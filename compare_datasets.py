"""compare_datasets.py — Compare v10 model trained on two dataset configurations.

Run:
    PYTHONPATH=. python compare_datasets.py

Trains the v10 ensemble twice using identical hyperparameters:
  [A] v1.12 only  — build_dataset.py (original, no augmentation)
  [B] v1.1 + v1.12 — build_dataset_v2.py (merged + subsampled at target_size hands)

Reports for each run:
  • Dataset statistics (chunks, dates, chunk sizes)
  • LODO cross-date AP  (leave-one-date-out, same protocol as train_v10.py)
  • Per-date AP breakdown
  • Frozen holdout score (last HOLDOUT_DAYS v1.12 dates — never touched during LODO)
  • Reward metric (0.65*AP + 0.35*recall) * human-safety-penalty
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.build_dataset_v2 import (
    AugmentMode,
    load_all_benchmark_examples,
)
from poker44_ml.features import chunk_features as base_cf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_DIR   = "data/benchmark_out"

# Frozen holdout: most recent v1.12 dates — never used in LODO training
HOLDOUT_DAYS    = 5
TOPK_FRAC       = 0.15

# Augmentation settings for run B
TARGET_SIZE     = 80          # hands per training chunk (live avg ~98)
MERGE_RATIO     = 1.5         # merge pool = TARGET_SIZE * MERGE_RATIO = 120 hands
SUBSAMPLE_DRAWS = 3           # windows per merged pool

# ---------------------------------------------------------------------------
# v10 estimators (identical to train_v10.py)
# ---------------------------------------------------------------------------

def make_estimators():
    return [
        lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
            min_child_samples=40, subsample=0.7, colsample_bytree=0.6,
            reg_lambda=5.0, reg_alpha=2.0, random_state=1, n_jobs=4, verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.02, num_leaves=63,
            min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
            reg_lambda=2.0, random_state=2, n_jobs=4, verbose=-1,
        ),
        ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=5, max_features=0.5,
            random_state=3, n_jobs=4,
        ),
        RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, max_features=0.5,
            random_state=4, n_jobs=4,
        ),
    ]

# ---------------------------------------------------------------------------
# Reward function (matches validator scoring)
# ---------------------------------------------------------------------------

def reward_metric(scores, labels):
    s  = np.asarray(scores, dtype=float)
    yt = np.asarray(labels, dtype=int)
    preds = (s >= 0.5).astype(int)
    tp = ((preds == 1) & (yt == 1)).sum()
    fp = ((preds == 1) & (yt == 0)).sum()
    tn = ((preds == 0) & (yt == 0)).sum()
    fn = ((preds == 0) & (yt == 1)).sum()
    fpr    = fp / max(tn + fp, 1)
    recall = tp / max(tp + fn, 1)
    ap     = (average_precision_score(yt, s)
              if yt.max() == 1 and yt.min() == 0 else 0.0)
    safety = 0.0 if fpr >= 0.10 else (1.0 - fpr) ** 2
    return {
        "reward": (0.65 * ap + 0.35 * recall) * safety,
        "ap":     ap,
        "fpr":    fpr,
        "recall": recall,
        "cliff":  fpr >= 0.10,
    }

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_feature_matrix(examples):
    fd    = [base_cf(e["chunk"]) for e in examples]
    names = sorted({k for d in fd for k in d})
    X     = np.array([[float(d.get(n, 0.0)) for n in names] for d in fd])
    y     = np.array([int(e["label"]) for e in examples])
    dates = [e["source_date"] for e in examples]
    return X, y, dates, names

# ---------------------------------------------------------------------------
# LODO evaluation
# ---------------------------------------------------------------------------

def lodo_eval(X, y, dates, holdout_set: set[str]):
    """LODO cross-date AP.  Dates in holdout_set are excluded from the loop
    (they must not appear in LODO training OR testing)."""
    all_dates = sorted(set(d for d in dates if d not in holdout_set))
    idx       = [i for i, d in enumerate(dates) if d not in holdout_set]
    X_lodo    = X[idx]; y_lodo = y[idx]
    dates_lodo = [dates[i] for i in idx]

    yt_all, ys_all = [], []
    per_date: list[dict] = []
    t0 = time.perf_counter()

    for date in all_dates:
        tr = [i for i, d in enumerate(dates_lodo) if d != date]
        te = [i for i, d in enumerate(dates_lodo) if d == date]
        if len(set(y_lodo[tr])) < 2 or len(te) == 0:
            continue

        ests = make_estimators()
        acc  = np.zeros(len(te))
        for est in ests:
            est.fit(X_lodo[tr], y_lodo[tr])
            acc += est.predict_proba(X_lodo[te])[:, 1]
        ys = list(acc / len(ests))
        yt = list(y_lodo[te])

        m = reward_metric(ys, yt)
        per_date.append({"date": date, "n": len(te), **m})
        yt_all += yt; ys_all += ys

    elapsed = time.perf_counter() - t0
    overall = reward_metric(ys_all, yt_all)
    return overall, per_date, elapsed

# ---------------------------------------------------------------------------
# Holdout evaluation (train on all non-holdout, score holdout)
# ---------------------------------------------------------------------------

def holdout_eval(X, y, dates, holdout_set: set[str], feat_names):
    tr = [i for i, d in enumerate(dates) if d not in holdout_set]
    te = [i for i, d in enumerate(dates) if d in holdout_set]
    if not te or len(set(y[tr])) < 2:
        return None, None

    ests = make_estimators()
    acc  = np.zeros(len(te))
    for est in ests:
        est.fit(X[tr], y[tr])
        acc += est.predict_proba(X[te])[:, 1]
    ys = list(acc / len(ests))
    yt = list(y[te])
    return reward_metric(ys, yt), [dates[i] for i in te]

# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 30, char: str = "█") -> str:
    filled = round(value * width)
    return char * filled + "░" * (width - filled)

def print_header(title: str) -> None:
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)

def print_dataset_stats(label: str, examples: list) -> None:
    versions  = Counter(e.get("version", "?") for e in examples)
    aug_types = Counter(e.get("aug_type", "none") for e in examples)
    labels    = Counter(e["label"] for e in examples)
    dates     = sorted(set(e["source_date"] for e in examples))
    sizes     = [len(e["chunk"]) for e in examples]

    print(f"\n  [{label}] Dataset Statistics")
    print(f"  {'─'*50}")
    print(f"  Total examples : {len(examples)}")
    print(f"  By version     : {dict(versions)}")
    print(f"  By label       : bot={labels[1]}  human={labels[0]}")
    print(f"  By aug_type    : {dict(aug_types)}")
    print(f"  Unique dates   : {len(dates)}  ({dates[0]} … {dates[-1]})")
    print(f"  Chunk sizes    : min={min(sizes)}  max={max(sizes)}  "
          f"mean={sum(sizes)/len(sizes):.1f}  (live avg ~98)")
    print(f"  Total hands    : {sum(sizes):,}")

def print_lodo_results(label: str, overall: dict, per_date: list,
                       elapsed: float) -> None:
    print(f"\n  [{label}] LODO Cross-Date Results  "
          f"({len(per_date)} dates, {elapsed:.1f}s)")
    print(f"  {'─'*50}")
    print(f"  OVERALL  AP={overall['ap']:.4f}  reward={overall['reward']:.4f}  "
          f"FPR={overall['fpr']:.4f}  recall={overall['recall']:.4f}")
    print(f"           cliff_hits={sum(1 for d in per_date if d['cliff'])}"
          f"/{len(per_date)}")
    print()
    print(f"  {'Date':<12} {'n':>4}  {'AP':>6}  {'Reward':>7}  "
          f"{'FPR':>6}  {'Recall':>6}  Cliff")
    print(f"  {'─'*58}")
    for d in per_date:
        cliff_flag = " !" if d["cliff"] else ""
        print(f"  {d['date']:<12} {d['n']:>4}  {d['ap']:>6.4f}  "
              f"{d['reward']:>7.4f}  {d['fpr']:>6.4f}  "
              f"{d['recall']:>6.4f}  {cliff_flag}")

def print_holdout_results(label: str, m: dict | None,
                          holdout_dates: list[str]) -> None:
    print(f"\n  [{label}] Frozen Holdout  "
          f"({len(holdout_dates)} dates: {holdout_dates[0]} … {holdout_dates[-1]})")
    print(f"  {'─'*50}")
    if m is None:
        print("  (not enough data)")
        return
    print(f"  AP={m['ap']:.4f}  reward={m['reward']:.4f}  "
          f"FPR={m['fpr']:.4f}  recall={m['recall']:.4f}  "
          f"cliff={'YES !' if m['cliff'] else 'no'}")

def print_comparison(res_a: dict, res_b: dict) -> None:
    print_header("SIDE-BY-SIDE COMPARISON")

    def _delta(key, a, b, higher_is_better=True):
        diff  = b[key] - a[key]
        arrow = ("▲" if diff > 0 else "▼") if abs(diff) > 0.0001 else "="
        color = arrow if (diff > 0) == higher_is_better else arrow
        return f"{arrow} {diff:+.4f}"

    rows = [
        ("Metric",          "[A] v1.12 only",   "[B] v1.1+v1.12",  "Delta B-A"),
        ("─" * 16,          "─" * 16,           "─" * 16,          "─" * 12),
        ("LODO AP",
            f"{res_a['lodo']['ap']:.4f}",
            f"{res_b['lodo']['ap']:.4f}",
            _delta("ap",     res_a["lodo"], res_b["lodo"])),
        ("LODO Reward",
            f"{res_a['lodo']['reward']:.4f}",
            f"{res_b['lodo']['reward']:.4f}",
            _delta("reward", res_a["lodo"], res_b["lodo"])),
        ("LODO FPR",
            f"{res_a['lodo']['fpr']:.4f}",
            f"{res_b['lodo']['fpr']:.4f}",
            _delta("fpr",    res_a["lodo"], res_b["lodo"], higher_is_better=False)),
        ("LODO Recall",
            f"{res_a['lodo']['recall']:.4f}",
            f"{res_b['lodo']['recall']:.4f}",
            _delta("recall", res_a["lodo"], res_b["lodo"])),
        ("LODO Cliff hits",
            f"{res_a['lodo_cliffs']}/{res_a['lodo_dates']}",
            f"{res_b['lodo_cliffs']}/{res_b['lodo_dates']}",
            ""),
        ("─" * 16,          "─" * 16,           "─" * 16,          "─" * 12),
        ("Holdout AP",
            f"{res_a['holdout']['ap']:.4f}" if res_a['holdout'] else "—",
            f"{res_b['holdout']['ap']:.4f}" if res_b['holdout'] else "—",
            _delta("ap", res_a["holdout"], res_b["holdout"])
                if res_a["holdout"] and res_b["holdout"] else ""),
        ("Holdout Reward",
            f"{res_a['holdout']['reward']:.4f}" if res_a['holdout'] else "—",
            f"{res_b['holdout']['reward']:.4f}" if res_b['holdout'] else "—",
            _delta("reward", res_a["holdout"], res_b["holdout"])
                if res_a["holdout"] and res_b["holdout"] else ""),
        ("─" * 16,          "─" * 16,           "─" * 16,          "─" * 12),
        ("Train examples",
            str(res_a["n_train"]),
            str(res_b["n_train"]),
            f"+{res_b['n_train'] - res_a['n_train']}"),
        ("Train dates",
            str(res_a["n_dates"]),
            str(res_b["n_dates"]),
            f"+{res_b['n_dates'] - res_a['n_dates']}"),
        ("Avg chunk size",
            f"{res_a['avg_size']:.1f} hands",
            f"{res_b['avg_size']:.1f} hands",
            f"{res_b['avg_size'] - res_a['avg_size']:+.1f}"),
    ]
    col_w = [18, 16, 16, 13]
    for row in rows:
        line = "  ".join(str(c).ljust(w) for c, w in zip(row, col_w))
        print(f"  {line}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(label: str, examples: list,
                   holdout_set: set[str]) -> dict:
    print_dataset_stats(label, examples)

    # Feature matrix (exclude holdout from entire training)
    lodo_exs = [e for e in examples if e["source_date"] not in holdout_set]
    X, y, dates, names = build_feature_matrix(lodo_exs)
    print(f"\n  [{label}] Feature matrix: {X.shape[0]} examples × {X.shape[1]} features")

    print(f"\n  [{label}] Running LODO cross-validation ...")
    overall, per_date, elapsed = lodo_eval(X, y, dates, holdout_set)
    print_lodo_results(label, overall, per_date, elapsed)

    # Holdout eval — train on all non-holdout examples, test on holdout
    # Need to rebuild matrix including holdout examples as test
    X_all, y_all, dates_all, _ = build_feature_matrix(examples)
    print(f"\n  [{label}] Running holdout evaluation ...")
    h_metric, h_dates = holdout_eval(X_all, y_all, dates_all, holdout_set, names)
    print_holdout_results(label, h_metric, h_dates or [])

    sizes = [len(e["chunk"]) for e in examples]
    return {
        "lodo":        overall,
        "lodo_cliffs": sum(1 for d in per_date if d["cliff"]),
        "lodo_dates":  len(per_date),
        "holdout":     h_metric,
        "n_train":     len(lodo_exs),
        "n_dates":     len(set(e["source_date"] for e in lodo_exs)),
        "avg_size":    sum(sizes) / len(sizes),
    }


def main():
    print_header("Poker44 Dataset Comparison: v1.12 only  vs  v1.1 + v1.12")
    print(f"\n  Config:")
    print(f"    benchmark_dir    = {BENCHMARK_DIR}")
    print(f"    holdout_days     = {HOLDOUT_DAYS} (most recent v1.12 dates)")
    print(f"    topk_frac        = {TOPK_FRAC}")
    print(f"    [B] target_size  = {TARGET_SIZE} hands  (live avg ~98)")
    print(f"    [B] merge_ratio  = {MERGE_RATIO}  (pool = {int(TARGET_SIZE*MERGE_RATIO)} hands)")
    print(f"    [B] sub_draws    = {SUBSAMPLE_DRAWS}")

    # ── Determine holdout dates (last N v1.12 dates) ─────────────────────
    print_header("Loading datasets")

    # Load v1.12 only first to determine holdout set
    exs_v112_raw = load_all_benchmark_examples(
        BENCHMARK_DIR,
        include_v11=False,
        include_v112=True,
        augment=AugmentMode.NONE,
    )
    all_v112_dates = sorted(set(e["source_date"] for e in exs_v112_raw))
    holdout_dates  = set(all_v112_dates[-HOLDOUT_DAYS:])
    print(f"\n  Holdout dates ({HOLDOUT_DAYS}): {sorted(holdout_dates)}")

    # ── Run A: v1.12 only, no augmentation (original build_dataset) ──────
    print_header("Run A: v1.12 only  [build_dataset.py — no augmentation]")
    exs_a = load_all_benchmark_examples(
        BENCHMARK_DIR,
        include_v11=False,
        include_v112=True,
        augment=AugmentMode.NONE,
    )
    res_a = run_experiment("A", exs_a, holdout_dates)

    # ── Run B: v1.1 + v1.12, merged + subsampled ─────────────────────────
    print_header("Run B: v1.1 + v1.12  [build_dataset_v2.py — BOTH augmentation]")
    exs_b = load_all_benchmark_examples(
        BENCHMARK_DIR,
        include_v11=True,
        include_v112=True,
        augment=AugmentMode.BOTH,
        target_size=TARGET_SIZE,
        merge_ratio=MERGE_RATIO,
        subsample_draws=SUBSAMPLE_DRAWS,
        seed=42,
    )
    res_b = run_experiment("B", exs_b, holdout_dates)

    # ── Side-by-side comparison ───────────────────────────────────────────
    print_comparison(res_a, res_b)
    print()


if __name__ == "__main__":
    main()
