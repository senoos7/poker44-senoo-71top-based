"""topk_sweep.py — Sweep topk_frac for v10 ensemble using LODO on v1.12 data only.

Strategy:
  - Train the v10 ensemble (3×LightGBM + ExtraTrees + RF) once per LODO fold.
  - Cache the raw ensemble scores for each test fold.
  - Apply _topk_squeeze at each frac in TOPK_FRACS without retraining.

Run:
    PYTHONPATH=. python topk_sweep.py

Output saved to logs/topk_sweep_<timestamp>.log
"""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.build_dataset_v2 import AugmentMode, load_all_benchmark_examples
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.model_v5 import _topk_squeeze

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_DIR = "data/benchmark_out"
TOPK_FRACS    = [0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

# ---------------------------------------------------------------------------
# Estimators  (identical to train_v10.py)
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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(raw_scores: np.ndarray, labels: np.ndarray,
                    frac: float) -> dict:
    """Apply topk_squeeze then compute AP, FPR, Recall, Reward."""
    scores = np.asarray(_topk_squeeze(raw_scores, frac), dtype=float)
    yt     = labels.astype(int)
    preds  = (scores >= 0.5).astype(int)

    tp = int(((preds == 1) & (yt == 1)).sum())
    fp = int(((preds == 1) & (yt == 0)).sum())
    tn = int(((preds == 0) & (yt == 0)).sum())
    fn = int(((preds == 0) & (yt == 1)).sum())

    fpr    = fp / max(tn + fp, 1)
    recall = tp / max(tp + fn, 1)
    ap     = (average_precision_score(yt, scores)
              if yt.max() == 1 and yt.min() == 0 else 0.0)
    safety = 0.0 if fpr >= 0.10 else (1.0 - fpr) ** 2
    reward = (0.65 * ap + 0.35 * recall) * safety
    cliff  = fpr >= 0.10

    return {
        "ap": ap, "fpr": fpr, "recall": recall,
        "reward": reward, "cliff": cliff,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

# ---------------------------------------------------------------------------
# Logger: write to file and stdout simultaneously
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, path: Path):
        self._f = open(path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg: str):
        self._stdout.write(msg)
        self._f.write(msg)

    def flush(self):
        self._stdout.flush()
        self._f.flush()

    def close(self):
        self._f.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = Path("logs") / f"topk_sweep_{ts}.log"
    logpath.parent.mkdir(exist_ok=True)
    tee = Tee(logpath)
    sys.stdout = tee

    print(f"topk_sweep  —  {ts}")
    print(f"log → {logpath}")
    print(f"dataset: v1.12 only  ({BENCHMARK_DIR})")
    print(f"topk_fracs: {TOPK_FRACS}")
    print()

    # ── Load v1.12 only ──────────────────────────────────────────────────
    t0  = time.perf_counter()
    exs = load_all_benchmark_examples(
        BENCHMARK_DIR,
        include_v11=False,
        include_v112=True,
        augment=AugmentMode.NONE,
    )
    print(f"Loaded {len(exs)} examples from v1.12")

    # Feature matrix
    fd    = [base_cf(e["chunk"]) for e in exs]
    names = sorted({k for d in fd for k in d})
    X     = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])
    y     = np.asarray([int(e["label"]) for e in exs])
    dates = [e["source_date"] for e in exs]
    uniq  = sorted(set(dates))

    print(f"Feature matrix: {X.shape[0]} × {X.shape[1]}")
    print(f"Dates: {uniq[0]} → {uniq[-1]}  ({len(uniq)} dates)")
    print(f"Labels: {int((y==1).sum())} bot / {int((y==0).sum())} human")
    print()

    # ── LODO: train once per fold, cache raw ensemble scores ─────────────
    print("=" * 66)
    print("LODO — training v10 ensemble (one pass, caching raw scores)")
    print("=" * 66)

    # raw_by_date[date] = (raw_scores_array, labels_array)
    raw_by_date: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for date in uniq:
        tr = [i for i, d in enumerate(dates) if d != date]
        te = [i for i, d in enumerate(dates) if d == date]
        if len(set(y[tr])) < 2 or not te:
            print(f"  skip {date} (insufficient label diversity)")
            continue

        t_fold = time.perf_counter()
        ests   = make_estimators()
        acc    = np.zeros(len(te), dtype=np.float64)
        for est in ests:
            est.fit(X[tr], y[tr])
            p    = est.predict_proba(X[te])
            acc += p[:, 1] if p.ndim == 2 else np.asarray(p)

        raw = acc / len(ests)
        raw_by_date[date] = (raw, y[te])
        elapsed = time.perf_counter() - t_fold
        print(f"  {date}  n={len(te):3d}  raw_mean={raw.mean():.4f}  ({elapsed:.0f}s)")

    total_lodo = time.perf_counter() - t0
    print(f"\nLODO training done in {total_lodo:.0f}s  ({len(raw_by_date)} folds)\n")

    # ── Sweep topk_frac ───────────────────────────────────────────────────
    print("=" * 66)
    print("TOPK_FRAC SWEEP")
    print("=" * 66)

    summary: list[dict] = []

    for frac in TOPK_FRACS:
        print(f"\n{'─'*66}")
        print(f"topk_frac = {frac:.2f}")
        print(f"{'─'*66}")
        print(f"  {'Date':<12} {'n':>3}  {'AP':>6}  {'Reward':>7}  "
              f"{'FPR':>6}  {'Recall':>6}  {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}  Cliff")

        all_raw, all_y = [], []
        per_date_metrics = []
        cliff_count = 0

        for date in sorted(raw_by_date):
            raw, yt = raw_by_date[date]
            m = compute_metrics(raw, yt, frac)
            per_date_metrics.append((date, len(yt), m))
            all_raw.append(raw)
            all_y.append(yt)
            if m["cliff"]:
                cliff_count += 1
            cliff_flag = " !" if m["cliff"] else ""
            print(f"  {date:<12} {len(yt):>3}  {m['ap']:>6.4f}  {m['reward']:>7.4f}  "
                  f"{m['fpr']:>6.4f}  {m['recall']:>6.4f}  "
                  f"{m['tp']:>3} {m['fp']:>3} {m['tn']:>3} {m['fn']:>3}  {cliff_flag}")

        # Overall: concatenate all test folds
        all_raw_cat = np.concatenate(all_raw)
        all_y_cat   = np.concatenate(all_y)
        overall     = compute_metrics(all_raw_cat, all_y_cat, frac)

        print(f"\n  OVERALL  AP={overall['ap']:.4f}  Reward={overall['reward']:.4f}  "
              f"FPR={overall['fpr']:.4f}  Recall={overall['recall']:.4f}  "
              f"Cliff_hits={cliff_count}/{len(per_date_metrics)}")

        summary.append({
            "frac":   frac,
            "ap":     overall["ap"],
            "reward": overall["reward"],
            "fpr":    overall["fpr"],
            "recall": overall["recall"],
            "cliffs": cliff_count,
            "n_folds": len(per_date_metrics),
        })

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("SUMMARY  —  topk_frac sweep on v1.12 LODO")
    print(f"{'='*66}")
    print(f"  {'frac':>6}  {'AP':>7}  {'Reward':>8}  {'FPR':>7}  {'Recall':>7}  "
          f"{'Cliffs':>8}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}")
    best_reward = max(s["reward"] for s in summary)
    for s in summary:
        marker = " ◄ best" if s["reward"] == best_reward else ""
        print(f"  {s['frac']:>6.2f}  {s['ap']:>7.4f}  {s['reward']:>8.4f}  "
              f"{s['fpr']:>7.4f}  {s['recall']:>7.4f}  "
              f"{s['cliffs']:>3}/{s['n_folds']}{marker}")

    total_elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print(f"Log saved to: {logpath}")

    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
