"""train_v10_search.py — Hyperparameter search to improve v10 AP score.

Trains 4 variants (v10-1 … v10-4) against the frozen baseline (v10).
Each variant uses LODO over all 37 v1.12 dates, topk_frac=0.15 throughout.

Focus: Average Precision (AP) — the primary ranking quality metric.
       All variants keep topk_frac=0.15 to isolate the effect of the
       ensemble itself rather than the safety head.

Variants
--------
v10-1  Stronger regularization:
       Higher reg_lambda/alpha, lower lr, more trees, larger min_child_samples.
       Hypothesis: current LGBMs are slightly noisy; tighter constraints →
       more consistent bot-vs-human ranking.

v10-2  XGBoost diversity:
       Drop 2 LGBMs, add 2 XGBClassifiers (level-wise splits vs LGBM leaf-wise).
       Hypothesis: different tree-growth algorithm reduces ensemble bias on hard dates.

v10-3  CatBoost + DART:
       Add CatBoostClassifier + one DART-mode LightGBM alongside base LGB/ET/RF.
       Hypothesis: CatBoost ordered-boosting and DART dropout attack different
       generalization failure modes.

v10-4  Large diverse ensemble (7 estimators):
       3×LGB (varied params) + 2×XGB (varied params) + ET + RF, all with
       strong regularization.
       Hypothesis: maximum diversity + quantity → lowest LODO variance → best AP.

Output
------
  models/bump_model_v10-{N}.joblib       final model (all data)
  models/bump_model_v10-{N}_meta.json    training metadata
  logs/train_v10_search_{ts}.log         full run log

Run:
    PYTHONPATH=. python -W ignore train_v10_search.py
"""
from __future__ import annotations

import datetime, json, math, sys, time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "/root/Skip/poker/poker_demo_2")

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.model_v10 import V10Model
from poker44_bump.model_v5 import _topk_squeeze

TOPK_FRAC = 0.15
BENCHMARK_DIR = "data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

Path("logs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
_ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_logpath = Path("logs") / f"train_v10_search_{_ts}.log"
_logfile = open(_logpath, "w", encoding="utf-8", buffering=1)
_real_stdout = sys.stdout

class _Tee:
    def write(self, m):  _real_stdout.write(m); _logfile.write(m)
    def flush(self):     _real_stdout.flush();  _logfile.flush()

sys.stdout = _Tee()

def log(msg: str = ""):
    print(msg)

# ---------------------------------------------------------------------------
# Data  (v1.12 only via build_dataset.py)
# ---------------------------------------------------------------------------

log(f"train_v10_search  —  {_ts}")
log(f"log → {_logpath}")
log(f"topk_frac = {TOPK_FRAC}")
log()

exs   = bd.load_benchmark_examples(
    sorted(Path("data/benchmark_out").glob(
        "benchmark_2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    ))
)
y     = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]
uniq  = sorted(set(dates))

fd    = [base_cf(e["chunk"]) for e in exs]
names = sorted({k for d in fd for k in d})
X     = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])

log(f"[data] {len(exs)} chunks / {len(uniq)} dates / {len(names)} features")
log(f"       labels: {int(y.sum())} bot / {int((y==0).sum())} human")
log()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(raw: np.ndarray, yt: np.ndarray, frac: float = TOPK_FRAC) -> dict:
    sq    = np.array(_topk_squeeze(raw, frac))
    pred  = sq > 0.5
    n_bot = int(yt.sum()); n_hum = int((yt == 0).sum())
    tp    = int((pred & (yt == 1)).sum())
    fp    = int((pred & (yt == 0)).sum())
    fpr   = fp / max(n_hum, 1)
    recall= tp / max(n_bot, 1)
    ap    = float(average_precision_score(yt, sq)) if yt.max() != yt.min() else 0.0
    saf   = 0.0 if fpr >= 0.10 else (1.0 - fpr) ** 2
    reward= (0.65 * ap + 0.35 * recall) * saf
    return dict(ap=ap, recall=recall, fpr=fpr, reward=reward,
                cliff=fpr >= 0.10, tp=tp, fp=fp)

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS: dict[str, dict] = {

    "v10": {
        "desc": "Baseline — original diverse avg ensemble (no cx_)",
        "estimators_fn": lambda: [
            lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1),
            lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
                               min_child_samples=40, subsample=0.7, colsample_bytree=0.6,
                               reg_lambda=5.0, reg_alpha=2.0, random_state=1, n_jobs=4, verbose=-1),
            lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=63,
                               min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
                               reg_lambda=2.0, random_state=2, n_jobs=4, verbose=-1),
            ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                 random_state=3, n_jobs=4),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                   random_state=4, n_jobs=4),
        ],
        "params": {
            "lgb1": dict(n_estimators=400, lr=0.03, num_leaves=31, min_child=20, reg_lambda=1.0),
            "lgb2": dict(n_estimators=300, lr=0.02, num_leaves=15, max_depth=4, min_child=40, reg_lambda=5.0, reg_alpha=2.0),
            "lgb3": dict(n_estimators=500, lr=0.02, num_leaves=63, min_child=30, reg_lambda=2.0),
            "et":   dict(n_estimators=400, min_samples_leaf=5, max_features=0.5),
            "rf":   dict(n_estimators=400, min_samples_leaf=5, max_features=0.5),
        },
    },

    "v10-1": {
        "desc": "Stronger regularization — higher reg, lower lr, more trees",
        "estimators_fn": lambda: [
            lgb.LGBMClassifier(n_estimators=700, learning_rate=0.015, num_leaves=31,
                               min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=3.0, reg_alpha=1.0, random_state=0, n_jobs=4, verbose=-1),
            lgb.LGBMClassifier(n_estimators=600, learning_rate=0.012, num_leaves=15, max_depth=4,
                               min_child_samples=50, subsample=0.7, colsample_bytree=0.6,
                               reg_lambda=8.0, reg_alpha=3.0, random_state=1, n_jobs=4, verbose=-1),
            lgb.LGBMClassifier(n_estimators=800, learning_rate=0.012, num_leaves=47,
                               min_child_samples=40, subsample=0.75, colsample_bytree=0.7,
                               reg_lambda=4.0, reg_alpha=1.0, random_state=2, n_jobs=4, verbose=-1),
            ExtraTreesClassifier(n_estimators=600, min_samples_leaf=8, max_features=0.4,
                                 random_state=3, n_jobs=4),
            RandomForestClassifier(n_estimators=600, min_samples_leaf=8, max_features=0.4,
                                   random_state=4, n_jobs=4),
        ],
        "params": {
            "lgb1": dict(n_estimators=700, lr=0.015, num_leaves=31, min_child=30, reg_lambda=3.0, reg_alpha=1.0),
            "lgb2": dict(n_estimators=600, lr=0.012, num_leaves=15, max_depth=4, min_child=50, reg_lambda=8.0, reg_alpha=3.0),
            "lgb3": dict(n_estimators=800, lr=0.012, num_leaves=47, min_child=40, reg_lambda=4.0, reg_alpha=1.0),
            "et":   dict(n_estimators=600, min_samples_leaf=8, max_features=0.4),
            "rf":   dict(n_estimators=600, min_samples_leaf=8, max_features=0.4),
        },
    },

    "v10-2": {
        "desc": "XGBoost diversity — replace 2 LGBMs with level-wise XGB",
        "estimators_fn": lambda: [
            lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1),
            lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=63,
                               min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
                               reg_lambda=2.0, random_state=2, n_jobs=4, verbose=-1),
            xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                               random_state=10, n_jobs=4, eval_metric="logloss",
                               verbosity=0, use_label_encoder=False),
            xgb.XGBClassifier(n_estimators=300, learning_rate=0.02, max_depth=3,
                               subsample=0.7, colsample_bytree=0.6, reg_alpha=2.0,
                               reg_lambda=3.0, random_state=11, n_jobs=4,
                               eval_metric="logloss", verbosity=0, use_label_encoder=False),
            ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                 random_state=3, n_jobs=4),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                   random_state=4, n_jobs=4),
        ],
        "params": {
            "lgb1": dict(n_estimators=400, lr=0.03, num_leaves=31, min_child=20, reg_lambda=1.0),
            "lgb3": dict(n_estimators=500, lr=0.02, num_leaves=63, min_child=30, reg_lambda=2.0),
            "xgb1": dict(n_estimators=400, lr=0.03, max_depth=4, reg_lambda=2.0),
            "xgb2": dict(n_estimators=300, lr=0.02, max_depth=3, reg_alpha=2.0, reg_lambda=3.0),
            "et":   dict(n_estimators=400, min_samples_leaf=5, max_features=0.5),
            "rf":   dict(n_estimators=400, min_samples_leaf=5, max_features=0.5),
        },
    },

    "v10-3": {
        "desc": "CatBoost + DART — ordered boosting + dropout regularization",
        "estimators_fn": lambda: [
            lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1),
            lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, boosting_type="dart", drop_rate=0.1,
                               random_state=5, n_jobs=4, verbose=-1),
            CatBoostClassifier(iterations=500, learning_rate=0.03, depth=5,
                               l2_leaf_reg=5.0, random_seed=20,
                               loss_function="Logloss", verbose=False,
                               allow_writing_files=False),
            ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                 random_state=3, n_jobs=4),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                   random_state=4, n_jobs=4),
        ],
        "params": {
            "lgb1":     dict(n_estimators=400, lr=0.03, num_leaves=31, min_child=20, reg_lambda=1.0),
            "lgb_dart": dict(n_estimators=300, lr=0.05, num_leaves=31, boosting="dart", drop_rate=0.1),
            "catboost": dict(iterations=500, lr=0.03, depth=5, l2_leaf_reg=5.0),
            "et":       dict(n_estimators=400, min_samples_leaf=5, max_features=0.5),
            "rf":       dict(n_estimators=400, min_samples_leaf=5, max_features=0.5),
        },
    },

    "v10-4": {
        "desc": "Large diverse ensemble — 3×LGB + 2×XGB + ET + RF (7 total, strong reg)",
        "estimators_fn": lambda: [
            lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=31,
                               min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=2.0, reg_alpha=0.5, random_state=0, n_jobs=4, verbose=-1),
            lgb.LGBMClassifier(n_estimators=500, learning_rate=0.015, num_leaves=15, max_depth=4,
                               min_child_samples=45, subsample=0.7, colsample_bytree=0.6,
                               reg_lambda=6.0, reg_alpha=2.0, random_state=1, n_jobs=4, verbose=-1),
            lgb.LGBMClassifier(n_estimators=700, learning_rate=0.015, num_leaves=47,
                               min_child_samples=35, subsample=0.75, colsample_bytree=0.65,
                               reg_lambda=3.0, reg_alpha=1.0, random_state=2, n_jobs=4, verbose=-1),
            xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                               random_state=10, n_jobs=4, eval_metric="logloss",
                               verbosity=0, use_label_encoder=False),
            xgb.XGBClassifier(n_estimators=350, learning_rate=0.02, max_depth=3,
                               subsample=0.75, colsample_bytree=0.65, reg_alpha=1.5,
                               reg_lambda=4.0, random_state=11, n_jobs=4,
                               eval_metric="logloss", verbosity=0, use_label_encoder=False),
            ExtraTreesClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.45,
                                 random_state=3, n_jobs=4),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.45,
                                   random_state=4, n_jobs=4),
        ],
        "params": {
            "lgb1": dict(n_estimators=600, lr=0.02, num_leaves=31, min_child=25, reg_lambda=2.0, reg_alpha=0.5),
            "lgb2": dict(n_estimators=500, lr=0.015, num_leaves=15, max_depth=4, min_child=45, reg_lambda=6.0, reg_alpha=2.0),
            "lgb3": dict(n_estimators=700, lr=0.015, num_leaves=47, min_child=35, reg_lambda=3.0, reg_alpha=1.0),
            "xgb1": dict(n_estimators=400, lr=0.03, max_depth=4, reg_lambda=2.0),
            "xgb2": dict(n_estimators=350, lr=0.02, max_depth=3, reg_alpha=1.5, reg_lambda=4.0),
            "et":   dict(n_estimators=500, min_samples_leaf=6, max_features=0.45),
            "rf":   dict(n_estimators=500, min_samples_leaf=6, max_features=0.45),
        },
    },
}

# ---------------------------------------------------------------------------
# LODO runner
# ---------------------------------------------------------------------------

def run_lodo(variant_name: str, estimators_fn) -> tuple[list[dict], float]:
    """Run LODO, return (per_date_results, elapsed_s)."""
    results = []
    t0 = time.time()

    for date in uniq:
        te = [i for i, d in enumerate(dates) if d == date]
        tr = [i for i, d in enumerate(dates) if d != date]
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr)) < 2:
            continue

        ests = estimators_fn()
        acc  = np.zeros(len(te))
        t_f  = time.time()
        for est in ests:
            est.fit(X[tr], y_tr)
            p    = est.predict_proba(X[te])
            acc += p[:, 1] if p.ndim == 2 else np.asarray(p)
        raw = acc / len(ests)

        m = metrics(raw, y_te)
        elapsed_f = time.time() - t_f
        cliff_mark = "  CLIFF" if m["cliff"] else ""
        log(f"  [{variant_name}] {date}  n={len(te):2d}  "
            f"AP={m['ap']:.4f}  Recall={m['recall']:.4f}  "
            f"FPR={m['fpr']:.4f}  Reward={m['reward']:.4f}{cliff_mark}  "
            f"({elapsed_f:.0f}s)")
        results.append({"date": date, "n": len(te), **m})

    return results, time.time() - t0


def aggregate(results: list[dict]) -> dict:
    """Pool all test fold predictions for overall metrics (not mean of per-date)."""
    ap   = float(np.mean([r["ap"]     for r in results]))
    rec  = float(np.mean([r["recall"] for r in results]))
    fpr  = float(np.mean([r["fpr"]    for r in results]))
    rew  = float(np.mean([r["reward"] for r in results]))
    cliffs = sum(1 for r in results if r["cliff"])
    return dict(ap=ap, recall=rec, fpr=fpr, reward=rew,
                cliffs=cliffs, n_dates=len(results))


def save_model_and_meta(variant_name: str, variant: dict,
                        agg_metrics: dict, elapsed: float,
                        per_date: list[dict]):
    """Final fit on all data, save joblib + metadata JSON."""
    ests = variant["estimators_fn"]()
    for est in ests:
        est.fit(X, y)
    model = V10Model(
        ests, names,
        topk_cfg={"positive_fraction": TOPK_FRAC},
        metadata={
            "model_version":  variant_name,
            "model_name":     f"poker44-bump-{variant_name}",
            "framework":      "avg-ensemble+topk",
            "n_estimators":   len(ests),
            "n_features":     len(names),
            "topk_frac":      TOPK_FRAC,
            "lodo_ap":        round(agg_metrics["ap"], 6),
            "lodo_recall":    round(agg_metrics["recall"], 6),
            "lodo_reward":    round(agg_metrics["reward"], 6),
            "lodo_fpr":       round(agg_metrics["fpr"], 6),
            "lodo_cliffs":    agg_metrics["cliffs"],
            "lodo_dates":     agg_metrics["n_dates"],
            "trained_at":     _ts,
        },
    )
    model_path = Path(f"models/bump_model_{variant_name}.joblib")
    joblib.dump(model, model_path)

    meta = {
        "variant":        variant_name,
        "description":    variant["desc"],
        "trained_at":     _ts,
        "topk_frac":      TOPK_FRAC,
        "n_estimators":   len(ests),
        "n_features":     len(names),
        "dataset":        "v1.12 only (benchmark_YYYY-MM-DD.json)",
        "n_examples":     len(exs),
        "n_dates":        len(uniq),
        "lodo_seconds":   round(elapsed, 1),
        "hyperparameters": variant["params"],
        "lodo_aggregate": agg_metrics,
        "lodo_per_date":  per_date,
    }
    meta_path = Path(f"models/bump_model_{variant_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    log(f"  → saved {model_path}  ({len(ests)} estimators, {len(names)} feats)")
    log(f"  → saved {meta_path}")
    return model_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

all_results: dict[str, dict] = {}

for variant_name, variant in VARIANTS.items():
    log()
    log("=" * 70)
    log(f"  {variant_name}  |  {variant['desc']}")
    log("=" * 70)

    per_date, elapsed = run_lodo(variant_name, variant["estimators_fn"])
    agg = aggregate(per_date)

    log()
    log(f"  [{variant_name}] LODO done in {elapsed:.0f}s  "
        f"({agg['n_dates']} dates)")
    log(f"  [{variant_name}] AGGREGATE  "
        f"AP={agg['ap']:.4f}  Recall={agg['recall']:.4f}  "
        f"FPR={agg['fpr']:.4f}  Reward={agg['reward']:.4f}  "
        f"Cliffs={agg['cliffs']}/{agg['n_dates']}")

    save_model_and_meta(variant_name, variant, agg, elapsed, per_date)
    all_results[variant_name] = {"agg": agg, "per_date": per_date}


# ---------------------------------------------------------------------------
# Summary comparison table
# ---------------------------------------------------------------------------

log()
log("=" * 78)
log("FINAL COMPARISON  (mean per-date, topk_frac=0.15)")
log("=" * 78)
log()
log(f"  {'Variant':<10}  {'AP':>7}  {'Recall':>7}  {'FPR':>7}  "
    f"{'Reward':>8}  {'Cliffs':>8}  Description")
log(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*30}")

baseline_ap = all_results["v10"]["agg"]["ap"]
for vname, res in all_results.items():
    a = res["agg"]
    delta = f"  ({a['ap']-baseline_ap:+.4f})" if vname != "v10" else "  (baseline)"
    log(f"  {vname:<10}  {a['ap']:>7.4f}  {a['recall']:>7.4f}  {a['fpr']:>7.4f}  "
        f"{a['reward']:>8.4f}  {a['cliffs']:>3}/{a['n_dates']:<4}  "
        f"{VARIANTS[vname]['desc'][:35]}{delta}")

log()

# Per-date AP comparison table
log("─" * 78)
log("PER-DATE AP BREAKDOWN")
log("─" * 78)
variant_names = list(all_results.keys())
header = f"  {'Date':<12}" + "".join(f"  {n:>8}" for n in variant_names)
log(header)
log("  " + "─" * (12 + 10 * len(variant_names)))

for date in uniq:
    row = f"  {date:<12}"
    for vname in variant_names:
        pd = {r["date"]: r for r in all_results[vname]["per_date"]}
        if date in pd:
            ap  = pd[date]["ap"]
            clif = "C" if pd[date]["cliff"] else " "
            row += f"  {ap:>7.4f}{clif}"
        else:
            row += f"  {'—':>8} "
    log(row)

log()
log(f"Log saved to: {_logpath}")

sys.stdout = _Tee._real_stdout if hasattr(_Tee, "_real_stdout") else _real_stdout
_logfile.close()
