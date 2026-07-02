"""4-way LODO comparison:
  A  features_v1 (293) + v10   ensemble (5 est)  — reference baseline
  B  features_v1 (293) + v10-4 ensemble (7 est)  — more estimators, user request
  C  features_v2 (367) + v10   ensemble (5 est)  — feature upgrade only
  D  features_v2 (367) + v10-4 ensemble (7 est)  — combined best (main goal)

Run:
    PYTHONPATH=. python -W ignore train_combo_lodo.py
"""
from __future__ import annotations
import sys, time, datetime, json
from pathlib import Path
sys.path.insert(0, ".")
sys.path.append("/root/Skip/poker/poker_demo_2")

import numpy as np, joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import xgboost as xgb

import training.build_dataset as bd
from poker44_ml.features    import chunk_features as cf_v1
from poker44_ml.features_v2 import chunk_features as cf_v2
from poker44_bump.model_v5  import _topk_squeeze
from poker44_bump.model_v10 import V10Model

# ── logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)
ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logpath = Path("logs") / f"train_combo_{ts}.log"
logfile = open(logpath, "w", encoding="utf-8", buffering=1)
_real   = sys.stdout

class Tee:
    def write(self, m): _real.write(m); logfile.write(m)
    def flush(self):    _real.flush();  logfile.flush()

sys.stdout = Tee()

TOPK_FRAC = 0.15
print(f"4-way combo LODO  —  {ts}")
print(f"log → {logpath}  |  topk_frac={TOPK_FRAC}")
print()

# ── data ──────────────────────────────────────────────────────────────────────
exs   = bd.load_benchmark_examples(
    sorted(Path("data/benchmark_out").glob(
        "benchmark_2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    ))
)
y     = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]
uniq  = sorted(set(dates))
print(f"[data] {len(exs)} chunks / {len(uniq)} dates  "
      f"({int(y.sum())} bot / {int((y==0).sum())} human)\n")

# ── feature matrices ──────────────────────────────────────────────────────────
print("Building feature matrices ...")
fd1 = [cf_v1(e["chunk"]) for e in exs]; n1 = sorted({k for d in fd1 for k in d})
X1  = np.asarray([[float(d.get(k, 0.0)) for k in n1] for d in fd1])
fd2 = [cf_v2(e["chunk"]) for e in exs]; n2 = sorted({k for d in fd2 for k in d})
X2  = np.asarray([[float(d.get(k, 0.0)) for k in n2] for d in fd2])
print(f"  features_v1: {X1.shape[1]} feats   features_v2: {X2.shape[1]} feats "
      f"(+{X2.shape[1]-X1.shape[1]} new)\n")

# ── ensemble factories ────────────────────────────────────────────────────────
def make_v10():
    """Original 5-estimator ensemble (3×LGB + ET + RF)."""
    return [
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
    ]

def make_v10_4():
    """7-estimator ensemble (3×LGB + 2×XGB + ET + RF), strong reg (v10-4)."""
    return [
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
                          random_state=10, n_jobs=4, eval_metric="logloss", verbosity=0),
        xgb.XGBClassifier(n_estimators=350, learning_rate=0.02, max_depth=3,
                          subsample=0.75, colsample_bytree=0.65, reg_alpha=1.5,
                          reg_lambda=4.0, random_state=11, n_jobs=4,
                          eval_metric="logloss", verbosity=0),
        ExtraTreesClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.45,
                             random_state=3, n_jobs=4),
        RandomForestClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.45,
                               random_state=4, n_jobs=4),
    ]

# ── metrics ───────────────────────────────────────────────────────────────────
def mets(raw, yt):
    sq     = np.array(_topk_squeeze(raw, TOPK_FRAC))
    pred   = sq > 0.5
    n_bot  = int(yt.sum()); n_hum = int((yt==0).sum())
    tp     = int((pred & (yt==1)).sum()); fp = int((pred & (yt==0)).sum())
    fpr    = fp / max(n_hum, 1); recall = tp / max(n_bot, 1)
    ap     = float(average_precision_score(yt, sq)) if yt.max() != yt.min() else 0.0
    saf    = 0.0 if fpr >= 0.10 else (1.0-fpr)**2
    return dict(ap=ap, recall=recall, fpr=fpr,
                reward=(0.65*ap+0.35*recall)*saf, cliff=fpr>=0.10)

# ── LODO runner ───────────────────────────────────────────────────────────────
def run_lodo(tag: str, X: np.ndarray, feat_names: list, est_fn) -> tuple:
    print(f"{'='*70}")
    print(f"  {tag}")
    print(f"  {X.shape[1]} features  |  {len(est_fn())} estimators  |  topk_frac={TOPK_FRAC}")
    print(f"{'='*70}")
    results = []; t0 = time.time()
    for date in uniq:
        te = [i for i,d in enumerate(dates) if d==date]
        tr = [i for i,d in enumerate(dates) if d!=date]
        if len(set(y[tr])) < 2: continue
        ests = est_fn(); acc = np.zeros(len(te)); tf = time.time()
        for est in ests:
            est.fit(X[tr], y[tr])
            p = est.predict_proba(X[te])
            acc += p[:,1] if p.ndim==2 else np.asarray(p)
        raw = acc / len(ests); m = mets(raw, y[te])
        cliff_mark = "  CLIFF" if m["cliff"] else ""
        print(f"  {date}  n={len(te):2d}  "
              f"AP={m['ap']:.4f}  Recall={m['recall']:.4f}  "
              f"FPR={m['fpr']:.4f}  Reward={m['reward']:.4f}{cliff_mark}  "
              f"({time.time()-tf:.0f}s)")
        results.append({"date": date, "n": len(te), **m})
    elapsed = time.time() - t0
    ap  = float(np.mean([r["ap"]     for r in results]))
    rec = float(np.mean([r["recall"] for r in results]))
    fpr = float(np.mean([r["fpr"]    for r in results]))
    rew = float(np.mean([r["reward"] for r in results]))
    clf = sum(1 for r in results if r["cliff"])
    print(f"\n  AGGREGATE  AP={ap:.4f}  Recall={rec:.4f}  FPR={fpr:.4f}  "
          f"Reward={rew:.4f}  Cliffs={clf}/{len(results)}  ({elapsed:.0f}s)\n")
    return results, dict(ap=ap, recall=rec, fpr=fpr, reward=rew,
                         cliffs=clf, n_dates=len(results),
                         n_feats=X.shape[1], n_ests=len(est_fn()))

# ── run all 4 variants ────────────────────────────────────────────────────────
runs = [
    ("A  feat_v1 + v10    (293f, 5est)", X1, n1, make_v10,   "v10_v1feat"),
    ("B  feat_v1 + v10-4  (293f, 7est)", X1, n1, make_v10_4, "v10-4_v1feat"),
    ("C  feat_v2 + v10    (367f, 5est)", X2, n2, make_v10,   "v10_v2feat"),
    ("D  feat_v2 + v10-4  (367f, 7est)", X2, n2, make_v10_4, "v10-4_v2feat"),
]

all_results: dict[str, tuple] = {}
for tag, X, feat_names, est_fn, model_id in runs:
    res, agg = run_lodo(tag, X, feat_names, est_fn)
    all_results[tag] = (res, agg, X, feat_names, est_fn, model_id)

# ── save best model (D) ───────────────────────────────────────────────────────
tag_D, (res_D, agg_D, X_D, n_D, est_fn_D, mid_D) = \
    "D  feat_v2 + v10-4  (367f, 7est)", all_results["D  feat_v2 + v10-4  (367f, 7est)"]
print("Fitting final model D on all data → models/bump_model_v10-4v2.joblib ...")
ests_final = est_fn_D()
for est in ests_final: est.fit(X_D, y)
model_D = V10Model(
    ests_final, n_D,
    topk_cfg={"positive_fraction": TOPK_FRAC},
    metadata={
        "model_version": "v10-4v2",
        "model_name":    "poker44-bump-v10-4v2",
        "framework":     "avg-ensemble(3lgbm,2xgb,et,rf)+topk+feat_v2",
        "n_estimators":  len(ests_final),
        "n_feats":       len(n_D),
        "lodo_ap":       round(agg_D["ap"], 6),
        "lodo_reward":   round(agg_D["reward"], 6),
        "lodo_cliffs":   agg_D["cliffs"],
        "trained_at":    ts,
    },
)
joblib.dump(model_D, "models/bump_model_v10-4v2.joblib")
meta_path = Path("models/bump_model_v10-4v2_meta.json")
with open(meta_path, "w") as f:
    json.dump({
        "variant": "v10-4v2",
        "description": "feat_v2 (367) + v10-4 ensemble (3×LGB + 2×XGB + ET + RF)",
        "trained_at": ts,
        "lodo_aggregate": agg_D,
        "lodo_per_date": res_D,
    }, f, indent=2, default=str)
print(f"  saved models/bump_model_v10-4v2.joblib  ({len(ests_final)} est, {len(n_D)} feats)")
print(f"  saved {meta_path}\n")

# ── comparison table ──────────────────────────────────────────────────────────
print("="*78)
print("FINAL COMPARISON  (mean per-date LODO, topk_frac=0.15)")
print("="*78)
print(f"  {'Variant':<34}  {'F':>3}  {'E':>2}  {'AP':>7}  {'Recall':>7}  "
      f"{'FPR':>6}  {'Reward':>8}  Cliffs")
print(f"  {'-'*34}  {'-'*3}  {'-'*2}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*6}")

ref_ap  = all_results["A  feat_v1 + v10    (293f, 5est)"][1]["ap"]
ref_rew = all_results["A  feat_v1 + v10    (293f, 5est)"][1]["reward"]
for tag, (res, agg, *_rest) in all_results.items():
    d_ap  = f" ({agg['ap']-ref_ap:+.4f})"  if tag != "A  feat_v1 + v10    (293f, 5est)" else " (baseline)"
    print(f"  {tag:<34}  {agg['n_feats']:>3}  {agg['n_ests']:>2}  "
          f"{agg['ap']:>7.4f}  {agg['recall']:>7.4f}  {agg['fpr']:>6.4f}  "
          f"{agg['reward']:>8.4f}  {agg['cliffs']}/{agg['n_dates']}{d_ap}")

# ── per-date AP matrix ────────────────────────────────────────────────────────
print()
print("─"*78)
print("PER-DATE AP  (A=feat_v1+v10  B=feat_v1+v10-4  C=feat_v2+v10  D=feat_v2+v10-4)")
print("─"*78)
keys   = list(all_results.keys())
labels = ["A", "B", "C", "D"]
header = f"  {'Date':<12}" + "".join(f"  {l:>7}" for l in labels) + "  best"
print(header)
print("  " + "─"*(12 + 10*len(labels) + 6))

for date in uniq:
    row_aps: list[float] = []
    for tag in keys:
        res = all_results[tag][0]
        d2r = {r["date"]: r for r in res}
        row_aps.append(d2r.get(date, {}).get("ap", float("nan")))
    best_idx = int(np.nanargmax(row_aps))
    row = f"  {date:<12}"
    for i, ap in enumerate(row_aps):
        marker = "◄" if i == best_idx else " "
        row += f"  {ap:>6.4f}{marker}"
    row += f"  {labels[best_idx]}"
    print(row)

# Best count per variant
counts = [0]*4
for date in uniq:
    row_aps = []
    for tag in keys:
        d2r = {r["date"]: r for r in all_results[tag][0]}
        row_aps.append(d2r.get(date, {}).get("ap", float("nan")))
    counts[int(np.nanargmax(row_aps))] += 1

print()
print("  Best AP wins per variant: " +
      "  ".join(f"{l}={c}" for l, c in zip(labels, counts)))
print(f"\nLog saved to: {logpath}")

sys.stdout = _real
logfile.close()
