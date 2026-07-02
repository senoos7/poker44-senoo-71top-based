"""LODO comparison: features_v1 (293) vs features_v2 (367) with v10 ensemble.

Run:
    PYTHONPATH=. python -W ignore train_featv2_lodo.py
"""
from __future__ import annotations
import sys, time, datetime
from pathlib import Path
sys.path.insert(0, ".")
sys.path.append("/root/Skip/poker/poker_demo_2")

import numpy as np, joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
import lightgbm as lgb

import training.build_dataset as bd
from poker44_ml.features    import chunk_features as cf_v1
from poker44_ml.features_v2 import chunk_features as cf_v2
from poker44_bump.model_v5  import _topk_squeeze
from poker44_bump.model_v10 import V10Model

# ── logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logpath = Path("logs") / f"train_featv2_{ts}.log"
logfile = open(logpath, "w", encoding="utf-8", buffering=1)
_real   = sys.stdout

class Tee:
    def write(self, m): _real.write(m); logfile.write(m)
    def flush(self):    _real.flush();  logfile.flush()

sys.stdout = Tee()

print(f"Feature-v2 upgrade LODO  —  {ts}")
print(f"log → {logpath}")
print()

TOPK_FRAC = 0.15

# ── data ──────────────────────────────────────────────────────────────────────
exs   = bd.load_benchmark_examples(
    sorted(Path("data/benchmark_out").glob(
        "benchmark_2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    ))
)
y     = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]
uniq  = sorted(set(dates))

# ── feature matrices ──────────────────────────────────────────────────────────
print("Computing features_v1 (293 feats) ...")
fd1 = [cf_v1(e["chunk"]) for e in exs]
n1  = sorted({k for d in fd1 for k in d})
X1  = np.asarray([[float(d.get(k, 0.0)) for k in n1] for d in fd1])
print(f"  shape: {X1.shape}")

print("Computing features_v2 (367 feats) ...")
fd2 = [cf_v2(e["chunk"]) for e in exs]
n2  = sorted({k for d in fd2 for k in d})
X2  = np.asarray([[float(d.get(k, 0.0)) for k in n2] for d in fd2])
print(f"  shape: {X2.shape}")

new_feats = sorted(set(n2) - set(n1))
print(f"\n{len(new_feats)} new features in v2:")
for f in new_feats:
    print(f"  {f}")
print()

# ── helpers ───────────────────────────────────────────────────────────────────
def make_v10():
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

def run_lodo(tag, X, feat_names):
    print(f"{'='*66}")
    print(f"LODO  [{tag}]  ({X.shape[1]} features, topk_frac={TOPK_FRAC})")
    print(f"{'='*66}")
    results = []; t0 = time.time()
    for date in uniq:
        te = [i for i,d in enumerate(dates) if d==date]
        tr = [i for i,d in enumerate(dates) if d!=date]
        if len(set(y[tr])) < 2: continue
        ests = make_v10(); acc = np.zeros(len(te)); tf = time.time()
        for est in ests:
            est.fit(X[tr], y[tr])
            p = est.predict_proba(X[te])
            acc += p[:,1]
        raw = acc / len(ests)
        m   = mets(raw, y[te])
        print(f"  {date}  n={len(te):2d}  AP={m['ap']:.4f}  "
              f"Recall={m['recall']:.4f}  FPR={m['fpr']:.4f}  "
              f"Reward={m['reward']:.4f}{'  CLIFF' if m['cliff'] else ''}  "
              f"({time.time()-tf:.0f}s)")
        results.append({"date": date, **m})

    elapsed = time.time() - t0
    ap  = float(np.mean([r["ap"]     for r in results]))
    rec = float(np.mean([r["recall"] for r in results]))
    fpr = float(np.mean([r["fpr"]    for r in results]))
    rew = float(np.mean([r["reward"] for r in results]))
    clf = sum(1 for r in results if r["cliff"])
    print(f"\n  [{tag}] AGGREGATE  AP={ap:.4f}  Recall={rec:.4f}  "
          f"FPR={fpr:.4f}  Reward={rew:.4f}  Cliffs={clf}/{len(results)}  "
          f"({elapsed:.0f}s)\n")
    return results, dict(ap=ap, recall=rec, fpr=fpr, reward=rew, cliffs=clf,
                         n_dates=len(results), n_feats=X.shape[1])

res1, agg1 = run_lodo("v10  features_v1", X1, n1)
res2, agg2 = run_lodo("v10v2 features_v2", X2, n2)

# ── save v10v2 model ──────────────────────────────────────────────────────────
print("Final fit on all data → models/bump_model_v10v2.joblib ...")
ests = make_v10()
for est in ests: est.fit(X2, y)
model = V10Model(ests, n2, topk_cfg={"positive_fraction": TOPK_FRAC},
                 metadata={"model_version": "v10v2",
                           "model_name": "poker44-bump-v10v2",
                           "framework": "avg-ensemble(lgbm,et,rf)+topk+feat_v2",
                           "n_estimators": len(ests), "n_feats": len(n2),
                           "lodo_ap": round(agg2["ap"], 6),
                           "lodo_reward": round(agg2["reward"], 6)})
joblib.dump(model, "models/bump_model_v10v2.joblib")
print("  saved models/bump_model_v10v2.joblib\n")

# ── side-by-side summary ──────────────────────────────────────────────────────
print("="*68)
print("SUMMARY  —  same v10 ensemble, features_v1 vs features_v2")
print("="*68)
print(f"  {'Variant':<22}  {'Feats':>5}  {'AP':>7}  {'Recall':>7}  "
      f"{'FPR':>7}  {'Reward':>8}  {'Cliffs'}")
print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")
for tag, agg in [("v10  (features_v1)", agg1), ("v10v2 (features_v2)", agg2)]:
    print(f"  {tag:<22}  {agg['n_feats']:>5}  {agg['ap']:>7.4f}  "
          f"{agg['recall']:>7.4f}  {agg['fpr']:>7.4f}  {agg['reward']:>8.4f}  "
          f"{agg['cliffs']}/{agg['n_dates']}")
d_ap  = agg2["ap"]     - agg1["ap"]
d_rew = agg2["reward"] - agg1["reward"]
d_rec = agg2["recall"] - agg1["recall"]
print(f"  {'Delta (v2 − v1)':<22}  {agg2['n_feats']-agg1['n_feats']:>+5}  "
      f"  {d_ap:>+7.4f}  {d_rec:>+7.4f}  {'':>7}  {d_rew:>+8.4f}")

print()
print("Per-date AP:  v1   v2   Δ")
print("-"*40)
d2r2 = {r["date"]: r for r in res2}
for r in res1:
    r2 = d2r2.get(r["date"], {})
    ap2 = r2.get("ap", float("nan"))
    delta = ap2 - r["ap"]
    mark = " ▲" if delta > 0.001 else (" ▼" if delta < -0.001 else "  ")
    print(f"  {r['date']}   {r['ap']:.4f}  {ap2:.4f}  {delta:+.4f}{mark}")

print(f"\nLog saved to: {logpath}")
sys.stdout = _real
logfile.close()
