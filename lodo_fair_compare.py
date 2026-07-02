"""Fair LODO (Leave-One-Date-Out) comparison:
  - v10       : 3×LightGBM + ExtraTrees + RF, uniform avg, no sample weights
  - pareto_v3 : CatBoost + RandomForest + ParetoForest, uniform avg,
                human_weight_multiplier=2.0 (mirrors original training command)

Both models are retrained from scratch on all dates except the held-out one,
then evaluated on the held-out date.  No pre-trained artifacts are used.

Run:
    PYTHONPATH=. python -W ignore lodo_fair_compare.py 2>&1 | tee lodo_fair.log
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/root/Skip/poker/poker_demo_2")
sys.path.insert(0, ".")   # local packages take priority

import numpy as np
import joblib
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
import multiprocessing as _mp
from poker44_bump.fast_pareto import FastParetoForestClassifier as ParetoForestClassifier

# ── load benchmark data ──────────────────────────────────────────────────────
exs   = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]
y      = np.array([int(e["label"]) for e in exs])
dates  = [e["source_date"] for e in exs]
uniq   = sorted(set(dates))

fd    = [base_cf(c) for c in chunks]
names = sorted({k for d in fd for k in d})
X     = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])

print(f"[data] {len(exs)} chunks / {len(uniq)} dates / {len(names)} features")
print(f"       dates: {uniq[0]} → {uniq[-1]}")
print(f"       labels: {int(y.sum())} bot / {int((y==0).sum())} human")
print()

# ── model factories ──────────────────────────────────────────────────────────
SEED = 42

def make_v10():
    """3×LightGBM + ExtraTrees + RF — mirrors train_v10.py exactly."""
    return [
        lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
                           min_child_samples=40, subsample=0.7, colsample_bytree=0.6,
                           reg_lambda=5.0, reg_alpha=2.0, random_state=SEED+1, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=63,
                           min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
                           reg_lambda=2.0, random_state=SEED+2, n_jobs=4, verbose=-1),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                             random_state=SEED+3, n_jobs=4),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                               random_state=SEED+4, n_jobs=4),
    ]

_PARETO_POOL = None   # persistent fork pool, created once before the LODO loop

def make_pareto():
    """CatBoost + RandomForest + FastParetoForest — mirrors original pareto_v3 command.
    ParetoForest uses a persistent fork pool (created once) to avoid repeated
    pool-creation overhead across 37 LODO folds.
    """
    pareto = ParetoForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=2,
                                    random_state=SEED+6, n_jobs=8)
    if _PARETO_POOL is not None:
        pareto.set_pool(_PARETO_POOL)
    return [
        CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=7,
                           l2_leaf_reg=3.0, random_seed=SEED+2,
                           loss_function="Logloss", eval_metric="PRAUC",
                           allow_writing_files=False, verbose=False),
        RandomForestClassifier(n_estimators=700, max_depth=12, min_samples_leaf=1,
                               class_weight="balanced_subsample",
                               random_state=SEED+4, n_jobs=4),
        pareto,
    ]

# ── reward function ───────────────────────────────────────────────────────────
def metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    if labels.max() == labels.min():
        return dict(ap=0.0, recall=0.0, fpr=0.0, reward=0.0, cliff=1)
    ap  = float(average_precision_score(labels, scores))
    p   = (scores >= 0.5).astype(int)
    tp  = int(((p==1)&(labels==1)).sum())
    fp  = int(((p==1)&(labels==0)).sum())
    tn  = int(((p==0)&(labels==0)).sum())
    fn  = int(((p==0)&(labels==1)).sum())
    fpr = fp / max(tn+fp, 1)
    rec = tp / max(tp+fn, 1)
    saf = 0.0 if fpr >= 0.10 else (1-fpr)**2
    return dict(ap=ap, recall=rec, fpr=fpr,
                reward=(0.65*ap + 0.35*rec)*saf,
                cliff=int(fpr >= 0.10))

# ── LODO runner ───────────────────────────────────────────────────────────────
def lodo(model_fn: callable, label: str,
         human_weight: float = 1.0) -> list[dict]:
    results = []
    n = len(uniq)
    for fold_i, d in enumerate(uniq):
        te_idx = [i for i,x in enumerate(dates) if x==d]
        tr_idx = [i for i,x in enumerate(dates) if x!=d]
        y_tr = y[tr_idx];  y_te = y[te_idx]
        if len(set(y_tr)) < 2:
            print(f"  [{label}] skip {d}: only one class in train")
            continue

        # sample weights for training
        w_tr = np.where(y_tr==0, human_weight, 1.0).astype(np.float64)

        ests = model_fn()
        acc  = np.zeros(len(te_idx))
        t0   = time.time()
        for est in ests:
            try:
                est.fit(X[tr_idx], y_tr, sample_weight=w_tr)
            except TypeError:
                est.fit(X[tr_idx], y_tr)
            p = est.predict_proba(X[te_idx])
            acc += (p[:,1] if p.ndim==2 else np.asarray(p))
        scores = acc / len(ests)
        elapsed = time.time() - t0

        r = metrics(scores, y_te)
        r["date"] = d
        r["n_bot"] = int(y_te.sum())
        r["n_hum"] = int((y_te==0).sum())
        results.append(r)
        print(f"  [{label}] {d}  AP={r['ap']:.4f}  Reward={r['reward']:.4f}"
              f"  FPR={r['fpr']:.3f}  Recall={r['recall']:.3f}"
              f"  {'CLIFF' if r['cliff'] else '     '}  ({elapsed:.0f}s)")
    return results

# ── run both ──────────────────────────────────────────────────────────────────
print(f"{'='*60}")
print("Running LODO for v10 (3×LightGBM + ExtraTrees + RF, no weights)...")
print(f"{'='*60}")
t_start = time.time()
res_v10 = lodo(make_v10, "v10", human_weight=1.0)
print(f"\nv10 LODO done in {time.time()-t_start:.0f}s\n")

print(f"{'='*60}")
print("Running LODO for pareto_v3 (CatBoost + RF + ParetoForest, hw=2.0)...")
print(f"{'='*60}")
t_start = time.time()
_ctx = _mp.get_context("fork")
_PARETO_POOL = _ctx.Pool(8)   # persistent pool — created once for all 37 folds
try:
    res_par = lodo(make_pareto, "pareto", human_weight=2.0)
finally:
    _PARETO_POOL.terminate()
    _PARETO_POOL = None
print(f"\npareto LODO done in {time.time()-t_start:.0f}s\n")

# ── aggregate & print ─────────────────────────────────────────────────────────
def agg(results: list[dict]) -> dict:
    def m(k): return float(np.mean([r[k] for r in results]))
    return dict(AP=m("ap"), Recall=m("recall"), FPR=m("fpr"),
                Reward=m("reward"),
                Cliff=100*sum(r["cliff"] for r in results)/len(results),
                n=len(results))

a10 = agg(res_v10)
ap  = agg(res_par)

print()
print(f"{'='*76}")
print(f"FAIR LODO SUMMARY  ({len(uniq)} dates, both models retrained per fold)")
print(f"{'='*76}")
print(f"{'Model':<40} {'AP':>7} {'Recall':>7} {'FPR':>7} {'Reward':>8} {'Cliff%':>7}")
print("-"*76)
print(f"{'v10  (3×LGB + ET + RF,   hw=1.0)':<40} {a10['AP']:7.4f} {a10['Recall']:7.4f} {a10['FPR']:7.4f} {a10['Reward']:8.4f} {a10['Cliff%']:7.1f}")
print(f"{'par  (CatBoost+RF+Pareto, hw=2.0)':<40} {ap['AP']:7.4f} {ap['Recall']:7.4f} {ap['FPR']:7.4f} {ap['Reward']:8.4f} {ap['Cliff%']:7.1f}")

delta_ap     = ap['AP']     - a10['AP']
delta_reward = ap['Reward'] - a10['Reward']
delta_fpr    = ap['FPR']    - a10['FPR']
print(f"\n{'Δ pareto - v10':<40} {delta_ap:+7.4f} {'':>7} {delta_fpr:+7.4f} {delta_reward:+8.4f}")
print()

# ── per-date breakdown ────────────────────────────────────────────────────────
print(f"\n{'Date':<12} {'bots':>5} {'hum':>4}  {'v10 AP':>7} {'v10 Rew':>8} {'par AP':>7} {'par Rew':>8}  winner")
print("-"*65)
d2p = {r["date"]: r for r in res_par}
d2v = {r["date"]: r for r in res_v10}
v10_wins = par_wins = ties = 0
for d in uniq:
    v = d2v.get(d); p = d2p.get(d)
    if not v or not p:
        continue
    w = "v10" if v["ap"]>p["ap"]+1e-4 else ("par" if p["ap"]>v["ap"]+1e-4 else "tie")
    if w=="v10": v10_wins+=1
    elif w=="par": par_wins+=1
    else: ties+=1
    print(f"{d:<12} {v['n_bot']:>5} {v['n_hum']:>4}  "
          f"{v['ap']:7.4f} {v['reward']:8.4f}  "
          f"{p['ap']:7.4f} {p['reward']:8.4f}  {w}")

print(f"\nDate-level AP wins:  v10={v10_wins}  pareto={par_wins}  tie={ties} / {len(uniq)} total dates")
