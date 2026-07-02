"""Fair LODO comparison with _topk_squeeze() head applied to both models.

Same training as lodo_fair_compare.py (retrain per fold from scratch).
After each fold's raw scores are computed, _topk_squeeze(frac=0.15) is applied
to simulate exactly how the miner serves scores in production.

Evaluation at threshold=0.5 (topk head guarantees top-15% > 0.5, rest < 0.5):
  - FPR   = fraction of human chunks in top-15% of that batch
  - Recall = fraction of bot chunks in top-15%
  - AP     = average precision using squeezed scores (ranking within top-k preserved)
  - Reward = (0.65*AP + 0.35*Recall) * (1-FPR)^2  if FPR < 0.10 else 0

Run:
    PYTHONPATH=. python -W ignore lodo_topk_compare.py 2>&1 | tee lodo_topk.log
"""
from __future__ import annotations
import sys, time, math, multiprocessing as _mp
sys.path.insert(0, "/root/Skip/poker/poker_demo_2")
sys.path.insert(0, ".")

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.model_v5 import _topk_squeeze
from poker44_bump.fast_pareto import FastParetoForestClassifier

# ── benchmark data ────────────────────────────────────────────────────────────
exs   = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]
y      = np.array([int(e["label"]) for e in exs])
dates  = [e["source_date"] for e in exs]
uniq   = sorted(set(dates))

fd    = [base_cf(c) for c in chunks]
names = sorted({k for d in fd for k in d})
X     = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])

TOPK_FRAC = 0.15
SEED = 42

print(f"[data] {len(exs)} chunks / {len(uniq)} dates / {len(names)} features")
print(f"       labels: {int(y.sum())} bot / {int((y==0).sum())} human")
print(f"       topk_frac={TOPK_FRAC}  (top-{TOPK_FRAC*100:.0f}% of each batch flagged as bot)")
print()

# ── topk-aware metrics ────────────────────────────────────────────────────────
def topk_metrics(raw_scores: np.ndarray, labels: np.ndarray) -> dict:
    """Apply _topk_squeeze then evaluate at threshold=0.5."""
    squeezed = np.array(_topk_squeeze(raw_scores, TOPK_FRAC))
    # At threshold=0.5: top-k chunks are predicted bot (score > 0.5)
    pred_bot = squeezed > 0.5
    n_bot = int(labels.sum())
    n_hum = int((labels == 0).sum())
    tp = int(((pred_bot) & (labels == 1)).sum())
    fp = int(((pred_bot) & (labels == 0)).sum())
    fpr    = fp / max(n_hum, 1)
    recall = tp / max(n_bot, 1)
    # AP uses squeezed scores (ranking within top-k is preserved)
    if labels.max() == labels.min():
        ap = 0.0
    else:
        ap = float(average_precision_score(labels, squeezed))
    saf    = 0.0 if fpr >= 0.10 else (1.0 - fpr) ** 2
    reward = (0.65 * ap + 0.35 * recall) * saf
    k      = max(0, min(len(labels), int(math.floor(len(labels) * TOPK_FRAC))))
    return dict(ap=ap, recall=recall, fpr=fpr, reward=reward,
                cliff=int(fpr >= 0.10), k=k, n=len(labels),
                n_bot=n_bot, n_hum=n_hum)

# also raw metrics (no topk head, threshold=0.5) for side-by-side
def raw_metrics(raw_scores: np.ndarray, labels: np.ndarray) -> dict:
    if labels.max() == labels.min():
        return dict(ap=0.0, recall=0.0, fpr=0.0, reward=0.0, cliff=1)
    ap  = float(average_precision_score(labels, raw_scores))
    p   = (raw_scores >= 0.5).astype(int)
    tp  = int(((p==1)&(labels==1)).sum())
    fp  = int(((p==1)&(labels==0)).sum())
    tn  = int(((p==0)&(labels==0)).sum())
    fn  = int(((p==0)&(labels==1)).sum())
    fpr = fp / max(tn+fp, 1)
    rec = tp / max(tp+fn, 1)
    saf = 0.0 if fpr >= 0.10 else (1.0-fpr)**2
    return dict(ap=ap, recall=rec, fpr=fpr,
                reward=(0.65*ap+0.35*rec)*saf, cliff=int(fpr>=0.10))

# ── model factories ───────────────────────────────────────────────────────────
def make_v10():
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

_PARETO_POOL = None

def make_pareto():
    pareto = FastParetoForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=2,
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

# ── LODO runner ───────────────────────────────────────────────────────────────
def lodo(model_fn, label, human_weight=1.0):
    results = []
    for d in uniq:
        te_idx = [i for i,x in enumerate(dates) if x==d]
        tr_idx = [i for i,x in enumerate(dates) if x!=d]
        y_tr = y[tr_idx]; y_te = y[te_idx]
        if len(set(y_tr)) < 2:
            continue
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
        raw = acc / len(ests)
        elapsed = time.time() - t0

        rm  = raw_metrics(raw, y_te)
        tkm = topk_metrics(raw, y_te)
        k   = tkm["k"]
        print(f"  [{label}] {d}  "
              f"raw AP={rm['ap']:.4f} Rew={rm['reward']:.4f} {'C' if rm['cliff'] else ' '}  │  "
              f"topk(k={k}/{len(te_idx)}) AP={tkm['ap']:.4f} Rew={tkm['reward']:.4f} Rec={tkm['recall']:.3f} "
              f"FPR={tkm['fpr']:.3f} {'CLIFF' if tkm['cliff'] else '     '}  ({elapsed:.0f}s)")
        results.append({"date": d, "raw": rm, "topk": tkm})
    return results

# ── aggregate helper ──────────────────────────────────────────────────────────
def agg(results, key):
    rs = [r[key] for r in results]
    return {m: float(np.mean([r[m] for r in rs]))
            for m in ["ap","recall","fpr","reward"]} | \
           {"cliff_pct": 100*sum(r["cliff"] for r in rs)/len(rs),
            "n": len(rs)}

# ── run ───────────────────────────────────────────────────────────────────────
print("="*70)
print("v10  LODO (3×LightGBM + ExtraTrees + RF, hw=1.0)")
print("="*70)
t0 = time.time()
res_v10 = lodo(make_v10, "v10", human_weight=1.0)
print(f"\nv10 done in {time.time()-t0:.0f}s\n")

print("="*70)
print("pareto_v3 LODO (CatBoost + RF + FastPareto, hw=2.0)")
print("="*70)
t0 = time.time()
_ctx = _mp.get_context("fork")
_PARETO_POOL = _ctx.Pool(8)
try:
    res_par = lodo(make_pareto, "par", human_weight=2.0)
finally:
    _PARETO_POOL.terminate()
    _PARETO_POOL = None
print(f"\npareto done in {time.time()-t0:.0f}s\n")

# ── summary ───────────────────────────────────────────────────────────────────
av_raw  = agg(res_v10, "raw");  av_tk = agg(res_v10, "topk")
ap_raw  = agg(res_par, "raw");  ap_tk = agg(res_par, "topk")

W = 44
print()
print("="*90)
print(f"FAIR LODO SUMMARY  ({len(uniq)} dates, both models retrained per fold, topk_frac={TOPK_FRAC})")
print("="*90)
print(f"\n{'── Raw scores (threshold=0.5, as before) ──':}")
print(f"{'Model':<{W}} {'AP':>7} {'Recall':>7} {'FPR':>7} {'Reward':>8} {'Cliff%':>7}")
print("-"*80)
print(f"{'v10  (3×LGB+ET+RF,   hw=1.0)':<{W}} {av_raw['ap']:7.4f} {av_raw['recall']:7.4f} {av_raw['fpr']:7.4f} {av_raw['reward']:8.4f} {av_raw['cliff_pct']:7.1f}")
print(f"{'par  (CatBoost+RF+Pareto, hw=2.0)':<{W}} {ap_raw['ap']:7.4f} {ap_raw['recall']:7.4f} {ap_raw['fpr']:7.4f} {ap_raw['reward']:8.4f} {ap_raw['cliff_pct']:7.1f}")

print(f"\n{'── With _topk_squeeze(frac=0.15) — production head ──':}")
print(f"{'Model':<{W}} {'AP':>7} {'Recall':>7} {'FPR':>7} {'Reward':>8} {'Cliff%':>7}")
print("-"*80)
print(f"{'v10  + topk':<{W}} {av_tk['ap']:7.4f} {av_tk['recall']:7.4f} {av_tk['fpr']:7.4f} {av_tk['reward']:8.4f} {av_tk['cliff_pct']:7.1f}")
print(f"{'par  + topk':<{W}} {ap_tk['ap']:7.4f} {ap_tk['recall']:7.4f} {ap_tk['fpr']:7.4f} {ap_tk['reward']:8.4f} {ap_tk['cliff_pct']:7.1f}")
dap = ap_tk['ap']-av_tk['ap']; drw = ap_tk['reward']-av_tk['reward']; dfpr = ap_tk['fpr']-av_tk['fpr']
print(f"{'Δ par+topk − v10+topk':<{W}} {dap:+7.4f} {'':>7} {dfpr:+7.4f} {drw:+8.4f}")

# ── per-date table ────────────────────────────────────────────────────────────
print()
print(f"\n{'Date':<12} {'k':>3}  "
      f"{'v10 raw':>8} {'v10+tk':>7} {'v10 Rew':>8}  "
      f"{'par raw':>8} {'par+tk':>7} {'par Rew':>8}  winner(topk)")
print("-"*82)
d2p = {r["date"]: r for r in res_par}
v10w = parw = ties = 0
for r in res_v10:
    p = d2p.get(r["date"], {})
    va = r["topk"]["ap"]; pa = p.get("topk",{}).get("ap", float("nan"))
    w = "v10" if va > pa+1e-4 else ("par" if pa > va+1e-4 else "tie")
    if w=="v10": v10w+=1
    elif w=="par": parw+=1
    else: ties+=1
    k  = r["topk"]["k"]
    cv = "C" if r["topk"]["cliff"] else " "
    cp = "C" if p.get("topk",{}).get("cliff") else " "
    print(f"{r['date']:<12} {k:>3}  "
          f"{r['raw']['ap']:8.4f} {r['topk']['ap']:7.4f} {r['topk']['reward']:8.4f}{cv}  "
          f"{p.get('raw',{}).get('ap',float('nan')):8.4f} {pa:7.4f} {p.get('topk',{}).get('reward',float('nan')):8.4f}{cp}  {w}")

print()
print(f"topk AP wins: v10={v10w}  pareto={parw}  tie={ties} / {len(uniq)} dates")
