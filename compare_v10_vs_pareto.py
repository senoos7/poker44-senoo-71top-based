"""LODO comparison: bump_model_v10 vs all293_cat_rf_pareto_v3.

Run:
    PYTHONPATH=. python -W ignore compare_v10_vs_pareto.py
"""
from __future__ import annotations
import sys, warnings
sys.path.insert(0, "/root/Skip/poker/poker_demo_2")
sys.path.insert(0, ".")   # local packages take priority
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from sklearn.metrics import average_precision_score

import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf

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
print()

# ── load models ──────────────────────────────────────────────────────────────
v10     = joblib.load("models/bump_model_v10.joblib")
pareto_d = joblib.load("/root/Skip/poker/poker_demo_2/models/all293_cat_rf_pareto_v3.joblib")
pareto  = pareto_d["models"][0]   # StackedEnsemble
pareto_feats = pareto_d["feature_names"]  # 293 feature names (same set)

# Build X aligned to pareto feature order
X_pareto = np.asarray([[float(d.get(n, 0.0)) for n in pareto_feats] for d in fd])

# ── reward function (same as head2head.py) ───────────────────────────────────
def reward_at(scores, labels, T=0.5):
    p  = (scores >= T).astype(int)
    yt = labels.astype(int)
    tp = ((p==1)&(yt==1)).sum(); fp = ((p==1)&(yt==0)).sum()
    tn = ((p==0)&(yt==0)).sum(); fn = ((p==0)&(yt==1)).sum()
    fpr = fp / max(tn+fp, 1)
    rec = tp / max(tp+fn, 1)
    ap  = average_precision_score(yt, scores) if yt.max()==1 and yt.min()==0 else 0.0
    saf = 0.0 if fpr >= 0.10 else (1-fpr)**2
    return dict(reward=(0.65*ap+0.35*rec)*saf, ap=ap, recall=rec, fpr=fpr,
                cliff=int(fpr>=0.10))

# ── LODO evaluation ───────────────────────────────────────────────────────────
def lodo(predict_fn, label):
    results = []
    for d in uniq:
        te = [i for i,x in enumerate(dates) if x==d]
        tr = [i for i,x in enumerate(dates) if x!=d]
        if len(set(y[tr])) < 2:
            continue
        scores_te = predict_fn(tr, te)
        r = reward_at(scores_te, y[te])
        r["date"] = d
        results.append(r)
    return results

def predict_v10(tr, te):
    # refit v10 ensemble on tr, predict te
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    import lightgbm as lgb
    ests = [
        lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,
                           min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
                           reg_lambda=1.0,random_state=0,n_jobs=4,verbose=-1),
        lgb.LGBMClassifier(n_estimators=300,learning_rate=0.02,num_leaves=15,max_depth=4,
                           min_child_samples=40,subsample=0.7,colsample_bytree=0.6,
                           reg_lambda=5.0,reg_alpha=2.0,random_state=1,n_jobs=4,verbose=-1),
        lgb.LGBMClassifier(n_estimators=500,learning_rate=0.02,num_leaves=63,
                           min_child_samples=30,subsample=0.7,colsample_bytree=0.7,
                           reg_lambda=2.0,random_state=2,n_jobs=4,verbose=-1),
        ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,max_features=0.5,
                             random_state=3,n_jobs=4),
        RandomForestClassifier(n_estimators=400,min_samples_leaf=5,max_features=0.5,
                               random_state=4,n_jobs=4),
    ]
    acc = np.zeros(len(te))
    for e in ests:
        e.fit(X[tr], y[tr])
        acc += e.predict_proba(X[te])[:,1]
    return acc / len(ests)

def predict_pareto(tr, te):
    # pareto is a pre-trained model — use it directly (no refit)
    return pareto.predict_proba(X_pareto[te])[:,1]

print("Running LODO for v10  (refits per date, may take a few minutes)...")
res_v10    = lodo(predict_v10,    "v10")
print("Running LODO for pareto (pre-trained, fast)...")
res_pareto = lodo(predict_pareto, "pareto")

# ── aggregate stats ───────────────────────────────────────────────────────────
def agg(results):
    return {
        "AP":      np.mean([r["ap"]     for r in results]),
        "Recall":  np.mean([r["recall"] for r in results]),
        "FPR":     np.mean([r["fpr"]    for r in results]),
        "Reward":  np.mean([r["reward"] for r in results]),
        "Cliff%":  100*sum(r["cliff"] for r in results)/len(results),
        "n_dates": len(results),
    }

a10 = agg(res_v10)
ap  = agg(res_pareto)

print()
print(f"{'Model':<32} {'AP':>7} {'Recall':>7} {'FPR':>7} {'Reward':>8} {'Cliff%':>7} {'Dates':>6}")
print("-"*76)
print(f"{'v10 (avg ensemble, LODO refit)':<32} {a10['AP']:7.4f} {a10['Recall']:7.4f} {a10['FPR']:7.4f} {a10['Reward']:8.4f} {a10['Cliff%']:7.1f} {a10['n_dates']:6d}")
print(f"{'pareto_v3 (pre-trained, eval only)':<32} {ap['AP']:7.4f} {ap['Recall']:7.4f} {ap['FPR']:7.4f} {ap['Reward']:8.4f} {ap['Cliff%']:7.1f} {ap['n_dates']:6d}")

print()
print("Note: v10 refits from scratch per date (true LODO).")
print("      pareto_v3 is evaluated without refit (it was trained on 35 of these dates).")
print("      A fair LODO comparison for pareto would require its training code.")

# ── per-date breakdown ────────────────────────────────────────────────────────
print()
print(f"{'Date':<12} {'v10 AP':>8} {'v10 Rew':>9} {'pareto AP':>10} {'pareto Rew':>11}")
print("-"*54)
d2p = {r["date"]: r for r in res_pareto}
for r in res_v10:
    p = d2p.get(r["date"], {})
    print(f"{r['date']:<12} {r['ap']:8.4f} {r['reward']:9.4f} "
          f"{p.get('ap',float('nan')):10.4f} {p.get('reward',float('nan')):11.4f}")
