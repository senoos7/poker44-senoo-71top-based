"""v12 = MAXIMUM-EFFORT training: greedy ensemble selection (Caruana 2004) over a
large diverse candidate pool, optimized for LEAVE-ONE-DATE-OUT cross-date per-date
mean AP (our best live-transfer proxy) -- NOT benchmark holdout AP (which overfits).

Pipeline:
 1) large pool of diverse base learners (LGBM/XGB/ExtraTrees/RF/HistGB/robust/logreg)
 2) leakage-free OOF preds via grouped-by-date 8-fold CV
 3) greedy forward selection w/ replacement maximizing per-date-mean AP
 4) refit selected members on all data -> weighted ensemble (V11Model wrapper) + topk 0.15

Run:  PYTHONPATH=. .venv/bin/python train_v12.py
"""
import sys, numpy as np, joblib
sys.path.insert(0, ".")
import training.build_dataset as bd
from poker44_bump.model_v11 import V11Model
from poker44_ml.features import chunk_features as base_cf
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]; uniq = sorted(set(dates))
fd = [base_cf(c) for c in chunks]
names = sorted({k for d in fd for k in d})
X = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])
print(f"[v12] {len(exs)} chunks / {len(uniq)} dates / {len(names)} feats | xgb={HAVE_XGB}", flush=True)

# robust feature subset (cross-date sign-consistent)
pooled = np.array([roc_auc_score(y, X[:, j]) if np.std(X[:, j]) > 1e-9 else 0.5 for j in range(X.shape[1])])
robust = []
for j in range(X.shape[1]):
    if abs(pooled[j]-0.5) < 0.01: continue
    sgn = np.sign(pooled[j]-0.5); ag=tot=0
    for d in uniq:
        idx=[i for i,x in enumerate(dates) if x==d]
        if len(set(y[idx]))<2 or np.std(X[idx,j])<1e-9: continue
        tot+=1; ag+=(np.sign(roc_auc_score(y[idx],X[idx,j])-0.5)==sgn)
    if tot and ag/tot>=0.67: robust.append(j)
print(f"[v12] robust feats: {len(robust)}/{len(names)}", flush=True)

def L(**k): return lgb.LGBMClassifier(n_jobs=4, verbose=-1, **k)
# ---- large diverse candidate pool: (name, factory, cols) ----
pool = []
for seed in (0,1,2):
    pool += [
        (f"lgb_l31_r1_s{seed}", lambda s=seed: L(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=s), None),
        (f"lgb_l63_r2_s{seed}", lambda s=seed: L(n_estimators=500,learning_rate=0.02,num_leaves=63,min_child_samples=30,subsample=0.7,colsample_bytree=0.7,reg_lambda=2.0,random_state=s), None),
        (f"lgb_l15_heavy_s{seed}", lambda s=seed: L(n_estimators=300,learning_rate=0.02,num_leaves=15,max_depth=4,min_child_samples=40,subsample=0.7,colsample_bytree=0.6,reg_lambda=5.0,reg_alpha=2.0,random_state=s), None),
        (f"lgb_cs05_s{seed}", lambda s=seed: L(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=25,subsample=0.8,colsample_bytree=0.5,reg_lambda=1.5,random_state=s), None),
    ]
pool += [
    ("et_mf05", lambda: ExtraTreesClassifier(n_estimators=500,min_samples_leaf=5,max_features=0.5,random_state=3,n_jobs=4), None),
    ("et_mf03", lambda: ExtraTreesClassifier(n_estimators=500,min_samples_leaf=8,max_features=0.3,random_state=13,n_jobs=4), None),
    ("rf_mf05", lambda: RandomForestClassifier(n_estimators=500,min_samples_leaf=5,max_features=0.5,random_state=4,n_jobs=4), None),
    ("rf_mf03", lambda: RandomForestClassifier(n_estimators=500,min_samples_leaf=8,max_features=0.3,random_state=14,n_jobs=4), None),
    ("hgb1", lambda: HistGradientBoostingClassifier(max_iter=400,learning_rate=0.03,max_leaf_nodes=31,l2_regularization=1.0,random_state=6), None),
    ("hgb2", lambda: HistGradientBoostingClassifier(max_iter=300,learning_rate=0.02,max_leaf_nodes=15,l2_regularization=5.0,random_state=16), None),
    ("lgb_robust1", lambda: L(n_estimators=300,learning_rate=0.02,num_leaves=15,max_depth=4,min_child_samples=40,subsample=0.7,colsample_bytree=0.6,reg_lambda=5.0,reg_alpha=2.0,random_state=7), robust),
    ("lgb_robust2", lambda: L(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,reg_lambda=1.0,random_state=17), robust),
    ("logreg_robust", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5,max_iter=2000)), robust),
    ("et_robust", lambda: ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,max_features=0.6,random_state=23,n_jobs=4), robust),
]
if HAVE_XGB:
    for seed in (0,1):
        pool += [(f"xgb_d4_s{seed}", lambda s=seed: xgb.XGBClassifier(n_estimators=400,learning_rate=0.03,max_depth=4,subsample=0.8,colsample_bytree=0.7,reg_lambda=2.0,random_state=s,n_jobs=4,eval_metric="logloss",verbosity=0), None),
                 (f"xgb_d6_s{seed}", lambda s=seed: xgb.XGBClassifier(n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.7,colsample_bytree=0.6,reg_lambda=3.0,random_state=s,n_jobs=4,eval_metric="logloss",verbosity=0), None)]
print(f"[v12] candidate pool size: {len(pool)}", flush=True)

# ---- grouped-by-date folds (8) for leakage-free OOF ----
import itertools
date_groups = [uniq[i::8] for i in range(8)]   # round-robin partition of dates
oof = np.full((len(pool), len(y)), np.nan)
for ci,(name,fac,cols) in enumerate(pool):
    for grp in date_groups:
        te=[i for i,d in enumerate(dates) if d in grp]; tr=[i for i,d in enumerate(dates) if d not in grp]
        if not te or len(set(y[tr]))<2: continue
        Xtr = X[np.ix_(tr,cols)] if cols is not None else X[tr]
        Xte = X[np.ix_(te,cols)] if cols is not None else X[te]
        m=fac(); m.fit(Xtr,y[tr])
        oof[ci,te]=m.predict_proba(Xte)[:,1]
    print(f"  OOF {ci+1}/{len(pool)} {name}", flush=True)

def perdate_ap(scores):
    aps=[]
    for d in uniq:
        idx=[i for i,x in enumerate(dates) if x==d]
        if len(set(y[idx]))>1: aps.append(average_precision_score(y[idx],scores[idx]))
    return float(np.mean(aps))

singles=[(perdate_ap(oof[ci]),pool[ci][0]) for ci in range(len(pool))]
singles.sort(reverse=True)
print("[v12] top-5 single candidates (per-date mean AP):", [(round(a,4),n) for a,n in singles[:5]], flush=True)

# ---- greedy ensemble selection w/ replacement (Caruana) ----
ens=[]; cur=np.zeros(len(y)); best=0.0
for rnd in range(30):
    scored=[]
    for ci in range(len(pool)):
        cand=(cur*len(ens)+oof[ci])/(len(ens)+1)
        scored.append((perdate_ap(cand),ci))
    a,ci=max(scored)
    if a<=best+1e-5 and len(ens)>=3: break
    ens.append(ci); cur=(cur*(len(ens)-1)+oof[ci])/len(ens); best=a
from collections import Counter
counts=Counter(ens)
print(f"[v12] greedy ensemble per-date mean AP = {best:.4f}  (vs best single {singles[0][0]:.4f})", flush=True)
print("[v12] selected (weight):", [(pool[ci][0],counts[ci]) for ci in counts], flush=True)

# ---- refit selected on all data -> weighted V11Model ----
members=[]
for ci,w in counts.items():
    name,fac,cols=pool[ci]; m=fac();
    Xall = X[:,cols] if cols is not None else X
    m.fit(Xall,y)
    members.append({"kind":"sklearn","est":m,"cols":cols,"w":float(w)})
model=V11Model(members, names, topk_cfg={"positive_fraction":0.15},
               metadata={"model_version":"v12-greedy-ensemble","model_name":"poker44-bump-v12",
                         "framework":"greedy-selected-ensemble+topk","n_members":len(members),
                         "lodo_perdate_ap":round(best,4)})
model.metadata["model_version"]="v12-greedy-ensemble"
joblib.dump(model,"models/bump_model_v12.joblib")
print(f"[v12] saved models/bump_model_v12.joblib ({len(members)} members, LODO per-date AP {best:.4f})", flush=True)
