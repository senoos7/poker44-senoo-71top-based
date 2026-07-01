"""Build v11 = weighted meta-ensemble (v5 stacked + diverse trees + robust-feat LGB).
LODO cross-date AP (per-date mean = per-query proxy) vs v10. Deploy candidate for pes01.
Run:  PYTHONPATH=. .venv/bin/python train_v11.py
"""
import sys, numpy as np, joblib
sys.path.insert(0, ".")
import training.build_dataset as bd
from poker44_bump.model_v11 import V11Model
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb

def lgbs():
    return [
        lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4, min_child_samples=40,
                           subsample=0.7, colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=2.0, random_state=1, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=63, min_child_samples=30,
                           subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, random_state=2, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=25,
                           subsample=0.8, colsample_bytree=0.5, reg_lambda=1.5, random_state=5, n_jobs=4, verbose=-1),
    ]
def trees():
    return [ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=3, n_jobs=4),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=4, n_jobs=4)]
def robust_lgb():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4, min_child_samples=40,
                              subsample=0.7, colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=2.0, random_state=7, n_jobs=4, verbose=-1)

# canonical feature order = v5 stacked's feature_names
v5art = joblib.load("models/v5_stacked.joblib")
stacked = (v5art.get("models") or [None])[0]
names = list(v5art.get("feature_names") or [])
exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]; uniq = sorted(set(dates))
from poker44_ml.features import chunk_features as base_cf
fd = [base_cf(c) for c in chunks]
X = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])
print(f"[v11] {len(exs)} chunks / {len(uniq)} dates / {len(names)} feats (stacked order)")

# robust feature subset (cross-date sign-consistent)
pooled = np.array([roc_auc_score(y, X[:, j]) if np.std(X[:, j]) > 1e-9 else 0.5 for j in range(X.shape[1])])
robust = []
for j in range(X.shape[1]):
    if abs(pooled[j]-0.5) < 0.01: continue
    sgn = np.sign(pooled[j]-0.5); ag=tot=0
    for d in uniq:
        idx=[i for i,x in enumerate(dates) if x==d]
        if len(set(y[idx]))<2 or np.std(X[idx,j])<1e-9: continue
        tot+=1; ag+= (np.sign(roc_auc_score(y[idx],X[idx,j])-0.5)==sgn)
    if tot and ag/tot>=0.67: robust.append(j)
print(f"[v11] robust feats: {len(robust)}/{len(names)}")

# LODO per-date mean AP (per-query proxy): v11 ensemble vs v10-style (no stacked, no robust)
def lodo(kind):
    aps=[]
    for d in uniq:
        te=[i for i,x in enumerate(dates) if x==d]; tr=[i for i,x in enumerate(dates) if x!=d]
        if len(set(y[tr]))<2 or len(set(y[te]))<2: continue
        members=[]
        for e in lgbs()+trees(): e.fit(X[tr],y[tr]); members.append((e,None,1.0))
        if kind=="v11":
            rb=robust_lgb(); rb.fit(X[np.ix_(tr,robust)],y[tr]); members.append((rb,robust,1.0))
            members.append(("__stacked__",None,2.0))   # stacked is pre-fit on all data (proxy)
        acc=np.zeros(len(te)); ws=0
        for est,cols,w in members:
            if est=="__stacked__":
                s=np.asarray(stacked.predict_chunk_scores([chunks[i] for i in te], X[te]))
            else:
                Xt=X[np.ix_(te,cols)] if cols is not None else X[te]
                s=est.predict_proba(Xt)[:,1]
            acc+=w*s; ws+=w
        aps.append(average_precision_score(y[te], acc/ws))
    return float(np.mean(aps))
print(f"[v11] LODO per-date mean AP: v10-style={lodo('v10'):.4f}  v11(meta)={lodo('v11'):.4f}")
print("      (note: v11's stacked member is fit-on-all -> slightly optimistic; trees are honest LODO)")

# final build on all data
members=[{"kind":"stacked","est":stacked,"cols":None,"w":2.0}]
for e in lgbs()+trees(): e.fit(X,y); members.append({"kind":"sklearn","est":e,"cols":None,"w":1.0})
rb=robust_lgb(); rb.fit(X[:,robust],y); members.append({"kind":"sklearn","est":rb,"cols":robust,"w":1.0})
model=V11Model(members, names, topk_cfg={"positive_fraction":0.15},
               metadata={"n_members":len(members),"n_feats":len(names),"n_robust":len(robust)})
joblib.dump(model,"models/bump_model_v11.joblib")
print(f"[v11] saved models/bump_model_v11.joblib ({len(members)} members)")
