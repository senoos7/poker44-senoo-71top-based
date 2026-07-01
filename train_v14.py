"""Build v14 = v10 averaged ensemble over base + REPETITION-INVARIANT features (no cx_).
LODO cross-date per-date-mean AP (live proxy) vs base-only. Deploy candidate.
Run:  PYTHONPATH=. .venv/bin/python train_v14.py
"""
import sys, time, warnings, numpy as np, joblib
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import training.build_dataset as bd
from poker44_bump.features import chunk_features as base_cf
from poker44_bump.features_repeat import _repeat_feats
from poker44_bump.model_v14 import V14Model
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb

def make_estimators():
    return [
        lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4, min_child_samples=40,
                           subsample=0.7, colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=2.0, random_state=1, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=63, min_child_samples=30,
                           subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, random_state=2, n_jobs=4, verbose=-1),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=3, n_jobs=4),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5, random_state=4, n_jobs=4),
    ]

exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks=[e["chunk"] for e in exs]; y=np.array([int(e["label"]) for e in exs])
dates=[e["source_date"] for e in exs]; uniq=sorted(set(dates))
t0=time.time(); fd=[{**base_cf(c), **_repeat_feats(c)} for c in chunks]; fdt=(time.time()-t0)/len(chunks)*1000
names=sorted({k for d in fd for k in d})
rp_names=[n for n in names if n.startswith("rp_")]
X=np.asarray([[float(d.get(n,0.0)) for n in names] for d in fd])
print(f"[v14] {len(exs)} chunks / {len(uniq)} dates / {len(names)} feats ({len(rp_names)} rp_) | feat {fdt:.2f} ms/chunk", flush=True)

def lodo(cols):
    oof=np.full(len(y),np.nan)
    for d in uniq:
        te=[i for i,x in enumerate(dates) if x==d]; tr=[i for i,x in enumerate(dates) if x!=d]
        if len(set(y[tr]))<2: continue
        acc=np.zeros(len(te))
        for e in make_estimators():
            e.fit(X[np.ix_(tr,cols)],y[tr]); acc+=e.predict_proba(X[np.ix_(te,cols)])[:,1]
        oof[te]=acc/5.0
    aps=[average_precision_score(y[[i for i,x in enumerate(dates) if x==d]],
         oof[[i for i,x in enumerate(dates) if x==d]]) for d in uniq
         if len(set(y[[i for i,x in enumerate(dates) if x==d]]))>1]
    return float(np.mean(aps))

base_cols=[i for i,n in enumerate(names) if not n.startswith("rp_")]
all_cols=list(range(len(names)))
ap_base=lodo(base_cols); ap_v14=lodo(all_cols)
print(f"[v14] LODO ensemble per-date AP:  base={ap_base:.4f}  base+rp(v14)={ap_v14:.4f}  (+{ap_v14-ap_base:.4f})", flush=True)

# final fit on all data over base+rp
ests=make_estimators()
for e in ests: e.fit(X,y)
model=V14Model(ests, names, topk_cfg={"positive_fraction":0.15},
               metadata={"model_version":"v14-repeat-ensemble","n_estimators":len(ests),
                         "n_feats":len(names),"n_rp":len(rp_names),"lodo_perdate_ap":round(ap_v14,4)})
model.metadata["model_version"]="v14-repeat-ensemble"
joblib.dump(model,"models/bump_model_v14.joblib")

mm=joblib.load("models/bump_model_v14.joblib")
t0=time.time(); sc=mm.predict_chunk_scores(chunks[:200]); dt=(time.time()-t0)/200*1000
pos=sum(1 for s in sc if s>0.5)
print(f"[v14] saved bump_model_v14.joblib | {dt:.2f} ms/chunk | pos {pos}/200 (~frac {pos/200:.2f}) | range [{min(sc):.3f},{max(sc):.3f}]", flush=True)
print("[v14] rp_ feats:", rp_names, flush=True)
