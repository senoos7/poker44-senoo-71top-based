"""Build v10 = diverse averaged ensemble (no meta-learner, no cx_) for live
generalization. Reports LODO cross-date AP vs single-model baselines.
Run:  PYTHONPATH=. .venv/bin/python train_v10.py
"""
import sys, numpy as np, joblib
sys.path.insert(0, ".")
import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.model_v10 import V10Model
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb

def make_estimators():
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

exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]; uniq = sorted(set(dates))
fd = [base_cf(c) for c in chunks]
names = sorted({k for d in fd for k in d})
X = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])
print(f"[v10] {len(exs)} chunks / {len(uniq)} dates / {len(names)} base feats (no cx_)")

def lodo_ensemble():
    yt, ys = [], []
    for d in uniq:
        te = [i for i, x in enumerate(dates) if x == d]; tr = [i for i, x in enumerate(dates) if x != d]
        if len(set(y[tr])) < 2: continue
        ests = make_estimators()
        acc = np.zeros(len(te))
        for e in ests:
            e.fit(X[tr], y[tr]); acc += e.predict_proba(X[te])[:, 1]
        ys += list(acc / len(ests)); yt += list(y[te])
    return average_precision_score(yt, ys)

def lodo_single():
    yt, ys = [], []
    for d in uniq:
        te = [i for i, x in enumerate(dates) if x == d]; tr = [i for i, x in enumerate(dates) if x != d]
        if len(set(y[tr])) < 2: continue
        e = make_estimators()[0]; e.fit(X[tr], y[tr])
        ys += list(e.predict_proba(X[te])[:, 1]); yt += list(y[te])
    return average_precision_score(yt, ys)

print(f"[v10] LODO cross-date AP: single-lgb={lodo_single():.4f}  ENSEMBLE={lodo_ensemble():.4f}")

# final fit on all data
ests = make_estimators()
for e in ests:
    e.fit(X, y)
model = V10Model(ests, names, topk_cfg={"positive_fraction": 0.15},
                 metadata={"n_estimators": len(ests), "n_feats": len(names)})
joblib.dump(model, "models/bump_model_v10.joblib")
print(f"[v10] saved models/bump_model_v10.joblib  ({len(ests)} estimators, {len(names)} feats)")
