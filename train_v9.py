"""Build v9 = lean/robust model to chase LIVE AP.
1) 293 base feats (no cx_); 2) keep only cross-date-ROBUST feats (sign consistent
across dates); 3) heavily-regularized LightGBM; 4) LODO cross-date AP sanity vs a
v5-like baseline; 5) save deployable artifact. Live AP can't be measured offline
-- this is a robustness-by-design hypothesis to validate live vs v5.

Run:  PYTHONPATH=. .venv/bin/python train_v9.py
"""
import sys, numpy as np, joblib
sys.path.insert(0, ".")
import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.model_v9 import V9Model
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

# heavy regularization: shallow, strong L1/L2, big leaves-min, subsample
LGB_HEAVY = dict(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
                 min_child_samples=40, subsample=0.7, colsample_bytree=0.6,
                 reg_lambda=5.0, reg_alpha=2.0, random_state=0, n_jobs=4, verbose=-1)
LGB_V5LIKE = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1)

exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]; uniq = sorted(set(dates))
fd = [base_cf(c) for c in chunks]
names = sorted({k for d in fd for k in d})
X = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])
print(f"[v9] {len(exs)} chunks / {len(uniq)} dates / {len(names)} base feats (no cx_)")

# robust feature screen: per-date AUC sign consistency + non-trivial pooled signal
pooled = np.array([roc_auc_score(y, X[:, j]) if np.std(X[:, j]) > 1e-9 else 0.5
                   for j in range(X.shape[1])])
robust = []
for j in range(X.shape[1]):
    if abs(pooled[j] - 0.5) < 0.01:
        continue
    sgn = np.sign(pooled[j] - 0.5); agree = 0; tot = 0
    for d in uniq:
        idx = [i for i, x in enumerate(dates) if x == d]
        yy = y[idx]; col = X[idx, j]
        if len(set(yy)) < 2 or np.std(col) < 1e-9:
            continue
        tot += 1
        if np.sign(roc_auc_score(yy, col) - 0.5) == sgn:
            agree += 1
    if tot and agree / tot >= 0.67:
        robust.append(j)
rnames = [names[j] for j in robust]
print(f"[v9] robust features kept: {len(rnames)}/{len(names)}")

def lodo(cols, params):
    yt, ys = [], []
    for d in uniq:
        te = [i for i, x in enumerate(dates) if x == d]; tr = [i for i, x in enumerate(dates) if x != d]
        if len(set(y[tr])) < 2: continue
        m = lgb.LGBMClassifier(**params).fit(X[np.ix_(tr, cols)], y[tr])
        ys += list(m.predict_proba(X[np.ix_(te, cols)])[:, 1]); yt += list(y[te])
    return average_precision_score(yt, ys)

allcols = list(range(X.shape[1]))
print(f"[v9] LODO cross-date AP:")
print(f"     v5-like (293 feats, moderate reg) = {lodo(allcols, LGB_V5LIKE):.4f}")
print(f"     v9 (robust feats, heavy reg)       = {lodo(robust, LGB_HEAVY):.4f}")

# final fit on all data, robust subset, heavy reg
clf = lgb.LGBMClassifier(**LGB_HEAVY).fit(X[:, robust], y)
model = V9Model(clf, rnames, topk_cfg={"positive_fraction": 0.15},
                metadata={"n_robust_feats": len(rnames), "n_base_feats": len(names),
                          "lgb_params": "heavy-reg"})
joblib.dump(model, "models/bump_model_v9.joblib")
print(f"[v9] saved models/bump_model_v9.joblib  feats={len(rnames)} (robust subset of {len(names)})")
