"""v13 = HYBRID of the two LIVE WINNERS (06-27 v2.0 daily epoch, controlled same-window A/B):
   v10 diverse averaged ensemble (0.419, #6) ~= v5 stacked (0.415, #8)  >>  v12-XGB (0.331) > v9-lean (0.295)
The LODO proxy had wrongly favored v12's single XGBoost; LIVE says diverse averaging/stacking
transfer best, single high-capacity overfits, and the robust-only direction (v9) underperforms.
phasberg (old #1) zeroed on the FPR cliff under deeper daily eval -> our topk head (FPR->0) is the edge.

v13 = weighted average of:  v5 stacked (raw probs, weight 5)  +  v10's exact 5 base learners (weight 1 each)
   => predict_raw = 0.5*v5_raw + 0.5*mean(v10 trees) = a 50/50 blend of the two co-winners.
No robust-only member (v9 lost live). No cx_ (overfits live). Conservative topk frac 0.15 (FPR-safe).
Combining two good, decorrelated models reduces variance -> the cure for the benchmark->live gap.

Run:  PYTHONPATH=. .venv/bin/python train_v13.py
"""
import sys, time, numpy as np, joblib
sys.path.insert(0, ".")
import training.build_dataset as bd
from poker44_bump.model_v11 import V11Model
from poker44_ml.features import chunk_features as base_cf
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb

# v10's EXACT 5 base learners (so the tree-half reproduces v10 precisely)
def make_trees():
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

# canonical feature order = v5 stacked's feature_names (stacked member needs this order)
v5art = joblib.load("models/v5_stacked.joblib")
stacked = (v5art.get("models") or [None])[0]
names = list(v5art.get("feature_names") or [])
exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]; uniq = sorted(set(dates))
fd = [base_cf(c) for c in chunks]
X = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])
print(f"[v13] {len(exs)} chunks / {len(uniq)} dates / {len(names)} feats (v5 order)", flush=True)

def perdate_ap(idx_scores):
    aps = []
    for d in uniq:
        te = [i for i, x in enumerate(dates) if x == d]
        if len(set(y[te])) > 1:
            aps.append(average_precision_score(y[te], idx_scores[te]))
    return float(np.mean(aps))

# LODO per-date mean AP: v10-trees (honest) vs v13-blend (stacked is fit-on-all -> optimistic on that half)
oof_trees = np.full(len(y), np.nan); oof_blend = np.full(len(y), np.nan)
for d in uniq:
    te = [i for i, x in enumerate(dates) if x == d]; tr = [i for i, x in enumerate(dates) if x != d]
    if len(set(y[tr])) < 2: continue
    acc = np.zeros(len(te))
    for e in make_trees():
        e.fit(X[tr], y[tr]); acc += e.predict_proba(X[te])[:, 1]
    tree_mean = acc / 5.0
    oof_trees[te] = tree_mean
    s_stack = np.asarray(stacked.predict_chunk_scores([chunks[i] for i in te], X[te]), dtype=float)
    oof_blend[te] = 0.5 * s_stack + 0.5 * tree_mean
m = ~np.isnan(oof_trees)
print(f"[v13] LODO per-date mean AP:  v10-trees(honest)={perdate_ap(np.where(m,oof_trees,0)):.4f}  "
      f"v13-blend={perdate_ap(np.where(m,oof_blend,0)):.4f}  (blend's stacked half fit-on-all=optimistic)", flush=True)

# final build on all data
trees = make_trees()
for e in trees: e.fit(X, y)
members = [{"kind": "stacked", "est": stacked, "cols": None, "w": 5.0}]
members += [{"kind": "sklearn", "est": e, "cols": None, "w": 1.0} for e in trees]
model = V11Model(members, names, topk_cfg={"positive_fraction": 0.15},
                 metadata={"model_version": "v13-hybrid-v5v10", "model_name": "poker44-bump-v13",
                           "framework": "hybrid(0.5*v5stacked+0.5*v10avg)+topk",
                           "n_members": len(members), "blend": "0.5*v5 + 0.5*v10"})
model.metadata["model_version"] = "v13-hybrid-v5v10"
joblib.dump(model, "models/bump_model_v13.joblib")

# sanity: load, predict, time, verify FPR-safe head
mm = joblib.load("models/bump_model_v13.joblib")
t0 = time.time(); sc = mm.predict_chunk_scores(chunks[:200]); dt = (time.time() - t0) / 200 * 1000
pos = sum(1 for s in sc if s > 0.5)
print(f"[v13] saved bump_model_v13.joblib ({len(members)} members) | {dt:.2f} ms/chunk | "
      f"pos {pos}/200 (~frac {pos/200:.2f}) | range [{min(sc):.3f},{max(sc):.3f}]", flush=True)
