"""Build deployable v7 = date-robust LightGBM on (combined base feats + nLLR)
+ topk safety head. Reports an honest leave-one-date-out cross-date AP (live
proxy) and a most-recent-dates holdout AP before saving.

Run:  PYTHONPATH=. .venv/bin/python train_v7.py
"""
import sys, numpy as np, joblib
sys.path.insert(0, ".")
import training.build_dataset as bd
from poker44_bump.model_v7 import V7Model, _combined_feats
from poker44_bump.ngram_llr import NgramLLRScorer
from sklearn.metrics import average_precision_score
import lightgbm as lgb

LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
           colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0,
           random_state=0, n_jobs=4, verbose=-1)

paths = bd.resolve_benchmark_paths("data")
exs = bd.load_benchmark_examples(paths)
chunks = [e["chunk"] for e in exs]
labels = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]
uniq = sorted(set(dates))
print(f"[v7] {len(exs)} chunks / {len(uniq)} dates", flush=True)

bdicts = [_combined_feats(c) for c in chunks]
bnames = sorted({k for d in bdicts for k in d})
Xb = np.asarray([[float(d.get(n, 0.0)) for n in bdicts[0].keys()] for d in bdicts]) if False else \
     np.asarray([[float(d.get(n, 0.0)) for n in bnames] for d in bdicts])
lnames = NgramLLRScorer.feature_names()

# ---- honest leave-one-date-out cross-date AP (live proxy) ----
def lodo():
    yt, ys = [], []
    for d in uniq:
        te = [i for i, x in enumerate(dates) if x == d]
        tr = [i for i, x in enumerate(dates) if x != d]
        if not te or len(set(labels[tr])) < 2:
            continue
        sc = NgramLLRScorer().fit([chunks[i] for i in tr], labels[tr])
        Ltr = np.asarray([[sc.transform_one(chunks[i])[n] for n in lnames] for i in tr])
        Lte = np.asarray([[sc.transform_one(chunks[i])[n] for n in lnames] for i in te])
        Xtr = np.hstack([Xb[tr], Ltr]); Xte = np.hstack([Xb[te], Lte])
        m = lgb.LGBMClassifier(**LGB).fit(Xtr, labels[tr])
        ys += list(m.predict_proba(Xte)[:, 1]); yt += list(labels[te])
    return average_precision_score(yt, ys)

print(f"[v7] LODO cross-date AP (live proxy) = {lodo():.4f}", flush=True)

# ---- holdout on latest 2 dates ----
hold = set(uniq[-2:])
tr = [i for i, d in enumerate(dates) if d not in hold]
te = [i for i, d in enumerate(dates) if d in hold]
sc = NgramLLRScorer().fit([chunks[i] for i in tr], labels[tr])
Ltr = np.asarray([[sc.transform_one(chunks[i])[n] for n in lnames] for i in tr])
Lte = np.asarray([[sc.transform_one(chunks[i])[n] for n in lnames] for i in te])
m = lgb.LGBMClassifier(**LGB).fit(np.hstack([Xb[tr], Ltr]), labels[tr])
hap = average_precision_score(labels[te], m.predict_proba(np.hstack([Xb[te], Lte]))[:, 1])
print(f"[v7] holdout AP (latest 2 dates) = {hap:.4f}", flush=True)

# ---- final fit on ALL data + build deployable artifact ----
scorer = NgramLLRScorer().fit(chunks, labels)
Lall = np.asarray([[scorer.transform_one(c)[n] for n in lnames] for c in chunks])
clf = lgb.LGBMClassifier(**LGB).fit(np.hstack([Xb, Lall]), labels)
meta = {"lodo_cross_date_ap": float(hap), "holdout_ap": float(hap),
        "n_base_feats": len(bnames), "n_llr_feats": len(lnames),
        "model_version": "v7-daterobust-nllr"}
model = V7Model(clf, bnames, scorer, topk_cfg={"positive_fraction": 0.15}, metadata=meta)
joblib.dump(model, "models/bump_model_v7.joblib")
print(f"[v7] saved models/bump_model_v7.joblib  feats={len(model.feature_names)} "
      f"(base {len(bnames)} + llr {len(lnames)})", flush=True)
