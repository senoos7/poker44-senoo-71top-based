"""Build v20 = v19 sanitization-invariant subset (drop hero/button + raw *_bb)
PLUS bucket-snapped amount features (d0's fuller fix: keep amount signal in a
bounded, sanitization-robust bucket-index form). Retrain v10 ensemble on native
sanitized benchmark. Compare LODO AP + LIVE degeneracy vs v19 and v10.

Run:  PYTHONPATH=. .venv/bin/python train_v20.py
"""
import sys, glob, json, time, warnings, numpy as np, joblib
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.features_bucket import bucket_amount_feats, feature_names as bkt_names
from poker44_bump.model_v20 import V20Model
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

def keep(name): return not ("hero" in name or "button" in name or "_bb" in name)

exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]; uniq = sorted(set(dates))
v10 = joblib.load("models/bump_model_v10.joblib")
base_keep = [n for n in v10.feature_names if keep(n)]
names = base_keep + bkt_names()        # v19 subset + 42 bucket feats
print(f"[v20] feats: v19-subset {len(base_keep)} + bucket {len(bkt_names())} = {len(names)}", flush=True)

def row(c):
    bf = base_cf(c) if c else {"hand_count": 0.0}; bf["hand_count"] = float(len(c))
    bf.update(bucket_amount_feats(c))
    return [float(bf.get(n, 0.0)) for n in names]
X = np.asarray([row(c) for c in chunks], dtype=np.float64)
print(f"[v20] X={X.shape}", flush=True)

def lodo(Xm):
    yt, ys = [], []
    for d in uniq:
        te = [i for i, x in enumerate(dates) if x == d]; tr = [i for i, x in enumerate(dates) if x != d]
        if len(set(y[tr])) < 2: continue
        ests = make_estimators(); acc = np.zeros(len(te))
        for e in ests:
            e.fit(Xm[tr], y[tr]); acc += e.predict_proba(Xm[te])[:, 1]
        ys += list(acc / len(ests)); yt += list(y[te])
    return average_precision_score(yt, ys)
t0 = time.time()
ap = lodo(X)
print(f"[v20] LODO cross-date AP = {ap:.4f}   (v10 full 0.8988, v19 invariant 0.8703)   [{time.time()-t0:.0f}s]", flush=True)

ests = make_estimators()
for e in ests: e.fit(X, y)
model = V20Model(ests, names, topk_cfg={"positive_fraction": 0.15},
                 metadata={"model_version": "v20-saninv-bucket", "model_name": "poker44-bump-v20",
                           "framework": "avg-ensemble(lgbm,et,rf)+topk @sanitization-invariant+bucket-amounts",
                           "n_estimators": len(ests), "n_feats": len(names), "lodo_ap": round(ap, 4)})
joblib.dump(model, "models/bump_model_v20.joblib")
print(f"[v20] saved models/bump_model_v20.joblib ({len(names)} feats)", flush=True)

# DECISIVE: degeneracy on LIVE
live = []
for f in glob.glob("live_capture/*.jsonl"):
    for l in open(f):
        l = l.strip()
        if l:
            try: live.append(json.loads(l)["chunk"])
            except Exception: pass
sample = live[:600]
mm = joblib.load("models/bump_model_v20.joblib")
r20 = mm.predict_raw(sample); r10 = v10.predict_raw(sample)
v19 = joblib.load("models/bump_model_v19.joblib"); r19 = v19.predict_raw(sample)
print(f"\n[v20] *** LIVE raw-prob (n={len(sample)}):", flush=True)
print(f"      v20 (saninv+bucket): mean={r20.mean():.3f} std={r20.std():.3f}", flush=True)
print(f"      v19 (saninv only):   mean={r19.mean():.3f} std={r19.std():.3f}", flush=True)
print(f"      v10 (full hero+raw): mean={r10.mean():.3f} std={r10.std():.3f}   (degenerate if std~0.02)", flush=True)
print(f"      => v20/v10 std ratio = {r20.std()/max(r10.std(),1e-9):.1f}x", flush=True)

def domain_auc(A, B):
    Xd = np.vstack([A, B]); yd = np.r_[np.zeros(len(A)), np.ones(len(B))]
    au = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, yd):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                               reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1)
        m.fit(Xd[tr], yd[tr]); au.append(roc_auc_score(yd[te], m.predict_proba(Xd[te])[:, 1]))
    return float(np.mean(au))
Xl = np.asarray([row(c) for c in sample], dtype=np.float64)
print(f"\n[v20] domain AUC benchmark vs live = {domain_auc(X[:600], Xl):.4f}  (still 1.0 expected; lower=gap closed)", flush=True)
