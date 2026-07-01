"""Build v18 = v10 architecture (avg ensemble lgbm+et+rf + topk) RETRAINED on
benchmark hands RESAMPLED into ~live-sized (80-100 hand) chunks, instead of the
native 30-40 hand benchmark chunks. Goal: remove the size-extrapolation component
of the live degeneracy (v10 raw-prob std on live = 0.020 ~ near-constant) by
letting the model see size-sensitive features (unique_share, signature shares)
at the FIXED live chunk size (subnet commit 0b6d742 pinned live to ~100 hands).

Decisive test (printed): does v18 raw-prob STD on captured live chunks rise far
above v10's 0.020 (un-degenerate)? + domain AUC v18-train vs live (size effect?).

Run:  PYTHONPATH=. .venv/bin/python train_v18.py
"""
import sys, glob, json, time, warnings, numpy as np, joblib
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.model_v10 import V10Model
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb

RNG = np.random.RandomState(0)
N_PER_GROUP = 8   # synthetic 100-hand chunks per (date,label) pool

def make_estimators():  # identical to v10
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

# ---- live chunk-size distribution (to match when resampling benchmark) ----
live = []
for f in glob.glob("live_capture/*.jsonl"):
    for l in open(f):
        l = l.strip()
        if l:
            try: live.append(json.loads(l)["chunk"])
            except Exception: pass
live_sizes = np.array([len(c) for c in live])
print(f"[v18] live chunks={len(live)}  size: min={live_sizes.min()} max={live_sizes.max()} mean={live_sizes.mean():.1f}", flush=True)

# ---- benchmark: pool hands by (date,label), resample into live-sized chunks ----
exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
v10 = joblib.load("models/bump_model_v10.joblib")
names = list(v10.feature_names)

pools = {}  # (date,label) -> list of hands
for e in exs:
    pools.setdefault((e["source_date"], int(e["label"])), []).extend(e["chunk"])

syn_chunks, syn_y, syn_date = [], [], []
for (date, lab), hands in pools.items():
    hands = list(hands)
    for k in range(N_PER_GROUP):
        s = int(RNG.choice(live_sizes))                 # match live size
        s = min(s, len(hands))
        idx = RNG.choice(len(hands), size=s, replace=False)
        syn_chunks.append([hands[i] for i in idx])
        syn_y.append(lab); syn_date.append(date)
syn_y = np.array(syn_y)
print(f"[v18] synthetic chunks={len(syn_chunks)}  bot={int(syn_y.sum())} human={int((1-syn_y).sum())}  "
      f"size: mean={np.mean([len(c) for c in syn_chunks]):.1f}", flush=True)

def rows(chunks):
    out = []
    for c in chunks:
        bf = base_cf(c) if c else {"hand_count": 0.0}; bf["hand_count"] = float(len(c))
        out.append([float(bf.get(n, 0.0)) for n in names])
    return np.asarray(out, dtype=np.float64)

X = rows(syn_chunks); y = syn_y; dates = syn_date; uniq = sorted(set(dates))
print(f"[v18] X={X.shape} dates={len(uniq)}", flush=True)

# ---- LODO cross-date AP (same protocol as v10) ----
def lodo():
    yt, ys = [], []
    for d in uniq:
        te = [i for i, x in enumerate(dates) if x == d]; tr = [i for i, x in enumerate(dates) if x != d]
        if len(set(y[tr])) < 2: continue
        ests = make_estimators(); acc = np.zeros(len(te))
        for e in ests:
            e.fit(X[tr], y[tr]); acc += e.predict_proba(X[te])[:, 1]
        ys += list(acc / len(ests)); yt += list(y[te])
    return average_precision_score(yt, ys)
t0 = time.time()
ap = lodo()
print(f"[v18] LODO cross-date AP (100-hand chunks) = {ap:.4f}   (v10 native-size LODO was 0.9091)   [{time.time()-t0:.0f}s]", flush=True)

# ---- final fit on all synthetic 100-hand chunks ----
ests = make_estimators()
for e in ests: e.fit(X, y)
model = V10Model(ests, names, topk_cfg={"positive_fraction": 0.15},
                 metadata={"model_version": "v18-size100", "model_name": "poker44-bump-v18",
                           "framework": "avg-ensemble(lgbm,et,rf)+topk @100-hand-chunks",
                           "n_estimators": len(ests), "n_feats": len(names),
                           "trained_chunk_size": "live-matched 80-100", "lodo_ap": round(ap, 4)})
joblib.dump(model, "models/bump_model_v18.joblib")
print(f"[v18] saved models/bump_model_v18.joblib", flush=True)

# ================= DECISIVE TEST: degeneracy on LIVE =================
mm = joblib.load("models/bump_model_v18.joblib")
sample = live[:600]
r18 = mm.predict_raw(sample); r10 = v10.predict_raw(sample)
print(f"\n[v18] *** LIVE raw-prob (n={len(sample)}):", flush=True)
print(f"      v18 (100-hand-trained): mean={r18.mean():.3f} std={r18.std():.3f}", flush=True)
print(f"      v10 (native-trained):   mean={r10.mean():.3f} std={r10.std():.3f}   (degenerate if std~0.02)", flush=True)
print(f"      => v18/v10 std ratio = {r18.std()/max(r10.std(),1e-9):.1f}x", flush=True)

# domain AUC: does training-feature distribution at live size move closer to live?
def domain_auc(A, B):
    Xd = np.vstack([A, B]); yd = np.r_[np.zeros(len(A)), np.ones(len(B))]
    au = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, yd):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                               reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1)
        m.fit(Xd[tr], yd[tr]); au.append(roc_auc_score(yd[te], m.predict_proba(Xd[te])[:, 1]))
    return float(np.mean(au))
Xl = rows(sample)
print(f"\n[v18] domain AUC  v18-train(100h) vs live = {domain_auc(X[:600], Xl):.4f}  "
      f"(v10 native-vs-live was 1.000; lower=size gap closed)", flush=True)
