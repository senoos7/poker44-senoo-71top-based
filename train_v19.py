"""Build v19 = v10 architecture retrained on the SANITIZATION-INVARIANT feature
subset (drops hero/button-relative cols + raw *_bb magnitude cols that go OOD /
collapse on the sanitized live feed), per guten-tag d0's features_v2 diagnosis:
hero-keyed + raw-amount features collapse to near-constant on live (their words:
live AP 0.42 vs 0.82 benchmark). We KEEP the hero-free / scale-free signal:
bucket-signature collisions, action/street/actor entropies, action shares,
switch/run-share structure, counts, pot_monotonic_rate, order-stats (size-robust).

Native benchmark chunk size (NOT resampled — v18 proved resampling kills the
collision signal). Train==serve already in place (build_dataset sanitizes).

DECISIVE: does v19 raw-prob STD on captured live rise far above v10's 0.020
(un-degenerate, like d0's non-degenerate live ranker)?

Run:  PYTHONPATH=. .venv/bin/python train_v19.py
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

def keep(name: str) -> bool:
    # drop sanitizer-fragile features: hero/button-relative + raw bb magnitudes
    if "hero" in name or "button" in name:
        return False
    if "_bb" in name:          # amount_*_bb, pot_*_bb, starting_stack_*_bb (raw magnitude)
        return False
    return True

# ---- data: native-size SANITIZED benchmark (same loader as v10) ----
exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]; uniq = sorted(set(dates))
v10 = joblib.load("models/bump_model_v10.joblib")
full_names = list(v10.feature_names)
names = [n for n in full_names if keep(n)]
dropped = [n for n in full_names if not keep(n)]
print(f"[v19] {len(exs)} chunks / {len(uniq)} dates | feats: {len(full_names)} -> KEEP {len(names)}  (dropped {len(dropped)})", flush=True)
print(f"[v19] dropped examples: {[n for n in dropped if 'hero' in n or 'button' in n][:4]} ... {[n for n in dropped if '_bb' in n][:4]}", flush=True)

fd = [base_cf(c) for c in chunks]
def to_X(fd, cols): return np.asarray([[float(d.get(n, 0.0)) for n in cols] for d in fd])
X = to_X(fd, names); Xfull = to_X(fd, full_names)

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
ap_full = lodo(Xfull); ap_sub = lodo(X)
print(f"[v19] LODO cross-date AP:  FULL-293 = {ap_full:.4f}   SANITIZATION-INVARIANT-{len(names)} = {ap_sub:.4f}   [{time.time()-t0:.0f}s]", flush=True)

# ---- final fit on all data (subset) ----
ests = make_estimators()
for e in ests: e.fit(X, y)
model = V10Model(ests, names, topk_cfg={"positive_fraction": 0.15},
                 metadata={"model_version": "v19-saninvariant", "model_name": "poker44-bump-v19",
                           "framework": "avg-ensemble(lgbm,et,rf)+topk @sanitization-invariant-feats",
                           "n_estimators": len(ests), "n_feats": len(names),
                           "dropped": "hero/button + raw *_bb", "lodo_ap": round(ap_sub, 4)})
joblib.dump(model, "models/bump_model_v19.joblib")
print(f"[v19] saved models/bump_model_v19.joblib ({len(names)} feats)", flush=True)

# ================= DECISIVE TEST: degeneracy on LIVE =================
live = []
for f in glob.glob("live_capture/*.jsonl"):
    for l in open(f):
        l = l.strip()
        if l:
            try: live.append(json.loads(l)["chunk"])
            except Exception: pass
sample = live[:600]
mm = joblib.load("models/bump_model_v19.joblib")
r19 = mm.predict_raw(sample); r10 = v10.predict_raw(sample)
print(f"\n[v19] *** LIVE raw-prob (n={len(sample)}):", flush=True)
print(f"      v19 (sanitization-invariant): mean={r19.mean():.3f} std={r19.std():.3f}", flush=True)
print(f"      v10 (full 293, hero+raw):     mean={r10.mean():.3f} std={r10.std():.3f}   (degenerate if std~0.02)", flush=True)
print(f"      => v19/v10 std ratio = {r19.std()/max(r10.std(),1e-9):.1f}x   (>>1 = un-degenerated)", flush=True)

# domain AUC: does dropping OOD features close the benchmark<->live gap?
def domain_auc(A, B):
    Xd = np.vstack([A, B]); yd = np.r_[np.zeros(len(A)), np.ones(len(B))]
    au = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, yd):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31,
                               reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1)
        m.fit(Xd[tr], yd[tr]); au.append(roc_auc_score(yd[te], m.predict_proba(Xd[te])[:, 1]))
    return float(np.mean(au))
Xl = to_X([base_cf(c) for c in sample], names)
print(f"\n[v19] domain AUC  v19-feats benchmark vs live = {domain_auc(X[:600], Xl):.4f}  (v10 full was 1.000; lower=gap closed)", flush=True)
