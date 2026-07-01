"""Build v17 = v10 ensemble + unsupervised per-feature quantile alignment (live->benchmark).
Verifies the alignment mechanically collapses the benchmark<->live domain gap.
Run:  PYTHONPATH=. .venv/bin/python train_v17.py
"""
import sys, json, glob, time, warnings, numpy as np, joblib
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import training.build_dataset as bd
from poker44_bump.features import chunk_features as base_cf
from poker44_bump.model_v17 import V17AlignedModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

NQ = 512
v10 = joblib.load("models/bump_model_v10.joblib")
names = list(v10.feature_names); ests = v10.estimators; wts = v10.weights
print(f"[v17] base=v10 ({len(ests)} estimators, {len(names)} feats)", flush=True)

def rows(chunks):
    out = []
    for c in chunks:
        c = list(c or []); bf = base_cf(c) if c else {"hand_count": 0.0}; bf["hand_count"] = float(len(c))
        out.append([float(bf.get(n, 0.0)) for n in names])
    return np.asarray(out, dtype=np.float64)

exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
bench = [e["chunk"] for e in exs]
live = []
for f in glob.glob("live_capture/*.jsonl"):
    for l in open(f):
        l = l.strip()
        if l:
            try: live.append(json.loads(l)["chunk"])
            except Exception: pass
print(f"[v17] benchmark={len(bench)} live={len(live)}", flush=True)
Xb = rows(bench); Xl = rows(live)

# per-feature quantile grids
grid = np.linspace(0, 1, NQ)
bench_sorted = [np.quantile(Xb[:, j], grid) for j in range(Xb.shape[1])]
live_sorted = [np.quantile(Xl[:, j], grid) for j in range(Xl.shape[1])]

model = V17AlignedModel(ests, names, wts, live_sorted, bench_sorted,
                        topk_cfg={"positive_fraction": 0.15},
                        metadata={"model_version": "v17-quantile-aligned", "base": "v10",
                                  "n_live_calib": len(live), "n_quantiles": NQ})
model.metadata["model_version"] = "v17-quantile-aligned"

# mechanical check: domain AUC raw vs aligned-live (lower=better aligned)
def domain_auc(A, B):
    X = np.vstack([A, B]); y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    au = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, reg_lambda=1.0,
                               random_state=0, n_jobs=4, verbose=-1)
        m.fit(X[tr], y[tr]); au.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(au))
Xl_al = model._align(Xl)
print(f"[v17] domain AUC  raw={domain_auc(Xb, Xl):.4f}  ALIGNED={domain_auc(Xb, Xl_al):.4f}  (lower=better; ~0.5=marginals matched)", flush=True)

joblib.dump(model, "models/bump_model_v17.joblib")
mm = joblib.load("models/bump_model_v17.joblib")
t0 = time.time(); sc = mm.predict_chunk_scores(live[:200]); dt = (time.time() - t0) / 200 * 1000
pos = sum(1 for s in sc if s > 0.5)
print(f"[v17] saved bump_model_v17.joblib | {dt:.2f} ms/chunk (on live 100-hand chunks) | pos {pos}/200 | range [{min(sc):.3f},{max(sc):.3f}]", flush=True)
# show how alignment changes raw scores on live (sanity: should differ from unaligned v10)
raw_al = mm.predict_raw(live[:200]); raw_un = v10.predict_raw(live[:200])
print(f"[v17] live raw-prob: aligned mean={raw_al.mean():.3f} std={raw_al.std():.3f} | v10-unaligned mean={raw_un.mean():.3f} std={raw_un.std():.3f}", flush=True)
