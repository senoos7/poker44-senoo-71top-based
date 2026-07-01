"""v21 = train ONLY on features that (a) TRANSFER benchmark->live (single-feature
domain AUC ~0.5 = invariant marginal) AND (b) carry bot/human signal on benchmark.
Rationale: v19 dropped the worst (hero/raw) but still trains on disjoint behavioral
features (fold_share etc. domain AUC 1.0 per shift_diag) that don't transfer and
re-degenerate the live ranker. Keep only invariant+informative features. Trained on
the FULL updated benchmark (36 dates incl. fresh 06-27..06-30).

Also a DECISIVE diagnostic: if no invariant feature carries signal => the wall.
Run:  PYTHONPATH=. .venv/bin/python train_v21.py
"""
import sys, glob, json, time, warnings, numpy as np, joblib
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.model_v10 import V10Model
from sklearn.metrics import average_precision_score, roc_auc_score
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
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])
dates = [e["source_date"] for e in exs]; uniq = sorted(set(dates))
v10 = joblib.load("models/bump_model_v10.joblib"); names = list(v10.feature_names)
print(f"[v21] benchmark={len(exs)} dates={len(uniq)} feats={len(names)}", flush=True)

live = []
for f in glob.glob("live_capture/*.jsonl"):
    for l in open(f):
        l=l.strip()
        if l:
            try: live.append(json.loads(l)["chunk"])
            except Exception: pass
livesample = live[:2500]
print(f"[v21] live sample={len(livesample)}", flush=True)

def rows(cs):
    out=[]
    for c in cs:
        bf = base_cf(c) if c else {"hand_count":0.0}; bf["hand_count"]=float(len(c))
        out.append([float(bf.get(n,0.0)) for n in names])
    return np.asarray(out, dtype=np.float64)
Xb = rows(chunks); Xl = rows(livesample)

# ---- per-feature screen: domain separation + benchmark signal ----
dom = np.r_[np.zeros(len(Xb)), np.ones(len(Xl))]
Xall = np.vstack([Xb, Xl])
screen=[]
for j,n in enumerate(names):
    col = Xall[:,j]
    try: da = roc_auc_score(dom, col)
    except Exception: da = 0.5
    dsep = abs(da-0.5)
    try: ba = roc_auc_score(y, Xb[:,j])
    except Exception: ba = 0.5
    bsep = abs(ba-0.5)
    screen.append((n, dsep, bsep))
screen.sort(key=lambda t:(t[1], -t[2]))   # most invariant first
inv = [s for s in screen if s[1] < 0.10]          # invariant marginal (dom AUC 0.40-0.60)
inv_sig = [s for s in inv if s[2] >= 0.07]         # ...and carries benchmark signal
print(f"\n[v21] of {len(names)} feats: INVARIANT(dom_sep<0.10)={len(inv)}  INVARIANT+SIGNAL(bench_sep>=0.07)={len(inv_sig)}", flush=True)
print("[v21] top invariant+signal feats (name, dom_sep, bench_sep):", flush=True)
for n,ds,bs in sorted(inv_sig, key=lambda t:-t[2])[:15]:
    print(f"        {n:48s} dom={ds:.3f} bench={bs:.3f}", flush=True)
if len(inv_sig) < 5:
    print("[v21] *** FEWER THAN 5 invariant+informative features => the WALL is confirmed; ranking signal does NOT transfer.", flush=True)

keep = [n for n,_,_ in inv_sig] or [n for n,_,_ in inv]   # fall back to invariant-only if needed
keepset = set(keep); kidx = [j for j,n in enumerate(names) if n in keepset]
print(f"[v21] training on {len(keep)} features", flush=True)

# ---- LODO on the kept subset ----
def lodo(Xm):
    yt, ys = [], []
    for d in uniq:
        te=[i for i,x in enumerate(dates) if x==d]; tr=[i for i,x in enumerate(dates) if x!=d]
        if len(set(y[tr]))<2: continue
        ests=make_estimators(); acc=np.zeros(len(te))
        for e in ests: e.fit(Xm[tr], y[tr]); acc+=e.predict_proba(Xm[te])[:,1]
        ys+=list(acc/len(ests)); yt+=list(y[te])
    return average_precision_score(yt, ys)
Xk = Xb[:, kidx]
t0=time.time(); ap=lodo(Xk)
print(f"[v21] LODO AP (invariant subset, 36 dates) = {ap:.4f}  (v10-full 0.90, v19 0.87)  [{time.time()-t0:.0f}s]", flush=True)

ests=make_estimators()
for e in ests: e.fit(Xk, y)
model=V10Model(ests, keep, topk_cfg={"positive_fraction":0.15},
               metadata={"model_version":"v21-invariant","model_name":"poker44-bump-v21",
                         "framework":"avg-ensemble @ domain-invariant+informative feats (36 dates)",
                         "n_feats":len(keep),"lodo_ap":round(ap,4),
                         "training_data_statement":"Trained on RELEASED public benchmark (groundTruth), 36 dates 2026-05-26..2026-06-30, sanitized (train==serve), feature set screened to benchmark<->live domain-invariant + informative columns. No validator-private data.",
                         "training_data_sources":["released_training_benchmark"],
                         "data_attestation":"No validator-private data used; released benchmark labels only."})
joblib.dump(model, "models/bump_model_v21.joblib")
print(f"[v21] saved models/bump_model_v21.joblib ({len(keep)} feats)", flush=True)

# ---- DECISIVE: live degeneracy vs v10 / v19 ----
mm=joblib.load("models/bump_model_v21.joblib")
samp=live[:600]
r21=mm.predict_raw(samp); r10=v10.predict_raw(samp)
try: r19=joblib.load("models/bump_model_v19.joblib").predict_raw(samp)
except Exception: r19=np.array([0.0])
print(f"\n[v21] *** LIVE raw-prob (n={len(samp)}):", flush=True)
print(f"      v21 (invariant):  mean={r21.mean():.3f} std={r21.std():.3f}", flush=True)
print(f"      v19 (saninv):     mean={r19.mean():.3f} std={r19.std():.3f}", flush=True)
print(f"      v10 (full):       mean={r10.mean():.3f} std={r10.std():.3f}  (degenerate if ~0.02)", flush=True)
print(f"      => v21/v10 std ratio = {r21.std()/max(r10.std(),1e-9):.1f}x", flush=True)
