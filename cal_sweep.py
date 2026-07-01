"""Cache walk-forward raw scores (ext features) once, then sweep conformal
(calib-window, buffer) in-memory to find 0-cliff max-reward calibration."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from poker44_bump.features_ext import chunk_features_ext
from poker44_bump.payload_view import prepare_hand_for_miner
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATA = HERE / "data"
G = []
for fp in sorted(glob.glob(str(DATA / "training_benchmark_*.txt"))):
    d = Path(fp).stem.split("_")[-1]; data = json.load(open(fp))["data"]
    for rec in data["chunks"]:
        for hands, label in zip(rec["chunks"], rec["groundTruth"]):
            vis = [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]
            if vis: G.append((d, int(label), chunk_features_ext(vis)))
cols = sorted(G[0][2].keys()); dates = sorted(set(d for d, _, _ in G))
X = np.array([[float(f.get(c, 0.0)) for c in cols] for _, _, f in G])
y = np.array([l for _, l, _ in G]); dt = np.array([d for d, _, _ in G])

def ens(s=42):
    return [LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,min_child_samples=8,verbose=-1,random_state=s),
            XGBClassifier(n_estimators=400,learning_rate=0.03,max_depth=4,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",verbosity=0,random_state=s),
            ExtraTreesClassifier(n_estimators=500,max_features="sqrt",min_samples_leaf=2,n_jobs=-1,random_state=s),
            RandomForestClassifier(n_estimators=500,max_features="sqrt",min_samples_leaf=2,n_jobs=-1,random_state=s)]
def avg(M, Xt): return np.mean([m.predict_proba(Xt)[:,1] for m in M], axis=0)

START, WMAX = 12, 10
cache = []  # (raw_test, y_test, raw_trail(list per trailing date newest-last), y_trail...)
for i in range(START, len(dates)):
    d = dates[i]; trail = dates[max(0,i-WMAX):i]; tr = dt < dates[max(0,i-WMAX)]
    M = ens()
    for m in M: m.fit(X[tr], y[tr])
    rt = avg(M, X[dt==d]); yt = y[dt==d]
    trail_raw = {td: (avg(M, X[dt==td]), y[dt==td]) for td in trail}
    cache.append((d, rt, yt, trail, trail_raw))
print(f"cached {len(cache)} test dates (ext feats={len(cols)})")

def q(a,p): return float(np.quantile(a,p)) if len(a) else 1.0
def rew(s,l,T):
    p=(s>=T).astype(int); yt=l.astype(int)
    tp=((p==1)&(yt==1)).sum();fp=((p==1)&(yt==0)).sum();tn=((p==0)&(yt==0)).sum();fn=((p==0)&(yt==1)).sum()
    fpr=fp/max(tn+fp,1);rec=tp/max(tp+fn,1)
    a=average_precision_score(yt,s) if (yt.max()==1 and yt.min()==0) else 0.0
    saf=0.0 if fpr>=0.10 else (1-fpr)**2
    return (0.65*a+0.35*rec)*saf,fpr,rec,a

print(f"\n{'W':>3}{'buf':>5}{'mean_rew':>10}{'cliff%':>8}{'mean_rec':>10}{'mean_AP':>9}")
best=None
for W in (5,8,10):
    for buf in (0.5,0.8,1.0,1.3):
        R=[];C=0;Rc=[];AP=[]
        for d,rt,yt,trail,traw in cache:
            use=trail[-W:]
            hc=np.concatenate([traw[td][0][traw[td][1]==0] for td in use]) if use else np.array([0.5])
            T=max(hc)+buf*(max(hc)-q(hc,0.9))
            r,fpr,rc,ap=rew(rt,yt,T); R.append(r);Rc.append(rc);AP.append(ap);C+=int(fpr>=0.10)
        mr=np.mean(R);cl=100*C/len(R)
        tag=""
        if cl==0 and (best is None or mr>best[2]): best=(W,buf,mr); tag="  <=="
        print(f"{W:>3}{buf:>5}{mr:>10.3f}{cl:>8.1f}{np.mean(Rc):>10.3f}{np.mean(AP):>9.4f}{tag}")
print(f"\nBEST 0-cliff: W={best[0]} buf={best[1]} reward={best[2]:.3f}  (vs king uid32 reward 0.536, #1 composite needs reward ~0.68)")
