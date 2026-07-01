"""Head-to-head walk-forward: baseline features vs extended (n-gram) features.
Train ensemble on dates < calib-window, fit conformal T on trailing held-out
window, test next date. Report mean AP / reward / cliff-rate / recall for each
feature set, plus the conformal-head deployed model's behavior."""
from __future__ import annotations
import glob, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from poker44_bump.features_ext import chunk_features_ext, _extra_feats
from poker44_bump.features import chunk_features
from poker44_bump.payload_view import prepare_hand_for_miner
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

DATA = HERE / "data"
groups = []  # (date, label, full_featdict)
for fp in sorted(glob.glob(str(DATA / "training_benchmark_*.txt"))):
    d = Path(fp).stem.split("_")[-1]
    data = json.load(open(fp))["data"]
    for rec in data["chunks"]:
        for hands, label in zip(rec["chunks"], rec["groundTruth"]):
            vis = [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]
            if vis:
                groups.append((d, int(label), chunk_features_ext(vis)))
dates = sorted(set(d for d, _, _ in groups))
base_cols = sorted(k for k in groups[0][2] if not k.startswith("cx_"))
ext_cols = sorted(groups[0][2].keys())
print(f"groups={len(groups)} dates={len(dates)} base_feats={len(base_cols)} ext_feats={len(ext_cols)}")

def mat(cols):
    X = np.array([[float(f.get(c, 0.0)) for c in cols] for _, _, f in groups])
    y = np.array([l for _, l, _ in groups]); dt = np.array([d for d, _, _ in groups])
    return X, y, dt

def ens(seed=42):
    return [LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,subsample=0.8,
              colsample_bytree=0.8,min_child_samples=8,verbose=-1,random_state=seed),
            XGBClassifier(n_estimators=400,learning_rate=0.03,max_depth=4,subsample=0.8,
              colsample_bytree=0.8,eval_metric="logloss",verbosity=0,random_state=seed),
            ExtraTreesClassifier(n_estimators=500,max_features="sqrt",min_samples_leaf=2,n_jobs=-1,random_state=seed),
            RandomForestClassifier(n_estimators=500,max_features="sqrt",min_samples_leaf=2,n_jobs=-1,random_state=seed)]
def avg(M,X): return np.mean([m.predict_proba(X)[:,1] for m in M],axis=0)
def q(a,p): return float(np.quantile(a,p)) if len(a) else 1.0

def reward_at(s,l,T):
    p=(s>=T).astype(int); yt=l.astype(int)
    tp=((p==1)&(yt==1)).sum();fp=((p==1)&(yt==0)).sum();tn=((p==0)&(yt==0)).sum();fn=((p==0)&(yt==1)).sum()
    fpr=fp/max(tn+fp,1);rec=tp/max(tp+fn,1)
    a=average_precision_score(yt,s) if(yt.max()==1 and yt.min()==0) else 0.0
    saf=0.0 if fpr>=0.10 else (1-fpr)**2
    return (0.65*a+0.35*rec)*saf,fpr,rec,a

def walkforward(cols, START=12, W=5, buf=0.5):
    X,y,dt = mat(cols)
    R=[];F=[];Rc=[];AP=[];cliff=0
    for i in range(START,len(dates)):
        d=dates[i]; cal=dates[i-W:i]; tr=dt<dates[i-W]; cm=np.isin(dt,cal)
        M=ens()
        for m in M: m.fit(X[tr],y[tr])
        rt=avg(M,X[dt==d]); yt=y[dt==d]
        hc=avg(M,X[cm])[y[cm]==0]
        T=max(hc)+buf*(max(hc)-q(hc,0.9))
        r,fpr,rec,ap=reward_at(rt,yt,T)
        R.append(r);F.append(fpr);Rc.append(rec);AP.append(ap);cliff+=int(fpr>=0.10)
    return dict(reward=np.mean(R),cliff=100*cliff/len(R),fpr=np.mean(F),recall=np.mean(Rc),ap=np.mean(AP),n=len(R))

print(f"\n{'feature set':14}{'mean_AP':>9}{'mean_rew':>9}{'cliff%':>8}{'mean_fpr':>9}{'mean_rec':>9}")
for name,cols in [("baseline",base_cols),("extended",ext_cols)]:
    r=walkforward(cols)
    print(f"{name:14}{r['ap']:9.4f}{r['reward']:9.3f}{r['cliff']:8.1f}{r['fpr']:9.3f}{r['recall']:9.3f}")
print(f"\n(walk-forward over {walkforward(base_cols)['n']} unseen test dates; king uid32: AP 0.824 reward 0.536 latency 17.9s)")
