"""Build v16 = decorrelated blend of (base+rp) tree-ensemble + Deep-Sets-Relation net.
8-fold grouped-by-date OOF to pick the blend weight honestly, then fit final on all data.
Run:  PYTHONPATH=. .venv/bin/python train_v16.py
"""
import sys, time, warnings, numpy as np, joblib
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import torch, torch.nn as nn
import training.build_dataset as bd
from poker44_bump.features import chunk_features as base_cf
from poker44_bump.features_repeat import _repeat_feats
from poker44_bump.model_v15 import DeepSetsRelation, featurize_chunk, MAX_ACTIONS
from poker44_bump.model_v16 import V16Model
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb
torch.manual_seed(0); np.random.seed(0)

exs=bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks=[e["chunk"] for e in exs]; y=np.array([int(e["label"]) for e in exs],np.float32)
dates=[e["source_date"] for e in exs]; uniq=sorted(set(dates))
fd=[{**base_cf(c),**_repeat_feats(c)} for c in chunks]; names=sorted({k for d in fd for k in d})
X=np.asarray([[float(d.get(n,0.0)) for n in names] for d in fd])
feats=[featurize_chunk(c) for c in chunks]; Hmax=max(f[0].shape[0] for f in feats); N=len(feats)
ACTS=np.zeros((N,Hmax,MAX_ACTIONS,4),np.int64); AMT=np.zeros((N,Hmax,MAX_ACTIONS),np.float32)
AMASK=np.zeros((N,Hmax,MAX_ACTIONS),np.float32); HMASK=np.zeros((N,Hmax),np.float32)
for i,(a,m,am,hm) in enumerate(feats): h=a.shape[0]; ACTS[i,:h]=a; AMT[i,:h]=m; AMASK[i,:h]=am; HMASK[i,:h]=hm
ACTS_t,AMT_t,AMASK_t,HMASK_t,Y_t=map(torch.from_numpy,(ACTS,AMT,AMASK,HMASK,y))
CFG=dict(d_act=24,d_hand=32,d_head=64,p=0.3)
print(f"[v16] {N} chunks / {len(uniq)} dates / {len(names)} feats ({sum(n.startswith('rp_') for n in names)} rp_)",flush=True)

def trees(): return [
    lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,random_state=0,n_jobs=4,verbose=-1),
    lgb.LGBMClassifier(n_estimators=300,learning_rate=0.02,num_leaves=15,max_depth=4,min_child_samples=40,subsample=0.7,colsample_bytree=0.6,reg_lambda=5.0,reg_alpha=2.0,random_state=1,n_jobs=4,verbose=-1),
    lgb.LGBMClassifier(n_estimators=500,learning_rate=0.02,num_leaves=63,min_child_samples=30,subsample=0.7,colsample_bytree=0.7,reg_lambda=2.0,random_state=2,n_jobs=4,verbose=-1),
    ExtraTreesClassifier(n_estimators=400,min_samples_leaf=5,max_features=0.5,random_state=3,n_jobs=4),
    RandomForestClassifier(n_estimators=400,min_samples_leaf=5,max_features=0.5,random_state=4,n_jobs=4)]

def train_ds(tr,seed,epochs=35):
    torch.manual_seed(seed); net=DeepSetsRelation(**CFG)
    opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-4); lf=nn.BCEWithLogitsLoss(); idx=np.array(tr)
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for s in range(0,len(idx),32):
            b=idx[s:s+32]; hm=HMASK_t[b].clone()
            keep=(torch.rand_like(hm)>0.25).float(); hm2=hm*keep
            ok=hm2.sum(1)>=torch.clamp(hm.sum(1),max=20.0); hm=torch.where(ok.unsqueeze(1).bool(),hm2,hm)
            opt.zero_grad(); loss=lf(net(ACTS_t[b],AMT_t[b],AMASK_t[b],hm),Y_t[b]); loss.backward(); opt.step()
    net.eval(); return net

groups=[uniq[i::8] for i in range(8)]; oof_t=np.full(N,np.nan); oof_d=np.full(N,np.nan); t0=time.time()
for gi,grp in enumerate(groups):
    te=[i for i,d in enumerate(dates) if d in grp]; tr=[i for i,d in enumerate(dates) if d not in grp]
    if not te or len(set(y[tr]))<2: continue
    acc=np.zeros(len(te))
    for e in trees(): e.fit(X[tr],y[tr]); acc+=e.predict_proba(X[te])[:,1]
    oof_t[te]=acc/5.0
    net=train_ds(tr,gi)
    with torch.no_grad(): oof_d[te]=torch.sigmoid(net(ACTS_t[te],AMT_t[te],AMASK_t[te],HMASK_t[te])).numpy()
    print(f"  [v16] fold {gi+1}/8 ({time.time()-t0:.0f}s)",flush=True)

def perdate(s):
    a=[average_precision_score(y[[i for i,x in enumerate(dates) if x==d]],s[[i for i,x in enumerate(dates) if x==d]])
       for d in uniq if len(set(y[[i for i,x in enumerate(dates) if x==d]]))>1]
    return float(np.mean(a))
m=~np.isnan(oof_t)
best_w,best_ap=0.0,0.0
for w in np.arange(0,0.76,0.05):
    ap=perdate(np.where(m,(1-w)*oof_t+w*oof_d,0))
    if ap>best_ap: best_ap,best_w=ap,float(w)
print(f"[v16] corr={np.corrcoef(oof_t[m],oof_d[m])[0,1]:.3f} | trees={perdate(np.where(m,oof_t,0)):.4f} ds={perdate(np.where(m,oof_d,0)):.4f} | BEST blend w={best_w:.2f} AP={best_ap:.4f}",flush=True)

# final fit on all data
ests=trees()
for e in ests: e.fit(X,y)
net=train_ds(list(range(N)),seed=7,epochs=60)
model=V16Model(ests,names,net.state_dict(),CFG,best_w,topk_cfg={"positive_fraction":0.15},
               metadata={"model_version":"v16-trees-deepsets-blend","blend_w":best_w,
                         "lodo_perdate_ap":round(best_ap,4),"n_feats":len(names),
                         "n_rp":sum(n.startswith('rp_') for n in names),"corr_tree_ds":round(float(np.corrcoef(oof_t[m],oof_d[m])[0,1]),3)})
model.metadata["model_version"]="v16-trees-deepsets-blend"
joblib.dump(model,"models/bump_model_v16.joblib")
mm=joblib.load("models/bump_model_v16.joblib")
t0=time.time(); sc=mm.predict_chunk_scores(chunks[:200]); dt=(time.time()-t0)/200*1000
pos=sum(1 for s in sc if s>0.5)
print(f"[v16] saved bump_model_v16.joblib | blend_w={best_w:.2f} | {dt:.2f} ms/chunk | pos {pos}/200 | range [{min(sc):.3f},{max(sc):.3f}]",flush=True)
