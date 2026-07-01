"""Build v15 = Deep-Sets hand-encoder + relation head (torch) with hand-subsample
augmentation. Honest LODO-style per-date AP via 8-fold grouped-by-date OOF.
Run:  PYTHONPATH=. .venv/bin/python train_v15.py
"""
import sys, time, warnings, numpy as np, joblib
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
import training.build_dataset as bd
from poker44_bump.model_v15 import DeepSetsRelation, V15Model, featurize_chunk, MAX_ACTIONS
from sklearn.metrics import average_precision_score

torch.manual_seed(0); np.random.seed(0)
exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks=[e["chunk"] for e in exs]; y=np.array([int(e["label"]) for e in exs],dtype=np.float32)
dates=[e["source_date"] for e in exs]; uniq=sorted(set(dates))
print(f"[v15] {len(exs)} chunks / {len(uniq)} dates | torch {torch.__version__}", flush=True)

# featurize once -> global padded arrays
feats=[featurize_chunk(c) for c in chunks]
Hmax=max(f[0].shape[0] for f in feats)
N=len(feats)
ACTS=np.zeros((N,Hmax,MAX_ACTIONS,4),np.int64); AMT=np.zeros((N,Hmax,MAX_ACTIONS),np.float32)
AMASK=np.zeros((N,Hmax,MAX_ACTIONS),np.float32); HMASK=np.zeros((N,Hmax),np.float32)
for i,(a,m,am,hm) in enumerate(feats):
    h=a.shape[0]; ACTS[i,:h]=a; AMT[i,:h]=m; AMASK[i,:h]=am; HMASK[i,:h]=hm
ACTS_t=torch.from_numpy(ACTS); AMT_t=torch.from_numpy(AMT); AMASK_t=torch.from_numpy(AMASK)
HMASK_t=torch.from_numpy(HMASK); Y_t=torch.from_numpy(y)

CFG=dict(d_act=24,d_hand=32,d_head=64,p=0.3)

def train_net(tr_idx, epochs=35, aug=True, seed=0):
    torch.manual_seed(seed)
    net=DeepSetsRelation(**CFG); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss()
    idx=np.array(tr_idx); bs=32
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for s in range(0,len(idx),bs):
            b=idx[s:s+bs]
            hm=HMASK_t[b].clone()
            if aug:  # hand-subsample augmentation: randomly drop valid hands (keep >=20)
                keep=(torch.rand_like(hm)>0.25).float()
                hm2=hm*keep
                # ensure at least min(20, n) hands remain per chunk
                ok=hm2.sum(1)>=torch.clamp(hm.sum(1),max=20.0)
                hm=torch.where(ok.unsqueeze(1).bool(),hm2,hm)
            opt.zero_grad()
            logit=net(ACTS_t[b],AMT_t[b],AMASK_t[b],hm)
            loss=lossf(logit,Y_t[b]); loss.backward(); opt.step()
    net.eval(); return net

def predict(net, idx):
    with torch.no_grad():
        out=[]
        for s in range(0,len(idx),128):
            b=np.array(idx[s:s+128])
            out.append(torch.sigmoid(net(ACTS_t[b],AMT_t[b],AMASK_t[b],HMASK_t[b])).numpy())
    return np.concatenate(out)

# 8-fold grouped-by-date OOF (leakage-free), per-date AP
groups=[uniq[i::8] for i in range(8)]
oof=np.full(N,np.nan)
t0=time.time()
for gi,grp in enumerate(groups):
    te=[i for i,d in enumerate(dates) if d in grp]; tr=[i for i,d in enumerate(dates) if d not in grp]
    if not te or len(set(y[tr]))<2: continue
    net=train_net(tr,epochs=35,aug=True,seed=gi)
    oof[te]=predict(net,te)
    print(f"  [v15] OOF fold {gi+1}/8 done ({time.time()-t0:.0f}s)", flush=True)
aps=[average_precision_score(y[[i for i,x in enumerate(dates) if x==d]],
     oof[[i for i,x in enumerate(dates) if x==d]]) for d in uniq
     if len(set(y[[i for i,x in enumerate(dates) if x==d]]))>1 and not np.isnan(oof[[i for i,x in enumerate(dates) if x==d]]).any()]
print(f"[v15] OOF per-date mean AP = {np.mean(aps):.4f}  (pooled AP {average_precision_score(y[~np.isnan(oof)],oof[~np.isnan(oof)]):.4f})", flush=True)

# final fit on all data (more epochs, seed-averaged 2 nets folded into one via best)
net=train_net(list(range(N)),epochs=60,aug=True,seed=7)
model=V15Model(net.state_dict(),CFG,topk_cfg={"positive_fraction":0.15},
               metadata={"model_version":"v15-deepsets-relation","oof_perdate_ap":round(float(np.mean(aps)),4),
                         "params":sum(p.numel() for p in net.parameters())})
model.metadata["model_version"]="v15-deepsets-relation"
joblib.dump(model,"models/bump_model_v15.joblib")

mm=joblib.load("models/bump_model_v15.joblib")
t0=time.time(); sc=mm.predict_chunk_scores(chunks[:200]); dt=(time.time()-t0)/200*1000
pos=sum(1 for s in sc if s>0.5)
print(f"[v15] saved bump_model_v15.joblib | {sum(p.numel() for p in net.parameters())} params | {dt:.2f} ms/chunk | pos {pos}/200 | range [{min(sc):.3f},{max(sc):.3f}]", flush=True)
