"""Train v22 (action-order sequence transformer) on 36 benchmark dates. 5-fold
grouped-by-date OOF AP + final fit. DECISIVE: live un-degeneration std vs v10/v19/v15.
Run:  PYTHONPATH=. .venv/bin/python train_v22.py
"""
import sys, glob, json, time, warnings, numpy as np, joblib
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
import torch, torch.nn as nn
import training.build_dataset as bd
from poker44_bump.model_v22 import SeqTransformer, V22Model
from poker44_bump.model_v15 import featurize_chunk, MAX_ACTIONS
from sklearn.metrics import average_precision_score

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(12)
exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks=[e["chunk"] for e in exs]; y=np.array([int(e["label"]) for e in exs],dtype=np.float32)
dates=[e["source_date"] for e in exs]; uniq=sorted(set(dates))
print(f"[v22] {len(exs)} chunks / {len(uniq)} dates | torch {torch.__version__}", flush=True)

feats=[featurize_chunk(c) for c in chunks]
Hmax=max(f[0].shape[0] for f in feats); N=len(feats)
ACTS=np.zeros((N,Hmax,MAX_ACTIONS,4),np.int64); AMT=np.zeros((N,Hmax,MAX_ACTIONS),np.float32)
AMASK=np.zeros((N,Hmax,MAX_ACTIONS),np.float32); HMASK=np.zeros((N,Hmax),np.float32)
for i,(a,m,am,hm) in enumerate(feats):
    h=a.shape[0]; ACTS[i,:h]=a; AMT[i,:h]=m; AMASK[i,:h]=am; HMASK[i,:h]=hm
ACTS_t=torch.from_numpy(ACTS); AMT_t=torch.from_numpy(AMT); AMASK_t=torch.from_numpy(AMASK)
HMASK_t=torch.from_numpy(HMASK); Y_t=torch.from_numpy(y)
CFG=dict(d=32,nhead=4,nlayers=1,d_head=64,p=0.3)

def train_net(tr_idx, epochs, seed=0, aug=True):
    torch.manual_seed(seed)
    net=SeqTransformer(**CFG); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(); idx=np.array(tr_idx); bs=48
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for s in range(0,len(idx),bs):
            b=idx[s:s+bs]; hm=HMASK_t[b].clone()
            if aug:
                keep=(torch.rand_like(hm)>0.25).float(); hm2=hm*keep
                ok=hm2.sum(1)>=torch.clamp(hm.sum(1),max=20.0)
                hm=torch.where(ok.unsqueeze(1).bool(),hm2,hm)
            opt.zero_grad()
            loss=lossf(net(ACTS_t[b],AMT_t[b],AMASK_t[b],hm),Y_t[b]); loss.backward(); opt.step()
    net.eval(); return net

def predict(net, idx):
    out=[]
    with torch.no_grad():
        for s in range(0,len(idx),128):
            b=np.array(idx[s:s+128])
            out.append(torch.sigmoid(net(ACTS_t[b],AMT_t[b],AMASK_t[b],HMASK_t[b])).numpy())
    return np.concatenate(out)

groups=[uniq[i::5] for i in range(5)]; oof=np.full(N,np.nan); t0=time.time()
for gi,grp in enumerate(groups):
    te=[i for i,d in enumerate(dates) if d in grp]; tr=[i for i,d in enumerate(dates) if d not in grp]
    if not te or len(set(y[tr]))<2: continue
    net=train_net(tr,epochs=22,seed=gi); oof[te]=predict(net,te)
    print(f"  [v22] OOF fold {gi+1}/5 ({time.time()-t0:.0f}s)", flush=True)
m=~np.isnan(oof)
print(f"[v22] OOF pooled AP = {average_precision_score(y[m],oof[m]):.4f}  (v10-trees 0.90, v15-deepsets 0.78)", flush=True)

net=train_net(list(range(N)),epochs=40,seed=7)
model=V22Model(net.state_dict(),CFG,topk_cfg={"positive_fraction":0.15},
               metadata={"model_version":"v22-seq-transformer","oof_ap":round(float(average_precision_score(y[m],oof[m])),4),
                         "params":sum(p.numel() for p in net.parameters()),"n_dates":len(uniq),
                         "training_data_statement":"Trained on RELEASED benchmark (groundTruth), 36 dates 2026-05-26..06-30, sanitized (train==serve). Per-hand action-order transformer + set-pool + repetition relation. No validator-private data.",
                         "training_data_sources":["released_training_benchmark"],
                         "data_attestation":"No validator-private data used; released benchmark labels only."})
joblib.dump(model,"models/bump_model_v22.joblib")
print(f"[v22] saved models/bump_model_v22.joblib | params={sum(p.numel() for p in net.parameters())}", flush=True)

# DECISIVE: live un-degeneration
live=[]
for f in glob.glob("live_capture/*.jsonl"):
    for l in open(f):
        l=l.strip()
        if l:
            try: live.append(json.loads(l)["chunk"])
            except: pass
samp=live[:600]; mm=joblib.load("models/bump_model_v22.joblib")
r22=mm.predict_raw(samp)
print(f"\n[v22] *** LIVE raw-prob (n={len(samp)}): mean={r22.mean():.3f} std={r22.std():.3f} range[{r22.min():.3f},{r22.max():.3f}]", flush=True)
for v,p in [("v19-saninv","models/bump_model_v19.joblib"),("v15-deepsets","models/bump_model_v15.joblib"),("v10-full","models/bump_model_v10.joblib")]:
    try:
        r=joblib.load(p).predict_raw(samp); print(f"      vs {v:14s} std={r.std():.3f} mean={r.mean():.3f}", flush=True)
    except Exception as e: print(f"      {v} err {e}", flush=True)
print(f"[v22] verdict: std>>0.066 (v19) => seq model adds transferable signal worth A/B; ~=0.05 => no breakthrough (wall holds)", flush=True)
