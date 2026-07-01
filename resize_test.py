"""Does the model (trained on 30-40 hand chunks) hold at LIVE chunk size (60-120)?
Pool same-label benchmark hands into ~80-hand chunks, score with the deployed
model, compare AP/reward vs the native 30-hand chunks."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from poker44_bump.payload_view import prepare_hand_for_miner
from poker44_bump.model import BumpModel
import joblib
from sklearn.metrics import average_precision_score

model=joblib.load(HERE/"models"/"bump_model.joblib"); T=model.threshold
DATA=HERE/"data"

def load_by_date():
    by={}
    for fp in sorted(glob.glob(str(DATA/"training_benchmark_*.txt"))):
        d=Path(fp).stem.split("_")[-1]; data=json.load(open(fp))["data"]
        groups=[]
        for rec in data["chunks"]:
            for hands,lab in zip(rec["chunks"],rec["groundTruth"]):
                vis=[prepare_hand_for_miner(h) for h in hands if isinstance(h,dict)]
                if vis: groups.append((int(lab),vis))
        by[d]=groups
    return by

def reward(s,l,T):
    s=np.array(s);l=np.array(l);p=(s>=T).astype(int)
    tp=((p==1)&(l==1)).sum();fp=((p==1)&(l==0)).sum();tn=((p==0)&(l==0)).sum();fn=((p==0)&(l==1)).sum()
    fpr=fp/max(tn+fp,1);rec=tp/max(tp+fn,1)
    ap=average_precision_score(l,s) if (l.max()==1 and l.min()==0) else 0.0
    saf=0.0 if fpr>=0.10 else (1-fpr)**2
    return (0.65*ap+0.35*rec)*saf,ap,fpr,rec

by=load_by_date()
# --- native 30-hand ---
nat_chunks=[]; nat_lab=[]
for d,gs in by.items():
    for lab,hs in gs: nat_chunks.append(hs); nat_lab.append(lab)
ns=model.predict_chunk_scores(nat_chunks)
r0=reward(ns,nat_lab,T)
print(f"NATIVE 30-40 hand chunks: n={len(nat_chunks)} avg_hands={np.mean([len(c) for c in nat_chunks]):.0f}")
print(f"  reward={r0[0]:.3f} AP={r0[1]:.3f} FPR={r0[2]:.3f} recall={r0[3]:.3f}\n")

# --- live-size: pool same-label hands within a date into ~TARGET-hand chunks ---
import random
for TARGET in (80, 120):
    rng=random.Random(0); big=[]; biglab=[]
    for d,gs in by.items():
        for lab in (0,1):
            pool=[h for l,hs in gs if l==lab for h in hs]
            rng.shuffle(pool)
            for i in range(0,len(pool)-TARGET+1,TARGET):
                big.append(pool[i:i+TARGET]); biglab.append(lab)
    if not big: continue
    bs=model.predict_chunk_scores(big)
    r=reward(bs,biglab,T)
    print(f"LIVE-SIZE ~{TARGET} hand chunks: n={len(big)} avg_hands={np.mean([len(c) for c in big]):.0f}")
    print(f"  reward={r[0]:.3f} AP={r[1]:.3f} FPR={r[2]:.3f} recall={r[3]:.3f}  (T={T:.3f} unchanged)")
