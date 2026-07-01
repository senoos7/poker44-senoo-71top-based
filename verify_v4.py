"""Load the saved v4 artifact and verify the topk head end-to-end through the
REAL validator scorer, per 40-chunk window. (In-sample on recent dates — a
mechanical sanity check; the honest unseen generalization result is from ab_head.py.)
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
TRAVIS = HERE.parent / "repos" / "Travis861_Poker44_v1"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(TRAVIS))
from poker44.score.scoring import reward as REAL_REWARD
from poker44_bump.payload_view import prepare_hand_for_miner

PATH = HERE / "models" / ("bump_model_v4.joblib" if "--v4" in sys.argv or True else "bump_model.joblib")
m = joblib.load(PATH)
md = m.metadata
print(f"artifact: {PATH.name}")
print(f"  model_version={md.get('model_version')} head_mode={m.head_mode} subsample={m.subsample}")
print(f"  topk_cfg={m.topk_cfg}")
print(f"  feats={len(m.feature_names)} train_chunk_size={m.train_chunk_size} oof_ap={md.get('oof_ap')}\n")

DATA = HERE / "data"
REWARD_WINDOW = 40
# most-recent ~6 dates (in-sample)
files = sorted(glob.glob(str(DATA / "training_benchmark_*.txt")))[-6:]
chunks, labels = [], []
for fp in files:
    data = json.load(open(fp))["data"]
    for rec in data["chunks"]:
        for hands, lab in zip(rec["chunks"], rec["groundTruth"]):
            vis = [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]
            if vis: chunks.append(vis); labels.append(int(lab))
labels = np.array(labels)
n_win = len(chunks)//REWARD_WINDOW
print(f"scoring {len(chunks)} recent chunks -> {n_win} windows of {REWARD_WINDOW}\n")

ws, aps, recs, fprs, safs = [], [], [], [], []
band_pos, band_neg = [], []
for w in range(n_win):
    sl = slice(w*REWARD_WINDOW, (w+1)*REWARD_WINDOW)
    sc = np.array(m.predict_chunk_scores(chunks[sl]))
    r, info = REAL_REWARD(sc, labels[sl])
    ws.append(r); aps.append(info['ap_score']); recs.append(info['bot_recall'])
    fprs.append(info['fpr']); safs.append(info['human_safety_penalty'])
    band_pos += [v for v in sc if v >= 0.5]; band_neg += [v for v in sc if v < 0.5]
mr = float(np.mean(ws))
print(f"per-window REAL reward: mean={mr:.3f} ap={np.mean(aps):.3f} recall={np.mean(recs):.3f} "
      f"fpr={np.mean(fprs):.3f} safety={np.mean(safs):.3f} cliffs={sum(f>=0.10 for f in fprs)}/{n_win}")
print(f"projected composite ~ {0.696*mr+0.127:.3f}")
if band_pos:
    print(f"positive band: [{min(band_pos):.3f},{max(band_pos):.3f}] (expect ~[0.501,0.509])  n={len(band_pos)}")
if band_neg:
    print(f"negative band: [{min(band_neg):.3f},{max(band_neg):.3f}] (expect [0,0.49])  n={len(band_neg)}")
print("NOTE: in-sample optimistic; unseen result is ab_head.py (~0.60 composite).")
