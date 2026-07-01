"""Local Poker44 validator harness.

Drives the deployed bump miner through the subnet's REAL scoring code
(poker44.score.scoring.reward) with the production reward_window=40, over
UNSEEN benchmark windows (model trained only on earlier dates). Mirrors the
validator's accumulate-then-score-last-40 path. Reports per-window reward and
the epoch-mean (composite is the mean of window scores)."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TRAVIS = HERE.parent / "repos" / "Travis861_Poker44_v1"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(TRAVIS))
from poker44.score.scoring import reward as REAL_REWARD          # authoritative validator scorer
from poker44_bump.payload_view import prepare_hand_for_miner
from poker44_bump.model import BumpModel
from train_bump import load_examples, make_ensemble, avg_proba
import numpy as np
from sklearn.model_selection import StratifiedKFold

REWARD_WINDOW = 40   # POKER44_REWARD_WINDOW default in production validator
DATA = HERE / "data"

# ---- 1. train miner on earlier dates only (honest holdout) ----
rows, y, dates = load_examples(str(DATA), "ext")
feat_names = sorted({k for r in rows for k in r})
X = np.array([[float(r.get(n, 0.0)) for n in feat_names] for r in rows])
uniq = sorted(set(dates.tolist()))
HOLD = 9
train_dates = set(uniq[:-HOLD]); test_dates = uniq[-HOLD:]
trm = np.array([d in train_dates for d in dates])
print(f"train dates={len(train_dates)} ({uniq[0]}..{uniq[-HOLD-1]}) | UNSEEN test dates={len(test_dates)} ({test_dates[0]}..{test_dates[-1]})")

# OOF on train for conformal T
oof = np.zeros(int(trm.sum()))
Xt, yt, dtt = X[trm], y[trm], dates[trm]
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for tr, va in skf.split(Xt, yt):
    M = make_ensemble(42); [m.fit(Xt[tr], yt[tr]) for m in M]; oof[va] = avg_proba(M, Xt[va])
recent = set(sorted(train_dates)[-10:])
hr = oof[(yt == 0) & np.isin(dtt, list(recent))]
final = make_ensemble(42); [m.fit(Xt, yt) for m in final]
def make_T(buf): return float(min(max(max(hr) + buf*(max(hr)-np.quantile(hr,0.9)), 0.05), 0.999))
import os
BUF = float(os.getenv("LV_BUF", "0.5"))
T = make_T(BUF)
model = BumpModel(final, feat_names, T, metadata={"feature_set": "ext"})
print(f"buffer={BUF} conformal T={T:.4f}\n")

# ---- 2. build unseen chunks, run miner ----
chunks, labels = [], []
for fp in sorted(glob.glob(str(DATA / "training_benchmark_*.txt"))):
    d = Path(fp).stem.split("_")[-1]
    if d not in test_dates: continue
    data = json.load(open(fp))["data"]
    for rec in data["chunks"]:
        for hands, lab in zip(rec["chunks"], rec["groundTruth"]):
            vis = [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]
            if vis: chunks.append(vis); labels.append(int(lab))
scores = np.array(model.predict_chunk_scores(chunks)); labels = np.array(labels)
print(f"miner scored {len(chunks)} unseen chunks (range [{scores.min():.3f},{scores.max():.3f}])")

# ---- 3. REAL validator scoring over rolling 40-chunk windows ----
print(f"\nreward_window={REWARD_WINDOW} (production). Per-window REAL reward:")
n_win = len(chunks) // REWARD_WINDOW
print(f"{'window':>7}{'reward':>9}{'ap':>8}{'recall':>8}{'fpr':>7}{'safety':>8}")
ws = []
for w in range(n_win):
    sl = slice(w*REWARD_WINDOW, (w+1)*REWARD_WINDOW)
    r, info = REAL_REWARD(scores[sl], labels[sl])
    ws.append(r)
    print(f"{w+1:>7}{r:>9.3f}{info['ap_score']:>8.3f}{info['bot_recall']:>8.3f}{info['fpr']:>7.3f}{info['human_safety_penalty']:>8.3f}")
# whole-set (single 40+ window the validator would hold)
r_all, info_all = REAL_REWARD(scores[:n_win*REWARD_WINDOW] if n_win else scores, labels[:n_win*REWARD_WINDOW] if n_win else labels)
print(f"\nEPOCH-MEAN reward over {n_win} windows = {np.mean(ws):.3f}  (composite = mean of window scores)")
print(f"pooled reward             = {r_all:.3f}  (ap={info_all['ap_score']:.3f} recall={info_all['bot_recall']:.3f} fpr={info_all['fpr']:.3f})")
print(f"\n--- benchmark vs targets ---")
print(f"  king uid32 live reward 0.536 / composite 0.583 ; provisional #1 uid136 reward 0.693 / composite 0.599")
print(f"  win threshold: reward ~0.68 -> composite ~0.60 (via composite~=0.70*reward+0.13)")
print(f"  our epoch-mean reward {np.mean(ws):.3f} -> projected composite ~ {0.696*np.mean(ws)+0.127:.3f}")
