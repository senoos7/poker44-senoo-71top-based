"""End-to-end verification of the deployable bump artifact.

(A) Deployment readiness: load saved model, score chunks shaped like a
    DetectionSynapse, check score length/range/latency + manifest.
(B) Honest generalization: train a holdout artifact on all-but-last-K dates
    (threshold fit on the recent-of-train window), then score the K unseen dates
    through the FULL inference path (validator-sanitize -> predict_chunk_scores).
"""
from __future__ import annotations
import glob, json, os, sys, time, tempfile, subprocess
from pathlib import Path
import numpy as np, joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = str(HERE)
from poker44_bump.payload_view import prepare_hand_for_miner
from poker44_bump.model import BumpModel
from sklearn.metrics import average_precision_score

DATA = HERE / "data"
files = sorted(glob.glob(str(DATA / "training_benchmark_*.txt")))
dates = [Path(f).stem.split("_")[-1] for f in files]

def load_date_chunks(fp):
    """Return [(validator_sanitized_chunk, label), ...] for one date file."""
    data = json.load(open(fp)).get("data", {})
    out = []
    for rec in data.get("chunks", []):
        for hands, label in zip(rec.get("chunks") or [], rec.get("groundTruth") or []):
            if not isinstance(hands, list):
                continue
            # validator would send payload_view(raw); benchmark hands -> same transform
            sanitized = [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]
            if sanitized:
                out.append((sanitized, int(label)))
    return out

def reward(scores, labels):
    s = np.asarray(scores); yt = np.asarray(labels)
    preds = (s >= 0.5).astype(int)
    tp=((preds==1)&(yt==1)).sum();fp=((preds==1)&(yt==0)).sum()
    tn=((preds==0)&(yt==0)).sum();fn=((preds==0)&(yt==1)).sum()
    fpr=fp/max(tn+fp,1);rec=tp/max(tp+fn,1)
    ap=average_precision_score(yt,s) if (yt.max()==1 and yt.min()==0) else 0.0
    saf=0.0 if fpr>=0.10 else (1-fpr)**2
    return dict(reward=(0.65*ap+0.35*rec)*saf, ap=ap, fpr=fpr, recall=rec)

# ---------- (A) deployment readiness on saved full model ----------
print("="*70, "\n(A) DEPLOYMENT READINESS (saved models/bump_model.joblib)")
model = joblib.load(HERE / "models" / "bump_model.joblib")
print(f"  loaded {type(model).__name__} | T={model.threshold:.4f} | feats={len(model.feature_names)}")
print(f"  manifest: {model.metadata.get('model_name')} v{model.metadata.get('model_version')} oof_ap={model.metadata.get('oof_ap')}")
sample = load_date_chunks(files[-1])
chunks = [c for c, _ in sample]
t0 = time.perf_counter()
scores = model.predict_chunk_scores(chunks)
dt = (time.perf_counter() - t0) * 1000
assert len(scores) == len(chunks), "score length mismatch!"
assert all(0.0 <= s <= 1.0 for s in scores), "score out of [0,1]!"
print(f"  scored {len(chunks)} chunks: len_ok={len(scores)==len(chunks)} range=[{min(scores):.3f},{max(scores):.3f}] "
      f"latency={dt:.1f}ms total ({dt/len(chunks):.2f}ms/chunk)")

# ---------- (B) honest generalization: holdout last K dates ----------
K = 3
print("="*70, f"\n(B) HONEST GENERALIZATION (train on {len(dates)-K} dates, test last {K} unseen)")
holdout_dates = dates[-K:]
with tempfile.TemporaryDirectory() as td:
    for f, d in zip(files, dates):
        if d not in holdout_dates:
            os.symlink(os.path.abspath(f), os.path.join(td, os.path.basename(f)))
    out = os.path.join(td, "holdout_model.joblib")
    r = subprocess.run([sys.executable, str(HERE/"train_bump.py"), "--data-dir", td,
                        "--output", out, "--calib-window", "6", "--buffer", "0.5"],
                       capture_output=True, text=True, env={**os.environ})
    for line in r.stdout.splitlines():
        if any(k in line for k in ("examples=", "OOF AP", "conformal T", "per-date")):
            print("   [train] " + line)
    hmodel = joblib.load(out)

all_scores, all_labels, per_date = [], [], []
t0 = time.perf_counter(); nchunks = 0
for d in holdout_dates:
    fp = str(DATA / f"training_benchmark_{d}.txt")
    items = load_date_chunks(fp)
    cks = [c for c, _ in items]; labs = [l for _, l in items]
    sc = hmodel.predict_chunk_scores(cks); nchunks += len(cks)
    m = reward(sc, labs)
    per_date.append((d, m))
    all_scores += sc; all_labels += labs
lat = (time.perf_counter()-t0)*1000
print(f"  per-date (unseen):")
for d, m in per_date:
    print(f"    {d}: reward={m['reward']:.3f} AP={m['ap']:.3f} FPR={m['fpr']:.3f} recall={m['recall']:.3f}")
agg = reward(all_scores, all_labels)
cliffs = sum(1 for _, m in per_date if m['fpr'] >= 0.10)
print(f"  POOLED unseen: reward={agg['reward']:.3f} AP={agg['ap']:.3f} FPR={agg['fpr']:.3f} "
      f"recall={agg['recall']:.3f} | cliff_dates={cliffs}/{K} | latency={lat/max(nchunks,1):.2f}ms/chunk")
print(f"\n  vs live leader: rankingQuality(AP) 0.887, reward 0.693, latency 23.7s/40chunks(latencyQuality=0)")
