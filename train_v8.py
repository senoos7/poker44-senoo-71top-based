"""Build deployable v8 = v6 stacked pipeline + static FPR-targeting threshold head.
Threshold t is set on benchmark-human stacked scores at a CONSERVATIVE in-sample
FPR target (in-sample AP is optimistic, so target tight; live-tune via env
POKER44_THRESHOLD). 0.30 hard cap backstop. The HEAD's cross-date superiority over
fixed-fraction is already validated (scratchpad/head_final.py, leakage-free OOF).

Run:  PYTHONPATH=. .venv/bin/python train_v8.py
"""
import sys, numpy as np, joblib
sys.path.insert(0, ".")
import training.build_dataset as bd
from poker44_bump.model_v8 import V8Model, _threshold_squeeze
from poker44_bump.model_v6 import _combined_feats
from sklearn.metrics import average_precision_score

IN_SAMPLE_FPR_TARGET = 0.01      # very tight: in-sample humans over-separate; live lands higher.
CAP = 0.22                        # backstop fraction (start safe; live-tune t up recall via env)

art = joblib.load("models/v6_stacked.joblib")
exs = bd.load_benchmark_examples(bd.resolve_benchmark_paths("data"))
chunks = [e["chunk"] for e in exs]; y = np.array([int(e["label"]) for e in exs])

stacked = (art.get("models") or [None])[0]
fnames = list(art.get("feature_names") or [])
fdicts = [_combined_feats(c) for c in chunks]          # compute features ONCE per chunk
rows = np.asarray([[float(d.get(n, 0.0)) for n in fnames] for d in fdicts])
scores = np.asarray(stacked.predict_chunk_scores(chunks, rows, apply_calibration=True))
print(f"v6 stacked in-sample AP={average_precision_score(y,scores):.4f}  "
      f"score range [{scores.min():.4f},{scores.max():.4f}]")

# threshold at the (1-target) quantile of HUMAN scores
t = float(np.quantile(scores[y == 0], 1.0 - IN_SAMPLE_FPR_TARGET))
print(f"threshold t={t:.5f}  (in-sample FPR target {IN_SAMPLE_FPR_TARGET})")

# in-sample sanity at this t with the head
shaped = np.asarray(_threshold_squeeze(scores, t, CAP))
pos = shaped > 0.5
tp=int((pos&(y==1)).sum()); fn=int((~pos&(y==1)).sum())
fp=int((pos&(y==0)).sum()); tn=int((~pos&(y==0)).sum())
print(f"in-sample @t: recall={tp/max(1,tp+fn):.3f} fpr={fp/max(1,fp+tn):.4f} "
      f"marked={pos.mean():.3f} bands [{shaped.min():.3f},{shaped.max():.3f}]")

model = V8Model(art, threshold=t, cap=CAP,
                metadata={"in_sample_fpr_target": IN_SAMPLE_FPR_TARGET,
                          "head": "static-fpr-threshold-on-raw"})
joblib.dump(model, "models/bump_model_v8.joblib")
print(f"saved models/bump_model_v8.joblib  ver={model.metadata['model_version']} "
      f"feats={len(model.feature_names)} t={model.threshold:.5f} cap={model.cap}")
