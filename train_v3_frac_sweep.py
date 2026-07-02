"""LODO sweep: feat_v3 + v10-4 with topk_frac in [0.11, 0.13, 0.15].

Trains each LODO fold once, caches raw scores, then applies all three
topk_frac values — no redundant training.
"""
import sys, logging, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.path.append("/root/Skip/poker/poker_demo_2")

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost  as xgb

import training.build_dataset as bd
from poker44_ml.features_v3 import chunk_features_v3
from poker44_bump.model_v5   import _topk_squeeze

# ── logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = Path(f"logs/train_v3_frac_{ts}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
log.info(f"Log: {log_path}")

FRACS = [0.11, 0.13, 0.15]

# ── data ──────────────────────────────────────────────────────────────────────
exs = bd.load_benchmark_examples(
    sorted(Path("data/benchmark_out").glob(
        "benchmark_2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    ))
)
dates = sorted(set(e["source_date"] for e in exs))
log.info(f"Loaded {len(exs)} examples / {len(dates)} dates")

log.info("Extracting v3 features …")
feats_v3 = [chunk_features_v3(e["chunk"]) for e in exs]
keys = sorted(feats_v3[0].keys())
X = np.array([[f[k] for k in keys] for f in feats_v3], dtype=np.float32)
labels   = np.array([e["label"]       for e in exs], dtype=np.int32)
date_arr = np.array([e["source_date"] for e in exs])
log.info(f"Matrix: {X.shape}")

# ── ensemble ──────────────────────────────────────────────────────────────────
def make_v10_4():
    return VotingClassifier(voting="soft", estimators=[
        ("lgb1", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=31,
                                     min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=2.0, reg_alpha=0.5, random_state=0, n_jobs=4, verbose=-1)),
        ("lgb2", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.015, num_leaves=15, max_depth=4,
                                     min_child_samples=45, subsample=0.7, colsample_bytree=0.6,
                                     reg_lambda=6.0, reg_alpha=2.0, random_state=1, n_jobs=4, verbose=-1)),
        ("lgb3", lgb.LGBMClassifier(n_estimators=700, learning_rate=0.015, num_leaves=47,
                                     min_child_samples=35, subsample=0.75, colsample_bytree=0.65,
                                     reg_lambda=3.0, reg_alpha=1.0, random_state=2, n_jobs=4, verbose=-1)),
        ("xgb1", xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                     random_state=10, n_jobs=4, eval_metric="logloss",
                                     verbosity=0, use_label_encoder=False)),
        ("xgb2", xgb.XGBClassifier(n_estimators=350, learning_rate=0.02, max_depth=3,
                                     subsample=0.75, colsample_bytree=0.65, reg_alpha=1.5,
                                     reg_lambda=4.0, random_state=11, n_jobs=4,
                                     eval_metric="logloss", verbosity=0, use_label_encoder=False)),
        ("et",  ExtraTreesClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.45,
                                     random_state=3, n_jobs=4)),
        ("rf",  RandomForestClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.45,
                                       random_state=4, n_jobs=4)),
    ])

# ── LODO: train once per fold, cache raw scores ───────────────────────────────
log.info("\nTraining LODO folds (feat_v3 + v10-4) …")
# results[frac][date] = dict(ap, recall, fpr, reward, n, k)
results = {f: {} for f in FRACS}

t0 = time.time()
for i, heldout in enumerate(dates):
    tr_mask = date_arr != heldout
    te_mask = date_arr == heldout
    X_tr, y_tr = X[tr_mask], labels[tr_mask]
    X_te, y_te = X[te_mask], labels[te_mask]
    if y_te.sum() == 0 or (1 - y_te).sum() == 0:
        continue

    clf = make_v10_4()
    clf.fit(X_tr, y_tr)
    raw = clf.predict_proba(X_te)[:, 1]

    for frac in FRACS:
        squeezed = np.array(_topk_squeeze(raw, frac=frac))
        ap = average_precision_score(y_te, squeezed)
        n  = len(y_te)
        k  = max(1, int(n * frac))
        thr = np.sort(squeezed)[-k]
        preds = (squeezed >= thr).astype(int)
        tp = int(((preds == 1) & (y_te == 1)).sum())
        fp = int(((preds == 1) & (y_te == 0)).sum())
        n_pos = int(y_te.sum()); n_neg = int((1 - y_te).sum())
        recall = tp / max(n_pos, 1)
        fpr    = fp / max(n_neg, 1)
        reward = ap * (0.0 if fpr > 0.10 else 1.0)
        results[frac][heldout] = dict(
            ap=ap, recall=recall, fpr=fpr, reward=reward, n=n, k=k, tp=tp, fp=fp
        )

    # per-date progress line (frac=0.15 for reference)
    r = results[0.15][heldout]
    cliff = " ***CLIFF***" if r["fpr"] > 0.10 else ""
    log.info(
        f"  [{i+1:2d}/{len(dates)}] {heldout}  n={r['n']:2d} k={r['k']}"
        f"  AP={r['ap']:.4f}  Recall={r['recall']:.3f}  FPR={r['fpr']:.3f}"
        f"  Rwd={r['reward']:.4f}{cliff}"
    )

elapsed = (time.time() - t0) / 60

# ── summary table ─────────────────────────────────────────────────────────────
log.info(f"\nTotal training time: {elapsed:.1f} min")
log.info("\n" + "="*82)
log.info("SUMMARY  feat_v3 + v10-4  — topk_frac sweep")
log.info("="*82)
header = (f"{'Date':<12}  "
          + "  ".join(f"frac={f:.2f} AP    Rwd" for f in FRACS))
log.info(header)
log.info("-"*82)

all_dates = [d for d in dates if d in results[FRACS[0]]]
for d in all_dates:
    row = f"{d:<12}  "
    for frac in FRACS:
        r = results[frac].get(d, {})
        ap  = r.get("ap",  0.0)
        rwd = r.get("reward", 0.0)
        fpr = r.get("fpr",  0.0)
        cliff = "!" if fpr > 0.10 else " "
        row += f"{ap:.4f}{cliff}  {rwd:.4f}    "
    log.info(row)

log.info("-"*82)
row = f"{'MEAN':<12}  "
for frac in FRACS:
    aps  = [results[frac][d]["ap"]     for d in all_dates if d in results[frac]]
    rwds = [results[frac][d]["reward"] for d in all_dates if d in results[frac]]
    fprs = [results[frac][d]["fpr"]    for d in all_dates if d in results[frac]]
    row += f"{np.mean(aps):.4f}   {np.mean(rwds):.4f}    "
    cliffs = sum(1 for f in fprs if f > 0.10)
    log.info(f"  frac={frac:.2f}: mean AP={np.mean(aps):.4f}  Reward={np.mean(rwds):.4f}  Cliffs={cliffs}")
log.info(row)

log.info(f"\nLog saved: {log_path}")
