"""Train production model: features_v3 + v10-4 + topk_frac=0.13.

Trains on ALL available v1.12 data (no holdout) and saves:
  models/bump_model_v10-4v3.joblib
  models/bump_model_v10-4v3_meta.json
"""
import sys, json, logging, time, joblib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.path.append("/root/Skip/poker/poker_demo_2")

from poker44_bump.model_v10 import V10Model

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
Path("models").mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = Path(f"logs/train_v3_final_{ts}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
log.info(f"Log: {log_path}")

TOPK_FRAC  = 0.13
MODEL_NAME = "bump_model_v10-4v3"

# ── data ──────────────────────────────────────────────────────────────────────
exs = bd.load_benchmark_examples(
    sorted(Path("data/benchmark_out").glob(
        "benchmark_2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    ))
)
dates = sorted(set(e["source_date"] for e in exs))
log.info(f"Loaded {len(exs)} examples / {len(dates)} dates  (full training set)")

log.info("Extracting features_v3 …")
feats = [chunk_features_v3(e["chunk"]) for e in exs]
keys  = sorted(feats[0].keys())
X     = np.array([[f[k] for k in keys] for f in feats], dtype=np.float32)
y     = np.array([e["label"] for e in exs], dtype=np.int32)
date_arr = np.array([e["source_date"] for e in exs])
log.info(f"Feature matrix: {X.shape}  (416 features)")
log.info(f"Labels: {y.sum()} bot / {(1-y).sum()} human")

# ── ensemble definition ───────────────────────────────────────────────────────
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

# ── LODO evaluation (for metadata) ───────────────────────────────────────────
log.info("\nRunning LODO for metadata …")
lodo_results = []
t0 = time.time()
for i, heldout in enumerate(dates):
    tr_mask = date_arr != heldout
    te_mask = date_arr == heldout
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]
    if y_te.sum() == 0 or (1 - y_te).sum() == 0:
        continue

    clf = make_v10_4()
    clf.fit(X_tr, y_tr)
    raw      = clf.predict_proba(X_te)[:, 1]
    squeezed = np.array(_topk_squeeze(raw, frac=TOPK_FRAC))
    ap       = average_precision_score(y_te, squeezed)

    n  = len(y_te); k = max(1, int(n * TOPK_FRAC))
    thr   = np.sort(squeezed)[-k]
    preds = (squeezed >= thr).astype(int)
    tp    = int(((preds == 1) & (y_te == 1)).sum())
    fp    = int(((preds == 1) & (y_te == 0)).sum())
    n_pos = int(y_te.sum()); n_neg = int((1-y_te).sum())
    recall = tp / max(n_pos, 1)
    fpr    = fp / max(n_neg, 1)
    reward = ap * (0.0 if fpr > 0.10 else 1.0)
    lodo_results.append(dict(
        date=heldout, n=n, k=k, ap=ap, recall=recall, fpr=fpr, reward=reward,
        cliff=bool(fpr > 0.10), tp=tp, fp=fp
    ))
    cliff_tag = " *** CLIFF ***" if fpr > 0.10 else ""
    log.info(
        f"  [{i+1:2d}/{len(dates)}] {heldout}  n={n:2d} k={k}"
        f"  AP={ap:.4f}  Recall={recall:.3f}  FPR={fpr:.3f}"
        f"  Reward={reward:.4f}{cliff_tag}"
    )

lodo_elapsed = (time.time() - t0) / 60
n_cliffs = sum(1 for r in lodo_results if r["cliff"])
mean_ap     = float(np.mean([r["ap"]     for r in lodo_results]))
mean_recall = float(np.mean([r["recall"] for r in lodo_results]))
mean_fpr    = float(np.mean([r["fpr"]    for r in lodo_results]))
mean_reward = float(np.mean([r["reward"] for r in lodo_results]))

log.info(f"\n  LODO summary: AP={mean_ap:.4f}  Recall={mean_recall:.3f}"
         f"  FPR={mean_fpr:.4f}  Reward={mean_reward:.4f}  Cliffs={n_cliffs}")
log.info(f"  LODO time: {lodo_elapsed:.1f} min")

# ── train on ALL data ─────────────────────────────────────────────────────────
log.info("\nTraining final model on all data …")
t1 = time.time()
clf_final = make_v10_4()
clf_final.fit(X, y)
train_elapsed = (time.time() - t1) / 60

# in-sample AP (informational only)
raw_all      = clf_final.predict_proba(X)[:, 1]
squeezed_all = np.array(_topk_squeeze(raw_all, frac=TOPK_FRAC))
insample_ap  = average_precision_score(y, squeezed_all)
log.info(f"  In-sample AP: {insample_ap:.4f}  (train time: {train_elapsed:.1f} min)")

# ── wrap as V10Model (required by miner.py) and save ─────────────────────────
fitted_ests = list(clf_final.estimators_)   # fitted estimators from VotingClassifier
v10_model = V10Model(
    estimators=fitted_ests,
    feature_names=keys,
    topk_cfg={"positive_fraction": TOPK_FRAC},
    metadata={
        "model_name":    f"poker44-bump-{MODEL_NAME}",
        "model_version": MODEL_NAME,
        "framework":     f"avg-ensemble(3lgbm+2xgb+et+rf)+topk-frac{TOPK_FRAC}+feat_v3",
        "features":      "features_v3",
        "n_feats":       len(keys),
        "n_estimators":  len(fitted_ests),
        "topk_frac":     TOPK_FRAC,
        "oof_ap":        mean_ap,
        "lodo_reward":   mean_reward,
        "lodo_cliffs":   n_cliffs,
        "trained_at":    ts,
    },
)
model_path = Path(f"models/{MODEL_NAME}.joblib")
joblib.dump(v10_model, model_path)
log.info(f"\nModel saved: {model_path}  ({model_path.stat().st_size/1e6:.1f} MB)  type=V10Model")

# ── save metadata ─────────────────────────────────────────────────────────────
meta = {
    "variant":     "v10-4v3",
    "description": "feat_v3 (416) + v10-4 ensemble (3×LGB + 2×XGB + ET + RF)",
    "features":    "features_v3",
    "n_features":  416,
    "ensemble":    "v10-4 (7 estimators)",
    "topk_frac":   TOPK_FRAC,
    "trained_at":  ts,
    "n_train":     len(exs),
    "n_dates":     len(dates),
    "date_range":  f"{dates[0]} → {dates[-1]}",
    "lodo_aggregate": {
        "ap":       mean_ap,
        "recall":   mean_recall,
        "fpr":      mean_fpr,
        "reward":   mean_reward,
        "cliffs":   n_cliffs,
        "n_dates":  len(lodo_results),
        "n_feats":  416,
        "n_ests":   7,
    },
    "lodo_per_date": lodo_results,
    "hyperparams": {
        "lgb1": {"n_estimators":600,"lr":0.02,"num_leaves":31,"min_child_samples":25,"reg_lambda":2.0,"reg_alpha":0.5},
        "lgb2": {"n_estimators":500,"lr":0.015,"num_leaves":15,"max_depth":4,"min_child_samples":45,"reg_lambda":6.0,"reg_alpha":2.0},
        "lgb3": {"n_estimators":700,"lr":0.015,"num_leaves":47,"min_child_samples":35,"reg_lambda":3.0,"reg_alpha":1.0},
        "xgb1": {"n_estimators":400,"lr":0.03,"max_depth":4,"reg_lambda":2.0},
        "xgb2": {"n_estimators":350,"lr":0.02,"max_depth":3,"reg_alpha":1.5,"reg_lambda":4.0},
        "et":   {"n_estimators":500,"min_samples_leaf":6,"max_features":0.45},
        "rf":   {"n_estimators":500,"min_samples_leaf":6,"max_features":0.45},
    },
    "feature_keys": keys,
}
meta_path = Path(f"models/{MODEL_NAME}_meta.json")
meta_path.write_text(json.dumps(meta, indent=2))
log.info(f"Metadata saved: {meta_path}")

log.info(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
log.info("Done.")
