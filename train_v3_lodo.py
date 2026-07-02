"""LODO comparison: features_v2+v10-4 vs features_v3+v10-4 vs features_v3+v10.

Runs three combinations so we can see both the FPR improvement from v3 features
and whether the larger v10-4 ensemble is still warranted with v3.
"""
import sys, logging, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.path.append("/root/Skip/poker/poker_demo_2")

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              VotingClassifier)
import lightgbm  as lgb
import xgboost   as xgb

import training.build_dataset as bd
from poker44_ml.features_v2 import chunk_features_v2
from poker44_ml.features_v3 import chunk_features_v3
from poker44_bump.model_v5   import _topk_squeeze

# ── logging ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = Path(f"logs/train_v3_{ts}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)
log = logging.getLogger(__name__)
log.info(f"Log: {log_path}")

# ── data ──────────────────────────────────────────────────────────────────────
exs = bd.load_benchmark_examples(
    sorted(Path("data/benchmark_out").glob(
        "benchmark_2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    ))
)
dates = sorted(set(e["source_date"] for e in exs))
log.info(f"Loaded {len(exs)} examples across {len(dates)} dates: {dates}")

# Pre-extract features for all examples
log.info("Extracting v2 features …")
feats_v2 = [chunk_features_v2(e["chunk"]) for e in exs]
log.info("Extracting v3 features …")
feats_v3 = [chunk_features_v3(e["chunk"]) for e in exs]

def to_matrix(feats_list):
    keys = sorted(feats_list[0].keys())
    return np.array([[f[k] for k in keys] for f in feats_list], dtype=np.float32), keys

X2, keys2 = to_matrix(feats_v2)
X3, keys3 = to_matrix(feats_v3)
labels = np.array([e["label"] for e in exs], dtype=np.int32)
date_arr = np.array([e["source_date"] for e in exs])
log.info(f"v2 matrix: {X2.shape}   v3 matrix: {X3.shape}")

# ── ensembles ─────────────────────────────────────────────────────────────────
def make_v10():
    return VotingClassifier(voting="soft", estimators=[
        ("lgb1", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=31,
                                     min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1)),
        ("lgb2", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.015, num_leaves=15,
                                     min_child_samples=30, subsample=0.7, colsample_bytree=0.6,
                                     reg_lambda=3.0, random_state=1, n_jobs=4, verbose=-1)),
        ("lgb3", lgb.LGBMClassifier(n_estimators=600, learning_rate=0.015, num_leaves=47,
                                     min_child_samples=25, subsample=0.75, colsample_bytree=0.65,
                                     reg_lambda=2.0, random_state=2, n_jobs=4, verbose=-1)),
        ("et",   ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                      random_state=3, n_jobs=4)),
        ("rf",   RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                        random_state=4, n_jobs=4)),
    ])

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
        ("et",   ExtraTreesClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.45,
                                      random_state=3, n_jobs=4)),
        ("rf",   RandomForestClassifier(n_estimators=500, min_samples_leaf=6, max_features=0.45,
                                        random_state=4, n_jobs=4)),
    ])

# ── LODO ──────────────────────────────────────────────────────────────────────
TOPK_FRAC = 0.15

def lodo_run(label: str, X: np.ndarray, make_clf):
    log.info(f"\n{'='*60}")
    log.info(f"  {label}")
    log.info(f"{'='*60}")
    results = {}
    for heldout in dates:
        tr_mask  = date_arr != heldout
        te_mask  = date_arr == heldout
        X_tr, y_tr = X[tr_mask], labels[tr_mask]
        X_te, y_te = X[te_mask], labels[te_mask]
        if y_te.sum() == 0 or (1 - y_te).sum() == 0:
            continue

        clf = make_clf()
        clf.fit(X_tr, y_tr)
        scores = clf.predict_proba(X_te)[:, 1]

        # topk squeeze
        squeezed = np.array(_topk_squeeze(scores, frac=TOPK_FRAC))
        ap = average_precision_score(y_te, squeezed)

        n = len(y_te); k = max(1, int(n * TOPK_FRAC))
        thr_idx = np.argsort(squeezed)[-k]
        thr = squeezed[thr_idx]
        preds = (squeezed >= thr).astype(int)
        tp = int(((preds == 1) & (y_te == 1)).sum())
        fp = int(((preds == 1) & (y_te == 0)).sum())
        fn = int(((preds == 0) & (y_te == 1)).sum())
        tn = int(((preds == 0) & (y_te == 0)).sum())
        n_pos = int(y_te.sum()); n_neg = int((1-y_te).sum())
        recall = tp / max(n_pos, 1)
        fpr    = fp / max(n_neg, 1)
        reward = ap * (0.0 if fpr > 0.10 else 1.0)
        results[heldout] = dict(ap=ap, recall=recall, fpr=fpr, reward=reward,
                                 n=n, tp=tp, fp=fp, fn=fn, tn=tn)
        cliff = " *** FPR CLIFF ***" if fpr > 0.10 else ""
        log.info(
            f"  {heldout}  n={n:2d}  "
            f"AP={ap:.4f}  Recall={recall:.3f}  FPR={fpr:.3f}  "
            f"Reward={reward:.4f}{cliff}"
        )

    aps     = [r["ap"]     for r in results.values()]
    recalls = [r["recall"] for r in results.values()]
    fprs    = [r["fpr"]    for r in results.values()]
    rewards = [r["reward"] for r in results.values()]
    log.info(
        f"  MEAN  AP={np.mean(aps):.4f}  Recall={np.mean(recalls):.3f}  "
        f"FPR={np.mean(fprs):.3f}  Reward={np.mean(rewards):.4f}  "
        f"[cliffs={sum(1 for f in fprs if f>0.10)}]"
    )
    return results

# ── run all 3 combos ──────────────────────────────────────────────────────────
t0 = time.time()
res_E = lodo_run("E  feat_v2 + v10-4  (367f, 7est)", X2, make_v10_4)
res_F = lodo_run("F  feat_v3 + v10-4  (416f, 7est)", X3, make_v10_4)
res_G = lodo_run("G  feat_v3 + v10    (416f, 5est)", X3, make_v10)

# ── summary table ─────────────────────────────────────────────────────────────
log.info("\n" + "="*70)
log.info("SUMMARY  (topk_frac=0.15, LODO over v1.12 dates)")
log.info("="*70)

header = f"{'Date':<12}  {'E AP':>6}  {'E Rwd':>6}  {'F AP':>6}  {'F Rwd':>6}  {'G AP':>6}  {'G Rwd':>6}  {'ΔAP(F-E)':>9}  {'ΔRwd(F-E)':>10}"
log.info(header)
log.info("-"*len(header))

for d in dates:
    rE = res_E.get(d, {}); rF = res_F.get(d, {}); rG = res_G.get(d, {})
    if not rE:
        continue
    e_ap  = rE["ap"];     e_rw = rE["reward"]
    f_ap  = rF.get("ap",0); f_rw = rF.get("reward",0)
    g_ap  = rG.get("ap",0); g_rw = rG.get("reward",0)
    cliff_e = " !" if rE["fpr"] > 0.10 else ""
    cliff_f = " !" if rF.get("fpr",0) > 0.10 else ""
    cliff_g = " !" if rG.get("fpr",0) > 0.10 else ""
    log.info(
        f"{d:<12}  {e_ap:6.4f}{cliff_e:<2}  {e_rw:6.4f}  "
        f"{f_ap:6.4f}{cliff_f:<2}  {f_rw:6.4f}  "
        f"{g_ap:6.4f}{cliff_g:<2}  {g_rw:6.4f}  "
        f"{(f_ap-e_ap):+9.4f}  {(f_rw-e_rw):+10.4f}"
    )

all_dates = [d for d in dates if d in res_E]
def avg(res, k): return np.mean([res[d][k] for d in all_dates if d in res])
log.info("-"*len(header))
log.info(
    f"{'MEAN':<12}  {avg(res_E,'ap'):6.4f}    {avg(res_E,'reward'):6.4f}  "
    f"{avg(res_F,'ap'):6.4f}    {avg(res_F,'reward'):6.4f}  "
    f"{avg(res_G,'ap'):6.4f}    {avg(res_G,'reward'):6.4f}  "
    f"{(avg(res_F,'ap')-avg(res_E,'ap')):+9.4f}  "
    f"{(avg(res_F,'reward')-avg(res_E,'reward')):+10.4f}"
)
log.info(f"\nCliffs — E:{sum(1 for d in all_dates if res_E.get(d,{}).get('fpr',0)>0.10)}  "
         f"F:{sum(1 for d in all_dates if res_F.get(d,{}).get('fpr',0)>0.10)}  "
         f"G:{sum(1 for d in all_dates if res_G.get(d,{}).get('fpr',0)>0.10)}")
log.info(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
log.info(f"Log saved: {log_path}")
