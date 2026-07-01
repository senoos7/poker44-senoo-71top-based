"""Self-contained trainer for the Poker44 'bump' miner.

Pipeline:
  1. Load all released benchmark dates (training_benchmark_<date>.txt).
  2. Build features = chunk_features(prepare_hand_for_miner(hand)) per chunk
     (matches the reference build_dataset miner_visible convention).
  3. 5-fold stratified OOF predictions with an averaged tree ensemble
     (LightGBM + XGBoost + ExtraTrees + RandomForest).
  4. Fit the CLIFF-ROBUST conformal threshold T on the most-recent W dates'
     OOF human scores: T = max(human) + BUF * (max(human) - q90(human)).
  5. Refit the ensemble on ALL data and save {models, feature_names, T, meta}.

Reports OOF AP and the simulated validator reward / per-date FPR at T.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from poker44_bump.features import chunk_features
from poker44_bump.features_ext import chunk_features_ext
from poker44_bump.payload_view import prepare_hand_for_miner
from poker44_bump.model import BumpModel, conformal_map

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def load_examples(data_dir: str, feature_set: str = "ext"):
    extractor = chunk_features_ext if feature_set == "ext" else chunk_features
    rows, labels, dates = [], [], []
    files = sorted(glob.glob(os.path.join(data_dir, "training_benchmark_*.txt")))
    if not files:
        raise FileNotFoundError(f"no benchmark files in {data_dir}")
    for fp in files:
        payload = json.load(open(fp))
        data = payload.get("data", payload)
        src_date = str(data.get("sourceDate") or Path(fp).stem.split("_")[-1])
        for rec in data.get("chunks", []):
            groups = rec.get("chunks") or []
            gts = rec.get("groundTruth") or rec.get("groundTruthLabels") or []
            rec_date = str(rec.get("sourceDate") or src_date)
            for hands, label in zip(groups, gts):
                if not isinstance(hands, list):
                    continue
                visible = [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]
                if not visible:
                    continue
                feats = extractor(visible)
                feats["hand_count"] = float(len(visible))
                rows.append(feats); labels.append(int(label)); dates.append(rec_date)
    return rows, np.array(labels), np.array(dates)


def make_ensemble(seed=42):
    return [
        LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                       colsample_bytree=0.8, min_child_samples=8, verbose=-1, random_state=seed),
        XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", verbosity=0, random_state=seed),
        ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                             n_jobs=-1, random_state=seed),
        RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                               n_jobs=-1, random_state=seed),
    ]


def avg_proba(models, X):
    return np.mean([np.asarray(m.predict_proba(X))[:, 1] for m in models], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--output", default=str(HERE / "models" / "bump_model.joblib"))
    ap.add_argument("--calib-window", type=int, default=6, help="recent dates for threshold fit")
    ap.add_argument("--buffer", type=float, default=-1.0,
                    help="drift buffer coef; <0 = auto-select smallest 0-cliff buffer (max reward)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--repo-commit", default="")
    ap.add_argument("--feature-set", choices=["base", "ext"], default="ext")
    args = ap.parse_args()

    rows, y, dates = load_examples(args.data_dir, feature_set=args.feature_set)
    feat_names = sorted({k for r in rows for k in r})
    train_chunk_size = int(np.median([float(r.get("hand_count", 0.0)) for r in rows]))
    X = np.array([[float(r.get(n, 0.0)) for n in feat_names] for r in rows], dtype=np.float64)
    uniq = sorted(set(dates.tolist()))
    print(f"examples={len(y)} feats={len(feat_names)} dates={len(uniq)} ({uniq[0]}..{uniq[-1]}) "
          f"bot={int(y.sum())} human={int((1-y).sum())}")

    # ---- OOF predictions ----
    oof = np.zeros(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for tr, va in skf.split(X, y):
        models = make_ensemble(args.seed)
        for m in models:
            m.fit(X[tr], y[tr])
        oof[va] = avg_proba(models, X[va])
    oof_ap = average_precision_score(y, oof)
    print(f"OOF AP (ranking quality) = {oof_ap:.4f}")

    # ---- conformal threshold: AUTO-SELECT buffer for 0 OOF cliffs, max reward ----
    recent = set(uniq[-args.calib_window:])
    hmask = (y == 0) & np.isin(dates, list(recent))
    human_recent = oof[hmask]
    hmax = float(human_recent.max()); hq90 = float(np.quantile(human_recent, 0.90))

    def reward_at(scores, labels, T):
        preds = (scores >= T).astype(int); yt = labels.astype(int)
        tp=((preds==1)&(yt==1)).sum();fp=((preds==1)&(yt==0)).sum()
        tn=((preds==0)&(yt==0)).sum();fn=((preds==0)&(yt==1)).sum()
        fpr=fp/max(tn+fp,1);rec=tp/max(tp+fn,1)
        a=average_precision_score(yt,scores) if (yt.max()==1 and yt.min()==0) else 0.0
        saf=0.0 if fpr>=0.10 else (1-fpr)**2
        return (0.65*a+0.35*rec)*saf, fpr, rec

    def eval_T(T):
        R=[];cliff=0;F=[]
        for d in uniq:
            m = dates==d
            r,fpr,_ = reward_at(oof[m], y[m], T); R.append(r); F.append(fpr); cliff+=int(fpr>=0.10)
        return float(np.mean(R)), cliff, float(np.max(F))

    if args.buffer >= 0:                       # fixed buffer (back-compat)
        buffers = [args.buffer]
    else:                                      # auto-tune: smallest safe buffer, max reward
        buffers = [round(0.3 + 0.1*k, 2) for k in range(0, 23)]  # 0.3 .. 2.5
    best = None
    for buf in buffers:
        T = float(min(max(hmax + buf*(hmax - hq90), 0.05), 0.999))
        mr, cliff, mfpr = eval_T(T)
        # prefer 0 cliffs, then max reward; fall back to fewest cliffs
        key = (cliff, -mr)
        if best is None or key < best[0]:
            best = (key, buf, T, mr, cliff, mfpr)
    _, buffer_used, T, mr, cliff, mfpr = best
    args.buffer = buffer_used
    print(f"conformal T={T:.4f} buffer={buffer_used} (auto) recent human max={hmax:.4f} q90={hq90:.4f} window={args.calib_window}d")
    print(f"per-date OOF @T: mean_reward={mr:.3f} cliff_hits={cliff}/{len(uniq)} max_fpr={mfpr:.3f}")

    # ---- topk head: AUTO-SELECT positive_fraction over 40-chunk OOF windows ----
    # The validator sends one ~40-chunk window per query and the topk squeeze is
    # batch-relative, so tune over 40-chunk windows (in load order) not per-date.
    def topk_squeeze(raw, frac, pf=0.501, pc=0.509, nc=0.49):
        rawa = np.asarray(raw, float); m = len(rawa); o = np.zeros(m)
        k = max(0, min(m, int(np.floor(m*frac))))
        order = np.argsort(-rawa, kind="stable"); pos, neg = order[:k], order[k:]
        if k > 0:
            dn = max(1, k-1)
            for rk, i in enumerate(pos): o[i] = pf + (1.0-rk/dn)*(pc-pf)
        if len(neg) > 0:
            nv = rawa[neg]; mn, mx = nv.min(), nv.max(); sp = max(mx-mn, 1e-9)
            for i in neg: o[i] = max(0.0, min(nc, (rawa[i]-mn)/sp*nc))
        return o
    WIN = 40
    nwin = len(oof)//WIN
    def eval_frac(frac):
        R, F = [], []
        for w in range(nwin):
            sl = slice(w*WIN, (w+1)*WIN)
            sc = topk_squeeze(oof[sl], frac); yy = y[sl].astype(int)
            preds = (sc >= 0.5).astype(int)
            tp = ((preds==1)&(yy==1)).sum(); fp = ((preds==1)&(yy==0)).sum()
            tn = ((preds==0)&(yy==0)).sum(); fn = ((preds==0)&(yy==1)).sum()
            fpr = fp/max(tn+fp,1); rec = tp/max(tp+fn,1)
            a = average_precision_score(yy, sc) if (yy.max()==1 and yy.min()==0) else 0.0
            saf = 0.0 if fpr>=0.10 else (1-fpr)**2
            R.append((0.65*a+0.35*rec)*saf); F.append(fpr)
        return (float(np.mean(R)) if R else 0.0, float(np.max(F)) if F else 0.0,
                sum(f>=0.10 for f in F))
    topk_frac = 0.15
    if nwin >= 2:
        bestf = None
        for frac in [round(0.10+0.025*k, 3) for k in range(0, 17)]:  # 0.10..0.50
            mrk, maxf, cl = eval_frac(frac)
            # safety-first: 0 cliffs AND max_fpr below margin, then max reward
            key = (cl == 0 and maxf < 0.03, mrk)
            if bestf is None or key > bestf[0]:
                bestf = (key, frac, mrk, maxf, cl)
        topk_frac = bestf[1]
        print(f"topk head auto frac={topk_frac} over {nwin} OOF windows: "
              f"mean_reward={bestf[2]:.3f} max_fpr={bestf[3]:.3f} cliffs={bestf[4]}")

    # ---- refit on ALL data, save ----
    final_models = make_ensemble(args.seed)
    for m in final_models:
        m.fit(X, y)
    meta = {
        "model_name": "poker44-bump-robust",
        "model_version": "bump-topk-v4-ext11" if args.feature_set == "ext" else "bump-topk-v1",
        "framework": "tree-ensemble+topk-safety-budget",
        "feature_set": args.feature_set,
        "ensemble_combiner": "mean(lgbm,xgb,extratrees,rf)",
        # scoring head: batch-relative topk safety budget (beats fixed-T conformal offline)
        "head_mode": "topk",
        "topk_cfg": {"positive_fraction": topk_frac,
                     "positive_floor": 0.501, "positive_ceiling": 0.509, "negative_ceiling": 0.49},
        # full-chunk scoring (no subsample) — more hands = stronger collision signal, matches leaders
        "subsample": False,
        "conformal_threshold": T, "calib_window_days": args.calib_window, "buffer_coef": args.buffer,
        "train_chunk_size": train_chunk_size, "bag": 5,
        "oof_ap": round(float(oof_ap), 5), "benchmark_rows": int(len(y)),
        "train_source_dates": uniq, "feature_count": len(feat_names),
        "repo_url": args.repo_url, "repo_commit": args.repo_commit,
        "built_unix": int(time.time()),
    }
    model = BumpModel(final_models, feat_names, T, metadata=meta)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"saved -> {args.output}  ({Path(args.output).stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
