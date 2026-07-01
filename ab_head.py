"""A/B the scoring HEAD on UNSEEN windows with the REAL validator scorer.

Phase 1 (slow, cached): train on earlier dates, compute RAW per-window bot-probs.
Phase 2 (fast): sweep heads (conformal vs topk_v1 fractions) on cached raw scores.
Scores PER 40-chunk window (validator sends one window per query; topk is batch-relative).
"""
from __future__ import annotations
import glob, json, sys, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TRAVIS = HERE.parent / "repos" / "Travis861_Poker44_v1"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(TRAVIS))
from poker44.score.scoring import reward as REAL_REWARD
from poker44_bump.model import BumpModel, conformal_map

REWARD_WINDOW = 40
DATA = HERE / "data"
HOLD = 9
CACHE = HERE / "_ab_raw_cache.npz"


def build_cache():
    from poker44_bump.payload_view import prepare_hand_for_miner
    from train_bump import load_examples, make_ensemble, avg_proba
    from sklearn.model_selection import StratifiedKFold
    rows, y, dates = load_examples(str(DATA), "ext")
    feat_names = sorted({k for r in rows for k in r})
    X = np.array([[float(r.get(n, 0.0)) for n in feat_names] for r in rows])
    uniq = sorted(set(dates.tolist()))
    train_dates = set(uniq[:-HOLD]); test_dates = uniq[-HOLD:]
    trm = np.array([d in train_dates for d in dates])
    Xt, yt, dtt = X[trm], y[trm], dates[trm]
    oof = np.zeros(int(trm.sum()))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, va in skf.split(Xt, yt):
        M = make_ensemble(42); [m.fit(Xt[tr], yt[tr]) for m in M]; oof[va] = avg_proba(M, Xt[va])
    recent = set(sorted(train_dates)[-10:])
    hr = oof[(yt == 0) & np.isin(dtt, list(recent))]
    T = float(min(max(max(hr) + 0.5*(max(hr)-np.quantile(hr, 0.9)), 0.05), 0.999))
    final = make_ensemble(42); [m.fit(Xt, yt) for m in final]
    chunks, labels = [], []
    for fp in sorted(glob.glob(str(DATA / "training_benchmark_*.txt"))):
        d = Path(fp).stem.split("_")[-1]
        if d not in test_dates: continue
        data = json.load(open(fp))["data"]
        for rec in data["chunks"]:
            for hands, lab in zip(rec["chunks"], rec["groundTruth"]):
                vis = [prepare_hand_for_miner(h) for h in hands if isinstance(h, dict)]
                if vis: chunks.append(vis); labels.append(int(lab))
    model = BumpModel(final, feat_names, T, metadata={"feature_set": "ext"})
    raw = np.array(model.predict_raw(chunks))
    np.savez(CACHE, raw=raw, labels=np.array(labels), T=np.array([T]),
             oof=oof, oof_y=yt, oof_dates=dtt,
             test_first=test_dates[0], test_last=test_dates[-1])
    print(f"CACHED raw for {len(raw)} unseen + {len(oof)} OOF chunks, T={T:.4f}, unseen {test_dates[0]}..{test_dates[-1]}")


def topk_squeeze(raw, frac, pf=0.501, pc=0.509, nc=0.49):
    n = len(raw); out = np.zeros(n)
    k = max(0, min(n, int(np.floor(n*frac))))
    order = np.argsort(-raw, kind="stable")
    pos, neg = order[:k], order[k:]
    if k > 0:
        denom = max(1, k-1)
        for r, i in enumerate(pos): out[i] = pf + (1.0 - r/denom)*(pc-pf)
    if len(neg) > 0:
        nv = raw[neg]; mn, mx = nv.min(), nv.max(); span = max(mx-mn, 1e-9)
        for i in neg: out[i] = max(0.0, min(nc, (raw[i]-mn)/span*nc))
    return out


def sweep():
    z = np.load(CACHE, allow_pickle=True)
    raw, labels, T = z["raw"], z["labels"], float(z["T"][0])
    n_win = len(raw)//REWARD_WINDOW
    print(f"unseen {z['test_first']}..{z['test_last']} | {len(raw)} chunks -> {n_win} windows | T={T:.4f}\n")

    def run(name, fn):
        ws, aps, recs, fprs, safs = [], [], [], [], []
        for w in range(n_win):
            sl = slice(w*REWARD_WINDOW, (w+1)*REWARD_WINDOW)
            sc = fn(raw[sl])
            r, info = REAL_REWARD(np.asarray(sc), labels[sl])
            ws.append(r); aps.append(info['ap_score']); recs.append(info['bot_recall'])
            fprs.append(info['fpr']); safs.append(info['human_safety_penalty'])
        mr = float(np.mean(ws)); cliffs = sum(1 for f in fprs if f >= 0.10)
        print(f"{name:<22} reward={mr:.3f} ap={np.mean(aps):.3f} recall={np.mean(recs):.3f} "
              f"fpr={np.mean(fprs):.3f} safety={np.mean(safs):.3f} cliffs={cliffs}/{n_win} -> comp~{0.696*mr+0.127:.3f}")
        return mr

    run("conformal (current)", lambda r: conformal_map(r, T, 0.02, 0.98))
    for frac in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        run(f"topk frac={frac}", lambda r, f=frac: topk_squeeze(r, f))
    print("\ncomp~ = 0.696*reward+0.127 (live regression). Targets: uid32 0.592, uid136 0.621.")

    # ---- robust frac tuning across ALL 40-windows (OOF train + unseen) ----
    print("\n=== robust frac tuning over OOF+unseen windows ===")
    oof, oy = z["oof"], z["oof_y"]
    allraw = np.concatenate([oof, raw]); ally = np.concatenate([oy, labels])
    nw = len(allraw)//REWARD_WINDOW
    print(f"total windows for tuning = {nw} ({len(allraw)} chunks)")
    best = None
    for frac in [round(0.10+0.025*k, 3) for k in range(0, 17)]:   # 0.10..0.50
        rs, fprs = [], []
        for w in range(nw):
            sl = slice(w*REWARD_WINDOW, (w+1)*REWARD_WINDOW)
            sc = topk_squeeze(allraw[sl], frac)
            r, info = REAL_REWARD(np.asarray(sc), ally[sl])
            rs.append(r); fprs.append(info['fpr'])
        mr = float(np.mean(rs)); maxf = float(np.max(fprs)); cliffs = sum(f >= 0.10 for f in fprs)
        worst = float(np.min(rs))
        flag = "OK" if maxf < 0.06 and cliffs == 0 else ("edge" if cliffs == 0 else "CLIFF")
        print(f"  frac={frac:<5} mean_reward={mr:.3f} worst_window={worst:.3f} max_fpr={maxf:.3f} cliffs={cliffs} [{flag}]")
        # prefer 0 cliffs + fpr margin, then max mean reward
        key = (cliffs == 0 and maxf < 0.06, mr)
        if best is None or key > best[0]:
            best = (key, frac, mr, maxf)
    print(f"-> robust pick: frac={best[1]} (mean_reward={best[2]:.3f}, max_fpr={best[3]:.3f})  safety-first w/ fpr<0.06 margin")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
    else:
        build_cache()
