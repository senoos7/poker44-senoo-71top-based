"""v10 LODO: sweep _topk_squeeze frac values (0.15, 0.16, 0.17, 0.18, 0.19, 0.20).

Trains v10 once per LODO fold, saves raw scores, then evaluates all frac values
without re-training.  Fast: same wall time as a single v10 LODO run (~2-3 min).
Data: v1.12 only (benchmark_YYYY-MM-DD.json files, May 26 – present).

Run:
    PYTHONPATH=. python -W ignore lodo_frac_sweep.py
Output: logs/topk_frac_sweep_<timestamp>.log
"""
from __future__ import annotations
import sys, time, math, datetime
from pathlib import Path
sys.path.insert(0, ".")

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb

sys.path.insert(0, "/root/Skip/poker/poker_demo_2")

import training.build_dataset as bd
from poker44_ml.features import chunk_features as base_cf
from poker44_bump.model_v5 import _topk_squeeze

# ── logging setup ─────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
_ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_logpath = Path("logs") / f"topk_frac_sweep_{_ts}.log"
_logfile = open(_logpath, "w", encoding="utf-8", buffering=1)
_real_stdout = sys.stdout

class _Tee:
    def write(self, m):  _real_stdout.write(m); _logfile.write(m)
    def flush(self):     _real_stdout.flush();  _logfile.flush()

sys.stdout = _Tee()

print(f"log → {_logpath}")
print(f"started: {_ts}")
print()

# ── data (v1.12 only — benchmark_YYYY-MM-DD.json files) ──────────────────────
exs   = bd.load_benchmark_examples(
    sorted(Path("data/benchmark_out").glob("benchmark_2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))
)
y      = np.array([int(e["label"]) for e in exs])
dates  = [e["source_date"] for e in exs]
uniq   = sorted(set(dates))

fd    = [base_cf(e["chunk"]) for e in exs]
names = sorted({k for d in fd for k in d})
X     = np.asarray([[float(d.get(n, 0.0)) for n in names] for d in fd])

FRACS  = [0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
SEED   = 42

print(f"[data] {len(exs)} chunks / {len(uniq)} dates / {len(names)} features")
print(f"       frac sweep: {FRACS}")
print()

# ── helpers ───────────────────────────────────────────────────────────────────
def make_v10():
    return [
        lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
                           min_child_samples=40, subsample=0.7, colsample_bytree=0.6,
                           reg_lambda=5.0, reg_alpha=2.0, random_state=SEED+1, n_jobs=4, verbose=-1),
        lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=63,
                           min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
                           reg_lambda=2.0, random_state=SEED+2, n_jobs=4, verbose=-1),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                             random_state=SEED+3, n_jobs=4),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                               random_state=SEED+4, n_jobs=4),
    ]

def eval_frac(raw: np.ndarray, labels: np.ndarray, frac: float) -> dict:
    """Apply _topk_squeeze(frac) and compute metrics at threshold=0.5."""
    sq  = np.array(_topk_squeeze(raw, frac))
    k   = max(0, min(len(raw), int(math.floor(len(raw) * frac))))
    bot = sq > 0.5                             # top-k flagged as bot
    n_bot = int(labels.sum())
    n_hum = int((labels == 0).sum())
    tp = int((bot & (labels == 1)).sum())
    fp = int((bot & (labels == 0)).sum())
    fpr    = fp / max(n_hum, 1)
    recall = tp / max(n_bot, 1)
    ap     = float(average_precision_score(labels, sq)) if labels.max() != labels.min() else 0.0
    saf    = 0.0 if fpr >= 0.10 else (1.0 - fpr) ** 2
    reward = (0.65 * ap + 0.35 * recall) * saf
    return dict(frac=frac, k=k, ap=ap, recall=recall, fpr=fpr,
                reward=reward, cliff=int(fpr >= 0.10))

# ── LODO with raw score collection ───────────────────────────────────────────
print("="*70)
print("Running LODO for v10 (trains once per fold, sweeps frac post-hoc)...")
print("="*70)

# fold_results[date] = list of eval_frac results for each frac
fold_results: dict[str, list[dict]] = {}

t_total = time.time()
for d in uniq:
    te_idx = [i for i,x in enumerate(dates) if x==d]
    tr_idx = [i for i,x in enumerate(dates) if x!=d]
    y_tr = y[tr_idx]; y_te = y[te_idx]
    if len(set(y_tr)) < 2:
        continue

    ests = make_v10()
    acc  = np.zeros(len(te_idx))
    t0   = time.time()
    for est in ests:
        est.fit(X[tr_idx], y_tr)
        p = est.predict_proba(X[te_idx])
        acc += p[:, 1]
    raw = acc / len(ests)
    elapsed = time.time() - t0

    evals = [eval_frac(raw, y_te, f) for f in FRACS]
    fold_results[d] = evals

    # compact per-fold line: show AP and Reward for each frac
    frac_str = "  ".join(
        f"f={e['frac']:.2f}→Rew={e['reward']:.3f}{'C' if e['cliff'] else ' '}"
        for e in evals
    )
    print(f"  {d} (k={[e['k'] for e in evals]}) │ {frac_str}  [{elapsed:.0f}s]")

print(f"\nLODO done in {time.time()-t_total:.0f}s\n")

# ── aggregate across dates ────────────────────────────────────────────────────
print("="*90)
print(f"SUMMARY: v10 + _topk_squeeze — frac sweep over {len(fold_results)} LODO dates")
print("="*90)
print(f"\n{'frac':>6} {'k/n':>6}  {'AP':>7} {'Recall':>7} {'FPR':>7} {'Reward':>8} {'Cliff%':>7}  {'Rew>0 days':>10}")
print("-"*72)

for fi, frac in enumerate(FRACS):
    fold_evals = [fold_results[d][fi] for d in fold_results]
    mean_k  = np.mean([e["k"] for e in fold_evals])
    mean_n  = np.mean([len(y[[i for i,x in enumerate(dates) if x==d]]) for d in fold_results])
    ap      = np.mean([e["ap"]     for e in fold_evals])
    recall  = np.mean([e["recall"] for e in fold_evals])
    fpr_m   = np.mean([e["fpr"]    for e in fold_evals])
    reward  = np.mean([e["reward"] for e in fold_evals])
    cliff_p = 100 * sum(e["cliff"] for e in fold_evals) / len(fold_evals)
    rew_pos = sum(1 for e in fold_evals if e["reward"] > 0)
    print(f"{frac:>6.2f} {mean_k:>3.0f}/{mean_n:>2.0f}  {ap:7.4f} {recall:7.4f} {fpr_m:7.4f} {reward:8.4f} {cliff_p:7.1f}  {rew_pos:>4}/{len(fold_evals)}")

# ── per-date table ────────────────────────────────────────────────────────────
print()
print(f"\n{'Date':<12}", end="")
for f in FRACS:
    print(f" {'frac='+str(f):>14}", end="")
print()
print(f"{'':12}", end="")
for _ in FRACS:
    print(f"  {'AP Rew Rec':>12}", end="")
print()
print("-" * (12 + 15 * len(FRACS)))

for d in sorted(fold_results):
    print(f"{d:<12}", end="")
    for e in fold_results[d]:
        cliff_mark = "C" if e["cliff"] else " "
        print(f"  {e['ap']:.3f} {e['reward']:.3f} {e['recall']:.2f}{cliff_mark}", end="")
    print()

# ── reward vs frac chart (ASCII) ─────────────────────────────────────────────
print()
print("Mean Reward by frac (ASCII bar):")
rewards = []
for fi, frac in enumerate(FRACS):
    fold_evals = [fold_results[d][fi] for d in fold_results]
    rewards.append((frac, np.mean([e["reward"] for e in fold_evals])))

max_rew = max(r for _, r in rewards)
for frac, rew in rewards:
    bar = "█" * int(rew / max_rew * 40)
    print(f"  frac={frac:.2f}  {bar:<40}  {rew:.4f}")
