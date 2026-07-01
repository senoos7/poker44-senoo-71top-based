"""Bucket-snapped amount features (d0 features_v2 idea): quantize bet/pot/stack
sizes to the validator's EXACT visible bb-bucket grid and use the BUCKET INDEX
(bounded 0..15) instead of the raw magnitude. This (a) cancels the sanitizer's
injected bucket noise and (b) keeps amount features BOUNDED so they don't go OOD
to extreme leaf values on the disjoint live bet-scale (live bb<=2.2 vs benchmark
<=126). Aggregated to the chunk with order-stats so 30-hand and ~90-hand chunks
look alike. Sanitization- and size-invariant; train==serve safe.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

# payload_view._VISIBLE_BB_BUCKETS
_BUCKETS = np.array((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0,
                     24.0, 36.0, 56.0, 84.0, 126.0))
_STATS = ("mean", "std", "min", "max", "q10", "q50", "q90")
_PREFIXES = ("bkt_amt_mean", "bkt_amt_max", "bkt_amt_std", "bkt_amt_nonzero",
             "bkt_pot_mean", "bkt_stack_mean")


def _bidx(v: float) -> float:
    v = max(0.0, float(v))
    return float(int(np.argmin(np.abs(_BUCKETS - v))))   # nearest bucket index 0..15


def _agg(prefix: str, vals: List[float], out: Dict[str, float]) -> None:
    if not vals:
        for s in _STATS:
            out[f"{prefix}_{s}"] = 0.0
        return
    a = np.asarray(vals, dtype=float)
    out[f"{prefix}_mean"] = float(a.mean()); out[f"{prefix}_std"] = float(a.std())
    out[f"{prefix}_min"] = float(a.min()); out[f"{prefix}_max"] = float(a.max())
    out[f"{prefix}_q10"] = float(np.quantile(a, 0.1))
    out[f"{prefix}_q50"] = float(np.quantile(a, 0.5))
    out[f"{prefix}_q90"] = float(np.quantile(a, 0.9))


def feature_names() -> List[str]:
    return [f"{p}_{s}" for p in _PREFIXES for s in _STATS]


def bucket_amount_feats(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    per_amt_mean: List[float] = []; per_amt_max: List[float] = []
    per_amt_std: List[float] = []; per_nonzero: List[float] = []
    per_pot: List[float] = []; per_stack: List[float] = []
    for hand in (chunk or []):
        hand = hand or {}
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        amts = [_bidx(float((a or {}).get("normalized_amount_bb", 0.0))) for a in actions]
        pots = [_bidx(float((a or {}).get("pot_before", 0.0)) / 0.02)
                for a in actions if (a or {}).get("pot_before")]
        stacks = [_bidx(float((p or {}).get("starting_stack", 0.0)) / 0.02)
                  for p in players if isinstance(p, dict)]
        if amts:
            aa = np.asarray(amts)
            per_amt_mean.append(float(aa.mean())); per_amt_max.append(float(aa.max()))
            per_amt_std.append(float(aa.std())); per_nonzero.append(float((aa > 0).mean()))
        if pots: per_pot.append(float(np.mean(pots)))
        if stacks: per_stack.append(float(np.mean(stacks)))
    out: Dict[str, float] = {}
    _agg("bkt_amt_mean", per_amt_mean, out)
    _agg("bkt_amt_max", per_amt_max, out)
    _agg("bkt_amt_std", per_amt_std, out)
    _agg("bkt_amt_nonzero", per_nonzero, out)
    _agg("bkt_pot_mean", per_pot, out)
    _agg("bkt_stack_mean", per_stack, out)
    return out
