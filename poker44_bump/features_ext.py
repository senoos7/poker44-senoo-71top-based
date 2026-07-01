"""Extended features: baseline chunk_features + screened n-gram/collision signals.

Only features that screened with |AUC-0.5| >= ~0.04 AND high per-date sign
consistency (>=0.67, i.e. they do NOT flip direction date-to-date) are added,
to lift ranking quality without overfitting the small benchmark. The two
strongest (multiset collisions) had sign_consistency 1.00.
"""
from __future__ import annotations
import math
from collections import Counter
from typing import Any, Dict, List

from poker44_bump.features import chunk_features


def _amt_bucket(v: float) -> str:
    if v <= 0: return "z"
    if v <= 0.5: return "xs"
    if v <= 1: return "s"
    if v <= 2: return "m"
    if v <= 5: return "l"
    return "xl"


def _ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)] if len(seq) >= n else []


def _share(pool):
    """(unique_share, top_share, norm_entropy) over a pooled list of tokens."""
    if not pool:
        return (1.0, 0.0, 0.0)
    c = Counter(pool); tot = sum(c.values())
    uniq = len(c) / len(pool)
    top = max(c.values()) / len(pool)
    if len(c) > 1:
        ent = -sum((v / tot) * math.log(v / tot + 1e-12) for v in c.values()) / math.log(len(c))
    else:
        ent = 0.0
    return (uniq, top, ent)


def _jaccard(a, b) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def _extra_feats(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    pool_tri, pool_betbi, pool_seat_act, pool_street_act, pool_4 = [], [], [], [], []
    action_multisets, bet_multisets, action_seqs = [], [], []
    for h in hands:
        acts = h.get("actions") or []
        at = [str((a or {}).get("action_type") or "").lower() for a in acts]
        bb = [_amt_bucket(float((a or {}).get("normalized_amount_bb", 0) or 0)) for a in acts]
        seats = [int((a or {}).get("actor_seat", 0) or 0) for a in acts]
        streets = [str((a or {}).get("street") or "").lower() for a in acts]
        pool_tri += _ngrams(at, 3)
        pool_betbi += _ngrams(bb, 2)
        pool_4 += _ngrams(at, 4)
        pool_seat_act += list(zip(seats, at))
        pool_street_act += list(zip(streets, at))
        action_multisets.append(tuple(sorted(Counter(at).items())))
        bet_multisets.append(tuple(sorted(Counter(bb).items())))
        action_seqs.append(tuple(at))
    n = float(max(1, len(hands)))
    tri_u, _, _ = _share(pool_tri)
    _, _, bbi_e = _share(pool_betbi)
    sa_u, _, _ = _share(pool_seat_act)
    sta_u, _, _ = _share(pool_street_act)
    g4_u, _, _ = _share(pool_4)
    ms = Counter(action_multisets)
    bms = Counter(bet_multisets)
    # pairwise inter-hand similarity (all pairs; chunk is ~30 hands)
    seqs = [list(s) for s in action_seqs]
    pairs = [(i, j) for i in range(len(seqs)) for j in range(i + 1, len(seqs))]
    if pairs:
        jac = sum(_jaccard(seqs[i], seqs[j]) for i, j in pairs) / len(pairs)
        exact = sum(1.0 for i, j in pairs if action_seqs[i] == action_seqs[j]) / len(pairs)
    else:
        jac, exact = 1.0, 0.0
    return {
        "cx_multiset_unique_share": len(ms) / n,
        "cx_multiset_top_share": max(ms.values()) / n,
        "cx_betmultiset_unique_share": len(bms) / n,
        "cx_betmultiset_top_share": max(bms.values()) / n,
        "cx_pair_jaccard_mean": jac,
        "cx_pair_exact_match_rate": exact,
        "cx_action4gram_unique_share": g4_u,
        "cx_trigram_unique_share": tri_u,
        "cx_streetact_unique_share": sta_u,
        "cx_betbigram_entropy": bbi_e,
        "cx_seatact_unique_share": sa_u,
    }


def chunk_features_ext(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    out = chunk_features(chunk)
    if chunk:
        out.update(_extra_feats(chunk))
    return out
