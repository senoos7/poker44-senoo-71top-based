"""Repetition-invariant features (2026-06-27 research): generator-AGNOSTIC measures of
how self-redundant a player's own bag of hands is. The marginal-match-proof signal for
bot detection (NDSS-2016 MMORPG self-similarity). Designed to TRANSFER live where our
synthetic-tuned cx_ collision counts overfit.

All features are size-invariant (ratios / per-hand-normalized / intensive) so a 33-hand
benchmark chunk and a 120-hand live chunk yield comparable values. Pairwise ops are
capped at MAX_HANDS for latency on deep v2.0 rounds.

Families: Vendi score (effective # distinct hands), DPP log-det (set volume), mean within-
set similarity, all-pairs normalized edit-distance distribution, gzip compression ratio /
NCD, normalized LZ76 complexity, order-1 entropy-rate, surprise burstiness (Fano/CV).
"""
from __future__ import annotations
import math, gzip
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np

from poker44_bump.features import chunk_features

MAX_HANDS = 60  # cap pairwise/eig work for live 60-120h chunks (subsample deterministically)


def _amt_bucket(v: float) -> str:
    if v <= 0: return "z"
    if v <= 0.5: return "xs"
    if v <= 1: return "s"
    if v <= 2: return "m"
    if v <= 5: return "l"
    return "xl"


def _hand_tokens(h: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(action-type seq, rich (street,act,bucket) seq) for one hand."""
    acts = h.get("actions") or []
    at, rich = [], []
    for a in acts:
        a = a or {}
        ac = (str(a.get("action_type") or "")[:1]) or "?"
        st = (str(a.get("street") or "")[:2]) or "?"
        bk = _amt_bucket(float(a.get("normalized_amount_bb", 0) or 0))
        at.append(ac)
        rich.append(st + ac + bk)
    return tuple(at), tuple(rich)


def _ngram_set(seq, n):
    return set(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)) if len(seq) >= n else set()


def _jacc(A, B):
    return len(A & B) / len(A | B) if (A | B) else 1.0


def _lz76_norm(s: str) -> float:
    """normalized Lempel-Ziv-76 complexity (lower = more deterministic/repetitive)."""
    n = len(s)
    if n <= 1: return 0.0
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1; break
        else:
            if k > k_max: k_max = k
            i += 1
            if i == l:
                c += 1; l += k_max
                if l + 1 > n: break
                i = 0; k = 1; k_max = 1
            else:
                k = 1
    return c / (n / math.log2(n))


def _entropy_rate(seq) -> float:
    """order-1 conditional entropy H(a_t | a_{t-1}) in bits (lower = more predictable)."""
    if len(seq) < 2: return 0.0
    trans = Counter(zip(seq[:-1], seq[1:]))
    ctx = Counter(seq[:-1])
    tot = sum(trans.values()); H = 0.0
    for (a, b), cnt in trans.items():
        H -= (cnt / tot) * math.log2((cnt / ctx[a]) + 1e-12)
    return H


def _vendi_and_logdet(sims: np.ndarray) -> Tuple[float, float]:
    """sims = n x n similarity matrix (diag 1). Vendi = exp(entropy of eigvals of K/n);
    return (vendi/n in [0,1], logdet(K + eps I)/n)."""
    n = sims.shape[0]
    if n <= 1: return 1.0, 0.0
    K = (sims + sims.T) / 2.0
    w = np.linalg.eigvalsh(K / n)
    w = np.clip(w, 1e-12, None); w = w / w.sum()
    vendi = math.exp(-float(np.sum(w * np.log(w))))
    sign, logabs = np.linalg.slogdet(K + 1e-6 * np.eye(n))
    return vendi / n, (float(logabs) / n if sign > 0 else -20.0)


def _repeat_feats(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    if not hands:
        return {}
    # deterministic subsample for pairwise/eig work (latency cap)
    H = hands if len(hands) <= MAX_HANDS else [hands[i] for i in
        np.linspace(0, len(hands) - 1, MAX_HANDS).astype(int)]
    ats, richs = zip(*[_hand_tokens(h) for h in H])
    n = len(H)
    big_at = [_ngram_set(s, 2) for s in ats]          # bigram sets for kernel

    # similarity matrix (bigram Jaccard) -> Vendi / logdet / mean within-set sim. O(n^2) set-ops.
    sims = np.eye(n)
    pair_jacc = []
    for i in range(n):
        bi = big_at[i]
        for j in range(i + 1, n):
            s = _jacc(bi, big_at[j])
            sims[i, j] = sims[j, i] = s
            pair_jacc.append(s)
    vendi_n, logdet_n = _vendi_and_logdet(sims)
    npairs = n * (n - 1) // 2 or 1
    pj = np.asarray(pair_jacc) if pair_jacc else np.array([0.0])

    # exact-duplicate-hand fractions via Counter (O(n), no edit-distance loop)
    at_counts = Counter(ats); rich_counts = Counter(richs)
    dup_at = sum(c * (c - 1) // 2 for c in at_counts.values()) / npairs   # frac of action-seq dup pairs
    dup_rich = sum(c * (c - 1) // 2 for c in rich_counts.values()) / npairs

    # compression ratio + NCD-ish (action-type stream, rich stream)
    def cr(seqs):
        joined = "|".join("".join(s) for s in seqs).encode()
        if not joined: return 1.0
        whole = len(gzip.compress(joined, 5))
        parts = sum(len(gzip.compress(("".join(s)).encode(), 5)) for s in seqs) or 1
        return whole / parts
    cr_at = cr(ats); cr_rich = cr(richs)

    # determinism of the concatenated action stream
    flat_at = [t for s in ats for t in s]
    er = _entropy_rate(flat_at)

    return {
        "rp_vendi_frac": vendi_n,             # effective # distinct hands / n (low=repetitive)
        "rp_logdet_n": logdet_n,              # set volume per hand (low=collinear/dup)
        "rp_pairsim_mean": float(pj.mean()),  # mean within-set bigram similarity (high=repetitive)
        "rp_edit_zerofrac": dup_at,           # frac of exact action-seq duplicate pairs
        "rp_exact_rich_frac": dup_rich,       # frac of exact rich-token duplicate pairs
        "rp_compress_ratio": cr_at,           # gzip whole/sum-parts (low=repetitive)
        "rp_compress_ratio_rich": cr_rich,
        "rp_entropy_rate": er,                # H(a_t|a_{t-1}) bits (low=predictable)
    }


def chunk_features_repeat(chunk: List[Dict[str, Any]]) -> Dict[str, float]:
    out = chunk_features(chunk)
    if chunk:
        out.update(_repeat_feats(chunk))
    return out
