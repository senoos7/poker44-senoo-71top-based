"""Class-conditional n-gram Markov log-likelihood-ratio (LLR) feature channel.

The synthetic bots are adversarially marginal-matched to humans (every aggregate
feature AUC ~ 0.50), so the residual fingerprint lives in higher-order SEQUENTIAL
transition structure. This module fits two smoothed class-conditional n-gram
language models over discretized action tokens -- one on HUMAN hands, one on BOT
hands -- and scores each hand by the per-token LLR  sum_t log[ P_bot(t|ctx) /
P_human(t|ctx) ].  Per-chunk aggregates of the ~30 hand LLRs become features.

Why this survives marginal-matching: the LLR is the smoothed, sign-contrasted,
low-dimensional view of the joint Markov structure that single-feature matching
cannot erase (Leslie-Eskin: k-mer spectrum == order-(k-1) Markov sufficient
statistics). Smoothing kills the zero-count artifacts that flip sign across dates.

MUST be fit out-of-fold (by date) during training to avoid leakage; the deployed
artifact fits on all training data. Picklable (pure python dict tables).
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_BOS = "^"   # beginning-of-hand context marker


def _bet_bin(v: float) -> str:
    try:
        v = float(v or 0.0)
    except Exception:
        v = 0.0
    if v <= 0:   return "z"
    if v <= 1:   return "a"
    if v <= 3:   return "b"
    if v <= 6:   return "c"
    return "d"


def _tokenize(hand: Dict[str, Any]) -> List[str]:
    """Action sequence -> token strings  (action_type | bet_bin | street)."""
    toks: List[str] = []
    for a in (hand.get("actions") or []):
        a = a or {}
        at = str(a.get("action_type") or "?").lower()
        st = str(a.get("street") or "?").lower()[:2]          # pf/fl/tu/ri
        bb = _bet_bin(a.get("normalized_amount_bb", 0))
        toks.append(f"{at}|{bb}|{st}")
    return toks


class NgramLLRScorer:
    """Smoothed class-conditional n-gram LMs -> per-hand LLR -> per-chunk feats."""

    def __init__(self, orders: Sequence[int] = (1, 2), alpha: float = 0.5) -> None:
        self.orders = tuple(orders)
        self.alpha = float(alpha)
        # per order: {context_tuple: {token: count}} for each class, + vocab
        self._bot: Dict[int, Dict[Tuple, Dict[str, float]]] = {}
        self._hum: Dict[int, Dict[Tuple, Dict[str, float]]] = {}
        self._vocab: set = set()
        self._fitted = False

    # ---- fitting -------------------------------------------------------
    @staticmethod
    def _count(hands: List[List[str]], order: int):
        tables: Dict[Tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for toks in hands:
            seq = [_BOS] * (order - 1) + toks
            for i in range(order - 1, len(seq)):
                ctx = tuple(seq[i - (order - 1):i])
                tables[ctx][seq[i]] += 1.0
        return tables

    def fit(self, chunks: Sequence[List[dict]], labels: Sequence[int]) -> "NgramLLRScorer":
        bot_hands: List[List[str]] = []
        hum_hands: List[List[str]] = []
        for chunk, lab in zip(chunks, labels):
            for hand in (chunk or []):
                toks = _tokenize(hand)
                if not toks:
                    continue
                self._vocab.update(toks)
                (bot_hands if int(lab) == 1 else hum_hands).append(toks)
        for o in self.orders:
            self._bot[o] = self._count(bot_hands, o)
            self._hum[o] = self._count(hum_hands, o)
        self._fitted = True
        return self

    # ---- scoring -------------------------------------------------------
    def _logp(self, tables: Dict[Tuple, Dict[str, float]], ctx: Tuple, tok: str) -> float:
        V = max(1, len(self._vocab))
        row = tables.get(ctx)
        a = self.alpha
        if row is None:
            return math.log(1.0 / V)                  # unseen context -> uniform
        tot = sum(row.values())
        return math.log((row.get(tok, 0.0) + a) / (tot + a * V))

    def _hand_llr(self, toks: List[str]) -> float:
        if not toks:
            return 0.0
        llr = 0.0
        n = 0
        for o in self.orders:
            seq = [_BOS] * (o - 1) + toks
            for i in range(o - 1, len(seq)):
                ctx = tuple(seq[i - (o - 1):i])
                tok = seq[i]
                llr += self._logp(self._bot[o], ctx, tok) - self._logp(self._hum[o], ctx, tok)
                n += 1
        return llr / max(1, n)                          # length-normalized

    def transform_one(self, chunk: List[dict]) -> Dict[str, float]:
        vals = [self._hand_llr(_tokenize(h)) for h in (chunk or []) if (h.get("actions"))]
        if not vals:
            return {k: 0.0 for k in self.feature_names()}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "llr_mean":   float(a.mean()),
            "llr_max":    float(a.max()),
            "llr_min":    float(a.min()),
            "llr_std":    float(a.std()),
            "llr_fracpos":float((a > 0).mean()),
            "llr_q10":    float(np.quantile(a, 0.10)),
            "llr_q50":    float(np.quantile(a, 0.50)),
            "llr_q90":    float(np.quantile(a, 0.90)),
            "llr_range":  float(a.max() - a.min()),
        }

    @staticmethod
    def feature_names() -> List[str]:
        return ["llr_mean", "llr_max", "llr_min", "llr_std", "llr_fracpos",
                "llr_q10", "llr_q50", "llr_q90", "llr_range"]
