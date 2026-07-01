"""v8 inference wrapper (Poker44 v2.0-optimized): v6's stacked pipeline + a
STATIC FPR-targeting THRESHOLD head (replaces the fixed-fraction topk head).

Why (validated on leakage-free cross-date OOF, deep-round sim): a fixed FRACTION
guesses the live bot rate -- too high a fraction systematically cliffs (FPR>=0.10)
when humans dominate the round, and v2.0's deep rounds remove the variance that
used to hide it (one cliffed round = 20% of the 5-round cycle). A threshold on the
(rank-preserving) score sits on the stable HUMAN score distribution, so its FPR is
~constant across bot rates while recall rises with bot density. It dominated
fixed-fraction at every bot rate 10-50% and never cliffed.

Head: mark chunk positive iff score > t (t learned to hold benchmark-human
FPR ~= target). Positives -> [0.501,0.509] by rank, negatives -> [0,0.49] by rank
(ranking/AP preserved, same band scheme as the topk head). A hard CAP fraction is
a backstop against pathological batches / live human-score drift. Picklable.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_bump.model_v6 import _combined_feats


def _threshold_squeeze(raw: np.ndarray, t: float, cap: float = 0.30,
                       pf: float = 0.501, pc: float = 0.509, nc: float = 0.49) -> List[float]:
    raw = np.asarray(raw, dtype=np.float64)
    n = len(raw)
    if n == 0:
        return []
    out = np.zeros(n, dtype=np.float64)
    pos = raw > t
    # backstop: never mark more than `cap` of the batch positive (bounds worst-case FPR)
    if pos.sum() > cap * n:
        order = np.argsort(-raw, kind="stable")
        pos = np.zeros(n, dtype=bool)
        pos[order[:int(cap * n)]] = True
    pidx = np.where(pos)[0]
    nidx = np.where(~pos)[0]
    if len(pidx) > 0:
        pr = raw[pidx]
        order = np.argsort(-pr, kind="stable")
        denom = max(1, len(pidx) - 1)
        ranks = np.empty(len(pidx)); ranks[order] = np.arange(len(pidx))
        for j, i in enumerate(pidx):
            out[i] = pf + (1.0 - ranks[j] / denom) * (pc - pf)
    if len(nidx) > 0:
        nv = raw[nidx]; mn, mx = float(nv.min()), float(nv.max()); span = max(mx - mn, 1e-9)
        for i in nidx:
            out[i] = max(0.0, min(nc, (float(raw[i]) - mn) / span * nc))
    return [float(v) for v in out]


class V8Model:
    """v6 stacked ensemble + static FPR-targeting threshold head. Drop-in bump model."""

    def __init__(self, artifact: Dict[str, Any], threshold: float,
                 cap: float = 0.30, metadata: Dict[str, Any] | None = None) -> None:
        models = list(artifact.get("models") or [])
        if not models:
            raise RuntimeError("artifact has no models")
        self.stacked = models[0]
        self.feature_names = list(artifact.get("feature_names") or [])
        self.threshold = float(threshold)            # score cutoff for positive
        self.cap = float(cap)
        self.metadata = dict(artifact.get("metadata") or {})
        if metadata:
            self.metadata.update(metadata)
        self.metadata["model_version"] = "v8-threshold-fpr"
        self.metadata["model_name"] = "poker44-bump-v8"
        self.metadata["framework"] = "stacked-trees+fpr-threshold"
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata["scoring_head"] = f"threshold_v1 (t={self.threshold:.5f}, cap={self.cap})"
        # miner banner reads .threshold (=0.5 boundary) — keep a separate decision cutoff
        self.head_mode = "threshold"
        self.subsample = False

    def _rows(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        rows = []
        for c in chunks:
            c = list(c or [])
            bf = _combined_feats(c) if c else {"hand_count": 0.0}
            bf["hand_count"] = float(len(c))
            rows.append([float(bf.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        rows = self._rows(chunks)
        # blended-isotonic calibration preserves ranking exactly (spearman 1.0)
        raw = self.stacked.predict_chunk_scores(chunks, rows, apply_calibration=True)
        return np.asarray(raw, dtype=np.float64)

    def predict_chunk_scores(self, chunks: Sequence[List[dict]]) -> List[float]:
        raw = self.predict_raw(chunks)
        t = float(os.getenv("POKER44_THRESHOLD", self.threshold))
        cap = float(os.getenv("POKER44_THRESHOLD_CAP", self.cap))
        return _threshold_squeeze(raw, t, cap)

    def score_chunk(self, chunk: List[dict]) -> float:
        return self.predict_chunk_scores([chunk])[0]
