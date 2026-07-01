"""v11 inference wrapper: a weighted META-ENSEMBLE synthesizing every recipe that
worked, for best live transfer.

Members (all over 293 no-cx_ base feats; canonical order = v5 stacked's feature_names):
  - v5 StackedEnsemble (our best single live model, 0.558)         weight 2
  - 4 diverse LightGBMs (seed/reg/feature-fraction variants)       weight 1 each
  - ExtraTrees + RandomForest                                       weight 1 each
  - 1 robust-feature LightGBM (only cross-date-robust feats, v9 idea) weight 1
Combine = weighted average of per-chunk probabilities -> conservative topk (0.15).

Rationale: cx_ overfits live, recall-boost hurts live, simpler/averaged generalizes
better live (live A/B). v11 maximizes diversity + includes our proven-best member +
the robust-feature direction -> the strongest variance reduction we can build on the
shared public data. Picklable, drop-in for the bump miner.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_ml.features import chunk_features as _base_cf   # 293 base feats, NO cx_
from poker44_bump.model_v5 import _topk_squeeze


class V11Model:
    def __init__(self, members: List[Dict[str, Any]], feature_names: Sequence[str],
                 topk_cfg: Dict[str, Any] | None = None,
                 metadata: Dict[str, Any] | None = None) -> None:
        # members: [{'kind':'stacked'|'sklearn', 'est':obj, 'cols':list|None, 'w':float}]
        self.members = members
        self.feature_names = list(feature_names)
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v11-meta-ensemble")
        self.metadata.setdefault("model_name", "poker44-bump-v11")
        self.metadata.setdefault("framework", "meta-ensemble(stack+lgbm+et+rf+robust)+topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = (
            f"topk_v1 (meta-ensemble, positive_fraction={self.topk_cfg.get('positive_fraction')})")
        self.threshold = 0.5
        self.head_mode = "topk"
        self.subsample = False

    def _rows(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        rows = []
        for c in chunks:
            c = list(c or [])
            bf = _base_cf(c) if c else {"hand_count": 0.0}
            bf["hand_count"] = float(len(c))
            rows.append([float(bf.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        rows = self._rows(chunks)
        acc = np.zeros(len(rows), dtype=np.float64); wsum = 0.0
        for m in self.members:
            w = float(m.get("w", 1.0))
            if m["kind"] == "stacked":
                s = np.asarray(m["est"].predict_chunk_scores(chunks, rows), dtype=np.float64)
            else:
                cols = m.get("cols")
                X = rows[:, cols] if cols is not None else rows
                p = m["est"].predict_proba(X)
                s = p[:, 1] if getattr(p, "ndim", 1) == 2 else np.asarray(p)
            acc += w * np.asarray(s, dtype=np.float64); wsum += w
        return acc / (wsum or 1.0)

    def predict_chunk_scores(self, chunks: Sequence[List[dict]]) -> List[float]:
        raw = self.predict_raw(chunks)
        frac = float(os.getenv("POKER44_TOPK_FRAC", self.topk_cfg.get("positive_fraction", 0.15)))
        return _topk_squeeze(
            raw, frac,
            float(self.topk_cfg.get("positive_floor", 0.501)),
            float(self.topk_cfg.get("positive_ceiling", 0.509)),
            float(self.topk_cfg.get("negative_ceiling", 0.49)),
        )

    def score_chunk(self, chunk: List[dict]) -> float:
        return self.predict_chunk_scores([chunk])[0]
