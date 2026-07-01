"""v9 inference wrapper (live-generalization-first): a heavily-regularized single
LightGBM over ONLY cross-date-ROBUST base features (no cx_ collision feats), +
conservative topk head (frac 0.15).

Rationale from the live A/B: simpler v5 (293 base feats) BEAT complex v6 (+cx_)
live, and recall-boosting hurt live -> the benchmark→live gap is overfitting to
synthetic quirks. v9 leans all the way into robustness: drop cx_, keep only
features whose sign is consistent across benchmark dates, shallow/strongly-
regularized trees, conservative recall. Goal = lift LIVE AP (validated live, not
offline). Picklable, drop-in for the bump miner.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_ml.features import chunk_features as _base_cf   # 293 base feats, NO cx_
from poker44_bump.model_v5 import _topk_squeeze


class V9Model:
    def __init__(self, clf: Any, feature_names: Sequence[str],
                 topk_cfg: Dict[str, Any] | None = None,
                 metadata: Dict[str, Any] | None = None) -> None:
        self.clf = clf
        self.feature_names = list(feature_names)
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v9-lean-robust")
        self.metadata.setdefault("model_name", "poker44-bump-v9")
        self.metadata.setdefault("framework", "lgbm-regularized+robustfeats+topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = (
            f"topk_v1 (lean/robust, positive_fraction={self.topk_cfg.get('positive_fraction')})")
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
        proba = self.clf.predict_proba(self._rows(chunks))
        raw = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else np.asarray(proba)
        return np.asarray(raw, dtype=np.float64)

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
