"""v14 inference wrapper: v10's diverse AVERAGED ENSEMBLE over base feats + REPETITION-
INVARIANT features (rp_*, generator-agnostic self-similarity) + conservative topk head.

Rationale (2026-06-27 research, 3 streams converged): the live gap is partly our cx_
collision features overfitting the synthetic benchmark. rp_* features (Vendi, set log-det,
compression ratio, entropy-rate, exact-dup fractions) measure intra-set repetition as a
GENERATOR-AGNOSTIC invariant -> should transfer live where cx_ does not. NO cx_ (known
live-loser: v6<v5). Full-chunk scoring (more hands -> stronger repetition signal). Averaging
reduces variance. Picklable, drop-in for the bump miner.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_bump.features import chunk_features as _base_cf
from poker44_bump.features_repeat import _repeat_feats
from poker44_bump.model_v5 import _topk_squeeze


class V14Model:
    def __init__(self, estimators: List[Any], feature_names: Sequence[str],
                 weights: Sequence[float] | None = None,
                 topk_cfg: Dict[str, Any] | None = None,
                 metadata: Dict[str, Any] | None = None) -> None:
        self.estimators = list(estimators)
        self.feature_names = list(feature_names)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.estimators)
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v14-repeat-ensemble")
        self.metadata.setdefault("model_name", "poker44-bump-v14")
        self.metadata.setdefault("framework", "avg-ensemble(lgbm,et,rf)+repeat-feats+topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = (
            f"topk_v1 (repeat-ensemble, positive_fraction={self.topk_cfg.get('positive_fraction')})")
        self.threshold = 0.5
        self.head_mode = "topk"
        self.subsample = False

    def _feat_dict(self, chunk: List[dict]) -> Dict[str, float]:
        c = list(chunk or [])
        d = _base_cf(c) if c else {"hand_count": 0.0}
        if c:
            d.update(_repeat_feats(c))
        d["hand_count"] = float(len(c))
        return d

    def _rows(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        rows = []
        for c in chunks:
            d = self._feat_dict(c)
            rows.append([float(d.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        X = self._rows(chunks)
        wsum = sum(self.weights) or 1.0
        acc = np.zeros(len(X), dtype=np.float64)
        for est, w in zip(self.estimators, self.weights):
            p = est.predict_proba(X)
            acc += float(w) * (p[:, 1] if getattr(p, "ndim", 1) == 2 else np.asarray(p))
        return acc / wsum

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
