"""v7 inference wrapper: v6's combined features + class-conditional n-gram LLR
channel + topk safety head, over a date-robustly trained classifier.

Feature vector per chunk = base combined feats (poker44_ml chunk_features ~293 +
our cx_ collision feats +11 = 304)  ++  n-gram Markov LLR aggregates (9). The
embedded NgramLLRScorer is fit on all training data; at inference it scores each
hand by class-conditional log-likelihood-ratio and aggregates over the chunk.

The classifier is any object exposing predict_proba (LightGBM, or a StackedEnsemble
adapter). Raw bot-probability is shaped by the same topk batch-safety head as
v5/v6 so chunk-level FPR stays under the 0.10 cliff while ranking (AP) is kept.

Picklable -> joblib-dumpable as the served artifact. Drop-in for the bump miner.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_ml.features import chunk_features as _base_cf
from poker44_bump.features_ext import _extra_feats
from poker44_bump.ngram_llr import NgramLLRScorer
from poker44_bump.model_v5 import _topk_squeeze


def _combined_feats(chunk: List[dict]) -> Dict[str, float]:
    f = dict(_base_cf(chunk))
    try:
        f.update(_extra_feats(chunk))
    except Exception:
        pass
    return f


class V7Model:
    """date-robust classifier + nLLR channel + topk head. Drop-in bump model."""

    def __init__(self, clf: Any, base_names: Sequence[str], llr_scorer: NgramLLRScorer,
                 topk_cfg: Dict[str, Any] | None = None,
                 metadata: Dict[str, Any] | None = None) -> None:
        self.clf = clf
        self.base_names = list(base_names)
        self.llr_names = NgramLLRScorer.feature_names()
        self.llr = llr_scorer
        self.feature_names = self.base_names + self.llr_names
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v7-daterobust-nllr")
        self.metadata.setdefault("model_name", "poker44-bump-v7")
        self.metadata.setdefault("framework", "lgbm+nllr+topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = (
            f"topk_v1 (daterobust+nLLR, positive_fraction={self.topk_cfg.get('positive_fraction')})")
        # miner startup banner reads these
        self.threshold = 0.5
        self.head_mode = "topk"
        self.subsample = False

    def _rows(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        rows = []
        for c in chunks:
            c = list(c or [])
            bf = _combined_feats(c) if c else {"hand_count": 0.0}
            bf["hand_count"] = float(len(c))
            lf = self.llr.transform_one(c) if c else {}
            base = [float(bf.get(n, 0.0)) for n in self.base_names]
            llr = [float(lf.get(n, 0.0)) for n in self.llr_names]
            rows.append(base + llr)
        return np.asarray(rows, dtype=np.float64)

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        X = self._rows(chunks)
        proba = self.clf.predict_proba(X)
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
