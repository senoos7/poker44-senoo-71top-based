"""v5 inference wrapper: leaders' stacked-tree pipeline + our topk safety head.

Loads a StackedEnsemble artifact produced by training/train_model_v2.py (the
proven uid-32/uid-136 pipeline: LightGBM+XGBoost+ExtraTrees+RandomForest base
learners -> logistic meta-learner -> isotonic calibration, over the vendored
poker44_ml `chunk_features`), gets the calibrated stacked ranking score per
chunk, then applies the batch-relative topk safety budget (same head as our v4
BumpModel) so chunk-level FPR stays under the 10% cliff while ranking (AP) is
preserved.

Picklable so it can be joblib-dumped as the served artifact. predict_chunk_scores
matches the BumpModel interface the miner expects.
"""
from __future__ import annotations
import math
import os
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_ml.features import chunk_features  # vendored leader features (~293)


def _topk_squeeze(raw: np.ndarray, frac: float,
                  pf: float = 0.501, pc: float = 0.509, nc: float = 0.49) -> List[float]:
    raw = np.asarray(raw, dtype=np.float64)
    n = len(raw)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return []
    k = max(0, min(n, int(math.floor(n * frac))))
    order = np.argsort(-raw, kind="stable")
    pos, neg = order[:k], order[k:]
    if k > 0:
        denom = max(1, k - 1)
        for rank, i in enumerate(pos):
            out[i] = pf + (1.0 - rank / denom) * (pc - pf)
    if len(neg) > 0:
        nv = raw[neg]; mn, mx = float(nv.min()), float(nv.max()); span = max(mx - mn, 1e-9)
        for i in neg:
            out[i] = max(0.0, min(nc, (float(raw[i]) - mn) / span * nc))
    return [float(v) for v in out]


class StackedTopkModel:
    """Leaders' StackedEnsemble + our topk head. Drop-in for the bump miner."""

    def __init__(self, artifact: Dict[str, Any], topk_cfg: Dict[str, Any] | None = None) -> None:
        models = list(artifact.get("models") or [])
        if not models:
            raise RuntimeError("artifact has no models")
        self.stacked = models[0]                      # StackedEnsemble
        self.feature_names = list(artifact.get("feature_names") or [])
        self.metadata = dict(artifact.get("metadata") or {})
        self.metadata.setdefault("model_version", "v5-stacked-topk")
        self.metadata.setdefault("model_name", "poker44-bump-stacked")
        self.metadata.setdefault("framework", "stacked-trees+topk")
        self.metadata.setdefault("ensemble_combiner", "stack(lgbm,xgb,extratrees,rf)+logreg-meta")
        self.metadata.setdefault("conformal_threshold", 0.5)  # miner manifest formats this
        self.topk_cfg = dict(topk_cfg or self.metadata.get("topk_cfg")
                             or {"positive_fraction": 0.15})
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = (
            f"topk_v1 (full stacked pipeline, positive_fraction={self.topk_cfg.get('positive_fraction')})")
        # miner startup log reads .threshold / .feature_names
        self.threshold = 0.5
        self.head_mode = "topk"
        self.subsample = False

    def _rows(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        rows = []
        for c in chunks:
            c = list(c or [])
            feats = chunk_features(c) if c else {"hand_count": 0.0}
            feats["hand_count"] = float(len(c))
            rows.append([float(feats.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        rows = self._rows(chunks)
        # StackedEnsemble.predict_chunk_scores(chunks, feature_rows) -> calibrated stacked score
        raw = self.stacked.predict_chunk_scores(chunks, rows)
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
