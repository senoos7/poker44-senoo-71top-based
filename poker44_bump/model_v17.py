"""v17 = v10 averaged ensemble + UNSUPERVISED per-feature QUANTILE ALIGNMENT (live->benchmark).

The 2026-06-29 domain diagnostic proved benchmark<->live are marginally DISJOINT (bet scale,
stacks, AND action-rates; single-feature domain AUC=1.0). A benchmark-trained model's splits
are calibrated to benchmark value ranges that NEVER occur live -> garbage. v17 fixes the
marginals: at inference, each feature's live value is mapped to the benchmark value at the
SAME empirical percentile (histogram matching), using precomputed benchmark/live quantile
grids. Trees then see in-distribution inputs; their learned partition applies to live RANKS.
Monotone per-feature (no live labels used). The bet/size/behavioral shifts collapse at once;
whether the bot/human signal survives is decided LIVE. topk head kept (neutral under the
2026-06-26 rank-based reward). Picklable, drop-in.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_bump.features import chunk_features as _base_cf
from poker44_bump.model_v5 import _topk_squeeze


class V17AlignedModel:
    def __init__(self, estimators, feature_names, weights,
                 live_sorted, bench_sorted, topk_cfg=None, metadata=None):
        # live_sorted[j], bench_sorted[j] = sorted feature-value arrays (the empirical
        # quantile grids) for feature j, from captured live + benchmark data.
        self.estimators = list(estimators)
        self.feature_names = list(feature_names)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.estimators)
        self.live_sorted = [np.asarray(a, dtype=np.float64) for a in live_sorted]
        self.bench_sorted = [np.asarray(a, dtype=np.float64) for a in bench_sorted]
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v17-quantile-aligned")
        self.metadata.setdefault("model_name", "poker44-bump-v17")
        self.metadata.setdefault("framework", "v10-ensemble + live->benchmark quantile-align + topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = f"topk_v1 (aligned, positive_fraction={self.topk_cfg.get('positive_fraction')})"
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

    def _align(self, X: np.ndarray) -> np.ndarray:
        """map each live value -> benchmark value at the same empirical percentile."""
        Xa = np.empty_like(X)
        for j in range(X.shape[1]):
            ls = self.live_sorted[j]; bs = self.bench_sorted[j]
            if ls.size < 2 or bs.size < 2:
                Xa[:, j] = X[:, j]
                continue
            q = np.searchsorted(ls, X[:, j], side="right") / ls.size      # live empirical CDF in [0,1]
            idx = np.clip((q * (bs.size - 1)).astype(np.int64), 0, bs.size - 1)
            Xa[:, j] = bs[idx]                                            # benchmark quantile fn
        return Xa

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        X = self._align(self._rows(chunks))
        wsum = sum(self.weights) or 1.0
        acc = np.zeros(len(X), dtype=np.float64)
        for est, w in zip(self.estimators, self.weights):
            p = est.predict_proba(X)
            acc += float(w) * (p[:, 1] if getattr(p, "ndim", 1) == 2 else np.asarray(p))
        return acc / wsum

    def predict_chunk_scores(self, chunks):
        raw = self.predict_raw(chunks)
        frac = float(os.getenv("POKER44_TOPK_FRAC", self.topk_cfg.get("positive_fraction", 0.15)))
        return _topk_squeeze(raw, frac,
                             float(self.topk_cfg.get("positive_floor", 0.501)),
                             float(self.topk_cfg.get("positive_ceiling", 0.509)),
                             float(self.topk_cfg.get("negative_ceiling", 0.49)))

    def score_chunk(self, chunk):
        return self.predict_chunk_scores([chunk])[0]
