"""Poker44 'bump' model: proven signature-collision features + averaged tree
ensemble + a CLIFF-ROBUST conformal head.

The conformal head maps the raw ensemble bot-probability `p` to a final risk
score that crosses 0.5 exactly at a calibrated threshold `T`, where `T` sits
just above the worst recent human score plus a drift buffer. This keeps
chunk-level FPR below the validator's 10% cliff under date-to-date drift while
preserving ranking (AP is invariant to this monotone map).

Inference convention matches the reference miner: `chunk_features` is applied
DIRECTLY to the incoming (already validator-sanitized) chunk.
"""
from __future__ import annotations
import hashlib
import math
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_bump.features import chunk_features
from poker44_bump.features_ext import chunk_features_ext


def _subsample_indices(n: int, k: int, salt: str) -> List[int]:
    """Deterministic size-k subsample of range(n) seeded by salt (reproducible)."""
    if n <= k:
        return list(range(n))
    order = sorted(range(n), key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).digest())
    return sorted(order[:k])


def conformal_map(p: float | np.ndarray, T: float, lo: float = 0.02, hi: float = 0.98):
    """Monotone map: p==T -> 0.5; p<T -> [lo,0.5); p>T -> (0.5,hi]. Preserves AP."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    T = float(min(max(T, 1e-4), 1.0 - 1e-4))
    below = 0.5 * (p / T)
    above = 0.5 + 0.5 * (p - T) / (1.0 - T)
    out = np.where(p >= T, above, below)
    return np.clip(out, lo, hi)


class BumpModel:
    """Picklable inference object. Stored fields are plain ensemble + conformal head."""

    def __init__(
        self,
        base_models: Sequence[Any],
        feature_names: Sequence[str],
        threshold: float,
        metadata: Dict[str, Any] | None = None,
        lo: float = 0.02,
        hi: float = 0.98,
        chunk_models: Sequence[Any] | None = None,
        meta_model: Any | None = None,
    ) -> None:
        self.base_models = list(base_models)
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.lo = float(lo)
        self.hi = float(hi)
        self.metadata = dict(metadata or {})
        # v5: optional chunk-level learner(s) (ChunkSetTransformer) blended with the
        # tree ensemble. meta_model (logistic) stacks [tree_mean_prob, chunk_prob];
        # if absent, fall back to a simple mean. Both empty => v4 tree-only behavior.
        self.chunk_models = list(chunk_models or [])
        self.meta_model = meta_model
        # size-invariance: subsample live chunks to the training size, bag over draws
        self.train_chunk_size = int(self.metadata.get("train_chunk_size", 0)) or 0
        self.bag = int(self.metadata.get("bag", 5))
        # subsampling OFF by default for full-chunk scoring (matches live leaders);
        # the topk head is batch-relative so absolute size calibration matters less.
        self.subsample = bool(self.metadata.get("subsample", False))
        # scoring head: "conformal" (fixed-T map) or "topk" (batch-relative safety budget)
        self.head_mode = str(self.metadata.get("head_mode", "conformal"))
        self.topk_cfg = dict(self.metadata.get("topk_cfg", {}) or {})

    # ---- feature path ----
    def _extract(self, chunk: List[dict]) -> Dict[str, float]:
        extractor = chunk_features_ext if self.metadata.get("feature_set") == "ext" else chunk_features
        return extractor(chunk) if chunk else {"hand_count": 0.0}

    def _vec(self, feats: Dict[str, float]) -> np.ndarray:
        return np.array([float(feats.get(n, 0.0)) for n in self.feature_names], dtype=np.float64)

    def _rows_for_chunk(self, chunk: List[dict]) -> np.ndarray:
        """One or more aligned feature rows. If the chunk is larger than the
        training size AND subsampling is enabled, return `bag` size-matched
        subsample rows (size-invariance). Otherwise score the FULL chunk
        (matches what the live leaders do — more hands = stronger collision
        signal)."""
        ts = self.train_chunk_size
        if getattr(self, "subsample", True) and ts and len(chunk) > ts:
            rows = []
            for b in range(max(1, self.bag)):
                idx = _subsample_indices(len(chunk), ts, salt=f"{len(chunk)}:{b}")
                rows.append(self._vec(self._extract([chunk[i] for i in idx])))
            return np.vstack(rows)
        return self._vec(self._extract(chunk))[None, :]

    def _base_raw(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for m in self.base_models:
            proba = np.asarray(m.predict_proba(X))
            cols.append(proba[:, 1] if proba.ndim == 2 else proba)
        return np.mean(np.vstack(cols), axis=0)

    def feature_matrix(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        # diagnostics only: single row per chunk (no bagging)
        if not chunks:
            return np.zeros((0, len(self.feature_names)), dtype=np.float64)
        return np.vstack([self._vec(self._extract(list(c or []))) for c in chunks])

    # ---- scoring ----
    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        clean = [list(c or []) for c in chunks]
        tree = np.empty(len(clean), dtype=np.float64)
        for i, c in enumerate(clean):
            rows = self._rows_for_chunk(c)
            tree[i] = float(np.mean(self._base_raw(rows)))  # bag-average over subsamples
        cms = getattr(self, "chunk_models", None) or []
        if not cms:
            return tree
        # v5: blend tree ensemble with chunk-level learner(s) (ChunkSetTransformer)
        cols = []
        for cm in cms:
            proba = np.asarray(cm.predict_proba(clean))
            cols.append(proba[:, 1] if proba.ndim == 2 else proba)
        st = np.mean(np.vstack(cols), axis=0)
        meta = getattr(self, "meta_model", None)
        if meta is not None:
            Z = np.column_stack([tree, st])
            p = np.asarray(meta.predict_proba(Z))
            return (p[:, 1] if p.ndim == 2 else p).astype(np.float64)
        w = float(self.metadata.get("blend_tree_weight", 0.5))
        return w * tree + (1.0 - w) * st

    def _topk_squeeze(self, raw: np.ndarray) -> List[float]:
        """Batch-relative safety budget (port of the live leader's topk_v1).

        Forces only the top `positive_fraction` of this batch above 0.5, into a
        razor band [floor, ceiling]; pushes the rest into [0, neg_ceiling].
        Ranking (hence AP) is preserved, FPR is driven toward 0 with a large
        margin, and recall is tuned via the fraction. Applied per query batch
        (the validator sends one window of chunks per call)."""
        cfg = getattr(self, "topk_cfg", {}) or {}
        frac = float(cfg.get("positive_fraction", 0.30))
        import os
        frac = float(os.getenv("POKER44_TOPK_FRAC", frac))   # per-process override for live A/B across keys
        pf = float(cfg.get("positive_floor", 0.501))
        pc = float(cfg.get("positive_ceiling", 0.509))
        nc = float(cfg.get("negative_ceiling", 0.49))
        raw = np.asarray(raw, dtype=np.float64)
        n = len(raw)
        out = np.zeros(n, dtype=np.float64)
        if n == 0:
            return []
        k = max(0, min(n, int(math.floor(n * frac))))
        order = np.argsort(-raw, kind="stable")           # high->low
        pos_idx, neg_idx = order[:k], order[k:]
        if k > 0:
            denom = max(1, k - 1)
            for rank, i in enumerate(pos_idx):
                rel = 1.0 - (rank / denom)
                out[i] = pf + rel * (pc - pf)
        if len(neg_idx) > 0:
            nv = raw[neg_idx]
            mn, mx = float(nv.min()), float(nv.max())
            span = max(mx - mn, 1e-9)
            for i in neg_idx:
                rel = (float(raw[i]) - mn) / span
                out[i] = max(0.0, min(nc, rel * nc))
        return [float(v) for v in out]

    def predict_chunk_scores(self, chunks: Sequence[List[dict]]) -> List[float]:
        raw = self.predict_raw(chunks)
        if getattr(self, "head_mode", "conformal") == "topk":
            return self._topk_squeeze(raw)
        final = conformal_map(raw, self.threshold, self.lo, self.hi)
        return [float(v) for v in final]

    def score_chunk(self, chunk: List[dict]) -> float:
        return self.predict_chunk_scores([chunk])[0]

    # diagnostics hook used by the reference miner (optional)
    def debug_score_components(self, chunks: Sequence[List[dict]]) -> Dict[str, List[float]]:
        raw = self.predict_raw(chunks)
        return {"raw_bot_prob": [float(v) for v in raw]}
