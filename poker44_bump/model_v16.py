"""v16 = DECORRELATED two-architecture blend (our best per the 2026-06-27 research):
  (1-w) * tree-ensemble[base + repetition-invariant rp_ feats]   (v14)
  +  w   * Deep-Sets+Relation set-encoder (torch)                 (v15)
Trees and deepsets are only Pearson ~0.32 correlated, so blending lifts LODO per-date AP
above either alone (0.911 -> ~0.919). Folds in all three research directions: rp_ features
(generator-agnostic repetition), Deep-Sets/Relation architecture, size-invariance +
hand-subsample augmentation. Conservative topk head (FPR-safe). Picklable, drop-in.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

try:
    import torch
except Exception:
    torch = None

from poker44_bump.features import chunk_features as _base_cf
from poker44_bump.features_repeat import _repeat_feats
from poker44_bump.model_v15 import DeepSetsRelation, featurize_chunk, MAX_ACTIONS
from poker44_bump.model_v5 import _topk_squeeze


class V16Model:
    def __init__(self, estimators, feature_names, ds_state, ds_cfg, blend_w,
                 weights=None, topk_cfg=None, metadata=None):
        self.estimators = list(estimators)
        self.feature_names = list(feature_names)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.estimators)
        self.ds_state = ds_state
        self.ds_cfg = dict(ds_cfg)
        self.blend_w = float(blend_w)
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v16-trees-deepsets-blend")
        self.metadata.setdefault("model_name", "poker44-bump-v16")
        self.metadata.setdefault("framework", "blend((base+rp)trees, deepsets-relation)+topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = f"topk_v1 (v16-blend w={self.blend_w}, frac={self.topk_cfg.get('positive_fraction')})"
        self.threshold = 0.5
        self.head_mode = "topk"
        self.subsample = False
        self._net = None

    def _rows(self, chunks):
        rows = []
        for c in chunks:
            c = list(c or [])
            d = _base_cf(c) if c else {"hand_count": 0.0}
            if c: d.update(_repeat_feats(c))
            d["hand_count"] = float(len(c))
            rows.append([float(d.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)

    def _net_eval(self):
        if self._net is None:
            net = DeepSetsRelation(**self.ds_cfg); net.load_state_dict(self.ds_state); net.eval()
            self._net = net
        return self._net

    def _ds_probs(self, chunks):
        net = self._net_eval()
        feats = [featurize_chunk(c) for c in chunks]
        Hmax = max(f[0].shape[0] for f in feats); B = len(feats)
        acts = np.zeros((B, Hmax, MAX_ACTIONS, 4), np.int64); amt = np.zeros((B, Hmax, MAX_ACTIONS), np.float32)
        amask = np.zeros((B, Hmax, MAX_ACTIONS), np.float32); hmask = np.zeros((B, Hmax), np.float32)
        for i, (a, m, am, hm) in enumerate(feats):
            h = a.shape[0]; acts[i, :h] = a; amt[i, :h] = m; amask[i, :h] = am; hmask[i, :h] = hm
        with torch.no_grad():
            p = torch.sigmoid(self._net_eval()(torch.from_numpy(acts), torch.from_numpy(amt),
                              torch.from_numpy(amask), torch.from_numpy(hmask))).numpy().astype(np.float64)
        return p

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        X = self._rows(chunks)
        wsum = sum(self.weights) or 1.0
        tree = np.zeros(len(X))
        for est, w in zip(self.estimators, self.weights):
            p = est.predict_proba(X)
            tree += float(w) * (p[:, 1] if getattr(p, "ndim", 1) == 2 else np.asarray(p))
        tree /= wsum
        ds = self._ds_probs(chunks)
        return (1.0 - self.blend_w) * tree + self.blend_w * ds

    def predict_chunk_scores(self, chunks):
        raw = self.predict_raw(chunks)
        frac = float(os.getenv("POKER44_TOPK_FRAC", self.topk_cfg.get("positive_fraction", 0.15)))
        return _topk_squeeze(raw, frac,
                             float(self.topk_cfg.get("positive_floor", 0.501)),
                             float(self.topk_cfg.get("positive_ceiling", 0.509)),
                             float(self.topk_cfg.get("negative_ceiling", 0.49)))

    def score_chunk(self, chunk):
        return self.predict_chunk_scores([chunk])[0]
