"""v6 inference wrapper: v5's stacked-tree pipeline + our cx_ collision features.

Identical to v5's StackedTopkModel except the per-chunk feature row is the
COMBINED set (vendored leader `chunk_features` ~293  +  our screened cx_
collision/n-gram features, +11 = 304). The v6 StackedEnsemble artifact
(models/v6_stacked.joblib) was trained on exactly this 304-feature set and
scored holdout AP 0.9425 vs v5's 0.919, so the cx_ features must be supplied
at inference in the same order (artifact feature_names drives the column order).

The topk safety head is unchanged from v5. Picklable -> joblib-dumpable as the
served artifact.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

from poker44_ml.features import chunk_features as _base_cf   # vendored leader feats (~293)
from poker44_bump.features_ext import _extra_feats           # our cx_ collision feats (+11)
from poker44_bump.model_v5 import StackedTopkModel, _topk_squeeze  # reuse head + base class


def _combined_feats(chunk: List[dict]) -> Dict[str, float]:
    f = dict(_base_cf(chunk))
    try:
        f.update(_extra_feats(chunk))
    except Exception:
        pass
    return f


class StackedTopkModelV6(StackedTopkModel):
    """v5 stacked pipeline + cx_ collision features + topk head."""

    def __init__(self, artifact: Dict[str, Any], topk_cfg: Dict[str, Any] | None = None) -> None:
        super().__init__(artifact, topk_cfg)
        self.metadata["model_version"] = self.metadata.get("model_version", "v6-stacked-topk")
        self.metadata["model_name"] = "poker44-bump-stacked-v6"
        self.metadata["framework"] = "stacked-trees+cx+topk"
        self.metadata["scoring_head"] = (
            f"topk_v1 (stacked+cx pipeline, positive_fraction={self.topk_cfg.get('positive_fraction')})")

    def _rows(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        rows = []
        for c in chunks:
            c = list(c or [])
            feats = _combined_feats(c) if c else {"hand_count": 0.0}
            feats["hand_count"] = float(len(c))
            rows.append([float(feats.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)
