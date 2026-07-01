"""v20 inference wrapper = v19 sanitization-invariant base feats (hero/raw-bb
dropped) PLUS bucket-snapped amount features (poker44_bump.features_bucket). Same
avg-ensemble + topk head as v10/v19. Picklable, drop-in. Feature vector built from
base chunk_features merged with bucket_amount_feats, selected by feature_names.
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np

from poker44_ml.features import chunk_features as _base_cf
from poker44_bump.features_bucket import bucket_amount_feats
from poker44_bump.model_v10 import V10Model


class V20Model(V10Model):
    def _rows(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        rows = []
        for c in chunks:
            c = list(c or [])
            bf = _base_cf(c) if c else {"hand_count": 0.0}
            bf["hand_count"] = float(len(c))
            bf.update(bucket_amount_feats(c))
            rows.append([float(bf.get(n, 0.0)) for n in self.feature_names])
        return np.asarray(rows, dtype=np.float64)
