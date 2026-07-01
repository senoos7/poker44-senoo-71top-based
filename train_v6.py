"""v6 = leaders' stacked pipeline trained on OUR richer feature set
(base 293 + 11 cx_ collision features). Patches build_dataset's feature
extractor to append our cx_ features, then runs the proven train_model_v2
pipeline. Goal: beat v5's honest holdout AP (0.919) via better ranking.

Run:  PYTHONPATH=. .venv/bin/python train_v6.py
"""
import os, sys
sys.path.insert(0, ".")

import training.build_dataset as bd
from poker44_ml.features import chunk_features as _base_cf
from poker44_bump.features_ext import _extra_feats


def _combined(payload):
    f = dict(_base_cf(payload))
    try:
        f.update(_extra_feats(payload))   # +11 cx_ collision features
    except Exception:
        pass
    return f


# patch the module global the trainer uses (build_dataset._feature_row -> chunk_features)
bd.chunk_features = _combined

os.environ.setdefault("POKER44_MODEL_VERSION", "v6-stacked-ext")
sys.argv = [
    "train_v6",
    "--benchmark-path", "data",
    "--disable-catboost",
    "--n-folds", "5",
    "--holdout-latest-days", "2",
    "--output", "models/v6_stacked.joblib",
]
from training.train_model_v2 import main
main()
