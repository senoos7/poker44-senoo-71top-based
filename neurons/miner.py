"""Poker44 skip miner: bump tree ensemble + cliff-robust topk head.

Drop-in miner for Poker44 subnet 126. Requires the Poker44-subnet package
importable (pip install -e . in the subnet repo) for BaseMinerNeuron and
DetectionSynapse. Loads the local BumpModel artifact and returns one risk
score per chunk, calibrated so chunk-level FPR stays under the validator's
10% cliff.

Manifest is built from hardcoded / env-var values rather than live git
introspection — each miner instance declares its own repo_url + repo_commit
via POKER44_MANIFEST_REPO_URL and POKER44_MANIFEST_REPO_COMMIT, so the
reported identity is stable and predictable regardless of the local git
checkout state.

Run:
  POKER44_BUMP_MODEL=$(pwd)/models/bump_model_v19.joblib \\
  POKER44_MANIFEST_REPO_URL=https://github.com/yourname/yourrepo \\
  POKER44_MANIFEST_REPO_COMMIT=abc1234 \\
  python neurons/miner.py --netuid 126 --wallet.name <c> --wallet.hotkey <h> \\
    --subtensor.network finney --axon.port 8091 \\
    --blacklist.force_validator_permit --logging.info
"""
# NOTE: do NOT add `from __future__ import annotations` here — bittensor's
# axon.attach() introspects forward()'s annotations at runtime with issubclass(),
# which fails if they are stringized. Keep annotations as real class objects.
import hashlib
import logging as stdlogging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bittensor as bt
import joblib

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker44_bump.model import BumpModel  # noqa: E402
try:
    from poker44_bump.live_capture import capture as _live_capture
except Exception:
    def _live_capture(*a, **k):
        return None

# ---------------------------------------------------------------------------
# Hardcoded manifest defaults — override via env vars in run_miners.sh
# These values are what gets broadcast to the validator; they must match a
# real public repo+commit that contains this miner's implementation.
# ---------------------------------------------------------------------------
_MANIFEST_REPO_URL_DEFAULT = "https://github.com/yourname/poker44-skip"
_MANIFEST_REPO_COMMIT_DEFAULT = "0000000"

_MANIFEST_MODEL_NAME_DEFAULT = "poker44-bump-robust"
_MANIFEST_MODEL_VERSION_DEFAULT = "bump-topk-v1"
_MANIFEST_FRAMEWORK_DEFAULT = "tree-ensemble+topk-safety-budget"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class _ScannerNoiseFilter(stdlogging.Filter):
    """Suppress common public-port probe errors emitted before miner routing."""

    _NOISY_SNIPPETS = ("UnknownSynapseError", "InvalidRequestNameError")
    _NOISY_REQUEST_NAMES = (
        "Synapse name ''",
        "Synapse name 'api'",
        "Synapse name 'mcp'",
        "Synapse name 'jsonrpc'",
        "Synapse name 'robots.txt'",
        "Could not parser request .",
    )

    def filter(self, record: stdlogging.LogRecord) -> bool:
        message = record.getMessage()
        if not any(s in message for s in self._NOISY_SNIPPETS):
            return True
        return not any(n in message for n in self._NOISY_REQUEST_NAMES)


def _install_scanner_log_filter() -> None:
    enabled = os.getenv("POKER44_SUPPRESS_SCANNER_ERRORS", "1").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    f = _ScannerNoiseFilter()
    for handler in getattr(bt.logging, "_handlers", []):
        handler.addFilter(f)
    for name in ("bittensor", "uvicorn.access", "uvicorn.error"):
        for handler in stdlogging.getLogger(name).handlers:
            handler.addFilter(f)


def _build_manifest(model_path: Path, model: BumpModel) -> Dict[str, Any]:
    """Build the model manifest from env-var overrides and model metadata.

    Repo URL and commit are taken from env vars (set per-miner in run_miners.sh)
    so they are stable, predictable, and don't drift with the local git state.
    All other fields are derived from the loaded model's metadata where
    available, falling back to env vars or compiled-in defaults.
    """
    md = model.metadata

    repo_url = (
        os.getenv("POKER44_MANIFEST_REPO_URL", "").strip()
        or md.get("repo_url", "")
        or _MANIFEST_REPO_URL_DEFAULT
    )
    repo_commit = (
        os.getenv("POKER44_MANIFEST_REPO_COMMIT", "").strip()
        or md.get("repo_commit", "")
        or _MANIFEST_REPO_COMMIT_DEFAULT
    )
    model_name = (
        os.getenv("POKER44_MANIFEST_MODEL_NAME", "").strip()
        or md.get("model_name", "")
        or _MANIFEST_MODEL_NAME_DEFAULT
    )
    model_version = (
        os.getenv("POKER44_MANIFEST_MODEL_VERSION", "").strip()
        or md.get("model_version", "")
        or _MANIFEST_MODEL_VERSION_DEFAULT
    )
    framework = (
        os.getenv("POKER44_MANIFEST_FRAMEWORK", "").strip()
        or md.get("framework", "")
        or _MANIFEST_FRAMEWORK_DEFAULT
    )
    training_data_statement = md.get("training_data_statement") or (
        f"Trained on {md.get('benchmark_rows', '?')} released benchmark chunks "
        f"(groundTruth labels). Cliff-robust topk head "
        f"(positive_fraction={model.topk_cfg.get('positive_fraction', 0.15):.2f})."
    )
    artifact_sha = _sha256(model_path) if model_path.is_file() else ""
    impl_sha = _sha256(Path(__file__).resolve())

    return {
        "schema_version": "1",
        "open_source": True,
        "repo_url": repo_url,
        "repo_commit": repo_commit,
        "model_name": model_name,
        "model_version": model_version,
        "framework": framework,
        "license": "MIT",
        "inference_mode": "local-joblib",
        "training_data_statement": training_data_statement,
        "training_data_sources": md.get("training_data_sources") or ["released_training_benchmark"],
        "private_data_attestation": (
            md.get("data_attestation")
            or "No validator-private data used; released benchmark labels only."
        ),
        "data_attestation": (
            md.get("data_attestation")
            or "No validator-private data used; released benchmark labels only."
        ),
        "implementation_files": [
            "neurons/miner.py",
            "poker44_bump/model.py",
            "poker44_bump/features.py",
            "poker44_bump/features_ext.py",
            "poker44_bump/payload_view.py",
        ],
        "implementation_sha256": impl_sha,
        "artifact_sha256": artifact_sha,
        "notes": (
            f"topk head, oof_ap={md.get('oof_ap')}, "
            f"combiner={md.get('ensemble_combiner')}, "
            f"features={len(getattr(model, 'feature_names', []) or [])}"
        ),
    }


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        _install_scanner_log_filter()

        root = Path(__file__).resolve().parents[1]
        self.model_path = Path(
            os.getenv("POKER44_BUMP_MODEL", str(root / "models" / "bump_model.joblib"))
        )
        self.model: BumpModel = joblib.load(self.model_path)
        self.model_manifest = _build_manifest(self.model_path, self.model)

        md = self.model.metadata
        bt.logging.info(
            f"Poker44 skip miner loaded | "
            f"head={self.model.head_mode} "
            f"T={self.model.threshold:.4f} "
            f"feats={len(getattr(self.model, 'feature_names', []) or [])} "
            f"oof_ap={md.get('oof_ap')} "
            f"path={self.model_path}"
        )
        bt.logging.info(
            f"Manifest | "
            f"model={self.model_manifest['model_name']} "
            f"version={self.model_manifest['model_version']} "
            f"repo={self.model_manifest['repo_url']} "
            f"commit={self.model_manifest['repo_commit']}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks: List[List[dict]] = [list(c or []) for c in (synapse.chunks or [])]
        caller = str(getattr(getattr(synapse, "dendrite", None), "hotkey", "") or "")
        t0 = time.perf_counter()

        try:
            scores = self.model.predict_chunk_scores(chunks)
        except Exception as err:
            bt.logging.warning(f"scoring failed ({err}); returning 0.5 for all chunks")
            scores = [0.5] * len(chunks)

        scores = [max(0.0, min(1.0, float(s))) for s in scores]

        try:
            _live_capture(chunks, scores, str(getattr(self.config.wallet, "hotkey", "")), caller)
        except Exception:
            pass

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        dt = (time.perf_counter() - t0) * 1000.0
        bt.logging.info(
            f"scored {len(chunks)} chunks | {dt:.1f}ms ({dt/max(len(chunks),1):.2f}ms/chunk) "
            f"bots={sum(synapse.predictions)} "
            f"range=[{min(scores) if scores else 0:.3f},{max(scores) if scores else 0:.3f}]"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 skip miner running...")
        while True:
            bt.logging.info(f"UID {miner.uid} | Incentive {miner.metagraph.I[miner.uid]}")
            time.sleep(300)
