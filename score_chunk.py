#!/usr/bin/env python3
"""CLI: score one or more chunks with the bump model.

Usage:
  python score_chunk.py path/to/chunk.json        # one chunk (list of hands) -> one score
  python score_chunk.py path/to/synapse.json      # {"chunks":[[...],[...]]} -> list of scores
The incoming hands are scored as-is (validator already sanitizes live payloads).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from poker44_bump.model import BumpModel  # noqa: E402

MODEL_PATH = HERE / "models" / "bump_model.joblib"


def _extract_chunks(payload):
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return [payload]        # single chunk = list of hand dicts
    if isinstance(payload, list):
        return payload          # already list of chunks
    raise ValueError("expected a chunk (list of hands) or {'chunks': [...]}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__); return 2
    model: BumpModel = joblib.load(MODEL_PATH)
    payload = json.loads(Path(sys.argv[1]).read_text())
    chunks = _extract_chunks(payload)
    t0 = time.perf_counter()
    scores = model.predict_chunk_scores(chunks)
    dt = (time.perf_counter() - t0) * 1000
    print(json.dumps({
        "threshold": model.threshold,
        "n_chunks": len(chunks),
        "risk_scores": [round(s, 6) for s in scores],
        "predictions": [s >= 0.5 for s in scores],
        "latency_ms_total": round(dt, 1),
        "latency_ms_per_chunk": round(dt / max(len(chunks), 1), 2),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
