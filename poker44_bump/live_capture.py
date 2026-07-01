"""Live validator-query capture (operational, local-only, gitignored output).

Persists the UNLABELED chunks validators send us at inference = the real live
distribution, for unsupervised domain-adaptation research (benchmark->live gap).
Env-gated (POKER44_CAPTURE=1), size-capped, fully fail-safe (never breaks serving).
Captures inputs only; no labels are produced here.
"""
from __future__ import annotations
import os, json, time, threading
from pathlib import Path

_LOCK = threading.Lock()
_DIR = Path(__file__).resolve().parents[1] / "live_capture"
_MAX_BYTES = int(os.getenv("POKER44_CAPTURE_MAX_BYTES", str(250 * 1024 * 1024)))  # 250MB/miner
_state = {"path": None, "full": False}


def capture(chunks, scores, hotkey: str, validator: str) -> None:
    if os.getenv("POKER44_CAPTURE", "0") != "1" or _state["full"] or not chunks:
        return
    try:
        _DIR.mkdir(exist_ok=True)
        if _state["path"] is None:
            _state["path"] = _DIR / f"capture_{(hotkey or 'self')[:10]}.jsonl"
        path = _state["path"]
        if path.exists() and path.stat().st_size >= _MAX_BYTES:
            _state["full"] = True
            return
        ts = time.time()
        with _LOCK:
            with open(path, "a") as f:
                for c, s in zip(chunks, scores):
                    rec = {"t": round(ts, 2), "v": (validator or "")[:8],
                           "n": len(c), "score": round(float(s), 5), "chunk": c}
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception:
        pass  # capture must NEVER affect serving
