"""build_dataset_v2.py — Unified benchmark loader with dual-format support and
size-normalizing augmentation.

Handles two fundamentally different API response schemas:

  v1.1  (benchmark.json, Apr 30 – May 8):
    { meta: {dates, ...}, chunks: [[hand,...], ...], groundTruth: [0,1,...] }
    • Flat list of hand-lists with a parallel label array
    • No per-chunk date — distributed from meta.dates sequentially
    • Chunk sizes: 40–80 hands (avg 60)
    • No hand_id field on individual hands

  v1.12 (benchmark_YYYY-MM-DD.json, May 26 – present):
    { data: { chunks: [{ chunkId, split, groundTruth, chunks: [[hand,..],..] }] } }
    • Nested chunk-groups with embedded labels
    • Per-chunk source_date from group metadata
    • Chunk sizes: 30–40 hands (avg 34)
    • hand_id present on each hand

Live validator reality: 80–100 hands per chunk (avg ~98).

Both dataset versions are significantly smaller than live. This module provides
size-normalizing augmentation strategies to close that gap:

  AugmentMode.NONE       – return chunks exactly as stored (baseline)
  AugmentMode.SUBSAMPLE  – draw K evenly-spaced random windows of target_size
                           from chunks that are meaningfully larger than the
                           target (>= target * merge_ratio, default 1.5x).
                           Chunks below that threshold are kept as-is.
  AugmentMode.MERGE      – merge consecutive same-date same-label chunks until
                           reaching target * merge_ratio hands. Produces larger
                           super-chunks that are genuinely useful to subsample.
                           Applied per (source_date, label) bucket.
  AugmentMode.BOTH       – MERGE to target * merge_ratio first, then SUBSAMPLE
                           K distinct windows of target_size from each merged
                           chunk. This is the recommended mode: each window is
                           a real contiguous slice at live-realistic size, and
                           the K draws are spread evenly so they cover different
                           hands with minimal overlap.

Typical usage:

    from training.build_dataset_v2 import load_all_benchmark_examples, AugmentMode

    # Load everything, merge small chunks up to ~80 hands, subsample large ones
    exs = load_all_benchmark_examples(
        "data/benchmark_out",
        augment=AugmentMode.BOTH,
        target_size=80,
        subsample_draws=3,
        seed=42,
    )
    chunks = [e["chunk"] for e in exs]
    labels = [e["label"]  for e in exs]
    dates  = [e["source_date"] for e in exs]   # use for LODO cross-validation

    # Or load a single file (auto-detects v1.1 vs v1.12):
    from training.build_dataset_v2 import load_benchmark_file
    exs = load_benchmark_file("data/benchmark_out/benchmark.json")
"""
from __future__ import annotations

import enum
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from poker44_bump.payload_view import prepare_hand_for_miner
from poker44_ml.features import chunk_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# Live validator sends ~80–100 hand chunks; train at this size for realism.
DEFAULT_TARGET_SIZE = 80
DEFAULT_SUBSAMPLE_DRAWS = 3
# Merge accumulates to target * MERGE_RATIO so subsampled windows are distinct.
# At 1.5x: 80-hand target → 120-hand pool → 3 draws span positions 0–40 (non-trivial spread).
DEFAULT_MERGE_RATIO = 1.5

# v1.1 meta date list has 9 dates covering 4280 chunks.
# We distribute chunks sequentially across those dates.
_V11_SOURCE_PREFIX = "v1.1"


# ---------------------------------------------------------------------------
# Augmentation mode
# ---------------------------------------------------------------------------

class AugmentMode(enum.Enum):
    NONE      = "none"       # as-is; no size normalization
    SUBSAMPLE = "subsample"  # random windows from large chunks
    MERGE     = "merge"      # merge small same-date same-label chunks
    BOTH      = "both"       # merge first, then subsample


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json_or_gz(path: str | Path) -> Any:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_benchmark_dir(
    path: str | Path | None,
    *,
    include_v11: bool = True,
    include_v112: bool = True,
) -> list[Path]:
    """Return sorted list of benchmark JSON files from a directory or direct path."""
    if path is None:
        path = REPO_ROOT / "data" / "benchmark_out"
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"Benchmark path not found: {p}")

    files: list[Path] = []
    if include_v11:
        # The large flat-format file (no date suffix)
        candidate = p / "benchmark.json"
        if candidate.exists():
            files.append(candidate)
    if include_v112:
        # Daily files: benchmark_YYYY-MM-DD.json
        files.extend(sorted(p.glob("benchmark_2[0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")))
    if not files:
        raise FileNotFoundError(f"No benchmark files found in {p}")
    return files


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def miner_visible_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the same sanitization miners receive from the validator."""
    return [
        prepare_hand_for_miner(hand)
        for hand in chunk
        if isinstance(hand, dict)
    ]


def _sanitize_and_featurize(
    chunk: list[dict[str, Any]],
    *,
    miner_visible: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    visible = miner_visible_chunk(chunk) if miner_visible else chunk
    if not visible:
        return [], {"hand_count": 0.0}
    feats = chunk_features(visible)
    feats["hand_count"] = float(len(visible))
    return visible, feats


# ---------------------------------------------------------------------------
# v1.1 loader  {meta, chunks: [[hand,...]], groundTruth: [0/1,...]}
# ---------------------------------------------------------------------------

def _is_v11_format(payload: dict) -> bool:
    """Detect flat v1.1 format: top-level chunks is a list of lists."""
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        return False
    return isinstance(raw_chunks[0], list)


def _assign_dates_v11(n_chunks: int, dates: list[str]) -> list[str]:
    """Distribute n_chunks sequentially across dates (round-robin remainder)."""
    if not dates:
        return [_V11_SOURCE_PREFIX] * n_chunks
    per_date, remainder = divmod(n_chunks, len(dates))
    mapping: list[str] = []
    for i, d in enumerate(dates):
        count = per_date + (1 if i < remainder else 0)
        mapping.extend([d] * count)
    return mapping


def _load_v11(
    path: Path,
    *,
    miner_visible: bool = True,
) -> list[dict[str, Any]]:
    payload = load_json_or_gz(path)
    if not _is_v11_format(payload):
        raise ValueError(f"Not a v1.1 flat-format file: {path}")

    flat_chunks: list[list[dict]] = payload["chunks"]
    labels: list[int] = [int(x) for x in payload["groundTruth"]]
    meta: dict = payload.get("meta", {})
    dates: list[str] = meta.get("dates", [])

    if len(flat_chunks) != len(labels):
        raise RuntimeError(
            f"v1.1: {len(flat_chunks)} chunks but {len(labels)} labels in {path}"
        )

    date_map = _assign_dates_v11(len(flat_chunks), dates)
    examples: list[dict[str, Any]] = []

    for i, (chunk, label) in enumerate(zip(flat_chunks, labels)):
        hand_chunk = [h for h in chunk if isinstance(h, dict)]
        if not hand_chunk:
            continue
        visible, feats = _sanitize_and_featurize(hand_chunk, miner_visible=miner_visible)
        if not visible:
            continue
        examples.append({
            "chunk":       visible,
            "label":       label,
            "source_date": date_map[i],
            "group_id":    f"v11_chunk_{i}",
            "group_hash":  "",
            "item_index":  i,
            "source_path": str(path),
            "version":     "v1.1",
            "features":    feats,
        })
    return examples


# ---------------------------------------------------------------------------
# v1.12 loader  {data: {chunks: [{chunkId, groundTruth, chunks}]}}
# ---------------------------------------------------------------------------

def _load_v112(
    path: Path,
    *,
    miner_visible: bool = True,
) -> list[dict[str, Any]]:
    payload = load_json_or_gz(path)
    root = payload.get("data", payload)

    groups = [g for g in root.get("chunks", []) if isinstance(g, dict)]
    if not groups:
        raise RuntimeError(f"v1.12: no chunk groups in {path}")

    source_date_top = str(root.get("sourceDate", ""))
    examples: list[dict[str, Any]] = []

    for gi, group in enumerate(groups):
        inner_chunks = group.get("chunks") or []
        gt = group.get("groundTruth") or []
        # groundTruthLabels encodes "bot"/"human" — fall back to groundTruth int list
        if not gt:
            raw_labels = group.get("groundTruthLabels") or []
            gt = [1 if str(l).lower() == "bot" else 0 for l in raw_labels]

        if len(inner_chunks) != len(gt):
            raise RuntimeError(
                f"v1.12 group {gi}: {len(inner_chunks)} chunks vs {len(gt)} labels in {path}"
            )

        source_date = str(group.get("sourceDate") or source_date_top or "")
        group_id = str(group.get("chunkId") or f"group_{gi}")
        group_hash = str(group.get("chunkHash") or "")
        split = str(group.get("split") or "")

        for ii, (chunk, label) in enumerate(zip(inner_chunks, gt)):
            if not isinstance(chunk, list):
                continue
            hand_chunk = [h for h in chunk if isinstance(h, dict)]
            if not hand_chunk:
                continue
            visible, feats = _sanitize_and_featurize(hand_chunk, miner_visible=miner_visible)
            if not visible:
                continue
            examples.append({
                "chunk":       visible,
                "label":       int(label),
                "source_date": source_date,
                "group_id":    group_id,
                "group_hash":  group_hash,
                "item_index":  ii,
                "source_path": str(path),
                "version":     "v1.12",
                "split":       split,
                "features":    feats,
            })
    return examples


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

def load_benchmark_file(
    path: str | Path,
    *,
    miner_visible: bool = True,
) -> list[dict[str, Any]]:
    """Load a single benchmark file, auto-detecting v1.1 vs v1.12 format."""
    p = Path(path)
    payload = load_json_or_gz(p)
    if _is_v11_format(payload):
        return _load_v11(p, miner_visible=miner_visible)
    return _load_v112(p, miner_visible=miner_visible)


# ---------------------------------------------------------------------------
# Augmentation: subsampling (large chunks → multiple smaller windows)
# ---------------------------------------------------------------------------

def _subsample_examples(
    examples: list[dict[str, Any]],
    *,
    target_size: int,
    draws: int,
    seed: int,
    min_size: int | None = None,
) -> list[dict[str, Any]]:
    """Draw K evenly-spaced random windows of target_size from large chunks.

    Only applied when len(chunk) >= min_size (default: target_size * 1.5).
    This guarantees the K windows are meaningfully different — each covers a
    distinct region of the pool with limited overlap.

    Start positions are spread evenly across the valid range [0, pool-target],
    then jittered by a small random offset so draws aren't perfectly regular.

    Chunks smaller than min_size are kept as-is (no subsampling applied).
    The original large chunk is NOT added — only the K windows are emitted,
    each at exactly target_size hands. This gives the model uniform chunk-size
    exposure at live-realistic size.
    """
    if min_size is None:
        min_size = math.ceil(target_size * DEFAULT_MERGE_RATIO)

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []

    for ex in examples:
        chunk = ex["chunk"]

        if len(chunk) < min_size:
            # Too small for meaningful subsampling — keep as-is.
            out.append(ex)
            continue

        max_start = len(chunk) - target_size
        if max_start <= 0:
            out.append(ex)
            continue

        # Evenly space draw anchor points across [0, max_start], then jitter.
        anchors = [round(i * max_start / (draws - 1)) for i in range(draws)] if draws > 1 else [0]
        jitter_range = max(1, max_start // (draws * 2))

        for draw_i, anchor in enumerate(anchors):
            jitter = rng.randint(-jitter_range, jitter_range)
            start = max(0, min(max_start, anchor + jitter))
            sub = chunk[start: start + target_size]
            feats = chunk_features(sub)
            feats["hand_count"] = float(len(sub))
            out.append({
                **ex,
                "chunk":      sub,
                "features":   feats,
                "group_id":   ex["group_id"] + f"_sub{draw_i}",
                "augmented":  True,
                "aug_type":   "subsample",
                "aug_start":  start,
                "aug_pool":   len(chunk),
            })
    return out


# ---------------------------------------------------------------------------
# Augmentation: merging (small same-date same-label → one larger chunk)
# ---------------------------------------------------------------------------

def _merge_examples(
    examples: list[dict[str, Any]],
    *,
    target_size: int,
    merge_ratio: float = DEFAULT_MERGE_RATIO,
) -> list[dict[str, Any]]:
    """Merge consecutive same-date same-label chunks to reach merge_target hands.

    merge_target = ceil(target_size * merge_ratio)  (default: 1.5x target)

    The larger merged pool is what makes subsequent subsampling produce K
    genuinely distinct windows. For example:
        target=80, ratio=1.5 → pool=120 hands
        3 subsample draws span start positions 0, 20, 40 → each covers a
        different 80-hand window with only ~75% overlap (vs 100% if pool=80).

    Chunks already >= merge_target pass through unchanged (they are large
    enough to subsample directly). Smaller chunks accumulate in order within
    each (source_date, label) bucket. The leftover at bucket end is emitted
    as-is even if shorter than merge_target; it will not be subsampled
    (subsampling threshold also requires >= merge_target hands).
    """
    merge_target = math.ceil(target_size * merge_ratio)

    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        key = (ex["source_date"], ex["label"])
        buckets[key].append(ex)

    out: list[dict[str, Any]] = []

    for (source_date, label), group in buckets.items():
        hand_pool: list[dict[str, Any]] = []
        source_ids: list[str] = []
        merge_index = 0

        def _emit(hands: list[dict], ids: list[str], idx: int) -> dict[str, Any]:
            feats = chunk_features(hands)
            feats["hand_count"] = float(len(hands))
            return {
                "chunk":       hands,
                "label":       label,
                "source_date": source_date,
                "group_id":    "merged_" + "_".join(ids[:3]) + (f"_+{len(ids)-3}" if len(ids) > 3 else ""),
                "group_hash":  "",
                "item_index":  idx,
                "source_path": group[0]["source_path"],
                "version":     group[0].get("version", ""),
                "features":    feats,
                "augmented":   len(ids) > 1,
                "aug_type":    "merge" if len(ids) > 1 else "none",
                "merged_from": ids,
                "merge_target": merge_target,
            }

        for ex in group:
            chunk = ex["chunk"]
            if len(chunk) >= merge_target:
                # Already large enough — flush pending pool, then pass through.
                if hand_pool:
                    out.append(_emit(hand_pool, source_ids, merge_index))
                    merge_index += 1
                    hand_pool, source_ids = [], []
                out.append(ex)
                continue

            hand_pool.extend(chunk)
            source_ids.append(ex["group_id"])

            if len(hand_pool) >= merge_target:
                # Emit the full merge_target pool (no truncation below merge_target).
                out.append(_emit(hand_pool[:merge_target], source_ids, merge_index))
                merge_index += 1
                hand_pool = hand_pool[merge_target:]
                source_ids = []

        # Flush remainder — may be < merge_target; subsampling won't touch it.
        if hand_pool:
            out.append(_emit(hand_pool, source_ids, merge_index))

    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_all_benchmark_examples(
    benchmark_dir: str | Path | None = None,
    *,
    include_v11: bool = True,
    include_v112: bool = True,
    augment: AugmentMode = AugmentMode.BOTH,
    target_size: int = DEFAULT_TARGET_SIZE,
    merge_ratio: float = DEFAULT_MERGE_RATIO,
    subsample_draws: int = DEFAULT_SUBSAMPLE_DRAWS,
    seed: int = 42,
    miner_visible: bool = True,
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """Load, sanitize, augment, and deduplicate all benchmark examples.

    Parameters
    ----------
    benchmark_dir:
        Directory containing benchmark JSON files, or a single file path.
        Defaults to <repo>/data/benchmark_out.
    include_v11:
        Include the large flat-format v1.1 file (benchmark.json).
    include_v112:
        Include v1.12 daily files (benchmark_YYYY-MM-DD.json).
    augment:
        Size-normalization strategy (see AugmentMode docstring).
    target_size:
        Hands per subsample window. Default 80, matching the lower end of
        live validator queries (80–100 hands).
    merge_ratio:
        Merge accumulates to ceil(target_size * merge_ratio) hands before
        emitting. Default 1.5 — at target=80 this gives a 120-hand pool so
        3 subsample draws span positions 0/20/40 with meaningful separation.
        Increase to 2.0 for larger pools and more distinct draws.
    subsample_draws:
        Number of evenly-spaced windows drawn per merged pool.
    seed:
        Random seed for reproducible subsampling.
    miner_visible:
        Apply validator sanitization (prepare_hand_for_miner). Always True
        for training to match live inference.
    deduplicate:
        Skip exact-duplicate examples by (source_path, group_id, item_index).

    Returns
    -------
    List of example dicts, each containing:
        chunk        – list of sanitized hand dicts (target_size hands each)
        label        – 1 (bot) or 0 (human)
        source_date  – YYYY-MM-DD string (use for LODO cross-validation)
        version      – "v1.1" or "v1.12"
        features     – dict of chunk-level features (pre-computed)
        augmented    – True if this example was created by augmentation
        aug_type     – "subsample" | "merge" | "none"
        aug_pool     – (subsample only) size of the source pool in hands
        aug_start    – (subsample only) start index within the pool
    """
    paths = resolve_benchmark_dir(
        benchmark_dir,
        include_v11=include_v11,
        include_v112=include_v112,
    )

    raw_v11:  list[dict[str, Any]] = []
    raw_v112: list[dict[str, Any]] = []

    for p in paths:
        payload = load_json_or_gz(p)
        if _is_v11_format(payload):
            raw_v11.extend(_load_v11(p, miner_visible=miner_visible))
        else:
            raw_v112.extend(_load_v112(p, miner_visible=miner_visible))

    do_merge     = augment in (AugmentMode.MERGE, AugmentMode.BOTH)
    do_subsample = augment in (AugmentMode.SUBSAMPLE, AugmentMode.BOTH)
    min_subsample = math.ceil(target_size * merge_ratio)

    aug_v11 = raw_v11
    if do_merge and aug_v11:
        aug_v11 = _merge_examples(aug_v11, target_size=target_size, merge_ratio=merge_ratio)
    if do_subsample and aug_v11:
        aug_v11 = _subsample_examples(
            aug_v11, target_size=target_size, draws=subsample_draws,
            seed=seed, min_size=min_subsample,
        )

    aug_v112 = raw_v112
    if do_merge and aug_v112:
        aug_v112 = _merge_examples(aug_v112, target_size=target_size, merge_ratio=merge_ratio)
    if do_subsample and aug_v112:
        aug_v112 = _subsample_examples(
            aug_v112, target_size=target_size, draws=subsample_draws,
            seed=seed, min_size=min_subsample,
        )

    all_examples = aug_v11 + aug_v112

    # --- Deduplication ---
    if deduplicate:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for ex in all_examples:
            key = "|".join([
                ex.get("source_path", ""),
                ex.get("group_id", ""),
                str(ex.get("item_index", "")),
            ])
            if key in seen:
                continue
            seen.add(key)
            unique.append(ex)
        all_examples = unique

    return all_examples


# ---------------------------------------------------------------------------
# Backward-compatible shim (drop-in replacement for build_dataset.load_benchmark_examples)
# ---------------------------------------------------------------------------

def load_benchmark_examples(
    paths: str | Path | list[str | Path],
    *,
    miner_visible: bool = True,
) -> list[dict[str, Any]]:
    """Backward-compatible loader: accepts file list or dir, no augmentation.

    This matches the original build_dataset.load_benchmark_examples signature
    so existing train_v*.py scripts continue to work without modification.
    For new training scripts prefer load_all_benchmark_examples() with
    explicit augmentation settings.
    """
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(p) for p in paths]

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()

    for p in path_list:
        if p.is_dir():
            sub = resolve_benchmark_dir(p)
        else:
            sub = [p]
        for fp in sub:
            for ex in load_benchmark_file(fp, miner_visible=miner_visible):
                key = "|".join([
                    ex.get("source_path", ""),
                    ex.get("group_id", ""),
                    str(ex.get("item_index", "")),
                ])
                if key in seen:
                    continue
                seen.add(key)
                examples.append(ex)

    if not examples:
        raise RuntimeError("No benchmark examples loaded.")
    return examples


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, collections

    parser = argparse.ArgumentParser(description="build_dataset_v2 smoke-test")
    parser.add_argument("--dir", default="data/benchmark_out")
    parser.add_argument("--augment", default="both",
                        choices=["none", "subsample", "merge", "both"])
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--merge-ratio", type=float, default=DEFAULT_MERGE_RATIO)
    parser.add_argument("--draws", type=int, default=DEFAULT_SUBSAMPLE_DRAWS)
    parser.add_argument("--no-v11",  action="store_true")
    parser.add_argument("--no-v112", action="store_true")
    args = parser.parse_args()

    mode = AugmentMode(args.augment)
    exs = load_all_benchmark_examples(
        args.dir,
        include_v11=not args.no_v11,
        include_v112=not args.no_v112,
        augment=mode,
        target_size=args.target_size,
        merge_ratio=args.merge_ratio,
        subsample_draws=args.draws,
    )

    versions  = collections.Counter(e.get("version", "?") for e in exs)
    labels    = collections.Counter(e["label"] for e in exs)
    aug_types = collections.Counter(e.get("aug_type", "none") for e in exs)
    dates     = sorted(set(e["source_date"] for e in exs))
    sizes     = [len(e["chunk"]) for e in exs]

    print(f"\n{'='*60}")
    print(f"  build_dataset_v2  augment={mode.value}  target={args.target_size}")
    print(f"{'='*60}")
    print(f"  Total examples : {len(exs)}")
    print(f"  By version     : {dict(versions)}")
    print(f"  By label       : {dict(labels)}  (0=human, 1=bot)")
    print(f"  By aug_type    : {dict(aug_types)}")
    print(f"  Unique dates   : {len(dates)}  ({dates[0]} … {dates[-1]})")
    print(f"  Chunk sizes    : min={min(sizes)} max={max(sizes)} "
          f"mean={sum(sizes)/len(sizes):.1f}")
    print(f"  Total hands    : {sum(sizes):,}")
    print(f"{'='*60}\n")
