"""Batch-level prior calibration for miner inference.

Blends model scores with rank-based targets, then enforces a target fraction
of scores below/above the validator threshold (0.5). Useful when model scores
drift low on live data but chunk ranking remains informative.

Default prior: 60% below 0.5 (human-like), 40% above 0.5 (bot-like).
"""

from __future__ import annotations

import random


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rank_percentiles(scores: list[float]) -> list[float]:
    n = len(scores)
    if n <= 1:
        return [0.5 for _ in scores]
    order = sorted(range(n), key=lambda idx: (scores[idx], idx))
    ranks = [0.0] * n
    for rank, idx in enumerate(order):
        ranks[idx] = (rank + 0.5) / n
    return ranks


def _rank_to_prob(rank_pct: float) -> float:
    return _clamp01(0.05 + 0.90 * float(rank_pct))


def _spread_below(count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.25]
    step = 0.44 / (count - 1)
    return [_clamp01(0.05 + step * index) for index in range(count)]


def _spread_above(count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.75]
    step = 0.44 / (count - 1)
    return [_clamp01(0.51 + step * index) for index in range(count)]


def build_frac_above_choices(
    low: float,
    high: float,
    *,
    step: float = 0.05,
) -> list[float]:
    """Discrete target bot-rate grid from ``low`` to ``high`` in ``step`` increments."""
    low_val = _clamp01(float(low))
    high_val = _clamp01(float(high))
    if low_val > high_val:
        low_val, high_val = high_val, low_val
    step_val = max(float(step), 1e-6)
    if low_val <= 0.0 or high_val >= 1.0:
        raise ValueError("frac_above choices must stay strictly inside (0, 1)")

    choices: list[float] = []
    value = low_val
    while value <= high_val + 1e-9:
        choices.append(round(value, 2))
        value += step_val
    if not choices:
        choices = [round(low_val, 2)]
    return choices


def pick_random_frac_above(
    low: float,
    high: float,
    *,
    step: float = 0.05,
    rng: random.Random | None = None,
) -> float:
    """Pick one target bot fraction from the discrete grid."""
    choices = build_frac_above_choices(low, high, step=step)
    picker = rng or random
    return float(picker.choice(choices))


def apply_batch_prior_calibration(
    scores: list[float],
    *,
    frac_above: float = 0.40,
    alpha: float = 0.70,
    min_chunks: int = 8,
) -> list[float]:
    """Hybrid batch calibration with a human/bot count prior.

    Parameters
    ----------
    scores:
        Base model risk scores in [0, 1], one per chunk.
    frac_above:
        Target fraction strictly above 0.5 (predicted bot). ``0.40`` implies
        a 60:40 human-to-bot prediction split at the 0.5 threshold.
    alpha:
        Weight on the base model score vs rank-derived target in [0, 1].
    min_chunks:
        Minimum batch size required to apply calibration.
    """
    if not scores:
        return []

    n = len(scores)
    if n < max(2, int(min_chunks)):
        return [_clamp01(value) for value in scores]

    frac_above = _clamp01(float(frac_above))
    if frac_above <= 0.0 or frac_above >= 1.0:
        return [_clamp01(value) for value in scores]

    alpha = _clamp01(float(alpha))
    rank_pct = _rank_percentiles(scores)
    targets = [_rank_to_prob(value) for value in rank_pct]
    blended = [
        _clamp01(alpha * base + (1.0 - alpha) * target)
        for base, target in zip(scores, targets)
    ]

    k_above = max(1, min(n - 1, int(round(n * frac_above))))
    k_below = n - k_above
    sorted_idx = sorted(range(n), key=lambda idx: (blended[idx], idx))

    below_values = _spread_below(k_below)
    above_values = _spread_above(k_above)
    final = [0.0] * n
    for slot, idx in enumerate(sorted_idx[:k_below]):
        final[idx] = below_values[slot]
    for slot, idx in enumerate(sorted_idx[k_below:]):
        final[idx] = above_values[slot]
    return [_clamp01(value) for value in final]
