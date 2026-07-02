"""Upgraded chunk feature engineering — v2.

Adds 74 new features on top of the original 293 → ~367 total.

New per-hand bases (8, each gets 7 aggregation stats = 56 new features):
  1. schema_bet_to_pot_ratio      – bet/raise size as fraction of pot_before
  2. schema_preflop_aggression_share – aggression rate on preflop actions only
  3. schema_postflop_aggression_share – aggression on flop/turn/river only
  4. schema_fold_to_aggression_rate – P(fold | facing bet/raise from another player)
  5. schema_reraise_rate          – P(reraise | facing bet/raise from another player)
  6. schema_flop_reach_rate       – 1 if hand reached the flop, else 0
  7. schema_river_reach_rate      – 1 if hand reached the river, else 0
  8. schema_sizing_cv             – coeff. of variation of bet/raise sizes (std/mean; 0=bot)

New chunk-level scalars (18 new features):
  Temporal trend (slope of per-hand series, normalised):
    schema_aggression_trend, schema_action_entropy_trend, schema_bet_to_pot_trend
  Half-chunk comparison (bots don't adjust):
    schema_aggression_{first,second}_half, schema_aggression_half_delta
    schema_entropy_{first,second}_half,    schema_entropy_half_delta
  Lag-1 autocorrelation (high corr = scripted):
    schema_aggression_lag1_corr, schema_bet_to_pot_lag1_corr, schema_action_count_lag1_corr
  Bet-size pattern buckets:
    schema_bet_size_bucket_top_share, schema_bet_size_bucket_unique_share
  Additional threshold rates:
    schema_very_high_aggression_hand_rate (≥0.6)
    schema_passive_hand_rate              (aggression=0)
    schema_short_hand_rate                (action_count ≤ 3)
    schema_consistent_sizing_hand_rate    (bet_to_pot_std ≤ 0.1 per hand)

Usage:
    from poker44_ml.features_v2 import chunk_features_v2
    feats = chunk_features_v2(chunk)   # drop-in replacement returning ~367 keys
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


# ── pure utility functions (identical to features.py) ─────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _entropy(values: list[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log(p + 1e-12)
    return _safe_div(ent, math.log(len(counts)))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    q = min(max(float(q), 0.0), 1.0)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _mean(values: list[float]) -> float:
    return _safe_div(sum(values), len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(max(0.0, _mean([(v - m) ** 2 for v in values])))


def _max_run_share(values: list[Any]) -> float:
    if not values:
        return 0.0
    longest = cur = 1
    for prev, cur_val in zip(values, values[1:]):
        if prev == cur_val:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return _safe_div(longest, len(values))


def _amount_bucket(value: float) -> str:
    if value <= 0.0:
        return "z"
    if value <= 0.5:
        return "xs"
    if value <= 1.0:
        return "s"
    if value <= 2.0:
        return "m"
    if value <= 5.0:
        return "l"
    return "xl"


def _aggregate_feature(prefix: str, values: list[float], out: dict[str, float]) -> None:
    out[f"{prefix}_mean"] = _mean(values)
    out[f"{prefix}_std"] = _std(values)
    out[f"{prefix}_min"] = min(values) if values else 0.0
    out[f"{prefix}_max"] = max(values) if values else 0.0
    out[f"{prefix}_q10"] = _quantile(values, 0.1)
    out[f"{prefix}_q50"] = _quantile(values, 0.5)
    out[f"{prefix}_q90"] = _quantile(values, 0.9)


# ── new utility functions ──────────────────────────────────────────────────────

def _linear_slope(values: list[float]) -> float:
    """OLS slope of values vs index [0, 1, …, n-1], normalised by n."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = _mean(values)
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0.0:
        return 0.0
    return num / den


def _lag1_corr(values: list[float]) -> float:
    """Pearson correlation between consecutive elements (lag-1 autocorrelation)."""
    if len(values) < 3:
        return 0.0
    x = values[:-1]
    y = values[1:]
    mx, my = _mean(x), _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return _safe_div(num, den)


def _bet_size_bucket(ratio: float) -> str:
    """Bucket a bet/pot ratio into qualitative size label."""
    if ratio <= 0.0:
        return "z"
    if ratio <= 0.30:
        return "micro"   # < 1/3 pot
    if ratio <= 0.55:
        return "half"    # ~half pot
    if ratio <= 0.80:
        return "tqtr"    # ~3/4 pot
    if ratio <= 1.10:
        return "pot"     # ~pot-sized
    return "over"        # overbet


# ── per-hand feature extraction (v2) ──────────────────────────────────────────

def _hand_features_v2(hand: dict[str, Any]) -> dict[str, float]:
    """Extract all per-hand features (original 40 + 8 new = 48 bases)."""
    metadata = hand.get("metadata") or {}
    players  = hand.get("players") or []
    streets  = hand.get("streets") or []
    actions  = hand.get("actions") or []

    max_seats    = max(1, _safe_int(metadata.get("max_seats"), 6))
    hero_seat    = _safe_int(metadata.get("hero_seat"), 0)
    button_seat  = _safe_int(metadata.get("button_seat"), 0)
    player_count = float(len(players))
    street_count = float(len(streets))
    action_count = float(len(actions))

    action_types:  list[str]   = []
    actor_seats:   list[int]   = []   # filtered (> 0 only) — used by original features
    all_actors:    list[int]   = []   # unfiltered (0 if missing) — used by new features
    street_names:  list[str]   = []
    amount_bb:     list[float] = []
    pot_before:    list[float] = []
    pot_after:     list[float] = []
    stack_bb:      list[float] = []
    raise_to_present = 0
    call_to_present  = 0

    for player in players:
        if isinstance(player, dict):
            stack_bb.append(_safe_div(_safe_float(player.get("starting_stack"), 0.0), 0.02))

    for action in actions:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type") or "").lower().strip()
        actor       = _safe_int(action.get("actor_seat"), 0)
        street      = str(action.get("street") or "").lower().strip()
        amt         = _safe_float(action.get("normalized_amount_bb"), 0.0)
        pb          = _safe_div(_safe_float(action.get("pot_before"), 0.0), 0.02)
        pa          = _safe_div(_safe_float(action.get("pot_after"),  0.0), 0.02)

        action_types.append(action_type)
        all_actors.append(actor)
        if actor > 0:
            actor_seats.append(actor)
        street_names.append(street)
        amount_bb.append(max(0.0, amt))
        pot_before.append(max(0.0, pb))
        pot_after.append(max(0.0, pa))
        raise_to_present += int(action.get("raise_to") is not None)
        call_to_present  += int(action.get("call_to") is not None)

    counts = Counter(action_types)
    meaningful = max(
        counts.get("call", 0) + counts.get("check", 0)
        + counts.get("bet", 0) + counts.get("raise", 0)
        + counts.get("fold", 0),
        1,
    )
    aggressive = counts.get("bet", 0) + counts.get("raise", 0)
    passive    = counts.get("call", 0) + counts.get("check", 0)

    preflop_n   = sum(1 for s in street_names if s == "preflop")
    postflop_n  = sum(1 for s in street_names if s not in {"", "preflop"})
    nonzero_amt = sum(1 for v in amount_bb if v > 0.0)
    hero_actions   = sum(1 for s in actor_seats if s == hero_seat and hero_seat > 0)
    button_actions = sum(1 for s in actor_seats if s == button_seat and button_seat > 0)

    pot_delta = [max(0.0, a - b) for a, b in zip(pot_after, pot_before)]
    monotonic = sum(
        1 for prev, cur in zip(pot_after, pot_after[1:]) if cur + 1e-9 >= prev
    )

    # ── new per-hand computations ──────────────────────────────────────────────

    # 1. Bet/raise size as fraction of pot_before
    bet_to_pot_list: list[float] = []
    for at, amt_val, pb_val in zip(action_types, amount_bb, pot_before):
        if at in ("bet", "raise") and pb_val > 0.0 and amt_val > 0.0:
            bet_to_pot_list.append(amt_val / pb_val)

    schema_bet_to_pot_ratio = _mean(bet_to_pot_list) if bet_to_pot_list else 0.0
    schema_sizing_cv = (
        _safe_div(_std(bet_to_pot_list), _mean(bet_to_pot_list))
        if bet_to_pot_list else 0.0
    )

    # 2 & 3. Street-specific aggression
    pf_meaningful = max(1, sum(
        1 for at, st in zip(action_types, street_names)
        if st == "preflop" and at in ("call", "check", "bet", "raise", "fold")
    ))
    pf_agg = sum(
        1 for at, st in zip(action_types, street_names)
        if st == "preflop" and at in ("bet", "raise")
    )
    postflop_total = max(1, sum(
        1 for st in street_names if st not in ("", "preflop")
    ))
    postflop_agg = sum(
        1 for at, st in zip(action_types, street_names)
        if st not in ("", "preflop") and at in ("bet", "raise")
    )
    schema_preflop_aggression_share  = _safe_div(pf_agg, pf_meaningful)
    schema_postflop_aggression_share = _safe_div(postflop_agg, postflop_total)

    # 4 & 5. Reactions to aggression (fold_to / reraise)
    agg_faced    = 0
    fold_to_agg  = 0
    reraise_cnt  = 0
    for i in range(1, len(action_types)):
        prev_type  = action_types[i - 1]
        prev_actor = all_actors[i - 1]
        curr_type  = action_types[i]
        curr_actor = all_actors[i]
        if prev_type in ("bet", "raise") and prev_actor != curr_actor:
            agg_faced += 1
            if curr_type == "fold":
                fold_to_agg += 1
            elif curr_type == "raise":
                reraise_cnt += 1

    schema_fold_to_aggression_rate = _safe_div(fold_to_agg, agg_faced)
    schema_reraise_rate            = _safe_div(reraise_cnt, agg_faced)

    # 6 & 7. Street reach flags
    streets_in_hand = set(street_names)
    schema_flop_reach_rate  = float("flop"  in streets_in_hand)
    schema_river_reach_rate = float("river" in streets_in_hand)

    # ── assemble output dict ───────────────────────────────────────────────────
    return {
        # ── original 40 bases ─────────────────────────────────────────────────
        "schema_player_count":           player_count,
        "schema_seat_utilization":       _safe_div(player_count, max_seats),
        "schema_action_count":           action_count,
        "schema_street_count":           street_count,
        "schema_call_share":             _safe_div(counts.get("call", 0),  meaningful),
        "schema_check_share":            _safe_div(counts.get("check", 0), meaningful),
        "schema_fold_share":             _safe_div(counts.get("fold", 0),  meaningful),
        "schema_bet_share":              _safe_div(counts.get("bet", 0),   meaningful),
        "schema_raise_share":            _safe_div(counts.get("raise", 0), meaningful),
        "schema_blind_share":            _safe_div(
            counts.get("small_blind", 0) + counts.get("big_blind", 0) + counts.get("ante", 0),
            max(1.0, action_count),
        ),
        "schema_allin_share":            _safe_div(counts.get("all_in", 0), max(1.0, action_count)),
        "schema_aggression_share":       _safe_div(aggressive, max(1.0, action_count)),
        "schema_passive_share":          _safe_div(passive, max(1.0, action_count)),
        "schema_preflop_share":          _safe_div(preflop_n,  max(1.0, action_count)),
        "schema_postflop_share":         _safe_div(postflop_n, max(1.0, action_count)),
        "schema_action_entropy":         _entropy(action_types),
        "schema_actor_entropy":          _entropy(actor_seats),
        "schema_street_entropy":         _entropy(street_names),
        "schema_unique_actor_share":     _safe_div(len(set(actor_seats)), max(1.0, player_count)),
        "schema_actor_switch_rate":      _safe_div(
            sum(1 for prev, cur in zip(actor_seats, actor_seats[1:]) if prev != cur),
            max(len(actor_seats) - 1, 1),
        ),
        "schema_actor_run_max_share":    _max_run_share(actor_seats),
        "schema_action_run_max_share":   _max_run_share(action_types),
        "schema_amount_mean_bb":         _mean(amount_bb),
        "schema_amount_std_bb":          _std(amount_bb),
        "schema_amount_q90_bb":          _quantile(amount_bb, 0.9),
        "schema_amount_max_bb":          max(amount_bb) if amount_bb else 0.0,
        "schema_nonzero_amount_share":   _safe_div(nonzero_amt, max(1.0, action_count)),
        "schema_pot_before_mean_bb":     _mean(pot_before),
        "schema_pot_after_mean_bb":      _mean(pot_after),
        "schema_pot_delta_mean_bb":      _mean(pot_delta),
        "schema_pot_growth_bb":          (
            max(pot_after) - min(pot_before) if pot_after and pot_before else 0.0
        ),
        "schema_pot_monotonic_rate":     _safe_div(monotonic, max(len(pot_after) - 1, 1)),
        "schema_raise_to_share":         _safe_div(raise_to_present, max(1.0, action_count)),
        "schema_call_to_share":          _safe_div(call_to_present,  max(1.0, action_count)),
        "schema_starting_stack_mean_bb": _mean(stack_bb),
        "schema_starting_stack_std_bb":  _std(stack_bb),
        "schema_starting_stack_iqr_bb":  _quantile(stack_bb, 0.75) - _quantile(stack_bb, 0.25),
        "schema_hero_action_share":      _safe_div(hero_actions,   max(1.0, action_count)),
        "schema_button_action_share":    _safe_div(button_actions, max(1.0, action_count)),
        "schema_hero_button_same":       float(hero_seat > 0 and hero_seat == button_seat),

        # ── 8 new per-hand bases ───────────────────────────────────────────────
        "schema_bet_to_pot_ratio":           schema_bet_to_pot_ratio,
        "schema_preflop_aggression_share":   schema_preflop_aggression_share,
        "schema_postflop_aggression_share":  schema_postflop_aggression_share,
        "schema_fold_to_aggression_rate":    schema_fold_to_aggression_rate,
        "schema_reraise_rate":               schema_reraise_rate,
        "schema_flop_reach_rate":            schema_flop_reach_rate,
        "schema_river_reach_rate":           schema_river_reach_rate,
        "schema_sizing_cv":                  schema_sizing_cv,
    }


# ── chunk-level feature aggregation (v2) ──────────────────────────────────────

def chunk_features_v2(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Compute all ~367 features for a chunk of hands.

    Drop-in replacement for chunk_features() returning ~74 extra keys.
    The original 293 keys are preserved with identical values.
    """
    if not chunk:
        return {"hand_count": 0.0}

    out: dict[str, float] = {"hand_count": float(len(chunk))}
    per_hand = [_hand_features_v2(hand) for hand in chunk]
    feature_names = sorted(per_hand[0].keys())

    # Aggregate each per-hand base with 7 statistics
    per_hand_series: dict[str, list[float]] = {}
    for name in feature_names:
        series = [float(feats[name]) for feats in per_hand]
        _aggregate_feature(name, series, out)
        per_hand_series[name] = series

    # ── original chunk-level signature features ────────────────────────────────
    action_signatures:        list[tuple[str, ...]] = []
    actor_signatures:         list[tuple[int, ...]] = []
    street_signatures:        list[tuple[str, ...]] = []
    amount_bucket_signatures: list[tuple[str, ...]] = []

    high_aggressive    = 0
    low_action_entropy = 0
    high_actor_entropy = 0
    long_action_hand   = 0

    # ── new per-hand accumulators ──────────────────────────────────────────────
    bet_size_buckets_per_hand: list[str] = []   # dominant bet-size bucket per hand
    consistent_sizing_count   = 0
    very_high_agg_count       = 0
    passive_hand_count         = 0
    short_hand_count           = 0

    for hand, feats in zip(chunk, per_hand):
        actions = hand.get("actions") or []
        action_types = tuple(
            str((a or {}).get("action_type") or "").lower().strip() for a in actions
        )
        actor_seq = tuple(
            _safe_int((a or {}).get("actor_seat"), 0) for a in actions
            if _safe_int((a or {}).get("actor_seat"), 0) > 0
        )
        street_seq = tuple(
            str((a or {}).get("street") or "").lower().strip() for a in actions
        )
        amounts = [
            max(0.0, _safe_float((a or {}).get("normalized_amount_bb"), 0.0))
            for a in actions
        ]
        amount_buckets = tuple(_amount_bucket(v) for v in amounts)

        action_signatures.append(action_types)
        actor_signatures.append(actor_seq)
        street_signatures.append(street_seq)
        amount_bucket_signatures.append(amount_buckets)

        # original threshold rates
        high_aggressive    += int(feats["schema_aggression_share"] >= 0.35)
        low_action_entropy += int(feats["schema_action_entropy"]   <= 0.35)
        high_actor_entropy += int(feats["schema_actor_entropy"]    >= 0.75)
        long_action_hand   += int(feats["schema_action_count"]     >= 12.0)

        # new threshold rates
        very_high_agg_count += int(feats["schema_aggression_share"] >= 0.60)
        passive_hand_count   += int(feats["schema_aggression_share"] == 0.0)
        short_hand_count     += int(feats["schema_action_count"]     <= 3.0)

        # bet-size bucket: dominant bucket of this hand's bet_to_pot_ratio
        btp = feats["schema_bet_to_pot_ratio"]
        bet_size_buckets_per_hand.append(_bet_size_bucket(btp))

        # consistent sizing: per-hand sizing_cv <= 0.1
        consistent_sizing_count += int(feats["schema_sizing_cv"] <= 0.10)

    n = float(len(chunk))

    # ── write original chunk-level signature features ──────────────────────────
    out["schema_action_signature_top_share"] = _safe_div(
        max(Counter(action_signatures).values()), n
    )
    out["schema_action_signature_unique_share"] = _safe_div(
        len(set(action_signatures)), n
    )
    out["schema_actor_signature_top_share"] = _safe_div(
        max(Counter(actor_signatures).values()), n
    )
    out["schema_actor_signature_unique_share"] = _safe_div(
        len(set(actor_signatures)), n
    )
    out["schema_street_signature_top_share"] = _safe_div(
        max(Counter(street_signatures).values()), n
    )
    out["schema_street_signature_unique_share"] = _safe_div(
        len(set(street_signatures)), n
    )
    out["schema_amount_bucket_signature_top_share"] = _safe_div(
        max(Counter(amount_bucket_signatures).values()), n
    )
    out["schema_amount_bucket_signature_unique_share"] = _safe_div(
        len(set(amount_bucket_signatures)), n
    )
    out["schema_high_aggression_hand_rate"]   = _safe_div(high_aggressive,    n)
    out["schema_low_action_entropy_hand_rate"] = _safe_div(low_action_entropy, n)
    out["schema_high_actor_entropy_hand_rate"] = _safe_div(high_actor_entropy, n)
    out["schema_long_action_hand_rate"]        = _safe_div(long_action_hand,   n)

    # ── NEW chunk-level features ───────────────────────────────────────────────

    # A. Temporal trends (slope of per-hand series)
    agg_series    = per_hand_series["schema_aggression_share"]
    ent_series    = per_hand_series["schema_action_entropy"]
    btp_series    = per_hand_series["schema_bet_to_pot_ratio"]
    cnt_series    = per_hand_series["schema_action_count"]

    out["schema_aggression_trend"]      = _linear_slope(agg_series)
    out["schema_action_entropy_trend"]  = _linear_slope(ent_series)
    out["schema_bet_to_pot_trend"]      = _linear_slope(btp_series)

    # B. First-half vs second-half comparison (bots stay flat)
    mid = max(1, len(chunk) // 2)
    agg_first  = _mean(agg_series[:mid])
    agg_second = _mean(agg_series[mid:]) if agg_series[mid:] else agg_first
    ent_first  = _mean(ent_series[:mid])
    ent_second = _mean(ent_series[mid:]) if ent_series[mid:] else ent_first

    out["schema_aggression_first_half"]  = agg_first
    out["schema_aggression_second_half"] = agg_second
    out["schema_aggression_half_delta"]  = abs(agg_first - agg_second)
    out["schema_entropy_first_half"]     = ent_first
    out["schema_entropy_second_half"]    = ent_second
    out["schema_entropy_half_delta"]     = abs(ent_first - ent_second)

    # C. Lag-1 autocorrelation (high = scripted / repetitive)
    out["schema_aggression_lag1_corr"]   = _lag1_corr(agg_series)
    out["schema_bet_to_pot_lag1_corr"]   = _lag1_corr(btp_series)
    out["schema_action_count_lag1_corr"] = _lag1_corr(cnt_series)

    # D. Bet-size pattern buckets (across hands in chunk)
    out["schema_bet_size_bucket_top_share"] = _safe_div(
        max(Counter(bet_size_buckets_per_hand).values()), n
    )
    out["schema_bet_size_bucket_unique_share"] = _safe_div(
        len(set(bet_size_buckets_per_hand)), n
    )

    # E. Additional threshold rates
    out["schema_very_high_aggression_hand_rate"]  = _safe_div(very_high_agg_count,    n)
    out["schema_passive_hand_rate"]               = _safe_div(passive_hand_count,     n)
    out["schema_short_hand_rate"]                 = _safe_div(short_hand_count,       n)
    out["schema_consistent_sizing_hand_rate"]     = _safe_div(consistent_sizing_count, n)

    return out


# ── compatibility shim ─────────────────────────────────────────────────────────

def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Alias → chunk_features_v2 (drop-in replacement for features.py)."""
    return chunk_features_v2(chunk)
