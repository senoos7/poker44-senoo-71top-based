"""Upgraded chunk feature engineering — v3.

Builds on v2 (367 features) and adds 40 new features targeting the FPR problem
diagnosed from v2: flop/river reach rates and lag correlations are bidirectional
(sometimes bots have high reach, sometimes low), causing human chunks with
consistent play to be mis-ranked above passive bots.

v3 strategy
-----------
1. Bet-size snapping  — bots program exact fractions (1/3, 1/2, 3/4, pot).
   Humans deviate. Very low false-positive risk.

2. Cross-hand consistency  — bots have near-zero hand-length and sizing variance
   across hands in a chunk.  A skilled human who bets consistently still varies
   between hands more than a bot.

3. Action-bigram entropy  — bots replay the same action-transition sequences.
   Measuring bigram (action[i]→action[i+1]) entropy at the chunk level captures
   repetitiveness that per-hand statistics miss.

4. Compound bot index  — requires snap_rate AND low sizing_cv AND low action
   complexity to all be bot-like simultaneously.  Prevents a single consistent
   human from triggering the signal.

5. Reach-rate consistency (CV)  — instead of raw flop/river reach rate (which
   is bidirectional), v3 also provides the CV of reach rates across hands.
   Bots are always consistently at the same reach level; humans vary.

New per-hand bases (+4, total 52 vs v2's 48 → 52×7 = 364 aggregated):
  1. schema_bet_snap_rate      — fraction of bets landing within ±8% of a
                                 standard fraction (1/4, 1/3, 1/2, 2/3, 3/4,
                                 1, 1.25, 1.5, 2)
  2. schema_action_bigram_ent  — normalised entropy of consecutive action-type
                                 pairs within this hand
  3. schema_bet_count          — absolute count of bet+raise actions (used to
                                 compute cross-hand CV at chunk level)
  4. schema_action_complexity  — unique (street, action_type) pairs / total
                                 actions.  Low = uniform pattern = bot.

New chunk-level scalars (+22, total ~52 vs v2's 30):
  Consistency:
    schema_hand_length_cv          — CV of action_count across hands
    schema_cross_hand_btp_std      — std of per-hand bet_to_pot_ratio mean
    schema_bet_count_cv            — CV of per-hand bet+raise count
    schema_flop_reach_cv           — CV of per-hand flop_reach_rate
    schema_river_reach_cv          — CV of per-hand river_reach_rate
    schema_action_complexity_cv    — CV of per-hand action_complexity
  Bet-size snapping:
    schema_snap_high_rate          — fraction of hands with snap_rate > 0.50
    schema_snap_consistent_rate    — fraction of hands with snap_rate > 0.60
    schema_bet_size_iqr            — IQR of all individual bet/pot ratios in chunk
    schema_bet_size_snap_concentration — fraction of bets in dominant snap bucket
  Action-bigram entropy:
    schema_chunk_bigram_entropy    — bigram entropy across all hands in chunk
    schema_low_bigram_entropy_rate — fraction of hands with bigram_ent < 0.40
  Compound signals:
    schema_compound_bot_index      — snap_rate_mean × (1−sizing_cv_mean) ×
                                     (1−action_complexity_mean); fires only when
                                     multiple signals align
    schema_snap_and_low_cv_rate    — fraction of hands with snap_rate>0.4 AND
                                     sizing_cv<0.25 simultaneously
  Reach-rate as consistency (replacing raw direction):
    schema_reach_delta_cv          — CV of (flop_reach − river_reach) per hand
  FPR-guard rates (human-positive signals):
    schema_reraise_hand_rate       — fraction of hands with reraise_rate > 0
    schema_check_raise_hand_rate   — fraction of hands with fold_to_agg=0 AND
                                     reraise_rate > 0 (proxy for check-raise)
    schema_high_action_complexity_rate — fraction of hands with complexity > 0.6
    schema_high_flop_reach_rate    — fraction of hands with flop_reach_rate = 1
    schema_varied_btp_hand_rate    — fraction of hands with sizing_cv > 0.30
    schema_diverse_bigram_hand_rate — fraction of hands with bigram_ent > 0.70

Usage:
    from poker44_ml.features_v3 import chunk_features_v3
    feats = chunk_features_v3(chunk)   # ~407 keys (superset of v2's ~367)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

# ── re-export all v2 internals ────────────────────────────────────────────────
from poker44_ml.features_v2 import (
    _safe_float, _safe_int, _safe_div, _clamp01,
    _entropy, _quantile, _mean, _std,
    _max_run_share, _amount_bucket,
    _aggregate_feature,
    _linear_slope, _lag1_corr, _bet_size_bucket,
    _hand_features_v2,
    chunk_features_v2,
)

# ── snap-to-fraction constants ────────────────────────────────────────────────
_SNAP_FRACTIONS = [0.25, 0.333, 0.5, 0.667, 0.75, 1.0, 1.25, 1.5, 2.0]
_SNAP_TOLERANCE = 0.08   # ±8% absolute tolerance


def _snaps_to_fraction(ratio: float) -> bool:
    """Return True if ratio is within ±SNAP_TOLERANCE of any standard fraction."""
    return any(abs(ratio - f) <= _SNAP_TOLERANCE for f in _SNAP_FRACTIONS)


def _nearest_snap_label(ratio: float) -> str:
    """Return label of nearest standard fraction, or 'none'."""
    if ratio <= 0.0:
        return "zero"
    best_dist = float("inf")
    best_label = "none"
    labels = ["q1", "third", "half", "twothird", "tqtr", "pot", "q5", "h5", "dbl"]
    for label, frac in zip(labels, _SNAP_FRACTIONS):
        d = abs(ratio - frac)
        if d < best_dist:
            best_dist = d
            best_label = label
    return best_label if best_dist <= _SNAP_TOLERANCE else "other"


# ── per-hand v3 features ──────────────────────────────────────────────────────

def _hand_features_v3(hand: dict[str, Any]) -> dict[str, float]:
    """All v2 features + 4 new per-hand bases (52 total)."""
    base = _hand_features_v2(hand)

    actions     = hand.get("actions") or []
    action_types: list[str]   = []
    street_names: list[str]   = []
    amount_bb:    list[float] = []
    pot_before:   list[float] = []

    for a in actions:
        if not isinstance(a, dict):
            continue
        action_types.append(str(a.get("action_type") or "").lower().strip())
        street_names.append(str(a.get("street") or "").lower().strip())
        amount_bb.append(max(0.0, _safe_float(a.get("normalized_amount_bb"), 0.0)))
        pot_before.append(max(0.0, _safe_div(
            _safe_float(a.get("pot_before"), 0.0), 0.02
        )))

    # 1. Bet-snap rate
    bet_snaps = 0; bet_total = 0
    for at, amt, pb in zip(action_types, amount_bb, pot_before):
        if at in ("bet", "raise") and pb > 0.0 and amt > 0.0:
            ratio = amt / pb
            bet_total += 1
            if _snaps_to_fraction(ratio):
                bet_snaps += 1
    schema_bet_snap_rate = _safe_div(bet_snaps, bet_total)

    # 2. Action-bigram entropy (consecutive action pairs within this hand)
    if len(action_types) >= 2:
        bigrams = list(zip(action_types, action_types[1:]))
        schema_action_bigram_ent = _entropy(bigrams)
    else:
        schema_action_bigram_ent = 0.0

    # 3. Bet count
    schema_bet_count = float(
        sum(1 for at in action_types if at in ("bet", "raise"))
    )

    # 4. Action complexity — unique (street, action_type) pairs / total actions
    if action_types:
        unique_combos = len(set(zip(street_names, action_types)))
        schema_action_complexity = _safe_div(unique_combos, len(action_types))
    else:
        schema_action_complexity = 0.0

    base["schema_bet_snap_rate"]      = schema_bet_snap_rate
    base["schema_action_bigram_ent"]  = schema_action_bigram_ent
    base["schema_bet_count"]          = schema_bet_count
    base["schema_action_complexity"]  = schema_action_complexity
    return base


# ── chunk-level v3 features ───────────────────────────────────────────────────

def chunk_features_v3(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Compute all ~407 features for a chunk of hands.

    Superset of v2 (367 features) — all v2 keys present with identical values.
    Adds 40 new chunk-level features targeted at reducing FPR.
    """
    if not chunk:
        return {"hand_count": 0.0}

    out: dict[str, float] = {"hand_count": float(len(chunk))}
    per_hand = [_hand_features_v3(hand) for hand in chunk]
    feature_names = sorted(per_hand[0].keys())

    # Aggregate each per-hand base with 7 statistics (same as v2)
    per_hand_series: dict[str, list[float]] = {}
    for name in feature_names:
        series = [float(feats[name]) for feats in per_hand]
        _aggregate_feature(name, series, out)
        per_hand_series[name] = series

    # ── re-compute original v2 chunk-level signature features ─────────────────
    action_signatures:        list[tuple[str, ...]] = []
    actor_signatures:         list[tuple[int, ...]] = []
    street_signatures:        list[tuple[str, ...]] = []
    amount_bucket_signatures: list[tuple[str, ...]] = []

    high_aggressive     = 0
    low_action_entropy  = 0
    high_actor_entropy  = 0
    long_action_hand    = 0
    very_high_agg_count = 0
    passive_hand_count  = 0
    short_hand_count    = 0
    consistent_sizing   = 0
    bet_size_buckets_per_hand: list[str] = []

    # ── new v3 per-hand accumulators ───────────────────────────────────────────
    snap_high_count          = 0   # snap_rate > 0.50
    snap_consistent_count    = 0   # snap_rate > 0.60
    snap_and_low_cv_count    = 0   # snap_rate > 0.40 AND sizing_cv < 0.25
    low_bigram_ent_count     = 0   # bigram_ent < 0.40
    diverse_bigram_count     = 0   # bigram_ent > 0.70
    high_complexity_count    = 0   # action_complexity > 0.60
    reraise_hand_count       = 0   # reraise_rate > 0
    check_raise_proxy_count  = 0   # fold_to_agg=0 AND reraise_rate > 0
    high_flop_reach_count    = 0   # flop_reach_rate = 1
    varied_btp_count         = 0   # sizing_cv > 0.30

    # chunk-level bet/pot ratio collection for IQR and snap concentration
    all_btp_chunk: list[float] = []
    snap_labels_chunk: list[str] = []

    # chunk-level bigrams across ALL hands
    chunk_bigrams: list[tuple[str, str]] = []

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

        # v2 threshold rates
        high_aggressive     += int(feats["schema_aggression_share"] >= 0.35)
        low_action_entropy  += int(feats["schema_action_entropy"]   <= 0.35)
        high_actor_entropy  += int(feats["schema_actor_entropy"]    >= 0.75)
        long_action_hand    += int(feats["schema_action_count"]     >= 12.0)
        very_high_agg_count += int(feats["schema_aggression_share"] >= 0.60)
        passive_hand_count  += int(feats["schema_aggression_share"] == 0.0)
        short_hand_count    += int(feats["schema_action_count"]     <= 3.0)
        consistent_sizing   += int(feats["schema_sizing_cv"]        <= 0.10)
        btp = feats["schema_bet_to_pot_ratio"]
        bet_size_buckets_per_hand.append(_bet_size_bucket(btp))

        # v3 threshold rates
        snap  = feats["schema_bet_snap_rate"]
        scv   = feats["schema_sizing_cv"]
        bgent = feats["schema_action_bigram_ent"]
        compl = feats["schema_action_complexity"]
        snap_high_count       += int(snap > 0.50)
        snap_consistent_count += int(snap > 0.60)
        snap_and_low_cv_count += int(snap > 0.40 and scv < 0.25)
        low_bigram_ent_count  += int(bgent < 0.40)
        diverse_bigram_count  += int(bgent > 0.70)
        high_complexity_count += int(compl > 0.60)
        reraise_hand_count    += int(feats["schema_reraise_rate"] > 0.0)
        check_raise_proxy_count += int(
            feats["schema_fold_to_aggression_rate"] == 0.0
            and feats["schema_reraise_rate"] > 0.0
        )
        high_flop_reach_count += int(feats["schema_flop_reach_rate"] == 1.0)
        varied_btp_count      += int(scv > 0.30)

        # chunk-level btp collection (for IQR / snap concentration)
        for a in actions:
            if not isinstance(a, dict):
                continue
            at  = str(a.get("action_type") or "").lower().strip()
            amt = max(0.0, _safe_float(a.get("normalized_amount_bb"), 0.0))
            pb  = max(0.0, _safe_div(
                _safe_float(a.get("pot_before"), 0.0), 0.02
            ))
            if at in ("bet", "raise") and pb > 0.0 and amt > 0.0:
                ratio = amt / pb
                all_btp_chunk.append(ratio)
                snap_labels_chunk.append(_nearest_snap_label(ratio))

        # chunk-level bigrams
        if len(action_types) >= 2:
            chunk_bigrams.extend(zip(action_types, action_types[1:]))

    n = float(len(chunk))

    # ── write v2 chunk-level signature features (identical values) ─────────────
    out["schema_action_signature_top_share"]   = _safe_div(max(Counter(action_signatures).values()), n)
    out["schema_action_signature_unique_share"]= _safe_div(len(set(action_signatures)), n)
    out["schema_actor_signature_top_share"]    = _safe_div(max(Counter(actor_signatures).values()), n)
    out["schema_actor_signature_unique_share"] = _safe_div(len(set(actor_signatures)), n)
    out["schema_street_signature_top_share"]   = _safe_div(max(Counter(street_signatures).values()), n)
    out["schema_street_signature_unique_share"]= _safe_div(len(set(street_signatures)), n)
    out["schema_amount_bucket_signature_top_share"]   = _safe_div(max(Counter(amount_bucket_signatures).values()), n)
    out["schema_amount_bucket_signature_unique_share"]= _safe_div(len(set(amount_bucket_signatures)), n)
    out["schema_high_aggression_hand_rate"]    = _safe_div(high_aggressive,    n)
    out["schema_low_action_entropy_hand_rate"] = _safe_div(low_action_entropy, n)
    out["schema_high_actor_entropy_hand_rate"] = _safe_div(high_actor_entropy, n)
    out["schema_long_action_hand_rate"]        = _safe_div(long_action_hand,   n)

    # v2 trend / half / lag features
    agg_series = per_hand_series["schema_aggression_share"]
    ent_series = per_hand_series["schema_action_entropy"]
    btp_series = per_hand_series["schema_bet_to_pot_ratio"]
    cnt_series = per_hand_series["schema_action_count"]

    out["schema_aggression_trend"]      = _linear_slope(agg_series)
    out["schema_action_entropy_trend"]  = _linear_slope(ent_series)
    out["schema_bet_to_pot_trend"]      = _linear_slope(btp_series)

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

    out["schema_aggression_lag1_corr"]   = _lag1_corr(agg_series)
    out["schema_bet_to_pot_lag1_corr"]   = _lag1_corr(btp_series)
    out["schema_action_count_lag1_corr"] = _lag1_corr(cnt_series)

    out["schema_bet_size_bucket_top_share"]    = _safe_div(max(Counter(bet_size_buckets_per_hand).values()), n)
    out["schema_bet_size_bucket_unique_share"] = _safe_div(len(set(bet_size_buckets_per_hand)), n)

    out["schema_very_high_aggression_hand_rate"]  = _safe_div(very_high_agg_count,  n)
    out["schema_passive_hand_rate"]               = _safe_div(passive_hand_count,   n)
    out["schema_short_hand_rate"]                 = _safe_div(short_hand_count,     n)
    out["schema_consistent_sizing_hand_rate"]     = _safe_div(consistent_sizing,    n)

    # ── NEW v3 chunk-level features ────────────────────────────────────────────

    # A. Cross-hand consistency (low = bot-like)
    out["schema_hand_length_cv"] = _safe_div(
        _std(cnt_series), max(_mean(cnt_series), 1e-9)
    )
    out["schema_cross_hand_btp_std"]    = _std(btp_series)
    out["schema_bet_count_cv"]          = _safe_div(
        _std(per_hand_series["schema_bet_count"]),
        max(_mean(per_hand_series["schema_bet_count"]), 1e-9),
    )
    out["schema_flop_reach_cv"]         = _safe_div(
        _std(per_hand_series["schema_flop_reach_rate"]),
        max(_mean(per_hand_series["schema_flop_reach_rate"]), 1e-9),
    )
    out["schema_river_reach_cv"]        = _safe_div(
        _std(per_hand_series["schema_river_reach_rate"]),
        max(_mean(per_hand_series["schema_river_reach_rate"]), 1e-9),
    )
    out["schema_action_complexity_cv"]  = _safe_div(
        _std(per_hand_series["schema_action_complexity"]),
        max(_mean(per_hand_series["schema_action_complexity"]), 1e-9),
    )

    # B. Bet-size snapping
    out["schema_snap_high_rate"]       = _safe_div(snap_high_count,       n)
    out["schema_snap_consistent_rate"] = _safe_div(snap_consistent_count, n)
    out["schema_bet_size_iqr"]         = (
        _quantile(all_btp_chunk, 0.75) - _quantile(all_btp_chunk, 0.25)
        if all_btp_chunk else 0.0
    )
    if snap_labels_chunk:
        top_snap = max(Counter(snap_labels_chunk).values())
        out["schema_bet_size_snap_concentration"] = _safe_div(top_snap, len(snap_labels_chunk))
    else:
        out["schema_bet_size_snap_concentration"] = 0.0

    # C. Action-bigram entropy
    out["schema_chunk_bigram_entropy"]   = _entropy(chunk_bigrams) if chunk_bigrams else 0.0
    out["schema_low_bigram_entropy_rate"]= _safe_div(low_bigram_ent_count, n)

    # D. Compound bot index  ∈ [0, 1]
    snap_mean  = _mean(per_hand_series["schema_bet_snap_rate"])
    scv_mean   = _mean(per_hand_series["schema_sizing_cv"])
    compl_mean = _mean(per_hand_series["schema_action_complexity"])
    out["schema_compound_bot_index"] = (
        snap_mean * max(0.0, 1.0 - scv_mean) * max(0.0, 1.0 - compl_mean)
    )
    out["schema_snap_and_low_cv_rate"] = _safe_div(snap_and_low_cv_count, n)

    # E. Reach-rate delta consistency
    reach_delta = [
        abs(f - r)
        for f, r in zip(
            per_hand_series["schema_flop_reach_rate"],
            per_hand_series["schema_river_reach_rate"],
        )
    ]
    out["schema_reach_delta_cv"] = _safe_div(
        _std(reach_delta), max(_mean(reach_delta), 1e-9)
    )

    # F. FPR-guard (human-positive) rates
    out["schema_reraise_hand_rate"]            = _safe_div(reraise_hand_count,      n)
    out["schema_check_raise_hand_rate"]        = _safe_div(check_raise_proxy_count, n)
    out["schema_high_action_complexity_rate"]  = _safe_div(high_complexity_count,   n)
    out["schema_high_flop_reach_rate"]         = _safe_div(high_flop_reach_count,   n)
    out["schema_varied_btp_hand_rate"]         = _safe_div(varied_btp_count,        n)
    out["schema_diverse_bigram_hand_rate"]     = _safe_div(diverse_bigram_count,    n)

    return out


# ── compatibility shim ────────────────────────────────────────────────────────

def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    """Alias → chunk_features_v3 (drop-in replacement for features_v2.py)."""
    return chunk_features_v3(chunk)
