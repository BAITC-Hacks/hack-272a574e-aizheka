"""Deterministic preflight shared by numerical and model-assisted actions."""

from math import isfinite


def validate_pilot(env, belief, size):
    if isinstance(size, bool) or not isinstance(size, int) or not 10 <= size <= 200:
        raise ValueError("invalid_sample_size")
    if belief.target_tariff == belief.current_tariff:
        raise ValueError("no_op_transition")
    if belief.arpu_segment not in {"LOW", "MID", "HIGH"}:
        raise ValueError("invalid_arpu_filter")
    if hasattr(env, "tariffs") and belief.target_tariff not in set(env.tariffs.tariff_plan_code):
        raise ValueError("unknown_target")
    if hasattr(env, "channels"):
        if belief.channel not in env.channels:
            raise ValueError("unknown_channel")
        actual_cost = float(env.channels[belief.channel]["cost_per_contact"])
        if actual_cost != belief.cost_per_contact:
            raise ValueError("stale_channel_cost")
    cost = size * belief.cost_per_contact
    if not isfinite(cost) or cost < 0:
        raise ValueError("invalid_cost")
    if env.pilots_left <= 0 or size > env.remaining_contacts or cost > env.remaining_budget:
        raise ValueError("insufficient_resources")
    if size > belief.audience_size:
        raise ValueError("insufficient_audience")
    if hasattr(env, "customer_profile"):
        rows = env.customer_profile
        actual = rows[(rows.current_tariff == belief.current_tariff)
                      & (rows.arpu_segment == belief.arpu_segment)]
        if len(actual) != belief.audience_size:
            raise ValueError("stale_audience")
