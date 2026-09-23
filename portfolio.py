"""Bounded beam search over campaign order, public resources and overlap."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Plan:
    value: float
    budget: float
    contacts: int
    chosen: tuple
    credited: np.ndarray


def optimize_portfolio(beliefs, env, beam_width=48):
    profile = env.customer_profile.sort_values("ID_NUMBER", kind="stable").reset_index(drop=True)
    arpu = profile.predicted_arpu.to_numpy(dtype=float)
    candidates = []
    for belief in sorted(beliefs, key=lambda item: item.candidate_id):
        if belief.blocked or belief.pilot_count == 0:
            continue
        if belief.target_tariff not in set(env.tariffs.tariff_plan_code):
            continue
        if belief.channel not in env.channels or belief.target_tariff == belief.current_tariff:
            continue
        indices = np.flatnonzero(
            (profile.current_tariff == belief.current_tariff)
            & (profile.arpu_segment == belief.arpu_segment)
        )[:5000]
        cost = float(env.channels[belief.channel]["cost_per_contact"])
        candidates.append((belief, indices, cost))

    # Pilot IDs are private. Use expected random coverage, never reconstruct IDs.
    # This estimates overlap; exact final-to-final overlap below is deterministic.
    pilot_credit = np.zeros(len(profile))
    for belief, _, _ in candidates:
        indices = np.flatnonzero(
            (profile.current_tariff == belief.current_tariff)
            & (profile.arpu_segment == belief.arpu_segment)
        )
        if not len(indices):
            continue
        coverage = min(1.0, belief.pilot_contacts / len(indices))
        conservative = max(0.0, belief.posterior_mean - 1.64 * belief.posterior_std)
        pilot_credit[indices] = np.maximum(pilot_credit[indices], coverage * conservative)

    initial = Plan(0.0, float(env.remaining_budget), int(env.remaining_contacts), (), pilot_credit)
    frontier = [initial]
    best = initial
    for _ in range(min(10, len(candidates))):
        extensions = []
        for plan in frontier:
            for index, (belief, all_indices, cost) in enumerate(candidates):
                if index in plan.chosen:
                    continue
                count = min(len(all_indices), plan.contacts)
                if cost > 0:
                    count = min(count, int(plan.budget // cost))
                if count <= 0:
                    continue
                indices = all_indices[:count]
                spend = count * cost
                z = 1.96 if spend > 0.5 * plan.budget else 1.64
                ratio = belief.posterior_mean - z * belief.posterior_std
                gain = float((arpu[indices] * np.maximum(ratio - plan.credited[indices], 0)).sum()) - spend
                if gain <= 0 or ratio <= 0:
                    continue
                credit = plan.credited.copy()
                credit[indices] = np.maximum(credit[indices], ratio)
                extensions.append(Plan(plan.value + gain, plan.budget - spend,
                                       plan.contacts - count, plan.chosen + (index,), credit))
        if not extensions:
            break
        extensions.sort(key=lambda plan: (-plan.value, plan.chosen))
        frontier = extensions[:beam_width]
        if frontier[0].value > best.value:
            best = frontier[0]
    if not best.chosen:
        return []
    return [campaign(candidates[index][0], order + 1) for order, index in enumerate(best.chosen)]


def campaign(belief, order):
    return {"campaign_name": f"portfolio_{order}_{belief.current_tariff}_{belief.target_tariff}",
            "filter_current_tariff": belief.current_tariff,
            "filter_arpu_segment": belief.arpu_segment,
            "target_tariff": belief.target_tariff, "channel": belief.channel}


def fallback_campaign(env):
    """Smallest valid free audience limits downside when evidence is weak."""
    tariffs = sorted(set(env.tariffs.tariff_plan_code))
    if len(tariffs) < 2 or "push" not in env.channels:
        raise ValueError("No valid free cross-tariff fallback available")
    profile = env.customer_profile
    columns = ["current_tariff", "arpu_segment", "data_segment", "call_segment"]
    columns = [column for column in columns if column in profile]
    groups = profile[profile.current_tariff.isin(tariffs)].groupby(columns, observed=True).size()
    if groups.empty:
        raise ValueError("No valid audience for fallback")
    key = min(groups.index, key=lambda key: (groups[key], key))
    filters = dict(zip(columns, key if isinstance(key, tuple) else (key,)))
    target = next(code for code in tariffs if code != filters["current_tariff"])
    return [{"campaign_name": "conservative_fallback", "target_tariff": target,
             "channel": "push", **{f"filter_{key}": value for key, value in filters.items()}}]
