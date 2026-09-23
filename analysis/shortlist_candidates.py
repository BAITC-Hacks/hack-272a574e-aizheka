"""Build a deterministic, uncertainty-aware campaign candidate shortlist."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ARPU_SEGMENTS = ("LOW", "MID", "HIGH")
CHANNELS = {
    "push": {"cost_per_contact": 0, "conversion_multiplier": 0.50},
    "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65},
    "digital_ads": {"cost_per_contact": 22, "conversion_multiplier": 0.85},
    "call": {"cost_per_contact": 160, "conversion_multiplier": 1.20},
}
MAX_CUSTOMERS_PER_CAMPAIGN = 5_000
MAX_TOTAL_CONTACTS = 15_000
TOTAL_BUDGET = 100_000
DEFAULT_SHORTLIST_SIZE = 32

PROFILE_COLUMNS = {
    "ID_NUMBER",
    "current_tariff",
    "arpu_segment",
    "predicted_arpu",
}
PRIOR_COLUMNS = {
    "tariff_plan_code_from",
    "tariff_plan_code_to",
    "arpu_segment",
    "sample_size",
    "posterior_mean_lift",
    "posterior_std_lift",
}
PRIOR_KEY = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _contact_ceiling(cost_per_contact: float) -> int:
    limit = min(MAX_CUSTOMERS_PER_CAMPAIGN, MAX_TOTAL_CONTACTS)
    if cost_per_contact > 0:
        limit = min(limit, int(TOTAL_BUDGET // cost_per_contact))
    return limit


def build_candidate_universe(
    profile: pd.DataFrame,
    priors: pd.DataFrame,
    tariff_codes: Iterable[str],
    channels: Mapping[str, Mapping[str, float]] = CHANNELS,
) -> pd.DataFrame:
    """Enumerate valid filter/target/channel combinations and attach prior proxies.

    Historical lift describes people who changed tariff, not campaign response.
    The value fields therefore compare candidates on a common prior scale; they
    are deliberately named ``proxy`` and must be updated from actual pilots.
    """

    _require_columns(profile, PROFILE_COLUMNS, "customer profile")
    _require_columns(priors, PRIOR_COLUMNS, "transition priors")

    known_tariffs = tuple(sorted({str(code) for code in tariff_codes}))
    if not known_tariffs:
        raise ValueError("At least one tariff code is required")
    if priors.duplicated(PRIOR_KEY).any():
        raise ValueError("Transition priors contain duplicate keys")

    working = profile.loc[
        profile["current_tariff"].isin(known_tariffs)
        & profile["arpu_segment"].isin(ARPU_SEGMENTS),
        list(PROFILE_COLUMNS),
    ].copy()
    working["predicted_arpu"] = pd.to_numeric(working["predicted_arpu"], errors="coerce")
    if working.empty:
        raise ValueError("No profile rows have a known tariff and supported ARPU segment")
    if not np.isfinite(working["predicted_arpu"]).all():
        raise ValueError("predicted_arpu must be finite for eligible profile rows")
    if (working["predicted_arpu"] < 0).any():
        raise ValueError("predicted_arpu must be nonnegative")

    prior_lookup = priors.set_index(PRIOR_KEY)
    records: list[dict[str, object]] = []
    grouped = working.groupby(["current_tariff", "arpu_segment"], observed=True)

    for (origin, arpu_segment), audience in sorted(grouped, key=lambda item: item[0]):
        audience = audience.sort_values("ID_NUMBER", kind="stable")
        for target in known_tariffs:
            if target == origin:
                continue
            prior_key = (origin, target, arpu_segment)
            prior = prior_lookup.loc[prior_key] if prior_key in prior_lookup.index else None

            for channel, channel_config in channels.items():
                cost_per_contact = float(channel_config["cost_per_contact"])
                multiplier = float(channel_config["conversion_multiplier"])
                contacted = audience.iloc[: _contact_ceiling(cost_per_contact)]
                contacts = int(len(contacted))
                modeled_arpu = float(contacted["predicted_arpu"].sum())
                campaign_cost = float(contacts * cost_per_contact)

                record: dict[str, object] = {
                    "candidate_id": f"{origin}__{arpu_segment}__{target}__{channel}",
                    "campaign_name": f"{origin}_{arpu_segment}_to_{target}_{channel}",
                    "filter_current_tariff": origin,
                    "filter_arpu_segment": arpu_segment,
                    "filter_data_segment": pd.NA,
                    "filter_call_segment": pd.NA,
                    "target_tariff": target,
                    "channel": channel,
                    "audience_size": int(len(audience)),
                    "modeled_contacts": contacts,
                    "modeled_arpu": modeled_arpu,
                    "cost_per_contact": cost_per_contact,
                    "campaign_cost": campaign_cost,
                    "channel_multiplier": multiplier,
                    "prior_available": prior is not None,
                    "sample_size": np.nan,
                    "posterior_mean_lift": np.nan,
                    "posterior_std_lift": np.nan,
                    "prior_gross_value_proxy": np.nan,
                    "prior_net_value_proxy": np.nan,
                    "uncertainty_value_proxy": np.nan,
                    "lower_net_value_proxy": np.nan,
                    "upper_net_value_proxy": np.nan,
                    "priority_score": np.nan,
                }

                if prior is not None:
                    mean_lift = float(prior["posterior_mean_lift"])
                    std_lift = float(prior["posterior_std_lift"])
                    gross_proxy = modeled_arpu * mean_lift * multiplier
                    uncertainty_proxy = modeled_arpu * std_lift * multiplier
                    net_proxy = gross_proxy - campaign_cost
                    record.update(
                        {
                            "sample_size": int(prior["sample_size"]),
                            "posterior_mean_lift": mean_lift,
                            "posterior_std_lift": std_lift,
                            "prior_gross_value_proxy": gross_proxy,
                            "prior_net_value_proxy": net_proxy,
                            "uncertainty_value_proxy": uncertainty_proxy,
                            "lower_net_value_proxy": net_proxy - uncertainty_proxy,
                            "upper_net_value_proxy": net_proxy + uncertainty_proxy,
                            "priority_score": net_proxy + 0.20 * uncertainty_proxy,
                        }
                    )
                records.append(record)

    universe = pd.DataFrame.from_records(records)
    numeric_columns = universe.select_dtypes(include=["number"]).columns
    universe[numeric_columns] = universe[numeric_columns].round(8)
    return universe.sort_values("candidate_id", kind="stable").reset_index(drop=True)


def _transition_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row["filter_current_tariff"]),
        str(row["target_tariff"]),
        str(row["filter_arpu_segment"]),
    )


def select_diverse_shortlist(
    universe: pd.DataFrame,
    shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
) -> pd.DataFrame:
    """Select exploitation and exploration candidates with deterministic diversity."""

    if not 20 <= shortlist_size <= 40:
        raise ValueError("shortlist_size must be between 20 and 40")

    eligible = universe.loc[
        universe["prior_available"]
        & np.isfinite(universe["prior_net_value_proxy"])
        & (universe["modeled_contacts"] >= 10)
        & (universe["upper_net_value_proxy"] > 0)
    ].copy()
    if eligible.empty:
        raise ValueError("No candidates have a usable prior and positive upper value proxy")
    if len(eligible.drop_duplicates(
        ["filter_current_tariff", "target_tariff", "filter_arpu_segment"]
    )) < shortlist_size:
        raise ValueError("Not enough distinct plausible transitions for the requested shortlist")

    selected: list[pd.Series] = []
    selected_ids: set[str] = set()
    selected_transitions: set[tuple[str, str, str]] = set()
    counts: dict[str, Counter[str]] = {
        "origin": Counter(),
        "target": Counter(),
        "segment": Counter(),
        "channel": Counter(),
    }

    def add(row: pd.Series, reason: str, caps: dict[str, int] | None = None) -> bool:
        candidate_id = str(row["candidate_id"])
        transition = _transition_key(row)
        dimensions = {
            "origin": str(row["filter_current_tariff"]),
            "target": str(row["target_tariff"]),
            "segment": str(row["filter_arpu_segment"]),
            "channel": str(row["channel"]),
        }
        if candidate_id in selected_ids or transition in selected_transitions:
            return False
        if caps and any(counts[name][value] >= caps[name] for name, value in dimensions.items()):
            return False
        picked = row.copy()
        picked["selection_reason"] = reason
        selected.append(picked)
        selected_ids.add(candidate_id)
        selected_transitions.add(transition)
        for name, value in dimensions.items():
            counts[name][value] += 1
        return True

    value_order = eligible.sort_values(
        ["prior_net_value_proxy", "sample_size", "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    uncertainty_order = eligible.sort_values(
        ["uncertainty_value_proxy", "upper_net_value_proxy", "candidate_id"],
        ascending=[False, False, True],
        kind="stable",
    )

    for segment in ARPU_SEGMENTS:
        for _, row in value_order.loc[value_order["filter_arpu_segment"] == segment].iterrows():
            if add(row, "coverage_arpu"):
                break
    for channel in CHANNELS:
        for _, row in value_order.loc[value_order["channel"] == channel].iterrows():
            if add(row, "coverage_channel"):
                break

    exploration_count = max(6, round(shortlist_size * 0.25))
    exploitation_target = shortlist_size - exploration_count
    value_caps = {"origin": 4, "target": 4, "segment": 10, "channel": 10}
    for _, row in value_order.iterrows():
        if len(selected) >= exploitation_target:
            break
        add(row, "expected_value", value_caps)

    exploration_caps = {"origin": 5, "target": 5, "segment": 12, "channel": 12}
    for _, row in uncertainty_order.iterrows():
        if len(selected) >= shortlist_size:
            break
        add(row, "uncertainty", exploration_caps)

    # The caps are diversity preferences, not validity rules. Relax them only if
    # necessary to produce the requested review set while retaining one channel
    # per origin/target/ARPU transition.
    if len(selected) < shortlist_size:
        fallback_order = eligible.sort_values(
            ["priority_score", "candidate_id"],
            ascending=[False, True],
            kind="stable",
        )
        for _, row in fallback_order.iterrows():
            if len(selected) >= shortlist_size:
                break
            add(row, "ranked_fill")

    shortlist = pd.DataFrame(selected).reset_index(drop=True)
    shortlist.insert(0, "selection_order", np.arange(1, len(shortlist) + 1))
    validate_shortlist(shortlist, set(universe["target_tariff"]) | set(universe["filter_current_tariff"]))
    return shortlist


def validate_shortlist(shortlist: pd.DataFrame, tariff_codes: set[str]) -> None:
    if not 20 <= len(shortlist) <= 40:
        raise ValueError("A shortlist must contain 20-40 candidates")
    if shortlist["candidate_id"].duplicated().any():
        raise ValueError("Candidate IDs must be unique")
    if shortlist.duplicated(
        ["filter_current_tariff", "target_tariff", "filter_arpu_segment"]
    ).any():
        raise ValueError("Only one channel may represent each transition in the shortlist")
    if not shortlist["filter_arpu_segment"].isin(ARPU_SEGMENTS).all():
        raise ValueError("Shortlist contains an unsupported ARPU segment")
    if not shortlist["channel"].isin(CHANNELS).all():
        raise ValueError("Shortlist contains an unsupported channel")
    if not shortlist["filter_current_tariff"].isin(tariff_codes).all():
        raise ValueError("Shortlist contains an unknown origin tariff")
    if not shortlist["target_tariff"].isin(tariff_codes).all():
        raise ValueError("Shortlist contains an unknown target tariff")
    if (shortlist["filter_current_tariff"] == shortlist["target_tariff"]).any():
        raise ValueError("Shortlist contains a no-op tariff transition")
    if not shortlist["prior_available"].all():
        raise ValueError("Every shortlisted candidate must have a historical prior")
    if not shortlist["modeled_contacts"].between(10, MAX_CUSTOMERS_PER_CAMPAIGN).all():
        raise ValueError("Shortlist violates campaign contact bounds")
    if (shortlist["campaign_cost"] > TOTAL_BUDGET).any():
        raise ValueError("A candidate exceeds the total monetary budget")
    if not np.isfinite(
        shortlist[
            [
                "posterior_mean_lift",
                "posterior_std_lift",
                "prior_net_value_proxy",
                "uncertainty_value_proxy",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Shortlist contains non-finite ranking values")


def summarize(universe: pd.DataFrame, shortlist: pd.DataFrame) -> dict[str, object]:
    plausible = universe.loc[
        universe["prior_available"] & (universe["upper_net_value_proxy"] > 0)
    ]
    return {
        "universe_candidates": int(len(universe)),
        "candidates_with_historical_prior": int(universe["prior_available"].sum()),
        "plausibly_positive_candidates": int(len(plausible)),
        "shortlist_size": int(len(shortlist)),
        "origin_tariffs": int(shortlist["filter_current_tariff"].nunique()),
        "target_tariffs": int(shortlist["target_tariff"].nunique()),
        "arpu_segments": sorted(shortlist["filter_arpu_segment"].unique().tolist()),
        "channels": sorted(shortlist["channel"].unique().tolist()),
        "selection_reasons": {
            str(key): int(value)
            for key, value in shortlist["selection_reason"].value_counts().sort_index().items()
        },
        "audience_size_min": int(shortlist["audience_size"].min()),
        "audience_size_max": int(shortlist["audience_size"].max()),
        "modeled_contacts_min": int(shortlist["modeled_contacts"].min()),
        "modeled_contacts_max": int(shortlist["modeled_contacts"].max()),
        "historical_sample_size_min": int(shortlist["sample_size"].min()),
        "historical_sample_size_median": float(shortlist["sample_size"].median()),
        "historical_sample_size_max": int(shortlist["sample_size"].max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--shortlist-size", type=int, default=DEFAULT_SHORTLIST_SIZE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "candidate_shortlist.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = pd.read_csv(args.root / "customer_profile.csv")
    priors = pd.read_csv(args.root / "analysis" / "transition_priors.csv")
    tariffs = pd.read_csv(args.root / "data" / "dict_tariff.csv")
    universe = build_candidate_universe(
        profile,
        priors,
        tariffs["tariff_plan_code"],
    )
    shortlist = select_diverse_shortlist(universe, args.shortlist_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(args.output, index=False)
    print(json.dumps(summarize(universe, shortlist), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
