"""Profile the supplied Beeline data and build conservative transition priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARPU_BINS = [-np.inf, 1_000, 5_000, np.inf]
ARPU_LABELS = ["LOW", "MID", "HIGH"]
CORE_TARIFF_COLUMNS = [
    "Data_in_PKG",
    "Min_another_operator_in_PKG",
    "Min_another_operator_and_city_in_PKG",
    "price_tariff",
    "tariff_plan_code",
]


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def duplicate_key_diagnostics(frame: pd.DataFrame, keys: list[str]) -> dict[str, int]:
    duplicate_rows = frame.loc[frame.duplicated(keys, keep=False)]
    duplicate_groups = duplicate_rows.groupby(keys, dropna=False)
    conflicting_groups = 0
    value_columns = [column for column in frame.columns if column not in keys]
    for _, rows in duplicate_groups:
        if any(rows[column].nunique(dropna=False) > 1 for column in value_columns):
            conflicting_groups += 1
    return {
        "duplicate_key_groups": int(duplicate_groups.ngroups),
        "duplicate_extra_rows": int(frame.duplicated(keys).sum()),
        "exact_duplicate_extra_rows": int(frame.duplicated().sum()),
        "conflicting_duplicate_key_groups": conflicting_groups,
    }


def read_inputs(root: Path) -> dict[str, pd.DataFrame]:
    frames = {
        "profile": pd.read_csv(root / "customer_profile.csv"),
        "changes": pd.read_csv(root / "data" / "change_tariff.csv"),
        "traffic": pd.read_csv(root / "data" / "traffic.csv"),
        "arpu": pd.read_csv(root / "data" / "arpu_monthly.csv"),
        "tariffs": pd.read_csv(root / "data" / "dict_tariff.csv"),
        "tariff_dictionary": pd.read_csv(root / "tariff_dictionary.csv"),
    }
    require_columns(
        frames["profile"],
        {
            "ID_NUMBER",
            "current_tariff",
            "ARPU_3m_avg",
            "predicted_arpu",
            "arpu_segment",
            "data_segment",
            "call_segment",
        },
        "customer_profile.csv",
    )
    require_columns(
        frames["changes"],
        {
            "ID_NUMBER",
            "TIME_KEY",
            "AVG_ARPU_PREV_3M",
            "AVG_ARPU_NEXT_3M",
            "tariff_plan_code_from",
            "tariff_plan_code_to",
        },
        "data/change_tariff.csv",
    )
    require_columns(frames["tariffs"], set(CORE_TARIFF_COLUMNS), "data/dict_tariff.csv")
    return frames


def build_transition_priors(changes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    deduplicated = changes.drop_duplicates().copy()
    tariff_codes = set(deduplicated["tariff_plan_code_from"]) | set(
        deduplicated["tariff_plan_code_to"]
    )
    possible_groups = len(tariff_codes) * (len(tariff_codes) - 1) * len(ARPU_LABELS)
    usable = deduplicated.loc[deduplicated["AVG_ARPU_PREV_3M"] >= 100].copy()
    usable["arpu_segment"] = pd.cut(
        usable["AVG_ARPU_PREV_3M"], bins=ARPU_BINS, labels=ARPU_LABELS
    ).astype("object")
    usable["arpu_change_ratio"] = (
        (usable["AVG_ARPU_NEXT_3M"] - usable["AVG_ARPU_PREV_3M"])
        / usable["AVG_ARPU_PREV_3M"]
    ).clip(-1, 3)
    usable["direction"] = np.select(
        [usable["arpu_change_ratio"] > 0.10, usable["arpu_change_ratio"] < -0.10],
        ["UPSELL", "DOWNSELL"],
        default="FLAT",
    )

    keys = ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"]
    grouped = (
        usable.groupby(keys, observed=True)
        .agg(
            sample_size=("ID_NUMBER", "size"),
            raw_mean_lift=("arpu_change_ratio", "mean"),
            raw_std_lift=("arpu_change_ratio", "std"),
            median_lift=("arpu_change_ratio", "median"),
            downsell_rate=("direction", lambda values: float((values == "DOWNSELL").mean())),
            flat_rate=("direction", lambda values: float((values == "FLAT").mean())),
            upsell_rate=("direction", lambda values: float((values == "UPSELL").mean())),
        )
        .reset_index()
    )

    origin_totals = (
        grouped.groupby(["tariff_plan_code_from", "arpu_segment"], observed=True)[
            "sample_size"
        ]
        .sum()
        .rename("origin_segment_changes")
        .reset_index()
    )
    grouped = grouped.merge(origin_totals, on=["tariff_plan_code_from", "arpu_segment"])
    grouped["destination_share_among_changers"] = (
        grouped["sample_size"] / grouped["origin_segment_changes"]
    )

    segment_rows: list[pd.DataFrame] = []
    hyperparameters: dict[str, dict[str, float]] = {}
    for segment, rows in grouped.groupby("arpu_segment", observed=True):
        observations = usable.loc[usable["arpu_segment"] == segment, "arpu_change_ratio"]
        prior_mean = float(observations.mean())
        fallback_variance = float(max(observations.var(ddof=1), 1e-6))
        raw_variance = rows["raw_std_lift"].pow(2).fillna(fallback_variance)
        mean_variance = raw_variance / rows["sample_size"].clip(lower=1)
        between_variance = float(rows["raw_mean_lift"].var(ddof=1))
        if not np.isfinite(between_variance):
            between_variance = 0.0
        tau2 = max(between_variance - float(mean_variance.mean()), 1e-6)

        precision_data = 1.0 / mean_variance.clip(lower=1e-9)
        precision_prior = 1.0 / tau2
        posterior_variance = 1.0 / (precision_data + precision_prior)
        posterior_mean = posterior_variance * (
            precision_data * rows["raw_mean_lift"] + precision_prior * prior_mean
        )

        enriched = rows.copy()
        enriched["segment_prior_mean"] = prior_mean
        enriched["between_group_variance"] = tau2
        enriched["posterior_mean_lift"] = posterior_mean
        enriched["posterior_std_lift"] = np.sqrt(posterior_variance)
        enriched["posterior_lower_95"] = posterior_mean - 1.96 * np.sqrt(posterior_variance)
        enriched["posterior_upper_95"] = posterior_mean + 1.96 * np.sqrt(posterior_variance)
        enriched["data_weight"] = precision_data / (precision_data + precision_prior)
        segment_rows.append(enriched)
        hyperparameters[str(segment)] = {
            "prior_mean": prior_mean,
            "between_group_variance": tau2,
            "usable_rows": int(len(observations)),
        }

    priors = pd.concat(segment_rows, ignore_index=True)
    priors = priors.sort_values(keys, kind="stable").reset_index(drop=True)
    numeric_columns = priors.select_dtypes(include=["number"]).columns
    priors[numeric_columns] = priors[numeric_columns].round(8)

    if int(priors["sample_size"].sum()) != len(usable):
        raise ValueError("Transition group counts do not reconcile to usable rows")
    if not np.isfinite(
        priors[
            [
                "posterior_mean_lift",
                "posterior_std_lift",
                "posterior_lower_95",
                "posterior_upper_95",
                "data_weight",
            ]
        ].to_numpy()
    ).all():
        raise ValueError("Transition priors contain non-finite posterior values")
    if not priors["data_weight"].between(0, 1).all():
        raise ValueError("Transition prior data weights must be between zero and one")
    if not np.allclose(
        priors[["downsell_rate", "flat_rate", "upsell_rate"]].sum(axis=1), 1
    ):
        raise ValueError("Transition direction rates do not sum to one")

    direction_counts = usable["direction"].value_counts().to_dict()
    summary = {
        "rows_total": int(len(changes)),
        "rows_after_exact_deduplication": int(len(deduplicated)),
        "rows_usable": int(len(usable)),
        "rows_excluded_prev_arpu_below_100": int(len(deduplicated) - len(usable)),
        "transition_groups": int(len(priors)),
        "possible_ordered_cross_tariff_groups": possible_groups,
        "observed_group_coverage": float(len(priors) / possible_groups),
        "groups_sample_size_lt_5": int((priors["sample_size"] < 5).sum()),
        "groups_sample_size_lt_20": int((priors["sample_size"] < 20).sum()),
        "sample_size_min": int(priors["sample_size"].min()),
        "sample_size_median": float(priors["sample_size"].median()),
        "sample_size_max": int(priors["sample_size"].max()),
        "direction_counts": {str(k): int(v) for k, v in direction_counts.items()},
        "hyperparameters": hyperparameters,
    }
    return priors, summary


def profile_data(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    profile = frames["profile"]
    changes = frames["changes"]
    traffic = frames["traffic"]
    arpu = frames["arpu"]
    tariffs = frames["tariffs"]
    tariff_dictionary = frames["tariff_dictionary"]
    known_tariffs = set(tariffs["tariff_plan_code"])

    tariff_comparison = tariffs[CORE_TARIFF_COLUMNS].sort_values("tariff_plan_code").reset_index(drop=True)
    dictionary_comparison = (
        tariff_dictionary[CORE_TARIFF_COLUMNS]
        .sort_values("tariff_plan_code")
        .reset_index(drop=True)
    )
    if not tariff_comparison.equals(dictionary_comparison):
        raise ValueError("Tariff dictionaries disagree on core fields")

    expected_segments = {
        "arpu_segment": {"LOW", "MID", "HIGH"},
        "data_segment": {"NON_USER", "LITE", "HEAVY"},
        "call_segment": {"LOW", "MEDIUM", "HIGH"},
    }
    invalid_segments = {
        column: sorted(set(profile[column].dropna()) - allowed)
        for column, allowed in expected_segments.items()
    }
    invalid_segments = {key: values for key, values in invalid_segments.items() if values}

    filter_group_sizes = {}
    for label, columns in {
        "arpu": ["arpu_segment"],
        "current_tariff": ["current_tariff"],
        "current_tariff_arpu": ["current_tariff", "arpu_segment"],
        "all_supported_filters": [
            "current_tariff",
            "arpu_segment",
            "data_segment",
            "call_segment",
        ],
    }.items():
        sizes = profile.groupby(columns, observed=True, dropna=False).size()
        filter_group_sizes[label] = {
            "groups": int(len(sizes)),
            "min": int(sizes.min()),
            "median": float(sizes.median()),
            "max": int(sizes.max()),
            "groups_over_campaign_limit": int((sizes > 5_000).sum()),
        }
    profile_ids = set(profile["ID_NUMBER"])
    change_ids = set(changes["ID_NUMBER"])
    traffic_ids = set(traffic["ID_NUMBER"])
    arpu_ids = set(arpu["ID_NUMBER"])

    duplicate_products = (
        tariffs.groupby(CORE_TARIFF_COLUMNS[:-1], dropna=False)["tariff_plan_code"]
        .agg(list)
        .loc[lambda values: values.str.len() > 1]
        .tolist()
    )
    profile_tariffs = set(profile["current_tariff"].dropna())
    history_tariffs = set(changes["tariff_plan_code_from"]) | set(
        changes["tariff_plan_code_to"]
    )

    expected_arpu = pd.cut(
        profile["ARPU_3m_avg"], bins=ARPU_BINS, labels=ARPU_LABELS
    ).astype("object")
    expected_data = profile["DATA_VOLUME"].map(
        lambda value: np.nan
        if pd.isna(value)
        else "NON_USER"
        if value == 0
        else "LITE"
        if value <= 2_000
        else "HEAVY"
    )
    call_minutes = profile["OUT_LOC_ONNET_MIN"] + profile["OUT_LOC_OFFNET_MIN"]
    expected_call = call_minutes.map(
        lambda value: np.nan
        if pd.isna(value)
        else "LOW"
        if value < 100
        else "MEDIUM"
        if value <= 400
        else "HIGH"
    )

    def mismatch_count(expected: pd.Series, actual: pd.Series) -> int:
        comparable = expected.notna()
        return int((expected.loc[comparable] != actual.loc[comparable]).sum())

    return {
        "profile": {
            "rows": int(len(profile)),
            "unique_ids": int(profile["ID_NUMBER"].nunique()),
            "duplicate_ids": int(profile["ID_NUMBER"].duplicated().sum()),
            "baseline_predicted_arpu": float(profile["predicted_arpu"].sum()),
            "missing_by_column": {
                key: int(value)
                for key, value in profile.isna().sum().items()
                if value
            },
            "invalid_segments": invalid_segments,
            "zero_predicted_arpu": int((profile["predicted_arpu"] == 0).sum()),
            "negative_predicted_arpu": int((profile["predicted_arpu"] < 0).sum()),
            "segment_mismatches_on_nonmissing_inputs": {
                "arpu_segment": mismatch_count(expected_arpu, profile["arpu_segment"]),
                "data_segment": mismatch_count(expected_data, profile["data_segment"]),
                "call_segment": mismatch_count(expected_call, profile["call_segment"]),
            },
            "filter_group_sizes": filter_group_sizes,
            "unknown_current_tariffs": sorted(profile_tariffs - known_tariffs),
        },
        "history": {
            "change_rows": int(len(changes)),
            "change_unique_ids": int(changes["ID_NUMBER"].nunique()),
            "change_duplicates": duplicate_key_diagnostics(
                changes, ["ID_NUMBER", "TIME_KEY"]
            ),
            "traffic_rows": int(len(traffic)),
            "traffic_unique_ids": int(traffic["ID_NUMBER"].nunique()),
            "traffic_duplicates": duplicate_key_diagnostics(
                traffic, ["ID_NUMBER", "time_key"]
            ),
            "arpu_rows": int(len(arpu)),
            "arpu_unique_ids": int(arpu["ID_NUMBER"].nunique()),
            "arpu_duplicates": duplicate_key_diagnostics(
                arpu, ["ID_NUMBER", "TIME_KEY"]
            ),
            "change_time_min": str(changes["TIME_KEY"].min()),
            "change_time_max": str(changes["TIME_KEY"].max()),
            "traffic_time_min": str(traffic["time_key"].min()),
            "traffic_time_max": str(traffic["time_key"].max()),
            "arpu_time_min": str(arpu["TIME_KEY"].min()),
            "arpu_time_max": str(arpu["TIME_KEY"].max()),
            "unknown_history_tariffs": sorted(history_tariffs - known_tariffs),
        },
        "joins": {
            "profile_ids_in_changes": int(len(profile_ids & change_ids)),
            "profile_ids_in_traffic": int(len(profile_ids & traffic_ids)),
            "profile_ids_in_arpu": int(len(profile_ids & arpu_ids)),
            "change_ids_in_traffic": int(len(change_ids & traffic_ids)),
            "change_ids_in_arpu": int(len(change_ids & arpu_ids)),
            "traffic_ids_in_arpu": int(len(traffic_ids & arpu_ids)),
        },
        "tariffs": {
            "rows": int(len(tariffs)),
            "unique_codes": int(tariffs["tariff_plan_code"].nunique()),
            "core_dictionaries_equal": True,
            "duplicate_product_groups": duplicate_products,
            "price_min": float(tariffs["price_tariff"].min()),
            "price_median": float(tariffs["price_tariff"].median()),
            "price_max": float(tariffs["price_tariff"].max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--priors-output", type=Path, default=Path("analysis/transition_priors.csv")
    )
    args = parser.parse_args()

    frames = read_inputs(args.root)
    priors, transition_summary = build_transition_priors(frames["changes"])
    summary = profile_data(frames)
    summary["transitions"] = transition_summary

    output = args.root / args.priors_output
    output.parent.mkdir(parents=True, exist_ok=True)
    priors.to_csv(output, index=False, lineterminator="\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
