from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from analysis.shortlist_candidates import (
    ARPU_SEGMENTS,
    CHANNELS,
    MAX_CUSTOMERS_PER_CAMPAIGN,
    TOTAL_BUDGET,
    build_candidate_universe,
    select_diverse_shortlist,
    validate_shortlist,
)


ROOT = Path(__file__).resolve().parents[1]


class CandidateUniverseTests(unittest.TestCase):
    def test_enumerates_cross_tariff_channels_and_applies_resource_caps(self) -> None:
        profile = pd.DataFrame(
            {
                "ID_NUMBER": range(1, 6_002),
                "current_tariff": ["tariff_1"] * 6_001,
                "arpu_segment": ["HIGH"] * 6_001,
                "predicted_arpu": [1_000.0] * 6_001,
            }
        )
        priors = pd.DataFrame(
            [
                {
                    "tariff_plan_code_from": "tariff_1",
                    "tariff_plan_code_to": "tariff_2",
                    "arpu_segment": "HIGH",
                    "sample_size": 25,
                    "posterior_mean_lift": 0.2,
                    "posterior_std_lift": 0.1,
                }
            ]
        )

        universe = build_candidate_universe(profile, priors, ["tariff_1", "tariff_2"])

        self.assertEqual(len(universe), 4)
        contacts = universe.set_index("channel")["modeled_contacts"].to_dict()
        self.assertEqual(contacts["push"], MAX_CUSTOMERS_PER_CAMPAIGN)
        self.assertEqual(contacts["sms"], MAX_CUSTOMERS_PER_CAMPAIGN)
        self.assertEqual(contacts["digital_ads"], 4_545)
        self.assertEqual(contacts["call"], 625)
        self.assertTrue((universe["campaign_cost"] <= TOTAL_BUDGET).all())
        self.assertTrue(universe["prior_available"].all())

    def test_unobserved_combinations_remain_visible_but_unranked(self) -> None:
        profile = pd.DataFrame(
            {
                "ID_NUMBER": range(20),
                "current_tariff": ["tariff_1"] * 20,
                "arpu_segment": ["MID"] * 20,
                "predicted_arpu": [2_000.0] * 20,
            }
        )
        priors = pd.DataFrame(
            columns=[
                "tariff_plan_code_from",
                "tariff_plan_code_to",
                "arpu_segment",
                "sample_size",
                "posterior_mean_lift",
                "posterior_std_lift",
            ]
        )

        universe = build_candidate_universe(profile, priors, ["tariff_1", "tariff_2"])

        self.assertEqual(len(universe), 4)
        self.assertFalse(universe["prior_available"].any())
        self.assertTrue(universe["prior_net_value_proxy"].isna().all())


class RealDataShortlistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        profile = pd.read_csv(ROOT / "customer_profile.csv")
        priors = pd.read_csv(ROOT / "analysis" / "transition_priors.csv")
        tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
        cls.tariff_codes = set(tariffs["tariff_plan_code"])
        cls.universe = build_candidate_universe(profile, priors, cls.tariff_codes)

    def test_default_shortlist_is_reproducible_valid_and_diverse(self) -> None:
        first = select_diverse_shortlist(self.universe)
        second = select_diverse_shortlist(self.universe)

        assert_frame_equal(first, second)
        validate_shortlist(first, self.tariff_codes)
        self.assertEqual(len(first), 32)
        self.assertEqual(set(first["filter_arpu_segment"]), set(ARPU_SEGMENTS))
        self.assertEqual(set(first["channel"]), set(CHANNELS))
        self.assertGreaterEqual(first["filter_current_tariff"].nunique(), 8)
        self.assertGreaterEqual(first["target_tariff"].nunique(), 8)

    def test_generated_filters_match_actual_audiences(self) -> None:
        shortlist = select_diverse_shortlist(self.universe)
        profile = pd.read_csv(ROOT / "customer_profile.csv")

        for row in shortlist.itertuples(index=False):
            audience = profile.loc[
                (profile["current_tariff"] == row.filter_current_tariff)
                & (profile["arpu_segment"] == row.filter_arpu_segment)
            ]
            self.assertEqual(len(audience), row.audience_size)
            self.assertLessEqual(row.modeled_contacts, len(audience))


if __name__ == "__main__":
    unittest.main()
