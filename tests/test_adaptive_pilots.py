from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from adaptive_pilots import (
    MAX_ADAPTIVE_PILOTS,
    MAX_PILOT_BUDGET,
    MAX_PILOT_CONTACTS,
    AdaptivePilotStrategy,
    CandidateBelief,
    expected_value_of_information,
)
from agent import Agent
from mock_environment import make_mock_env


ROOT = Path(__file__).resolve().parents[1]


def belief(
    candidate_id: str,
    mean: float = 0.05,
    std: float = 0.20,
    channel: str = "sms",
    cost: float = 4.0,
    audience_size: int = 500,
) -> CandidateBelief:
    cumulative = np.cumsum(np.full(audience_size, 2_000.0))
    return CandidateBelief(
        candidate_id=candidate_id,
        current_tariff=f"origin_{candidate_id}",
        arpu_segment="MID",
        target_tariff=f"target_{candidate_id}",
        channel=channel,
        cost_per_contact=cost,
        cumulative_arpu=cumulative,
        prior_mean=mean,
        prior_std=std,
        posterior_mean=mean,
        posterior_variance=std**2,
    )


class StubEnvironment:
    def __init__(
        self,
        outcomes: dict[str, float],
        reject_target: str | None = None,
        reject_all: bool = False,
    ):
        self.total_budget = 100_000.0
        self.max_total_contacts = 15_000
        self.remaining_budget = self.total_budget
        self.remaining_contacts = self.max_total_contacts
        self.pilots_left = 20
        self.pilot_history: list[dict[str, object]] = []
        self.outcomes = outcomes
        self.reject_target = reject_target
        self.reject_all = reject_all

    def run_pilot(self, target_tariff: str, channel: str, n_customers: int, **_) -> dict:
        if self.reject_all or target_tariff == self.reject_target:
            self.reject_target = None
            raise RuntimeError("synthetic rejection")
        costs = {"push": 0.0, "sms": 4.0, "digital_ads": 22.0, "call": 160.0}
        cost = n_customers * costs[channel]
        self.remaining_budget -= cost
        self.remaining_contacts -= n_customers
        self.pilots_left -= 1
        result = {
            "target_tariff": target_tariff,
            "channel": channel,
            "n_customers": n_customers,
            "cost": cost,
            "observed_lift_ratio": self.outcomes[target_tariff],
        }
        self.pilot_history.append(result)
        return result


class BayesianUpdateTests(unittest.TestCase):
    def test_observation_moves_mean_and_reduces_uncertainty(self) -> None:
        candidate = belief("a", mean=0.0, std=0.2)
        before = candidate.posterior_std

        candidate.update(0.30, 100)

        self.assertGreater(candidate.posterior_mean, 0.0)
        self.assertLess(candidate.posterior_mean, 0.30)
        self.assertLess(candidate.posterior_std, before)
        self.assertEqual(candidate.pilot_count, 1)
        self.assertEqual(candidate.pilot_contacts, 100)

    def test_information_value_is_larger_for_uncertain_candidate(self) -> None:
        uncertain = belief("uncertain", std=0.30)
        certain = belief("certain", std=0.03)

        uncertain_value = expected_value_of_information(uncertain, 100, 500, 100_000)
        certain_value = expected_value_of_information(certain, 100, 500, 100_000)

        self.assertGreater(uncertain_value, certain_value)
        self.assertGreaterEqual(certain_value, 0.0)

    def test_changed_observation_changes_final_decision(self) -> None:
        positive = belief("positive", mean=0.0)
        negative = belief("negative", mean=0.0)
        positive.update(0.30, 200)
        negative.update(-0.30, 200)
        strategy = AdaptivePilotStrategy([negative, positive])
        env = StubEnvironment({})

        campaigns = strategy.select_final_campaigns(env)

        self.assertEqual(campaigns[0]["target_tariff"], positive.target_tariff)


class AdaptivePlannerTests(unittest.TestCase):
    def test_planner_reserves_resources_and_handles_rejection(self) -> None:
        candidates = [
            belief("push", channel="push", cost=0.0),
            belief("sms", channel="sms", cost=4.0),
            belief("call", channel="call", cost=160.0),
        ]
        outcomes = {item.target_tariff: 0.10 for item in candidates}
        env = StubEnvironment(outcomes, reject_target=candidates[0].target_tariff)
        strategy = AdaptivePilotStrategy(candidates)

        strategy.run_pilots(env)

        contacts_used = env.max_total_contacts - env.remaining_contacts
        budget_used = env.total_budget - env.remaining_budget
        self.assertLessEqual(len(env.pilot_history), MAX_ADAPTIVE_PILOTS)
        self.assertLessEqual(contacts_used, MAX_PILOT_CONTACTS)
        self.assertLessEqual(budget_used, MAX_PILOT_BUDGET)
        self.assertIn(candidates[0].candidate_id, strategy.rejected_candidates)
        self.assertGreater(len(env.pilot_history), 0)

    def test_call_pilot_is_capped_by_single_pilot_spend(self) -> None:
        candidate = belief("call", channel="call", cost=160.0)
        strategy = AdaptivePilotStrategy([candidate])
        env = StubEnvironment({candidate.target_tariff: 0.10})

        choice = strategy.choose_next_pilot(env, contacts_used=0, budget_used=0.0)

        self.assertIsNotNone(choice)
        self.assertLessEqual(choice[1], 50)

    def test_all_rejected_pilots_stop_and_return_fallback(self) -> None:
        candidates = [belief("push", channel="push", cost=0.0)]
        strategy = AdaptivePilotStrategy(candidates)
        env = StubEnvironment({}, reject_all=True)

        campaigns = strategy.run(env)

        self.assertEqual(len(strategy.rejected_candidates), 1)
        self.assertEqual(len(env.pilot_history), 0)
        self.assertEqual(len(campaigns), 1)
        self.assertEqual(campaigns[0]["channel"], "push")

    def test_small_audience_is_not_piloted(self) -> None:
        candidate = belief("small", channel="push", cost=0.0, audience_size=9)
        strategy = AdaptivePilotStrategy([candidate])
        env = StubEnvironment({candidate.target_tariff: 0.10})

        choice = strategy.choose_next_pilot(env, contacts_used=0, budget_used=0.0)

        self.assertIsNone(choice)

    def test_deployment_reserve_stops_exploration(self) -> None:
        candidate = belief("reserve", channel="push", cost=0.0)
        strategy = AdaptivePilotStrategy([candidate])
        env = StubEnvironment({candidate.target_tariff: 0.10})
        env.remaining_contacts = 10_000

        strategy.run_pilots(env)

        self.assertEqual(env.pilot_history, [])
        self.assertEqual(
            strategy.stop_reason,
            "no_positive_decision_benefit_or_resources",
        )


class MockEnvironmentIntegrationTests(unittest.TestCase):
    def test_agent_is_valid_and_bounded_across_fixed_seeds(self) -> None:
        for seed in range(3):
            with self.subTest(seed=seed):
                env, _ = make_mock_env(seed=seed)
                agent = Agent(ROOT / "analysis" / "candidate_shortlist.csv")

                campaigns = agent.act(env)

                self.assertGreaterEqual(len(campaigns), 1)
                self.assertLessEqual(len(campaigns), 10)
                self.assertGreater(len(env.pilot_history), 0)
                self.assertLessEqual(len(env.pilot_history), MAX_ADAPTIVE_PILOTS)
                self.assertGreaterEqual(env.remaining_budget, 60_000.0)
                self.assertGreaterEqual(env.remaining_contacts, 10_000)
                self.assertTrue(
                    all(
                        item["target_tariff"] != item["filter_current_tariff"]
                        for item in campaigns
                    )
                )


if __name__ == "__main__":
    unittest.main()
