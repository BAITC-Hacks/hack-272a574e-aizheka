"""Bayesian pilot learning and resource-aware campaign exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, erf, exp, isfinite, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


OBSERVATION_STD = 0.804
PRIOR_RESPONSE_SCALE = 0.10
PRIOR_STD_FLOOR = 0.12
PRIOR_STD_CEILING = 0.40
TARGET_POSTERIOR_STD = 0.06
MIN_PILOT_CUSTOMERS = 10
MAX_PILOT_CUSTOMERS = 200
MAX_ADAPTIVE_PILOTS = 8
MAX_PILOTS_PER_CANDIDATE = 2
MAX_PILOTS_PER_AUDIENCE = 2
MAX_PILOT_CONTACTS = 1_600
MAX_PILOT_BUDGET = 20_000.0
MAX_SINGLE_PILOT_SPEND = 8_000.0
DEPLOYMENT_CONTACT_RESERVE = 10_000
DEPLOYMENT_BUDGET_RESERVE = 60_000.0
MIN_DECISION_BENEFIT = 1_000.0
MAX_FINAL_CAMPAIGNS = 6
MAX_CUSTOMERS_PER_CAMPAIGN = 5_000
CONSERVATIVE_Z = 1.64
HIGH_COMMITMENT_Z = 1.96
HIGH_COMMITMENT_BUDGET_SHARE = 0.50

SHORTLIST_COLUMNS = {
    "selection_order",
    "candidate_id",
    "filter_current_tariff",
    "filter_arpu_segment",
    "target_tariff",
    "channel",
    "posterior_mean_lift",
    "posterior_std_lift",
    "sample_size",
    "channel_multiplier",
    "cost_per_contact",
}


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def expected_positive_part(mean: float, std: float, threshold: float) -> float:
    """Return E[max(X - threshold, 0)] for a normal random variable X."""

    if std <= 0:
        return max(mean - threshold, 0.0)
    z_score = (mean - threshold) / std
    return (mean - threshold) * normal_cdf(z_score) + std * normal_pdf(z_score)


@dataclass
class CandidateBelief:
    candidate_id: str
    current_tariff: str
    arpu_segment: str
    target_tariff: str
    channel: str
    cost_per_contact: float
    cumulative_arpu: np.ndarray = field(repr=False)
    prior_mean: float
    prior_std: float
    posterior_mean: float
    posterior_variance: float
    pilot_count: int = 0
    pilot_contacts: int = 0
    blocked: bool = False

    @property
    def posterior_std(self) -> float:
        return sqrt(self.posterior_variance)

    @property
    def audience_size(self) -> int:
        return int(len(self.cumulative_arpu))

    @property
    def audience_arpu(self) -> float:
        return self.arpu_for(self.audience_size)

    @property
    def average_arpu(self) -> float:
        return 0.0 if self.audience_size == 0 else self.audience_arpu / self.audience_size

    @property
    def audience_key(self) -> tuple[str, str]:
        return self.current_tariff, self.arpu_segment

    def arpu_for(self, contacts: int) -> float:
        contacts = min(max(int(contacts), 0), self.audience_size)
        return 0.0 if contacts == 0 else float(self.cumulative_arpu[contacts - 1])

    def capacity(self, contacts: int, budget: float) -> int:
        capacity = min(self.audience_size, MAX_CUSTOMERS_PER_CAMPAIGN, max(int(contacts), 0))
        if self.cost_per_contact > 0:
            capacity = min(capacity, max(int(budget // self.cost_per_contact), 0))
        return capacity

    def update(self, observed_lift_ratio: float, n_customers: int) -> None:
        if not isfinite(observed_lift_ratio):
            raise ValueError("Pilot lift observation must be finite")
        if n_customers <= 0:
            raise ValueError("Pilot sample size must be positive")
        observation_variance = OBSERVATION_STD**2 / n_customers
        prior_precision = 1.0 / self.posterior_variance
        observation_precision = 1.0 / observation_variance
        updated_variance = 1.0 / (prior_precision + observation_precision)
        self.posterior_mean = updated_variance * (
            prior_precision * self.posterior_mean
            + observation_precision * observed_lift_ratio
        )
        self.posterior_variance = updated_variance
        self.pilot_count += 1
        self.pilot_contacts += int(n_customers)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def build_beliefs(shortlist: pd.DataFrame, profile: pd.DataFrame) -> list[CandidateBelief]:
    """Map historical transition priors to weak campaign-response priors."""

    _require_columns(shortlist, SHORTLIST_COLUMNS, "candidate shortlist")
    _require_columns(
        profile,
        {"ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu"},
        "customer profile",
    )
    audiences: dict[tuple[str, str], np.ndarray] = {}
    for (tariff, segment), rows in profile.groupby(
        ["current_tariff", "arpu_segment"], observed=True
    ):
        sorted_rows = rows.sort_values("ID_NUMBER", kind="stable")
        values = pd.to_numeric(sorted_rows["predicted_arpu"], errors="coerce").to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Eligible audience ARPU values must be finite and nonnegative")
        audiences[(str(tariff), str(segment))] = np.cumsum(values, dtype=float)

    beliefs: list[CandidateBelief] = []
    for row in shortlist.sort_values("selection_order", kind="stable").itertuples(index=False):
        audience_key = (str(row.filter_current_tariff), str(row.filter_arpu_segment))
        cumulative_arpu = audiences.get(audience_key)
        if cumulative_arpu is None or len(cumulative_arpu) < MIN_PILOT_CUSTOMERS:
            continue
        historical_signal = float(row.posterior_mean_lift) * float(row.channel_multiplier)
        prior_mean = float(np.clip(
            PRIOR_RESPONSE_SCALE * historical_signal,
            -0.20,
            0.35,
        ))
        sample_size = max(float(row.sample_size), 1.0)
        historical_uncertainty = (
            PRIOR_RESPONSE_SCALE
            * float(row.posterior_std_lift)
            * float(row.channel_multiplier)
        )
        sparse_floor = 0.24 / sqrt(sample_size)
        prior_std = float(np.clip(
            max(PRIOR_STD_FLOOR, historical_uncertainty, sparse_floor),
            PRIOR_STD_FLOOR,
            PRIOR_STD_CEILING,
        ))
        beliefs.append(
            CandidateBelief(
                candidate_id=str(row.candidate_id),
                current_tariff=audience_key[0],
                arpu_segment=audience_key[1],
                target_tariff=str(row.target_tariff),
                channel=str(row.channel),
                cost_per_contact=float(row.cost_per_contact),
                cumulative_arpu=cumulative_arpu,
                prior_mean=prior_mean,
                prior_std=prior_std,
                posterior_mean=prior_mean,
                posterior_variance=prior_std**2,
            )
        )
    if not beliefs:
        raise ValueError("No shortlist candidates have a pilotable audience")
    return beliefs


def expected_value_of_information(
    belief: CandidateBelief,
    n_customers: int,
    deployment_contacts: int,
    deployment_budget: float,
) -> float:
    """Expected gain from deciding after a sample instead of deciding now."""

    capacity = belief.capacity(deployment_contacts, deployment_budget)
    deployment_arpu = belief.arpu_for(capacity)
    if capacity <= 0 or deployment_arpu <= 0 or n_customers <= 0:
        return 0.0
    threshold = capacity * belief.cost_per_contact / deployment_arpu
    observation_variance = OBSERVATION_STD**2 / n_customers
    posterior_variance = 1.0 / (
        1.0 / belief.posterior_variance + 1.0 / observation_variance
    )
    mean_shift_std = sqrt(max(belief.posterior_variance - posterior_variance, 0.0))
    future_value = deployment_arpu * expected_positive_part(
        belief.posterior_mean,
        mean_shift_std,
        threshold,
    )
    current_value = deployment_arpu * max(belief.posterior_mean - threshold, 0.0)
    return max(future_value - current_value, 0.0)


class AdaptivePilotStrategy:
    def __init__(self, beliefs: list[CandidateBelief]):
        self.beliefs = beliefs
        self.stop_reason = "not_started"
        self.rejected_candidates: list[str] = []

    @classmethod
    def from_frames(
        cls,
        shortlist: pd.DataFrame,
        profile: pd.DataFrame,
    ) -> "AdaptivePilotStrategy":
        return cls(build_beliefs(shortlist, profile))

    @classmethod
    def from_csv(
        cls,
        shortlist_path: Path,
        profile: pd.DataFrame,
    ) -> "AdaptivePilotStrategy":
        return cls.from_frames(pd.read_csv(shortlist_path), profile)

    @staticmethod
    def _resource_reserves(env) -> tuple[int, float]:
        contact_reserve = min(
            DEPLOYMENT_CONTACT_RESERVE,
            int(getattr(env, "max_total_contacts", DEPLOYMENT_CONTACT_RESERVE) * 2 / 3),
        )
        budget_reserve = min(
            DEPLOYMENT_BUDGET_RESERVE,
            float(getattr(env, "total_budget", DEPLOYMENT_BUDGET_RESERVE) * 0.60),
        )
        return contact_reserve, budget_reserve

    def _pilot_size(
        self,
        belief: CandidateBelief,
        env,
        contacts_used: int,
        budget_used: float,
    ) -> int:
        if belief.blocked or belief.pilot_count >= MAX_PILOTS_PER_CANDIDATE:
            return 0
        audience_pilots = sum(
            item.pilot_count for item in self.beliefs if item.audience_key == belief.audience_key
        )
        if audience_pilots >= MAX_PILOTS_PER_AUDIENCE:
            return 0

        contact_reserve, budget_reserve = self._resource_reserves(env)
        contact_capacity = min(
            MAX_PILOT_CONTACTS - contacts_used,
            int(env.remaining_contacts) - contact_reserve,
            belief.audience_size,
            MAX_PILOT_CUSTOMERS,
        )
        budget_capacity = min(
            float(MAX_PILOT_BUDGET - budget_used),
            float(env.remaining_budget) - budget_reserve,
        )
        if belief.cost_per_contact > 0:
            contact_capacity = min(
                contact_capacity,
                int(max(budget_capacity, 0.0) // belief.cost_per_contact),
                int(MAX_SINGLE_PILOT_SPEND // belief.cost_per_contact),
            )
        if contact_capacity < MIN_PILOT_CUSTOMERS:
            return 0

        target_precision = max(
            1.0 / TARGET_POSTERIOR_STD**2 - 1.0 / belief.posterior_variance,
            0.0,
        )
        desired = ceil(OBSERVATION_STD**2 * target_precision)
        desired = max(desired, 40 if contact_capacity >= 40 else MIN_PILOT_CUSTOMERS)
        return int(min(desired, contact_capacity))

    def _pilot_utility(self, belief: CandidateBelief, sample_size: int, env) -> float:
        if sample_size < MIN_PILOT_CUSTOMERS:
            return float("-inf")
        remaining_contacts = max(int(env.remaining_contacts) - sample_size, 0)
        remaining_budget = float(env.remaining_budget) - sample_size * belief.cost_per_contact
        information_value = expected_value_of_information(
            belief,
            sample_size,
            remaining_contacts,
            remaining_budget,
        )
        pilot_net = sample_size * (
            belief.average_arpu * belief.posterior_mean - belief.cost_per_contact
        )
        return information_value + pilot_net

    def choose_next_pilot(
        self,
        env,
        contacts_used: int,
        budget_used: float,
    ) -> tuple[CandidateBelief, int, float] | None:
        ranked: list[tuple[float, str, CandidateBelief, int]] = []
        for belief in self.beliefs:
            sample_size = self._pilot_size(belief, env, contacts_used, budget_used)
            if sample_size == 0:
                continue
            utility = self._pilot_utility(belief, sample_size, env)
            ranked.append((utility, belief.candidate_id, belief, sample_size))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (-item[0], item[1]))
        utility, _, belief, sample_size = ranked[0]
        if utility < MIN_DECISION_BENEFIT:
            return None
        return belief, sample_size, utility

    def run_pilots(self, env) -> None:
        starting_contacts = int(env.remaining_contacts)
        starting_budget = float(env.remaining_budget)
        attempts = 0

        while attempts < MAX_ADAPTIVE_PILOTS:
            if int(env.pilots_left) <= 0:
                self.stop_reason = "environment_pilot_limit"
                return
            contacts_used = starting_contacts - int(env.remaining_contacts)
            budget_used = starting_budget - float(env.remaining_budget)
            next_pilot = self.choose_next_pilot(env, contacts_used, budget_used)
            if next_pilot is None:
                self.stop_reason = "no_positive_decision_benefit_or_resources"
                return
            belief, sample_size, _ = next_pilot
            attempts += 1
            try:
                result = env.run_pilot(
                    target_tariff=belief.target_tariff,
                    channel=belief.channel,
                    n_customers=sample_size,
                    filter_arpu_segment=belief.arpu_segment,
                    filter_current_tariff=belief.current_tariff,
                )
                belief.update(
                    float(result["observed_lift_ratio"]),
                    int(result["n_customers"]),
                )
            except (RuntimeError, ValueError, KeyError, TypeError):
                belief.blocked = True
                self.rejected_candidates.append(belief.candidate_id)

        self.stop_reason = "adaptive_pilot_attempt_cap"

    @staticmethod
    def _campaign(belief: CandidateBelief, order: int) -> dict[str, object]:
        return {
            "campaign_name": f"adaptive_{order}_{belief.current_tariff}_{belief.target_tariff}",
            "filter_arpu_segment": belief.arpu_segment,
            "filter_current_tariff": belief.current_tariff,
            "target_tariff": belief.target_tariff,
            "channel": belief.channel,
        }

    def select_final_campaigns(self, env) -> list[dict[str, object]]:
        remaining_contacts = int(env.remaining_contacts)
        remaining_budget = float(env.remaining_budget)
        scored: list[tuple[float, float, str, CandidateBelief, int]] = []
        for belief in self.beliefs:
            if belief.blocked or belief.pilot_count == 0:
                continue
            contacts = belief.capacity(remaining_contacts, remaining_budget)
            if contacts < MIN_PILOT_CUSTOMERS:
                continue
            arpu = belief.arpu_for(contacts)
            expected_net = arpu * belief.posterior_mean - contacts * belief.cost_per_contact
            campaign_cost = contacts * belief.cost_per_contact
            risk_z = (
                HIGH_COMMITMENT_Z
                if campaign_cost > HIGH_COMMITMENT_BUDGET_SHARE * remaining_budget
                else CONSERVATIVE_Z
            )
            conservative_net = (
                arpu * (belief.posterior_mean - risk_z * belief.posterior_std)
                - contacts * belief.cost_per_contact
            )
            scored.append(
                (conservative_net, expected_net, belief.candidate_id, belief, contacts)
            )
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))

        campaigns: list[dict[str, object]] = []
        used_audiences: set[tuple[str, str]] = set()
        for conservative_net, _, _, belief, _ in scored:
            if len(campaigns) >= MAX_FINAL_CAMPAIGNS or conservative_net <= 0:
                break
            if belief.audience_key in used_audiences:
                continue
            contacts = belief.capacity(remaining_contacts, remaining_budget)
            if contacts < MIN_PILOT_CUSTOMERS:
                continue
            campaigns.append(self._campaign(belief, len(campaigns) + 1))
            used_audiences.add(belief.audience_key)
            remaining_contacts -= contacts
            remaining_budget -= contacts * belief.cost_per_contact

        if campaigns:
            return campaigns

        measured = [item for item in scored if item[1] > 0]
        if measured:
            return [self._campaign(measured[0][3], 1)]
        push = [
            belief
            for belief in self.beliefs
            if belief.channel == "push" and not belief.blocked
        ]
        fallback_pool = push or [belief for belief in self.beliefs if not belief.blocked]
        if not fallback_pool:
            fallback_pool = [belief for belief in self.beliefs if belief.channel == "push"]
        if not fallback_pool:
            fallback_pool = self.beliefs
        fallback = max(
            fallback_pool,
            key=lambda item: (item.posterior_mean, -item.cost_per_contact, item.candidate_id),
        )
        return [self._campaign(fallback, 1)]

    def run(self, env) -> list[dict[str, object]]:
        self.run_pilots(env)
        return self.select_final_campaigns(env)
