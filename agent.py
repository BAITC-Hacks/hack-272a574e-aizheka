"""Adaptive Bayesian agent for the Beeline campaign environment."""

from __future__ import annotations

from pathlib import Path

from adaptive_pilots import AdaptivePilotStrategy


class Agent:
    def __init__(self, shortlist_path: str | Path | None = None):
        root = Path(__file__).resolve().parent
        self.shortlist_path = Path(
            shortlist_path or root / "analysis" / "candidate_shortlist.csv"
        )
        self.last_strategy: AdaptivePilotStrategy | None = None

    def act(self, env) -> list[dict[str, object]]:
        strategy = AdaptivePilotStrategy.from_csv(
            self.shortlist_path,
            env.customer_profile,
        )
        self.last_strategy = strategy
        return strategy.run(env)
