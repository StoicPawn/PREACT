"""Adaptive policy optimisation agents for PREACT."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ..simulation.engine import SimulationEngine
from ..simulation.policy import PolicyAdjustment, PolicyParameters
from ..simulation.scenario import Scenario
from ..simulation.results import SimulationResults

RewardFunction = Callable[[SimulationResults], float]


@dataclass
class AgentTrainingResult:
    """Summary of an agent optimisation session."""

    best_policy: PolicyParameters
    best_reward: float
    reward_history: list[float]


class AdaptivePolicyAgent:
    """Tabular Q-learning agent exploring policy adjustments."""

    def __init__(
        self,
        engine: SimulationEngine,
        *,
        actions: Iterable[PolicyAdjustment] | None = None,
        learning_rate: float = 0.3,
        discount_factor: float = 0.85,
        exploration_rate: float = 0.2,
        random_state: int = 7,
    ) -> None:
        self.engine = engine
        self.actions = tuple(actions or self._default_actions())
        if not self.actions:
            raise ValueError("At least one policy adjustment action must be provided")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.rng = np.random.default_rng(random_state)
        self.q_table: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.actions), dtype=float)
        )

    @staticmethod
    def _default_actions() -> Sequence[PolicyAdjustment]:
        return (
            PolicyAdjustment(),
            PolicyAdjustment(base_deduction_delta=500.0),
            PolicyAdjustment(base_deduction_delta=-500.0),
            PolicyAdjustment(child_subsidy_multiplier=1.25),
            PolicyAdjustment(unemployment_benefit_multiplier=1.4),
            PolicyAdjustment(
                base_deduction_delta=300.0,
                unemployment_benefit_multiplier=1.2,
            ),
        )

    def _state_from_results(self, results: SimulationResults) -> tuple[int, ...]:
        kpis = results.kpis()
        unemployment_rate = float(kpis["unemployment_rate"])
        inflation = float(kpis["cpi"]) / 100.0
        sentiment = float(kpis["sentiment"]) / 100.0
        unemployment_bucket = int(np.digitize(unemployment_rate, bins=[0.05, 0.08, 0.12, 0.2]))
        inflation_bucket = int(np.digitize(inflation, bins=[0.9, 1.05, 1.2, 1.4]))
        sentiment_bucket = int(np.digitize(sentiment, bins=[0.4, 0.6, 0.75, 0.85]))
        return (unemployment_bucket, inflation_bucket, sentiment_bucket)

    def _select_action(self, state: tuple[int, ...]) -> int:
        if self.rng.random() < self.exploration_rate:
            return int(self.rng.integers(0, len(self.actions)))
        q_values = self.q_table[state]
        max_value = np.max(q_values)
        candidates = np.flatnonzero(np.isclose(q_values, max_value))
        if len(candidates) == 0:
            return int(self.rng.integers(0, len(self.actions)))
        return int(self.rng.choice(candidates))

    def _update_q(
        self,
        state: tuple[int, ...],
        action_idx: int,
        reward: float,
        next_state: tuple[int, ...],
    ) -> None:
        next_q = self.q_table[next_state]
        best_next = float(np.max(next_q))
        current = self.q_table[state][action_idx]
        target = reward + self.discount_factor * best_next
        self.q_table[state][action_idx] = (1 - self.learning_rate) * current + self.learning_rate * target

    def train(
        self,
        scenario: Scenario,
        *,
        episodes: int,
        reward_fn: RewardFunction,
    ) -> AgentTrainingResult:
        """Run episodic training and return the best policy discovered."""

        baseline_results = self.engine.run(scenario)
        state = self._state_from_results(baseline_results)
        baseline_reward = reward_fn(baseline_results)
        best_policy = scenario.policy
        best_reward = baseline_reward
        history = [baseline_reward]
        current_policy = scenario.policy

        for episode in range(episodes):
            action_idx = self._select_action(state)
            adjustment = self.actions[action_idx]
            candidate_policy = adjustment.apply(current_policy)
            candidate_scenario = scenario.with_policy(
                candidate_policy,
                name=f"AdaptiveEpisode-{episode+1}",
            )
            results = self.engine.run(candidate_scenario)
            reward = reward_fn(results)
            next_state = self._state_from_results(results)
            self._update_q(state, action_idx, reward, next_state)

            if reward > best_reward:
                best_reward = reward
                best_policy = candidate_policy

            history.append(reward)
            state = next_state
            current_policy = candidate_policy

        return AgentTrainingResult(
            best_policy=best_policy,
            best_reward=best_reward,
            reward_history=history,
        )

    def suggest_policy(
        self,
        scenario: Scenario,
        *,
        episodes: int,
        reward_fn: RewardFunction,
    ) -> PolicyParameters:
        """Convenience helper returning the best policy only."""

        result = self.train(scenario, episodes=episodes, reward_fn=reward_fn)
        return result.best_policy
