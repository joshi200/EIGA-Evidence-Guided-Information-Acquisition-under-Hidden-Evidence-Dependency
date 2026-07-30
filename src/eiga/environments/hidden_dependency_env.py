from __future__ import annotations

from typing import Any

import numpy as np

from .actions import ActionType, decode_action
from .config import EnvironmentConfig
from .episode import HiddenEpisode
from .graph_generator import ScenarioGraphGenerator
from .oracle import (
    build_dependency_matrix,
    compute_effective_evidence,
    observed_agreement,
)
from .scenarios import ScenarioType


class HiddenDependencyEnv:
    """Gym-like environment with no hard dependency on Gymnasium.

    The API follows reset()/step() conventions and can later be wrapped as a
    Gymnasium environment without changing benchmark logic.
    """

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self.generator = ScenarioGraphGenerator(self.config)
        self._rng = np.random.default_rng()
        self._episode: HiddenEpisode | None = None
        self.action_space_n = (
            2 * self.config.num_tools + self.config.num_answer_classes + 1
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
        self._rng = np.random.default_rng(seed)
        requested = (options or {}).get("scenario", self.config.default_scenario)
        scenario = requested if isinstance(requested, ScenarioType) else ScenarioType(requested)
        self._episode = self.generator.generate(self._rng, scenario)
        return self._build_observation(), self._public_info()

    def step(
        self, action_index: int
    ) -> tuple[dict[str, np.ndarray | float], float, bool, bool, dict[str, Any]]:
        episode = self._require_episode()
        if episode.terminated:
            raise RuntimeError("episode has already terminated")
        if not isinstance(action_index, (int, np.integer)):
            raise TypeError("action_index must be an integer")
        if not 0 <= int(action_index) < self.action_space_n:
            raise ValueError(f"action index {action_index} is out of range")

        mask = self._build_action_mask()
        if not mask[int(action_index)]:
            raise ValueError(f"action {action_index} is invalid in the current state")

        action = decode_action(
            int(action_index), self.config.num_tools, self.config.num_answer_classes
        )
        if action.action_type is ActionType.QUERY:
            reward = self._handle_query(int(action.target))
        elif action.action_type is ActionType.DIAGNOSE:
            reward = self._handle_diagnostic(int(action.target))
        elif action.action_type is ActionType.ANSWER:
            reward = self._handle_answer(int(action.target))
        else:
            reward = self._handle_abstention()

        episode.current_step += 1
        if not episode.terminated and episode.current_step >= self.config.max_steps:
            episode.terminated = True
            episode.terminal_reason = "max_steps"

        terminated = episode.terminated
        info = self._public_info()
        if terminated:
            info["oracle"] = self.get_oracle_state()
        return self._build_observation(), reward, terminated, False, info

    def get_oracle_state(self) -> dict[str, Any]:
        episode = self._require_episode()
        return {
            "scenario": episode.scenario.value,
            "true_answer": episode.true_answer,
            "tool_lineages": episode.tool_lineages.copy(),
            "dependency_matrix": build_dependency_matrix(episode.tool_lineages),
            "true_effective_evidence": compute_effective_evidence(
                episode.tool_lineages, episode.queried_mask
            ),
            "observed_agreement": observed_agreement(
                episode.tool_claims, episode.queried_mask
            ),
            "terminal_reason": episode.terminal_reason,
            "submitted_answer": episode.submitted_answer,
        }

    def _handle_query(self, target: int) -> float:
        episode = self._require_episode()
        episode.queried_mask[target] = True
        episode.remaining_budget = self._spend(self.config.query_cost)
        episode.action_history.append({"type": "query", "target": target})
        return -self.config.query_cost

    def _handle_diagnostic(self, target: int) -> float:
        episode = self._require_episode()
        queried_indices = np.flatnonzero(episode.queried_mask)
        for other in queried_indices:
            same = episode.tool_lineages[target] == episode.tool_lineages[other]
            reported = bool(same)
            if self._rng.random() > self.config.diagnostic_accuracy:
                reported = not reported
            value = float(reported)
            episode.diagnostic_matrix[target, other] = value
            episode.diagnostic_matrix[other, target] = value
        episode.diagnostic_matrix[target, target] = 1.0
        episode.diagnosed_mask[target] = True
        episode.remaining_budget = self._spend(self.config.diagnostic_cost)
        episode.action_history.append({"type": "diagnose", "target": target})
        return -self.config.diagnostic_cost

    def _handle_answer(self, answer: int) -> float:
        episode = self._require_episode()
        correct = answer == episode.true_answer
        reward = self.config.correct_reward if correct else self.config.incorrect_reward
        if self._is_false_consensus(answer):
            reward += self.config.false_consensus_penalty
        episode.submitted_answer = answer
        episode.terminated = True
        episode.terminal_reason = "answer"
        episode.action_history.append(
            {"type": "answer", "answer": answer, "correct": correct}
        )
        return reward

    def _handle_abstention(self) -> float:
        episode = self._require_episode()
        episode.terminated = True
        episode.terminal_reason = "abstain"
        episode.action_history.append({"type": "abstain"})
        return self.config.abstain_reward

    def _is_false_consensus(self, submitted_answer: int) -> bool:
        episode = self._require_episode()
        queried = episode.queried_mask
        if queried.sum() < 2 or submitted_answer == episode.true_answer:
            return False
        agreement = float(np.mean(episode.tool_claims[queried] == submitted_answer))
        effective = compute_effective_evidence(episode.tool_lineages, queried)
        return (
            agreement >= self.config.false_consensus_agreement_threshold
            and effective <= self.config.false_consensus_max_effective_evidence
        )

    def _build_observation(self) -> dict[str, np.ndarray | float]:
        episode = self._require_episode()
        n = self.config.num_tools
        claims = np.full(n, -1, dtype=np.int64)
        confidences = np.zeros(n, dtype=np.float32)
        metadata = np.full(n, -1, dtype=np.int64)
        claims[episode.queried_mask] = episode.tool_claims[episode.queried_mask]
        confidences[episode.queried_mask] = episode.tool_confidences[episode.queried_mask]
        metadata[episode.queried_mask] = episode.metadata_signatures[episode.queried_mask]
        return {
            "tool_claims": claims,
            "tool_confidences": confidences,
            "queried_mask": episode.queried_mask.copy(),
            "diagnosed_mask": episode.diagnosed_mask.copy(),
            "metadata_signatures": metadata,
            "diagnostic_matrix": episode.diagnostic_matrix.copy(),
            "remaining_budget": float(episode.remaining_budget),
            "normalised_step": float(episode.current_step / self.config.max_steps),
            "action_mask": self._build_action_mask(),
        }

    def _build_action_mask(self) -> np.ndarray:
        episode = self._require_episode()
        cfg = self.config
        mask = np.zeros(self.action_space_n, dtype=bool)
        if episode.terminated:
            return mask
        if episode.remaining_budget + 1e-9 >= cfg.query_cost:
            mask[: cfg.num_tools] = ~episode.queried_mask
        if episode.remaining_budget + 1e-9 >= cfg.diagnostic_cost:
            mask[cfg.num_tools : 2 * cfg.num_tools] = (
                episode.queried_mask & ~episode.diagnosed_mask
            )
        answer_start = 2 * cfg.num_tools
        mask[answer_start : answer_start + cfg.num_answer_classes] = True
        mask[-1] = True
        return mask

    def _public_info(self) -> dict[str, Any]:
        episode = self._require_episode()
        return {
            "scenario": episode.scenario.value,
            "remaining_budget": float(episode.remaining_budget),
            "current_step": episode.current_step,
        }

    def _spend(self, cost: float) -> float:
        episode = self._require_episode()
        remaining = episode.remaining_budget - cost
        if remaining < -1e-8:
            raise RuntimeError("budget became negative")
        return max(0.0, remaining)

    def _require_episode(self) -> HiddenEpisode:
        if self._episode is None:
            raise RuntimeError("call reset before interacting with the environment")
        return self._episode
