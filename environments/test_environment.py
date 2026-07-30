import numpy as np
import pytest

from eiga.environments import EnvironmentConfig, HiddenDependencyEnv, ScenarioType
from eiga.environments.oracle import compute_effective_evidence


def test_hidden_lineage_not_in_observation():
    env = HiddenDependencyEnv()
    observation, info = env.reset(seed=1)
    assert "tool_lineages" not in observation
    assert "true_answer" not in observation
    assert "oracle" not in info


def test_query_reveals_only_target_and_masks_repeat():
    env = HiddenDependencyEnv()
    obs, _ = env.reset(seed=1)
    assert np.all(obs["tool_claims"] == -1)
    obs, reward, done, _, _ = env.step(0)
    assert reward == -env.config.query_cost
    assert not done
    assert obs["tool_claims"][0] in (0, 1)
    assert not obs["action_mask"][0]


def test_unqueried_tool_cannot_be_diagnosed():
    env = HiddenDependencyEnv()
    obs, _ = env.reset(seed=1)
    assert not obs["action_mask"][env.config.num_tools]
    with pytest.raises(ValueError):
        env.step(env.config.num_tools)


def test_diagnostic_is_symmetric_and_noisy_binary():
    env = HiddenDependencyEnv(EnvironmentConfig(diagnostic_accuracy=1.0))
    obs, _ = env.reset(seed=2)
    obs, *_ = env.step(0)
    obs, *_ = env.step(1)
    obs, *_ = env.step(env.config.num_tools)
    matrix = obs["diagnostic_matrix"]
    assert matrix[0, 1] in (0.0, 1.0)
    assert matrix[0, 1] == matrix[1, 0]


def test_answer_terminates_and_exposes_oracle_only_at_end():
    env = HiddenDependencyEnv()
    _, _ = env.reset(seed=4)
    answer_index = 2 * env.config.num_tools
    _, _, done, _, info = env.step(answer_index)
    assert done
    assert "oracle" in info


def test_budget_never_negative():
    env = HiddenDependencyEnv(EnvironmentConfig(initial_budget=1.0, query_cost=1.0))
    obs, _ = env.reset(seed=5)
    obs, *_ = env.step(0)
    assert obs["remaining_budget"] == 0.0
    assert not obs["action_mask"][: env.config.num_tools].any()


def test_effective_evidence_duplicate_invariant():
    lineages = np.array([0, 0, 1])
    assert compute_effective_evidence(lineages, np.array([True, False, False])) == 1
    assert compute_effective_evidence(lineages, np.array([True, True, False])) == 1
    assert compute_effective_evidence(lineages, np.array([True, True, True])) == 2


def test_false_consensus_penalty_can_trigger():
    cfg = EnvironmentConfig(
        num_tools=6,
        initial_budget=6.0,
        false_consensus_max_effective_evidence=1,
    )
    env = HiddenDependencyEnv(cfg)
    obs, _ = env.reset(seed=10, options={"scenario": ScenarioType.CORRELATED_INCORRECT})
    episode = env._require_episode()
    wrong = 1 - episode.true_answer
    wrong_indices = np.flatnonzero(episode.tool_claims == wrong)
    for idx in wrong_indices:
        obs, *_ = env.step(int(idx))
    answer_index = 2 * cfg.num_tools + wrong
    _, reward, done, _, _ = env.step(answer_index)
    assert done
    assert reward == cfg.incorrect_reward + cfg.false_consensus_penalty
