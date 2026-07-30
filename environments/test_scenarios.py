import numpy as np
import pytest

from eiga.environments import EnvironmentConfig, ScenarioType
from eiga.environments.graph_generator import ScenarioGraphGenerator


@pytest.fixture
def generator():
    return ScenarioGraphGenerator(EnvironmentConfig(num_tools=6))


def test_every_scenario_is_reproducible(generator):
    for scenario in ScenarioType:
        a = generator.generate(np.random.default_rng(123), scenario)
        b = generator.generate(np.random.default_rng(123), scenario)
        np.testing.assert_array_equal(a.tool_lineages, b.tool_lineages)
        np.testing.assert_array_equal(a.tool_claims, b.tool_claims)
        np.testing.assert_array_equal(a.metadata_signatures, b.metadata_signatures)


def test_shared_lineage_implies_shared_claim(generator):
    for scenario in ScenarioType:
        episode = generator.generate(np.random.default_rng(3), scenario)
        for lineage in np.unique(episode.tool_lineages):
            claims = np.unique(episode.tool_claims[episode.tool_lineages == lineage])
            assert claims.size == 1


def test_independent_agreement_is_all_true_and_independent(generator):
    e = generator.generate(np.random.default_rng(1), ScenarioType.INDEPENDENT_AGREEMENT)
    assert np.unique(e.tool_lineages).size == 6
    assert np.all(e.tool_claims == e.true_answer)


def test_correlated_incorrect_has_false_correlated_majority(generator):
    e = generator.generate(np.random.default_rng(1), ScenarioType.CORRELATED_INCORRECT)
    wrong = 1 - e.true_answer
    assert np.mean(e.tool_claims == wrong) >= 0.75
    wrong_lineages = np.unique(e.tool_lineages[e.tool_claims == wrong])
    assert wrong_lineages.size == 1


def test_minority_truth_has_independent_true_sources(generator):
    e = generator.generate(
        np.random.default_rng(9), ScenarioType.MINORITY_INDEPENDENT_TRUTH
    )
    true_mask = e.tool_claims == e.true_answer
    false_mask = ~true_mask
    assert true_mask.sum() < false_mask.sum()
    assert np.unique(e.tool_lineages[true_mask]).size == true_mask.sum()
    assert np.unique(e.tool_lineages[false_mask]).size == 1
