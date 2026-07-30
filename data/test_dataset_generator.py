import json

import numpy as np

from eiga.data.generator import DatasetGenerationConfig, SupervisedDatasetGenerator
from eiga.environments.scenarios import ScenarioType


def small_generator(**overrides):
    values = {
        "train_episodes_per_scenario": 1,
        "validation_episodes_per_scenario": 1,
        "test_episodes_per_scenario": 1,
        "base_seed": 123,
        "diagnostic_probability": 1.0,
    }
    values.update(overrides)
    return SupervisedDatasetGenerator(
        generation_config=DatasetGenerationConfig(**values)
    )


def test_generation_is_deterministic():
    first = small_generator().generate()
    second = small_generator().generate()
    assert first["train"][2].to_json_dict() == second["train"][2].to_json_dict()


def test_splits_have_disjoint_seeds():
    datasets = small_generator().generate()
    seed_sets = {
        split: {record.seed for record in records}
        for split, records in datasets.items()
    }
    assert seed_sets["train"].isdisjoint(seed_sets["validation"])
    assert seed_sets["train"].isdisjoint(seed_sets["test"])
    assert seed_sets["validation"].isdisjoint(seed_sets["test"])


def test_every_scenario_is_present_in_each_split():
    datasets = small_generator().generate()
    expected = {scenario.value for scenario in ScenarioType}
    for records in datasets.values():
        assert {record.scenario for record in records} == expected


def test_unqueried_features_do_not_reveal_lineage():
    record = small_generator(include_empty_prefix=True).generate()["train"][0]
    assert not record.queried_mask.any()
    assert np.all(record.tool_claims == -1)
    assert np.all(record.metadata_signatures == -1)
    assert np.all(record.tool_confidences == 0.0)


def test_dependency_mask_only_supervises_observed_tools():
    records = small_generator(include_empty_prefix=False).generate()["train"]
    record = records[0]
    expected = record.queried_mask[:, None] & record.queried_mask[None, :]
    np.testing.assert_array_equal(record.target_dependency_mask, expected)


def test_effective_evidence_matches_unique_queried_lineages():
    records = small_generator(include_empty_prefix=False).generate()["train"]
    for record in records:
        expected = np.unique(
            record.target_lineage_labels[record.queried_mask]
        ).size
        assert record.target_effective_evidence == expected


def test_write_creates_jsonl_and_manifest(tmp_path):
    paths = small_generator(max_prefixes_per_episode=2).write(tmp_path)
    assert paths["train"].exists()
    assert paths["validation"].exists()
    assert paths["test"].exists()
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["format_version"] == 1
    assert manifest["records_per_split"]["train"] > 0
    first_line = paths["train"].read_text().splitlines()[0]
    record = json.loads(first_line)
    assert set(record) == {"episode_id", "prefix_id", "seed", "split", "scenario", "features", "targets"}
