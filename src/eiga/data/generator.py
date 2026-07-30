from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Iterable

import numpy as np

from eiga.environments.actions import ActionType, AgentAction, encode_action
from eiga.environments.config import EnvironmentConfig
from eiga.environments.hidden_dependency_env import HiddenDependencyEnv
from eiga.environments.oracle import (
    build_dependency_matrix,
    compute_effective_evidence,
    observed_agreement,
)
from eiga.environments.scenarios import ScenarioType

from .records import EvidenceTrainingRecord


@dataclass(frozen=True)
class DatasetGenerationConfig:
    """Controls deterministic, scenario-balanced supervised data generation."""

    train_episodes_per_scenario: int = 200
    validation_episodes_per_scenario: int = 50
    test_episodes_per_scenario: int = 50
    base_seed: int = 17
    include_empty_prefix: bool = True
    include_diagnostics: bool = True
    diagnostic_probability: float = 0.30
    max_prefixes_per_episode: int | None = None

    def __post_init__(self) -> None:
        counts = (
            self.train_episodes_per_scenario,
            self.validation_episodes_per_scenario,
            self.test_episodes_per_scenario,
        )
        if any(value < 0 for value in counts):
            raise ValueError("episode counts must be non-negative")
        if not 0.0 <= self.diagnostic_probability <= 1.0:
            raise ValueError("diagnostic_probability must lie in [0, 1]")
        if self.max_prefixes_per_episode is not None and self.max_prefixes_per_episode < 1:
            raise ValueError("max_prefixes_per_episode must be positive")


class SupervisedDatasetGenerator:
    """Generate episode-prefix supervision for independent/pairwise/latent EBMs."""

    def __init__(
        self,
        environment_config: EnvironmentConfig | None = None,
        generation_config: DatasetGenerationConfig | None = None,
    ) -> None:
        self.environment_config = environment_config or EnvironmentConfig()
        self.generation_config = generation_config or DatasetGenerationConfig()

    def generate(self) -> dict[str, list[EvidenceTrainingRecord]]:
        datasets: dict[str, list[EvidenceTrainingRecord]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        episode_id = 0
        for split, count in self._split_counts().items():
            for scenario_index, scenario in enumerate(ScenarioType):
                for local_index in range(count):
                    seed = self._episode_seed(split, scenario_index, local_index)
                    records = self.generate_episode(
                        episode_id=episode_id,
                        split=split,
                        scenario=scenario,
                        seed=seed,
                    )
                    datasets[split].extend(records)
                    episode_id += 1
        return datasets

    def generate_episode(
        self,
        *,
        episode_id: int,
        split: str,
        scenario: ScenarioType,
        seed: int,
    ) -> list[EvidenceTrainingRecord]:
        env = HiddenDependencyEnv(self.environment_config)
        observation, _ = env.reset(seed=seed, options={"scenario": scenario})
        rng = np.random.default_rng(seed ^ 0x5DEECE66D)

        records: list[EvidenceTrainingRecord] = []
        prefix_id = 0
        if self.generation_config.include_empty_prefix:
            records.append(
                self._record_from_state(
                    env, observation, episode_id, prefix_id, split, scenario, seed
                )
            )
            prefix_id += 1

        query_order = rng.permutation(self.environment_config.num_tools).tolist()
        for tool_index in query_order:
            action = encode_action(
                AgentAction(ActionType.QUERY, int(tool_index)),
                self.environment_config.num_tools,
                self.environment_config.num_answer_classes,
            )
            if not bool(observation["action_mask"][action]):
                continue
            observation, _, terminated, _, _ = env.step(action)
            if terminated:
                break
            records.append(
                self._record_from_state(
                    env, observation, episode_id, prefix_id, split, scenario, seed
                )
            )
            prefix_id += 1
            if self._limit_reached(records):
                break

            if (
                self.generation_config.include_diagnostics
                and rng.random() < self.generation_config.diagnostic_probability
            ):
                diagnostic = encode_action(
                    AgentAction(ActionType.DIAGNOSE, int(tool_index)),
                    self.environment_config.num_tools,
                    self.environment_config.num_answer_classes,
                )
                if bool(observation["action_mask"][diagnostic]):
                    observation, _, terminated, _, _ = env.step(diagnostic)
                    if terminated:
                        break
                    records.append(
                        self._record_from_state(
                            env,
                            observation,
                            episode_id,
                            prefix_id,
                            split,
                            scenario,
                            seed,
                        )
                    )
                    prefix_id += 1
                    if self._limit_reached(records):
                        break

        return records

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        datasets = self.generate()
        paths: dict[str, Path] = {}

        for split, records in datasets.items():
            jsonl_path = output / f"{split}.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_json_dict(), separators=(",", ":")))
                    handle.write("\n")
            paths[split] = jsonl_path

        manifest = self._manifest(datasets)
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        paths["manifest"] = manifest_path
        return paths

    def _record_from_state(
        self,
        env: HiddenDependencyEnv,
        observation: dict,
        episode_id: int,
        prefix_id: int,
        split: str,
        scenario: ScenarioType,
        seed: int,
    ) -> EvidenceTrainingRecord:
        episode = env._require_episode()  # privileged generator boundary
        queried = np.asarray(observation["queried_mask"], dtype=bool)
        dependency = build_dependency_matrix(episode.tool_lineages)
        dependency_mask = queried[:, None] & queried[None, :]
        false_consensus = np.asarray(
            [
                self._false_consensus_for_answer(env, answer)
                for answer in range(self.environment_config.num_answer_classes)
            ],
            dtype=bool,
        )
        return EvidenceTrainingRecord(
            episode_id=episode_id,
            prefix_id=prefix_id,
            seed=seed,
            split=split,
            scenario=scenario.value,
            tool_claims=np.asarray(observation["tool_claims"], dtype=np.int64).copy(),
            tool_confidences=np.asarray(
                observation["tool_confidences"], dtype=np.float32
            ).copy(),
            queried_mask=queried.copy(),
            diagnosed_mask=np.asarray(
                observation["diagnosed_mask"], dtype=bool
            ).copy(),
            metadata_signatures=np.asarray(
                observation["metadata_signatures"], dtype=np.int64
            ).copy(),
            diagnostic_matrix=np.asarray(
                observation["diagnostic_matrix"], dtype=np.float32
            ).copy(),
            remaining_budget=float(observation["remaining_budget"]),
            normalised_step=float(observation["normalised_step"]),
            target_answer=int(episode.true_answer),
            target_dependency_matrix=dependency.astype(np.float32),
            target_dependency_mask=dependency_mask,
            target_lineage_labels=episode.tool_lineages.astype(np.int64).copy(),
            target_effective_evidence=compute_effective_evidence(
                episode.tool_lineages, queried
            ),
            target_observed_agreement=observed_agreement(episode.tool_claims, queried),
            target_false_consensus_by_answer=false_consensus,
        )

    def _false_consensus_for_answer(
        self, env: HiddenDependencyEnv, answer: int
    ) -> bool:
        episode = env._require_episode()
        queried = episode.queried_mask
        if queried.sum() < 2 or answer == episode.true_answer:
            return False
        agreement = float(np.mean(episode.tool_claims[queried] == answer))
        effective = compute_effective_evidence(episode.tool_lineages, queried)
        return bool(
            agreement
            >= self.environment_config.false_consensus_agreement_threshold
            and effective
            <= self.environment_config.false_consensus_max_effective_evidence
        )

    def _split_counts(self) -> dict[str, int]:
        cfg = self.generation_config
        return {
            "train": cfg.train_episodes_per_scenario,
            "validation": cfg.validation_episodes_per_scenario,
            "test": cfg.test_episodes_per_scenario,
        }

    def _episode_seed(self, split: str, scenario_index: int, local_index: int) -> int:
        split_offset = {"train": 0, "validation": 1_000_000, "test": 2_000_000}
        return int(
            self.generation_config.base_seed
            + split_offset[split]
            + scenario_index * 10_000
            + local_index
        )

    def _limit_reached(self, records: list[EvidenceTrainingRecord]) -> bool:
        maximum = self.generation_config.max_prefixes_per_episode
        return maximum is not None and len(records) >= maximum

    def _manifest(
        self, datasets: dict[str, list[EvidenceTrainingRecord]]
    ) -> dict:
        scenario_counts: dict[str, dict[str, int]] = {}
        for split, records in datasets.items():
            counts = {scenario.value: 0 for scenario in ScenarioType}
            for record in records:
                counts[record.scenario] += 1
            scenario_counts[split] = counts
        return {
            "format_version": 1,
            "environment_config": asdict(self.environment_config),
            "generation_config": asdict(self.generation_config),
            "records_per_split": {
                split: len(records) for split, records in datasets.items()
            },
            "records_per_scenario": scenario_counts,
            "feature_fields": [
                "tool_claims",
                "tool_confidences",
                "queried_mask",
                "diagnosed_mask",
                "metadata_signatures",
                "diagnostic_matrix",
                "remaining_budget",
                "normalised_step",
            ],
            "target_fields": [
                "answer",
                "dependency_matrix",
                "dependency_mask",
                "lineage_labels",
                "effective_evidence",
                "observed_agreement",
                "false_consensus_by_answer",
            ],
        }


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
