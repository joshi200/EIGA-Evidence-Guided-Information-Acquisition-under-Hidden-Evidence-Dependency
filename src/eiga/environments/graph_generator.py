from dataclasses import dataclass

import numpy as np

from .config import EnvironmentConfig
from .episode import HiddenEpisode
from .scenarios import ScenarioType


@dataclass
class ScenarioGraphGenerator:
    config: EnvironmentConfig

    def generate(
        self,
        rng: np.random.Generator,
        scenario: ScenarioType | None = None,
    ) -> HiddenEpisode:
        scenario = scenario or self.config.default_scenario
        true_answer = int(rng.integers(0, 2))
        wrong_answer = 1 - true_answer

        lineages, claims = self._construct_scenario(
            scenario=scenario,
            true_answer=true_answer,
            wrong_answer=wrong_answer,
            rng=rng,
        )
        lineages, claims = self._shuffle_tools(lineages, claims, rng)

        source_claims = self._source_claims_from_tools(lineages, claims)
        confidences = np.clip(
            np.where(claims == true_answer, 0.78, 0.82)
            + rng.normal(0.0, 0.06, self.config.num_tools),
            0.50,
            0.99,
        )
        metadata = self._generate_metadata(lineages, rng)
        n = self.config.num_tools

        return HiddenEpisode(
            scenario=scenario,
            true_answer=true_answer,
            tool_lineages=lineages,
            source_claims=source_claims,
            tool_claims=claims,
            tool_confidences=confidences.astype(np.float32),
            metadata_signatures=metadata,
            queried_mask=np.zeros(n, dtype=bool),
            diagnosed_mask=np.zeros(n, dtype=bool),
            diagnostic_matrix=np.full((n, n), -1.0, dtype=np.float32),
            remaining_budget=self.config.initial_budget,
        )

    def _construct_scenario(
        self,
        scenario: ScenarioType,
        true_answer: int,
        wrong_answer: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.config.num_tools

        if scenario is ScenarioType.INDEPENDENT_AGREEMENT:
            lineages = np.arange(n)
            claims = np.full(n, true_answer)

        elif scenario is ScenarioType.INDEPENDENT_MIXED:
            lineages = np.arange(n)
            claims = np.full(n, true_answer)
            wrong_count = max(1, n // 3)
            claims[:wrong_count] = wrong_answer

        elif scenario is ScenarioType.CORRELATED_CORRECT:
            majority = max(2, int(np.ceil(0.75 * n)))
            lineages = np.concatenate(
                [np.zeros(majority, dtype=int), np.arange(1, n - majority + 1)]
            )
            claims = np.full(n, true_answer)

        elif scenario is ScenarioType.CORRELATED_INCORRECT:
            majority = max(2, int(np.ceil(0.75 * n)))
            lineages = np.concatenate(
                [np.zeros(majority, dtype=int), np.arange(1, n - majority + 1)]
            )
            claims = np.concatenate(
                [np.full(majority, wrong_answer), np.full(n - majority, true_answer)]
            )

        elif scenario is ScenarioType.MINORITY_INDEPENDENT_TRUTH:
            false_majority = max(3, int(np.ceil(0.60 * n)))
            true_count = n - false_majority
            if true_count < 1:
                raise ValueError("minority truth requires at least one true tool")
            lineages = np.concatenate(
                [np.zeros(false_majority, dtype=int), np.arange(1, true_count + 1)]
            )
            claims = np.concatenate(
                [np.full(false_majority, wrong_answer), np.full(true_count, true_answer)]
            )

        elif scenario is ScenarioType.BALANCED_CONFLICT:
            true_count = n // 2
            false_count = n - true_count
            lineages = np.arange(n)
            claims = np.concatenate(
                [np.full(true_count, true_answer), np.full(false_count, wrong_answer)]
            )

        elif scenario is ScenarioType.MIXED_DEPENDENCY:
            # Two duplicated groups plus independent tools. Claims are lineage-consistent.
            group_a = max(2, n // 3)
            group_b = max(2, n // 3)
            if group_a + group_b > n:
                group_b = n - group_a
            remainder = n - group_a - group_b
            lineages = np.concatenate(
                [
                    np.zeros(group_a, dtype=int),
                    np.ones(group_b, dtype=int),
                    np.arange(2, 2 + remainder),
                ]
            )
            claims = np.concatenate(
                [
                    np.full(group_a, true_answer),
                    np.full(group_b, wrong_answer),
                    rng.integers(0, 2, size=remainder),
                ]
            )
        else:
            raise ValueError(f"unsupported scenario: {scenario}")

        return lineages.astype(np.int64), claims.astype(np.int64)

    @staticmethod
    def _shuffle_tools(
        lineages: np.ndarray, claims: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        order = rng.permutation(lineages.size)
        return lineages[order], claims[order]

    @staticmethod
    def _source_claims_from_tools(
        lineages: np.ndarray, claims: np.ndarray
    ) -> np.ndarray:
        values = []
        for lineage in sorted(np.unique(lineages).tolist()):
            lineage_claims = np.unique(claims[lineages == lineage])
            if lineage_claims.size != 1:
                raise RuntimeError("all tools in a lineage must share one claim")
            values.append(int(lineage_claims[0]))
        return np.asarray(values, dtype=np.int64)

    def _generate_metadata(
        self, lineages: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        unique_lineages = np.unique(lineages)
        max_signature = max(self.config.num_tools, unique_lineages.size + 1)
        signatures = np.empty_like(lineages)
        for i, lineage in enumerate(lineages):
            if rng.random() < self.config.metadata_accuracy:
                signatures[i] = lineage
            else:
                alternatives = [x for x in range(max_signature) if x != lineage]
                signatures[i] = int(rng.choice(alternatives))
        return signatures.astype(np.int64)
