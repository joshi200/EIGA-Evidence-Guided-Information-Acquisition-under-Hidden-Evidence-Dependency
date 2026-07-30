from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvidenceTrainingRecord:
    """One agent-visible episode prefix paired with privileged training labels.

    All feature fields mirror the environment observation. Oracle information is
    kept only in explicitly named target fields so training code can audit data
    leakage easily.
    """

    episode_id: int
    prefix_id: int
    seed: int
    split: str
    scenario: str

    tool_claims: np.ndarray
    tool_confidences: np.ndarray
    queried_mask: np.ndarray
    diagnosed_mask: np.ndarray
    metadata_signatures: np.ndarray
    diagnostic_matrix: np.ndarray
    remaining_budget: float
    normalised_step: float

    target_answer: int
    target_dependency_matrix: np.ndarray
    target_dependency_mask: np.ndarray
    target_lineage_labels: np.ndarray
    target_effective_evidence: int
    target_observed_agreement: float
    target_false_consensus_by_answer: np.ndarray

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "prefix_id": self.prefix_id,
            "seed": self.seed,
            "split": self.split,
            "scenario": self.scenario,
            "features": {
                "tool_claims": self.tool_claims.tolist(),
                "tool_confidences": self.tool_confidences.tolist(),
                "queried_mask": self.queried_mask.astype(bool).tolist(),
                "diagnosed_mask": self.diagnosed_mask.astype(bool).tolist(),
                "metadata_signatures": self.metadata_signatures.tolist(),
                "diagnostic_matrix": self.diagnostic_matrix.tolist(),
                "remaining_budget": self.remaining_budget,
                "normalised_step": self.normalised_step,
            },
            "targets": {
                "answer": self.target_answer,
                "dependency_matrix": self.target_dependency_matrix.tolist(),
                "dependency_mask": self.target_dependency_mask.astype(bool).tolist(),
                "lineage_labels": self.target_lineage_labels.tolist(),
                "effective_evidence": self.target_effective_evidence,
                "observed_agreement": self.target_observed_agreement,
                "false_consensus_by_answer": (
                    self.target_false_consensus_by_answer.astype(bool).tolist()
                ),
            },
        }
