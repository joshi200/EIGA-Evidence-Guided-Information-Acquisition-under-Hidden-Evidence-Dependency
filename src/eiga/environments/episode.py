from dataclasses import dataclass, field

import numpy as np

from .scenarios import ScenarioType


@dataclass
class HiddenEpisode:
    scenario: ScenarioType
    true_answer: int
    tool_lineages: np.ndarray
    source_claims: np.ndarray
    tool_claims: np.ndarray
    tool_confidences: np.ndarray
    metadata_signatures: np.ndarray
    queried_mask: np.ndarray
    diagnosed_mask: np.ndarray
    diagnostic_matrix: np.ndarray
    remaining_budget: float
    current_step: int = 0
    terminated: bool = False
    terminal_reason: str | None = None
    submitted_answer: int | None = None
    action_history: list[dict] = field(default_factory=list)

    @property
    def num_lineages(self) -> int:
        return int(np.unique(self.tool_lineages).size)
