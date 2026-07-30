from dataclasses import dataclass

from .scenarios import ScenarioType


@dataclass(frozen=True)
class EnvironmentConfig:
    num_tools: int = 6
    num_answer_classes: int = 2
    initial_budget: float = 5.0
    query_cost: float = 1.0
    diagnostic_cost: float = 0.5
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    abstain_reward: float = -0.2
    false_consensus_penalty: float = -0.5
    metadata_accuracy: float = 0.75
    diagnostic_accuracy: float = 0.90
    max_steps: int = 10
    false_consensus_agreement_threshold: float = 0.75
    false_consensus_max_effective_evidence: int = 1
    default_scenario: ScenarioType = ScenarioType.MIXED_DEPENDENCY

    def __post_init__(self) -> None:
        if self.num_tools < 2:
            raise ValueError("num_tools must be at least 2")
        if self.num_answer_classes != 2:
            raise ValueError("phase 7.2 currently supports binary answers only")
        if self.initial_budget <= 0:
            raise ValueError("initial_budget must be positive")
        if self.query_cost <= 0 or self.diagnostic_cost <= 0:
            raise ValueError("action costs must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        for name, value in (
            ("metadata_accuracy", self.metadata_accuracy),
            ("diagnostic_accuracy", self.diagnostic_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
