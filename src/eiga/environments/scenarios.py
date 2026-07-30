from enum import Enum


class ScenarioType(str, Enum):
    """Controlled hidden-dependency regimes used by the benchmark."""

    INDEPENDENT_AGREEMENT = "independent_agreement"
    INDEPENDENT_MIXED = "independent_mixed"
    CORRELATED_CORRECT = "correlated_correct"
    CORRELATED_INCORRECT = "correlated_incorrect"
    MINORITY_INDEPENDENT_TRUTH = "minority_independent_truth"
    BALANCED_CONFLICT = "balanced_conflict"
    MIXED_DEPENDENCY = "mixed_dependency"
