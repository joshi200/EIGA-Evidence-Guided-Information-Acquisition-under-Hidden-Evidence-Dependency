import numpy as np


def build_dependency_matrix(tool_lineages: np.ndarray) -> np.ndarray:
    return (tool_lineages[:, None] == tool_lineages[None, :]).astype(np.float32)


def compute_effective_evidence(
    tool_lineages: np.ndarray, queried_mask: np.ndarray
) -> int:
    if not queried_mask.any():
        return 0
    return int(np.unique(tool_lineages[queried_mask]).size)


def observed_agreement(tool_claims: np.ndarray, queried_mask: np.ndarray) -> float:
    claims = tool_claims[queried_mask]
    if claims.size == 0:
        return 0.0
    counts = np.bincount(claims, minlength=2)
    return float(counts.max() / claims.size)
