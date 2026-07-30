import torch

from eiga.models.pairwise_ebm import (
    PairwiseEBMConfig,
    PairwiseEvidenceBeliefModule,
    pairwise_ebm_loss,
)


def make_batch(batch_size: int = 3, num_tools: int = 6) -> dict[str, torch.Tensor]:
    queried = torch.tensor([[1, 1, 1, 0, 0, 0]] * batch_size, dtype=torch.bool)
    dependency = torch.zeros(batch_size, num_tools, num_tools)
    dependency[:, 0, 0] = 1
    dependency[:, 1, 1] = 1
    dependency[:, 2, 2] = 1
    dependency[:, 0, 1] = dependency[:, 1, 0] = 1
    return {
        "tool_claims": torch.tensor([[1, 1, 0, -1, -1, -1]] * batch_size),
        "tool_confidences": torch.rand(batch_size, num_tools),
        "queried_mask": queried,
        "diagnosed_mask": torch.zeros(batch_size, num_tools, dtype=torch.bool),
        "metadata_signatures": torch.tensor([[0, 0, 2, -1, -1, -1]] * batch_size),
        "diagnostic_matrix": torch.zeros(batch_size, num_tools, num_tools),
        "remaining_budget": torch.ones(batch_size),
        "normalised_step": torch.full((batch_size,), 0.5),
        "target_answer": torch.ones(batch_size, dtype=torch.long),
        "target_dependency_matrix": dependency,
        "target_dependency_mask": queried[:, :, None] & queried[:, None, :],
        "target_effective_evidence": torch.full((batch_size,), 2, dtype=torch.long),
        "target_observed_agreement": torch.full((batch_size,), 2 / 3),
        "target_false_consensus_by_answer": torch.zeros(batch_size, 2),
    }


def test_output_shapes_and_symmetry() -> None:
    model = PairwiseEvidenceBeliefModule(PairwiseEBMConfig())
    output = model(make_batch())
    assert output.answer_logits.shape == (3, 2)
    assert output.dependency_probabilities.shape == (3, 6, 6)
    assert torch.allclose(
        output.dependency_probabilities,
        output.dependency_probabilities.transpose(1, 2),
        atol=1e-6,
    )


def test_unqueried_tools_have_zero_weight() -> None:
    model = PairwiseEvidenceBeliefModule(PairwiseEBMConfig())
    output = model(make_batch())
    assert torch.all(output.evidence_weights[:, 3:] == 0)
    assert torch.allclose(output.evidence_weights.sum(dim=-1), torch.ones(3))


def test_loss_is_finite_and_backpropagates() -> None:
    model = PairwiseEvidenceBeliefModule(PairwiseEBMConfig())
    batch = make_batch()
    output = model(batch)
    loss, components = pairwise_ebm_loss(output, batch)
    assert torch.isfinite(loss)
    assert "dependency_loss" in components
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tool_permutation_equivariance() -> None:
    model = PairwiseEvidenceBeliefModule(PairwiseEBMConfig(dropout=0.0)).eval()
    batch = make_batch(batch_size=1)
    permutation = torch.tensor([2, 0, 1, 3, 4, 5])
    inverse = torch.argsort(permutation)
    permuted = {key: value.clone() for key, value in batch.items()}
    for key in ["tool_claims", "tool_confidences", "queried_mask", "diagnosed_mask", "metadata_signatures"]:
        permuted[key] = batch[key][:, permutation]
    for key in ["diagnostic_matrix", "target_dependency_matrix", "target_dependency_mask"]:
        permuted[key] = batch[key][:, permutation][:, :, permutation]
    with torch.no_grad():
        original = model(batch)
        changed = model(permuted)
    restored = changed.dependency_probabilities[:, inverse][:, :, inverse]
    assert torch.allclose(original.dependency_probabilities, restored, atol=1e-5)
    assert torch.allclose(original.answer_logits, changed.answer_logits, atol=1e-5)
