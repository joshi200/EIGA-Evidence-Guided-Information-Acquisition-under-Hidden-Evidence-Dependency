import torch

from eiga.models.latent_ebm import (
    LatentEBMConfig,
    LatentEvidenceBeliefModule,
    latent_ebm_loss,
)


def make_batch(batch_size: int = 3, num_tools: int = 6):
    queried = torch.tensor([[1, 1, 1, 0, 0, 0]] * batch_size, dtype=torch.bool)
    dependency = torch.zeros(batch_size, num_tools, num_tools)
    dependency[:, 0, 0] = dependency[:, 1, 1] = dependency[:, 2, 2] = 1
    dependency[:, 0, 1] = dependency[:, 1, 0] = 1
    return {
        "tool_claims": torch.tensor([[1, 1, 0, -1, -1, -1]] * batch_size),
        "tool_confidences": torch.tensor([[.8, .7, .9, 0, 0, 0]] * batch_size),
        "queried_mask": queried,
        "diagnosed_mask": torch.zeros(batch_size, num_tools, dtype=torch.bool),
        "metadata_signatures": torch.tensor([[0, 0, 1, -1, -1, -1]] * batch_size),
        "diagnostic_matrix": torch.zeros(batch_size, num_tools, num_tools),
        "remaining_budget": torch.ones(batch_size),
        "normalised_step": torch.full((batch_size,), .5),
        "target_answer": torch.ones(batch_size, dtype=torch.long),
        "target_dependency_matrix": dependency,
        "target_dependency_mask": queried[:, :, None] & queried[:, None, :],
        "target_effective_evidence": torch.full((batch_size,), 2, dtype=torch.long),
        "target_observed_agreement": torch.full((batch_size,), 2 / 3),
        "target_false_consensus_by_answer": torch.zeros(batch_size, 2),
    }


def test_latent_model_output_shapes():
    model = LatentEvidenceBeliefModule(LatentEBMConfig())
    output = model(make_batch())
    assert output.answer_logits.shape == (3, 2)
    assert output.assignment_probabilities.shape == (3, 6, 6)
    assert output.dependency_probabilities.shape == (3, 6, 6)
    assert output.slot_occupancy.shape == (3, 6)


def test_unqueried_tools_have_zero_assignment_mass():
    model = LatentEvidenceBeliefModule(LatentEBMConfig())
    output = model(make_batch())
    assert torch.allclose(output.assignment_probabilities[:, 3:], torch.zeros_like(output.assignment_probabilities[:, 3:]))


def test_dependency_matrix_is_symmetric_and_diagonal_one_for_queried():
    model = LatentEvidenceBeliefModule(LatentEBMConfig())
    output = model(make_batch())
    assert torch.allclose(output.dependency_probabilities, output.dependency_probabilities.transpose(1, 2), atol=1e-6)
    diagonal = output.dependency_probabilities.diagonal(dim1=1, dim2=2)
    assert torch.all(diagonal[:, :3] == 1)
    assert torch.all(diagonal[:, 3:] == 0)


def test_latent_loss_is_finite_and_backpropagates():
    model = LatentEvidenceBeliefModule(LatentEBMConfig())
    output = model(make_batch())
    loss, components = latent_ebm_loss(output, make_batch())
    assert torch.isfinite(loss)
    assert "dependency_loss" in components
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tool_permutation_preserves_global_prediction_in_eval_mode():
    torch.manual_seed(4)
    model = LatentEvidenceBeliefModule(LatentEBMConfig(dropout=0.0)).eval()
    batch = make_batch(batch_size=1)
    permutation = torch.tensor([2, 0, 1, 3, 4, 5])
    permuted = dict(batch)
    for key in ["tool_claims", "tool_confidences", "queried_mask", "diagnosed_mask", "metadata_signatures"]:
        permuted[key] = batch[key][:, permutation]
    permuted["diagnostic_matrix"] = batch["diagnostic_matrix"][:, permutation][:, :, permutation]
    with torch.no_grad():
        original = model(batch).answer_logits
        changed = model(permuted).answer_logits
    assert torch.allclose(original, changed, atol=1e-5)
