import torch

from eiga.models.independent_ebm import (
    IndependentEBMConfig,
    IndependentEvidenceBeliefModule,
    independent_ebm_loss,
)


def make_batch(batch_size: int = 4, num_tools: int = 6):
    return {
        "tool_claims": torch.randint(-1, 2, (batch_size, num_tools)),
        "tool_confidences": torch.rand(batch_size, num_tools),
        "queried_mask": torch.randint(0, 2, (batch_size, num_tools)).bool(),
        "diagnosed_mask": torch.randint(0, 2, (batch_size, num_tools)).bool(),
        "remaining_budget": torch.rand(batch_size),
        "normalised_step": torch.rand(batch_size),
        "target_answer": torch.randint(0, 2, (batch_size,)),
        "target_effective_evidence": torch.randint(0, num_tools + 1, (batch_size,)),
        "target_observed_agreement": torch.rand(batch_size),
        "target_false_consensus_by_answer": torch.randint(0, 2, (batch_size, 2)).float(),
    }


def test_forward_shapes():
    model = IndependentEvidenceBeliefModule(IndependentEBMConfig())
    output = model(make_batch())
    assert output.answer_logits.shape == (4, 2)
    assert output.effective_evidence_logits.shape == (4, 7)
    assert output.false_consensus_logits.shape == (4, 2)
    assert output.observed_agreement.shape == (4,)
    assert output.belief_embedding.shape == (4, 128)


def test_loss_is_finite_and_backpropagates():
    model = IndependentEvidenceBeliefModule(IndependentEBMConfig())
    batch = make_batch()
    output = model(batch)
    loss, components = independent_ebm_loss(output, batch)
    assert torch.isfinite(loss)
    assert set(components) == {
        "loss", "answer_loss", "effective_evidence_loss",
        "false_consensus_loss", "agreement_loss"
    }
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_permutation_invariance():
    torch.manual_seed(4)
    model = IndependentEvidenceBeliefModule(IndependentEBMConfig(dropout=0.0)).eval()
    batch = make_batch(batch_size=2)
    permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    permuted = dict(batch)
    for key in ["tool_claims", "tool_confidences", "queried_mask", "diagnosed_mask"]:
        permuted[key] = batch[key][:, permutation]
    with torch.no_grad():
        first = model(batch)
        second = model(permuted)
    assert torch.allclose(first.answer_logits, second.answer_logits, atol=1e-6)
    assert torch.allclose(
        first.effective_evidence_logits, second.effective_evidence_logits, atol=1e-6
    )
