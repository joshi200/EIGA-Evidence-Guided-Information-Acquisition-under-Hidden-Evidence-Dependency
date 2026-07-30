from pathlib import Path

from eiga.data.torch_dataset import InMemoryEvidenceDataset
from eiga.models.latent_ebm import LatentEBMConfig, LatentEvidenceBeliefModule
from eiga.training.independent_trainer import TrainingConfig
from eiga.training.latent_trainer import LatentEBMTrainer
import torch


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


def records_from_batch():
    batch = make_batch(batch_size=4)
    records = []
    for i in range(4):
        records.append({
            "features": {
                "tool_claims": batch["tool_claims"][i].tolist(),
                "tool_confidences": batch["tool_confidences"][i].tolist(),
                "queried_mask": batch["queried_mask"][i].tolist(),
                "diagnosed_mask": batch["diagnosed_mask"][i].tolist(),
                "metadata_signatures": batch["metadata_signatures"][i].tolist(),
                "diagnostic_matrix": batch["diagnostic_matrix"][i].tolist(),
                "remaining_budget": 1.0,
                "normalised_step": .5,
            },
            "targets": {
                "answer": 1,
                "dependency_matrix": batch["target_dependency_matrix"][i].tolist(),
                "dependency_mask": batch["target_dependency_mask"][i].tolist(),
                "lineage_labels": [0, 0, 1, 2, 3, 4],
                "effective_evidence": 2,
                "observed_agreement": 2 / 3,
                "false_consensus_by_answer": [0, 0],
            },
        })
    return records


def test_latent_trainer_saves_checkpoint(tmp_path: Path):
    dataset = InMemoryEvidenceDataset(records_from_batch())
    model = LatentEvidenceBeliefModule(LatentEBMConfig(hidden_dim=24, claim_embedding_dim=8, metadata_embedding_dim=6, slot_iterations=1))
    trainer = LatentEBMTrainer(model, TrainingConfig(epochs=1, batch_size=2, patience=1, device="cpu"))
    summary = trainer.fit(dataset, dataset, tmp_path)
    assert summary["epochs_completed"] == 1
    assert (tmp_path / "best.pt").exists()
    assert "dependency_accuracy" in trainer.evaluate(dataset)
