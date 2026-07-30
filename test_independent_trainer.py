from pathlib import Path

from eiga.data.generator import DatasetGenerationConfig, SupervisedDatasetGenerator
from eiga.data.torch_dataset import InMemoryEvidenceDataset
from eiga.models.independent_ebm import IndependentEBMConfig, IndependentEvidenceBeliefModule
from eiga.training.independent_trainer import IndependentEBMTrainer, TrainingConfig


def test_trainer_writes_checkpoint(tmp_path: Path):
    generated = SupervisedDatasetGenerator(
        generation_config=DatasetGenerationConfig(
            train_episodes_per_scenario=1,
            validation_episodes_per_scenario=1,
            test_episodes_per_scenario=0,
            max_prefixes_per_episode=2,
        )
    ).generate()
    train = InMemoryEvidenceDataset([r.to_json_dict() for r in generated["train"]])
    validation = InMemoryEvidenceDataset(
        [r.to_json_dict() for r in generated["validation"]]
    )
    model = IndependentEvidenceBeliefModule(
        IndependentEBMConfig(hidden_dim=32, claim_embedding_dim=8, dropout=0.0)
    )
    trainer = IndependentEBMTrainer(
        model,
        TrainingConfig(epochs=2, batch_size=8, patience=2, device="cpu"),
    )
    summary = trainer.fit(train, validation, tmp_path)
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "history.json").exists()
    assert summary["epochs_completed"] >= 1
