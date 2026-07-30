from pathlib import Path

from eiga.data.generator import DatasetGenerationConfig, SupervisedDatasetGenerator
from eiga.data.torch_dataset import InMemoryEvidenceDataset
from eiga.models.pairwise_ebm import PairwiseEBMConfig, PairwiseEvidenceBeliefModule
from eiga.training.pairwise_trainer import PairwiseEBMTrainer, PairwiseTrainingConfig


def test_pairwise_trainer_smoke(tmp_path: Path) -> None:
    generator = SupervisedDatasetGenerator(
        generation_config=DatasetGenerationConfig(
            train_episodes_per_scenario=2,
            validation_episodes_per_scenario=1,
            test_episodes_per_scenario=0,
            base_seed=9,
        )
    )
    generated = generator.generate()
    train = InMemoryEvidenceDataset([r.to_json_dict() for r in generated["train"]])
    validation = InMemoryEvidenceDataset([r.to_json_dict() for r in generated["validation"]])
    model = PairwiseEvidenceBeliefModule(PairwiseEBMConfig())
    trainer = PairwiseEBMTrainer(model, PairwiseTrainingConfig(
        epochs=1, batch_size=8, patience=1, device="cpu"
    ))
    summary = trainer.fit(train, validation, tmp_path)
    assert summary["epochs_completed"] == 1
    assert (tmp_path / "best.pt").exists()
    assert "dependency_accuracy" in trainer.evaluate(validation)
