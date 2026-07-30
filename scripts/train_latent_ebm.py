from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eiga.data.torch_dataset import EvidenceJsonlDataset
from eiga.models.latent_ebm import LatentEBMConfig, LatentEvidenceBeliefModule
from eiga.training.independent_trainer import TrainingConfig
from eiga.training.latent_trainer import LatentEBMTrainer


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the latent EIGA belief module")
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latent_ebm"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/latent_ebm.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training/latent_ebm.yaml"))
    args = parser.parse_args()

    model = LatentEvidenceBeliefModule(LatentEBMConfig(**load_yaml(args.model_config)))
    trainer = LatentEBMTrainer(model, TrainingConfig(**load_yaml(args.training_config)))
    train = EvidenceJsonlDataset(args.data_dir / "train.jsonl")
    validation = EvidenceJsonlDataset(args.data_dir / "validation.jsonl")
    test = EvidenceJsonlDataset(args.data_dir / "test.jsonl")
    summary = trainer.fit(train, validation, args.output_dir)
    metrics = trainer.evaluate(test)
    print({"training": summary, "test": metrics})


if __name__ == "__main__":
    main()
