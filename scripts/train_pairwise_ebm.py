from __future__ import annotations

import argparse
from pathlib import Path

from eiga.data.torch_dataset import EvidenceJsonlDataset
from eiga.models.pairwise_ebm import PairwiseEBMConfig, PairwiseEvidenceBeliefModule
from eiga.training.pairwise_trainer import PairwiseEBMTrainer, PairwiseTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pairwise EBM")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/pairwise_ebm"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = EvidenceJsonlDataset(args.data / "train.jsonl")
    validation = EvidenceJsonlDataset(args.data / "validation.jsonl")
    test = EvidenceJsonlDataset(args.data / "test.jsonl")
    sample = train[0]
    model = PairwiseEvidenceBeliefModule(PairwiseEBMConfig(
        num_tools=int(sample["tool_claims"].shape[0]),
        num_answer_classes=int(sample["target_false_consensus_by_answer"].shape[0]),
        hidden_dim=args.hidden_dim,
    ))
    trainer = PairwiseEBMTrainer(model, PairwiseTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    ))
    summary = trainer.fit(train, validation, args.output)
    metrics = trainer.evaluate(test)
    print(f"Best validation loss: {summary['best_validation_loss']:.4f}")
    print("Test metrics:")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
