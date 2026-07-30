from __future__ import annotations

import argparse
from pathlib import Path

from eiga.data.torch_dataset import EvidenceJsonlDataset
from eiga.models.independent_ebm import IndependentEBMConfig, IndependentEvidenceBeliefModule
from eiga.training.independent_trainer import IndependentEBMTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the independent EBM baseline")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/independent_ebm"))
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
    example = train[0]
    num_tools = int(example["tool_claims"].shape[0])
    num_answers = int(example["target_false_consensus_by_answer"].shape[0])
    model = IndependentEvidenceBeliefModule(
        IndependentEBMConfig(
            num_tools=num_tools,
            num_answer_classes=num_answers,
            hidden_dim=args.hidden_dim,
        )
    )
    trainer = IndependentEBMTrainer(
        model,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        ),
    )
    summary = trainer.fit(train, validation, args.output)
    test_metrics = trainer.evaluate(test)
    print(f"Best validation loss: {summary['best_validation_loss']:.4f}")
    print("Test metrics:")
    for name, value in sorted(test_metrics.items()):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
