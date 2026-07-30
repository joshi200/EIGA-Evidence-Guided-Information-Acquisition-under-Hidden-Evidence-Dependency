from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from eiga.data.torch_dataset import EvidenceJsonlDataset
from eiga.models.latent_ebm import (
    LatentEBMConfig,
    LatentEBMLossConfig,
    LatentEvidenceBeliefModule,
)
from eiga.training.independent_trainer import TrainingConfig
from eiga.training.latent_trainer import LatentEBMTrainer


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def experiment_grid(
    base_model: LatentEBMConfig,
    base_loss: LatentEBMLossConfig,
) -> list[tuple[str, LatentEBMConfig, LatentEBMLossConfig, str]]:
    experiments: list[tuple[str, LatentEBMConfig, LatentEBMLossConfig, str]] = [
        ("full", base_model, base_loss, "Full latent EBM"),
        (
            "no_dependency_loss",
            base_model,
            replace(base_loss, dependency_weight=0.0),
            "Remove dependency supervision",
        ),
        (
            "no_false_consensus_loss",
            base_model,
            replace(base_loss, false_consensus_weight=0.0),
            "Remove false-consensus supervision",
        ),
        (
            "no_occupancy_loss",
            base_model,
            replace(base_loss, occupancy_weight=0.0),
            "Remove slot-occupancy regularisation",
        ),
        (
            "no_entropy_regularisation",
            base_model,
            replace(base_loss, assignment_entropy_weight=0.0),
            "Remove assignment-entropy regularisation",
        ),
        (
            "answer_only",
            base_model,
            LatentEBMLossConfig(
                answer_weight=1.0,
                dependency_weight=0.0,
                effective_evidence_weight=0.0,
                occupancy_weight=0.0,
                false_consensus_weight=0.0,
                agreement_weight=0.0,
                assignment_entropy_weight=0.0,
            ),
            "Answer prediction only",
        ),
        (
            "slot_iterations_1",
            replace(base_model, slot_iterations=1),
            base_loss,
            "One slot-refinement iteration",
        ),
    ]

    for slots in (2, 4, 6, 8):
        experiments.append((
            f"latent_slots_{slots}",
            replace(base_model, max_latent_sources=slots),
            base_loss,
            f"Use {slots} latent slots",
        ))

    for temperature in (0.25, 0.50, 0.75, 1.00):
        experiments.append((
            f"temperature_{str(temperature).replace('.', '_')}",
            replace(base_model, assignment_temperature=temperature),
            base_loss,
            f"Assignment temperature {temperature:.2f}",
        ))

    # Remove exact duplicates of the full configuration (slots=6 and temp=0.75).
    unique: list[tuple[str, LatentEBMConfig, LatentEBMLossConfig, str]] = []
    seen: set[tuple[tuple, tuple]] = set()
    for name, model_cfg, loss_cfg, description in experiments:
        key = (tuple(asdict(model_cfg).items()), tuple(asdict(loss_cfg).items()))
        if key not in seen:
            seen.add(key)
            unique.append((name, model_cfg, loss_cfg, description))
    return unique


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["experiment"]), []).append(row)

    metrics = [
        "test_loss",
        "answer_accuracy",
        "dependency_accuracy",
        "effective_evidence_accuracy",
        "effective_evidence_mae",
        "false_consensus_accuracy",
        "observed_agreement_mae",
        "best_validation_loss",
        "duration_seconds",
    ]
    output: list[dict[str, Any]] = []
    for experiment, group in grouped.items():
        summary: dict[str, Any] = {
            "experiment": experiment,
            "description": group[0]["description"],
            "seeds": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Latent EBM ablation results on the held-out test set. Values are mean $\pm$ standard deviation across seeds. Higher is better for accuracy; lower is better for MAE.}",
        r"\label{tab:latent-ablation}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & Answer Acc. & Dependency Acc. & Eff. Evidence MAE & False Consensus Acc. \\",
        r"\midrule",
    ]
    for row in rows:
        name = str(row["experiment"]).replace("_", r"\_")
        lines.append(
            f"{name} & "
            f"{100 * row['answer_accuracy_mean']:.2f} $\\pm$ {100 * row['answer_accuracy_std']:.2f} & "
            f"{100 * row['dependency_accuracy_mean']:.2f} $\\pm$ {100 * row['dependency_accuracy_std']:.2f} & "
            f"{row['effective_evidence_mae_mean']:.3f} $\\pm$ {row['effective_evidence_mae_std']:.3f} & "
            f"{100 * row['false_consensus_accuracy_mean']:.2f} $\\pm$ {100 * row['false_consensus_accuracy_std']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Latent EBM ablations")
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latent_ablations"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/latent_ebm.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training/latent_ebm.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 123])
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment names. Omit to run the full grid.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = EvidenceJsonlDataset(args.data_dir / "train.jsonl")
    validation = EvidenceJsonlDataset(args.data_dir / "validation.jsonl")
    test = EvidenceJsonlDataset(args.data_dir / "test.jsonl")

    base_model = LatentEBMConfig(**load_yaml(args.model_config))
    base_training = TrainingConfig(**load_yaml(args.training_config))
    base_loss = LatentEBMLossConfig()
    grid = experiment_grid(base_model, base_loss)
    if args.experiments:
        requested = set(args.experiments)
        grid = [item for item in grid if item[0] in requested]
        missing = requested - {item[0] for item in grid}
        if missing:
            raise ValueError(f"Unknown experiments: {sorted(missing)}")

    all_rows: list[dict[str, Any]] = []
    for experiment, model_cfg, loss_cfg, description in grid:
        for seed in args.seeds:
            print(f"\n=== {experiment} | seed={seed} ===", flush=True)
            training_cfg = replace(base_training, seed=seed)
            run_dir = args.output_dir / experiment / f"seed_{seed}"
            model = LatentEvidenceBeliefModule(model_cfg)
            trainer = LatentEBMTrainer(model, training_cfg, loss_cfg)
            training_summary = trainer.fit(train, validation, run_dir)
            test_metrics = trainer.evaluate(test)
            row: dict[str, Any] = {
                "experiment": experiment,
                "description": description,
                "seed": seed,
                "checkpoint": str(run_dir / "best.pt"),
                "best_validation_loss": training_summary["best_validation_loss"],
                "best_epoch": training_summary["best_epoch"],
                "epochs_completed": training_summary["epochs_completed"],
                "duration_seconds": training_summary["duration_seconds"],
                "test_loss": test_metrics["loss"],
                "answer_accuracy": test_metrics["answer_accuracy"],
                "dependency_accuracy": test_metrics["dependency_accuracy"],
                "effective_evidence_accuracy": test_metrics["effective_evidence_accuracy"],
                "effective_evidence_mae": test_metrics["effective_evidence_mae"],
                "false_consensus_accuracy": test_metrics["false_consensus_accuracy"],
                "observed_agreement_mae": test_metrics["observed_agreement_mae"],
                **{f"model_{k}": v for k, v in asdict(model_cfg).items()},
                **{f"loss_{k}": v for k, v in asdict(loss_cfg).items()},
            }
            all_rows.append(row)
            (run_dir / "test_metrics.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            write_csv(args.output_dir / "raw_results.csv", all_rows)

    aggregated = aggregate(all_rows)
    write_csv(args.output_dir / "summary.csv", aggregated)
    (args.output_dir / "summary.json").write_text(
        json.dumps(aggregated, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ablation_table.tex").write_text(
        latex_table(aggregated), encoding="utf-8"
    )
    print(f"\nFinished. Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
