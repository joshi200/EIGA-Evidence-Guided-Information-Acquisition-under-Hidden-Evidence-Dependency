from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eiga.data.torch_dataset import move_batch_to_device
from eiga.models.independent_ebm import (
    IndependentEBMLossConfig,
    IndependentEvidenceBeliefModule,
    independent_ebm_loss,
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    patience: int = 5
    seed: int = 17
    num_workers: int = 0
    device: str = "auto"


class IndependentEBMTrainer:
    def __init__(
        self,
        model: IndependentEvidenceBeliefModule,
        config: TrainingConfig | None = None,
        loss_config: IndependentEBMLossConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.loss_config = loss_config or IndependentEBMLossConfig()
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def fit(
        self,
        train_dataset,
        validation_dataset,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        self._set_seed(self.config.seed)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        train_loader = self._loader(train_dataset, shuffle=True)
        validation_loader = self._loader(validation_dataset, shuffle=False)

        history: list[dict[str, float]] = []
        best_loss = float("inf")
        epochs_without_improvement = 0
        best_path = output / "best.pt"

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            row = {
                "epoch": float(epoch),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"validation_{k}": v for k, v in validation_metrics.items()},
            }
            history.append(row)
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                epochs_without_improvement = 0
                self._save_checkpoint(best_path, epoch, best_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        summary = {
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
            "device": str(self.device),
            "history": history,
        }
        (output / "history.json").write_text(json.dumps(summary, indent=2))
        return summary

    @torch.no_grad()
    def evaluate(self, dataset) -> dict[str, float]:
        return self._run_epoch(self._loader(dataset, shuffle=False), training=False)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        examples = 0
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            batch_size = int(batch["target_answer"].shape[0])
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = self.model(batch)
                loss, components = independent_ebm_loss(
                    output, batch, self.loss_config
                )
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
            metrics = self._batch_metrics(output, batch, components)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * batch_size
            examples += batch_size
        if examples == 0:
            raise ValueError("DataLoader produced no batches")
        return {name: value / examples for name, value in totals.items()}

    @staticmethod
    def _batch_metrics(output, batch, components: dict[str, Tensor]) -> dict[str, float]:
        answer_accuracy = (
            output.answer_logits.argmax(dim=-1) == batch["target_answer"]
        ).float().mean()
        evidence_mae = (
            output.effective_evidence_logits.argmax(dim=-1).float()
            - batch["target_effective_evidence"].float()
        ).abs().mean()
        fc_predictions = (torch.sigmoid(output.false_consensus_logits) >= 0.5).float()
        fc_accuracy = (
            fc_predictions == batch["target_false_consensus_by_answer"]
        ).float().mean()
        metrics = {name: float(value.item()) for name, value in components.items()}
        metrics.update(
            {
                "answer_accuracy": float(answer_accuracy.item()),
                "effective_evidence_mae": float(evidence_mae.item()),
                "false_consensus_accuracy": float(fc_accuracy.item()),
            }
        )
        return metrics

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            generator=generator,
        )

    def _save_checkpoint(self, path: Path, epoch: int, validation_loss: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_config": asdict(self.config),
                "loss_config": asdict(self.loss_config),
                "model_config": asdict(self.model.config),
            },
            path,
        )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
