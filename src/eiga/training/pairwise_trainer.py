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
from eiga.models.pairwise_ebm import (
    PairwiseEBMLossConfig,
    PairwiseEvidenceBeliefModule,
    pairwise_ebm_loss,
)


@dataclass(frozen=True)
class PairwiseTrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    patience: int = 5
    seed: int = 17
    num_workers: int = 0
    device: str = "auto"


class PairwiseEBMTrainer:
    def __init__(
        self,
        model: PairwiseEvidenceBeliefModule,
        config: PairwiseTrainingConfig | None = None,
        loss_config: PairwiseEBMLossConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PairwiseTrainingConfig()
        self.loss_config = loss_config or PairwiseEBMLossConfig()
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def fit(self, train_dataset, validation_dataset, output_dir: str | Path) -> dict[str, Any]:
        self._set_seed(self.config.seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_loader = self._loader(train_dataset, shuffle=True)
        validation_loader = self._loader(validation_dataset, shuffle=False)
        best_path = output_dir / "best.pt"
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        stale_epochs = 0

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            history.append({
                "epoch": float(epoch),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"validation_{k}": v for k, v in validation_metrics.items()},
            })
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                stale_epochs = 0
                self._save_checkpoint(best_path, epoch, best_loss)
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        summary = {
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
            "device": str(self.device),
            "history": history,
        }
        (output_dir / "history.json").write_text(json.dumps(summary, indent=2))
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
                loss, components = pairwise_ebm_loss(output, batch, self.loss_config)
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
        return {name: total / examples for name, total in totals.items()}

    @staticmethod
    def _batch_metrics(output, batch, components: dict[str, Tensor]) -> dict[str, float]:
        answer_accuracy = (
            output.answer_logits.argmax(dim=-1) == batch["target_answer"]
        ).float().mean()
        evidence_mae = (
            output.effective_evidence_logits.argmax(dim=-1).float()
            - batch["target_effective_evidence"].float()
        ).abs().mean()
        pair_mask = batch["target_dependency_mask"].bool()
        upper = torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
        mask = pair_mask & upper
        if mask.any():
            predictions = (output.dependency_probabilities[mask] >= 0.5).float()
            targets = batch["target_dependency_matrix"][mask]
            dependency_accuracy = (predictions == targets).float().mean()
        else:
            dependency_accuracy = torch.tensor(1.0, device=output.answer_logits.device)
        metrics = {name: float(value.item()) for name, value in components.items()}
        metrics.update({
            "answer_accuracy": float(answer_accuracy.item()),
            "effective_evidence_mae": float(evidence_mae.item()),
            "dependency_accuracy": float(dependency_accuracy.item()),
        })
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
        torch.save({
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_config": asdict(self.config),
            "loss_config": asdict(self.loss_config),
            "model_config": asdict(self.model.config),
        }, path)

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
