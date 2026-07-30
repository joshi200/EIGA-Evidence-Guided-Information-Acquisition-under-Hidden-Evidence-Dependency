from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eiga.data.torch_dataset import move_batch_to_device
from eiga.models.latent_ebm import (
    LatentEBMLossConfig,
    LatentEvidenceBeliefModule,
    latent_ebm_loss,
)
from eiga.training.independent_trainer import TrainingConfig


class LatentEBMTrainer:
    def __init__(
        self,
        model: LatentEvidenceBeliefModule,
        config: TrainingConfig | None = None,
        loss_config: LatentEBMLossConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.loss_config = loss_config or LatentEBMLossConfig()
        self._set_seed(self.config.seed)
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def fit(self, train_dataset, validation_dataset, output_dir: str | Path) -> dict[str, Any]:
        self._set_seed(self.config.seed)
        started = time.perf_counter()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        train_loader = self._loader(train_dataset, True)
        validation_loader = self._loader(validation_dataset, False)
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        stale = 0
        best_epoch = 0
        best_path = output / "best.pt"

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, True)
            validation_metrics = self._run_epoch(validation_loader, False)
            history.append({
                "epoch": float(epoch),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"validation_{k}": v for k, v in validation_metrics.items()},
            })
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                best_epoch = epoch
                stale = 0
                self._save_checkpoint(best_path, epoch, best_loss)
            else:
                stale += 1
                if stale >= self.config.patience:
                    break

        checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        summary = {
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "duration_seconds": time.perf_counter() - started,
            "device": str(self.device),
            "seed": self.config.seed,
            "history": history,
        }
        (output / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    @torch.no_grad()
    def evaluate(self, dataset) -> dict[str, float]:
        return self._run_epoch(self._loader(dataset, False), False)

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        examples = 0
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            size = int(batch["target_answer"].shape[0])
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = self.model(batch)
                loss, parts = latent_ebm_loss(output, batch, self.loss_config)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
            metrics = self._batch_metrics(output, batch, parts)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * size
            examples += size
        if not examples:
            raise ValueError("DataLoader produced no batches")
        return {name: value / examples for name, value in totals.items()}

    @staticmethod
    def _batch_metrics(output, batch, parts: dict[str, Tensor]) -> dict[str, float]:
        answer_prediction = output.answer_logits.argmax(-1)
        answer_accuracy = (answer_prediction == batch["target_answer"]).float().mean()

        evidence_prediction = output.effective_evidence_logits.argmax(-1)
        evidence_target = batch["target_effective_evidence"]
        evidence_accuracy = (evidence_prediction == evidence_target).float().mean()
        evidence_mae = (evidence_prediction.float() - evidence_target.float()).abs().mean()

        false_consensus_prediction = (
            torch.sigmoid(output.false_consensus_logits) >= 0.5
        ).float()
        false_consensus_accuracy = (
            false_consensus_prediction == batch["target_false_consensus_by_answer"]
        ).float().mean()

        agreement_mae = (
            output.observed_agreement - batch["target_observed_agreement"]
        ).abs().mean()

        mask = batch["target_dependency_mask"].bool()
        eye = torch.eye(mask.shape[-1], device=mask.device, dtype=torch.bool).unsqueeze(0)
        mask = mask & ~eye
        if mask.any():
            prediction = output.dependency_probabilities[mask] >= 0.5
            target = batch["target_dependency_matrix"][mask] >= 0.5
            dependency_accuracy = (prediction == target).float().mean()
        else:
            dependency_accuracy = torch.tensor(1.0, device=mask.device)

        metrics = {name: float(value.item()) for name, value in parts.items()}
        metrics.update({
            "answer_accuracy": float(answer_accuracy.item()),
            "effective_evidence_accuracy": float(evidence_accuracy.item()),
            "effective_evidence_mae": float(evidence_mae.item()),
            "false_consensus_accuracy": float(false_consensus_accuracy.item()),
            "observed_agreement_mae": float(agreement_mae.item()),
            "dependency_accuracy": float(dependency_accuracy.item()),
        })
        return metrics

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            generator=torch.Generator().manual_seed(self.config.seed),
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
