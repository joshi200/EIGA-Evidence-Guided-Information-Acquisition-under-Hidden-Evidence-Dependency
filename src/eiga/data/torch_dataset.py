from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


class EvidenceJsonlDataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset for EIGA JSONL records.

    Offsets are indexed once, while records are parsed lazily. This keeps memory
    usage low for large generated datasets and preserves deterministic ordering.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._offsets = self._build_offsets()
        if not self._offsets:
            raise ValueError(f"Dataset is empty: {self.path}")

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            payload = json.loads(handle.readline())
        return tensorise_record(payload)


class InMemoryEvidenceDataset(Dataset[dict[str, Tensor]]):
    """Small in-memory dataset used by tests and smoke experiments."""

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return tensorise_record(self.records[index])


def tensorise_record(record: dict[str, Any]) -> dict[str, Tensor]:
    features = record["features"]
    targets = record["targets"]
    return {
        "tool_claims": torch.tensor(features["tool_claims"], dtype=torch.long),
        "tool_confidences": torch.tensor(
            features["tool_confidences"], dtype=torch.float32
        ),
        "queried_mask": torch.tensor(features["queried_mask"], dtype=torch.bool),
        "diagnosed_mask": torch.tensor(
            features["diagnosed_mask"], dtype=torch.bool
        ),
        "metadata_signatures": torch.tensor(
            features["metadata_signatures"], dtype=torch.long
        ),
        "diagnostic_matrix": torch.tensor(
            features["diagnostic_matrix"], dtype=torch.float32
        ),
        "remaining_budget": torch.tensor(
            features["remaining_budget"], dtype=torch.float32
        ),
        "normalised_step": torch.tensor(
            features["normalised_step"], dtype=torch.float32
        ),
        "target_answer": torch.tensor(targets["answer"], dtype=torch.long),
        "target_dependency_matrix": torch.tensor(
            targets["dependency_matrix"], dtype=torch.float32
        ),
        "target_dependency_mask": torch.tensor(
            targets["dependency_mask"], dtype=torch.bool
        ),
        "target_lineage_labels": torch.tensor(
            targets["lineage_labels"], dtype=torch.long
        ),
        "target_effective_evidence": torch.tensor(
            targets["effective_evidence"], dtype=torch.long
        ),
        "target_observed_agreement": torch.tensor(
            targets["observed_agreement"], dtype=torch.float32
        ),
        "target_false_consensus_by_answer": torch.tensor(
            targets["false_consensus_by_answer"], dtype=torch.float32
        ),
    }


def move_batch_to_device(
    batch: dict[str, Tensor], device: torch.device | str
) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def iterate_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
