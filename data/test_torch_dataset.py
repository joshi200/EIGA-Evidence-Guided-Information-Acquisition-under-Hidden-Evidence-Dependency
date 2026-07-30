import json

import torch

from eiga.data.torch_dataset import EvidenceJsonlDataset


def test_jsonl_dataset_tensorises_record(tmp_path):
    record = {
        "features": {
            "tool_claims": [-1, 1],
            "tool_confidences": [0.0, 0.8],
            "queried_mask": [False, True],
            "diagnosed_mask": [False, False],
            "metadata_signatures": [0, 1],
            "diagnostic_matrix": [[-1.0, -1.0], [-1.0, -1.0]],
            "remaining_budget": 4.0,
            "normalised_step": 0.1,
        },
        "targets": {
            "answer": 1,
            "dependency_matrix": [[1.0, 0.0], [0.0, 1.0]],
            "dependency_mask": [[False, False], [False, True]],
            "lineage_labels": [0, 1],
            "effective_evidence": 1,
            "observed_agreement": 1.0,
            "false_consensus_by_answer": [False, False],
        },
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = EvidenceJsonlDataset(path)
    item = dataset[0]
    assert len(dataset) == 1
    assert item["tool_claims"].dtype == torch.long
    assert item["queried_mask"].dtype == torch.bool
    assert item["target_answer"].item() == 1
