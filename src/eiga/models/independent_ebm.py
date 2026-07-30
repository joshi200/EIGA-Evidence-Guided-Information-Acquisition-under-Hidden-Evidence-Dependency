from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class IndependentEBMConfig:
    num_tools: int = 6
    num_answer_classes: int = 2
    hidden_dim: int = 128
    claim_embedding_dim: int = 24
    dropout: float = 0.10


@dataclass
class IndependentEBMOutput:
    answer_logits: Tensor
    effective_evidence_logits: Tensor
    false_consensus_logits: Tensor
    observed_agreement: Tensor
    belief_embedding: Tensor


class IndependentEvidenceBeliefModule(nn.Module):
    """Permutation-invariant baseline that assumes queried tools are independent.

    It deliberately excludes provenance signatures and pairwise diagnostics. Each
    observed tool is encoded separately and pooled with a DeepSets-style sum/mean.
    This gives a strong neural aggregation baseline while preserving the exact
    independence assumption the proposed dependency-aware models must improve on.
    """

    def __init__(self, config: IndependentEBMConfig) -> None:
        super().__init__()
        self.config = config
        self.unseen_claim_index = config.num_answer_classes
        self.claim_embedding = nn.Embedding(
            config.num_answer_classes + 1, config.claim_embedding_dim
        )
        tool_input_dim = config.claim_embedding_dim + 3
        self.tool_encoder = nn.Sequential(
            nn.Linear(tool_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        global_input_dim = config.hidden_dim * 2 + 3
        self.belief_encoder = nn.Sequential(
            nn.Linear(global_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.answer_head = nn.Linear(config.hidden_dim, config.num_answer_classes)
        self.effective_evidence_head = nn.Linear(
            config.hidden_dim, config.num_tools + 1
        )
        self.false_consensus_head = nn.Linear(
            config.hidden_dim, config.num_answer_classes
        )
        self.agreement_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, batch: dict[str, Tensor]) -> IndependentEBMOutput:
        claims = batch["tool_claims"].long()
        queried = batch["queried_mask"].bool()
        confidences = batch["tool_confidences"].float()

        safe_claims = torch.where(
            queried,
            claims.clamp(min=0, max=self.config.num_answer_classes - 1),
            torch.full_like(claims, self.unseen_claim_index),
        )
        claim_features = self.claim_embedding(safe_claims)
        per_tool = torch.cat(
            [
                claim_features,
                confidences.unsqueeze(-1),
                queried.float().unsqueeze(-1),
                batch["diagnosed_mask"].float().unsqueeze(-1),
            ],
            dim=-1,
        )
        encoded = self.tool_encoder(per_tool)
        mask = queried.float().unsqueeze(-1)
        encoded = encoded * mask
        sum_pool = encoded.sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = sum_pool / count
        queried_fraction = queried.float().mean(dim=1, keepdim=True)
        global_features = torch.cat(
            [
                sum_pool,
                mean_pool,
                queried_fraction,
                batch["remaining_budget"].float().unsqueeze(-1),
                batch["normalised_step"].float().unsqueeze(-1),
            ],
            dim=-1,
        )
        belief = self.belief_encoder(global_features)
        return IndependentEBMOutput(
            answer_logits=self.answer_head(belief),
            effective_evidence_logits=self.effective_evidence_head(belief),
            false_consensus_logits=self.false_consensus_head(belief),
            observed_agreement=self.agreement_head(belief).squeeze(-1),
            belief_embedding=belief,
        )


@dataclass(frozen=True)
class IndependentEBMLossConfig:
    answer_weight: float = 1.0
    effective_evidence_weight: float = 0.5
    false_consensus_weight: float = 0.5
    agreement_weight: float = 0.25


def independent_ebm_loss(
    output: IndependentEBMOutput,
    batch: dict[str, Tensor],
    config: IndependentEBMLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    cfg = config or IndependentEBMLossConfig()
    answer_loss = F.cross_entropy(output.answer_logits, batch["target_answer"])
    evidence_loss = F.cross_entropy(
        output.effective_evidence_logits, batch["target_effective_evidence"]
    )
    false_consensus_loss = F.binary_cross_entropy_with_logits(
        output.false_consensus_logits,
        batch["target_false_consensus_by_answer"],
    )
    agreement_loss = F.mse_loss(
        output.observed_agreement, batch["target_observed_agreement"]
    )
    total = (
        cfg.answer_weight * answer_loss
        + cfg.effective_evidence_weight * evidence_loss
        + cfg.false_consensus_weight * false_consensus_loss
        + cfg.agreement_weight * agreement_loss
    )
    return total, {
        "loss": total.detach(),
        "answer_loss": answer_loss.detach(),
        "effective_evidence_loss": evidence_loss.detach(),
        "false_consensus_loss": false_consensus_loss.detach(),
        "agreement_loss": agreement_loss.detach(),
    }
