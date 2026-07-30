from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PairwiseEBMConfig:
    num_tools: int = 6
    num_answer_classes: int = 2
    hidden_dim: int = 128
    claim_embedding_dim: int = 24
    metadata_embedding_dim: int = 16
    pair_hidden_dim: int = 128
    dropout: float = 0.10


@dataclass
class PairwiseEBMOutput:
    answer_logits: Tensor
    effective_evidence_logits: Tensor
    false_consensus_logits: Tensor
    observed_agreement: Tensor
    dependency_logits: Tensor
    dependency_probabilities: Tensor
    evidence_weights: Tensor
    belief_embedding: Tensor


class PairwiseEvidenceBeliefModule(nn.Module):
    """Dependency-aware evidence aggregator with explicit pairwise supervision.

    Each queried tool is encoded independently. A symmetric pair representation
    predicts whether two tools share an evidence lineage. Those probabilities
    down-weight redundant tools before global evidence aggregation.
    """

    def __init__(self, config: PairwiseEBMConfig) -> None:
        super().__init__()
        self.config = config
        self.unseen_claim_index = config.num_answer_classes
        self.unknown_metadata_index = config.num_tools

        self.claim_embedding = nn.Embedding(
            config.num_answer_classes + 1, config.claim_embedding_dim
        )
        self.metadata_embedding = nn.Embedding(
            config.num_tools + 1, config.metadata_embedding_dim
        )
        tool_input_dim = (
            config.claim_embedding_dim
            + config.metadata_embedding_dim
            + 3
            + 3
        )
        self.tool_encoder = nn.Sequential(
            nn.Linear(tool_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        pair_input_dim = config.hidden_dim * 4 + 2
        self.pair_head = nn.Sequential(
            nn.Linear(pair_input_dim, config.pair_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.pair_hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.pair_hidden_dim, 1),
        )

        global_dim = config.hidden_dim * 2 + config.num_tools + 3
        self.belief_encoder = nn.Sequential(
            nn.Linear(global_dim, config.hidden_dim),
            nn.GELU(),
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
        self.agreement_head = nn.Sequential(nn.Linear(config.hidden_dim, 1), nn.Sigmoid())

    def forward(self, batch: dict[str, Tensor]) -> PairwiseEBMOutput:
        queried = batch["queried_mask"].bool()
        claims = batch["tool_claims"].long()
        metadata = batch["metadata_signatures"].long()

        safe_claims = torch.where(
            queried,
            claims.clamp(0, self.config.num_answer_classes - 1),
            torch.full_like(claims, self.unseen_claim_index),
        )
        safe_metadata = torch.where(
            queried,
            metadata.clamp(0, self.config.num_tools - 1),
            torch.full_like(metadata, self.unknown_metadata_index),
        )
        diagnostic = torch.nan_to_num(
            batch["diagnostic_matrix"].float(), nan=0.0
        ).clamp(-1.0, 1.0)
        diagnostic_observed = (diagnostic != 0).float()
        diagnostic_summary = torch.stack(
            [
                diagnostic.mean(dim=-1),
                diagnostic.abs().amax(dim=-1),
                diagnostic_observed.mean(dim=-1),
            ],
            dim=-1,
        )
        tool_features = torch.cat(
            [
                self.claim_embedding(safe_claims),
                self.metadata_embedding(safe_metadata),
                diagnostic_summary,
                batch["tool_confidences"].float().unsqueeze(-1),
                queried.float().unsqueeze(-1),
                batch["diagnosed_mask"].float().unsqueeze(-1),
            ],
            dim=-1,
        )
        tools = self.tool_encoder(tool_features)
        tools = tools * queried.float().unsqueeze(-1)

        left = tools[:, :, None, :].expand(-1, -1, self.config.num_tools, -1)
        right = tools[:, None, :, :].expand(-1, self.config.num_tools, -1, -1)
        direct_diagnostic = diagnostic.unsqueeze(-1)
        metadata_match = (safe_metadata[:, :, None] == safe_metadata[:, None, :]).float().unsqueeze(-1)
        pair_features = torch.cat(
            [
                left + right,
                torch.abs(left - right),
                left * right,
                (left + right) / 2,
                direct_diagnostic,
                metadata_match,
            ],
            dim=-1,
        )
        dependency_logits = self.pair_head(pair_features).squeeze(-1)
        dependency_logits = 0.5 * (
            dependency_logits + dependency_logits.transpose(1, 2)
        )

        pair_mask = queried[:, :, None] & queried[:, None, :]
        dependency_probabilities = torch.sigmoid(dependency_logits) * pair_mask.float()
        eye = torch.eye(
            self.config.num_tools, dtype=torch.bool, device=tools.device
        ).unsqueeze(0)
        dependency_probabilities = torch.where(
            eye & pair_mask,
            torch.ones_like(dependency_probabilities),
            dependency_probabilities,
        )

        redundancy = (dependency_probabilities * (~eye).float()).sum(dim=-1)
        raw_weights = queried.float() / (1.0 + redundancy)
        evidence_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        weighted_sum = torch.einsum("bn,bnd->bd", evidence_weights, tools)
        masked_sum = tools.sum(dim=1)
        queried_count = queried.float().sum(dim=1, keepdim=True)
        independent_mass = raw_weights.sum(dim=1, keepdim=True)
        global_features = torch.cat(
            [
                weighted_sum,
                masked_sum / queried_count.clamp_min(1.0),
                torch.sort(raw_weights, dim=-1, descending=True).values,
                independent_mass / max(float(self.config.num_tools), 1.0),
                batch["remaining_budget"].float().unsqueeze(-1),
                batch["normalised_step"].float().unsqueeze(-1),
            ],
            dim=-1,
        )
        belief = self.belief_encoder(global_features)
        return PairwiseEBMOutput(
            answer_logits=self.answer_head(belief),
            effective_evidence_logits=self.effective_evidence_head(belief),
            false_consensus_logits=self.false_consensus_head(belief),
            observed_agreement=self.agreement_head(belief).squeeze(-1),
            dependency_logits=dependency_logits,
            dependency_probabilities=dependency_probabilities,
            evidence_weights=evidence_weights,
            belief_embedding=belief,
        )


@dataclass(frozen=True)
class PairwiseEBMLossConfig:
    answer_weight: float = 1.0
    dependency_weight: float = 1.0
    effective_evidence_weight: float = 0.5
    false_consensus_weight: float = 0.5
    agreement_weight: float = 0.25


def pairwise_ebm_loss(
    output: PairwiseEBMOutput,
    batch: dict[str, Tensor],
    config: PairwiseEBMLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    cfg = config or PairwiseEBMLossConfig()
    answer_loss = F.cross_entropy(output.answer_logits, batch["target_answer"])

    pair_mask = batch["target_dependency_mask"].bool()
    off_diagonal = ~torch.eye(
        pair_mask.shape[-1], dtype=torch.bool, device=pair_mask.device
    ).unsqueeze(0)
    upper_triangle = torch.triu(
        torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
    )
    supervised_pairs = pair_mask & off_diagonal & upper_triangle
    if supervised_pairs.any():
        dependency_loss = F.binary_cross_entropy_with_logits(
            output.dependency_logits[supervised_pairs],
            batch["target_dependency_matrix"][supervised_pairs],
        )
    else:
        dependency_loss = output.answer_logits.sum() * 0.0

    evidence_loss = F.cross_entropy(
        output.effective_evidence_logits, batch["target_effective_evidence"]
    )
    false_consensus_loss = F.binary_cross_entropy_with_logits(
        output.false_consensus_logits, batch["target_false_consensus_by_answer"]
    )
    agreement_loss = F.mse_loss(
        output.observed_agreement, batch["target_observed_agreement"]
    )
    total = (
        cfg.answer_weight * answer_loss
        + cfg.dependency_weight * dependency_loss
        + cfg.effective_evidence_weight * evidence_loss
        + cfg.false_consensus_weight * false_consensus_loss
        + cfg.agreement_weight * agreement_loss
    )
    return total, {
        "loss": total.detach(),
        "answer_loss": answer_loss.detach(),
        "dependency_loss": dependency_loss.detach(),
        "effective_evidence_loss": evidence_loss.detach(),
        "false_consensus_loss": false_consensus_loss.detach(),
        "agreement_loss": agreement_loss.detach(),
    }
