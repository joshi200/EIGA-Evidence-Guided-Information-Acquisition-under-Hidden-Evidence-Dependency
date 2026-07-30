from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentEBMConfig:
    num_tools: int = 6
    num_answer_classes: int = 2
    max_latent_sources: int = 6
    hidden_dim: int = 128
    claim_embedding_dim: int = 24
    metadata_embedding_dim: int = 16
    slot_iterations: int = 3
    dropout: float = 0.10
    assignment_temperature: float = 0.75


@dataclass
class LatentEBMOutput:
    answer_logits: Tensor
    effective_evidence_logits: Tensor
    false_consensus_logits: Tensor
    observed_agreement: Tensor
    assignment_probabilities: Tensor
    dependency_probabilities: Tensor
    slot_occupancy: Tensor
    belief_embedding: Tensor


class LatentEvidenceBeliefModule(nn.Module):
    """Infers a permutation-invariant posterior over latent evidence lineages.

    Each queried tool is softly assigned to one of K exchangeable source slots.
    The induced pairwise dependency posterior is A A^T, avoiding arbitrary
    lineage-label supervision. Slot representations are then pooled to produce
    answer, effective-evidence and false-consensus beliefs.
    """

    def __init__(self, config: LatentEBMConfig) -> None:
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
        diagnostic_dim = config.num_tools
        tool_input_dim = (
            config.claim_embedding_dim
            + config.metadata_embedding_dim
            + diagnostic_dim
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

        self.initial_slots = nn.Parameter(
            torch.randn(config.max_latent_sources, config.hidden_dim) * 0.02
        )
        self.slot_query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.tool_key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.tool_value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.slot_update = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.slot_norm = nn.LayerNorm(config.hidden_dim)

        global_dim = config.hidden_dim * 2 + config.max_latent_sources + 3
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

    def forward(self, batch: dict[str, Tensor]) -> LatentEBMOutput:
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
        diagnostic = batch["diagnostic_matrix"].float()
        diagnostic = torch.nan_to_num(diagnostic, nan=0.0).clamp(-1.0, 1.0)

        tool_features = torch.cat(
            [
                self.claim_embedding(safe_claims),
                self.metadata_embedding(safe_metadata),
                diagnostic,
                batch["tool_confidences"].float().unsqueeze(-1),
                queried.float().unsqueeze(-1),
                batch["diagnosed_mask"].float().unsqueeze(-1),
            ],
            dim=-1,
        )
        tools = self.tool_encoder(tool_features)
        tools = tools * queried.float().unsqueeze(-1)

        assignments, slots = self._infer_slots(tools, queried)
        dependency = torch.bmm(assignments, assignments.transpose(1, 2))
        pair_mask = queried[:, :, None] & queried[:, None, :]
        dependency = dependency * pair_mask.float()
        eye = torch.eye(self.config.num_tools, device=dependency.device).unsqueeze(0)
        dependency = torch.where(eye.bool() & pair_mask, torch.ones_like(dependency), dependency)

        occupancy = 1.0 - torch.prod(1.0 - assignments.clamp(0.0, 1.0), dim=1)
        slot_mass = assignments.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        slot_summary = (slots * occupancy.unsqueeze(-1)).sum(dim=1)
        occupied_count = occupancy.sum(dim=1, keepdim=True).clamp_min(1e-6)
        slot_mean = slot_summary / occupied_count
        queried_fraction = queried.float().mean(dim=1, keepdim=True)

        global_features = torch.cat(
            [
                slot_summary,
                slot_mean,
                occupancy,
                queried_fraction,
                batch["remaining_budget"].float().unsqueeze(-1),
                batch["normalised_step"].float().unsqueeze(-1),
            ],
            dim=-1,
        )
        belief = self.belief_encoder(global_features)
        return LatentEBMOutput(
            answer_logits=self.answer_head(belief),
            effective_evidence_logits=self.effective_evidence_head(belief),
            false_consensus_logits=self.false_consensus_head(belief),
            observed_agreement=self.agreement_head(belief).squeeze(-1),
            assignment_probabilities=assignments,
            dependency_probabilities=dependency,
            slot_occupancy=occupancy,
            belief_embedding=belief,
        )

    def _infer_slots(self, tools: Tensor, queried: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = tools.shape[0]
        slots = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1)
        keys = self.tool_key(tools)
        values = self.tool_value(tools)
        scale = self.config.hidden_dim ** -0.5
        assignments = tools.new_zeros(
            batch_size, self.config.num_tools, self.config.max_latent_sources
        )

        for _ in range(self.config.slot_iterations):
            queries = self.slot_query(self.slot_norm(slots))
            logits = torch.einsum("bnd,bkd->bnk", keys, queries) * scale
            logits = logits / max(self.config.assignment_temperature, 1e-4)
            logits = logits.masked_fill(~queried.unsqueeze(-1), -1e9)
            assignments = torch.softmax(logits, dim=-1)
            assignments = assignments * queried.float().unsqueeze(-1)
            normaliser = assignments.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
            updates = torch.einsum("bnk,bnd->bkd", assignments, values)
            updates = updates / normaliser
            slots = self.slot_update(
                updates.reshape(-1, self.config.hidden_dim),
                slots.reshape(-1, self.config.hidden_dim),
            ).reshape(batch_size, self.config.max_latent_sources, self.config.hidden_dim)
        return assignments, slots


@dataclass(frozen=True)
class LatentEBMLossConfig:
    answer_weight: float = 1.0
    dependency_weight: float = 1.0
    effective_evidence_weight: float = 0.5
    occupancy_weight: float = 0.25
    false_consensus_weight: float = 0.5
    agreement_weight: float = 0.25
    assignment_entropy_weight: float = 0.01


def latent_ebm_loss(
    output: LatentEBMOutput,
    batch: dict[str, Tensor],
    config: LatentEBMLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    cfg = config or LatentEBMLossConfig()
    answer_loss = F.cross_entropy(output.answer_logits, batch["target_answer"])

    pair_mask = batch["target_dependency_mask"].bool()
    off_diagonal = ~torch.eye(
        pair_mask.shape[-1], device=pair_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    supervised_pairs = pair_mask & off_diagonal
    if supervised_pairs.any():
        dependency_loss = F.binary_cross_entropy(
            output.dependency_probabilities[supervised_pairs].clamp(1e-6, 1 - 1e-6),
            batch["target_dependency_matrix"][supervised_pairs],
        )
    else:
        dependency_loss = output.answer_logits.sum() * 0.0

    evidence_loss = F.cross_entropy(
        output.effective_evidence_logits, batch["target_effective_evidence"]
    )
    occupancy_count = output.slot_occupancy.sum(dim=-1)
    occupancy_loss = F.smooth_l1_loss(
        occupancy_count, batch["target_effective_evidence"].float()
    )
    false_consensus_loss = F.binary_cross_entropy_with_logits(
        output.false_consensus_logits, batch["target_false_consensus_by_answer"]
    )
    agreement_loss = F.mse_loss(
        output.observed_agreement, batch["target_observed_agreement"]
    )
    p = output.assignment_probabilities.clamp_min(1e-8)
    queried = batch["queried_mask"].float()
    entropy_per_tool = -(p * p.log()).sum(dim=-1)
    entropy_loss = (entropy_per_tool * queried).sum() / queried.sum().clamp_min(1.0)

    total = (
        cfg.answer_weight * answer_loss
        + cfg.dependency_weight * dependency_loss
        + cfg.effective_evidence_weight * evidence_loss
        + cfg.occupancy_weight * occupancy_loss
        + cfg.false_consensus_weight * false_consensus_loss
        + cfg.agreement_weight * agreement_loss
        + cfg.assignment_entropy_weight * entropy_loss
    )
    return total, {
        "loss": total.detach(),
        "answer_loss": answer_loss.detach(),
        "dependency_loss": dependency_loss.detach(),
        "effective_evidence_loss": evidence_loss.detach(),
        "occupancy_loss": occupancy_loss.detach(),
        "false_consensus_loss": false_consensus_loss.detach(),
        "agreement_loss": agreement_loss.detach(),
        "assignment_entropy": entropy_loss.detach(),
    }
