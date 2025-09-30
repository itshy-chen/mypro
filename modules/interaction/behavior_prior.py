from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


INTENT_LABELS = ("yield", "overtake", "approach", "irrelevant")
NUM_INTENTS = len(INTENT_LABELS)


@dataclass
class BehaviorPriorOutput:
    """Container for the behaviour prior forward pass results."""

    logits: torch.Tensor
    masked_logits: torch.Tensor
    prob: torch.Tensor
    intent_embedding: torch.Tensor
    attention_bias: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class FeatureEncoder(nn.Module):
    """Light-weight projection and normalisation block for feature tensors."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.LazyLinear(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        x = self.proj(x)
        x = self.norm(x)
        return x.view(*orig_shape[:-1], -1)


class InteractionBehaviorPrior(nn.Module):
    """Predict a distribution over neighbour interaction intents.

    The module consumes fused target/neighbor representations, additional edge
    descriptors and geometric priors to estimate the interaction intent between a
    target vehicle ``i`` and each of its neighbours ``j``. The resulting
    distribution is re-used in multiple downstream components as both an
    explanatory signal and as a soft prior for the cross-attention weights.

    Args:
        hidden_dim: Shared hidden dimensionality used across the feature
            encoders.
        intent_embed_dim: Size of the learnable intent embeddings.
        mlp_hidden_dim: Hidden dimensionality of the classification head.
        dropout: Dropout probability applied inside the MLP head.
        include_edge: When ``True`` edge features are consumed.
        include_geom: When ``True`` geometric prior summaries are consumed.
        include_future_context: Whether to use the optional neighbour future
            context (available for iterations ``r >= 2``).
        bias_weights: Optional tensor of per-class weights that control how the
            categorical distribution is mapped to an attention bias scalar.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        intent_embed_dim: int = 64,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
        include_edge: bool = True,
        include_geom: bool = True,
        include_future_context: bool = False,
        bias_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.include_edge = include_edge
        self.include_geom = include_geom
        self.include_future_context = include_future_context

        self.target_encoder = FeatureEncoder(hidden_dim)
        self.neighbour_encoder = FeatureEncoder(hidden_dim)
        self.edge_encoder = FeatureEncoder(hidden_dim) if include_edge else None
        self.geom_encoder = FeatureEncoder(hidden_dim) if include_geom else None
        self.future_encoder = (
            FeatureEncoder(hidden_dim) if include_future_context else None
        )

        feature_blocks = 2
        if include_edge:
            feature_blocks += 1
        if include_geom:
            feature_blocks += 1
        if include_future_context:
            feature_blocks += 1

        fusion_dim = hidden_dim * feature_blocks
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.logit_head = nn.Linear(mlp_hidden_dim, NUM_INTENTS)

        self.intent_embeddings = nn.Embedding(NUM_INTENTS, intent_embed_dim)
        nn.init.xavier_uniform_(self.intent_embeddings.weight)

        if bias_weights is None:
            bias_weights = torch.tensor([1.0, 1.2, 0.8, 0.0])
        if bias_weights.shape != (NUM_INTENTS,):
            raise ValueError(
                "`bias_weights` must be of shape [NUM_INTENTS]."
            )
        self.register_buffer("bias_weights", bias_weights.clone())

    def forward(
        self,
        target_feat: torch.Tensor,
        neighbour_feat: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
        geom_prior: Optional[torch.Tensor] = None,
        neighbour_mask: Optional[torch.Tensor] = None,
        future_context: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> BehaviorPriorOutput:
        """Run a forward pass of the behaviour prior.

        Args:
            target_feat: Tensor with shape ``[B, A, D_t]`` (or ``[B, A, K, D_t]``)
                containing the fused history/map features of the target agents.
            neighbour_feat: Tensor of shape ``[B, A, K, D_n]`` describing the
                neighbour agents.
            edge_feat: Optional tensor ``[B, A, K, D_e]`` capturing pairwise
                interaction context for the current refinement round.
            geom_prior: Optional tensor ``[B, A, K, D_g]`` aggregating geometric
                risk priors produced by ``geometric_priors``.
            neighbour_mask: Optional binary tensor ``[B, A, K]`` marking valid
                neighbours (``True`` for valid entries).
            future_context: Optional tensor ``[B, A, K, D_f]`` with neighbour
                future representations from previous refinement rounds.
            temperature: Optional temperature scaling applied to the logits prior
                to the softmax normalisation.
        """

        if neighbour_feat.dim() != 4:
            raise ValueError("`neighbour_feat` must have shape [B, A, K, D_n].")

        B, A, K, _ = neighbour_feat.shape

        target_feat = self._align_target(target_feat, B, A, K)
        neighbour_encoded = self.neighbour_encoder(neighbour_feat)
        target_encoded = self.target_encoder(target_feat)

        feature_parts = [target_encoded, neighbour_encoded]

        if self.include_edge:
            if edge_feat is None:
                raise ValueError("`edge_feat` is required when `include_edge=True`.")
            edge_encoded = self.edge_encoder(edge_feat)
            feature_parts.append(edge_encoded)

        if self.include_geom:
            if geom_prior is None:
                raise ValueError("`geom_prior` is required when `include_geom=True`.")
            geom_encoded = self.geom_encoder(geom_prior)
            feature_parts.append(geom_encoded)

        if self.include_future_context:
            if future_context is None:
                raise ValueError(
                    "`future_context` is required when ``include_future_context=True``."
                )
            future_encoded = self.future_encoder(future_context)
            feature_parts.append(future_encoded)

        fused = torch.cat(feature_parts, dim=-1)
        fused = self.fusion_norm(fused)

        hidden = self.mlp(fused)
        logits = self.logit_head(hidden)

        if temperature <= 0.0:
            raise ValueError("`temperature` must be positive.")
        scaled_logits = logits / temperature

        masked_logits = self._apply_mask(scaled_logits, neighbour_mask)
        prob = F.softmax(masked_logits, dim=-1)
        prob = self._mask_prob(prob, neighbour_mask)

        intent_embedding = torch.matmul(prob, self.intent_embeddings.weight)
        if neighbour_mask is not None:
            intent_embedding = intent_embedding * neighbour_mask.unsqueeze(-1).to(
                intent_embedding.dtype
            )

        attention_bias = self.compute_attention_bias(prob, neighbour_mask)
        diagnostics = self._gather_diagnostics(prob, neighbour_mask)

        return BehaviorPriorOutput(
            logits=logits,
            masked_logits=masked_logits,
            prob=prob,
            intent_embedding=intent_embedding,
            attention_bias=attention_bias,
            diagnostics=diagnostics,
        )

    def compute_attention_bias(
        self,
        prob: torch.Tensor,
        neighbour_mask: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Convert a probability distribution into an additive attention bias."""

        bias = torch.matmul(prob, self.bias_weights)
        bias = bias * scale
        if neighbour_mask is not None:
            bias = bias * neighbour_mask.to(bias.dtype)
        return bias

    def _align_target(
        self, target_feat: torch.Tensor, batch: int, agents: int, neighbours: int
    ) -> torch.Tensor:
        if target_feat.dim() == 3:
            if target_feat.shape[0] != batch or target_feat.shape[1] != agents:
                raise ValueError(
                    "`target_feat` must align with neighbour batch dimensions."
                )
            target_feat = target_feat.unsqueeze(2).expand(-1, -1, neighbours, -1)
        elif target_feat.dim() == 4:
            if (
                target_feat.shape[0] != batch
                or target_feat.shape[1] != agents
                or target_feat.shape[2] not in (1, neighbours)
            ):
                raise ValueError(
                    "`target_feat` must broadcast to neighbour dimension."
                )
            if target_feat.shape[2] == 1 and neighbours > 1:
                target_feat = target_feat.expand(-1, -1, neighbours, -1)
            elif target_feat.shape[2] != neighbours:
                raise ValueError(
                    "`target_feat`'s neighbour axis does not match the expected K."
                )
        else:
            raise ValueError("`target_feat` must be a 3-D or 4-D tensor.")
        return target_feat

    def _apply_mask(
        self, logits: torch.Tensor, neighbour_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if neighbour_mask is None:
            return logits
        if neighbour_mask.shape != logits.shape[:3]:
            raise ValueError("`neighbour_mask` must match the first three dims of logits.")
        mask = neighbour_mask.to(dtype=torch.bool).unsqueeze(-1)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        return masked_logits

    def _mask_prob(
        self, prob: torch.Tensor, neighbour_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if neighbour_mask is None:
            return prob
        mask = neighbour_mask.to(prob.dtype).unsqueeze(-1)
        return prob * mask

    def _gather_diagnostics(
        self, prob: torch.Tensor, neighbour_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if neighbour_mask is None:
            mean_prob = prob.mean(dim=(0, 1, 2))
            entropy = self._entropy(prob).mean()
            valid = torch.tensor(
                prob.shape[0] * prob.shape[1] * prob.shape[2], device=prob.device
            )
        else:
            mask = neighbour_mask.to(prob.dtype)
            denom = mask.sum().clamp_min(1.0)
            weighted_prob = prob * mask.unsqueeze(-1)
            mean_prob = weighted_prob.sum(dim=(0, 1, 2)) / denom
            entropy = (self._entropy(prob) * mask).sum() / denom
            valid = mask.sum()
        finite = torch.isfinite(prob).all()
        return {
            "mean_prob": mean_prob.detach(),
            "entropy": entropy.detach(),
            "valid_pairs": valid.detach(),
            "is_finite": finite.unsqueeze(0),
        }

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        prob = prob.clamp(min=1e-8)
        return -(prob * prob.log()).sum(dim=-1)


__all__ = [
    "InteractionBehaviorPrior",
    "BehaviorPriorOutput",
    "INTENT_LABELS",
]