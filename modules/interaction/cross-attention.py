"""Cross-attention modules used for agent interaction."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_agent_features(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather per-agent features according to ``indices``.

    ``features`` is expected to be ``[B, A, ...]``. The function returns a tensor
    with shape ``[B, A, K, ...]`` where ``K`` corresponds to the neighbour
    dimension in ``indices``.
    """

    if features.dim() < 3:
        raise ValueError("`features` must have at least three dimensions [B, A, ...].")

    B, A = features.shape[:2]
    if indices.shape[:2] != (B, A):
        raise ValueError("Batch/agent dimensions of `indices` must match `features`.")

    K = indices.shape[2]
    extra_shape = features.shape[2:]
    expanded = features.unsqueeze(1).expand(B, A, A, *extra_shape)
    gather_index = indices.view(B, A, K, *([1] * len(extra_shape)))
    gather_index = gather_index.expand(-1, -1, -1, *extra_shape)
    gathered = torch.gather(expanded, 2, gather_index)
    return gathered


class AgentCrossAttention(nn.Module):
    """Multi-head cross-attention operating on a target agent and its neighbours."""

    def __init__(
        self,
        d_query: int,
        d_key: int,
        *,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("`d_model` must be divisible by `num_heads`.")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_query, d_model)
        self.k_proj = nn.Linear(d_key, d_model)
        self.v_proj = nn.Linear(d_key, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        *,
        neighbour_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention.

        Args:
            query: ``[B, A, L_q, D_q]`` tensor for the target agents.
            key/value: ``[B, A, N, L_k, D_k]`` tensors for the neighbours.
            neighbour_mask: Optional ``[B, A, N]`` mask (``True`` for valid).
            attn_bias: Optional ``[B, A, N]`` additive bias (logit space).
        Returns:
            Tuple with the attended features ``[B, A, L_q, d_model]`` and the
            attention weights ``[B, A, num_heads, L_q, N, L_k]``.
        """

        if value is None:
            value = key

        B, A, L_q, _ = query.shape
        _, _, N, L_k, _ = key.shape

        q = self.q_proj(query).view(B * A, L_q, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [BA, heads, L_q, head_dim]

        k = self.k_proj(key).view(B * A, N, L_k, self.num_heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4).contiguous().view(B * A, self.num_heads, N * L_k, self.head_dim)

        v = self.v_proj(value).view(B * A, N, L_k, self.num_heads, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4).contiguous().view(B * A, self.num_heads, N * L_k, self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [BA, heads, L_q, N*L_k]

        if attn_bias is not None:
            bias = attn_bias.unsqueeze(-1)
            bias = bias.expand(-1, -1, -1, L_k)
            bias = bias.reshape(B * A, 1, 1, N * L_k)
            scores = scores + bias

        if neighbour_mask is not None:
            mask = neighbour_mask.unsqueeze(-1).expand(-1, -1, -1, L_k)
            mask = mask.reshape(B * A, 1, 1, N * L_k)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # [BA, heads, L_q, head_dim]
        context = context.transpose(1, 2).contiguous().view(B * A, L_q, self.num_heads * self.head_dim)
        context = self.out_proj(context)
        context = context.view(B, A, L_q, -1)

        attn_weights = attn.view(B, A, self.num_heads, L_q, N, L_k)
        return context, attn_weights


class FutureFutureCrossAttention(nn.Module):
    """Cross-attention between target futures and neighbour futures (FF-CA)."""

    def __init__(
        self,
        d_future: int,
        *,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = AgentCrossAttention(
            d_query=d_future,
            d_key=d_future,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        future_feat: torch.Tensor,
        neighbour_indices: torch.Tensor,
        *,
        geom_bias: Optional[torch.Tensor] = None,
        neighbour_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run FF-CA.

        Args:
            future_feat: ``[B, A, L_f, D]`` future features for all agents.
            neighbour_indices: ``[B, A, K]`` selected neighbour indices.
            geom_bias: Optional ``[B, A, K]`` additive bias (typically from
                :func:`geom_soft_prior`).
            neighbour_mask: Optional ``[B, A, K]`` validity mask.
        Returns:
            Tuple ``(context, weights)`` matching :class:`AgentCrossAttention`.
        """

        neighbours = _gather_agent_features(future_feat, neighbour_indices)
        context, weights = self.attn(
            future_feat,
            neighbours,
            neighbours,
            neighbour_mask=neighbour_mask,
            attn_bias=geom_bias,
        )
        return context, weights


class HistoryFutureCrossAttention(nn.Module):
    """Cross-attention from history queries to neighbour futures (HF-CA)."""

    def __init__(
        self,
        d_history: int,
        d_future: int,
        *,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = AgentCrossAttention(
            d_query=d_history,
            d_key=d_future,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        history_feat: torch.Tensor,
        future_feat: torch.Tensor,
        neighbour_indices: torch.Tensor,
        *,
        geom_bias: Optional[torch.Tensor] = None,
        neighbour_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run HF-CA."""

        neighbours = _gather_agent_features(future_feat, neighbour_indices)
        context, weights = self.attn(
            history_feat,
            neighbours,
            neighbours,
            neighbour_mask=neighbour_mask,
            attn_bias=geom_bias,
        )
        return context, weights