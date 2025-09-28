from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


INTENT_LABELS = ("yield", "overtake", "approach", "irrelevant")
NUM_INTENTS = len(INTENT_LABELS)


class InteractionIntentPredictor(nn.Module):
    """Predict interaction intent distribution between target and neighbor agents.

    The module consumes the spatio-temporal representations produced by the
    Mamba-based history encoder, map encoder and the future decoder stack.  For
    every target agent ``i`` and its neighbor ``j`` it outputs a categorical
    distribution over interaction intents, together with an expectation scalar
    (w.r.t. a learnable intent value prior) and an attention bias term that can
    be re-used by downstream attention modules.

    Args:
        hidden_dim: Size of the hidden representations inside the predictor.
        dropout: Dropout probability applied after each hidden layer.
        intent_values: Optional tensor initialising the expectation prior for
            each intent class. When ``None`` a linearly spaced prior in
            ``[0, 1]`` is used.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        intent_values: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.target_encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.neighbor_encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.joint_encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.intent_head = nn.LazyLinear(NUM_INTENTS)
        self.bias_head = nn.LazyLinear(1)

        if intent_values is None:
            intent_values = torch.linspace(0.0, 1.0, steps=NUM_INTENTS)
        if intent_values.shape != (NUM_INTENTS,):  # pragma: no cover - defensive
            raise ValueError(
                "`intent_values` must be a 1-D tensor with length equal to the"
                f" number of intents ({NUM_INTENTS})."
            )
        self.intent_values = nn.Parameter(intent_values.clone())

    def forward(
        self,
        h_tilde: torch.Tensor,
        H_seq: torch.Tensor,
        H_f: torch.Tensor,
        Y_pred: torch.Tensor,
        W_attn: torch.Tensor,
        nbr_index: torch.Tensor,
        nbr_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run intent prediction for every target-neighbor pair.

        Args:
            h_tilde: Historical fusion features of shape ``[B, N_tgt, N_nbr, D_h]``.
            H_seq: Target sequence-level history features ``[B, N_tgt, T_obs, D]``
                (or ``[B, T_obs, D]`` for a single target case).
            H_f: Future encoder outputs ``[B, N_tgt, T_fut, D]``.
            Y_pred: Predicted future trajectories ``[B, N_tgt, K, T_pred, 2]``.
            W_attn: Attention weights/features per neighbor
                ``[B, N_tgt, N_nbr, D_w]``.
            nbr_index: Neighbour meta information (e.g. relative pose or index)
                ``[B, N_tgt, N_nbr, D_idx]``.
            nbr_mask: Optional binary mask ``[B, N_tgt, N_nbr]`` indicating
                valid neighbours.

        Returns:
            Dictionary with ``logits``, ``prob``, ``expected`` and ``attn_bias``.
        """

        if h_tilde.dim() != 4:
            raise ValueError("`h_tilde` must be of shape [B, N_tgt, N_nbr, D_h].")

        B, N_tgt, N_nbr, _ = h_tilde.shape

        target_feat = self._encode_target_context(H_seq, H_f, Y_pred, B, N_tgt)
        neighbor_feat = self._encode_neighbor_context(h_tilde, W_attn, nbr_index)

        target_feat = self.target_encoder(target_feat)
        neighbor_feat = self.neighbor_encoder(neighbor_feat)

        target_expanded = target_feat.unsqueeze(2).expand(-1, -1, N_nbr, -1)
        joint_input = torch.cat([target_expanded, neighbor_feat], dim=-1)
        joint_feat = self.joint_encoder(joint_input)

        logits = self.intent_head(joint_feat)
        prob = F.softmax(logits, dim=-1)

        expected = torch.matmul(prob, self.intent_values)
        attn_bias = self.bias_head(joint_feat).squeeze(-1)

        if nbr_mask is not None:
            nbr_mask = nbr_mask.to(dtype=logits.dtype)
            mask_expand = nbr_mask.unsqueeze(-1)
            prob = prob * mask_expand
            expected = expected * nbr_mask
            attn_bias = attn_bias * nbr_mask

        return {
            "logits": logits,
            "prob": prob,
            "expected": expected,
            "attn_bias": attn_bias,
        }

    def _encode_target_context(
        self,
        H_seq: torch.Tensor,
        H_f: torch.Tensor,
        Y_pred: torch.Tensor,
        batch: int,
        n_targets: int,
    ) -> torch.Tensor:
        """Aggregate per-target temporal context into a fixed-size feature."""

        hist = self._pool_temporal(H_seq, expected_batch=batch, expected_agents=n_targets)
        future = self._pool_temporal(H_f, expected_batch=batch, expected_agents=n_targets)
        traj = self._summarise_predictions(Y_pred, batch=batch, n_targets=n_targets)

        features = [tensor for tensor in (hist, future, traj) if tensor is not None]
        if not features:
            raise ValueError("At least one of the target features must be provided.")
        return torch.cat(features, dim=-1)

    def _encode_neighbor_context(
        self,
        h_tilde: torch.Tensor,
        W_attn: torch.Tensor,
        nbr_index: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate neighbour-specific features."""

        features = [h_tilde]
        if W_attn is not None:
            if W_attn.shape[:3] != h_tilde.shape[:3]:
                raise ValueError("`W_attn` must match the first three dims of h_tilde`.")
            features.append(W_attn)
        if nbr_index is not None:
            if nbr_index.shape[:3] != h_tilde.shape[:3]:
                raise ValueError(
                    "`nbr_index` must match the first three dims of `h_tilde`."
                )
            features.append(nbr_index.to(h_tilde.dtype))
        return torch.cat(features, dim=-1)

    @staticmethod
    def _pool_temporal(
        tensor: Optional[torch.Tensor],
        *,
        expected_batch: int,
        expected_agents: int,
    ) -> Optional[torch.Tensor]:
        """Pool temporal tensors into ``[B, N, D]`` format."""

        if tensor is None:
            return None
        if tensor.dim() == 4:
            if tensor.shape[0] != expected_batch or tensor.shape[1] != expected_agents:
                raise ValueError("Temporal tensor batch/agent dims do not match.")
            last = tensor[..., -1, :]
            mean = tensor.mean(dim=-2)
            max_ = tensor.amax(dim=-2)
            return torch.cat([last, mean, max_], dim=-1)
        if tensor.dim() == 3:
            if tensor.shape[0] != expected_batch:
                raise ValueError("Temporal tensor batch dim does not match.")
            last = tensor[..., -1, :]
            mean = tensor.mean(dim=-2)
            max_ = tensor.amax(dim=-2)
            pooled = torch.cat([last, mean, max_], dim=-1)
            return pooled.unsqueeze(1).expand(-1, expected_agents, -1)
        raise ValueError("Temporal tensors must have 3 or 4 dimensions.")

    @staticmethod
    def _summarise_predictions(
        pred: Optional[torch.Tensor],
        *,
        batch: int,
        n_targets: int,
    ) -> Optional[torch.Tensor]:
        """Summarise the distribution over future trajectories."""

        if pred is None:
            return None
        if pred.dim() != 5:
            raise ValueError("`Y_pred` must be of shape [B, N_tgt, K, T, 2].")
        if pred.shape[0] != batch or pred.shape[1] != n_targets:
            raise ValueError("`Y_pred` batch/target dims do not match history features.")

        first = pred[..., 0, :]
        last = pred[..., -1, :]
        displacement = last - first  # [B, N, K, 2]
        disp_mean = displacement.mean(dim=-2)
        disp_std = displacement.std(dim=-2)

        step = pred[..., 1:, :] - pred[..., :-1, :]
        speed = torch.norm(step, dim=-1)  # [B, N, K, T-1]
        speed_mean = speed.mean(dim=-1).mean(dim=-1, keepdim=True)
        speed_max = speed.amax(dim=-1).amax(dim=-1, keepdim=True)

        return torch.cat([disp_mean, disp_std, speed_mean, speed_max], dim=-1)
