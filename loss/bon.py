"""Best-of-N trajectory regression loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


LossReduction = Literal["mean", "sum", "none"]
ErrorType = Literal["l1", "huber"]
CoordMode = Literal["absolute", "delta"]


@dataclass
class BestOfNConfig:
    """Configuration for :func:`best_of_n_loss`.

    Attributes
    ----------
    error_type:
        Regression penalty to use. ``"l1"`` uses the absolute error, ``"huber"``
        uses the smooth L1 loss with ``huber_delta`` as the transition point.
    huber_delta:
        The delta parameter for the Huber penalty. Only used when
        ``error_type == "huber"``.
    coord_mode:
        Whether predictions correspond to absolute coordinates (default) or to
        per-step displacements (``"delta"``). When ``"delta"`` is selected the
        ground-truth sequence is differenced along the time dimension so the
        loss is computed on (Δx, Δy).
    final_step_weight:
        Optional multiplicative weight applied to the last time-step error. Set
        to ``1.0`` to disable the weighting.
    reduction:
        Specifies how to reduce the per-agent losses. ``"mean"`` (default)
        averages over valid agents with equal weight, ``"sum"`` accumulates the
        losses, while ``"none"`` returns the per-agent tensor without further
        reduction.
    return_per_agent:
        When ``True`` the function returns the per-agent loss tensor in
        addition to the scalar. This is useful for debugging or for applying
        custom re-weighting schemes outside the function.
    eps:
        Numerical stability constant used when normalising by the number of
        valid time-steps.
    """

    error_type: ErrorType = "l1"
    huber_delta: float = 1.0
    coord_mode: CoordMode = "absolute"
    final_step_weight: float = 1.0
    reduction: LossReduction = "mean"
    return_per_agent: bool = False
    eps: float = 1e-6


def _prepare_targets(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    coord_mode: CoordMode,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if coord_mode == "absolute":
        return y_pred, y_gt

    if coord_mode != "delta":  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported coord_mode: {coord_mode}")

    gt_delta = torch.diff(y_gt, dim=-2, prepend=y_gt[..., :1, :])
    return y_pred, gt_delta


def _compute_step_loss(
    diff: torch.Tensor,
    error_type: ErrorType,
    huber_delta: float,
) -> torch.Tensor:
    if error_type == "l1":
        return diff.abs().sum(dim=-1)
    if error_type != "huber":  # pragma: no cover - defensive
        raise ValueError(f"Unsupported error_type: {error_type}")

    abs_diff = diff.abs()
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive for Huber loss")
    quadratic = 0.5 * abs_diff.pow(2) / huber_delta
    linear = abs_diff - 0.5 * huber_delta
    per_coord = torch.where(abs_diff <= huber_delta, quadratic, linear)
    return per_coord.sum(dim=-1)


def best_of_n_loss(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    valid_agent: Optional[torch.Tensor] = None,
    valid_time: Optional[torch.Tensor] = None,
    config: Optional[BestOfNConfig] = None,
):
    """Compute the Best-of-N regression loss.

    Parameters
    ----------
    y_pred:
        Tensor of shape ``[B, A, K, M, H, 2]`` with candidate trajectories.
    y_gt:
        Tensor of shape ``[B, A, H, 2]`` containing the reference trajectories.
    valid_agent:
        Optional boolean/float mask of shape ``[B, A]`` indicating valid agents.
    valid_time:
        Optional boolean/float mask of shape ``[B, A, H]`` indicating valid
        future time-steps.
    config:
        :class:`BestOfNConfig` controlling the behaviour of the loss.

    Returns
    -------
    loss_scalar:
        Scalar tensor with the reduced loss value (depending on ``reduction``).
    winner_idx:
        Tensor of indices with shape ``[B, A, 2]`` storing the winning
        candidate indices ``(k, m)``.
    per_agent_loss:
        Optional tensor of shape ``[B, A]`` with the loss per agent. Returned
        only when ``config.return_per_agent`` is ``True`` or when
        ``config.reduction == "none"``.
    """

    if config is None:
        config = BestOfNConfig()

    if y_pred.ndim != 6:
        raise ValueError("y_pred must have 6 dimensions [B, A, K, M, H, 2]")
    if y_gt.ndim != 4:
        raise ValueError("y_gt must have 4 dimensions [B, A, H, 2]")

    b, a, k, m, h, c = y_pred.shape
    if c != 2 or y_gt.shape[-1] != 2:
        raise ValueError("The coordinate dimension must be of size 2")
    if y_gt.shape[0] != b or y_gt.shape[1] != a or y_gt.shape[2] != h:
        raise ValueError("Ground truth shape mismatch with predictions")

    y_pred_eval, y_target = _prepare_targets(y_pred, y_gt, config.coord_mode)

    diff = y_pred_eval - y_target.unsqueeze(2).unsqueeze(3)
    per_step_loss = _compute_step_loss(diff, config.error_type, config.huber_delta)

    time_weights = torch.ones_like(per_step_loss)
    if config.final_step_weight != 1.0:
        time_weights[..., -1] = time_weights[..., -1] * config.final_step_weight

    if valid_time is not None:
        if valid_time.shape != (b, a, h):
            raise ValueError("valid_time must have shape [B, A, H]")
        time_weights = time_weights * valid_time.unsqueeze(2).unsqueeze(3)

    weighted_loss = per_step_loss * time_weights
    normaliser = time_weights.sum(dim=-1).clamp_min(config.eps)
    candidate_loss = weighted_loss.sum(dim=-1) / normaliser

    if valid_time is not None:
        zero_mask = normaliser <= config.eps
        candidate_loss = candidate_loss.masked_fill(zero_mask, 0.0)

    candidate_loss_flat = candidate_loss.view(b, a, k * m)
    best_values, best_indices = candidate_loss_flat.min(dim=-1)

    winner_k = best_indices // m
    winner_m = best_indices % m
    winner_idx = torch.stack([winner_k, winner_m], dim=-1)

    if valid_agent is not None:
        if valid_agent.shape != (b, a):
            raise ValueError("valid_agent must have shape [B, A]")
        agent_mask = valid_agent.to(candidate_loss.dtype)
    else:
        agent_mask = torch.ones((b, a), dtype=candidate_loss.dtype, device=candidate_loss.device)

    per_agent_loss = best_values
    masked_loss = per_agent_loss * agent_mask
    valid_counts = agent_mask.sum().clamp_min(config.eps)

    if valid_agent is not None:
        invalid_agents = agent_mask <= 0
        winner_idx = winner_idx.masked_fill(invalid_agents.unsqueeze(-1), -1)

    if config.reduction == "mean":
        loss_scalar = masked_loss.sum() / valid_counts
    elif config.reduction == "sum":
        loss_scalar = masked_loss.sum()
    elif config.reduction == "none":
        loss_scalar = masked_loss
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported reduction: {config.reduction}")

    if config.reduction != "none" and config.return_per_agent:
        return loss_scalar, winner_idx, per_agent_loss
    if config.reduction == "none":
        return loss_scalar, winner_idx, per_agent_loss
    return loss_scalar, winner_idx
