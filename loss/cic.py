"""Consistency and Intervention Consistency (CIC) loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

Reduction = Literal["mean", "sum", "none"]
TrajectoryMetric = Literal["l1", "l2"]


@dataclass
class CICLossConfig:
    """Configuration for :func:`run_cic_intervention`.

    Attributes
    ----------
    consistency_weight:
        Weight applied to the low-risk consistency term.
    sensitivity_weight:
        Weight applied to the high-risk sensitivity term.
    consistency_score_weight:
        Optional weight applied to the score difference regulariser for
        low-risk interventions.
    sensitivity_score_weight:
        Optional weight applied to the score degradation term for high-risk
        interventions.
    sensitivity_margin:
        Margin (in metres) the high-risk interventions must exceed.
    score_margin:
        Minimum drop in the candidate scores required for high-risk
        interventions.
    reduction:
        Reduction applied across agents.
    trajectory_metric:
        Metric used to measure the distance between baseline and intervened
        trajectories.
    max_low_edges:
        Optional limit on the number of low-risk edges to intervene per sample.
    max_high_edges:
        Optional limit on the number of high-risk edges to intervene per sample.
    low_strength:
        Multiplicative factor applied to the intervention strength for low-risk
        edges. Forward hook implementations can use this value to adjust their
        masks.
    high_strength:
        Multiplicative factor applied to the intervention strength for high-risk
        edges.
    eps:
        Numerical stability constant.
    """

    consistency_weight: float = 1.0
    sensitivity_weight: float = 1.0
    consistency_score_weight: float = 0.0
    sensitivity_score_weight: float = 0.0
    sensitivity_margin: float = 1.0
    score_margin: float = 0.5
    reduction: Reduction = "mean"
    trajectory_metric: TrajectoryMetric = "l2"
    max_low_edges: Optional[int] = None
    max_high_edges: Optional[int] = None
    low_strength: float = 1.0
    high_strength: float = 1.0
    eps: float = 1e-6


@dataclass
class CandidateSet:
    """Container for model outputs under a particular intervention."""

    trajectories: torch.Tensor
    scores: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.trajectories.ndim != 6:
            raise ValueError("trajectories must have shape [B, A, K, M, H, 2]")
        if self.trajectories.size(-1) != 2:
            raise ValueError("The coordinate dimension must be 2")
        if self.scores is not None:
            expected = self.trajectories.shape[:-2]
            if self.scores.shape != expected:
                raise ValueError(f"scores must have shape {expected}")


def _subsample_edges(edges: Optional[torch.Tensor], limit: Optional[int]) -> Optional[torch.Tensor]:
    if edges is None or limit is None:
        return edges
    if edges.ndim != 3:
        raise ValueError("edges must have shape [B, E, ...]")
    if edges.size(1) <= limit:
        return edges
    perm = torch.randperm(edges.size(1), device=edges.device)[:limit]
    return edges[:, perm]


def _trajectory_distance(
    base: torch.Tensor,
    intervened: torch.Tensor,
    metric: TrajectoryMetric,
) -> torch.Tensor:
    diff = base - intervened
    if metric == "l2":
        per_step = torch.linalg.norm(diff, dim=-1)
    elif metric == "l1":
        per_step = diff.abs().sum(dim=-1)
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported trajectory metric: {metric}")
    return per_step.mean(dim=-1)


def _agent_mask(valid_agent: Optional[torch.Tensor], shape: Tuple[int, int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if valid_agent is None:
        return torch.ones(shape, dtype=dtype, device=device)
    if valid_agent.shape != shape:
        raise ValueError(f"valid_agent must have shape {shape}")
    return valid_agent.to(dtype=dtype, device=device)


def _reduce_agents(per_agent_loss: torch.Tensor, agent_mask: torch.Tensor, reduction: Reduction, eps: float):
    masked = per_agent_loss * agent_mask
    if reduction == "mean":
        denom = agent_mask.sum().clamp_min(eps)
        scalar = masked.sum() / denom
    elif reduction == "sum":
        scalar = masked.sum()
    elif reduction == "none":
        scalar = masked
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported reduction: {reduction}")
    return scalar, masked


def run_cic_intervention(
    forward_fn: Callable[[Optional[Dict[str, torch.Tensor]]], CandidateSet],
    low_risk_edges: Optional[torch.Tensor] = None,
    high_risk_edges: Optional[torch.Tensor] = None,
    valid_agent: Optional[torch.Tensor] = None,
    config: Optional[CICLossConfig] = None,
):
    """Execute CIC interventions and compute the associated loss.

    Parameters
    ----------
    forward_fn:
        Callable that performs a forward pass. It must accept a dictionary with
        the keys ``"edges"`` (tensor describing the edges to drop), ``"mode"``
        (``"low"`` or ``"high"``) and ``"strength"`` (float). Passing ``None``
        should execute the baseline forward pass.
    low_risk_edges:
        Tensor with shape ``[B, E_low, D]`` describing the low-risk edges to be
        dropped together. The content of the last dimension is left to the
        calling code (e.g. neighbour indices).
    high_risk_edges:
        Tensor with shape ``[B, E_high, D]`` describing the high-risk edges.
    valid_agent:
        Optional mask ``[B, A]`` to exclude invalid agents from the reduction.
    config:
        Optional :class:`CICLossConfig` controlling weights and margins.

    Returns
    -------
    cic_loss_scalar:
        Scalar tensor with the reduced CIC loss (or per-agent tensor when
        ``reduction == "none"``).
    diagnostics:
        Dictionary with aggregated metrics used for monitoring.
    """

    if config is None:
        config = CICLossConfig()

    baseline = forward_fn(None)
    if not isinstance(baseline, CandidateSet):
        raise TypeError("forward_fn must return a CandidateSet instance")
    baseline.validate()

    b, a = baseline.trajectories.shape[:2]
    agent_mask = _agent_mask(valid_agent, (b, a), baseline.trajectories.dtype, baseline.trajectories.device)

    diagnostics: Dict[str, torch.Tensor] = {}
    per_agent_losses = baseline.trajectories.new_zeros((b, a))

    low_edges = _subsample_edges(low_risk_edges, config.max_low_edges)
    high_edges = _subsample_edges(high_risk_edges, config.max_high_edges)

    if config.consistency_weight > 0 and low_edges is not None and low_edges.numel() > 0:
        low_result = forward_fn({"edges": low_edges, "mode": "low", "strength": config.low_strength})
        if not isinstance(low_result, CandidateSet):
            raise TypeError("forward_fn must return a CandidateSet instance")
        low_result.validate()
        dist = _trajectory_distance(baseline.trajectories, low_result.trajectories, config.trajectory_metric)
        dist = dist.mean(dim=(-2, -1))
        per_agent_losses = per_agent_losses + config.consistency_weight * dist
        diagnostics["consistency_mean"] = dist.mean().detach()
        diagnostics["low_edges"] = torch.tensor(float(low_edges.size(1)), device=dist.device)
        if config.consistency_score_weight > 0 and baseline.scores is not None and low_result.scores is not None:
            score_diff = (baseline.scores - low_result.scores).abs().mean(dim=(-2, -1))
            per_agent_losses = per_agent_losses + config.consistency_score_weight * score_diff
            diagnostics["consistency_score_mean"] = score_diff.mean().detach()
    else:
        diagnostics["consistency_mean"] = torch.tensor(0.0, device=per_agent_losses.device)
        diagnostics["low_edges"] = torch.tensor(0.0, device=per_agent_losses.device)

    if config.sensitivity_weight > 0 and high_edges is not None and high_edges.numel() > 0:
        high_result = forward_fn({"edges": high_edges, "mode": "high", "strength": config.high_strength})
        if not isinstance(high_result, CandidateSet):
            raise TypeError("forward_fn must return a CandidateSet instance")
        high_result.validate()
        dist = _trajectory_distance(baseline.trajectories, high_result.trajectories, config.trajectory_metric)
        dist = dist.mean(dim=(-2, -1))
        violation = F.relu(config.sensitivity_margin - dist)
        per_agent_losses = per_agent_losses + config.sensitivity_weight * violation
        diagnostics["sensitivity_violation"] = violation.mean().detach()
        diagnostics["high_edges"] = torch.tensor(float(high_edges.size(1)), device=dist.device)
        hit_rate = (dist > config.sensitivity_margin).float().mean()
        diagnostics["sensitivity_hit_rate"] = hit_rate.detach()
        if config.sensitivity_score_weight > 0 and baseline.scores is not None and high_result.scores is not None:
            score_drop = baseline.scores - high_result.scores
            score_drop = score_drop.mean(dim=(-2, -1))
            score_violation = F.relu(config.score_margin - score_drop)
            per_agent_losses = per_agent_losses + config.sensitivity_score_weight * score_violation
            diagnostics["sensitivity_score_violation"] = score_violation.mean().detach()
    else:
        diagnostics["sensitivity_violation"] = torch.tensor(0.0, device=per_agent_losses.device)
        diagnostics["high_edges"] = torch.tensor(0.0, device=per_agent_losses.device)
        diagnostics["sensitivity_hit_rate"] = torch.tensor(0.0, device=per_agent_losses.device)

    loss_scalar, masked = _reduce_agents(per_agent_losses, agent_mask, config.reduction, config.eps)
    if config.reduction == "none":
        diagnostics["per_agent_loss"] = masked.detach()

    return loss_scalar, diagnostics