"""Differentiable safety terms for trajectory candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F


NeighborAggregation = Literal["max", "mean", "softmin"]
CandidateAggregation = Literal["mean", "sum"]


@dataclass
class SafetyLossConfig:
    """Configuration for :func:`compute_safety_terms`.

    Attributes
    ----------
    enable_map_term:
        If ``True`` the map consistency penalty is computed.
    enable_interaction_term:
        If ``True`` the neighbour interaction penalty is computed.
    map_weight:
        Weight assigned to the map term when the combined scalar loss is
        requested.
    interaction_weight:
        Weight assigned to the interaction term when computing the scalar loss.
    map_safe_margin:
        Margin (in metres) inside the drivable area considered safe. Positions
        farther inside than the margin receive no penalty.
    map_softplus_beta:
        Controls the softness of the hinge used for map violations.
    interaction_margin:
        Desired minimum distance to neighbouring agents. Distances smaller than
        this value are penalised.
    interaction_softness:
        Temperature used in the smooth penalty for inter-agent distances.
    neighbor_topk:
        If provided, limits the number of neighbours considered per agent by
        selecting the first ``neighbor_topk`` entries.
    candidate_topk:
        If provided, limits the number of trajectory hypotheses processed along
        the ``K`` dimension. The remaining hypotheses keep zero penalties.
    neighbor_aggregation:
        Strategy used to aggregate neighbour penalties. ``"max"`` selects the
        worst neighbour, ``"mean"`` averages them and ``"softmin"`` performs a
        differentiable soft-minimum controlled by ``interaction_softness``.
    candidate_aggregation:
        Reduction over the time dimension for each candidate. ``"mean"`` uses a
        weighted average, ``"sum"`` accumulates the penalties.
    eps:
        Numerical stability constant.
    """

    enable_map_term: bool = True
    enable_interaction_term: bool = True
    map_weight: float = 1.0
    interaction_weight: float = 1.0
    map_safe_margin: float = 0.0
    map_softplus_beta: float = 10.0
    interaction_margin: float = 1.5
    interaction_softness: float = 0.5
    neighbor_topk: Optional[int] = None
    candidate_topk: Optional[int] = None
    neighbor_aggregation: NeighborAggregation = "softmin"
    candidate_aggregation: CandidateAggregation = "mean"
    eps: float = 1e-6


def _map_penalty(
    positions: torch.Tensor,
    map_query: Callable[[torch.Tensor], torch.Tensor],
    config: SafetyLossConfig,
    valid_time: Optional[torch.Tensor],
) -> torch.Tensor:
    distances = map_query(positions)
    if distances.shape != positions.shape[:-1]:
        raise ValueError("map_query must return a tensor matching [B, A, K, M, H]")

    violation = F.softplus((config.map_safe_margin - distances) * config.map_softplus_beta)
    violation = violation / max(config.map_softplus_beta, config.eps)

    if valid_time is not None:
        violation = violation * valid_time.unsqueeze(2).unsqueeze(3)
    return violation


def _aggregate_time(
    penalty: torch.Tensor,
    valid_time: Optional[torch.Tensor],
    config: SafetyLossConfig,
) -> torch.Tensor:
    if valid_time is not None:
        weights = valid_time.unsqueeze(2).unsqueeze(3)
        weighted = penalty * weights
        normaliser = weights.sum(dim=-1).clamp_min(config.eps)
    else:
        weights = None
        weighted = penalty
        normaliser = penalty.new_full(penalty.shape[:-1], penalty.shape[-1], dtype=penalty.dtype)

    if config.candidate_aggregation == "mean":
        return weighted.sum(dim=-1) / normaliser

    if config.candidate_aggregation == "sum":
        if weights is not None:
            return weighted.sum(dim=-1)
        return penalty.sum(dim=-1)

    raise ValueError(f"Unsupported candidate aggregation: {config.candidate_aggregation}")


def _interaction_penalty(
    positions: torch.Tensor,
    neighbor_trajs: torch.Tensor,
    config: SafetyLossConfig,
    valid_time: Optional[torch.Tensor],
    neighbor_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    b, a, k, m, h, _ = positions.shape
    _, _, n, _, _ = neighbor_trajs.shape
    rel = positions.unsqueeze(4) - neighbor_trajs.unsqueeze(2).unsqueeze(3)
    distances = torch.linalg.norm(rel, dim=-1)

    if neighbor_mask is not None:
        if neighbor_mask.shape != (b, a, n):
            raise ValueError("neighbor_mask must have shape [B, A, N]")
        mask = neighbor_mask.unsqueeze(2).unsqueeze(3).unsqueeze(-1)
        safe_distance = distances.new_full((), config.interaction_margin * 4.0)
        distances = torch.where(mask.bool(), distances, safe_distance)
    else:
        mask = None

    proximity = F.softplus((config.interaction_margin - distances) / config.interaction_softness)
    proximity = proximity * config.interaction_softness

    if valid_time is not None:
        proximity = proximity * valid_time.unsqueeze(2).unsqueeze(3).unsqueeze(4)

    if config.neighbor_aggregation == "max":
        penalty = proximity.max(dim=4).values
    elif config.neighbor_aggregation == "mean":
        if mask is None:
            penalty = proximity.mean(dim=4)
        else:
            weights = mask.to(proximity.dtype)
            penalty = (proximity * weights).sum(dim=4) / weights.sum(dim=4).clamp_min(config.eps)
    elif config.neighbor_aggregation == "softmin":
        logits = -proximity / max(config.interaction_softness, config.eps)
        penalty = -(torch.logsumexp(logits, dim=4) * config.interaction_softness)
    else:
        raise ValueError(f"Unsupported neighbor aggregation: {config.neighbor_aggregation}")

    return penalty


def compute_safety_terms(
    y_pred: torch.Tensor,
    map_query: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    neighbor_trajs: Optional[torch.Tensor] = None,
    valid_agent: Optional[torch.Tensor] = None,
    valid_time: Optional[torch.Tensor] = None,
    neighbor_mask: Optional[torch.Tensor] = None,
    config: Optional[SafetyLossConfig] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute differentiable safety terms for candidate trajectories.

    Parameters
    ----------
    y_pred:
        Candidate trajectories with shape ``[B, A, K, M, H, 2]``.
    map_query:
        Callable receiving a tensor of shape ``[B, A, K, M, H, 2]`` and returning
        signed distances to the feasible region with shape ``[B, A, K, M, H]``.
    neighbor_trajs:
        Optional tensor of neighbour trajectories with shape ``[B, A, N, H, 2]``.
    valid_agent:
        Optional mask of shape ``[B, A]``.
    valid_time:
        Optional mask of shape ``[B, A, H]``.
    neighbor_mask:
        Optional mask of shape ``[B, A, N]`` specifying valid neighbours.
    config:
        :class:`SafetyLossConfig` instance.

    Returns
    -------
    safety_terms:
        Dictionary with per-candidate penalties. Each value has shape
        ``[B, A, K, M]``.
    safety_loss_scalar:
        Optional scalar tensor with the aggregated safety loss when either map
        or interaction terms are enabled.
    diagnostics:
        Dictionary with summary statistics for monitoring.
    """

    if config is None:
        config = SafetyLossConfig()

    if y_pred.ndim != 6:
        raise ValueError("y_pred must have shape [B, A, K, M, H, 2]")
    b, a, k, m, h, _ = y_pred.shape

    if config.candidate_topk is not None and (config.candidate_topk <= 0 or config.candidate_topk > k):
        raise ValueError("candidate_topk must be within (0, K]")
    if config.neighbor_topk is not None and neighbor_trajs is not None:
        if config.neighbor_topk <= 0:
            raise ValueError("neighbor_topk must be positive")
        neighbor_trajs = neighbor_trajs[:, :, : config.neighbor_topk, ...]
        if neighbor_mask is not None:
            neighbor_mask = neighbor_mask[..., : config.neighbor_topk]

    processed_pred = y_pred
    if config.candidate_topk is not None:
        processed_pred = processed_pred[:, :, : config.candidate_topk, ...]

    safety_terms: Dict[str, torch.Tensor] = {}
    diagnostics: Dict[str, torch.Tensor] = {}

    map_penalty_reduced = None
    if config.enable_map_term and map_query is not None:
        map_penalty = _map_penalty(processed_pred, map_query, config, valid_time)
        map_penalty_reduced = _aggregate_time(map_penalty, valid_time, config)
        full = torch.zeros((b, a, k, m), device=y_pred.device, dtype=y_pred.dtype)
        full[:, :, : map_penalty_reduced.shape[2], :] = map_penalty_reduced
        safety_terms["map"] = full
        diagnostics["map_mean"] = map_penalty_reduced.mean()
    elif config.enable_map_term:
        raise ValueError("map_query must be provided when enable_map_term is True")

    interaction_penalty_reduced = None
    if config.enable_interaction_term and neighbor_trajs is not None:
        penalty = _interaction_penalty(processed_pred, neighbor_trajs, config, valid_time, neighbor_mask)
        penalty = _aggregate_time(penalty, valid_time, config)
        interaction_penalty_reduced = penalty
        full_term = torch.zeros((b, a, k, m), device=y_pred.device, dtype=y_pred.dtype)
        full_term[:, :, : penalty.shape[2], :] = penalty
        safety_terms["interaction"] = full_term
        diagnostics["interaction_mean"] = penalty.mean()
    elif config.enable_interaction_term:
        raise ValueError("neighbor_trajs must be provided when enable_interaction_term is True")

    if valid_agent is not None:
        if valid_agent.shape != (b, a):
            raise ValueError("valid_agent must have shape [B, A]")
        agent_mask = valid_agent.to(y_pred.dtype)
    else:
        agent_mask = torch.ones((b, a), device=y_pred.device, dtype=y_pred.dtype)

    safety_loss_scalar = None
    total = None
    if map_penalty_reduced is not None:
        total = config.map_weight * map_penalty_reduced
    if interaction_penalty_reduced is not None:
        total = interaction_penalty_reduced * config.interaction_weight if total is None else total + config.interaction_weight * interaction_penalty_reduced

    if total is not None:
        agent_mask_expanded = agent_mask.unsqueeze(-1).unsqueeze(-1)
        masked = total * agent_mask_expanded
        loss_per_agent = masked.mean(dim=(-2, -1))
        denom = agent_mask.sum().clamp_min(config.eps)
        safety_loss_scalar = (loss_per_agent * agent_mask).sum() / denom

    return safety_terms, safety_loss_scalar, diagnostics