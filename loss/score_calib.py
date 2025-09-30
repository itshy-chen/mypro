"""Score calibration and risk-aware rescoring for trajectory candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

Reduction = Literal["mean", "sum", "none"]
LossType = Literal["cross_entropy", "bce", "rank"]


@dataclass
class ScoreCalibrationConfig:
    """Configuration container for :func:`calibrate_scores`.

    Attributes
    ----------
    temperature:
        Temperature used when computing the softmax probabilities. Values larger
        than 1.0 flatten the distribution while values below 1.0 sharpen it.
    loss_type:
        Calibration loss to use. ``"cross_entropy"`` supervises the winner using
        categorical cross-entropy. ``"rank"`` applies a pairwise hinge loss
        ensuring the winner scores higher than the other candidates by a margin.
        ``"bce"`` interprets the winner as a binary target per candidate.
    risk_weights:
        Optional dictionary mapping safety term names to the weight applied when
        subtracting them from the raw scores.
    reduction:
        Reduction mode for the calibration loss. ``"mean"`` averages over valid
        agents, ``"sum"`` accumulates and ``"none"`` returns the per-agent
        losses.
    ece_bins:
        Number of bins used when computing the Expected Calibration Error.
    compute_ece:
        Whether to compute ECE diagnostics.
    margin:
        Margin used by the pairwise ranking loss.
    eps:
        Numerical stability constant.
    """

    temperature: float = 1.0
    loss_type: LossType = "cross_entropy"
    risk_weights: Optional[Dict[str, float]] = None
    reduction: Reduction = "mean"
    ece_bins: int = 10
    compute_ece: bool = True
    margin: float = 0.2
    eps: float = 1e-6


def _apply_safety_terms(
    scores: torch.Tensor,
    safety_terms: Optional[Dict[str, torch.Tensor]],
    weights: Optional[Dict[str, float]],
) -> torch.Tensor:
    if not safety_terms or not weights:
        return scores
    adjusted = scores
    for key, term in safety_terms.items():
        weight = weights.get(key)
        if weight is None:
            continue
        if term.shape != scores.shape:
            raise ValueError(f"Safety term '{key}' has shape {term.shape} but expected {scores.shape}")
        adjusted = adjusted - weight * term
    return adjusted


def _valid_agent_mask(mask: Optional[torch.Tensor], shape: Tuple[int, int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(shape, dtype=dtype, device=device)
    if mask.shape != shape:
        raise ValueError(f"valid_agent must have shape {shape}")
    return mask.to(dtype=dtype, device=device)


def _cross_entropy_loss(
    logits: torch.Tensor,
    winner_idx: torch.Tensor,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    b, a, k, m = logits.shape
    flat_logits = logits.view(b, a, k * m)
    scaled_logits = flat_logits / max(temperature, eps)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    linear_idx = winner_idx[..., 0] * m + winner_idx[..., 1]
    gather = log_probs.gather(-1, linear_idx.unsqueeze(-1)).squeeze(-1)
    loss = -gather
    return loss


def _bce_loss(
    logits: torch.Tensor,
    winner_idx: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    b, a, k, m = logits.shape
    flat_logits = logits.view(b, a, k * m)
    linear_idx = winner_idx[..., 0] * m + winner_idx[..., 1]
    target = torch.zeros_like(flat_logits)
    target.scatter_(-1, linear_idx.unsqueeze(-1), 1.0)
    loss = F.binary_cross_entropy_with_logits(flat_logits, target, reduction="none")
    loss = loss.mean(dim=-1)
    return loss


def _ranking_loss(
    logits: torch.Tensor,
    winner_idx: torch.Tensor,
    margin: float,
    eps: float,
) -> torch.Tensor:
    b, a, k, m = logits.shape
    flat_logits = logits.view(b, a, k * m)
    linear_idx = winner_idx[..., 0] * m + winner_idx[..., 1]
    winner_scores = flat_logits.gather(-1, linear_idx.unsqueeze(-1))
    diff = winner_scores - flat_logits
    loss = F.relu(margin - diff)
    loss = loss.mean(dim=-1)
    return loss


def _reduce_loss(
    per_agent_loss: torch.Tensor,
    agent_mask: torch.Tensor,
    reduction: Reduction,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    masked = per_agent_loss * agent_mask
    if reduction == "mean":
        denom = agent_mask.sum().clamp_min(eps)
        scalar = masked.sum() / denom
    elif reduction == "sum":
        scalar = masked.sum()
    elif reduction == "none":
        scalar = masked
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return scalar, masked


def _ece(
    logits: torch.Tensor,
    winner_idx: torch.Tensor,
    agent_mask: torch.Tensor,
    bins: int,
    temperature: float,
    eps: float,
) -> torch.Tensor:
    b, a, k, m = logits.shape
    flat_logits = logits.view(b, a, k * m)
    probs = F.softmax(flat_logits / max(temperature, eps), dim=-1)
    linear_idx = winner_idx[..., 0] * m + winner_idx[..., 1]
    winner_prob = probs.gather(-1, linear_idx.unsqueeze(-1)).squeeze(-1)
    mask = agent_mask > 0.0
    conf = winner_prob[mask]
    acc = torch.ones_like(conf)
    bin_edges = torch.linspace(0, 1, bins + 1, device=logits.device, dtype=logits.dtype)
    ece = logits.new_tensor(0.0)
    for i in range(bins):
        mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
        if mask.any():
            bin_conf = conf[mask].mean()
            bin_acc = acc[mask].mean()
            ece = ece + conf[mask].numel() * torch.abs(bin_conf - bin_acc)
    total = conf.numel()
    if total == 0:
        return ece
    return ece / total


def calibrate_scores(
    raw_scores: torch.Tensor,
    winner_idx: torch.Tensor,
    safety_terms: Optional[Dict[str, torch.Tensor]] = None,
    valid_agent: Optional[torch.Tensor] = None,
    config: Optional[ScoreCalibrationConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Rescore candidate trajectories using safety terms and calibrate logits."""

    if config is None:
        config = ScoreCalibrationConfig()

    if raw_scores.ndim != 4:
        raise ValueError("raw_scores must have shape [B, A, K, M]")

    b, a, k, m = raw_scores.shape
    if winner_idx.shape != (b, a, 2):
        raise ValueError("winner_idx must have shape [B, A, 2]")

    adjusted_scores = _apply_safety_terms(raw_scores, safety_terms, config.risk_weights)

    agent_mask = _valid_agent_mask(valid_agent, (b, a), raw_scores.dtype, raw_scores.device)

    loss_scalar: torch.Tensor
    if config.loss_type == "cross_entropy":
        per_agent = _cross_entropy_loss(adjusted_scores, winner_idx, config.temperature, config.eps)
    elif config.loss_type == "bce":
        per_agent = _bce_loss(adjusted_scores, winner_idx, config.eps)
    elif config.loss_type == "rank":
        per_agent = _ranking_loss(adjusted_scores, winner_idx, config.margin, config.eps)
    else:
        raise ValueError(f"Unsupported loss_type: {config.loss_type}")

    loss_scalar, masked_loss = _reduce_loss(per_agent, agent_mask, config.reduction, config.eps)

    rescored = adjusted_scores
    diagnostics: Dict[str, torch.Tensor] = {"loss": loss_scalar.detach()}
    if config.compute_ece:
        diagnostics["ece"] = _ece(adjusted_scores.detach(), winner_idx, agent_mask, config.ece_bins, config.temperature, config.eps)
    if config.reduction == "none":
        diagnostics["per_agent_loss"] = masked_loss.detach()

    return rescored, loss_scalar, diagnostics