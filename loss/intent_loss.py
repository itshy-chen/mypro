"""Weak supervision and distillation losses for interaction intent predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

@dataclass
class IntentLossConfig:
    """Configuration for :func:`intent_loss`.

    Attributes
    ----------
    supervision_weight:
        Weight applied to the pseudo-label supervision term.
    distill_weight:
        Weight applied to the distillation alignment term.
    temperature:
        Temperature used for the distillation soft targets.
    label_smoothing:
        Amount of label smoothing applied to the hard pseudo-labels.
    class_weights:
        Optional tensor with per-class weights to mitigate imbalance.
    focal_gamma:
        Focal loss gamma parameter. Set to ``0`` to disable the focal re-weighting.
    reduction:
        Reduction applied over the agent dimension.
    eps:
        Numerical stability constant.
    """

    supervision_weight: float = 1.0
    distill_weight: float = 1.0
    temperature: float = 1.0
    label_smoothing: float = 0.0
    class_weights: Optional[torch.Tensor] = None
    focal_gamma: float = 0.0
    reduction: str = "mean"
    eps: float = 1e-6


def _prepare_masks(
    valid_agent: Optional[torch.Tensor],
    pseudo_mask: Optional[torch.Tensor],
    distill_mask: Optional[torch.Tensor],
    shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
):
    if valid_agent is None:
        agent_mask = torch.ones(shape, device=device, dtype=dtype)
    else:
        if valid_agent.shape != shape:
            raise ValueError(f"valid_agent must have shape {shape}")
        agent_mask = valid_agent.to(device=device, dtype=dtype)

    if pseudo_mask is not None:
        if pseudo_mask.shape != shape:
            raise ValueError("pseudo_mask must have shape [B, A]")
        pseudo_mask = pseudo_mask.to(device=device, dtype=dtype)
    if distill_mask is not None:
        if distill_mask.shape != shape:
            raise ValueError("distill_mask must have shape [B, A]")
        distill_mask = distill_mask.to(device=device, dtype=dtype)
    return agent_mask, pseudo_mask, distill_mask


def _supervision_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor],
    config: IntentLossConfig,
) -> torch.Tensor:
    b, a, k, c = logits.shape
    flat_logits = logits.view(b * a * k, c)
    labels = labels.view(b * a * k)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    probs = log_probs.exp()
    one_hot = F.one_hot(labels.long(), num_classes=c).to(log_probs.dtype)
    if config.label_smoothing > 0:
        smooth = config.label_smoothing / c
        one_hot = one_hot * (1 - config.label_smoothing) + smooth
    ce = -(one_hot * log_probs).sum(dim=-1)
    if config.class_weights is not None:
        weights = config.class_weights.to(log_probs.device, dtype=log_probs.dtype)
        ce = ce * weights[labels.long()]
    if config.focal_gamma > 0:
        pt = (one_hot * probs).sum(dim=-1).clamp_min(config.eps)
        ce = ((1 - pt) ** config.focal_gamma) * ce
    ce = ce.view(b, a, k)
    if mask is not None:
        ce = ce * mask.unsqueeze(-1)
    return ce.mean(dim=-1)


def _distillation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor],
    config: IntentLossConfig,
) -> torch.Tensor:
    b, a, k, c = logits.shape
    student_log_probs = F.log_softmax(logits / max(config.temperature, config.eps), dim=-1)
    if targets.shape != logits.shape:
        raise ValueError("distillation targets must have shape [B, A, K, C]")
    teacher_probs = F.softmax(targets / max(config.temperature, config.eps), dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none") * (config.temperature ** 2)
    kl = kl.sum(dim=-1)
    if mask is not None:
        kl = kl * mask.unsqueeze(-1)
    return kl.mean(dim=-1)


def _reduce(loss: torch.Tensor, agent_mask: torch.Tensor, reduction: str, eps: float) -> torch.Tensor:
    masked = loss * agent_mask
    if reduction == "mean":
        denom = agent_mask.sum().clamp_min(eps)
        return masked.sum() / denom
    if reduction == "sum":
        return masked.sum()
    if reduction == "none":
        return masked
    raise ValueError(f"Unsupported reduction: {reduction}")


def intent_loss(
    logits: torch.Tensor,
    pseudo_labels: Optional[torch.Tensor] = None,
    pseudo_mask: Optional[torch.Tensor] = None,
    distill_targets: Optional[torch.Tensor] = None,
    distill_mask: Optional[torch.Tensor] = None,
    valid_agent: Optional[torch.Tensor] = None,
    config: Optional[IntentLossConfig] = None,
):
    """Compute intent supervision and distillation losses."""

    if config is None:
        config = IntentLossConfig()

    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, A, K, 4]")

    b, a, k, c = logits.shape
    if c != 4:
        raise ValueError("Expected 4 intent classes")

    device = logits.device
    dtype = logits.dtype
    agent_mask, pseudo_mask, distill_mask = _prepare_masks(
        valid_agent,
        pseudo_mask,
        distill_mask,
        (b, a),
        device,
        dtype,
    )

    loss_terms = []
    diagnostics = {}

    if config.supervision_weight > 0 and pseudo_labels is not None:
        if pseudo_labels.shape != (b, a, k):
            raise ValueError("pseudo_labels must have shape [B, A, K]")
        sup = _supervision_loss(logits, pseudo_labels, pseudo_mask, config)
        loss_terms.append(config.supervision_weight * sup)
        diagnostics["supervision_mean"] = sup.mean().detach()
    else:
        diagnostics["supervision_mean"] = torch.tensor(0.0, device=device)

    if config.distill_weight > 0 and distill_targets is not None:
        distill = _distillation_loss(logits, distill_targets, distill_mask, config)
        loss_terms.append(config.distill_weight * distill)
        diagnostics["distill_mean"] = distill.mean().detach()
    else:
        diagnostics["distill_mean"] = torch.tensor(0.0, device=device)

    if not loss_terms:
        return torch.tensor(0.0, device=device, dtype=dtype), diagnostics

    total = sum(loss_terms)
    loss_scalar = _reduce(total, agent_mask, config.reduction, config.eps)
    if config.reduction == "none":
        diagnostics["per_agent_loss"] = loss_scalar.detach()

    return loss_scalar, diagnostics
