from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Container holding the results of a validation/test run."""

    metrics: Dict[str, float]
    per_scene: Dict[str, Dict[str, float]] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    num_batches: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "per_scene": self.per_scene,
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "samples": self.samples,
            "num_batches": self.num_batches,
        }


def _move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return type(batch)({k: _move_to_device(v, device) for k, v in batch.items()})
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device) for v in batch)
    return batch


def _extract_metrics(output: Any) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if output is None:
        return metrics
    if isinstance(output, Mapping):
        maybe_metrics = output.get("metrics")
        if isinstance(maybe_metrics, Mapping):
            for key, value in maybe_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
                elif torch.is_tensor(value) and value.dim() == 0:
                    metrics[key] = float(value.detach().cpu())
        for key, value in output.items():
            if key == "metrics":
                continue
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif torch.is_tensor(value) and value.dim() == 0:
                metrics[key] = float(value.detach().cpu())
    return metrics


def evaluate(
    cfg: Mapping[str, Any],
    model: nn.Module,
    dataloader: Iterable[Any],
    *,
    device: Optional[torch.device] = None,
    topk: Optional[int] = None,
) -> EvalReport:
    """Run batched evaluation for the provided dataloader."""

    if device is None:
        first_param = next(model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")

    model_was_training = model.training
    model.eval()

    metric_sums: MutableMapping[str, float] = defaultdict(float)
    metric_counts: MutableMapping[str, float] = defaultdict(float)
    per_scene_sums: MutableMapping[str, MutableMapping[str, float]] = defaultdict(lambda: defaultdict(float))
    per_scene_counts: MutableMapping[str, float] = defaultdict(float)
    collected_samples: List[Dict[str, Any]] = []

    rounds = int(cfg.get("training", {}).get("rounds", 3))

    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_to_device(batch, device)
            batch_metrics: Dict[str, float] = {}
            for round_idx in range(rounds):
                if hasattr(model, "inference_round"):
                    output = model.inference_round(batch, round_idx=round_idx, topk=topk)
                elif hasattr(model, "validation_step"):
                    output = model.validation_step(batch, round_idx=round_idx)
                else:
                    if round_idx > 0:
                        continue
                    output = model(batch)
                batch_metrics.update(_extract_metrics(output))
                if isinstance(output, Mapping):
                    sample = output.get("sample")
                    if sample is not None:
                        collected_samples.append(sample)
            if not batch_metrics:
                continue
            batch_count += 1
            weight = float(batch_metrics.pop("weight", 1.0))
            for key, value in batch_metrics.items():
                metric_sums[key] += value * weight
                metric_counts[key] += weight
            scene_label = None
            if isinstance(batch, Mapping):
                scene_label = batch.get("scene_type") or batch.get("scene_label")
            if scene_label is not None:
                per_scene_counts[str(scene_label)] += weight
                for key, value in batch_metrics.items():
                    per_scene_sums[str(scene_label)][key] += value * weight

    metrics = {k: metric_sums[k] / max(metric_counts[k], 1.0) for k in metric_sums}
    per_scene = {
        scene: {k: sums[k] / max(per_scene_counts[scene], 1.0) for k in sums}
        for scene, sums in per_scene_sums.items()
    }

    if model_was_training:
        model.train()

    return EvalReport(
        metrics=metrics,
        per_scene=per_scene,
        samples=collected_samples,
        num_batches=batch_count,
    )


__all__ = ["EvalReport", "evaluate"]