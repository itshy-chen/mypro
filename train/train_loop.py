from __future__ import annotations

import copy

import dataclasses
import importlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .evaluator import EvalReport, evaluate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    """Represents the state of a training run.

    Attributes
    ----------
    work_dir:
        Directory that contains checkpoints and logs for the run.
    epoch:
        Index of the last finished epoch (0 based).
    global_step:
        Number of optimisation steps that have been performed.
    best_metric:
        Best validation metric that has been seen so far.
    best_metric_name:
        Name of the metric that is being monitored for early stopping.
    best_checkpoint_path:
        Path to the checkpoint file that stores the best model weights.
    last_eval_report:
        Evaluation report returned by :func:`train.evaluator.evaluate` for the
        most recent validation run.
    history:
        List of dictionaries containing logged metrics.
    stopped_early:
        ``True`` when the loop terminated because of the early stopping
        criteria, ``False`` otherwise.
    resume_checkpoint:
        Path of the checkpoint that has been used for warm starting the run.
    summary:
        A short, human readable summary of the training outcome.
    """

    work_dir: Path
    epoch: int = 0
    global_step: int = 0
    best_metric: float = math.inf
    best_metric_name: str = "val_loss"
    best_checkpoint_path: Optional[Path] = None
    last_eval_report: Optional[EvalReport] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False
    resume_checkpoint: Optional[Path] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def ensure_logging(log_level: int = logging.INFO) -> None:
    """Initialise a default logging configuration.

    The project relies heavily on logging for observability.  When the client
    code has not set up any handlers we install a simple stdout handler so the
    emitted log records do not vanish silently.
    """

    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=log_level, format="[%(asctime)s][%(levelname)s] %(message)s")


def resolve_target(target: Union[str, Callable[..., Any]]) -> Callable[..., Any]:
    """Resolve a dotted path or return the callable as-is."""

    if callable(target):
        return target
    if not isinstance(target, str):
        raise TypeError(f"Cannot resolve target from {target!r}")
    if ":" in target:
        module_name, attr = target.split(":", 1)
    else:
        module_name, _, attr = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Cannot import target from string without module name: {target!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def instantiate_spec(
    spec: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Instantiate an object described by ``spec``.

    The helper understands three formats:

    1. ``None`` which returns ``None``.
    2. A callable which is invoked with ``*args`` and ``**kwargs``.
    3. A mapping that contains at least a ``"target"`` key.  Additional
       arguments can be supplied under ``"args"``/``"kwargs"`` or directly as
       remaining dictionary keys.
    """

    if spec is None:
        return None
    if isinstance(spec, DataLoader):
        return spec
    if callable(spec):
        return spec(*args, **kwargs)
    if isinstance(spec, str):
        target = resolve_target(spec)
        return target(*args, **kwargs)
    if isinstance(spec, Mapping):
        spec_dict = dict(spec)
        target = spec_dict.pop("target", None)
        if target is None:
            # ``type``/``class`` style keys are common in older configuration
            # files.  Accept them as aliases for ``target`` so that legacy
            # configs keep working.
            for alias in ("type", "class", "cls", "callable", "factory"):
                target = spec_dict.pop(alias, None)
                if target is not None:
                    break
        if target is None:
            raise ValueError(
                "Mapping specifications must provide a 'target' key (or one of"
                " the supported aliases: type/class/cls/callable/factory)"
            )
        target_fn = resolve_target(target)
        spec_args = list(spec_dict.pop("args", []))
        spec_kwargs = dict(spec_dict.pop("kwargs", {}))
        if "params" in spec_dict:
            params = spec_dict.pop("params")
            if isinstance(params, Mapping):
                spec_kwargs.update(params)
            elif isinstance(params, Sequence) and not isinstance(params, (str, bytes)):
                spec_args.extend(params)
            else:
                spec_args.append(params)
        # Remaining keys are treated as keyword arguments.  This allows concise
        # YAML/JSON configuration files.
        spec_kwargs.update(spec_dict)
        return target_fn(*spec_args, *args, **spec_kwargs, **kwargs)
    return spec


def _move_to_device(data: Any, device: torch.device) -> Any:
    """Recursively move a batch of data to the provided device."""

    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return type(data)({k: _move_to_device(v, device) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(_move_to_device(v, device) for v in data)
    return data


def _extract_loss(output: Any) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Extract a scalar loss value and optional logging dictionary."""

    if output is None:
        return None, {}
    if isinstance(output, Mapping):
        loss = output.get("loss")
        if loss is None and "losses" in output:
            losses = output["losses"]
            if isinstance(losses, Mapping):
                loss = sum(losses.values())
        log_values = {}
        for key, value in output.items():
            if isinstance(value, (int, float)):
                log_values[key] = float(value)
            elif torch.is_tensor(value) and value.dim() == 0:
                log_values[key] = float(value.detach())
            elif key == "losses" and isinstance(value, Mapping):
                log_values.update({f"loss/{k}": float(v) for k, v in value.items()})
        return loss, log_values
    if torch.is_tensor(output):
        return output, {"loss": float(output.detach())}
    if isinstance(output, Sequence) and output:
        maybe_loss = output[0]
        if torch.is_tensor(maybe_loss):
            return maybe_loss, {"loss": float(maybe_loss.detach())}
    raise TypeError(f"Unsupported output type for loss extraction: {type(output)!r}")


def _ensure_dataloader(
    loader_spec: Any,
    dataset_spec: Any,
    *,
    default_batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a dataloader from the provided specification."""

    if loader_spec is not None:
        loader = instantiate_spec(loader_spec)
        if not isinstance(loader, DataLoader):
            raise TypeError("The loader specification must resolve to a DataLoader instance")
        return loader
    dataset = instantiate_spec(dataset_spec)
    if dataset is None:
        raise ValueError("Neither dataloader nor dataset specification provided")
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset specification must resolve to a torch.utils.data.Dataset")
    return DataLoader(dataset, batch_size=default_batch_size, shuffle=shuffle)


def _serialize_config(cfg: Any) -> Any:
    """Convert a configuration object into a JSON/YAML friendly structure."""

    if isinstance(cfg, Mapping):
        return {k: _serialize_config(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_serialize_config(v) for v in cfg]
    if callable(cfg):
        return f"<callable {cfg.__module__}.{getattr(cfg, '__qualname__', cfg.__name__)}>"
    if isinstance(cfg, Path):
        return str(cfg)
    return cfg


def _get_learning_rate(optimizer: Optional[torch.optim.Optimizer]) -> float:
    if optimizer is None or not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------


def train(cfg: Mapping[str, Any]) -> TrainState:
    """Execute the end-to-end training loop.

    Parameters
    ----------
    cfg:
        A nested mapping containing everything that is required to build the
        model, data pipeline and optimisation strategy.  The function expects
        (but does not strictly require) the following keys:

        ``training``
            General optimisation settings such as number of epochs or logging
            cadence.  Recognised options include:

            ``max_epochs`` (int):
                Number of epochs to train for.
            ``device`` (str):
                Device identifier passed to :class:`torch.device`.
            ``work_dir`` (str):
                Output directory for checkpoints and logs.
            ``log_every`` (int):
                Interval (in optimisation steps) for emitting progress logs.
            ``val_interval`` (int):
                Run validation every *n* epochs (defaults to 1).
            ``metric`` (str):
                Name of the metric used to select the best checkpoint.
            ``metric_mode`` (str):
                Either ``"min"`` or ``"max"`` depending on the optimisation
                direction of the monitored metric.
            ``early_stop_patience`` (int):
                Stop training if the monitored metric has not improved for the
                specified number of validation runs.

        ``model``
            A specification understood by :func:`instantiate_spec` that returns
            the ``torch.nn.Module`` to train.

        ``optimizer`` / ``scheduler``
            Specifications for the optimiser and optional learning rate
            scheduler.

        ``data``
            Dataloader or dataset specifications under the keys
            ``train_dataloader``/``train_dataset`` and
            ``val_dataloader``/``val_dataset``.
    """

    ensure_logging()
    cfg = copy.deepcopy(cfg)
    training_cfg = dict(cfg.get("training", {}))
    data_cfg = dict(cfg.get("data", {}))

    device = torch.device(training_cfg.get("device", "cpu"))
    random_seed = training_cfg.get("seed", 42)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    try:
        import numpy as np

        np.random.seed(random_seed)
    except Exception:  # pragma: no cover - numpy might not be available
        logger.debug("NumPy not available - skipping NumPy seeding")

    work_dir = Path(training_cfg.get("work_dir", "./outputs")).expanduser().resolve()
    ckpt_dir = work_dir / "checkpoints"
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = work_dir / "config_snapshot.yaml"
    if not config_snapshot.exists():
        try:
            import yaml

            with config_snapshot.open("w", encoding="utf-8") as fp:
                yaml.safe_dump(_serialize_config(cfg), fp, allow_unicode=True, sort_keys=False)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("Unable to serialise configuration: %s", exc)

    callbacks = []
    for cb_spec in training_cfg.get("callbacks", []):
        callbacks.append(instantiate_spec(cb_spec))

    def _trigger(event_name: str, *cb_args: Any, **cb_kwargs: Any) -> None:
        for cb in callbacks:
            handler = getattr(cb, event_name, None)
            if handler is not None:
                handler(*cb_args, **cb_kwargs)

    # ------------------------------------------------------------------
    # Build data pipeline
    # ------------------------------------------------------------------
    default_batch_size = int(training_cfg.get("batch_size", 1))
    train_loader = _ensure_dataloader(
        data_cfg.get("train_dataloader"),
        data_cfg.get("train_dataset"),
        default_batch_size=default_batch_size,
        shuffle=bool(training_cfg.get("shuffle", True)),
    )
    val_loader = None
    if "val_dataloader" in data_cfg or "val_dataset" in data_cfg:
        val_loader = _ensure_dataloader(
            data_cfg.get("val_dataloader"),
            data_cfg.get("val_dataset"),
            default_batch_size=int(training_cfg.get("val_batch_size", default_batch_size)),
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # Build model and optimisation stack
    # ------------------------------------------------------------------
    model = instantiate_spec(cfg.get("model"))
    if not isinstance(model, nn.Module):
        raise TypeError("Model specification must resolve to a torch.nn.Module instance")
    model.to(device)

    optimizer_cfg = cfg.get("optimizer")
    optimizer: Optional[torch.optim.Optimizer] = None
    if optimizer_cfg is not None:
        optimizer = instantiate_spec(optimizer_cfg, model.parameters())
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Optimizer specification must resolve to a torch.optim.Optimizer")

    scheduler_cfg = cfg.get("scheduler")
    scheduler = None
    if scheduler_cfg is not None:
        scheduler = instantiate_spec(scheduler_cfg, optimizer)

    scaler = None
    use_amp = bool(training_cfg.get("amp", False)) and torch.cuda.is_available()
    if use_amp:
        scaler = GradScaler()

    scheduler_step = training_cfg.get("scheduler_step", "step")
    gradient_accum = max(int(training_cfg.get("grad_accum", 1)), 1)
    grad_clip = training_cfg.get("grad_clip", None)
    loss_scale = float(training_cfg.get("loss_scale", 1.0))

    # ------------------------------------------------------------------
    # Resume from checkpoint if requested
    # ------------------------------------------------------------------
    state = TrainState(work_dir=work_dir)
    start_epoch = 0
    resume_path = training_cfg.get("resume_from")
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = load_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0))
        state.global_step = int(checkpoint.get("global_step", 0))
        state.best_metric = float(checkpoint.get("best_metric", state.best_metric))
        state.best_metric_name = checkpoint.get("best_metric_name", state.best_metric_name)
        state.best_checkpoint_path = Path(checkpoint.get("best_checkpoint_path", ckpt_dir / "best.ckpt"))
        state.resume_checkpoint = resume_path
        torch.set_rng_state(checkpoint.get("rng_state", torch.get_rng_state()))
        if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        try:  # pragma: no cover - numpy optional
            import numpy as np

            if "numpy_rng_state" in checkpoint:
                np.random.set_state(tuple(checkpoint["numpy_rng_state"]))
        except Exception:
            pass
        random_state = checkpoint.get("random_state")
        if random_state is not None:
            random.setstate(tuple(random_state))
        logger.info("Resumed training from %s", resume_path)

    # ------------------------------------------------------------------
    # Book keeping for metrics and early stopping
    # ------------------------------------------------------------------
    metric_name = training_cfg.get("metric", "val_loss")
    metric_mode = training_cfg.get("metric_mode", "min")
    if metric_mode not in {"min", "max"}:
        raise ValueError("training.metric_mode must be either 'min' or 'max'")
    state.best_metric_name = metric_name
    state.best_metric = -math.inf if metric_mode == "max" else math.inf
    better = (lambda a, b: a > b) if metric_mode == "max" else (lambda a, b: a < b)
    early_stop_patience = training_cfg.get("early_stop_patience")
    epochs_since_improvement = 0

    max_epochs = int(training_cfg.get("max_epochs", 1))
    log_every = max(int(training_cfg.get("log_every", 10)), 1)
    val_interval = max(int(training_cfg.get("val_interval", 1)), 1)

    _trigger("on_train_start", state=state, config=cfg)

    model.train()
    optimizer_zero_grad = getattr(optimizer, "zero_grad", None)
    if optimizer_zero_grad is not None:
        optimizer_zero_grad(set_to_none=True)

    last_log_time = time.time()

    for epoch in range(start_epoch, max_epochs):
        state.epoch = epoch
        _trigger("on_epoch_start", state=state)
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            step_in_epoch = batch_idx + 1
            accumulate_step = (step_in_epoch - 1) % gradient_accum + 1
            if optimizer is not None and accumulate_step == 1 and optimizer_zero_grad is not None:
                optimizer_zero_grad(set_to_none=True)

            batch = _move_to_device(batch, device)
            round_outputs: List[Any] = []
            total_loss_tensor = None
            rounds = int(training_cfg.get("rounds", 3))
            for round_idx in range(rounds):
                output = None
                if hasattr(model, "training_round"):
                    output = model.training_round(batch, round_idx=round_idx)
                elif hasattr(model, "training_step"):
                    output = model.training_step(batch, round_idx=round_idx)
                elif round_idx == 0:
                    if hasattr(model, "forward"):
                        output = model(batch)
                    else:
                        raise AttributeError("Model does not provide a forward or training_step method")
                if output is None:
                    continue
                loss, _ = _extract_loss(output)
                if loss is None:
                    raise ValueError("Model output does not contain a loss value")
                if loss_scale != 1.0:
                    loss = loss * loss_scale
                round_outputs.append(output)
                total_loss_tensor = loss if total_loss_tensor is None else total_loss_tensor + loss

            if total_loss_tensor is None:
                raise RuntimeError("No loss was produced during the forward pass")

            loss_to_backward = total_loss_tensor / gradient_accum
            if scaler is not None:
                with autocast():
                    scaler.scale(loss_to_backward).backward()
                if accumulate_step == gradient_accum and optimizer is not None:
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss_to_backward.backward()
                if accumulate_step == gradient_accum:
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if optimizer is not None:
                        optimizer.step()

            if accumulate_step == gradient_accum:
                if optimizer is not None and optimizer_zero_grad is not None:
                    optimizer_zero_grad(set_to_none=True)
                if optimizer is not None and scheduler is not None and scheduler_step == "step":
                    scheduler.step()
                state.global_step += 1

            if accumulate_step == gradient_accum and state.global_step % log_every == 0:
                now = time.time()
                elapsed = now - last_log_time
                last_log_time = now
                log_record: Dict[str, Any] = {
                    "step": int(state.global_step),
                    "epoch": int(epoch),
                    "lr": _get_learning_rate(optimizer),
                    "elapsed": elapsed,
                }
                for output in round_outputs:
                    _, log_values = _extract_loss(output)
                    for key, value in log_values.items():
                        log_record.setdefault(key, 0.0)
                        log_record[key] += value
                if round_outputs:
                    for key in list(log_record.keys()):
                        if key.startswith("loss/"):
                            log_record[key] /= len(round_outputs)
                state.history.append(log_record)
                loss_value = log_record.get("loss")
                if loss_value is None:
                    logger.info(
                        "Step %s (epoch %s): metrics=%s",
                        log_record.get("step"),
                        log_record.get("epoch"),
                        {k: v for k, v in log_record.items() if k not in {"step", "epoch", "elapsed"}},
                    )
                else:
                    logger.info(
                        "Step %s (epoch %s): loss=%.4f lr=%.4g",
                        log_record.get("step"),
                        log_record.get("epoch"),
                        loss_value,
                        log_record.get("lr", 0.0),
                    )
                _trigger("on_log", log=log_record, state=state)

        # Epoch end -----------------------------------------------------
        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        if val_loader is not None and (epoch + 1) % val_interval == 0:
            eval_report = evaluate(cfg, model, val_loader, device=device)
            state.last_eval_report = eval_report
            metric_value = eval_report.metrics.get(metric_name)
            if metric_value is None:
                logger.warning(
                    "Validation report does not contain the monitored metric '%s'. Available metrics: %s",
                    metric_name,
                    sorted(eval_report.metrics.keys()),
                )
            else:
                improved = better(metric_value, state.best_metric)
                if improved:
                    state.best_metric = metric_value
                    epochs_since_improvement = 0
                    best_ckpt_path = ckpt_dir / "best.ckpt"
                    save_checkpoint(
                        best_ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch + 1,
                        global_step=state.global_step,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        best_checkpoint_path=best_ckpt_path,
                    )
                    state.best_checkpoint_path = best_ckpt_path
                    _trigger("on_checkpoint", state=state, path=best_ckpt_path, is_best=True)
                else:
                    epochs_since_improvement += 1
                logger.info("Validation (%s=%s)", metric_name, metric_value)
            _trigger("on_validation_end", state=state, report=eval_report)

            if early_stop_patience is not None and epochs_since_improvement >= early_stop_patience:
                state.stopped_early = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    epochs_since_improvement,
                )
                break

        latest_ckpt = ckpt_dir / "latest.ckpt"
        save_checkpoint(
            latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=state.global_step,
            metric_name=metric_name,
            metric_value=state.best_metric,
            best_checkpoint_path=state.best_checkpoint_path,
        )
        _trigger("on_checkpoint", state=state, path=latest_ckpt, is_best=False)
        _trigger("on_epoch_end", state=state)

        if state.stopped_early:
            break

    state.summary = {
        "best_metric_name": state.best_metric_name,
        "best_metric": state.best_metric,
        "best_checkpoint": str(state.best_checkpoint_path) if state.best_checkpoint_path else None,
        "epochs": state.epoch + 1,
        "global_step": state.global_step,
        "stopped_early": state.stopped_early,
    }
    _trigger("on_train_end", state=state)
    return state


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int,
    global_step: int,
    metric_name: str,
    metric_value: Optional[float],
    best_checkpoint_path: Optional[Path],
) -> None:
    """Persist the full training state to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": metric_value,
        "best_metric_name": metric_name,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "rng_state": torch.get_rng_state(),
        "random_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()
    try:  # pragma: no cover - numpy optional
        import numpy as np

        checkpoint["numpy_rng_state"] = np.random.get_state()
    except Exception:
        pass
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    *,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device, None] = None,
) -> Dict[str, Any]:
    """Load a checkpoint file and optionally restore the provided objects."""

    checkpoint = torch.load(path, map_location=map_location)
    if model is not None:
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            logger.warning("Model state mismatch when loading %s", path)
            if missing:
                logger.warning("Missing keys: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys: %s", unexpected)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


__all__ = [
    "TrainState",
    "train",
    "instantiate_spec",
    "save_checkpoint",
    "load_checkpoint",
]
