"""PyTorch Lightning training entrypoint for Argoverse V2 models.

This script provides a command line interface that wires together:

1. Configuration parsing with optional command line overrides.
2. Initialisation of the model and data module.
3. Trainer configuration with common callbacks such as checkpointing and
   learning-rate logging.
4. Launching the Lightning training loop with reproducibility guarantees.

The script expects model definitions that are compatible with Lightning's
``LightningModule`` API.  They can be specified declaratively via YAML config
files using the existing ``instantiate_spec`` helper from ``train.train_loop``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Union

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, LightningLoggerBase

from datamodules.av2_datamodule import Av2DataModule
from train.train_loop import instantiate_spec

LOGGER = logging.getLogger("train")


def _setup_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, MutableMapping):
        raise TypeError("Configuration root must be a mapping")
    return dict(data)


def _parse_override(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_override(config: MutableMapping[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected key=value format")
    key, raw_value = override.split("=", 1)
    target: MutableMapping[str, Any] = config
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = target.get(part)
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            target[part] = next_value
        target = next_value  # type: ignore[assignment]
    target[parts[-1]] = _parse_override(raw_value)


def _parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_devices(value: str) -> Union[str, int, List[int]]:
    if value.lower() == "auto":
        return "auto"
    if "," in value:
        return [int(item) for item in value.split(",") if item]
    return int(value)


def build_datamodule(args: argparse.Namespace, config: Mapping[str, Any]) -> Av2DataModule:
    data_cfg = dict(config.get("data_module", {}))
    data_root = args.data_root or data_cfg.pop("data_root", None)
    if not data_root:
        raise ValueError("The dataset root directory must be provided via --data-root or config")
    data_root_path = Path(data_root).expanduser().resolve()
    if not data_root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root_path}")
    dataset_cfg = data_cfg.pop("dataset", {})
    train_batch_size = args.batch_size or data_cfg.pop("train_batch_size", 32)
    val_batch_size = args.val_batch_size or data_cfg.pop("val_batch_size", train_batch_size)
    num_workers = args.num_workers if args.num_workers is not None else data_cfg.pop("num_workers", 8)
    pin_memory = args.pin_memory if args.pin_memory is not None else data_cfg.pop("pin_memory", True)
    shuffle = data_cfg.pop("shuffle", True)
    return Av2DataModule(
        data_root=str(data_root_path),
        dataset=dataset_cfg,
        train_batch_size=train_batch_size,
        test_batch_size=val_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        logger=LOGGER,
    )


def build_model(config: Mapping[str, Any]) -> pl.LightningModule:
    model_spec = config.get("model")
    if model_spec is None:
        raise ValueError("Configuration must define a 'model' section")
    model = instantiate_spec(model_spec)
    if not isinstance(model, pl.LightningModule):
        raise TypeError("Model specification must resolve to a LightningModule instance")
    return model


def _prepare_callbacks(raw_callbacks: Iterable[Any]) -> List[Callback]:
    callbacks: List[Callback] = []
    if isinstance(raw_callbacks, Mapping):
        iterable: Iterable[Any] = raw_callbacks.values()
    elif isinstance(raw_callbacks, (str, bytes)):
        iterable = [raw_callbacks]
    else:
        iterable = raw_callbacks
    for item in iterable:
        if isinstance(item, Callback):
            callbacks.append(item)
            continue
        instantiated = instantiate_spec(item)
        if not isinstance(instantiated, Callback):
            raise TypeError(
                "Callback specifications must resolve to pytorch_lightning.Callback instances",
            )
        callbacks.append(instantiated)
    return callbacks


def build_trainer(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[pl.Trainer, ModelCheckpoint]:
    trainer_cfg = dict(config.get("trainer", {}))
    max_epochs = args.max_epochs if args.max_epochs is not None else trainer_cfg.pop("max_epochs", 64)
    accelerator = args.accelerator or trainer_cfg.pop("accelerator", "auto")
    devices = _resolve_devices(args.devices) if args.devices is not None else trainer_cfg.pop("devices", "auto")
    precision = args.precision or trainer_cfg.pop("precision", 32)

    checkpoint_cb = ModelCheckpoint(
        monitor="val/minFDE",
        mode="min",
        save_top_k=5,
        filename="epoch{epoch:02d}-minFDE{val_minFDE:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks_cfg = trainer_cfg.pop("callbacks", [])
    callbacks = _prepare_callbacks(callbacks_cfg)
    callbacks.extend([checkpoint_cb, lr_monitor])

    logger_cfg = trainer_cfg.pop("logger", None)
    if logger_cfg is None:
        logger = CSVLogger(save_dir="lightning_logs", name=args.run_name)
    else:
        logger = instantiate_spec(logger_cfg)
    if not isinstance(logger, LightningLoggerBase):
        raise TypeError("Logger specification must resolve to a LightningLoggerBase instance")

    trainer_kwargs: Dict[str, Any] = {
        "max_epochs": max_epochs,
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "logger": logger,
        "callbacks": callbacks,
        "default_root_dir": str(args.default_root_dir),
    }
    gradient_clip_val = trainer_cfg.pop("gradient_clip_val", None)
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    trainer_kwargs.update(trainer_cfg)
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer, checkpoint_cb


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Argoverse V2 models with PyTorch Lightning")
    parser.add_argument("--config", type=Path, help="Path to a YAML configuration file with model/trainer settings")
    parser.add_argument("--override", action="append", default=[], help="Override config values, e.g. model.args.lr=1e-4")
    parser.add_argument("--data-root", type=Path, help="Root directory of the Argoverse V2 dataset")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of dataloader worker processes")
    parser.add_argument("--pin-memory", type=_parse_bool, default=None, help="Whether to enable pin_memory")
    parser.add_argument("--devices", default=None, help="Device specification passed to Lightning Trainer (e.g. 1, 0,1,2 or auto)")
    parser.add_argument("--accelerator", default=None, help="Accelerator type (cpu, gpu, auto, etc.)")
    parser.add_argument("--precision", default=None, help="Numerical precision for the Trainer (32, 16-mixed, bf16-mixed, ...)")
    parser.add_argument("--max-epochs", type=int, default=None, help="Maximum number of training epochs (default 64)")
    parser.add_argument("--run-name", default="av2", help="Name of the training run for logger naming")
    parser.add_argument("--default-root-dir", type=Path, default=Path("lightning_logs"), help="Root directory for trainer outputs")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed for reproducibility")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    _setup_logging()
    args = parse_args(argv)

    config = _load_yaml(args.config)
    for override in args.override:
        _apply_override(config, override)

    pl.seed_everything(args.seed, workers=True)

    data_module = build_datamodule(args, config)
    model = build_model(config)
    trainer, checkpoint_cb = build_trainer(args, config)

    LOGGER.info("Starting training with max_epochs=%s", trainer.max_epochs)
    trainer.fit(model=model, datamodule=data_module)
    if checkpoint_cb.best_model_path:
        LOGGER.info("Best checkpoint saved to %s", checkpoint_cb.best_model_path)
    loggers = trainer.loggers if isinstance(trainer.loggers, list) else [trainer.logger] if trainer.logger else []
    log_dirs = [getattr(logger, "log_dir", None) for logger in loggers if getattr(logger, "log_dir", None)]
    if log_dirs:
        LOGGER.info("Training completed. Logs available at: %s", ", ".join(log_dirs))
    else:
        LOGGER.info("Training completed.")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()