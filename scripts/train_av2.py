from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from torch import nn
from torch.utils.data import DataLoader
import yaml

from datasets.av2_dataset import Av2Dataset, collate_fn
from train.train_loop import TrainState, instantiate_spec, train as train_loop

LOGGER = logging.getLogger("scripts.train_av2")


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Configuration root must be a mapping")
    return data


def _parse_override(value: str) -> Any:
    """Parse a command line override value using YAML semantics."""

    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_override(config: Dict[str, Any], override: str) -> None:
    """Apply a dotted key override (``foo.bar=baz``) to the configuration."""

    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected key=value format")
    key, raw_value = override.split("=", 1)
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = _parse_override(raw_value)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_single_loader(
    *,
    dataset: Av2Dataset,
    loader_cfg: Dict[str, Any],
    default_batch_size: int,
    shuffle_default: bool,
) -> DataLoader:
    """Create a torch DataLoader for the provided dataset."""

    batch_size = int(loader_cfg.get("batch_size", default_batch_size))
    shuffle = bool(loader_cfg.get("shuffle", shuffle_default))
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", False))
    drop_last = bool(loader_cfg.get("drop_last", False))
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0)) if num_workers > 0 else False
    prefetch_factor = loader_cfg.get("prefetch_factor")
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }
    if persistent_workers:
        loader_kwargs["persistent_workers"] = True
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def build_data_components(config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Instantiate the Argoverse 2 data pipeline for training and validation."""

    data_cfg = dict(config.get("data", {}))
    root = data_cfg.get("root")
    if not root:
        raise ValueError("'data.root' must be provided in the configuration")
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    dataset_params = dict(data_cfg.get("dataset", {}))
    LOGGER.info("Preparing Argoverse 2 dataset from %s", root_path)
    train_split = data_cfg.get("train_split", "train")
    if not (root_path / train_split).exists():
        raise FileNotFoundError(f"Training split '{train_split}' not found under {root_path}")
    train_dataset = Av2Dataset(data_root=root_path, split=train_split, logger=LOGGER, **dataset_params)
    LOGGER.info("Loaded %d training scenarios", len(train_dataset))

    training_cfg = dict(config.get("training", {}))
    default_batch_size = int(training_cfg.get("batch_size", 1))

    train_loader_cfg = dict(data_cfg.get("train_loader", {}))
    train_loader = _build_single_loader(
        dataset=train_dataset,
        loader_cfg=train_loader_cfg,
        default_batch_size=default_batch_size,
        shuffle_default=bool(training_cfg.get("shuffle", True)),
    )

    val_loader: Optional[DataLoader] = None
    val_split = data_cfg.get("val_split", "val")
    if val_split:
        if not (root_path / val_split).exists():
            LOGGER.warning("Validation split '%s' not found under %s; skipping validation", val_split, root_path)
        else:
            val_dataset = Av2Dataset(data_root=root_path, split=val_split, logger=LOGGER, **dataset_params)
            LOGGER.info("Loaded %d validation scenarios", len(val_dataset))
            val_loader_cfg = dict(data_cfg.get("val_loader", {}))
            val_loader = _build_single_loader(
                dataset=val_dataset,
                loader_cfg=val_loader_cfg,
                default_batch_size=int(training_cfg.get("val_batch_size", default_batch_size)),
                shuffle_default=False,
            )
    return train_loader, val_loader


def build_model(config: Dict[str, Any]) -> nn.Module:
    """Instantiate the model as described in the configuration."""

    model_spec = config.get("model")
    if model_spec is None:
        raise ValueError("Configuration must provide a 'model' section")
    model = instantiate_spec(model_spec)
    if not isinstance(model, nn.Module):
        raise TypeError("Model specification must resolve to a torch.nn.Module instance")
    config["model"] = model
    return model


def _stringify_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify_paths(v) for v in obj]
    return obj


def write_report(state: TrainState, work_dir: Path) -> Path:
    report = state.to_dict()
    report["status"] = "success"
    report_path = work_dir / "train_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(_stringify_paths(report), handle, indent=2, ensure_ascii=False)
    return report_path


def prepare_logging(level: str, work_dir: Optional[Path]) -> None:
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if work_dir is not None:
        log_file = work_dir / "train.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on the Argoverse 2 dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override configuration values using dot notation (e.g. training.max_epochs=10)",
    )
    parser.add_argument("--work-dir", type=Path, help="Optional directory for outputs and checkpoints")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> TrainState:
    args = parse_args(argv)
    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)

    training_cfg = config.setdefault("training", {})
    if args.work_dir is not None:
        training_cfg["work_dir"] = str(args.work_dir)
    work_dir = Path(training_cfg.get("work_dir", "./outputs/av2")).expanduser().resolve()
    _ensure_directory(work_dir)

    prepare_logging(args.log_level, work_dir)
    LOGGER.info("Using work directory %s", work_dir)

    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    try:
        train_loader, val_loader = build_data_components(config)
        config.setdefault("data", {})["train_dataloader"] = train_loader
        if val_loader is not None:
            config["data"]["val_dataloader"] = val_loader
        model = build_model(config)
        LOGGER.info("Model %s initialised with %d parameters", model.__class__.__name__, sum(p.numel() for p in model.parameters()))
    except Exception as exc:
        LOGGER.exception("Failed to prepare training components: %s", exc)
        raise

    state: Optional[TrainState] = None
    try:
        state = train_loop(config)
    except Exception as exc:
        LOGGER.exception("Training failed: %s", exc)
        failure_report = work_dir / "train_report.json"
        payload = {
            "status": "failed",
            "error": str(exc),
        }
        with failure_report.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        raise

    assert state is not None
    report_path = write_report(state, work_dir)
    if state.best_checkpoint_path:
        LOGGER.info("Best checkpoint stored at %s", state.best_checkpoint_path)
    LOGGER.info("Training complete. Report saved to %s", report_path)
    return state


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

