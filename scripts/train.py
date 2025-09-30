from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from train import TrainState, train_loop

LOGGER = logging.getLogger("train.script")


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_override(config: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected format key=value")
    key, value = override.split("=", 1)
    target = config
    keys = key.split(".")
    for part in keys[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[keys[-1]] = _parse_value(value)


def main() -> TrainState:
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Configuration overrides in the form key=value. Nested keys use dot notation.",
    )
    parser.add_argument("--work-dir", type=Path, help="Override the training output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)

    if args.work_dir is not None:
        config.setdefault("training", {})["work_dir"] = str(args.work_dir)

    LOGGER.info("Starting training with configuration from %s", args.config)

    state = train(config)

    LOGGER.info("Training finished. Best %s=%s", state.best_metric_name, state.best_metric)
    if state.best_checkpoint_path:
        LOGGER.info("Best checkpoint saved to %s", state.best_checkpoint_path)

    return state


if __name__ == "__main__":
    main()