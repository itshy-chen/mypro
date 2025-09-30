from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch
from torch.utils.data import DataLoader, Dataset

from train import EvalReport, evaluate, instantiate_spec, load_checkpoint

LOGGER = logging.getLogger("eval.script")


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def build_dataloader(
    data_cfg: Dict[str, Any],
    split: str,
    *,
    default_batch_size: int = 1,
) -> DataLoader:
    loader_keys = [
        f"{split}_dataloader",
        f"{split}_loader",
        f"{split}_dataloder",
    ]
    dataset_keys = [
        f"{split}_dataset",
        f"{split}_data",
    ]
    loader_spec = None
    for key in loader_keys:
        if key in data_cfg:
            loader_spec = data_cfg[key]
            break
    if loader_spec is not None:
        loader = instantiate_spec(loader_spec)
        if not isinstance(loader, DataLoader):
            raise TypeError(f"Loader specification for split '{split}' must resolve to a DataLoader")
        return loader

    dataset_spec = None
    for key in dataset_keys:
        if key in data_cfg:
            dataset_spec = data_cfg[key]
            break
    dataset = instantiate_spec(dataset_spec) if dataset_spec is not None else None
    if dataset is None:
        raise ValueError(f"No dataloader or dataset configuration found for split '{split}'")
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset specification must resolve to a torch Dataset")
    return DataLoader(dataset, batch_size=default_batch_size, shuffle=False)


def main() -> Optional[EvalReport]:
    parser = argparse.ArgumentParser(description="Evaluation entrypoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint containing model weights")
    parser.add_argument("--config", type=Path, help="Configuration file. Defaults to config snapshot next to checkpoint")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate (default: val)")
    parser.add_argument("--device", default="cpu", help="Torch device for evaluation")
    parser.add_argument("--output", type=Path, help="Optional path to store the evaluation report as JSON")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    config_path = args.config
    if config_path is None:
        snapshot = args.checkpoint.parent / "config_snapshot.yaml"
        if snapshot.exists():
            config_path = snapshot
        else:
            raise FileNotFoundError("Configuration file not provided and snapshot could not be located")

    cfg = load_config(config_path)
    data_cfg = dict(cfg.get("data", {}))
    training_cfg = dict(cfg.get("training", {}))

    model = instantiate_spec(cfg.get("model"))
    device = torch.device(args.device)
    model.to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)

    default_bs = int(training_cfg.get("val_batch_size", training_cfg.get("batch_size", 1)))
    dataloader = build_dataloader(data_cfg, args.split, default_batch_size=default_bs)

    report = evaluate(cfg, model, dataloader, device=device)
    LOGGER.info("Evaluation metrics: %s", report.metrics)

    if args.output is not None:
        payload = report.to_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        LOGGER.info("Saved evaluation report to %s", args.output)

    return report


if __name__ == "__main__":
    main()

