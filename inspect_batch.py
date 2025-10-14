#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_batch.py
用法示例：
    python inspect_batch.py \
      --data-root "/mnt/d/python/Repetition-trajectory prediction/mypro/av2-test.small/data_root" \
      --batch-size 2 --num-workers 0 --split train
"""

import os
import sys
import argparse
import inspect
import importlib
from typing import Any, Dict

def add_cwd_to_sys_path():
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "av2-test.small"),
                   help="数据根目录；也可用环境变量 DATA_ROOT 指定")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--split", default="train", help="尽力传给 Dataset；传不进去会自动忽略")
    return p.parse_args()

# ---------- Pretty print ----------
def _is_tg(obj):
    try:
        from torch_geometric.data import Batch, HeteroData
        return isinstance(obj, (Batch, HeteroData))
    except Exception:
        return False

def describe(obj, prefix=""):
    import torch
    from torch import Tensor

    if isinstance(obj, Tensor):
        print(f"{prefix}Tensor  shape={list(obj.shape)}  dtype={obj.dtype}  device={obj.device}")
        return
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{prefix}Dict(keys={keys})")
        for k, v in obj.items():
            print(f"{prefix}  ├─[{k}] -> ", end="")
            describe(v, prefix + "  │   ")
        return
    if isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__}(len={len(obj)})")
        for i, v in enumerate(obj[:10]):
            print(f"{prefix}  ├─[{i}] -> ", end="")
            describe(v, prefix + "  │   ")
        return
    if _is_tg(obj):
        print(f"{prefix}{obj.__class__.__name__}")
        try:
            if hasattr(obj, 'num_graphs'):
                print(f"{prefix}  ├─num_graphs={obj.num_graphs}")
            if hasattr(obj, 'keys'):
                for store_key in obj.keys():
                    store = obj[store_key]
                    fields = list(store.keys())
                    print(f"{prefix}  ├─[{store_key}] fields={fields}")
                    for f in fields:
                        val = store[f]
                        try:
                            import torch
                            if isinstance(val, torch.Tensor):
                                print(f"{prefix}  │    • {f}: shape={list(val.shape)} dtype={val.dtype} dev={val.device}")
                        except Exception:
                            pass
        except Exception as e:
            print(f"{prefix}  (summary error: {e})")
        return
    print(f"{prefix}{type(obj).__name__}: {obj}")

# ---------- Reflection helpers ----------
def safe_get_class(mod_name: str, cls_name: str):
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)
    except Exception:
        return None

def filter_kwargs_by_signature(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return {}

def try_datamodule(args):
    Av2DM = safe_get_class("datamodules.av2_datamodule", "Av2DataModule")
    if Av2DM is None:
        print("[DataModule] 未找到 datamodules.av2_datamodule.Av2DataModule")
        return None
    # 准备一份尽可能通用的 kwargs，只传签名里接受的字段
    raw = dict(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        split=args.split,               # 不一定存在，待过滤
        train_split=args.split,         # 以防万一
    )
    kw = filter_kwargs_by_signature(Av2DM.__init__, raw)
    print(f"[DataModule] 将使用参数：{kw}")
    try:
        dm = Av2DM(**kw)
        if hasattr(dm, "prepare_data"): dm.prepare_data()
        if hasattr(dm, "setup"): dm.setup("fit")
        # 优先找 train_dataloader / *_dataloader
        loader = None
        for name in ("train_dataloader", "fit_dataloader"):
            if hasattr(dm, name):
                loader = getattr(dm, name)()
                break
        if loader is None:
            # 兜底：找以 _dataloader 结尾的方法
            for name in dir(dm):
                if name.endswith("_dataloader") and callable(getattr(dm, name)):
                    loader = getattr(dm, name)()
                    break
        if loader is None:
            print("[DataModule] 未找到可调用的 *dataloader 方法")
            return None
        batch = next(iter(loader))
        print("\n[DataModule] Got one batch:")
        describe(batch)
        return batch
    except Exception as e:
        print(f"[DataModule] 失败：{e}")
        return None

def try_dataset(args):
    Av2DS = safe_get_class("datasets.av2_dataset", "Av2Dataset")
    if Av2DS is None:
        print("[Dataset] 未找到 datasets.av2_dataset.Av2Dataset")
        return None
    collate_fn = None
    try:
        mod_ds = importlib.import_module("datasets.av2_dataset")
        collate_fn = getattr(mod_ds, "collate_fn", None)
    except Exception:
        pass

    # 组装最常见的构造参数，按签名过滤
    raw = dict(
        data_root=args.data_root,
        split=args.split,
        mode=args.split,         # 有些项目用 mode
        partition=args.split,    # 有些项目用 partition
    )
    kw = filter_kwargs_by_signature(Av2DS.__init__, raw)
    print(f"[Dataset] 将使用参数：{kw}")

    try:
        ds = Av2DS(**kw)
    except Exception as e:
        print(f"[Dataset] 实例化失败：{e}")
        return None

    # 构造 DataLoader
    try:
        from torch.utils.data import DataLoader
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        batch = next(iter(loader))
        print("\n[Plain DataLoader] Got one batch:")
        describe(batch)
        return batch
    except Exception as e:
        print(f"[Dataset] DataLoader 失败：{e}")
        return None

def main():
    add_cwd_to_sys_path()
    # 先确认 torch 是否可用
    try:
        import torch  # noqa
    except Exception:
        print("❌ 未安装 torch：请先在当前环境安装 PyTorch 再运行本脚本。")
        print("例如（CUDA 12.1）：\n  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
              "--extra-index-url https://download.pytorch.org/whl/cu121 "
              "torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121")
        sys.exit(1)

    args = parse_args()

    # 先尝试 DataModule，失败再尝试 Dataset
    batch = try_datamodule(args)
    if batch is not None:
        return
    try_dataset(args)

if __name__ == "__main__":
    main()
