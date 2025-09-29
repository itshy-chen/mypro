# -*- coding: utf-8 -*-
'''verify_av2_realdata.py

适配你的 Av2DataModule（见 datamodules/av2_datamodule.py）的本地验证脚本：
- 按给定 split(train/val/test) 实例化 Av2DataModule
- 取一个 batch，做形状/掩码/时间戳等契约检查（键名使用常见候选，尽量自适应）
- 用一个极小的自适配 MLP 跑一次 forward + backward，验证数值稳定
- （可选）如果有 utils.geometry 的 to_agent_frame/to_global_frame，则做一次坐标往返小检验

运行示例：
  python verify_av2_realdata.py --data-root /data/av2 --split train --batch 2
  python verify_av2_realdata.py --data-root /data/av2 --split val --batch 2 \
      --dataset-cfg "{\"history_len\":50,\"future_len\":60}'''

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------
# 打印与断言
# ---------------------------
def log(msg: str):
    print(f"[verify] {msg}")

def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)

# ---------------------------
# 导入你的 Av2DataModule
# ---------------------------
def import_av2_dm():
    """
    固定按你的文件与类名导入：
      from datamodules.av2_datamodule import Av2DataModule
    """
    from datamodules.av2_datamodule import Av2DataModule
    return Av2DataModule

# ---------------------------
# 取一个 DataLoader（按 split）
# ---------------------------
def get_dataloader(dm, split: str):
    split = split.lower()
    if split == "train":
        return dm.train_dataloader()
    elif split == "val":
        return dm.val_dataloader()
    elif split == "test":
        return dm.test_dataloader()
    else:
        raise ValueError(f"未知 split: {split}")

# ---------------------------
# 从 batch 里取张量/键名自适应
# ---------------------------
def _pick(batch: Any, keys: List[str]) -> Optional[Any]:
    if isinstance(batch, dict):
        for k in keys:
            if k in batch:
                return batch[k]
    return None

def _to_tensor(x) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.as_tensor(x)
    except Exception:
        return None

# ---------------------------
# 批次契约 + 一步训练
# ---------------------------
def check_batch_and_one_step_train(batch: Any):
    """
    期望（尽量自适应你的 collate_fn 输出）：
      - 历史：hist / history / past  -> (B, Th, Fin)
      - 未来：future / fut / target / y -> (B, Tf, Fo)
      - 掩码：hist_mask / past_mask, fut_mask / future_mask / target_mask / y_mask
      - 时间戳（可选）：timestamps
    """
    torch.manual_seed(123)

    # 主要键名候选（可以按你项目实际再补充）
    hist = _pick(batch, ["hist", "history", "past"])
    fut  = _pick(batch, ["future", "fut", "target", "y"])
    hist = _to_tensor(hist)
    fut  = _to_tensor(fut)

    assert_true(hist is not None and fut is not None,
                f"未在 batch 中找到历史/未来张量；实际 keys={list(batch.keys()) if isinstance(batch, dict) else type(batch)}")

    assert_true(hist.ndim == 3, f"hist 需要 (B,Th,Fin)，当前 {tuple(hist.shape)}")
    assert_true(fut.ndim  == 3, f"future 需要 (B,Tf,Fo)，当前 {tuple(fut.shape)}")

    B, Th, Fin = hist.shape
    B2, Tf, Fo = fut.shape
    assert_true(B == B2, f"batch 维度不一致：hist B={B}, future B={B2}")
    assert_true(Fin > 0 and Fo > 0, "特征维不可为 0")

    # 掩码（可选）
    mask_h = _to_tensor(_pick(batch, ["hist_mask", "history_mask", "past_mask"]))
    mask_f = _to_tensor(_pick(batch, ["fut_mask", "future_mask", "target_mask", "y_mask"]))
    if mask_h is not None:
        assert_true(mask_h.shape == (B, Th), f"hist_mask 形状应为 (B,Th)，实际 {tuple(mask_h.shape)}")
        assert_true(mask_h.any(dim=1).all(), "hist_mask 每个样本至少应有一个 True")
    if mask_f is not None:
        assert_true(mask_f.shape == (B, Tf), f"fut_mask 形状应为 (B,Tf)，实际 {tuple(mask_f.shape)}")
        assert_true(mask_f.any(dim=1).all(), "fut_mask 每个样本至少应有一个 True")

    # 时间戳（可选检查）
    ts = _to_tensor(_pick(batch, ["timestamps", "ts"]))
    if ts is not None and ts.ndim == 2:  # (B, T?)
        # 允许非严格递增（不同目标/采样策略可能导致相等）
        assert_true(torch.all(ts[:, 1:] >= ts[:, :-1]), "时间戳未保持非递减")

    # 极小自适配模型（只用于数值通路验证，不代表你的真实模型）
    class Tiny(torch.nn.Module):
        def __init__(self, Fin, Fo, Tf):
            super().__init__()
            self.inp = torch.nn.Linear(Fin, 64)
            self.act = torch.nn.SiLU()
            self.out = torch.nn.Linear(64, Fo)
            self.Tf  = Tf
        def forward(self, x):  # x: (B,Th,Fin)
            B, T, F = x.shape
            z = self.out(self.act(self.inp(x.reshape(B*T, F)))).reshape(B, T, -1)
            if T != self.Tf:
                # 用线性插值把时间维对齐到 Tf
                z = torch.nn.functional.interpolate(
                    z.permute(0, 2, 1), size=self.Tf,
                    mode="linear", align_corners=False
                ).permute(0, 2, 1)
            return z

    model = Tiny(Fin, Fo, Tf)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pred = model(hist)
    assert_true(pred.shape == (B, Tf, Fo), f"预测形状不匹配：{tuple(pred.shape)} vs {(B, Tf, Fo)}")

    loss = ((pred - fut) ** 2).mean()
    assert_true(torch.isfinite(loss).item(), "loss 出现 NaN/Inf")
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert_true(torch.isfinite(p.grad).all().item(), "梯度出现 NaN/Inf")
    opt.step()

    log(f"batch 检查通过 & 一步训练 OK | loss={loss.item():.6f} | hist={tuple(hist.shape)} | fut={tuple(fut.shape)}")

# ---------------------------
# 坐标往返（如果你实现了 utils.geometry）
# ---------------------------
def try_geometry_roundtrip():
    try:
        from utils.geometry import to_agent_frame, to_global_frame
    except Exception:
        log("跳过坐标往返：未找到 utils.geometry 的 to_agent_frame/to_global_frame")
        return
    pts = torch.randn(16, 2) * 50
    origin = torch.tensor([10.0, -5.0])
    heading = torch.tensor(0.75)  # rad
    local = to_agent_frame(pts, origin, heading)
    back  = to_global_frame(local, origin, heading)
    err = (back - pts).norm(dim=-1).max().item()
    assert_true(err < 1e-4, f"坐标往返最大误差过大：{err}")
    log(f"坐标往返 OK，max_err={err:.2e}")

# ---------------------------
# 主流程
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="真实数据根目录")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0, help="建议验证时设 0，避免多进程文件句柄问题")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--dataset-cfg", type=str, default="{}", help='传给 Av2Dataset 的 dict，JSON 字符串，如 {"history_len":50}')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    assert_true(data_root.exists(), f"数据根目录不存在：{data_root}")

    # 解析 dataset 配置
    try:
        dataset_cfg: Dict[str, Any] = json.loads(args.dataset_cfg) if args.dataset_cfg else {}
        assert isinstance(dataset_cfg, dict)
    except Exception as e:
        raise SystemExit(f"--dataset-cfg 需要是 JSON 字符串，解析失败：{e}")

    log(f"data_root = {data_root}")
    log(f"split     = {args.split}")
    log(f"batch     = {args.batch}")
    log(f"dataset   = {dataset_cfg}")

    # 导入并实例化 DataModule
    Av2DM = import_av2_dm()
    dm = Av2DM(
        data_root=str(data_root),
        dataset=dataset_cfg,
        train_batch_size=args.batch,
        test_batch_size=args.batch,
        shuffle=False,                  # 验证建议关闭随机
        num_workers=args.num_workers,   # 验证建议 0
        pin_memory=args.pin_memory,
        test=(args.split == "test"),
        logger=None,
    )

    # setup：fit 用于 train/val，test 用于 test
    try:
        if args.split in ("train", "val"):
            dm.setup(stage="fit")
        else:
            dm.setup(stage="test")
    except Exception as e:
        raise SystemExit(f"DataModule.setup 失败：{e}")

    # 取 DataLoader
    try:
        dl = get_dataloader(dm, args.split)
    except Exception as e:
        raise SystemExit(f"获取 {args.split}_dataloader 失败：{e}")

    # 拉一个 batch
    try:
        batch = next(iter(dl))
    except StopIteration:
        raise SystemExit("DataLoader 返回空（没有样本）")
    except Exception as e:
        raise SystemExit(f"从 DataLoader 取 batch 失败：{e}")

    # 批次契约 + 一步训练
    check_batch_and_one_step_train(batch)

    # 坐标往返（若有）
    try_geometry_roundtrip()

    log("✅ 验证完成：真实数据最小通路正常。")

if __name__ == "__main__":
 main()
