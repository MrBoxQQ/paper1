import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import pathlib

import numpy as np
import torch

from dataset import PcrdFeatureDataset
from torch.utils.data import DataLoader
from utils.model import Model


@dataclass(frozen=True)
class EvalConfig:
    cache_root: Path
    ckpt: Path
    n_class: Optional[int] = None
    n_domain: Optional[int] = None
    in_channel: int = 1
    batch_size: int = 64
    num_workers: int = 0
    preload_to_device: bool = False
    seed: int = 42


@torch.no_grad()
def evaluate(model: Model, loader, *, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_correct = 0
    d_correct = 0
    total = 0
    for x, y, d in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        d = d.to(device, non_blocking=True)
        out = model(x, lambd=0.0)
        y_pred = out.y_logits.argmax(dim=1)
        d_pred = out.d_logits.argmax(dim=1)
        y_correct += int((y_pred == y).sum().item())
        d_correct += int((d_pred == d).sum().item())
        total += int(y.numel())
    if total == 0:
        return {"y_acc": 0.0, "d_acc": 0.0}
    return {
        "y_acc": float(y_correct / total),
        "d_acc": float(d_correct / total),
    }


def _torch_load_ckpt(ckpt_path: Path, map_location: str | torch.device):
        return torch.load(ckpt_path, map_location=map_location, weights_only=True)


def _load_ckpt_cfg(ckpt_path: Path) -> Dict:
    ckpt = _torch_load_ckpt(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        return ckpt["cfg"]
    return {}


def _load_model(
    ckpt_path: Path,
    *,
    n_class: int,
    n_domain: int,
    in_channel: int,
    device: torch.device,
) -> Model:
    ckpt = _torch_load_ckpt(ckpt_path, map_location=device)
    model = Model(n_class=int(n_class), n_domain=int(n_domain), in_channels=int(in_channel)).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model


def run_eval(cfg: EvalConfig) -> Dict[str, float]:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = make_test_loader(
        cfg.cache_root,
        batch_size=int(cfg.batch_size),
        in_channel=int(cfg.in_channel),
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        preload_to_device=bool(cfg.preload_to_device),
        device=device,
    )

    ckpt_cfg = _load_ckpt_cfg(cfg.ckpt)
    n_class = int(cfg.n_class) if cfg.n_class is not None else int(ckpt_cfg.get("n_class", 0))
    n_domain = int(cfg.n_domain) if cfg.n_domain is not None else int(ckpt_cfg.get("n_domain", 0))
    if n_class <= 0 or n_domain <= 0:
        raise ValueError("n_class and n_domain must be provided either via args or checkpoint cfg")

    model = _load_model(
        cfg.ckpt,
        n_class=n_class,
        n_domain=n_domain,
        in_channel=int(cfg.in_channel),
        device=device,
    )

    metrics = evaluate(model, test_loader, device=device)
    return metrics


def make_test_loader(
    cache_root: Path,
    *,
    batch_size: int,
    in_channel: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    mmap: bool = True,
    preload_to_device: bool = False,
    device: Optional[torch.device | str] = None,
) -> DataLoader:
    if preload_to_device:
        num_workers = 0
        pin_memory = False
    test_set = PcrdFeatureDataset(
        cache_root,
        split="test",
        mmap=mmap,
        preload_to_device=preload_to_device,
        device=device,
        in_channel=in_channel,
    )
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", type=str)
    p.add_argument("--ckpt", type=str)
    p.add_argument("--n_class", type=int, default=None)
    p.add_argument("--n_domain", type=int, default=None)
    p.add_argument("--in_channel", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--preload_to_device", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    cfg = EvalConfig(
        cache_root=Path("/data/out"),
        ckpt=Path("/best_test.pth"),
        n_class=None if args.n_class is None else int(args.n_class),
        n_domain=None if args.n_domain is None else int(args.n_domain),
        in_channel=int(args.in_channel),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        preload_to_device=bool(args.preload_to_device),
        seed=int(args.seed),
    )
    metrics = run_eval(cfg)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
