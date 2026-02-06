import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


@dataclass(frozen=True)
class PcrdCachePaths:
    root: Path
    split: str

    @property
    def split_dir(self) -> Path:
        return self.root / self.split

    @property
    def features(self) -> Path:
        return self.split_dir / "features.npy"

    @property
    def targets(self) -> Path:
        return self.split_dir / "targets.npy"

    @property
    def meta(self) -> Path:
        return self.split_dir / "meta.json"


class PcrdFeatureDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        *,
        split: str,
        in_channel: int = 2,  
        mmap: bool = True,
        preload_to_device: bool = False,
        device: Optional[torch.device | str] = None,
    ):
        self.paths = PcrdCachePaths(Path(cache_root), split=str(split))
        self.in_channel = int(in_channel)
        if self.in_channel not in (1, 2):
            raise ValueError(f"in_channel must be 1 or 2, got: {self.in_channel}")

        self._meta: Optional[Dict] = None
        if self.paths.meta.exists():
            self._meta = json.loads(self.paths.meta.read_text(encoding="utf-8"))

        self._targets_np = np.load(self.paths.targets)

        self._features = np.load(self.paths.features, mmap_mode="r" if (mmap and not preload_to_device) else None)

        self._x_dev: Optional[torch.Tensor] = None
        self._y_dev: Optional[torch.Tensor] = None
        self._d_dev: Optional[torch.Tensor] = None

        if self._features.ndim != 4:
            raise ValueError(f"features.npy must be 4D (N,2,52,52), got shape={self._features.shape}")
        if self._targets_np.ndim != 2 or self._targets_np.shape[1] != 2:
            raise ValueError(f"targets.npy must be 2D (N,2), got shape={self._targets_np.shape}")
        if self._features.shape[0] != self._targets_np.shape[0]:
            raise ValueError("features/targets have mismatched N")

        n_samples = None if self._meta is None else self._meta.get("n_samples")
        if n_samples is None:
            self._n = int(self._features.shape[0])
        else:
            self._n = int(n_samples)
            if self._n < 0 or self._n > int(self._features.shape[0]):
                raise ValueError(f"meta.json has invalid n_samples: {self._n}")

        if preload_to_device:
            dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x_np = np.asarray(self._features[: self._n], dtype=np.float32)

            if self.in_channel == 1:
                x_np = x_np[:, :1, :, :]

            y_np = np.asarray(self._targets_np[: self._n, 0], dtype=np.int64)
            d_np = np.asarray(self._targets_np[: self._n, 1], dtype=np.int64)
            self._x_dev = torch.from_numpy(x_np).to(dev, non_blocking=False)
            self._y_dev = torch.from_numpy(y_np).to(dev, non_blocking=False)
            self._d_dev = torch.from_numpy(d_np).to(dev, non_blocking=False)

            self._features = None

    @property
    def meta(self) -> Optional[Dict]:
        return self._meta

    @property
    def domains_np(self) -> np.ndarray:
        return np.asarray(self._targets_np[: self._n, 1], dtype=np.int64)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)

        if self._x_dev is not None:
            x = self._x_dev[idx]
            y = self._y_dev[idx]
            d = self._d_dev[idx]
            return x, y, d

        raw_data = self._features[idx]

        if self.in_channel == 1:
            raw_data = raw_data[:1] 

        x = torch.from_numpy(np.asarray(raw_data, dtype=np.float32).copy())
        y = torch.tensor(int(self._targets_np[idx, 0]), dtype=torch.long)
        d = torch.tensor(int(self._targets_np[idx, 1]), dtype=torch.long)
        return x, y, d


class RandomSubsetSampler(Sampler[int]):
    def __init__(self, data_source, fraction: float, *, generator=None, shuffle=True):
        assert 0 < fraction <= 1
        self.data_source = data_source
        self.fraction = fraction
        self.shuffle = shuffle
        self.generator = generator
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return int(np.ceil(len(self.data_source) * self.fraction))

    def __iter__(self):
        n = len(self.data_source)
        k = len(self)
        g = self.generator if self.generator is not None else torch.Generator()
        g = g.manual_seed(42 + self.epoch)

        if self.shuffle:
            perm = torch.randperm(n, generator=g).tolist()
        else:
            perm = list(range(n))
        return iter(perm[:k])


def make_dataloaders(
    cache_root: str | Path,
    *,
    batch_size: int,
    in_channel: int = 2,
    num_workers: int = 0,
    pin_memory: bool = True,
    mmap: bool = True,
    preload_to_device: bool = False,
    device: Optional[torch.device | str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if preload_to_device:
        num_workers = 0
        pin_memory = False
    train_set = PcrdFeatureDataset(
        cache_root, split="train", mmap=mmap, preload_to_device=preload_to_device, device=device, in_channel=in_channel
    )
    sampler = RandomSubsetSampler(train_set, fraction=0.5, shuffle = True)

    val_set = PcrdFeatureDataset(
        cache_root, split="val", mmap=mmap, preload_to_device=preload_to_device, device=device, in_channel=in_channel
    )
    
    test_set = PcrdFeatureDataset(
        cache_root, split="test", mmap=mmap, preload_to_device=preload_to_device, device=device, in_channel=in_channel
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler = sampler,
        # shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
