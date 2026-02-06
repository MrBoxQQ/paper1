import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List
from tqdm import tqdm
import numpy as np
from utils.utils import cgs
import matplotlib.pyplot as plt

def plot_ifft_results(abs_e):
    cols=8
    m, n, num_carriers = np.asarray(abs_e).shape
    
    rows = math.ceil(m / cols)
    
    shifted_data = np.fft.ifftshift(abs_e, axes=2)
    time_domain_abs = np.abs(np.fft.ifft(shifted_data, axis=2))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2), 
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()
    
    x_axis = np.arange(num_carriers)
    
    for i in range(m):
        ax = axes_flat[i]
        channel_data = time_domain_abs[i, :, :]
        mean_data = np.mean(channel_data, axis=0)

        ax.plot(x_axis, channel_data.T, color='gray', alpha=0.05, linewidth=0.5)
        
        ax.plot(x_axis, mean_data, color='tab:red', linewidth=1.5)
        ax.set_yscale("log")
        ax.set_title(f"CH {i}", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)

    for j in range(m, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.supxlabel("Time Bin (IFFT Points)")
    fig.supylabel("Magnitude")
    fig.suptitle(f"Time Domain Impulse Response (m={m}, n={n})", fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



@dataclass(frozen=True)
class Cfg0:
    raw_root: Optional[Path] = None
    cache_root: Optional[Path] = None
    test_domain_id: int = 3
    val_ratio: float = 0.1
    seed: int = 42
    k_sc: int = 52
    t_win: int = 52
    energy_ratio: float = 0.95
    apply_ifftshift: bool = True
    gap: int = 60
    domain_map: Optional[Dict[str, int]] = None
    label_map: Optional[Dict[str, int]] = None



def _iter_raw_files(
    raw_root: Path, *, domain_map: Optional[Dict[str, int]], label_map: Optional[Dict[str, int]]
) -> Iterable[Tuple[int, int, Path]]:
    for domain_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        if domain_map is None:
            try:
                domain_id = int(domain_dir.name)
            except ValueError as e:
                raise ValueError(
                    f"domain directory name must be convertible to int (e.g., 0/1/2/3), "
                    f"or provide domain_map.json, but got {domain_dir.name!r}"
                ) from e
        else:
            if domain_dir.name not in domain_map:
                raise ValueError(f"domain_map does not include directory name {domain_dir.name!r}")
            domain_id = int(domain_map[domain_dir.name])

        for npy_path in sorted(domain_dir.glob("*.npy")):
            if label_map is None:
                try:
                    label_id = int(npy_path.stem)
                except ValueError as e:
                    raise ValueError(
                        f"label filename must be numeric (e.g., 0.npy..7.npy), "
                        f"or provide label_map.json, but got {npy_path.name!r}"
                    ) from e
            else:
                if npy_path.stem not in label_map:
                    raise ValueError(f"label_map does not include filename {npy_path.stem!r}")
                label_id = int(label_map[npy_path.stem])

            yield domain_id, label_id, npy_path


def _to_data_fixed(arr: np.ndarray, *, k_sc: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.shape[0] == k_sc:
            return arr
        if arr.shape[1] == k_sc:
            return arr.T
        raise ValueError(f"2D data must have one dimension equal to K={k_sc}, got shape={arr.shape}")
    # if arr.ndim == 3:
    #     if arr.shape[0] == k_sc:
    #         return arr.reshape(k_sc, -1)
    #     if arr.shape[1] == k_sc:
    #         return np.transpose(arr, (1, 0, 2)).reshape(k_sc, -1)
    #     if arr.shape[2] == k_sc:
    #         return np.transpose(arr, (2, 0, 1)).reshape(k_sc, -1)
    #     raise ValueError(f"3D data cannot resolve K={k_sc} dimension, got shape={arr.shape}")

    raise ValueError(f"Unsupported array dimensions: ndim={arr.ndim}, shape={arr.shape}")


def _open_split_memmaps(cache_root: Path, split: str, *, n: int, t_win: int, k_sc: int):
    split_dir = cache_root / split
    _ensure_dir_empty_create(split_dir)
    features_path = split_dir / "features.npy"
    targets_path = split_dir / "targets.npy"
    meta_path = split_dir / "meta.json"
    features_mm = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float64, shape=(n, 1, t_win, k_sc))
    targets_mm = np.lib.format.open_memmap(targets_path, mode="w+", dtype=np.int64, shape=(n, 2))
    return features_mm, targets_mm, meta_path

def _ensure_dir_empty_create(path: Path) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)



def rpv(cfg: Cfg0) -> Dict[str, int]:
    if not (0.0 <= float(cfg.val_ratio) < 1.0):
        raise ValueError(f"val_ratio must be in [0,1), got {cfg.val_ratio!r}")

    if cfg.raw_root is None or cfg.cache_root is None:
        raise ValueError("raw_root and cache_root must be provided")
    raw_root = Path(cfg.raw_root)
    cache_root = Path(cfg.cache_root)
    _ensure_dir_empty_create(cache_root)

    items = list(_iter_raw_files(raw_root, domain_map=cfg.domain_map, label_map=cfg.label_map))
    n_files = int(len(items))
    if n_files == 0:
        raise ValueError(f"no .npy files found under raw_root: {raw_root}")

    test_domain_id = int(cfg.test_domain_id)
    max_domain_id = max(int(d) for d, _l, _p in items)
    max_label_id = max(int(l) for _d, l, _p in items)
    n_domain = int(max_domain_id + 1)
    n_class = int(max_label_id + 1)

    if test_domain_id < 0 or test_domain_id >= n_domain:
        raise ValueError(
            f"test_domain_id={test_domain_id} is outside the parsed domain range [0,{n_domain})"
        )

    print("Collecting per-packet features...")
    samples_by_group: Dict[Tuple[int, int], List[Tuple[np.ndarray, int, int]]] = {}
    test_samples: List[Tuple[np.ndarray, int, int]] = []
    abs_e = []
    for domain_id, label_id, path in tqdm(items, desc="Collecting per-packet"):
        domain_id = int(domain_id)
        label_id = int(label_id)
        
        arr = np.load(path, allow_pickle=False)
        data_fixed = _to_data_fixed(arr, k_sc=int(cfg.k_sc))
        _, n_frames = data_fixed.shape
        if n_frames == 0:
            continue

        gadf_list, abs_e_l = cgs(
            data_fixed,
            energy_ratio=float(cfg.energy_ratio),
            apply_ifftshift=bool(cfg.apply_ifftshift),
        )
        abs_e.append(abs_e_l)

        for gadf in gadf_list:
            x = gadf[None, ...].astype(np.float64, copy=False)
            if domain_id == test_domain_id:
                test_samples.append((x, label_id, domain_id))
            else:
                group_key = (domain_id, label_id)
                if group_key not in samples_by_group:
                    samples_by_group[group_key] = []
                samples_by_group[group_key].append((x, label_id, domain_id))
    # plot_ifft_results(abs_e)
    print("Splitting training and validation sets...")
    train_samples: List[Tuple[np.ndarray, int, int]] = []
    val_samples: List[Tuple[np.ndarray, int, int]] = []

    for group_key, group_samples in samples_by_group.items():
        n_samples = len(group_samples)
        if n_samples == 0:
            continue

        n_val = int(np.floor(float(cfg.val_ratio) * n_samples))
        if n_val == 0:
            train_samples.extend(group_samples)
            continue

        if n_samples <= n_val + cfg.gap:
            train_samples.extend(group_samples)
            continue

        n_train = n_samples - n_val - cfg.gap
        if n_train <= 0:
            train_samples.extend(group_samples)
            continue

        train_samples.extend(group_samples[:n_train])
        val_samples.extend(group_samples[n_train + cfg.gap:])

    n_train = len(train_samples)
    n_val = len(val_samples)
    n_test = len(test_samples)
    
    print(f"Split complete: train={n_train}, val={n_val}, test={n_test}")

    train_results = train_samples
    val_results = val_samples
    test_results = test_samples

    print("Saving results...")
    
    train_f, train_t, train_meta_path = _open_split_memmaps(
        cache_root, "train", n=n_train, t_win=int(cfg.t_win), k_sc=int(cfg.k_sc)
    )
    val_f, val_t, val_meta_path = _open_split_memmaps(
        cache_root, "val", n=n_val, t_win=int(cfg.t_win), k_sc=int(cfg.k_sc)
    )
    test_f, test_t, test_meta_path = _open_split_memmaps(
        cache_root, "test", n=n_test, t_win=int(cfg.t_win), k_sc=int(cfg.k_sc)
    )
    
    for i, (x, label_id, domain_id) in enumerate(train_results):
        train_f[i] = x
        train_t[i] = np.asarray([label_id, domain_id], dtype=np.int64)
    
    for i, (x, label_id, domain_id) in enumerate(val_results):
        val_f[i] = x
        val_t[i] = np.asarray([label_id, domain_id], dtype=np.int64)

    for i, (x, label_id, domain_id) in enumerate(test_results):
        test_f[i] = x
        test_t[i] = np.asarray([label_id, domain_id], dtype=np.int64)

    train_f.flush()
    train_t.flush()
    val_f.flush()
    val_t.flush()
    test_f.flush()
    test_t.flush()

    domain_counts = np.zeros((n_domain,), dtype=np.int64)
    val_counts = np.zeros((n_domain,), dtype=np.int64)
    train_counts = np.zeros((n_domain,), dtype=np.int64)
    
    for _x, _label_id, domain_id in train_results:
        domain_counts[domain_id] += 1
        train_counts[domain_id] += 1

    for _x, _label_id, domain_id in val_results:
        domain_counts[domain_id] += 1
        val_counts[domain_id] += 1

    for _x, _label_id, domain_id in test_results:
        domain_counts[domain_id] += 1

    common_meta = {
        "n_class": int(n_class),
        "n_domain": int(n_domain),
        "domain_ids": sorted({int(d) for d, _l, _p in items}),
        "label_ids": sorted({int(l) for _d, l, _p in items}),
        "t_win": int(cfg.t_win),
        "k_sc": int(cfg.k_sc),
        "energy_ratio": float(cfg.energy_ratio),
        "apply_ifftshift": bool(cfg.apply_ifftshift),
        "raw_root": str(raw_root),
        "cache_root": str(cache_root),
        "split_method": "leave_one_domain_out_sequential_split_with_gap",
        "test_domain_id": int(test_domain_id),
        "val_ratio": float(cfg.val_ratio),
        "seed": int(cfg.seed),
        "gap_size": int(cfg.gap),
        "domain_counts": {str(i): int(domain_counts[i]) for i in range(n_domain)},
        "val_counts": {str(i): int(val_counts[i]) for i in range(n_domain)},
        "train_counts": {str(i): int(train_counts[i]) for i in range(n_domain)},
    }

    train_meta = dict(common_meta)
    train_meta.update({"split": "train", "n_samples": int(n_train), "n_files": n_files})
    train_meta_path.write_text(json.dumps(train_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    val_meta = dict(common_meta)
    val_meta.update({"split": "val", "n_samples": int(n_val), "n_files": n_files})
    val_meta_path.write_text(json.dumps(val_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    test_meta = dict(common_meta)
    test_meta.update({"split": "test", "n_samples": int(n_test), "n_files": n_files})
    test_meta_path.write_text(json.dumps(test_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"n_files": n_files, "n_train": int(n_train), "n_val": int(n_val), "n_test": int(n_test)}








def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", type=str, default=None)
    p.add_argument("--cache_root", type=str, default=None)
    p.add_argument("--test_domain_id", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k_sc", type=int, default=52)
    p.add_argument("--energy_ratio", type=float, default=0.999)
    p.add_argument("--no_ifftshift", action="store_true", default=False)
    p.add_argument("--domain_map", type=str, default=None, help="Optional JSON: {domain_dir_name: domain_id}")
    p.add_argument("--label_map", type=str, default=None, help="Optional JSON: {label_stem: label_id}")
    args = p.parse_args()

    domain_map = None if args.domain_map is None else json.loads(Path(args.domain_map).read_text(encoding="utf-8"))
    label_map = None if args.label_map is None else json.loads(Path(args.label_map).read_text(encoding="utf-8"))

    cfg = Cfg0(
        raw_root=None if args.raw_root is None else Path(args.raw_root),
        cache_root=None if args.cache_root is None else Path(args.cache_root),
        test_domain_id=int(args.test_domain_id),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        k_sc=int(args.k_sc),
        energy_ratio=float(args.energy_ratio),
        apply_ifftshift=not bool(args.no_ifftshift),
        domain_map=domain_map,
        label_map=label_map,
    )
    stats = rpv(cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
