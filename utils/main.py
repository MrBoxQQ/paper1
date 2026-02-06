import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from utils.utils import cgs


def _ensure_dir_empty_create(path: Path) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _to_data_fixed(arr: np.ndarray, *, k_sc: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] != k_sc:
            raise ValueError(f"1D data must have length K={k_sc}, got shape={arr.shape}")
        return arr.reshape(k_sc, 1)
    if arr.ndim == 2:
        if arr.shape[0] == k_sc:
            return arr
        if arr.shape[1] == k_sc:
            return arr.T
        raise ValueError(f"2D data must have one dimension equal to K={k_sc}, got shape={arr.shape}")
    raise ValueError(f"Unsupported array dimensions: ndim={arr.ndim}, shape={arr.shape}")


def _open_test_memmaps(cache_root: Path, *, n: int, t_win: int, k_sc: int):
    test_dir = cache_root / "test"
    _ensure_dir_empty_create(test_dir)
    features_path = test_dir / "features.npy"
    targets_path = test_dir / "targets.npy"
    meta_path = test_dir / "meta.json"
    features_mm = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float64, shape=(n, 1, t_win, k_sc))
    targets_mm = np.lib.format.open_memmap(targets_path, mode="w+", dtype=np.int64, shape=(n, 2))
    return features_mm, targets_mm, meta_path


@dataclass(frozen=True)
class Cfg1:
    csi_path: Optional[Path] = None
    labels_json: Optional[Path] = None
    cache_root: Optional[Path] = None
    k_sc: int = 52
    t_win: int = 52
    energy_ratio: float = 0.95
    apply_ifftshift: bool = True


def rpv1(cfg: Cfg1) -> Dict[str, int]:
    if cfg.csi_path is None or cfg.labels_json is None or cfg.cache_root is None:
        raise ValueError("csi_path, labels_json, and cache_root must be provided")

    csi_path = Path(cfg.csi_path)
    labels_json = Path(cfg.labels_json)
    cache_root = Path(cfg.cache_root)

    _ensure_dir_empty_create(cache_root)

    arr = np.load(csi_path, allow_pickle=False)
    data_fixed = _to_data_fixed(arr, k_sc=int(cfg.k_sc))
    _, n_frames = data_fixed.shape
    if n_frames == 0:
        raise ValueError("input sequence has 0 frames")

    labels_raw = json.loads(labels_json.read_text(encoding="utf-8"))
    if isinstance(labels_raw, dict) and "labels" in labels_raw:
        labels = labels_raw["labels"]
    else:
        labels = labels_raw
    if not isinstance(labels, list):
        raise ValueError("labels_json must be a JSON list or a dict with 'labels'")

    gadf_stack, _abs_e = cgs(
        data_fixed,
        energy_ratio=float(cfg.energy_ratio),
        apply_ifftshift=bool(cfg.apply_ifftshift),
    )

    n_samples = int(gadf_stack.shape[0])
    if gadf_stack.shape[1] != int(cfg.t_win) or gadf_stack.shape[2] != int(cfg.k_sc):
        raise ValueError(
            f"gadf shape {(gadf_stack.shape[1], gadf_stack.shape[2])} "
            f"does not match (t_win, k_sc)=({cfg.t_win}, {cfg.k_sc})"
        )
    if len(labels) != n_samples:
        raise ValueError(f"labels length must match samples ({n_samples}), got {len(labels)}")

    test_f, test_t, test_meta_path = _open_test_memmaps(
        cache_root, n=n_samples, t_win=int(cfg.t_win), k_sc=int(cfg.k_sc)
    )

    domain_id = 1 # just work in tarin, not used in eval
    for i, gadf in enumerate(gadf_stack):
        label_id = int(labels[i])
        test_f[i] = gadf[None, ...].astype(np.float64, copy=False)
        test_t[i] = np.asarray([label_id, domain_id], dtype=np.int64)

    test_f.flush()
    test_t.flush()

    label_ids = sorted({int(x) for x in labels})
    n_class = int(max(label_ids) + 1) if label_ids else 0
    n_domain = int(domain_id + 1)
    common_meta = {
        "n_class": int(n_class),
        "n_domain": int(n_domain),
        "domain_ids": [int(domain_id)],
        "label_ids": label_ids,
        "t_win": int(cfg.t_win),
        "k_sc": int(cfg.k_sc),
        "energy_ratio": float(cfg.energy_ratio),
        "apply_ifftshift": bool(cfg.apply_ifftshift),
        "csi_path": str(csi_path),
        "cache_root": str(cache_root),
        "split_method": "single_sequence_labels_json_test_only",
        "split": "test",
        "n_samples": int(n_samples),
    }
    test_meta_path.write_text(json.dumps(common_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"n_samples": int(n_samples)}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csi_path", type=str, default=None)
    p.add_argument("--labels_json", type=str, default=None)
    p.add_argument("--cache_root", type=str, default=None)
    p.add_argument("--k_sc", type=int, default=52)
    p.add_argument("--t_win", type=int, default=52)
    p.add_argument("--energy_ratio", type=float, default=0.995)
    p.add_argument("--no_ifftshift", action="store_true", default=False)
    args = p.parse_args()

    cfg = Cfg1(
        csi_path=None if args.csi_path is None else Path(args.csi_path),
        labels_json=None if args.labels_json is None else Path(args.labels_json),
        cache_root=None if args.cache_root is None else Path(args.cache_root),
        k_sc=int(args.k_sc),
        t_win=int(args.t_win),
        energy_ratio=float(args.energy_ratio),
        apply_ifftshift=not bool(args.no_ifftshift),
    )
    stats = rpv1(cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
