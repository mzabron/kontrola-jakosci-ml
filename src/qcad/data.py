from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config

INDEX_COLUMNS = [
    "id", "category", "category_type", "source_split", "defect_type",
    "label", "image_path", "mask_path",
]


def build_index(root: Path = config.DATA_ROOT, categories=config.CATEGORIES) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for category in categories:
        for source_split in ("train", "test"):
            split_dir = root / category / source_split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"missing directory: {split_dir}")
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                defect_type = defect_dir.name
                for img in sorted(defect_dir.glob("*.png")):
                    mask_path = ""
                    if defect_type != "good":
                        mask = root / category / "ground_truth" / defect_type / f"{img.stem}_mask.png"
                        if not mask.exists():
                            raise FileNotFoundError(f"missing mask: {img}")
                        mask_path = mask.relative_to(root).as_posix()
                    rows.append({
                        "id": f"{category}/{source_split}/{defect_type}/{img.stem}",
                        "category": category,
                        "category_type": config.category_type(category),
                        "source_split": source_split,
                        "defect_type": defect_type,
                        "label": int(defect_type != "good"),
                        "image_path": img.relative_to(root).as_posix(),
                        "mask_path": mask_path,
                    })
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def _resize_and_crop(img: Image.Image, resample, resize: int, crop: int) -> Image.Image:
    if img.width != img.height:
        raise ValueError(f"non-square image: {img.size}")
    img = img.resize((resize, resize), resample=resample)
    off = (resize - crop) // 2
    return img.crop((off, off, off + crop, off + crop))


def load_image(path: Path, resize: int = config.RESIZE, crop: int = config.CROP) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = _resize_and_crop(img, Image.Resampling.BILINEAR, resize, crop)
        return np.asarray(img, dtype=np.uint8).copy()


def load_mask(path: Path | None, resize: int = config.RESIZE, crop: int = config.CROP) -> np.ndarray:
    if path is None:
        return np.zeros((crop, crop), dtype=bool)
    with Image.open(path) as m:
        m = m.convert("L")
        m = _resize_and_crop(m, Image.Resampling.NEAREST, resize, crop)
        return np.asarray(m) > 127


@dataclass
class ImageSet:
    category: str
    subset: str
    ids: np.ndarray
    images: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    defect_types: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)


def _cache_path(category: str, resize: int, crop: int) -> Path:
    return config.CACHE_DIR / f"{category}_r{resize}_c{crop}.npz"


def load_category_arrays(category: str, index: pd.DataFrame, root: Path = config.DATA_ROOT,
                         resize: int = config.RESIZE, crop: int = config.CROP,
                         use_cache: bool = True) -> dict[str, np.ndarray]:
    rows = index[index["category"] == category]
    cache = _cache_path(category, resize, crop)
    if use_cache and cache.exists():
        data = np.load(cache, allow_pickle=False)
        if set(data["ids"]) == set(rows["id"]):
            return {k: data[k] for k in ("ids", "images", "masks")}
    root = Path(root)
    images = np.stack([load_image(root / p, resize, crop) for p in rows["image_path"]])
    masks = np.stack([load_mask(root / p if p else None, resize, crop) for p in rows["mask_path"]])
    arrays = {"ids": rows["id"].to_numpy(dtype=str), "images": images, "masks": masks}
    if use_cache:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **arrays)
    return arrays


def load_split_file(path: Path = config.SPLIT_FILE) -> pd.DataFrame:
    split = pd.read_csv(path, keep_default_na=False)
    split["label"] = split["label"].astype(int)
    split["train_holdout"] = split["train_holdout"].astype(bool)
    return split


def select_rows(split: pd.DataFrame, category: str, subset: str,
                train_fraction: float = 1.0) -> pd.DataFrame:
    rows = split[split["category"] == category]
    if subset == "train":
        rows = rows[(rows["subset"] == "train") & ~rows["train_holdout"]]
        if train_fraction < 1.0:
            n = max(1, int(round(train_fraction * len(rows))))
            rows = rows[rows["fewshot_rank"] < n]
    elif subset == "holdout":
        rows = rows[(rows["subset"] == "train") & rows["train_holdout"]]
    elif subset in ("val", "test"):
        rows = rows[rows["subset"] == subset]
    else:
        raise ValueError(f"unknown subset: {subset}")
    return rows


def load_subset(category: str, subset: str, split: pd.DataFrame | None = None,
                train_fraction: float = 1.0, root: Path = config.DATA_ROOT) -> ImageSet:
    split = load_split_file() if split is None else split
    rows = select_rows(split, category, subset, train_fraction)
    arrays = load_category_arrays(category, split, root)
    pos = {i: k for k, i in enumerate(arrays["ids"])}
    idx = np.array([pos[i] for i in rows["id"]], dtype=int)
    return ImageSet(
        category=category,
        subset=subset,
        ids=rows["id"].to_numpy(dtype=str),
        images=arrays["images"][idx],
        masks=arrays["masks"][idx],
        labels=rows["label"].to_numpy(dtype=int),
        defect_types=rows["defect_type"].to_numpy(dtype=str),
    )


def to_tensor(images: np.ndarray, imagenet_norm: bool = False):
    import torch

    x = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    if imagenet_norm:
        mean = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std
    return x
