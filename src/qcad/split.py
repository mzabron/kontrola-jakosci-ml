from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import config


def _rng(seed: int, *names: str) -> np.random.Generator:
    key = zlib.crc32("/".join(names).encode("utf-8"))
    return np.random.default_rng([seed, key])


def make_split(index: pd.DataFrame, seed: int = config.SPLIT_SEED,
               val_fraction: float = config.VAL_FRACTION,
               holdout_fraction: float = config.HOLDOUT_FRACTION) -> pd.DataFrame:
    split = index.copy()
    split["subset"] = ""
    split["train_holdout"] = False
    split["fewshot_rank"] = -1

    for category, cat_rows in split.groupby("category", sort=False):
        test_rows = cat_rows[cat_rows["source_split"] == "test"]
        for defect_type, grp in test_rows.groupby("defect_type", sort=True):
            ids = grp.index.to_numpy()
            perm = _rng(seed, category, "test", defect_type).permutation(len(ids))
            n_val = int(round(val_fraction * len(ids)))
            if len(ids) >= 2:
                n_val = min(max(n_val, 1), len(ids) - 1)
            split.loc[ids[perm[:n_val]], "subset"] = "val"
            split.loc[ids[perm[n_val:]], "subset"] = "test"

        train_ids = cat_rows[cat_rows["source_split"] == "train"].index.to_numpy()
        perm = _rng(seed, category, "train").permutation(len(train_ids))
        n_hold = max(1, int(round(holdout_fraction * len(train_ids))))
        hold, fit = train_ids[perm[:n_hold]], train_ids[perm[n_hold:]]
        split.loc[train_ids, "subset"] = "train"
        split.loc[hold, "train_holdout"] = True
        split.loc[fit, "fewshot_rank"] = np.arange(len(fit))

    assert (split["subset"] != "").all()
    return split


def summarize_split(split: pd.DataFrame) -> pd.DataFrame:
    s = split.assign(
        part=np.where(split["subset"] == "train",
                      np.where(split["train_holdout"], "holdout", "train"),
                      split["subset"])
    )
    table = (s.groupby(["category", "part", "label"]).size()
               .unstack(["part", "label"], fill_value=0))
    out = pd.DataFrame(index=table.index)
    out["train"] = table.get(("train", 0), 0)
    out["holdout"] = table.get(("holdout", 0), 0)
    for part in ("val", "test"):
        out[f"{part}_good"] = table.get((part, 0), 0)
        out[f"{part}_defect"] = table.get((part, 1), 0)
    out = out.loc[[c for c in config.CATEGORIES if c in out.index]]
    out.loc["total"] = out.sum()
    return out.astype(int)
