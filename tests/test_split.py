import numpy as np
import pandas as pd
import pytest

from qcad import config
from qcad.data import build_index, load_split_file, select_rows
from qcad.split import make_split

from .conftest import requires_data, requires_split


@pytest.fixture(scope="module")
def split():
    return load_split_file()


@requires_split
def test_every_image_assigned_once(split):
    assert split["id"].is_unique
    assert len(split) == 5354
    assert set(split["subset"]) == {"train", "val", "test"}
    assert (split.loc[split["source_split"] == "train", "subset"] == "train").all()
    assert split.loc[split["source_split"] == "test", "subset"].isin(["val", "test"]).all()


@requires_split
def test_train_contains_only_good(split):
    assert (split.loc[split["subset"] == "train", "label"] == 0).all()
    assert not split.loc[split["subset"] != "train", "train_holdout"].any()


@requires_split
def test_val_fraction_and_stratification(split):
    test_folder = split[split["source_split"] == "test"]
    frac = (test_folder["subset"] == "val").mean()
    assert frac == pytest.approx(config.VAL_FRACTION, abs=0.01)
    for (cat, dt), grp in test_folder.groupby(["category", "defect_type"]):
        if len(grp) >= 2:
            assert set(grp["subset"]) == {"val", "test"}, f"{cat}/{dt} missing val or test"


@requires_split
def test_holdout_and_fewshot_ranks(split):
    for cat in config.CATEGORIES:
        train = split[(split["category"] == cat) & (split["subset"] == "train")]
        fit = train[~train["train_holdout"]]
        assert train["train_holdout"].sum() >= 1
        assert sorted(fit["fewshot_rank"]) == list(range(len(fit)))
        assert (train.loc[train["train_holdout"], "fewshot_rank"] == -1).all()


@requires_split
def test_fewshot_subsets_are_nested(split):
    prev = set()
    for frac in (0.1, 0.25, 0.5, 1.0):
        ids = set(select_rows(split, "bottle", "train", frac)["id"])
        assert prev <= ids
        prev = ids
    assert len(prev) == len(select_rows(split, "bottle", "train"))


@requires_data
@requires_split
def test_split_is_reproducible(split):
    regenerated = make_split(build_index(), seed=config.SPLIT_SEED)
    cols = ["id", "subset", "train_holdout", "fewshot_rank"]
    pd.testing.assert_frame_equal(
        regenerated[cols].reset_index(drop=True),
        split[cols].reset_index(drop=True), check_dtype=False)


@requires_data
def test_different_seed_gives_different_split():
    index = build_index(categories=("bottle",))
    a, b = make_split(index, seed=1), make_split(index, seed=2)
    assert not np.array_equal(a["subset"].to_numpy(), b["subset"].to_numpy())
