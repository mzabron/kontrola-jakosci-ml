import numpy as np
import pytest

from qcad import config
from qcad.data import load_subset, to_tensor

from .conftest import requires_data, requires_split


@requires_data
@requires_split
def test_loader_shapes_and_types():
    s = load_subset("bottle", "test")
    n = len(s)
    assert s.images.shape == (n, config.CROP, config.CROP, 3) and s.images.dtype == np.uint8
    assert s.masks.shape == (n, config.CROP, config.CROP) and s.masks.dtype == bool
    assert len(s.labels) == len(s.defect_types) == len(s.ids) == n


@requires_data
@requires_split
def test_grayscale_category_has_three_identical_channels():
    s = load_subset("screw", "train")
    assert s.images.shape[-1] == 3
    assert np.array_equal(s.images[..., 0], s.images[..., 1])


@requires_data
@requires_split
def test_masks_consistent_with_labels():
    s = load_subset("carpet", "val")
    assert not s.masks[s.labels == 0].any(), "good image has defect pixels"
    assert s.masks[s.labels == 1].any(axis=(1, 2)).mean() > 0.95


@requires_data
@requires_split
def test_train_is_good_only_and_excludes_holdout():
    train, hold = load_subset("toothbrush", "train"), load_subset("toothbrush", "holdout")
    assert train.labels.sum() == 0 and hold.labels.sum() == 0
    assert not set(train.ids) & set(hold.ids)


def test_to_tensor_ranges():
    imgs = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
    x = to_tensor(imgs)
    assert tuple(x.shape) == (2, 3, 4, 4) and float(x.max()) == 1.0
    xn = to_tensor(imgs, imagenet_norm=True)
    assert float(xn[0, 0, 0, 0]) == pytest.approx((1 - config.IMAGENET_MEAN[0]) / config.IMAGENET_STD[0], rel=1e-6)


def test_every_category_has_polish_name():
    assert set(config.CATEGORY_NAMES_PL) == set(config.CATEGORIES)
    assert config.category_name_pl("bottle") == "butelka"


@requires_split
def test_every_defect_type_has_polish_name():
    from qcad.data import load_split_file

    types = set(load_split_file()["defect_type"])
    assert types <= set(config.DEFECT_NAMES_PL)
    assert config.defect_name_pl("scratch") == "zarysowanie"
