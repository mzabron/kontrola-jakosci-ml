import numpy as np
import pandas as pd
import pytest
import yaml

from qcad import config
from qcad.methods import create_method
from qcad.methods.shallow import DETECTORS, extract_block
from qcad.tuning import expand_grid, tune

from .conftest import requires_data, requires_split


def _stripes(n, rng, size=224):
    x = np.arange(size)
    base = 128 + 60 * np.sin(2 * np.pi * x / 16)
    imgs = np.repeat(base[None, None, :, None], size, axis=1).repeat(3, axis=3)
    imgs = np.repeat(imgs, n, axis=0) + rng.normal(0, 5, (n, size, size, 1))
    return np.clip(imgs, 0, 255).astype(np.uint8)


def _defective(images, rng):
    out = images.copy()
    for img in out:
        y, x = rng.integers(20, 150, 2)
        img[y:y + 50, x:x + 50] = rng.integers(0, 256, (50, 50, 3))
    return out


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


@pytest.mark.parametrize("name,dim", [("hog", 6084), ("lbp", 448), ("glcm", 512), ("color", 768)])
def test_feature_shapes(name, dim):
    imgs = _stripes(3, np.random.default_rng(0))
    feats = extract_block(imgs, name, n_jobs=1)
    assert feats.shape == (3, dim)
    assert np.isfinite(feats).all()


def test_feature_disk_cache():
    imgs = _stripes(2, np.random.default_rng(0))
    first = extract_block(imgs, "lbp", n_jobs=1)
    assert len(list((config.CACHE_DIR / "features").glob("lbp_*.npy"))) == 1
    np.testing.assert_array_equal(first, extract_block(imgs, "lbp", n_jobs=1))
    np.testing.assert_array_equal(first, extract_block(imgs, "lbp", n_jobs=1, use_cache=False))
    extract_block(imgs, "color", n_jobs=1, use_cache=False)
    assert not list((config.CACHE_DIR / "features").glob("color_*.npy"))


@pytest.mark.parametrize("detector", DETECTORS)
def test_defects_score_higher(detector):
    rng = np.random.default_rng(1)
    train, good = _stripes(40, rng), _stripes(10, rng)
    bad = _defective(_stripes(10, rng), rng)
    method = create_method("shallow", detector=detector, pca_components=10, n_jobs=1)
    method.fit(train)
    scores = method.predict(np.concatenate([good, bad])).scores
    assert scores.shape == (20,)
    assert scores[10:].mean() > scores[:10].mean()


def test_pca_capped_by_samples():
    imgs = _stripes(5, np.random.default_rng(2))
    method = create_method("shallow", features=["hog", "color"], detector="mahalanobis",
                           pca_components=50, n_jobs=1).fit(imgs)
    assert all(out <= 4 for _, out in method.get_config()["block_dims"].values())


def test_model_size():
    imgs = _stripes(10, np.random.default_rng(4))
    method = create_method("shallow", features=["color"], detector="mahalanobis",
                           pca_components=5, n_jobs=1).fit(imgs)
    assert method.model_size() == 2 * 768 + 5 * 768 + 768 + 1 + 5 + 5 * 5
    for detector in DETECTORS:
        assert create_method("shallow", detector=detector, pca_components=5,
                             n_jobs=1).fit(imgs).model_size() > 0


def test_seeded_detector_is_deterministic():
    rng = np.random.default_rng(3)
    train, test = _stripes(30, rng), _stripes(5, rng)
    a = create_method("shallow", detector="iforest", seed=7, n_jobs=1).fit(train).predict(test).scores
    b = create_method("shallow", detector="iforest", seed=7, n_jobs=1).fit(train).predict(test).scores
    np.testing.assert_array_equal(a, b)


def test_unknown_options_rejected():
    with pytest.raises(ValueError):
        create_method("shallow", features=["sift"])
    with pytest.raises(ValueError):
        create_method("shallow", detector="knn")


def test_expand_grid():
    combos = expand_grid({"a": [1, 2], "b": [[0], [0, 1]]})
    assert combos == [{"a": 1, "b": [0]}, {"a": 1, "b": [0, 1]}, {"a": 2, "b": [0]}, {"a": 2, "b": [0, 1]}]


@requires_data
@requires_split
def test_tuning_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", config.PROJECT_ROOT / "cache")
    out, finals = tune({"method": "shallow", "params": {"detector": "mahalanobis"},
                        "grid": {"features": [["color"], ["lbp"]], "pca_components": [5, 10]},
                        "select_per": ["features"], "categories": ["toothbrush"], "seeds": [0], "tag": "t"})
    grid = pd.read_csv(out / "val_grid.csv")
    assert len(grid) == 4 and grid["image_auroc"].between(0, 1).all()
    assert {f["tag"] for f in finals} == {"t_color", "t_lbp"}
    best = yaml.safe_load((out / "best_t_color.yaml").read_text(encoding="utf-8"))
    assert best["params"]["features"] == ["color"] and best["seeds"] == [0]
