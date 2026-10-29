import numpy as np
import pandas as pd
import pytest
import torch

from qcad import config
from qcad.methods import create_method
from qcad.methods.autoencoder import (AUGMENTATIONS, ConvAutoencoder, augment, patch_positions,
                                      random_crops, ssim_map)
from qcad.runner import run_experiment

from .conftest import requires_data, requires_split

FAST = {"max_epochs": 3, "samples_per_epoch": 64, "batch_size": 16, "patience": 1, "eval_batch": 8}


def _stripes(n, rng, size=224):
    x = np.arange(size)
    base = 128 + 60 * np.sin(2 * np.pi * x / 16)
    imgs = np.repeat(base[None, None, :, None], size, axis=1).repeat(3, axis=3)
    imgs = np.repeat(imgs, n, axis=0) + rng.normal(0, 5, (n, size, size, 1))
    return np.clip(imgs, 0, 255).astype(np.uint8)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


def test_augmentation_table_complete():
    assert set(AUGMENTATIONS) == set(config.CATEGORIES)
    assert not AUGMENTATIONS["metal_nut"]["hflip"] and not AUGMENTATIONS["metal_nut"]["vflip"]
    assert AUGMENTATIONS["transistor"]["shift"] == 0 and AUGMENTATIONS["transistor"]["rotate"] == 0


@pytest.mark.parametrize("size,channels", [(224, 3), (224, 1), (128, 1)])
def test_autoencoder_shapes(size, channels):
    model = ConvAutoencoder(size, channels, latent_dim=16)
    x = torch.rand(2, channels, size, size)
    assert model(x).shape == x.shape
    assert model.encoder(x).shape == (2, 16, 1, 1)


def test_ssim_map():
    x = torch.rand(2, 1, 64, 64)
    np.testing.assert_allclose(ssim_map(x, x).numpy(), 1.0, atol=1e-4)
    assert ssim_map(x, torch.rand(2, 1, 64, 64)).mean() < 0.5
    assert ssim_map(x, x).shape == x.shape


def test_augment_and_crops_keep_shape():
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(4, 3, 224, 224)
    for spec in AUGMENTATIONS.values():
        assert augment(x, spec, gen).shape == x.shape
    assert random_crops(x, 128, gen).shape == (4, 3, 128, 128)
    assert patch_positions(224, 128, 32) == [0, 32, 64, 96]
    assert patch_positions(224, 128, 64) == [0, 64, 96]


@pytest.mark.parametrize("loss,category", [("mse", "bottle"), ("ssim", "carpet")])
def test_fit_predict(loss, category):
    rng = np.random.default_rng(0)
    method = create_method("ae", loss=loss, latent_dim=16, category=category, device="cpu", **FAST)
    method.fit(_stripes(6, rng), holdout=_stripes(2, rng))
    pred = method.predict(_stripes(3, rng))
    assert pred.scores.shape == (3,) and pred.maps.shape == (3, 224, 224)
    assert np.isfinite(pred.maps).all()
    assert method.patch_size == (128 if category == "carpet" else None)
    assert method.model_size() == sum(p.numel() for p in method.model_.parameters())
    info = method.fit_info()
    assert 1 <= info["epochs_run"] <= 3 and info["best_epoch"] < info["epochs_run"]


def test_mse_warmup_excluded_from_early_stopping():
    rng = np.random.default_rng(2)
    method = create_method("ae", loss="ssim", latent_dim=16, device="cpu", mse_warmup=2,
                           **{**FAST, "max_epochs": 3, "patience": 5})
    method.fit(_stripes(4, rng), holdout=_stripes(2, rng))
    assert method.fit_info()["best_epoch"] == 2


def test_model_cache_reuses_training():
    rng = np.random.default_rng(1)
    train, hold, test = _stripes(6, rng), _stripes(2, rng), _stripes(2, rng)
    params = dict(loss="mse", latent_dim=16, category="bottle", device="cpu", **FAST)
    first = create_method("ae", **params).fit(train, holdout=hold).predict(test)
    assert len(list((config.CACHE_DIR / "ae").glob("*.pt"))) == 1
    again = create_method("ae", **{**params, "score": "mean"}).fit(train, holdout=hold).predict(test)
    np.testing.assert_allclose(first.maps, again.maps, rtol=1e-5, atol=1e-6)
    create_method("ae", **{**params, "seed": 1}).fit(train, holdout=hold)
    assert len(list((config.CACHE_DIR / "ae").glob("*.pt"))) == 2


def test_score_reductions():
    method = create_method("ae", score="top", score_top=0.5)
    maps = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
    assert method.reduce_scores(maps)[0] == 2.5
    method.score = "max"
    assert method.reduce_scores(maps)[0] == 3.0
    method.score = "mean"
    assert method.reduce_scores(maps)[0] == 1.5


@requires_data
@requires_split
def test_autoencoder_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    out = run_experiment({"method": "ae", "categories": ["toothbrush"], "seeds": [0],
                          "params": {"latent_dim": 16, "model_cache": False, **FAST}})
    metrics = pd.read_csv(out / "metrics.csv")
    test = metrics[metrics["subset"] == "test"].iloc[0]
    assert 0.0 <= test["pixel_auroc"] <= 1.0 and 0.0 <= test["au_pro"] <= 1.0
    assert test["epochs_run"] >= 1 and not np.isnan(test["holdout_loss"])
    assert test["model_size"] > 0
