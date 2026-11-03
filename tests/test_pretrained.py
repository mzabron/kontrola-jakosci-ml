import numpy as np
import pandas as pd
import pytest
import torch

from qcad import config
from qcad.methods import create_method
from qcad.methods.pretrained import BACKBONES, cnn_features, fit_gaussians, mahalanobis, original_embedding
from qcad.runner import run_experiment

from .conftest import requires_data, requires_split


def _images(n, seed=0, size=64):
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 110, (1, size, size, 3))
    return np.clip(base + rng.normal(0, 4, (n, size, size, 3)), 0, 255).astype(np.uint8)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")


def test_gaussians_match_numpy():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(5, 20, 4)).astype(np.float32))
    q = torch.from_numpy(rng.normal(size=(5, 3, 4)).astype(np.float32))
    mean, inv = fit_gaussians(x, epsilon=0.01, chunk=2)
    dist = mahalanobis(q, mean, inv, chunk=2).numpy()
    for p in range(5):
        cov = np.cov(x[p].numpy(), rowvar=False) + 0.01 * np.eye(4)
        delta = q[p].numpy() - x[p].numpy().mean(axis=0)
        expected = np.sqrt(np.einsum("nd,de,ne->n", delta, np.linalg.inv(cov), delta))
        np.testing.assert_allclose(dist[p], expected, rtol=1e-4)


def test_padim_shapes_and_size():
    train, test = _images(6), _images(3, seed=1)
    test[0, 20:40, 20:40] = 255
    m = create_method("padim", backbone="resnet18", n_features=20, pretrained=False, seed=0).fit(train)
    pred = m.predict(test)
    assert pred.scores.shape == (3,) and pred.maps.shape == (3, 64, 64)
    assert np.all(np.isfinite(pred.scores)) and np.argmax(pred.scores) == 0
    p = 16 * 16
    assert m.model_size() == p * 20 + p * 20 * 20 + 20
    assert m.get_config()["embedding_dim"] == 448


def test_padim_seed_selects_features():
    train = _images(4)
    a = create_method("padim", backbone="resnet18", n_features=10, pretrained=False, seed=0).fit(train)
    b = create_method("padim", backbone="resnet18", n_features=10, pretrained=False, seed=1).fit(train)
    c = create_method("padim", backbone="resnet18", n_features=10, pretrained=False, seed=0).fit(train)
    assert torch.equal(a.index_, c.index_) and not torch.equal(a.index_, b.index_)


def test_padim_rejects_bad_params():
    with pytest.raises(ValueError):
        create_method("padim", backbone="vgg16")
    with pytest.raises(ValueError):
        create_method("padim", backbone="resnet18", n_features=1000, pretrained=False).fit(_images(3))


def test_patchcore_shapes_and_size():
    train, test = _images(4), _images(2, seed=1)
    m = create_method("patchcore", backbone="resnet18", coreset_ratio=0.25, pretrained=False).fit(train)
    pred = m.predict(test)
    assert pred.scores.shape == (2,) and pred.maps.shape == (2, 64, 64)
    assert np.all(np.isfinite(pred.scores))
    assert m.get_config()["n_patches"] == 4 * 8 * 8
    assert m.model_size() == int(4 * 8 * 8 * 0.25) * (128 + 256)


def test_patchcore_original_aggregation():
    train, test = _images(4), _images(2, seed=1)
    m = create_method("patchcore", backbone="resnet18", coreset_ratio=0.25, aggregation="original",
                      embed_dim=64, pretrained=False).fit(train)
    pred = m.predict(test)
    assert pred.maps.shape == (2, 64, 64) and np.all(np.isfinite(pred.scores))
    assert list(m.model_.memory_bank.shape) == [int(4 * 8 * 8 * 0.25), 64]
    assert m.get_config()["embed_dim"] == 64
    with pytest.raises(ValueError):
        create_method("patchcore", aggregation="sum")


def test_original_embedding_weights_layers_equally():
    feats = {"layer2": torch.ones(1, 4, 2, 2), "layer3": torch.zeros(1, 8, 1, 1)}
    emb = original_embedding(feats, ["layer2", "layer3"], 4)
    assert emb.shape == (1, 4, 2, 2)
    assert torch.allclose(emb[0, :, 0, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_cnn_block_in_shallow():
    feats = cnn_features(_images(3), backbone="resnet18", pretrained=False)
    assert feats.shape == (3, 128 + 256) and feats.dtype == np.float32
    m = create_method("shallow", features=["cnn"], detector="mahalanobis", pca_components=2,
                      cnn_backbone="resnet18", cnn_pretrained=False, n_jobs=1).fit(_images(6))
    assert m.predict(_images(2, seed=1)).scores.shape == (2,)
    assert m.get_config()["block_dims"] == {"cnn": [384, 2]}
    assert set(BACKBONES) == {"resnet18", "wide_resnet50_2"}


@requires_data
@requires_split
def test_padim_bottle_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    out = run_experiment({"method": "padim", "params": {"backbone": "resnet18"},
                          "categories": ["bottle"], "seeds": [0], "tag": "test"})
    metrics = pd.read_csv(out / "metrics.csv")
    test = metrics[metrics["subset"] == "test"].iloc[0]
    assert test["image_auroc"] > 0.95 and test["pixel_auroc"] > 0.9
