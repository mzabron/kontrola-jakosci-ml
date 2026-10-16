import numpy as np
import pytest

from qcad.evaluation import (au_pro, evaluate, image_auroc, per_defect_auroc, pixel_auroc,
                             pro_curve, select_threshold, threshold_metrics)


def _masks(n=40, size=64, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, size, size), dtype=bool)
    for i in range(n // 2):
        y, x = rng.integers(0, size - 12, 2)
        h, w = rng.integers(3, 12, 2)
        masks[i, y:y + h, x:x + w] = True
    return masks


def test_image_auroc_random_is_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 20000)
    assert image_auroc(rng.random(20000), labels) == pytest.approx(0.5, abs=0.01)


def test_image_auroc_oracle_and_inverted():
    labels = np.array([0, 0, 1, 1, 1])
    assert image_auroc(labels + 0.1 * np.arange(5), labels) == 1.0
    assert image_auroc(-labels.astype(float), labels) == 0.0


def test_image_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    s, y = rng.random(60), rng.integers(0, 2, 60)
    pos, neg = s[y == 1], s[y == 0]
    pairwise = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert image_auroc(s, y) == pytest.approx(pairwise)


def test_threshold_metrics_counts():
    scores = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    labels = np.array([0, 1, 0, 1, 1])
    m = threshold_metrics(scores, labels, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_flag_everything_has_high_f1_but_zero_mcc():
    labels = np.array([1] * 73 + [0] * 27)
    m = threshold_metrics(np.ones(100), labels, 0.5)
    assert m["f1"] == pytest.approx(2 * 0.73 / 1.73)
    assert m["mcc"] == 0.0 and m["balanced_accuracy"] == 0.5


def test_select_threshold_f1_separable():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    labels = np.array([0, 0, 0, 1, 1])
    thr = select_threshold(scores, labels, "f1")
    assert threshold_metrics(scores, labels, thr)["f1"] == 1.0


def test_select_threshold_percentile_uses_only_good():
    scores = np.concatenate([np.arange(100.0), [1e6]])
    labels = np.concatenate([np.zeros(100), [1]])
    assert select_threshold(scores, labels, "percentile") == pytest.approx(np.percentile(np.arange(100.0), 99))


def test_per_defect_auroc():
    scores = np.array([0.1, 0.2, 0.9, 0.05, 0.95])
    labels = np.array([0, 0, 1, 1, 1])
    types = np.array(["good", "good", "a", "b", "b"])
    out = per_defect_auroc(scores, labels, types)
    assert out["a"] == 1.0
    assert out["b"] == pytest.approx(0.5)


def test_pixel_metrics_oracle_is_perfect():
    masks = _masks()
    noise = np.random.default_rng(0).random(masks.shape) * 0.1
    maps = masks + noise
    assert pixel_auroc(maps, masks) == pytest.approx(1.0)
    assert au_pro(maps, masks) == pytest.approx(1.0, abs=1e-6)


def test_pixel_metrics_random():
    masks = _masks(n=60, size=64)
    maps = np.random.default_rng(3).random(masks.shape)
    assert pixel_auroc(maps, masks) == pytest.approx(0.5, abs=0.02)
    assert au_pro(maps, masks) == pytest.approx(0.15, abs=0.02)


def test_pro_curve_hand_computed():
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 2:4, 2:4] = True
    maps = np.zeros((1, 4, 4))
    maps[0, 0, 0] = 1.0
    maps[0, 2, 2] = 0.9
    fprs, pros = pro_curve(maps, mask)
    assert pros[1] == pytest.approx(0.5)
    assert pros[2] == pytest.approx(0.625)
    assert fprs[1] == fprs[2] == 0.0


def test_evaluate_without_maps_gives_nan_pixel_metrics():
    out = evaluate(np.array([0.1, 0.9]), np.array([0, 1]), thresholds={"f1": 0.5})
    assert out["image_auroc"] == 1.0
    assert np.isnan(out["pixel_auroc"]) and np.isnan(out["au_pro"])
    assert out["f1_f1"] == 1.0
