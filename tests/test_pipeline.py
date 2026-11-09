import json

import pandas as pd

from qcad import config
from qcad.runner import cost_config, expand_fractions, is_cost_run, run_experiment, thread_limit

from .conftest import requires_data, requires_split


@requires_data
@requires_split
def test_random_method_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    out = run_experiment({"method": "random", "categories": ["toothbrush"], "seeds": [0]})
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["subset"]) == {"val", "test"}
    test = metrics[metrics["subset"] == "test"].iloc[0]
    assert 0.0 <= test["image_auroc"] <= 1.0
    assert 0.35 < test["pixel_auroc"] < 0.65
    assert test["model_size"] == 0
    scores = pd.read_csv(out / "scores.csv")
    assert set(scores["subset"]) == {"holdout", "val", "test"}
    info = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert info["environment"]["split_sha256"]
    assert (tmp_path / "summary.csv").exists()


def test_expand_fractions_and_cost_config():
    cfg = {"method": "shallow", "params": {"detector": "gmm"}, "seeds": [0, 1, 2],
           "train_fraction": [0.1, 0.25, 0.5], "tag": "gmm"}
    runs = expand_fractions(cfg)
    assert [r["train_fraction"] for r in runs] == [0.1, 0.25, 0.5]
    assert [r["tag"] for r in runs] == ["gmm_f010", "gmm_f025", "gmm_f050"]
    assert expand_fractions({"method": "random"}) == [{"method": "random"}]
    cost = cost_config(cfg)
    assert cost["seeds"] == [0] and cost["train_fraction"] == 1.0 and cost["tag"] == "gmm_cost"
    assert cost["threads"] == 1
    assert cost["params"] == {"detector": "gmm", "feature_cache": False, "n_jobs": 1}
    assert is_cost_run({"config": cost}) and not is_cost_run({"config": cfg})
    assert expand_fractions(cost) == [cost]


@requires_data
@requires_split
def test_train_fraction_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    out = run_experiment({"method": "random", "categories": ["toothbrush"], "seeds": [0],
                          "train_fraction": 0.1, "tag": "f010"})
    metrics = pd.read_csv(out / "metrics.csv")
    assert set(metrics["n_train"]) == {5}
    assert out.name.endswith("_random_f010")


def test_thread_limit():
    from threadpoolctl import threadpool_info

    with thread_limit(1):
        assert all(p["num_threads"] == 1 for p in threadpool_info())
