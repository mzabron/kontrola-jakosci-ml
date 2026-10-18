import json

import pandas as pd

from qcad import config
from qcad.runner import run_experiment

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
    scores = pd.read_csv(out / "scores.csv")
    assert set(scores["subset"]) == {"holdout", "val", "test"}
    info = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert info["environment"]["split_sha256"]
    assert (tmp_path / "summary.csv").exists()
