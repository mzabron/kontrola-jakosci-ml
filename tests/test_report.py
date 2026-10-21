import json

import pandas as pd

from qcad import config
from qcad.report import (ablation_table, category_table, hyperparameter_table, latex_ablation_table,
                         latex_category_table, latex_hyperparameter_table)

PARAMS = {"ocsvm": {"pca_components": None, "nu": 0.01, "gamma_factor": 16.0},
          "gmm": {"pca_components": 200, "n_components": 3, "covariance_type": "diag"}}


def _metrics():
    rows = []
    for detector, features, val in (("ocsvm", "hog", 0.7), ("ocsvm", "color", 0.6), ("gmm", "lbp-glcm", 0.8)):
        params = json.dumps({"detector": detector, **PARAMS[detector]})
        for seed in (0, 1):
            for i, cat in enumerate(config.CATEGORIES):
                for subset in ("val", "test"):
                    rows.append({"detector": detector, "features": features, "val_select": val, "seed": seed,
                                 "category": cat, "category_type": config.category_type(cat), "subset": subset,
                                 "image_auroc": 0.5 + 0.01 * i + 0.001 * seed, "image_ap": 0.8, "f1_mcc": 0.3,
                                 "f1_balanced_accuracy": 0.6, "pct_mcc": 0.2, "params": params})
    return pd.DataFrame(rows)


def test_ablation_table_groups_and_order():
    table = ablation_table(_metrics())
    assert list(table["detector"]) == ["ocsvm", "ocsvm", "gmm"]
    assert list(table["features"]) == ["hog", "color", "lbp-glcm"]
    expected = pd.Series([0.5 + 0.01 * i for i in range(5)]).mean() + 0.0005
    assert abs(table.loc[0, "image_auroc_texture"] - expected) < 1e-9
    assert table.loc[0, "image_auroc_all_std"] > 0
    tex = latex_ablation_table(table)
    assert "OC-SVM & HOG" in tex and "kolor" in tex and "0,5" in tex


def test_category_table_uses_validation_choice():
    table, choice = category_table(_metrics())
    assert choice == {"gmm": "lbp-glcm", "ocsvm": "hog"}
    assert list(table.index) == list(config.CATEGORIES)
    tex = latex_category_table(table, choice)
    assert "butelka" in tex and "średnia: tekstury" in tex


def test_hyperparameter_table():
    table = hyperparameter_table(_metrics())
    assert list(table["detector"]) == ["ocsvm", "ocsvm", "gmm"]
    tex = latex_hyperparameter_table(table)
    assert "bez PCA" in tex and r"$\nu$ = 0,01, $g$ = 16" in tex
    assert "3 skł., kowariancja diagonalna" in tex and "0,800" in tex
