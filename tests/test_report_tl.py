import pandas as pd

from qcad import config
from qcad.report_tl import (category_table, cross_table, latex_category_table, latex_cross_table,
                            latex_summary_table, run_variant, summary_table)


def _metrics():
    rows = []
    for variant, shift in (("padim_r18", 0.0), ("padim_wr50", 0.05), ("patchcore_concat", 0.08),
                           ("patchcore_original", 0.1)):
        for seed in (0, 1):
            for i, cat in enumerate(config.CATEGORIES):
                for subset in ("val", "test"):
                    rows.append({"variant": variant, "seed": seed, "category": cat, "subset": subset,
                                 "category_type": config.category_type(cat),
                                 "image_auroc": 0.8 + 0.01 * i + shift + 0.001 * seed, "image_ap": 0.9,
                                 "pixel_auroc": 0.9 + shift, "au_pro": 0.8 + shift, "f1_mcc": 0.6,
                                 "model_size": 1000})
    return pd.DataFrame(rows)


def test_run_variant():
    assert run_variant({"config": {"method": "padim", "params": {"backbone": "wide_resnet50_2"}}}) == "padim_wr50"
    assert run_variant({"config": {"method": "patchcore", "params": {}}}) == "patchcore_concat"
    assert run_variant({"config": {"method": "patchcore", "params": {"aggregation": "original"}}}) == \
        "patchcore_original"
    assert run_variant({"config": {"method": "ae", "params": {}}}) is None


def test_summary_and_category_tables():
    metrics = _metrics()
    summary = summary_table(metrics)
    assert list(summary["variant"]) == ["padim_r18", "padim_wr50", "patchcore_concat", "patchcore_original"]
    assert summary.loc[0, "image_auroc_all_std"] > 0
    tex = latex_summary_table(summary)
    assert "PaDiM R18 &" in tex and "PatchCore (anomalib) &" in tex and r"\textbf{" in tex
    table = category_table(metrics)
    assert list(table.index) == list(config.CATEGORIES)
    assert "patchcore_concat_image_auroc" not in table.columns
    tex = latex_category_table(table)
    assert "dywan" in tex and "średnia: tekstury" in tex and r"\multicolumn{3}{c}{AU-PRO}" in tex


def test_cross_table():
    rows = []
    for detector in ("ocsvm", "mahalanobis"):
        for features, base, val in (("hog", 0.7, 0.6), ("lbp-glcm", 0.8, 0.8), ("cnn", 0.9, 0.9)):
            for i, cat in enumerate(config.CATEGORIES):
                for subset in ("val", "test"):
                    rows.append({"detector": detector, "features": features, "seed": 0, "category": cat,
                                 "subset": subset, "category_type": config.category_type(cat),
                                 "image_auroc": base, "val_select": val})
    table = cross_table(pd.DataFrame(rows))
    assert list(table["detector"]) == ["ocsvm", "mahalanobis"]
    assert set(table["hand_features"]) == {"lbp-glcm"}
    assert abs(table.loc[0, "gain_all"] - 0.1) < 1e-9
    tex = latex_cross_table(table)
    assert "LBP + GLCM" in tex and "+0,100" in tex and "Mahalanobis" in tex
