import json

import pandas as pd

from qcad import config
from qcad.report_ae import (augmentation_table, category_table, latex_augmentation_table, latex_category_table,
                            latex_summary_table, latex_tuning_table, summary_table, tuning_table)


def _metrics():
    rows = []
    for loss, shift in (("mse", 0.0), ("ssim", 0.1)):
        for seed in (0, 1):
            for i, cat in enumerate(config.CATEGORIES):
                for subset in ("val", "test"):
                    rows.append({"loss": loss, "seed": seed, "category": cat, "subset": subset,
                                 "category_type": config.category_type(cat),
                                 "image_auroc": 0.5 + 0.01 * i + shift * seed, "image_ap": 0.8,
                                 "pixel_auroc": 0.7 + shift, "au_pro": 0.5, "f1_mcc": 0.3})
    return pd.DataFrame(rows)


def test_summary_and_category_tables():
    metrics = _metrics()
    summary = summary_table(metrics)
    assert list(summary["loss"]) == ["mse", "ssim"]
    assert summary.loc[0, "image_auroc_all_std"] == 0.0 and summary.loc[1, "image_auroc_all_std"] > 0
    tex = latex_summary_table(summary)
    assert tex.startswith(r"\setlength") and "MSE &" in tex and "SSIM &" in tex
    table = category_table(metrics)
    assert list(table.index) == list(config.CATEGORIES)
    assert abs(table.loc["carpet", "ssim_image_auroc"] - 0.55) < 1e-9
    tex = latex_category_table(table)
    assert "dywan" in tex and r"$^{\dagger}$" in tex and "średnia: obiekty" in tex


def test_tuning_and_augmentation_tables():
    val = pd.DataFrame([{"combo": i, "loss": json.dumps(loss), "map_sigma": s, "score": json.dumps(r),
                         "image_auroc_all": 0.7 + 0.01 * i}
                        for i, (loss, s, r) in enumerate((l, s, r) for l in ("mse", "ssim") for s in (0.0, 2.0)
                                                         for r in ("max", "mean", "top"))])
    table = tuning_table(val)
    assert table.shape == (6, 2) and list(table.index[0]) == [0.0, "max"]
    assert r"\textbf{0,810}" in latex_tuning_table(table)
    aug = augmentation_table()
    assert aug.set_index("category").loc["transistor", ["flips", "rotation", "shift"]].tolist() == ["", "", ""]
    tex = latex_augmentation_table(aug)
    assert "±" not in tex and r"$\pm$180$^{\circ}$" in tex
