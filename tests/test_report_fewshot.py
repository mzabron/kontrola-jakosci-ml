import pandas as pd

from qcad import config
from qcad.report_fewshot import (cost_table, fraction_table, latex_cost_table, latex_fraction_table,
                                 latex_group_table, method_key, train_sizes)


def _metrics():
    rows = []
    for method, base, seeds in (("patchcore_original", 0.9, (0, 1)), ("ae_mse", 0.7, (0, 1, 2)),
                                ("mahalanobis_lbp-glcm", 0.6, (0,))):
        for fraction in (0.1, 0.5, 1.0):
            for seed in seeds if fraction == 1.0 else seeds[:1]:
                for i, cat in enumerate(config.CATEGORIES):
                    for subset in ("val", "test"):
                        rows.append({"method": method, "fraction": fraction, "seed": seed, "category": cat,
                                     "subset": subset, "category_type": config.category_type(cat),
                                     "image_auroc": base + 0.1 * fraction - 0.05 * seed, "au_pro": base - 0.1,
                                     "n_train": round((50 + i) * fraction),
                                     "fit_time_s": 0.05 if seed == 0 else 100.0 + seed,
                                     "test_time_per_image_ms": 2.0, "model_size": 12345.0})
    return pd.DataFrame(rows)


def test_method_key():
    assert method_key({"config": {"method": "shallow",
                                  "params": {"detector": "gmm", "features": ["lbp", "glcm"]}}}) == "gmm_lbp-glcm"
    assert method_key({"config": {"method": "ae", "params": {"loss": "ssim"}}}) == "ae_ssim"
    assert method_key({"config": {"method": "padim", "params": {"backbone": "resnet18"}}}) == "padim_r18"


def test_fraction_table_uses_fraction_seeds():
    table = fraction_table(_metrics())
    assert list(table.index) == ["patchcore_original", "ae_mse", "mahalanobis_lbp-glcm"]
    assert abs(table.loc["ae_mse", 1.0] - 0.8) < 1e-9
    assert table.loc["ae_mse", "n_seeds"] == 1
    sizes = train_sizes(_metrics())
    tex = latex_fraction_table(table, sizes)
    assert "PatchCore &" in tex and "Mahalanobis (LBP + GLCM)" in tex and r"10\,\%" in tex
    assert r"\textbf{" in tex and "$-$0,090" in tex and "śr." in tex
    groups = {g: fraction_table(_metrics(), "image_auroc", g) for g in ("texture", "object", "all")}
    tex = latex_group_table(groups)
    assert "Tekstury" in tex and "Obiekty" in tex


def test_cost_table_skips_cached_seed():
    metrics = _metrics()
    table = cost_table(metrics, pd.DataFrame())
    ae = table.set_index("method").loc["ae_mse"]
    assert ae["source"] == "main" and ae["n_seeds"] == 2
    assert abs(ae["fit_time_s"] - 101.5) < 1e-9
    cost = metrics[(metrics["method"] == "mahalanobis_lbp-glcm") & (metrics["fraction"] == 1.0)].copy()
    cost["fit_time_s"] = 5.3
    table = cost_table(metrics, cost)
    row = table.set_index("method").loc["mahalanobis_lbp-glcm"]
    assert row["source"] == "cost" and row["device"] == "CPU (1 wątek)"
    tex = latex_cost_table(table)
    assert r"$1{,}2 \cdot 10^{4}$" in tex and "5,3" in tex and "Autoenkoder (MSE)" in tex
