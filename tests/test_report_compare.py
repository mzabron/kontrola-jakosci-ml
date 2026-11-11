import numpy as np
import pandas as pd

from qcad import config
from qcad.report_compare import (category_rows, category_table, group_table, label_positions, latex_group_table,
                                 latex_size_table, representatives, size_edges, size_table)


def _methods(tmp_path):
    methods = {}
    for key, family, base, val in (("mahalanobis_lbp-glcm", "hand", 0.7, 0.8), ("ocsvm_hog", "hand", 0.6, 0.9),
                                   ("ae_mse", "ae", 0.75, None), ("patchcore_original", "pretrained", 0.95, None)):
        rows = []
        for i, cat in enumerate(config.CATEGORIES):
            shift = 0.1 if config.category_type(cat) == "texture" else 0.0
            rows.append({"seed": 0, "category": cat, "subset": "test", "category_type": config.category_type(cat),
                         "image_auroc": base + shift, "au_pro": np.nan if family == "hand" else base - 0.1})
        path = tmp_path / key
        path.mkdir()
        scores = []
        for cat in ("carpet", "bottle"):
            for k, (label, area) in enumerate(((0, 0.0), (0, 0.0), (1, 0.001), (1, 0.02), (1, 0.2))):
                scores.append({"seed": 0, "id": f"{cat}/{k}", "subset": "test", "label": label,
                               "defect_type": "x", "score": float(k) if key != "ocsvm_hog" else float(-k)})
        pd.DataFrame(scores).to_csv(path / "scores.csv", index=False)
        methods[key] = {"name": key, "family": family, "metrics": pd.DataFrame(rows), "path": path, "val": val}
    return methods


def _areas():
    rows = []
    for cat in ("carpet", "bottle"):
        for k, (label, area) in enumerate(((0, 0.0), (0, 0.0), (1, 0.001), (1, 0.02), (1, 0.2))):
            rows.append({"id": f"{cat}/{k}", "category": cat, "label": label, "area": area})
    return pd.DataFrame(rows)


def test_group_and_category_tables(tmp_path):
    methods = _methods(tmp_path)
    keys = representatives(methods)
    assert keys == ["ocsvm_hog", "ae_mse", "patchcore_original"]
    groups = group_table(methods)
    row = groups.set_index("method").loc["patchcore_original"]
    assert abs(row["gap"] - 0.1) < 1e-9 and abs(row["au_pro"] - 0.85) < 1e-9
    tex = latex_group_table(groups)
    assert "+0,100" in tex and "--" in tex and r"\midrule" in tex
    table = category_table(methods, keys)
    assert list(table.index) == list(config.CATEGORIES)
    rows = category_rows(table)
    assert list(rows.index[:6]) == ["dywan", "siatka", "skóra", "płytka", "drewno", "średnia: tekstury"]
    assert rows.index[-1] == "średnia" and len(rows) == 18
    assert abs(rows.loc["średnia: obiekty", "patchcore_original"] - 0.95) < 1e-9


def test_size_table(tmp_path):
    methods = _methods(tmp_path)
    areas = _areas()
    lo, hi = size_edges(areas)
    assert lo < hi
    table = size_table(methods, ["ae_mse", "ocsvm_hog"], areas)
    assert table.loc["ae_mse"].tolist() == [1.0, 1.0, 1.0]
    assert table.loc["ocsvm_hog"].tolist() == [0.0, 0.0, 0.0]
    tex = latex_size_table(table, methods, (lo, hi))
    assert "małe" in tex and "duże" in tex and r"\textbf{1,000}" in tex


def test_label_positions_spread_close_points():
    data = pd.DataFrame({"test_time_per_image_ms": [4.0, 4.1, 4.2, 100.0],
                         "image_auroc": [0.890, 0.891, 0.892, 0.891]}, index=["a", "b", "c", "d"])
    pos = label_positions(data)
    ys = sorted(pos[k][2] for k in "abc")
    assert all(b - a >= 0.0075 - 1e-12 for a, b in zip(ys, ys[1:]))
    assert pos["d"][2] == 0.891
