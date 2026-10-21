from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config

DETECTOR_NAMES_PL = {"ocsvm": "OC-SVM", "iforest": "Isolation Forest", "gmm": "GMM", "mahalanobis": "Mahalanobis"}
FEATURE_SET_NAMES_PL = {"hog": "HOG", "lbp-glcm": "LBP + GLCM", "color": "kolor",
                        "hog-lbp-glcm-color": "HOG + LBP + GLCM + kolor"}
GROUP_NAMES_PL = {"texture": "tekstury", "object": "obiekty", "all": "łącznie"}
COVARIANCE_NAMES_PL = {"full": "pełna", "diag": "diagonalna", "ledoitwolf": "Ledoit-Wolf", "empirical": "empiryczna"}


def collect_shallow_runs(runs_dir: Path | None = None) -> pd.DataFrame:
    runs_dir = Path(runs_dir or config.RESULTS_DIR / "runs")
    latest = {}
    for path in sorted(runs_dir.glob("*_shallow_*")):
        info_path = path / "config.json"
        if not info_path.exists() or not (path / "metrics.csv").exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        params = info["config"].get("params") or {}
        if float(info.get("train_fraction", 1.0)) != 1.0:
            continue
        key = (params.get("detector", "ocsvm"), "-".join(params.get("features", ())))
        latest[key] = (path, info)
    frames = []
    for (detector, features), (path, info) in latest.items():
        m = pd.read_csv(path / "metrics.csv")
        tuning = info["config"].get("tuning") or {}
        frames.append(m.assign(detector=detector, features=features, run_id=path.name,
                               val_select=tuning.get("val_all"),
                               params=json.dumps(info["config"].get("params") or {}, sort_keys=True)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _seed_mean_std(part: pd.DataFrame, metric: str) -> tuple[float, float]:
    per_seed = part.groupby("seed")[metric].mean()
    return per_seed.mean(), per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0


def ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    rows = []
    for (detector, features), part in test.groupby(["detector", "features"], sort=False):
        row = {"detector": detector, "features": features, "n_seeds": part["seed"].nunique()}
        for group in ("texture", "object", "all"):
            sel = part if group == "all" else part[part["category_type"] == group]
            row[f"image_auroc_{group}"], row[f"image_auroc_{group}_std"] = _seed_mean_std(sel, "image_auroc")
        for metric in ("image_ap", "f1_mcc", "f1_balanced_accuracy", "pct_mcc"):
            row[metric], _ = _seed_mean_std(part, metric)
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(DETECTOR_NAMES_PL)} | {f: i for i, f in enumerate(FEATURE_SET_NAMES_PL)}
    return out.sort_values(["detector", "features"], key=lambda s: s.map(order)).reset_index(drop=True)


def category_table(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    choice = (metrics.drop_duplicates(["detector", "features"])
              .sort_values("val_select", ascending=False).drop_duplicates("detector")
              .set_index("detector")["features"].to_dict())
    test = metrics[metrics["subset"] == "test"]
    cols = {}
    for detector in DETECTOR_NAMES_PL:
        if detector in choice:
            part = test[(test["detector"] == detector) & (test["features"] == choice[detector])]
            cols[detector] = part.groupby("category")["image_auroc"].mean()
    table = pd.DataFrame(cols).reindex(list(config.CATEGORIES))
    return table, choice


def hyperparameter_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (detector, features), part in metrics.groupby(["detector", "features"], sort=False):
        params = json.loads(part["params"].iloc[0])
        rows.append({"detector": detector, "features": features,
                     "pca_components": params.get("pca_components"),
                     "nu": params.get("nu"), "gamma_factor": params.get("gamma_factor"),
                     "n_estimators": params.get("n_estimators"),
                     "n_components": params.get("n_components"),
                     "covariance_type": params.get("covariance_type"),
                     "covariance": params.get("covariance"),
                     "n_seeds": part["seed"].nunique(), "val_image_auroc": part["val_select"].iloc[0]})
    out = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(DETECTOR_NAMES_PL)} | {f: i for i, f in enumerate(FEATURE_SET_NAMES_PL)}
    return out.sort_values(["detector", "features"], key=lambda s: s.map(order)).reset_index(drop=True)


def _num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _short(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _detector_params(r: pd.Series) -> str:
    if r["detector"] == "ocsvm":
        return rf"$\nu$ = {_short(r['nu'])}, $g$ = {_short(r['gamma_factor'])}"
    if r["detector"] == "iforest":
        return f"{int(r['n_estimators'])} drzew"
    if r["detector"] == "gmm":
        return f"{int(r['n_components'])} skł., kowariancja {COVARIANCE_NAMES_PL[r['covariance_type']]}"
    covariance = "ledoitwolf" if pd.isna(r["covariance"]) else r["covariance"]
    return f"kowariancja {COVARIANCE_NAMES_PL[covariance]}"


def _cell(mean: float, std: float, n_seeds: int, best: bool) -> str:
    text = _num(mean) + (rf"\,$\pm$\,{_num(std)}" if n_seeds > 1 else "")
    return rf"\textbf{{{text}}}" if best else text


def latex_ablation_table(table: pd.DataFrame) -> str:
    lines = [
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        r"\thead{Detektor} & \thead{Cechy} & \thead{AUROC\\tekstury} & \thead{AUROC\\obiekty} & "
        r"\thead{AUROC\\łącznie} & \thead{AP\\łącznie} & \thead{MCC\\(próg F1)} \\",
        r"\midrule",
    ]
    best = {c: table[c].max() for c in ("image_auroc_texture", "image_auroc_object", "image_auroc_all",
                                         "image_ap", "f1_mcc")}
    previous = None
    for _, r in table.iterrows():
        if previous is not None and r["detector"] != previous:
            lines.append(r"\midrule")
        name = DETECTOR_NAMES_PL[r["detector"]] if r["detector"] != previous else ""
        previous = r["detector"]
        cells = [_cell(r[f"image_auroc_{g}"], r[f"image_auroc_{g}_std"], r["n_seeds"],
                       r[f"image_auroc_{g}"] == best[f"image_auroc_{g}"]) for g in ("texture", "object", "all")]
        cells += [_cell(r[m], 0.0, 1, r[m] == best[m]) for m in ("image_ap", "f1_mcc")]
        lines.append(" & ".join([name, FEATURE_SET_NAMES_PL[r["features"]], *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_category_table(table: pd.DataFrame, choice: dict[str, str]) -> str:
    dets = list(table.columns)
    header = " & ".join(rf"\thead{{{DETECTOR_NAMES_PL[d]}\\({FEATURE_SET_NAMES_PL[choice[d]]})}}" for d in dets)
    lines = [rf"\begin{{tabular}}{{l {'r ' * len(dets)}}}", r"\toprule",
             rf"\thead{{Kategoria}} & {header} \\", r"\midrule"]
    for group, cats in (("texture", config.TEXTURES), ("object", config.OBJECTS)):
        for c in cats:
            row = table.loc[c]
            cells = [_cell(row[d], 0.0, 1, row[d] == row.max()) for d in dets]
            lines.append(" & ".join([config.category_name_pl(c), *cells]) + r" \\")
        mean = table.loc[list(cats)].mean()
        lines.append(" & ".join([rf"\textit{{średnia: {GROUP_NAMES_PL[group]}}}",
                                 *[rf"\textit{{{_num(mean[d])}}}" for d in dets]]) + r" \\")
        lines.append(r"\midrule")
    mean = table.mean()
    lines.append(" & ".join([r"\textbf{Średnia}", *[rf"\textbf{{{_num(mean[d])}}}" for d in dets]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_hyperparameter_table(table: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{l l r l r}", r"\toprule",
             r"\thead{Detektor} & \thead{Cechy} & \thead{$k$\\(PCA)} & \thead{Parametry detektora} & "
             r"\thead{AUROC\\(walidacja)} \\", r"\midrule"]
    previous = None
    for _, r in table.iterrows():
        if previous is not None and r["detector"] != previous:
            lines.append(r"\midrule")
        name = DETECTOR_NAMES_PL[r["detector"]] if r["detector"] != previous else ""
        previous = r["detector"]
        k = "bez PCA" if pd.isna(r["pca_components"]) else str(int(r["pca_components"]))
        lines.append(" & ".join([name, FEATURE_SET_NAMES_PL[r["features"]], k, _detector_params(r),
                                 _num(r["val_image_auroc"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_shallow_tables(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or config.TABLES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = collect_shallow_runs()
    if metrics.empty:
        raise SystemExit("no shallow runs found")
    ablation = ablation_table(metrics)
    categories, choice = category_table(metrics)
    hyper = hyperparameter_table(metrics)
    paths = [out_dir / "plytkie_ablacja.csv", out_dir / "tab_plytkie_ablacja.tex",
             out_dir / "plytkie_kategorie.csv", out_dir / "tab_plytkie_kategorie.tex",
             out_dir / "plytkie_hiperparametry.csv", out_dir / "tab_plytkie_hiperparametry.tex"]
    ablation.to_csv(paths[0], index=False)
    paths[1].write_text(latex_ablation_table(ablation), encoding="utf-8")
    categories.rename(columns=lambda d: f"{d}_{choice[d]}").to_csv(paths[2], index_label="category")
    paths[3].write_text(latex_category_table(categories, choice), encoding="utf-8")
    hyper.to_csv(paths[4], index=False)
    paths[5].write_text(latex_hyperparameter_table(hyper), encoding="utf-8")
    return paths
