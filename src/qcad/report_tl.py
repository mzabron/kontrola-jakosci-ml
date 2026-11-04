from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config
from .data import load_split_file, load_subset
from .eda import TEXT_PRIMARY, TEXT_WIDTH_IN, _save, _show, setup_style
from .report import (DETECTOR_NAMES_PL, FEATURE_SET_NAMES_PL, GROUP_NAMES_PL, HANDCRAFTED, _cell, _num,
                     collect_shallow_runs, hyperparameter_table, latex_hyperparameter_table)
from .runner import build_method, fit_method

VARIANT_NAMES_PL = {"padim_r18": "PaDiM R18", "padim_wr50": "PaDiM WR50",
                    "patchcore_concat": "PatchCore (anomalib)", "patchcore_original": "PatchCore"}
CATEGORY_VARIANTS = ("padim_r18", "padim_wr50", "patchcore_original")
FIGURE_VARIANTS = ("padim_r18", "padim_wr50", "patchcore_original")


def run_variant(info: dict) -> str | None:
    cfg = info["config"]
    params = cfg.get("params") or {}
    if cfg.get("method") == "padim":
        return {"resnet18": "padim_r18", "wide_resnet50_2": "padim_wr50"}.get(params.get("backbone", "resnet18"))
    if cfg.get("method") == "patchcore":
        return f"patchcore_{params.get('aggregation', 'concat')}"
    return None


def collect_tl_runs(runs_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    runs_dir = Path(runs_dir or config.RESULTS_DIR / "runs")
    latest = {}
    for path in sorted(runs_dir.glob("*")):
        info_path = path / "config.json"
        if not info_path.exists() or not (path / "metrics.csv").exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        variant = run_variant(info)
        if variant is None or float(info.get("train_fraction", 1.0)) != 1.0:
            continue
        latest[variant] = (path, info)
    frames = [pd.read_csv(path / "metrics.csv").assign(variant=v, run_id=path.name)
              for v, (path, _) in latest.items()]
    infos = {v: info for v, (_, info) in latest.items()}
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), infos


def _seed_stats(part: pd.DataFrame, metric: str) -> tuple[float, float]:
    per_seed = part.groupby("seed")[metric].mean()
    return per_seed.mean(), per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    rows = []
    for variant in VARIANT_NAMES_PL:
        part = test[test["variant"] == variant]
        if part.empty:
            continue
        row = {"variant": variant, "n_seeds": part["seed"].nunique()}
        for group in ("texture", "object", "all"):
            sel = part if group == "all" else part[part["category_type"] == group]
            row[f"image_auroc_{group}"], row[f"image_auroc_{group}_std"] = _seed_stats(sel, "image_auroc")
        for metric in ("image_ap", "pixel_auroc", "au_pro", "f1_mcc"):
            row[metric], row[f"{metric}_std"] = _seed_stats(part, metric)
        row["model_size"] = part["model_size"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def category_table(metrics: pd.DataFrame, variants=CATEGORY_VARIANTS) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    out = {}
    for variant in variants:
        part = test[test["variant"] == variant]
        if part.empty:
            continue
        grouped = part.groupby("category")
        for metric in ("image_auroc", "pixel_auroc", "au_pro"):
            out[f"{variant}_{metric}"] = grouped[metric].mean()
    return pd.DataFrame(out).reindex(list(config.CATEGORIES))


def cross_table(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    hand = metrics[metrics["features"].isin(HANDCRAFTED)]
    choice = (hand.drop_duplicates(["detector", "features"]).sort_values("val_select", ascending=False)
              .drop_duplicates("detector").set_index("detector")["features"].to_dict())
    rows = []
    for detector in DETECTOR_NAMES_PL:
        cnn = test[(test["detector"] == detector) & (test["features"] == "cnn")]
        if cnn.empty or detector not in choice:
            continue
        row = {"detector": detector, "hand_features": choice[detector]}
        for prefix, part in (("hand", test[(test["detector"] == detector) & (test["features"] == choice[detector])]),
                             ("cnn", cnn)):
            row[f"{prefix}_n_seeds"] = part["seed"].nunique()
            for group in ("texture", "object", "all"):
                sel = part if group == "all" else part[part["category_type"] == group]
                row[f"{prefix}_{group}"], row[f"{prefix}_{group}_std"] = _seed_stats(sel, "image_auroc")
        row["gain_all"] = row["cnn_all"] - row["hand_all"]
        rows.append(row)
    return pd.DataFrame(rows)


def latex_summary_table(table: pd.DataFrame) -> str:
    lines = [r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l r r r r r r r}", r"\toprule",
             r"\thead{Metoda} & \thead{AUROC\\tekstury} & \thead{AUROC\\obiekty} & \thead{AUROC\\łącznie} & "
             r"\thead{AP\\łącznie} & \thead{AUROC\\pikseli} & \thead{AU-PRO} & \thead{MCC\\(próg F1)} \\",
             r"\midrule"]
    cols = ["image_auroc_texture", "image_auroc_object", "image_auroc_all", "image_ap", "pixel_auroc",
            "au_pro", "f1_mcc"]
    best = {c: table[c].round(3).max() for c in cols}
    for _, r in table.iterrows():
        cells = [_cell(r[c], r[f"{c}_std"], r["n_seeds"] if c.startswith("image_auroc") else 1,
                       round(r[c], 3) == best[c]) for c in cols]
        lines.append(" & ".join([VARIANT_NAMES_PL[r["variant"]], *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_category_table(table: pd.DataFrame) -> str:
    variants = [v for v in VARIANT_NAMES_PL if f"{v}_image_auroc" in table.columns]
    metrics = (("image_auroc", "AUROC obrazów"), ("au_pro", "AU-PRO"))
    n = len(variants)
    spec = "l " + " ".join("r" * n for _ in metrics)
    head = " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{name}}}" for _, name in metrics)
    rules = " ".join(rf"\cmidrule(lr){{{2 + k * n}-{1 + (k + 1) * n}}}" for k in range(len(metrics)))
    sub = " & ".join(VARIANT_NAMES_PL[v] for _ in metrics for v in variants)
    lines = [r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             rf" & {head} \\", rules, rf"Kategoria & {sub} \\", r"\midrule"]

    def cells(values: pd.Series, style: str | None) -> list[str]:
        out = []
        for metric, _ in metrics:
            vals = {v: round(values[f"{v}_{metric}"], 3) for v in variants}
            top = max(vals.values())
            for v in variants:
                text = _num(vals[v])
                if style:
                    text = rf"\{style}{{{text}}}"
                elif vals[v] == top:
                    text = rf"\textbf{{{text}}}"
                out.append(text)
        return out

    for group, cats in (("texture", config.TEXTURES), ("object", config.OBJECTS)):
        for c in cats:
            lines.append(" & ".join([config.category_name_pl(c), *cells(table.loc[c], None)]) + r" \\")
        mean = table.loc[list(cats)].mean()
        lines.append(" & ".join([rf"\textit{{średnia: {GROUP_NAMES_PL[group]}}}", *cells(mean, "textit")]) + r" \\")
        lines.append(r"\midrule")
    lines.append(" & ".join([r"\textbf{Średnia}", *cells(table.mean(), "textbf")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_cross_table(table: pd.DataFrame) -> str:
    lines = [r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l l r r r r r r r}", r"\toprule",
             r" & \multicolumn{4}{c}{Cechy projektowane ręcznie} & \multicolumn{3}{c}{Cechy CNN (WideResNet-50)} & \\",
             r"\cmidrule(lr){2-5} \cmidrule(lr){6-8}",
             r"\thead{Detektor} & \thead{Zestaw} & \thead{tekstury} & \thead{obiekty} & \thead{łącznie} & "
             r"\thead{tekstury} & \thead{obiekty} & \thead{łącznie} & \thead{Różnica} \\", r"\midrule"]
    for _, r in table.iterrows():
        cells = [FEATURE_SET_NAMES_PL[r["hand_features"]]]
        for prefix in ("hand", "cnn"):
            cells += [_num(r[f"{prefix}_texture"]), _num(r[f"{prefix}_object"]),
                      _cell(r[f"{prefix}_all"], r[f"{prefix}_all_std"], r[f"{prefix}_n_seeds"], False)]
        cells.append(f"{r['gain_all']:+.3f}".replace(".", ",").replace("-", "$-$"))
        lines.append(" & ".join([DETECTOR_NAMES_PL[r["detector"]], *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _fit(info: dict, category: str, seed: int, split: pd.DataFrame):
    cfg = info["config"]
    method = build_method(cfg["method"], dict(cfg.get("params") or {}), seed, category)
    train = load_subset(category, "train", split)
    holdout = load_subset(category, "holdout", split)
    return fit_method(method, train.images, holdout.images)


def plot_maps(out_dir: Path, infos: dict, categories=("hazelnut", "screw", "carpet", "tile"), seed: int = 0) -> None:
    split = load_split_file()
    rng = np.random.default_rng(seed)
    variants = [v for v in FIGURE_VARIANTS if v in infos]
    fig, axes = plt.subplots(len(categories), 1 + len(variants),
                             figsize=(TEXT_WIDTH_IN * (1 + len(variants)) / 5, 1.3 * len(categories) + 0.4),
                             squeeze=False)
    for r, cat in enumerate(categories):
        test = load_subset(cat, "test", split)
        k = rng.choice(np.flatnonzero(test.labels == 1))
        img = test.images[k:k + 1]
        _show(axes[r, 0], img[0], test.masks[k], title="obraz testowy" if r == 0 else None)
        axes[r, 0].set_ylabel(config.category_name_pl(cat), fontsize=8, color=TEXT_PRIMARY)
        for j, variant in enumerate(variants):
            method = _fit(infos[variant], cat, seed, split)
            amap = method.predict(img).maps[0]
            ax = axes[r, 1 + j]
            _show(ax, amap / max(float(amap.max()), 1e-12), test.masks[k],
                  title=f"mapa anomalii ({VARIANT_NAMES_PL[variant]})" if r == 0 else None)
            ax.images[0].set_cmap("inferno")
            ax.images[0].set_clim(0, 1)
            del method
            gc.collect()
            torch.cuda.empty_cache()
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    _save(fig, out_dir, "tl_mapy")


def write_tl_report(tables_dir: Path | None = None, figures_dir: Path | None = None,
                    figures: bool = True) -> list[Path]:
    tables_dir = Path(tables_dir or config.TABLES_DIR)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics, infos = collect_tl_runs()
    if metrics.empty:
        raise SystemExit("no padim/patchcore runs found")
    summary = summary_table(metrics)
    categories = category_table(metrics)
    frames = {"tl_wyniki.csv": (summary, False), "tl_kategorie.csv": (categories, True)}
    texts = {"tab_tl_wyniki.tex": latex_summary_table(summary),
             "tab_tl_kategorie.tex": latex_category_table(categories)}
    shallow = collect_shallow_runs()
    if not shallow.empty and (shallow["features"] == "cnn").any():
        cross = cross_table(shallow)
        hyper = hyperparameter_table(shallow[shallow["features"] == "cnn"])
        frames |= {"krzyzowy.csv": (cross, False), "krzyzowy_hiperparametry.csv": (hyper, False)}
        texts |= {"tab_krzyzowy.tex": latex_cross_table(cross),
                  "tab_krzyzowy_hiperparametry.tex": latex_hyperparameter_table(hyper)}
    saved = []
    for name, (df, index) in frames.items():
        df.to_csv(tables_dir / name, index=index, index_label="category" if index else None)
        saved.append(tables_dir / name)
    for name, text in texts.items():
        (tables_dir / name).write_text(text, encoding="utf-8")
        saved.append(tables_dir / name)
    if figures:
        setup_style()
        out = Path(figures_dir or config.FIGURES_DIR / "tl")
        plot_maps(out, infos)
        saved += sorted(out.glob("*.pdf"))
    return saved
