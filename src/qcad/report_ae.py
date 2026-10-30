from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config
from .data import load_split_file, load_subset
from .eda import TEXT_PRIMARY, TEXT_WIDTH_IN, _save, _show, setup_style
from .methods.autoencoder import AUGMENTATIONS, augment, random_crops
from .report import GROUP_NAMES_PL, _cell, _num
from .runner import build_method, fit_method

LOSS_NAMES_PL = {"mse": "MSE", "ssim": "SSIM"}
SCORE_NAMES_PL = {"max": "maksimum", "mean": "średnia", "top": "średnia z 1\\% największych"}
UNSTABLE_STD = 0.05


def collect_ae_runs(runs_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    runs_dir = Path(runs_dir or config.RESULTS_DIR / "runs")
    latest = {}
    for path in sorted(runs_dir.glob("*_ae_*")):
        info_path = path / "config.json"
        if not info_path.exists() or not (path / "metrics.csv").exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if float(info.get("train_fraction", 1.0)) != 1.0:
            continue
        latest[(info["config"].get("params") or {}).get("loss", "mse")] = (path, info)
    frames = [pd.read_csv(path / "metrics.csv").assign(loss=loss, run_id=path.name)
              for loss, (path, _) in latest.items()]
    infos = {loss: info for loss, (_, info) in latest.items()}
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), infos


def _seed_stats(part: pd.DataFrame, metric: str) -> tuple[float, float]:
    per_seed = part.groupby("seed")[metric].mean()
    return per_seed.mean(), per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    rows = []
    for loss in LOSS_NAMES_PL:
        part = test[test["loss"] == loss]
        if part.empty:
            continue
        row = {"loss": loss, "n_seeds": part["seed"].nunique()}
        for group in ("texture", "object", "all"):
            sel = part if group == "all" else part[part["category_type"] == group]
            row[f"image_auroc_{group}"], row[f"image_auroc_{group}_std"] = _seed_stats(sel, "image_auroc")
        for metric in ("image_ap", "pixel_auroc", "au_pro", "f1_mcc"):
            row[metric], row[f"{metric}_std"] = _seed_stats(part, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def category_table(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    out = {}
    for loss in LOSS_NAMES_PL:
        part = test[test["loss"] == loss]
        if part.empty:
            continue
        grouped = part.groupby("category")
        for metric in ("image_auroc", "pixel_auroc", "au_pro"):
            out[f"{loss}_{metric}"] = grouped[metric].mean()
        out[f"{loss}_image_auroc_std"] = grouped["image_auroc"].std(ddof=1).fillna(0.0)
    return pd.DataFrame(out).reindex(list(config.CATEGORIES))


def tuning_table(val_summary: pd.DataFrame) -> pd.DataFrame:
    df = val_summary.copy()
    for col in ("loss", "score"):
        df[col] = df[col].map(lambda v: json.loads(v) if isinstance(v, str) and v.startswith('"') else v)
    table = df.pivot_table(index=["map_sigma", "score"], columns="loss", values="image_auroc_all")
    order = [(s, r) for s in sorted(df["map_sigma"].unique()) for r in ("max", "mean", "top")]
    return table.reindex(order)


def augmentation_table() -> pd.DataFrame:
    rows = []
    for cat in config.CATEGORIES:
        spec = AUGMENTATIONS[cat]
        flips = [name for key, name in (("hflip", "poziome"), ("vflip", "pionowe")) if spec[key]]
        rotation = []
        if spec["rot90"]:
            rotation.append("o 90°")
        if spec["rotate"] > 0:
            rotation.append(f"±{spec['rotate']:g}°")
        rows.append({"category": cat, "category_type": config.category_type(cat),
                     "flips": " i ".join(flips), "rotation": ", ".join(rotation),
                     "shift": f"±{spec['shift']} px" if spec["shift"] else ""})
    return pd.DataFrame(rows)


def latex_summary_table(table: pd.DataFrame) -> str:
    lines = [r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l r r r r r r r}", r"\toprule",
             r"\thead{Strata} & \thead{AUROC\\tekstury} & \thead{AUROC\\obiekty} & \thead{AUROC\\łącznie} & "
             r"\thead{AP\\łącznie} & \thead{AUROC\\pikseli} & \thead{AU-PRO} & \thead{MCC\\(próg F1)} \\",
             r"\midrule"]
    cols = ["image_auroc_texture", "image_auroc_object", "image_auroc_all", "image_ap", "pixel_auroc",
            "au_pro", "f1_mcc"]
    best = {c: table[c].max() for c in cols}
    for _, r in table.iterrows():
        cells = [_cell(r[c], r[f"{c}_std"], r["n_seeds"], r[c] == best[c]) for c in cols]
        lines.append(" & ".join([LOSS_NAMES_PL[r["loss"]], *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _pair_cells(row: pd.Series, metric: str, losses: list[str], mark_unstable: bool = False) -> list[str]:
    values = {loss: round(row[f"{loss}_{metric}"], 3) for loss in losses}
    top = max(values.values())
    cells = []
    for loss in losses:
        text = _num(values[loss])
        if mark_unstable and row[f"{loss}_image_auroc_std"] > UNSTABLE_STD:
            text += r"$^{\dagger}$"
        cells.append(rf"\textbf{{{text}}}" if len(losses) > 1 and values[loss] == top else text)
    return cells


def latex_category_table(table: pd.DataFrame) -> str:
    losses = [loss for loss in LOSS_NAMES_PL if f"{loss}_image_auroc" in table.columns]
    metrics = (("image_auroc", "AUROC obrazów"), ("pixel_auroc", "AUROC pikseli"), ("au_pro", "AU-PRO"))
    n = len(losses)
    spec = "l " + " ".join("r" * n for _ in metrics)
    head = " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{name}}}" for _, name in metrics)
    rules = " ".join(rf"\cmidrule(lr){{{2 + k * n}-{1 + (k + 1) * n}}}" for k in range(len(metrics)))
    sub = " & ".join(LOSS_NAMES_PL[loss] for _ in metrics for loss in losses)
    lines = [r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             rf" & {head} \\", rules, rf"Kategoria & {sub} \\", r"\midrule"]
    for group, cats in (("texture", config.TEXTURES), ("object", config.OBJECTS)):
        for c in cats:
            row = table.loc[c]
            cells = [cell for metric, _ in metrics
                     for cell in _pair_cells(row, metric, losses, metric == "image_auroc")]
            lines.append(" & ".join([config.category_name_pl(c), *cells]) + r" \\")
        mean = table.loc[list(cats)].mean()
        cells = [rf"\textit{{{_num(mean[f'{loss}_{metric}'])}}}" for metric, _ in metrics for loss in losses]
        lines.append(" & ".join([rf"\textit{{średnia: {GROUP_NAMES_PL[group]}}}", *cells]) + r" \\")
        lines.append(r"\midrule")
    mean = table.mean()
    cells = [rf"\textbf{{{_num(mean[f'{loss}_{metric}'])}}}" for metric, _ in metrics for loss in losses]
    lines.append(" & ".join([r"\textbf{Średnia}", *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_tuning_table(table: pd.DataFrame) -> str:
    losses = [loss for loss in LOSS_NAMES_PL if loss in table.columns]
    best = {loss: table[loss].max() for loss in losses}
    lines = [rf"\begin{{tabular}}{{r l {'r ' * len(losses)}}}", r"\toprule",
             r"\thead{$\sigma$ [px]} & \thead{Wynik obrazu} & "
             + " & ".join(rf"\thead{{{LOSS_NAMES_PL[loss]}}}" for loss in losses) + r" \\", r"\midrule"]
    previous = None
    for (sigma, score), r in table.iterrows():
        if previous is not None and sigma != previous:
            lines.append(r"\midrule")
        label = f"{sigma:g}" if sigma != previous else ""
        previous = sigma
        cells = [rf"\textbf{{{_num(r[loss])}}}" if r[loss] == best[loss] else _num(r[loss]) for loss in losses]
        lines.append(" & ".join([label, SCORE_NAMES_PL[score], *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_augmentation_table(table: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{l l l l l}", r"\toprule",
             r"\thead{Kategoria} & \thead{Wycinek} & \thead{Odbicia} & \thead{Obrót} & \thead{Przesunięcie} \\",
             r"\midrule"]
    for group, cats in (("texture", config.TEXTURES), ("object", config.OBJECTS)):
        for _, r in table[table["category"].isin(cats)].iterrows():
            crop = r"łatka $128 \times 128$" if group == "texture" else "cały obraz"
            cells = [(v or "--").replace("±", r"$\pm$").replace("°", r"$^{\circ}$")
                     for v in (r["flips"], r["rotation"], r["shift"])]
            lines.append(" & ".join([config.category_name_pl(r["category"]), crop, *cells]) + r" \\")
        if group == "texture":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _to_display(x: torch.Tensor) -> np.ndarray:
    img = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return img[..., 0] if img.shape[-1] == 1 else img


def plot_augmentations(out_dir: Path, categories=("carpet", "bottle", "cable", "screw"), n: int = 5,
                       seed: int = 0) -> None:
    split = load_split_file()
    gen = torch.Generator().manual_seed(seed)
    fig, axes = plt.subplots(len(categories), n + 1, figsize=(TEXT_WIDTH_IN, 1.2 * len(categories) + 0.2),
                             squeeze=False)
    for r, cat in enumerate(categories):
        image = load_subset(cat, "train", split).images[:1]
        x = torch.as_tensor(image).permute(0, 3, 1, 2).float().div(255).repeat(n, 1, 1, 1)
        aug = augment(x, AUGMENTATIONS[cat], gen)
        if config.category_type(cat) == "texture":
            aug = random_crops(aug, 128, gen)
        _show(axes[r, 0], _to_display(x[0]), title="oryginał" if r == 0 else None)
        for c in range(n):
            _show(axes[r, c + 1], _to_display(aug[c]), title=f"przykład {c + 1}" if r == 0 else None)
        axes[r, 0].set_ylabel(config.category_name_pl(cat), fontsize=8, color=TEXT_PRIMARY)
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    _save(fig, out_dir, "ae_augmentacja")


def _load_model(info: dict, category: str, seed: int, split: pd.DataFrame):
    params = dict(info["config"].get("params") or {})
    method = build_method("ae", params, seed, category)
    train = load_subset(category, "train", split)
    holdout = load_subset(category, "holdout", split)
    return fit_method(method, train.images, holdout.images)


def plot_reconstructions(out_dir: Path, infos: dict, categories=("hazelnut", "screw", "carpet", "tile"),
                         seed: int = 0) -> None:
    split = load_split_file()
    rng = np.random.default_rng(seed)
    losses = [loss for loss in LOSS_NAMES_PL if loss in infos]
    ncol = 1 + 2 * len(losses)
    fig, axes = plt.subplots(len(categories), ncol, figsize=(TEXT_WIDTH_IN, 1.3 * len(categories) + 0.4),
                             squeeze=False)
    for r, cat in enumerate(categories):
        test = load_subset(cat, "test", split)
        k = rng.choice(np.flatnonzero(test.labels == 1))
        img = test.images[k:k + 1]
        _show(axes[r, 0], img[0], test.masks[k], title="obraz testowy" if r == 0 else None)
        axes[r, 0].set_ylabel(config.category_name_pl(cat), fontsize=8, color=TEXT_PRIMARY)
        for j, loss in enumerate(losses):
            method = _load_model(infos[loss], cat, seed, split)
            with torch.no_grad():
                x = method._prepare(img)
                recon = _to_display(method.reconstruct(x)[0])
            amap = method.anomaly_maps(img)[0]
            ax_r, ax_m = axes[r, 1 + 2 * j], axes[r, 2 + 2 * j]
            name = LOSS_NAMES_PL[loss]
            _show(ax_r, recon, title=f"rekonstrukcja ({name})" if r == 0 else None)
            if recon.ndim == 2:
                ax_r.images[0].set_cmap("gray")
                ax_r.images[0].set_clim(0, 1)
            _show(ax_m, amap / max(float(amap.max()), 1e-12), test.masks[k],
                  title=f"mapa anomalii ({name})" if r == 0 else None)
            ax_m.images[0].set_cmap("inferno")
            ax_m.images[0].set_clim(0, 1)
            torch.cuda.empty_cache()
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    _save(fig, out_dir, "ae_rekonstrukcje")


def write_ae_report(tables_dir: Path | None = None, figures_dir: Path | None = None,
                    figures: bool = True) -> list[Path]:
    tables_dir = Path(tables_dir or config.TABLES_DIR)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics, infos = collect_ae_runs()
    if metrics.empty:
        raise SystemExit("no ae runs found")
    summary = summary_table(metrics)
    categories = category_table(metrics)
    augment_tab = augmentation_table()
    paths = {"ae_wyniki.csv": summary, "ae_kategorie.csv": categories, "ae_augmentacja.csv": augment_tab}
    tuning_ids = {info["config"].get("tuning", {}).get("run_id") for info in infos.values()} - {None}
    tuning = None
    if len(tuning_ids) == 1:
        val_path = config.RESULTS_DIR / "tuning" / tuning_ids.pop() / "val_summary.csv"
        if val_path.exists():
            tuning = tuning_table(pd.read_csv(val_path))
            paths["ae_strojenie.csv"] = tuning.reset_index()
    saved = []
    for name, df in paths.items():
        df.to_csv(tables_dir / name, index=name == "ae_kategorie.csv", index_label="category")
        saved.append(tables_dir / name)
    texts = {"tab_ae_wyniki.tex": latex_summary_table(summary),
             "tab_ae_kategorie.tex": latex_category_table(categories),
             "tab_ae_augmentacja.tex": latex_augmentation_table(augment_tab)}
    if tuning is not None:
        texts["tab_ae_strojenie.tex"] = latex_tuning_table(tuning)
    for name, text in texts.items():
        (tables_dir / name).write_text(text, encoding="utf-8")
        saved.append(tables_dir / name)
    if figures:
        setup_style()
        out = Path(figures_dir or config.FIGURES_DIR / "ae")
        plot_augmentations(out)
        plot_reconstructions(out, infos)
        saved += sorted(out.glob("*.pdf"))
    return saved
