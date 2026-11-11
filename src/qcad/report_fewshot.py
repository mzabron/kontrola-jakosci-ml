from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from . import config
from .eda import GRID, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_WIDTH_IN, _save, setup_style
from .report import GROUP_NAMES_PL, _num
from .report_tl import run_variant
from .runner import is_cost_run

METHOD_NAMES_PL = {
    "patchcore_original": "PatchCore",
    "padim_wr50": "PaDiM WR50",
    "padim_r18": "PaDiM R18",
    "ae_mse": "Autoenkoder (MSE)",
    "ocsvm_cnn": "OC-SVM (CNN)",
    "iforest_cnn": "Isolation Forest (CNN)",
    "gmm_cnn": "GMM (CNN)",
    "mahalanobis_cnn": "Mahalanobis (CNN)",
    "ocsvm_hog": "OC-SVM (HOG)",
    "iforest_lbp-glcm": "Isolation Forest (LBP + GLCM)",
    "gmm_lbp-glcm": "GMM (LBP + GLCM)",
    "mahalanobis_lbp-glcm": "Mahalanobis (LBP + GLCM)",
}
FAMILIES = {
    "pretrained": ("patchcore_original", "padim_wr50", "padim_r18"),
    "ae": ("ae_mse",),
    "cnn": ("ocsvm_cnn", "iforest_cnn", "gmm_cnn", "mahalanobis_cnn"),
    "hand": ("ocsvm_hog", "iforest_lbp-glcm", "gmm_lbp-glcm", "mahalanobis_lbp-glcm"),
}
FAMILY_OF = {m: f for f, methods in FAMILIES.items() for m in methods}
DEVICE_NAMES_PL = {"pretrained": "GPU", "ae": "GPU", "cnn": "GPU + CPU (1 wątek)", "hand": "CPU (1 wątek)"}
MAP_METHODS = FAMILIES["pretrained"] + FAMILIES["ae"]
DETECTOR_COLORS = {"ocsvm": "#2a78d6", "iforest": "#1baf7a", "gmm": "#eb6834", "mahalanobis": "#8e5bd0"}
MAP_COLORS = {"patchcore_original": "#2a78d6", "padim_wr50": "#1baf7a", "padim_r18": "#8e5bd0",
              "ae_mse": "#eb6834"}
CACHED_FIT_RATIO = 0.01


def method_key(info: dict) -> str | None:
    cfg = info["config"]
    params = cfg.get("params") or {}
    if cfg.get("method") == "shallow":
        return f"{params.get('detector', 'ocsvm')}_{'-'.join(params.get('features', ()))}"
    if cfg.get("method") == "ae":
        return f"ae_{params.get('loss', 'mse')}"
    return run_variant(info)


def collect_fewshot_runs(runs_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_dir = Path(runs_dir or config.RESULTS_DIR / "runs")
    runs, cost = {}, {}
    for path in sorted(runs_dir.glob("*")):
        info_path = path / "config.json"
        if not info_path.exists() or not (path / "metrics.csv").exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        key = method_key(info)
        if key not in METHOD_NAMES_PL:
            continue
        if is_cost_run(info):
            cost[key] = path
        else:
            runs[(key, round(float(info.get("train_fraction", 1.0)), 4))] = path

    def load(items) -> pd.DataFrame:
        frames = [pd.read_csv(path / "metrics.csv").assign(method=key, fraction=fraction, run_id=path.name)
                  for (key, fraction), path in items]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return load(runs.items()), load(((k, 1.0), p) for k, p in cost.items())


def _fraction_seeds(part: pd.DataFrame) -> set:
    reduced = part[part["fraction"] < 1.0]
    return set((reduced if not reduced.empty else part)["seed"].unique())


def fraction_table(metrics: pd.DataFrame, metric: str = "image_auroc", group: str = "all") -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    if group != "all":
        test = test[test["category_type"] == group]
    rows = []
    for method in METHOD_NAMES_PL:
        part = test[test["method"] == method]
        if part.empty or part[metric].isna().all():
            continue
        part = part[part["seed"].isin(_fraction_seeds(part))]
        row = {"method": method, "n_seeds": part["seed"].nunique()}
        for fraction, sel in part.groupby("fraction"):
            row[fraction] = sel.groupby("seed")[metric].mean().mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("method") if rows else pd.DataFrame()


def train_sizes(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"].drop_duplicates(["fraction", "category"])
    return test.groupby("fraction")["n_train"].agg(["mean", "min", "max"])


def long_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("image_auroc", "image_ap", "pixel_auroc", "au_pro", "f1_mcc"):
        for group in ("texture", "object", "all"):
            table = fraction_table(metrics, metric, group)
            for method, r in table.iterrows():
                for fraction in [c for c in table.columns if c != "n_seeds"]:
                    if not pd.isna(r[fraction]):
                        rows.append({"method": method, "metric": metric, "group": group, "fraction": fraction,
                                     "value": r[fraction], "n_seeds": int(r["n_seeds"])})
    return pd.DataFrame(rows)


def category_table(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    rows = []
    for method in METHOD_NAMES_PL:
        part = test[test["method"] == method]
        if part.empty:
            continue
        part = part[part["seed"].isin(_fraction_seeds(part))]
        values = part.groupby(["category", "fraction"])["image_auroc"].mean().unstack("fraction")
        rows.append(values.assign(method=method).reset_index())
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.rename(columns=lambda c: f"f{round(100 * c):03d}" if isinstance(c, float) else c)


def _seed_means(part: pd.DataFrame) -> pd.DataFrame:
    per_seed = part.groupby("seed")[["fit_time_s", "test_time_per_image_ms", "model_size"]].mean()
    fitted = per_seed["fit_time_s"] >= CACHED_FIT_RATIO * per_seed["fit_time_s"].median()
    return per_seed[fitted]


def cost_table(metrics: pd.DataFrame, cost_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_NAMES_PL:
        source = "cost"
        part = cost_metrics[cost_metrics["method"] == method] if not cost_metrics.empty else cost_metrics
        if part.empty:
            source = "main"
            part = metrics[(metrics["method"] == method) & (metrics["fraction"] == 1.0)]
        part = part[part["subset"] == "test"] if not part.empty else part
        if part.empty:
            continue
        per_seed = _seed_means(part)
        rows.append({"method": method, "source": source, "device": DEVICE_NAMES_PL[FAMILY_OF[method]],
                     "n_seeds": len(per_seed), "fit_time_s": per_seed["fit_time_s"].mean(),
                     "test_time_per_image_ms": per_seed["test_time_per_image_ms"].mean(),
                     "model_size": per_seed["model_size"].mean()})
    return pd.DataFrame(rows)


def _pct(fraction: float) -> str:
    return rf"{round(100 * fraction)}\,\%"


def _signed(x: float) -> str:
    return f"{x:+.3f}".replace(".", ",").replace("-", "$-$")


def _sig(x: float) -> str:
    if x >= 100:
        return f"{x:.0f}"
    digits = max(0, 1 - math.floor(math.log10(x))) if x > 0 else 1
    return f"{x:.{digits}f}".replace(".", ",")


def _sci(x: float) -> str:
    exp = math.floor(math.log10(x))
    return rf"${f'{x / 10 ** exp:.1f}'.replace('.', '{,}')} \cdot 10^{{{exp}}}$"


def _family_blocks(table: pd.DataFrame):
    for family, methods in FAMILIES.items():
        block = [m for m in methods if m in table.index]
        if block:
            yield family, block


def latex_fraction_table(table: pd.DataFrame, sizes: pd.DataFrame) -> str:
    fractions = sorted(c for c in table.columns if c != "n_seeds")
    head = " & ".join(rf"\thead{{{_pct(f)}\\(śr. {round(sizes.loc[f, 'mean'])} obr.)}}" for f in fractions)
    lines = [r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{l {'r ' * len(fractions)}r}}", r"\toprule",
             rf"\thead{{Metoda}} & {head} & \thead{{Różnica\\{_pct(fractions[0])} -- {_pct(fractions[-1])}}} \\",
             r"\midrule"]
    best = {f: table[f].round(3).max() for f in fractions}
    first = True
    for _, block in _family_blocks(table):
        if not first:
            lines.append(r"\midrule")
        first = False
        for method in block:
            r = table.loc[method]
            cells = []
            for f in fractions:
                text = _num(r[f]) if not pd.isna(r[f]) else "--"
                cells.append(rf"\textbf{{{text}}}" if round(r[f], 3) == best[f] else text)
            cells.append(_signed(r[fractions[0]] - r[fractions[-1]]))
            lines.append(" & ".join([METHOD_NAMES_PL[method], *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_group_table(tables: dict[str, pd.DataFrame]) -> str:
    fractions = sorted(c for c in tables["texture"].columns if c != "n_seeds")
    lo, hi = fractions[0], fractions[-1]
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l r r r r r r}", r"\toprule",
             rf" & \multicolumn{{3}}{{c}}{{{GROUP_NAMES_PL['texture'].capitalize()}}} & "
             rf"\multicolumn{{3}}{{c}}{{{GROUP_NAMES_PL['object'].capitalize()}}} \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             rf"\thead{{Metoda}} & {_pct(lo)} & {_pct(hi)} & Różnica & {_pct(lo)} & {_pct(hi)} & Różnica \\",
             r"\midrule"]
    first = True
    for _, block in _family_blocks(tables["all"]):
        if not first:
            lines.append(r"\midrule")
        first = False
        for method in block:
            cells = []
            for group in ("texture", "object"):
                r = tables[group].loc[method]
                cells += [_num(r[lo]), _num(r[hi]), _signed(r[lo] - r[hi])]
            lines.append(" & ".join([METHOD_NAMES_PL[method], *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def latex_cost_table(table: pd.DataFrame) -> str:
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l l r r r}", r"\toprule",
             r"\thead{Metoda} & \thead{Sprzęt} & \thead{Czas uczenia\\{[s]}} & \thead{Czas oceny\\{[ms/obraz]}} & "
             r"\thead{Rozmiar modelu\\{[liczba wartości]}} \\", r"\midrule"]
    indexed = table.set_index("method")
    first = True
    for _, block in _family_blocks(indexed):
        if not first:
            lines.append(r"\midrule")
        first = False
        for method in block:
            r = indexed.loc[method]
            lines.append(" & ".join([METHOD_NAMES_PL[method], r["device"], _sig(r["fit_time_s"]),
                                     _sig(r["test_time_per_image_ms"]), _sci(r["model_size"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def comma_formatter(digits: int = 2) -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{v:.{digits}f}".replace(".", ","))


def plot_fractions(table: pd.DataFrame, out_dir: Path) -> None:
    fractions = sorted(c for c in table.columns if c != "n_seeds")
    x = [100 * f for f in fractions]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.0), sharey=True)
    left = [m for m in MAP_METHODS if m in table.index]
    for method in left:
        axes[0].plot(x, table.loc[method, fractions], marker="o", markersize=3.5, linewidth=1.4,
                     color=MAP_COLORS[method], label=METHOD_NAMES_PL[method])
    for method in FAMILIES["cnn"] + FAMILIES["hand"]:
        if method not in table.index:
            continue
        detector = method.split("_")[0]
        cnn = FAMILY_OF[method] == "cnn"
        axes[1].plot(x, table.loc[method, fractions], marker="o" if cnn else "s", markersize=3.2,
                     linewidth=1.4, linestyle="-" if cnn else "--", color=DETECTOR_COLORS[detector],
                     label=METHOD_NAMES_PL[method])
    titles = ("metody z mapami anomalii", "detektory płytkie")
    for ax, title in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.minorticks_off()
        ax.set_xlabel("udział obrazów uczących [%]")
        ax.set_title(title, color=TEXT_PRIMARY)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=7,
                  handlelength=2.2, columnspacing=1.0, labelcolor=TEXT_SECONDARY)
    axes[0].set_ylabel("AUROC obrazów (zbiór testowy)")
    axes[0].yaxis.set_major_formatter(comma_formatter())
    lo = np.nanmin(table[fractions].to_numpy())
    axes[0].set_ylim(math.floor((lo - 0.01) * 20) / 20, 1.0)
    fig.subplots_adjust(wspace=0.08)
    _save(fig, out_dir, "fewshot_auroc")


def write_fewshot_report(tables_dir: Path | None = None, figures_dir: Path | None = None,
                         figures: bool = True) -> list[Path]:
    tables_dir = Path(tables_dir or config.TABLES_DIR)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics, cost_metrics = collect_fewshot_runs()
    if metrics.empty or not (metrics["fraction"] < 1.0).any():
        raise SystemExit("no training fraction runs found")
    auroc = fraction_table(metrics)
    groups = {g: fraction_table(metrics, "image_auroc", g) for g in ("texture", "object", "all")}
    aupro = fraction_table(metrics[metrics["method"].isin(MAP_METHODS)], "au_pro")
    sizes = train_sizes(metrics)
    cost = cost_table(metrics, cost_metrics)
    frames = {"fewshot_wyniki.csv": long_table(metrics), "fewshot_kategorie.csv": category_table(metrics),
              "fewshot_liczba_obrazow.csv": sizes.reset_index(), "koszt.csv": cost}
    texts = {"tab_fewshot_auroc.tex": latex_fraction_table(auroc, sizes),
             "tab_fewshot_grupy.tex": latex_group_table(groups),
             "tab_fewshot_aupro.tex": latex_fraction_table(aupro, sizes),
             "tab_koszt.tex": latex_cost_table(cost)}
    saved = []
    for name, df in frames.items():
        df.to_csv(tables_dir / name, index=False)
        saved.append(tables_dir / name)
    for name, text in texts.items():
        (tables_dir / name).write_text(text, encoding="utf-8")
        saved.append(tables_dir / name)
    if figures:
        setup_style()
        out = Path(figures_dir or config.FIGURES_DIR / "fewshot")
        plot_fractions(auroc, out)
        saved += sorted(out.glob("*.pdf"))
    return saved
