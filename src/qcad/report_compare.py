from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config
from .data import load_split_file, load_subset
from .eda import GRID, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_WIDTH_IN, _save, setup_style
from .report import DETECTOR_NAMES_PL, FEATURE_SET_NAMES_PL, HANDCRAFTED, _num, collect_shallow_runs
from .report_ae import collect_ae_runs
from .report_fewshot import collect_fewshot_runs, comma_formatter, cost_table, fraction_table
from .report_tl import collect_tl_runs

FAMILY_NAMES_PL = {"hand": "cechy projektowane ręcznie", "ae": "autoenkoder", "cnn": "cechy CNN z detektorami płytkimi",
                   "pretrained": "cechy sieci pretrenowanych"}
FAMILY_COLORS = {"hand": "#eb6834", "ae": "#8e5bd0", "cnn": "#1baf7a", "pretrained": "#2a78d6"}
TL_NAMES_PL = {"padim_r18": "PaDiM R18", "padim_wr50": "PaDiM WR50", "patchcore_original": "PatchCore"}
SIZE_NAMES_PL = ("małe", "średnie", "duże")
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])


def _selected_features(shallow: pd.DataFrame) -> dict[str, str]:
    hand = shallow[shallow["features"].isin(HANDCRAFTED)]
    return (hand.drop_duplicates(["detector", "features"]).sort_values("val_select", ascending=False)
            .drop_duplicates("detector").set_index("detector")["features"].to_dict())


def collect_methods(runs_dir: Path | None = None) -> dict[str, dict]:
    runs_dir = Path(runs_dir or config.RESULTS_DIR / "runs")
    methods = {}
    shallow = collect_shallow_runs(runs_dir)
    if not shallow.empty:
        choice = _selected_features(shallow)
        for features, family in (("hand", "hand"), ("cnn", "cnn")):
            for detector in DETECTOR_NAMES_PL:
                feat = choice.get(detector) if features == "hand" else "cnn"
                part = shallow[(shallow["detector"] == detector) & (shallow["features"] == feat)]
                if part.empty:
                    continue
                label = FEATURE_SET_NAMES_PL[feat] if features == "hand" else "CNN"
                methods[f"{detector}_{feat}"] = {
                    "name": f"{DETECTOR_NAMES_PL[detector]} ({label})", "family": family, "metrics": part,
                    "path": runs_dir / part["run_id"].iloc[0], "val": part["val_select"].iloc[0]}
    ae, _ = collect_ae_runs(runs_dir)
    for loss in ("mse", "ssim"):
        part = ae[ae["loss"] == loss] if not ae.empty else ae
        if not part.empty:
            methods[f"ae_{loss}"] = {"name": f"Autoenkoder ({loss.upper()})", "family": "ae", "metrics": part,
                                     "path": runs_dir / part["run_id"].iloc[0]}
    tl, _ = collect_tl_runs(runs_dir)
    for variant, name in TL_NAMES_PL.items():
        part = tl[tl["variant"] == variant] if not tl.empty else tl
        if not part.empty:
            methods[variant] = {"name": name, "family": "pretrained", "metrics": part,
                                "path": runs_dir / part["run_id"].iloc[0]}
    return methods


def representatives(methods: dict[str, dict]) -> list[str]:
    keys = []
    for family in ("hand", "cnn"):
        cands = [k for k, m in methods.items() if m["family"] == family and m.get("val") is not None]
        if cands:
            keys.append(max(cands, key=lambda k: methods[k]["val"]))
    keys += [k for k in ("ae_mse", "ae_ssim", "padim_wr50", "patchcore_original") if k in methods]
    order = {"hand": 0, "ae": 1, "cnn": 2, "pretrained": 3}
    return sorted(keys, key=lambda k: (order[methods[k]["family"]], list(methods).index(k)))


def _seed_mean(part: pd.DataFrame, metric: str) -> float:
    return part.groupby("seed")[metric].mean().mean()


def group_table(methods: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, m in methods.items():
        test = m["metrics"][m["metrics"]["subset"] == "test"]
        row = {"method": key, "name": m["name"], "family": m["family"], "n_seeds": test["seed"].nunique()}
        for group in ("texture", "object", "all"):
            sel = test if group == "all" else test[test["category_type"] == group]
            row[f"image_auroc_{group}"] = _seed_mean(sel, "image_auroc")
        row["gap"] = row["image_auroc_texture"] - row["image_auroc_object"]
        row["au_pro"] = _seed_mean(test, "au_pro") if test["au_pro"].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def category_table(methods: dict[str, dict], keys: list[str]) -> pd.DataFrame:
    cols = {}
    for key in keys:
        test = methods[key]["metrics"][methods[key]["metrics"]["subset"] == "test"]
        cols[key] = test.groupby("category")["image_auroc"].mean()
    return pd.DataFrame(cols).reindex(list(config.CATEGORIES))


def defect_table(methods: dict[str, dict], keys: list[str]) -> pd.DataFrame:
    cols = {}
    for key in keys:
        d = pd.read_csv(methods[key]["path"] / "per_defect.csv")
        d = d[d["subset"] == "test"]
        cols[key] = d.groupby(["category", "defect_type"])["image_auroc"].mean()
    table = pd.DataFrame(cols)
    order = {c: i for i, c in enumerate(config.CATEGORIES)}
    return table.sort_index(key=lambda idx: idx.map(order) if idx.name == "category" else idx)


def defect_areas(split: pd.DataFrame | None = None) -> pd.DataFrame:
    split = load_split_file() if split is None else split
    rows = []
    for category in config.CATEGORIES:
        test = load_subset(category, "test", split)
        for i, label, mask in zip(test.ids, test.labels, test.masks):
            rows.append({"id": i, "category": category, "label": int(label), "area": float(mask.mean())})
    return pd.DataFrame(rows)


def size_edges(areas: pd.DataFrame) -> tuple[float, float]:
    defect = areas.loc[areas["label"] == 1, "area"]
    return float(defect.quantile(1 / 3)), float(defect.quantile(2 / 3))


def size_table(methods: dict[str, dict], keys: list[str], areas: pd.DataFrame) -> pd.DataFrame:
    lo, hi = size_edges(areas)
    sized = areas.assign(size=np.select([areas["area"] <= lo, areas["area"] <= hi], [0, 1], 2))
    rows = []
    for key in keys:
        scores = pd.read_csv(methods[key]["path"] / "scores.csv")
        scores = scores[scores["subset"] == "test"].merge(sized[["id", "category", "size"]], on="id")
        row = {"method": key}
        for size in range(3):
            per_seed = []
            for _, s in scores.groupby("seed"):
                aucs = []
                for _, c in s.groupby("category"):
                    good = c[c["label"] == 0]
                    bad = c[(c["label"] == 1) & (c["size"] == size)]
                    if len(good) and len(bad):
                        y = np.r_[np.zeros(len(good)), np.ones(len(bad))]
                        aucs.append(roc_auc_score(y, np.r_[good["score"], bad["score"]]))
                per_seed.append(np.mean(aucs))
            row[SIZE_NAMES_PL[size]] = float(np.mean(per_seed))
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def brightness_table(split: pd.DataFrame | None = None) -> pd.DataFrame:
    split = load_split_file() if split is None else split
    rows = []
    for category in config.CATEGORIES:
        row = {"category": category}
        for subset in ("train", "holdout", "val", "test"):
            data = load_subset(category, subset, split)
            gray = data.images[..., :3].astype(np.float64) @ GRAY_WEIGHTS
            means = gray.mean(axis=(1, 2))
            labels = np.asarray(data.labels)
            row[f"{subset}_good"] = float(means[labels == 0].mean()) if (labels == 0).any() else np.nan
            if subset in ("val", "test"):
                row[f"{subset}_defect"] = float(means[labels == 1].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _cell(value: float, best: bool) -> str:
    text = _num(value)
    return rf"\textbf{{{text}}}" if best else text


def _signed(x: float) -> str:
    return f"{x:+.3f}".replace(".", ",").replace("-", "$-$")


def latex_group_table(table: pd.DataFrame) -> str:
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l r r r r r}", r"\toprule",
             r"\thead{Metoda} & \thead{AUROC\\tekstury} & \thead{AUROC\\obiekty} & \thead{AUROC\\łącznie} & "
             r"\thead{Różnica\\tekstury -- obiekty} & \thead{AU-PRO} \\", r"\midrule"]
    previous = None
    for _, r in table.iterrows():
        if previous is not None and r["family"] != previous:
            lines.append(r"\midrule")
        previous = r["family"]
        pro = "--" if pd.isna(r["au_pro"]) else _num(r["au_pro"])
        lines.append(" & ".join([r["name"], _num(r["image_auroc_texture"]), _num(r["image_auroc_object"]),
                                 _num(r["image_auroc_all"]), _signed(r["gap"]), pro]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def category_rows(table: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for group, cats in (("texture", config.TEXTURES), ("object", config.OBJECTS)):
        rows = table.loc[list(cats)].copy()
        rows.index = [config.category_name_pl(c) for c in cats]
        mean = table.loc[list(cats)].mean().to_frame(f"średnia: {'tekstury' if group == 'texture' else 'obiekty'}").T
        parts += [rows, mean]
    parts.append(table.mean().to_frame("średnia").T)
    return pd.concat(parts)


def latex_size_table(table: pd.DataFrame, methods: dict[str, dict], edges: tuple[float, float]) -> str:
    lo, hi = (f"{100 * e:.2f}".replace(".", ",") for e in edges)
    heads = (rf"\thead{{małe\\($\le$\,{lo}\,\%)}}", rf"\thead{{średnie\\({lo}--{hi}\,\%)}}",
             rf"\thead{{duże\\($>$\,{hi}\,\%)}}")
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l r r r}", r"\toprule",
             rf"\thead{{Metoda}} & {' & '.join(heads)} \\", r"\midrule"]
    best = {c: table[c].round(3).max() for c in SIZE_NAMES_PL}
    for key, r in table.iterrows():
        lines.append(" & ".join([methods[key]["name"], *[_cell(r[c], round(r[c], 3) == best[c])
                                                           for c in SIZE_NAMES_PL]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def plot_categories(table: pd.DataFrame, methods: dict[str, dict], out_dir: Path) -> None:
    rows = category_rows(table)
    keys = list(rows.columns)
    values = rows.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 0.24 * len(rows) + 0.9))
    im = ax.imshow(values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    for i, row in enumerate(values):
        best = np.round(row, 3).max()
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.3f}".replace(".", ","), ha="center", va="center", fontsize=7,
                    fontweight="bold" if round(v, 3) == best else "normal",
                    color="white" if v <= 0.47 or v >= 0.93 else TEXT_PRIMARY)
    ax.set_yticks(range(len(rows)), rows.index, fontsize=7.5)
    for label in ax.get_yticklabels():
        if label.get_text().startswith("średnia"):
            label.set_fontstyle("italic")
    ax.set_xticks(range(len(keys)), [methods[k]["name"].replace(" (", "\n(") for k in keys], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    n_tex = len(config.TEXTURES)
    for y in (n_tex + 0.5, len(rows) - 1.5):
        ax.axhline(y, color="white", linewidth=3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("AUROC obrazów (zbiór testowy)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.yaxis.set_major_formatter(comma_formatter(1))
    _save(fig, out_dir, "porownanie_kategorie")


def label_positions(data: pd.DataFrame, gap: float = 0.0075, near: float = 0.1) -> dict[str, tuple]:
    logs = np.log10(data["test_time_per_image_ms"]).sort_values()
    groups, current = [], [logs.index[0]]
    for prev, key in zip(logs.index[:-1], logs.index[1:]):
        if logs[key] - logs[prev] > near:
            groups.append(current)
            current = []
        current.append(key)
    groups.append(current)
    out = {}
    for group in groups:
        previous = -np.inf
        for k in sorted(group, key=lambda k: data.loc[k, "image_auroc"]):
            y = data.loc[k, "image_auroc"]
            label_y = max(y, previous + gap)
            out[k] = (data.loc[k, "test_time_per_image_ms"], y, label_y)
            previous = label_y
    return out


def plot_quality_time(out_dir: Path, runs_dir: Path | None = None) -> pd.DataFrame:
    metrics, cost_metrics = collect_fewshot_runs(runs_dir)
    from .report_fewshot import FAMILY_OF, METHOD_NAMES_PL
    quality = fraction_table(metrics)[1.0]
    cost = cost_table(metrics, cost_metrics).set_index("method")
    data = cost.join(quality.rename("image_auroc"))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.4))
    for key, r in data.iterrows():
        ax.scatter(r["test_time_per_image_ms"], r["image_auroc"], s=26, color=FAMILY_COLORS[FAMILY_OF[key]],
                   zorder=3)
    for key, (x, y, label_y) in label_positions(data).items():
        ax.annotate(METHOD_NAMES_PL[key], (x, y), xytext=(x * 1.18, label_y), textcoords="data", fontsize=6.3,
                    color=TEXT_SECONDARY, va="center",
                    arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.6, "shrinkA": 0, "shrinkB": 3})
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(comma_formatter())
    ax.set_xlabel("czas oceny obrazu [ms] (skala logarytmiczna)")
    ax.set_ylabel("AUROC obrazów (zbiór testowy)")
    ax.grid(True, which="major", color=GRID)
    ax.set_axisbelow(True)
    lo, hi = data["test_time_per_image_ms"].min(), data["test_time_per_image_ms"].max()
    ax.set_xlim(10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi)) * 1.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=FAMILY_COLORS[f], label=FAMILY_NAMES_PL[f])
               for f in ("hand", "ae", "cnn", "pretrained")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7)
    _save(fig, out_dir, "porownanie_jakosc_czas")
    return data


def write_compare_report(tables_dir: Path | None = None, figures_dir: Path | None = None,
                         figures: bool = True) -> list[Path]:
    tables_dir = Path(tables_dir or config.TABLES_DIR)
    tables_dir.mkdir(parents=True, exist_ok=True)
    methods = collect_methods()
    if not methods:
        raise SystemExit("no runs found")
    keys = representatives(methods)
    groups = group_table(methods)
    categories = category_table(methods, keys)
    defects = defect_table(methods, keys)
    areas = defect_areas()
    sizes = size_table(methods, keys, areas)
    edges = size_edges(areas)
    brightness = brightness_table()
    frames = {"porownanie_grupy.csv": (groups, False), "porownanie_kategorie.csv": (categories, True),
              "porownanie_typy_defektow.csv": (defects, True),
              "porownanie_wielkosc.csv": (sizes.assign(edge_low=edges[0], edge_high=edges[1]), True),
              "jasnosc.csv": (brightness, False)}
    texts = {"tab_porownanie_grupy.tex": latex_group_table(groups),
             "tab_porownanie_wielkosc.tex": latex_size_table(sizes, methods, edges)}
    saved = []
    for name, (df, index) in frames.items():
        df.to_csv(tables_dir / name, index=index)
        saved.append(tables_dir / name)
    for name, text in texts.items():
        (tables_dir / name).write_text(text, encoding="utf-8")
        saved.append(tables_dir / name)
    if figures:
        setup_style()
        out = Path(figures_dir or config.FIGURES_DIR / "porownanie")
        plot_categories(categories, methods, out)
        plot_quality_time(out)
        saved += sorted(out.glob("*.pdf"))
    return saved
