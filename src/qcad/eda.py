from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image
from scipy import ndimage
from skimage.measure import find_contours

from . import config
from .data import load_image, load_mask

REFERENCE_COUNTS = {
    "carpet": (280, 28, 89, 5, 97), "grid": (264, 21, 57, 5, 170),
    "leather": (245, 32, 92, 5, 99), "tile": (230, 33, 84, 5, 86),
    "wood": (247, 19, 60, 5, 168), "bottle": (209, 20, 63, 3, 68),
    "cable": (224, 58, 92, 8, 151), "capsule": (219, 23, 109, 5, 114),
    "hazelnut": (391, 40, 70, 4, 136), "metal_nut": (220, 22, 93, 4, 132),
    "pill": (267, 26, 141, 7, 245), "screw": (320, 41, 119, 5, 135),
    "toothbrush": (60, 12, 30, 1, 66), "transistor": (213, 60, 40, 4, 44),
    "zipper": (240, 32, 119, 7, 177),
}

COLOR_TRAIN = "#2a78d6"
COLOR_GOOD = "#1baf7a"
COLOR_DEFECT = "#eb6834"
COLOR_TEXTURE = "#2a78d6"
COLOR_OBJECT = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"
TEXT_WIDTH_IN = 6.3
TITLE_WRAP = 15


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out_dir / f"{name}.png", metadata={"Software": None})
    plt.close(fig)


def _pl(category: str) -> str:
    return config.category_name_pl(category)


def _defect_label(defect_type: str) -> str:
    return config.defect_name_pl(defect_type)


def image_stats(index: pd.DataFrame, root: Path = config.DATA_ROOT) -> pd.DataFrame:
    sizes, modes = [], []
    for p in index["image_path"]:
        with Image.open(root / p) as img:
            sizes.append(img.size)
            modes.append(img.mode)
    out = index.copy()
    out["width"] = [s[0] for s in sizes]
    out["height"] = [s[1] for s in sizes]
    out["mode"] = modes
    return out


def defect_stats(index: pd.DataFrame, root: Path = config.DATA_ROOT,
                 resize: int = config.RESIZE, crop: int = config.CROP) -> tuple[pd.DataFrame, pd.DataFrame]:
    structure = np.ones((3, 3), dtype=int)
    lo, hi = (resize - crop) / 2 / resize, (resize + crop) / 2 / resize
    img_rows, reg_rows = [], []
    for _, r in index[index["label"] == 1].iterrows():
        with Image.open(root / r["mask_path"]) as m:
            mask = np.asarray(m.convert("L")) > 127
        h, w = mask.shape
        labeled, n_reg = ndimage.label(mask, structure=structure)
        area = mask.sum()
        y0, y1, x0, x1 = int(round(lo * h)), int(round(hi * h)), int(round(lo * w)), int(round(hi * w))
        inside = mask[y0:y1, x0:x1].sum()
        mask_pp = load_mask(root / r["mask_path"], resize, crop)
        _, n_reg_pp = ndimage.label(mask_pp, structure=structure)
        img_rows.append({
            "id": r["id"], "category": r["category"], "defect_type": r["defect_type"],
            "defect_area_pct": 100 * area / mask.size, "n_regions": n_reg,
            "frac_inside_crop": inside / area if area else np.nan,
            "n_regions_after_preproc": n_reg_pp, "defect_pixels_after_preproc": int(mask_pp.sum()),
        })
        if n_reg:
            sizes = ndimage.sum_labels(mask, labeled, index=np.arange(1, n_reg + 1))
            for s in sizes:
                reg_rows.append({"category": r["category"], "defect_type": r["defect_type"],
                                 "region_area_pct": 100 * s / mask.size})
    return pd.DataFrame(img_rows), pd.DataFrame(reg_rows)


def category_table(stats: pd.DataFrame, defects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in config.CATEGORIES:
        s = stats[stats["category"] == cat]
        d = defects[defects["category"] == cat]
        train = int(((s["source_split"] == "train")).sum())
        test_good = int(((s["source_split"] == "test") & (s["label"] == 0)).sum())
        test_def = int(((s["source_split"] == "test") & (s["label"] == 1)).sum())
        n_types = s.loc[s["label"] == 1, "defect_type"].nunique()
        n_reg = int(d["n_regions"].sum())
        ref = REFERENCE_COUNTS[cat]
        rows.append({
            "category": cat, "category_pl": _pl(cat), "category_type": config.category_type(cat),
            "train_good": train, "test_good": test_good, "test_defect": test_def,
            "n_defect_types": n_types, "n_regions": n_reg,
            "image_size": "/".join(sorted({str(w) for w in s["width"]})),
            "mode": "/".join(sorted(set(s["mode"]))),
            "matches_paper": (train, test_good, test_def, n_types, n_reg) == ref,
        })
    table = pd.DataFrame(rows)
    total = {"category": "total", "category_pl": "suma", "category_type": "",
             **{c: int(table[c].sum()) for c in ("train_good", "test_good", "test_defect",
                                                  "n_defect_types", "n_regions")},
             "image_size": "", "mode": "", "matches_paper": bool(table["matches_paper"].all())}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


def _category_order() -> list[str]:
    return list(config.TEXTURES) + list(config.OBJECTS)


def _group_labels(ax, y: np.ndarray) -> None:
    n_tex = len(config.TEXTURES)
    for name, ys in (("tekstury", y[:n_tex]), ("obiekty", y[n_tex:])):
        ax.text(-0.33, ys.mean(), name, transform=ax.get_yaxis_transform(), rotation=90,
                va="center", ha="center", color=TEXT_SECONDARY)


def plot_counts(table: pd.DataFrame, out_dir: Path) -> None:
    t = table.set_index("category").loc[_category_order()]
    labels = [_pl(c) for c in t.index]
    y = np.arange(len(t))[::-1].astype(float)
    y[len(config.TEXTURES):] -= 0.6
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.2))
    left = np.zeros(len(t))
    for col, color, name in (("train_good", COLOR_TRAIN, "uczące (dobre)"),
                             ("test_good", COLOR_GOOD, "testowe dobre"),
                             ("test_defect", COLOR_DEFECT, "testowe wadliwe")):
        ax.barh(y, t[col], left=left, height=0.72, color=color, label=name,
                edgecolor="white", linewidth=1.0)
        left += t[col].to_numpy()
    for yi, total in zip(y, left):
        ax.text(total + 5, yi, f"{int(total)}", va="center", fontsize=7.5, color=TEXT_SECONDARY)
    ax.set_yticks(y, labels)
    ax.set_xlabel("liczba obrazów")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(0, left.max() * 1.08)
    _group_labels(ax, y)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    _save(fig, out_dir, "eda_liczebnosc_kategorii")


def plot_defect_sizes(regions: pd.DataFrame, out_dir: Path) -> None:
    order = _category_order()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.2))
    data = [regions.loc[regions["category"] == c, "region_area_pct"].to_numpy() for c in order]
    pos = np.arange(len(order))[::-1].astype(float)
    pos[len(config.TEXTURES):] -= 0.6
    bp = ax.boxplot(data, positions=pos, orientation="horizontal", widths=0.6, patch_artist=True,
                    showfliers=True, flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.5,
                                                 "markeredgewidth": 0},
                    medianprops={"color": TEXT_PRIMARY, "linewidth": 1.2},
                    whiskerprops={"color": TEXT_SECONDARY}, capprops={"color": TEXT_SECONDARY})
    for patch, flier, cat in zip(bp["boxes"], bp["fliers"], order):
        color = COLOR_TEXTURE if cat in config.TEXTURES else COLOR_OBJECT
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(color)
        flier.set_markerfacecolor(color)
    ax.set_yticks(pos, [_pl(c) for c in order])
    ax.set_xscale("log")
    ax.set_xlabel("pole obszaru defektu [% powierzchni obrazu] (skala logarytmiczna)")
    ax.xaxis.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=COLOR_TEXTURE, label="tekstury"),
                       Patch(color=COLOR_OBJECT, label="obiekty")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    _save(fig, out_dir, "eda_wielkosc_defektow")


def _show(ax, img, mask=None, title=None):
    ax.imshow(img)
    if mask is not None and mask.any():
        for c in find_contours(mask.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=COLOR_DEFECT, linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        title = "\n".join(textwrap.fill(line, TITLE_WRAP, break_long_words=False, break_on_hyphens=False)
                          for line in title.split("\n"))
        ax.set_title(title, fontsize=7.5, pad=2, color=TEXT_PRIMARY, linespacing=1.0)


def plot_examples(index: pd.DataFrame, out_dir: Path, root: Path = config.DATA_ROOT, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    order = _category_order()
    fig, axes = plt.subplots(5, 6, figsize=(TEXT_WIDTH_IN, 6.8))
    for k, cat in enumerate(order):
        rows = index[(index["category"] == cat) & (index["source_split"] == "test")]
        good = rows[rows["label"] == 0].iloc[rng.integers(0, (rows["label"] == 0).sum())]
        bad_rows = rows[rows["label"] == 1]
        bad = bad_rows.iloc[rng.integers(0, len(bad_rows))]
        ax_g, ax_b = axes[k // 3, 2 * (k % 3)], axes[k // 3, 2 * (k % 3) + 1]
        _show(ax_g, load_image(root / good["image_path"]), title=_pl(cat))
        _show(ax_b, load_image(root / bad["image_path"]), load_mask(root / bad["mask_path"]),
              title=_defect_label(bad["defect_type"]))
    fig.subplots_adjust(wspace=0.04, hspace=0.45)
    _save(fig, out_dir, "eda_przyklady_kategorii")


def plot_defect_types(index: pd.DataFrame, categories: tuple[str, ...], out_dir: Path,
                      root: Path = config.DATA_ROOT, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    per_cat = []
    for cat in categories:
        rows = index[(index["category"] == cat) & (index["source_split"] == "test")]
        types = ["good"] + sorted(rows.loc[rows["label"] == 1, "defect_type"].unique())
        per_cat.append((cat, rows, types))
    ncol = max(len(t) for _, _, t in per_cat)
    fig, axes = plt.subplots(len(categories), ncol, figsize=(TEXT_WIDTH_IN, 1.45 * len(categories) + 0.2),
                             squeeze=False)
    for r, (cat, rows, types) in enumerate(per_cat):
        for c in range(ncol):
            ax = axes[r, c]
            if c >= len(types):
                ax.axis("off")
                continue
            sel = rows[rows["defect_type"] == types[c]]
            ex = sel.iloc[rng.integers(0, len(sel))]
            mask = load_mask(root / ex["mask_path"]) if ex["mask_path"] else None
            title = _defect_label(types[c])
            _show(ax, load_image(root / ex["image_path"]), mask, title=title)
        axes[r, 0].set_ylabel(_pl(cat), fontsize=8, color=TEXT_PRIMARY)
        axes[r, 0].yaxis.label.set_visible(True)
    fig.subplots_adjust(wspace=0.04, hspace=0.4)
    _save(fig, out_dir, "eda_typy_defektow")


def plot_crop_check(defects: pd.DataFrame, index: pd.DataFrame, out_dir: Path,
                    root: Path = config.DATA_ROOT, n: int = 4) -> None:
    worst = defects.sort_values("frac_inside_crop").head(n)
    if worst.empty:
        return
    fig, axes = plt.subplots(1, len(worst), figsize=(TEXT_WIDTH_IN, 1.9), squeeze=False)
    for ax, (_, r) in zip(axes[0], worst.iterrows()):
        row = index[index["id"] == r["id"]].iloc[0]
        with Image.open(root / row["image_path"]) as img:
            full = np.asarray(img.convert("RGB").resize((config.RESIZE, config.RESIZE)))
        with Image.open(root / row["mask_path"]) as m:
            mask = np.asarray(m.convert("L").resize((config.RESIZE, config.RESIZE), Image.Resampling.NEAREST)) > 127
        name = f"{_pl(r['category'])} / {_defect_label(r['defect_type'])}"
        _show(ax, full, mask, title=f"{name}\nw wycinku: {100 * r['frac_inside_crop']:.0f}%")
        off = (config.RESIZE - config.CROP) / 2
        ax.add_patch(plt.Rectangle((off, off), config.CROP, config.CROP, fill=False,
                                   edgecolor=TEXT_PRIMARY, linewidth=1.0, linestyle="--"))
    _save(fig, out_dir, "eda_wycinek_a_defekty")


def latex_category_table(table: pd.DataFrame, defect_counts: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabularx}{\textwidth}{l l r r r r r >{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\thead{Kategoria} & \thead{Typ} & \thead{Uczące\\(dobre)} & \thead{Test\\dobre} & "
        r"\thead{Test\\wadliwe} & \thead{Obszary\\defektów} & \thead{Bok\\{[}px{]}} & \thead{Typy defektów} \\",
        r"\midrule",
    ]
    for _, r in table.iterrows():
        if r["category"] == "total":
            lines.append(r"\midrule")
            lines.append(rf"\textbf{{Suma}} & & {r['train_good']} & {r['test_good']} & {r['test_defect']} & "
                         rf"{r['n_regions']} & & {r['n_defect_types']} typy \\")
            continue
        types = defect_counts[defect_counts["category"] == r["category"]]
        type_list = ", ".join(f"{_defect_label(t)} ({n})" for t, n in zip(types["defect_type"], types["count"]))
        gray = r"\textsuperscript{*}" if r["mode"] == "L" else ""
        type_pl = config.category_type_name_pl(r["category_type"])
        lines.append(rf"{_pl(r['category'])}{gray} & {type_pl} & {r['train_good']} & "
                     rf"{r['test_good']} & {r['test_defect']} & {r['n_regions']} & {r['image_size']} & "
                     rf"{type_list} \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(lines) + "\n"


def latex_split_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"\thead{Kategoria} & \thead{Uczący} & \thead{Odłożony} & \thead{Walidacja\\dobre} & "
        r"\thead{Walidacja\\wadliwe} & \thead{Test\\dobre} & \thead{Test\\wadliwe} \\",
        r"\midrule",
    ]
    for cat, r in summary.iterrows():
        if cat == "total":
            lines.append(r"\midrule")
            name = r"\textbf{Suma}"
        else:
            name = _pl(cat)
        lines.append(rf"{name} & {r['train']} & {r['holdout']} & {r['val_good']} & {r['val_defect']} & "
                     rf"{r['test_good']} & {r['test_defect']} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
