import argparse
from pathlib import Path

from qcad import config
from qcad import eda
from qcad.data import build_index, load_split_file
from qcad.split import summarize_split


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory data analysis of MVTec AD.")
    ap.add_argument("--root", type=Path, default=config.DATA_ROOT)
    args = ap.parse_args()

    eda.setup_style()
    tables, figs = config.TABLES_DIR, config.FIGURES_DIR / "eda"
    tables.mkdir(parents=True, exist_ok=True)

    index = build_index(args.root)
    stats = eda.image_stats(index, args.root)
    defects, regions = eda.defect_stats(index, args.root)

    table = eda.category_table(stats, defects)
    table.to_csv(tables / "eda_kategorie.csv", index=False)
    defect_counts = (index[index["label"] == 1].groupby(["category", "defect_type"]).size()
                     .rename("count").reset_index())
    defect_counts.to_csv(tables / "eda_typy_defektow.csv", index=False)
    defects.to_csv(tables / "eda_defekty_obrazy.csv", index=False)
    regions.to_csv(tables / "eda_defekty_obszary.csv", index=False)
    (tables / "tab_kategorie.tex").write_text(eda.latex_category_table(table, defect_counts), encoding="utf-8")

    crop = defects[defects["frac_inside_crop"] < 1.0]
    crop.to_csv(tables / "eda_wycinek_obciete.csv", index=False)

    if config.SPLIT_FILE.exists():
        summary = summarize_split(load_split_file())
        summary.to_csv(tables / "podzial_licznosci.csv")
        (tables / "tab_podzial.tex").write_text(eda.latex_split_table(summary), encoding="utf-8")

    eda.plot_counts(table, figs)
    eda.plot_defect_sizes(regions, figs)
    eda.plot_examples(index, figs, args.root)
    eda.plot_defect_types(index, ("carpet", "screw"), figs, args.root)
    eda.plot_crop_check(defects, index, figs, args.root)

    gray = sorted(stats.loc[stats["mode"] == "L", "category"].unique())
    print(table.to_string(index=False))
    print(f"\nmatches paper: {bool(table['matches_paper'].iloc[-1])}")
    print(f"grayscale: {', '.join(gray)}")
    print(f"defect area %: median {defects['defect_area_pct'].median():.2f}, "
          f"min {defects['defect_area_pct'].min():.4f}, max {defects['defect_area_pct'].max():.1f}")
    print(f"defect regions: {len(regions)}, median area {regions['region_area_pct'].median():.3f}%")
    print(f"crop: partial {len(crop)}, full {(defects['frac_inside_crop'] == 0).sum()}, "
          f"empty mask {(defects['defect_pixels_after_preproc'] == 0).sum()}")
    print(f"saved: {figs}, {tables}")


if __name__ == "__main__":
    main()
