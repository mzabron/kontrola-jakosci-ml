import argparse
from pathlib import Path

from qcad import config
from qcad.data import build_index
from qcad.eda import latex_split_table
from qcad.split import make_split, summarize_split


def main() -> None:
    ap = argparse.ArgumentParser(description="Create the data split.")
    ap.add_argument("--root", type=Path, default=config.DATA_ROOT)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing split")
    args = ap.parse_args()

    out = config.SPLITS_DIR / f"split_seed{args.seed}.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"split exists: {out} (use --force)")

    split = make_split(build_index(args.root), seed=args.seed)
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split.to_csv(out, index=False)

    summary = summarize_split(split)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.TABLES_DIR / "podzial_licznosci.csv")
    (config.TABLES_DIR / "tab_podzial.tex").write_text(latex_split_table(summary), encoding="utf-8")
    print(summary.to_string())
    print(f"saved: {out} ({len(split)} images)")


if __name__ == "__main__":
    main()
