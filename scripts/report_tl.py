import argparse

from qcad.report_tl import write_tl_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build transfer learning result tables and figures.")
    ap.add_argument("--no-figures", action="store_true", help="tables only")
    args = ap.parse_args()
    for path in write_tl_report(figures=not args.no_figures):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
