import argparse

from qcad.report_fewshot import write_fewshot_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build training fraction and cost tables and figures.")
    ap.add_argument("--no-figures", action="store_true", help="tables only")
    args = ap.parse_args()
    for path in write_fewshot_report(figures=not args.no_figures):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
