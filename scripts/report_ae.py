import argparse

from qcad.report_ae import write_ae_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build autoencoder result tables and figures.")
    ap.add_argument("--no-figures", action="store_true", help="tables only")
    args = ap.parse_args()
    for path in write_ae_report(figures=not args.no_figures):
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
