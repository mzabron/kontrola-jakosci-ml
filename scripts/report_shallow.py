import argparse

from qcad.report import write_shallow_tables


def main() -> None:
    argparse.ArgumentParser(description="Build shallow-method result tables.").parse_args()
    for path in write_shallow_tables():
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
