import argparse
from pathlib import Path

import yaml

from qcad.runner import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an experiment from a YAML config.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--categories", nargs="+", help="override categories")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.categories:
        cfg["categories"] = args.categories
    run_experiment(cfg)


if __name__ == "__main__":
    main()
