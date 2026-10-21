import argparse
from pathlib import Path

import yaml

from qcad.runner import run_experiment
from qcad.tuning import tune


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid search on the validation set.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--categories", nargs="+", help="override categories")
    ap.add_argument("--run", action="store_true", help="run final experiments with the best settings")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.categories:
        cfg["categories"] = args.categories
    _, finals = tune(cfg)
    if args.run:
        for final in finals:
            run_experiment(final)


if __name__ == "__main__":
    main()
