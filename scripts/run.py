import argparse
from pathlib import Path

import yaml

from qcad.runner import cost_config, expand_fractions, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an experiment from a YAML config.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--categories", nargs="+", help="override categories")
    ap.add_argument("--fractions", nargs="+", type=float, help="override train_fraction")
    ap.add_argument("--cost", action="store_true", help="cost run: one seed, 100%%, no caches")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.categories:
        cfg["categories"] = args.categories
    if args.fractions:
        cfg["train_fraction"] = args.fractions
    if args.cost:
        cfg = cost_config(cfg)
    for run_cfg in expand_fractions(cfg):
        run_experiment(run_cfg)


if __name__ == "__main__":
    main()
