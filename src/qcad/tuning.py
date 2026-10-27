from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from . import config
from .data import load_split_file, load_subset
from .evaluation import image_ap, image_auroc
from .runner import build_method, fit_method, set_seed

FINAL_KEYS = ("method", "categories", "seeds", "train_fraction", "split_file")


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _label(value) -> str:
    return "-".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)


def _key(value) -> str:
    return json.dumps(value, sort_keys=True)


def tune(cfg: dict) -> tuple[Path, list[dict]]:
    method_name = cfg["method"]
    base = dict(cfg.get("params") or {})
    grid = dict(cfg.get("grid") or {})
    select_per = list(cfg.get("select_per") or [])
    missing = [k for k in select_per if k not in grid]
    if missing:
        raise ValueError(f"select_per keys not in grid: {missing}")
    metric = cfg.get("select_metric", "image_auroc")
    categories = config.CATEGORIES if cfg.get("categories", "all") == "all" else tuple(cfg["categories"])
    seeds = list(cfg.get("tune_seeds", [0]))
    train_fraction = float(cfg.get("train_fraction", 1.0))
    split = load_split_file(Path(cfg.get("split_file", config.SPLIT_FILE)))
    combos = expand_grid(grid)

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{method_name}" + (f"_{cfg['tag']}" if cfg.get("tag") else "")
    out_dir = config.RESULTS_DIR / "tuning" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for category in categories:
        train = load_subset(category, "train", split, train_fraction)
        holdout = load_subset(category, "holdout", split)
        val = load_subset(category, "val", split)
        for ci, combo in enumerate(combos):
            for seed in seeds:
                set_seed(seed)
                method = build_method(method_name, {**base, **combo}, seed, category)
                scores = fit_method(method, train.images, holdout.images).predict(val.images).scores
                rows.append({"combo": ci, **{k: _key(v) for k, v in combo.items()},
                             "seed": seed, "category": category,
                             "category_type": config.category_type(category),
                             "image_auroc": image_auroc(scores, val.labels),
                             "image_ap": image_ap(scores, val.labels)})
        best = max((r for r in rows if r["category"] == category), key=lambda r: r[metric])
        print(f"[tune {method_name}] {category:<11} best val {metric}={best[metric]:.3f} (combo {best['combo']})")

    val_grid = pd.DataFrame(rows)
    val_grid.to_csv(out_dir / "val_grid.csv", index=False)
    summary = summarize_grid(val_grid, list(grid), metric)
    summary.to_csv(out_dir / "val_summary.csv", index=False)

    finals = []
    groups = summary.groupby(select_per, sort=False) if select_per else [((), summary)]
    for _, part in groups:
        best = part.sort_values([f"{metric}_all", "combo"], ascending=[False, True]).iloc[0]
        combo = combos[int(best["combo"])]
        final = {k: cfg[k] for k in FINAL_KEYS if k in cfg}
        final["params"] = {**base, **combo}
        suffix = "_".join(_label(combo[k]) for k in select_per)
        final["tag"] = "_".join(t for t in (cfg.get("tag"), suffix) if t)
        final["tuning"] = {"run_id": run_id, "combo": int(best["combo"]), "metric": metric,
                           "val_all": float(best[f"{metric}_all"]),
                           "val_texture": float(best[f"{metric}_texture"]),
                           "val_object": float(best[f"{metric}_object"])}
        path = out_dir / f"best_{final['tag'] or method_name}.yaml"
        path.write_text(yaml.safe_dump(final, sort_keys=False, allow_unicode=True), encoding="utf-8")
        finals.append(final)
        print(f"best ({final['tag']}): {combo} val {metric}={best[f'{metric}_all']:.3f} -> {path.name}")
    (out_dir / "config.json").write_text(json.dumps({"run_id": run_id, "config": cfg}, indent=2,
                                                    ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_dir}")
    return out_dir, finals


def summarize_grid(val_grid: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    per_cat = val_grid.groupby(["combo", *keys, "category", "category_type"], sort=False)[metric].mean().reset_index()
    out = per_cat.groupby(["combo", *keys], sort=False)[metric].mean().rename(f"{metric}_all").to_frame()
    for group in ("texture", "object"):
        part = per_cat[per_cat["category_type"] == group]
        out[f"{metric}_{group}"] = part.groupby(["combo", *keys], sort=False)[metric].mean()
    out = out.reset_index()
    return out.sort_values(f"{metric}_all", ascending=False)
