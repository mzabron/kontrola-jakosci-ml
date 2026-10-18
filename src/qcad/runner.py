from __future__ import annotations

import hashlib
import inspect
import json
import platform
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import load_split_file, load_subset
from .evaluation import evaluate, per_defect_auroc, select_threshold
from .methods import METHODS, create_method


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _timed(fn, *args):
    _sync_cuda()
    t0 = time.perf_counter()
    out = fn(*args)
    _sync_cuda()
    return out, time.perf_counter() - t0


def _environment(split_file: Path) -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import sklearn
        import torch

        env.update(torch=torch.__version__, sklearn=sklearn.__version__,
                   cuda=torch.cuda.is_available(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=config.PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip() != ""
        env.update(git_commit=commit, git_dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        env.update(git_commit=None)
    env["split_file"] = str(split_file.relative_to(config.PROJECT_ROOT)) \
        if split_file.is_relative_to(config.PROJECT_ROOT) else str(split_file)
    env["split_sha256"] = hashlib.sha256(split_file.read_bytes()).hexdigest()
    return env


def run_experiment(cfg: dict) -> Path:
    method_name = cfg["method"]
    params = dict(cfg.get("params") or {})
    categories = config.CATEGORIES if cfg.get("categories", "all") == "all" else tuple(cfg["categories"])
    seeds = list(cfg.get("seeds", [0]))
    train_fraction = float(cfg.get("train_fraction", 1.0))
    split_file = Path(cfg.get("split_file", config.SPLIT_FILE))
    split = load_split_file(split_file)

    accepts_seed = "seed" in inspect.signature(METHODS[method_name].__init__).parameters
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{method_name}" + (f"_{cfg['tag']}" if cfg.get("tag") else "")
    out_dir = config.RESULTS_DIR / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows, defect_rows, score_rows, method_cfg = [], [], [], {}
    for seed in seeds:
        for category in categories:
            set_seed(seed)
            method = create_method(method_name, **({**params, "seed": seed} if accepts_seed else params))
            method_cfg = method.get_config()
            train = load_subset(category, "train", split, train_fraction)
            holdout = load_subset(category, "holdout", split)
            val = load_subset(category, "val", split)
            test = load_subset(category, "test", split)

            _, fit_time = _timed(method.fit, train.images)
            pred_hold = method.predict(holdout.images)
            pred_val = method.predict(val.images)
            pred_test, test_time = _timed(method.predict, test.images)

            thresholds = {
                "f1": select_threshold(pred_val.scores, val.labels, "f1"),
                "pct": select_threshold(pred_hold.scores, None, "percentile"),
            }
            common = {"seed": seed, "category": category,
                      "category_type": config.category_type(category),
                      "n_train": len(train), "fit_time_s": fit_time,
                      "test_time_per_image_ms": 1000 * test_time / len(test)}
            for subset, data, pred in (("val", val, pred_val), ("test", test, pred_test)):
                m = evaluate(pred.scores, data.labels, thresholds=thresholds,
                             maps=pred.maps, masks=data.masks)
                metric_rows.append({**common, "subset": subset, **m})
                for dt, auc in per_defect_auroc(pred.scores, data.labels, data.defect_types).items():
                    defect_rows.append({"seed": seed, "category": category, "subset": subset,
                                        "defect_type": dt, "image_auroc": auc,
                                        "n_defect": int(np.sum(data.defect_types == dt))})
                for i, s, lab, dt in zip(data.ids, pred.scores, data.labels, data.defect_types):
                    score_rows.append({"seed": seed, "id": i, "subset": subset,
                                       "label": int(lab), "defect_type": dt, "score": float(s)})
            for i, s in zip(holdout.ids, pred_hold.scores):
                score_rows.append({"seed": seed, "id": i, "subset": "holdout",
                                   "label": 0, "defect_type": "good", "score": float(s)})
            test_row = metric_rows[-1]
            print(f"[{method_name} seed={seed}] {category:<11} image_auroc={test_row['image_auroc']:.3f} "
                  f"pixel_auroc={test_row['pixel_auroc']:.3f} au_pro={test_row['au_pro']:.3f} "
                  f"f1={test_row['f1_f1']:.3f} mcc={test_row['f1_mcc']:.3f}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(defect_rows).to_csv(out_dir / "per_defect.csv", index=False)
    pd.DataFrame(score_rows).to_csv(out_dir / "scores.csv", index=False)
    run_info = {"run_id": run_id, "config": cfg, "method_config": method_cfg,
                "train_fraction": train_fraction, "seeds": seeds,
                "categories": list(categories), "environment": _environment(split_file)}
    (out_dir / "config.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarize(metrics, run_id, method_name)
    summary_path = config.RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, mode="a", index=False, header=not summary_path.exists())
    print("\ntest set means:\n" + summary.to_string(index=False))
    print(f"\nsaved: {out_dir}")
    return out_dir


SUMMARY_METRICS = ["image_auroc", "image_ap", "pixel_auroc", "au_pro",
                   "f1_f1", "f1_mcc", "f1_balanced_accuracy", "pct_f1", "pct_mcc", "pct_balanced_accuracy",
                   "fit_time_s", "test_time_per_image_ms"]


def summarize(metrics: pd.DataFrame, run_id: str, method_name: str) -> pd.DataFrame:
    test = metrics[metrics["subset"] == "test"]
    rows = []
    for group, sel in (("texture", test["category_type"] == "texture"),
                       ("object", test["category_type"] == "object"),
                       ("all", slice(None))):
        part = test[sel]
        if part.empty:
            continue
        per_seed = part.groupby("seed")[SUMMARY_METRICS].mean()
        row = {"run_id": run_id, "method": method_name, "group": group,
               "n_categories": part["category"].nunique(), "n_seeds": len(per_seed)}
        for m in SUMMARY_METRICS:
            row[m] = per_seed[m].mean()
            row[f"{m}_std"] = per_seed[m].std(ddof=1) if len(per_seed) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
