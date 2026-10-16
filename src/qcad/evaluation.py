from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PRO_FPR_LIMIT = 0.3
PERCENTILE = 99.0


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def image_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if labels.max() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def select_threshold(scores: np.ndarray, labels: np.ndarray | None = None,
                     strategy: str = "f1", percentile: float = PERCENTILE) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if strategy == "f1":
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        return float(thresholds[np.argmax(f1[:-1])])
    if strategy == "percentile":
        good = scores if labels is None else scores[np.asarray(labels) == 0]
        return float(np.percentile(good, percentile))
    raise ValueError(f"unknown threshold strategy: {strategy}")


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = np.asarray(scores) >= threshold
    labels = np.asarray(labels).astype(bool)
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    tn = int(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": float(mcc),
        "accuracy": (tp + tn) / len(labels),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def per_defect_auroc(scores: np.ndarray, labels: np.ndarray, defect_types: np.ndarray) -> dict:
    scores, labels, defect_types = map(np.asarray, (scores, labels, defect_types))
    good = labels == 0
    out = {}
    for dt in sorted(set(defect_types[labels == 1])):
        sel = good | (defect_types == dt)
        out[dt] = image_auroc(scores[sel], labels[sel])
    return out


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    masks = np.asarray(masks, dtype=bool).ravel()
    if masks.all() or not masks.any():
        return float("nan")
    return float(roc_auc_score(masks, np.asarray(maps, dtype=np.float32).ravel()))


def pro_curve(maps: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maps = np.asarray(maps, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    structure = np.ones((3, 3), dtype=int)
    fp_changes = np.zeros(maps.shape, dtype=np.float64)
    pro_changes = np.zeros(maps.shape, dtype=np.float64)
    num_ok_pixels, num_regions = 0, 0
    for i, mask in enumerate(masks):
        labeled, n_comp = ndimage.label(mask, structure=structure)
        num_regions += n_comp
        ok = labeled == 0
        num_ok_pixels += int(ok.sum())
        fp_changes[i][ok] = 1.0
        for k in range(1, n_comp + 1):
            region = labeled == k
            pro_changes[i][region] = 1.0 / region.sum()
    if num_regions == 0 or num_ok_pixels == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    flat = maps.ravel()
    order = np.argsort(flat, kind="stable")[::-1]
    flat = flat[order]
    fprs = np.cumsum(fp_changes.ravel()[order]) / num_ok_pixels
    pros = np.cumsum(pro_changes.ravel()[order]) / num_regions
    keep = np.append(np.diff(flat) != 0, True)
    fprs = np.clip(fprs[keep], None, 1.0)
    pros = np.clip(pros[keep], None, 1.0)
    return np.concatenate(([0.0], fprs, [1.0])), np.concatenate(([0.0], pros, [1.0]))


def _area_up_to(x: np.ndarray, y: np.ndarray, x_max: float) -> float:
    inside = x <= x_max
    xs, ys = x[inside], y[inside]
    if xs[-1] < x_max:
        j = np.searchsorted(x, x_max, side="left")
        y_at = ys[-1] + (y[j] - ys[-1]) * (x_max - xs[-1]) / (x[j] - xs[-1])
        xs, ys = np.append(xs, x_max), np.append(ys, y_at)
    return float(np.trapezoid(ys, xs))


def au_pro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = PRO_FPR_LIMIT) -> float:
    if not np.asarray(masks, dtype=bool).any():
        return float("nan")
    fprs, pros = pro_curve(maps, masks)
    return _area_up_to(fprs, pros, fpr_limit) / fpr_limit


def evaluate(scores: np.ndarray, labels: np.ndarray, *,
             thresholds: dict[str, float] | None = None,
             maps: np.ndarray | None = None, masks: np.ndarray | None = None) -> dict:
    out = {
        "n_images": int(len(labels)),
        "n_defect": int(np.sum(labels)),
        "image_auroc": image_auroc(scores, labels),
        "image_ap": image_ap(scores, labels),
        "pixel_auroc": float("nan"),
        "au_pro": float("nan"),
    }
    if maps is not None and masks is not None:
        out["pixel_auroc"] = pixel_auroc(maps, masks)
        out["au_pro"] = au_pro(maps, masks)
    for name, thr in (thresholds or {}).items():
        for key, val in threshold_metrics(scores, labels, thr).items():
            out[f"{name}_{key}"] = val
    return out
