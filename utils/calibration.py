from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import roc_curve


def truth_contains(truth: str, label: str) -> bool:
    parts = [p.strip() for p in str(truth).split(",") if p.strip()]
    return any(p == label for p in parts)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    # Remove NaNs
    ok = np.isfinite(scores)
    y_true = y_true[ok]
    scores = scores[ok]

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    J = tpr - fpr
    j_idx = int(np.argmax(J))
    thr = float(thresholds[j_idx])

    meta = {
        "youden_J": float(J[j_idx]),
        "tpr": float(tpr[j_idx]),
        "fpr": float(fpr[j_idx]),
    }
    return thr, meta


def bootstrap_youden_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    seed: int = 23,
) -> Tuple[float, Dict[str, Any]]:
    """Bootstrap Youden threshold on finalized scores.

    Deterministic given `seed`. Returns median threshold and a small summary payload.
    """

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    ok = np.isfinite(scores)
    y_true = y_true[ok]
    scores = scores[ok]

    if y_true.size == 0:
        raise ValueError("No valid scores for bootstrapping")

    rng = np.random.default_rng(seed)
    n = int(y_true.shape[0])

    thrs = []
    Js = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        y_boot = y_true[idx]
        # Skip degenerate resamples where only one class is present
        if len(np.unique(y_boot)) < 2:
            continue
        thr, meta = youden_threshold(y_boot, scores[idx])
        thrs.append(thr)
        Js.append(float(meta.get("youden_J", np.nan)))

    thrs = np.asarray(thrs, dtype=float)
    Js = np.asarray(Js, dtype=float)

    thr_med = float(np.nanmedian(thrs))
    ci_lo, ci_hi = np.nanpercentile(thrs, [2.5, 97.5]).astype(float)

    meta_out: Dict[str, Any] = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "threshold_median": thr_med,
        "threshold_ci95": [float(ci_lo), float(ci_hi)],
        "youden_J_median": float(np.nanmedian(Js)),
    }
    return thr_med, meta_out


def write_calibration_report(path_json: str | Path, payload: Dict[str, Any]) -> None:
    path_json = Path(path_json)
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with path_json.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
