from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_roc_pdf(y_true: np.ndarray, scores: np.ndarray, title: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    ok = np.isfinite(scores)
    y_true = y_true[ok]
    scores = scores[ok]

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(fpr, tpr, color="#0072B2", lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_score_hist(
    scores: np.ndarray,
    y_true: np.ndarray,
    title: str,
    xlabel: str,
    out_path: str | Path,
    *,
    neg_label: str = "negative",
    pos_label: str = "positive",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    ok = np.isfinite(scores)
    y_true = y_true[ok]
    scores = scores[ok]

    s0 = scores[y_true == 0]
    s1 = scores[y_true == 1]

    plt.figure(figsize=(6.2, 4.6))
    bins = 30
    plt.hist(s0, bins=bins, alpha=0.6, label=str(neg_label), color="#D55E00", density=True)
    plt.hist(s1, bins=bins, alpha=0.6, label=str(pos_label), color="#009E73", density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
