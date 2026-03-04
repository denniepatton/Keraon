from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ReferenceModel:
    """Single reusable artifact for Keraon inference and calibration."""

    df_train: pd.DataFrame
    scaling_params: Dict[str, Any]

    feature_selection: Dict[str, Any]
    ctdpheno_gda: Dict[str, Any]
    keraon: Dict[str, Any]

    calibration: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "df_train": self.df_train,
            "scaling_params": self.scaling_params,
            "feature_selection": self.feature_selection,
            "ctdpheno_gda": self.ctdpheno_gda,
            "keraon": self.keraon,
            "calibration": self.calibration,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ReferenceModel":
        return ReferenceModel(
            df_train=payload["df_train"],
            scaling_params=payload["scaling_params"],
            feature_selection=payload.get("feature_selection", {}),
            ctdpheno_gda=payload.get("ctdpheno_gda", {}),
            keraon=payload.get("keraon", {}),
            calibration=payload.get("calibration"),
        )


def save_reference_model(model: ReferenceModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_reference_model(path: str | Path) -> ReferenceModel:
    """Load a current `reference_model.pickle` produced by this codebase."""
    path = Path(path)
    with path.open("rb") as f:
        obj = pickle.load(f)

    # New format: dict with required keys
    if isinstance(obj, dict) and "df_train" in obj and "scaling_params" in obj:
        return ReferenceModel.from_dict(obj)

    raise ValueError(f"Unrecognized reference model pickle format: {path}")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_tsv(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=True)
