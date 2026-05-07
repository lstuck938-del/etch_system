"""Metrics, stratified group CV splits, and unified IO for outputs/{model}/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import (
    GROUP_KEYS,
    N_SPLITS,
    OUTPUT_ROOT,
    RANDOM_SEED,
    TARGET,
)

N_TARGET_BINS = 5


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def output_dir(model_name: str) -> Path:
    d = OUTPUT_ROOT / model_name
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def _target_bins(y: np.ndarray, n_bins: int = N_TARGET_BINS) -> pd.Series:
    """Quantile-bin the continuous target for stratified regression CV."""
    y_s = pd.Series(np.asarray(y, dtype=float))
    try:
        binned = pd.qcut(y_s, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        binned = pd.cut(y_s, bins=min(n_bins, max(int(y_s.nunique()), 1)), labels=False)
    return binned.fillna(0).astype(int).astype(str)


def _collapse_rare(labels: pd.Series, fallback: pd.Series, min_count: int = N_SPLITS) -> pd.Series:
    """Replace labels with fallback where a stratum is too sparse."""
    counts = labels.value_counts()
    rare = labels.map(counts).fillna(0) < min_count
    out = labels.copy()
    out.loc[rare] = fallback.loc[rare]
    return out.astype(str)


def stratification_labels(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Build robust labels: target-bin + material + time, with rare strata collapsed.

    The primary label follows the requested target-bin/material/time stratification.
    Because GroupKFold keeps whole process-condition groups together, some exact
    combinations can be too rare; those fall back to target-bin+material, then
    target-bin, preserving the most important regression balance.
    """
    target_bin = _target_bins(y)
    material = df.get("substrate", pd.Series("unknown", index=df.index)).astype(str).reset_index(drop=True)
    if "time_min" in df.columns:
        time = df["time_min"].astype(str).reset_index(drop=True)
    else:
        time = pd.Series("unknown", index=df.index)

    by_bin = target_bin
    by_bin_material = target_bin + "|mat=" + material
    full = by_bin_material + "|t=" + time
    labels = _collapse_rare(full, by_bin_material)
    labels = _collapse_rare(labels, by_bin)
    counts = labels.value_counts()
    rare = labels.map(counts).fillna(0) < N_SPLITS
    labels.loc[rare] = "rare"
    return labels.to_numpy()


def cv_split(df: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = stratification_labels(df.reset_index(drop=True), y)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    return list(splitter.split(df, labels, groups))


def save_oof_csv(
    df: pd.DataFrame,
    oof: np.ndarray,
    model_name: str,
    out_dir: Path,
    extra_cols: dict[str, np.ndarray] | None = None,
) -> None:
    keep_cols = GROUP_KEYS + ["position", "time_min", "etch_depth_nm", TARGET]
    out = df[keep_cols].copy()
    out[f"pred_{model_name}"] = oof
    if extra_cols:
        for k, v in extra_cols.items():
            out[k] = v
    out.to_csv(out_dir / "oof.csv", index=False, encoding="utf-8-sig")


def save_metrics_json(out_dir: Path, payload: dict) -> None:
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
