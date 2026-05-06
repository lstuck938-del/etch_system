"""Unified data loading: long-table read + Tukey IQR filter + sample weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PATH, GROUP_KEYS, SHEET_NAME, TARGET


def tukey_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    if s.notna().sum() < 4:
        return pd.Series(True, index=s.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(True, index=s.index)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return s.between(lo, hi)


def load_long() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df = df.dropna(subset=["time_min", "etch_depth_nm"]).reset_index(drop=True)
    df = df[df["time_min"] > 0].reset_index(drop=True)
    df[TARGET] = df["etch_depth_nm"] / df["time_min"]

    keep = df.groupby(GROUP_KEYS, dropna=False)[TARGET].transform(tukey_mask)
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"Tukey filter dropped {n_drop} rows ({n_drop / len(df):.1%}) within {'+'.join(GROUP_KEYS)}")
    df = df[keep].reset_index(drop=True)

    rep_counts = df.groupby(GROUP_KEYS + ["position"], dropna=False)[TARGET].transform("count")
    df["sample_weight"] = 1.0 / rep_counts
    return df


def make_groups(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    s = df[GROUP_KEYS].astype(str).agg("|".join, axis=1)
    return s.to_numpy(), int(s.nunique())
