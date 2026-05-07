"""Shared paths, feature lists, CV/seed settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "刻蚀实验_建模数据表.xlsx"
SHEET_NAME = "长表(Long)"
OUTPUT_ROOT = ROOT / "outputs"

RANDOM_SEED = 20260896
N_SPLITS = 5

TARGET = "etch_rate_nm_per_min"
GROUP_KEYS = ["discharge_condition", "substrate"]
NUMERIC = ["pressure_Pa", "power_W", "time_min", "Ar_frac", "O2_frac", "CF4_frac"]
CATEGORICAL = ["substrate", "position", "photoresist"]
FEATURES = NUMERIC + CATEGORICAL
