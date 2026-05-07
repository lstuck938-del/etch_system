"""Small HTTP API for the PIML CCP etch dashboard.

The training script currently writes OOF predictions/metrics but does not save a
reloadable PyTorch checkpoint. This service therefore exposes a data-calibrated
predictor over the processed experiment table and the saved PIML metrics, so the
frontend talks to real backend data instead of its previous in-browser mock.
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    from .config import OUTPUT_ROOT, TARGET
    from .data import load_long
except ImportError:  # Allows: python backend/src/api.py
    BACKEND_ROOT = Path(__file__).resolve().parent.parent
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    from src.config import OUTPUT_ROOT, TARGET
    from src.data import load_long


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_HTML = PROJECT_ROOT / "frontend" / "PIML CCP Etch Dashboard.html"
MODEL_NAME = "piml"
NUMERIC = ["pressure_Pa", "power_W", "time_min", "Ar_frac", "O2_frac", "CF4_frac"]
MATERIALS = ["Si", "SiO2", "SiN"]
POSITIONS = ["center", "edge"]
OOF_PATH = OUTPUT_ROOT / MODEL_NAME / "oof.csv"


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, fallback: float) -> float:
    try:
        if value is None or value == "":
            return fallback
        out = float(value)
        return out if math.isfinite(out) else fallback
    except (TypeError, ValueError):
        return fallback


def _as_fraction(value: Any, fallback: float) -> float:
    out = _safe_float(value, fallback)
    if out > 1.0:
        out /= 100.0
    return min(1.0, max(0.0, out))


def _normalize_position(value: Any) -> str:
    text = str(value or "center").strip().lower()
    if text in {"edge", "边缘"}:
        return "edge"
    return "center"


def _normalize_material(value: Any) -> str:
    text = str(value or "SiO2").replace("₂", "2").strip()
    return text if text in MATERIALS else "SiO2"


def _feature_matrix(df: pd.DataFrame, pred_rate: np.ndarray | None = None) -> np.ndarray:
    num = df[NUMERIC].astype(float).to_numpy()
    mat = df["substrate"].map(_normalize_material)
    pos = df["position"].map(_normalize_position)
    mat_oh = np.column_stack([(mat == m).astype(float).to_numpy() for m in MATERIALS])
    pos_edge = (pos == "edge").astype(float).to_numpy()[:, None]
    if pred_rate is None:
        pred_rate = np.zeros(len(df), dtype=float)
    return np.column_stack([num, mat_oh, pos_edge, np.asarray(pred_rate, dtype=float)])


def _recipe_feature(recipe: dict[str, Any], pred_rate: float) -> np.ndarray:
    row = pd.DataFrame([{
        **{col: float(recipe[col]) for col in NUMERIC},
        "substrate": recipe["substrate"],
        "position": recipe["position"],
    }])
    return _feature_matrix(row, np.array([pred_rate], dtype=float))


def _fit_residual_uncertainty_model(df: pd.DataFrame) -> dict[str, Any]:
    """Fit an auxiliary residual model from saved PIML OOF predictions.

    This model is separate from PIML: it learns only the magnitude of the OOF
    residual and is used to size prediction intervals.
    """
    if not OOF_PATH.exists():
        return {"available": False, "reason": f"missing {OOF_PATH}"}

    oof = pd.read_csv(OOF_PATH)
    pred_col = f"pred_{MODEL_NAME}"
    required = {"substrate", "position", TARGET, pred_col}
    if not required.issubset(oof.columns):
        return {"available": False, "reason": "oof.csv missing columns for residual model"}

    train = oof.dropna(subset=list(required)).copy()
    if len(train) != len(df):
        return {"available": False, "reason": "oof.csv row count does not match processed data"}
    for col in NUMERIC:
        train[col] = df[col].to_numpy()
    train["substrate"] = train["substrate"].map(_normalize_material)
    train["position"] = train["position"].map(_normalize_position)
    pred = train[pred_col].astype(float).to_numpy()
    residual = train[TARGET].astype(float).to_numpy() - pred
    abs_residual = np.abs(residual)

    x = _feature_matrix(train, pred)
    y = np.log1p(abs_residual)
    model = RandomForestRegressor(
        n_estimators=240,
        min_samples_leaf=10,
        random_state=20260896,
        n_jobs=1,
    )
    model.fit(x, y)

    pred_abs = np.expm1(model.predict(x))
    pred_abs = np.clip(pred_abs, 1e-6, None)
    normalized_error = abs_residual / pred_abs
    q95 = float(np.quantile(normalized_error, 0.95))
    mean_abs = float(abs_residual.mean())
    by_material = {
        m: float(abs_residual[train["substrate"].to_numpy() == m].mean())
        for m in MATERIALS
        if (train["substrate"].to_numpy() == m).any()
    }
    return {
        "available": True,
        "model": model,
        "calibration_q95": max(q95, 1.0),
        "mean_abs_residual": mean_abs,
        "mean_abs_residual_by_material": by_material,
        "n_rows": int(len(train)),
        "target": "log1p(abs(y_true - pred_piml_oof))",
    }


@lru_cache(maxsize=1)
def app_state() -> dict[str, Any]:
    df = load_long()
    df = df.dropna(subset=NUMERIC + [TARGET, "substrate", "position"]).reset_index(drop=True)
    df["position"] = df["position"].map(_normalize_position)
    df["substrate"] = df["substrate"].map(_normalize_material)

    x_num = df[NUMERIC].astype(float).to_numpy()
    mean = x_num.mean(axis=0)
    std = x_num.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    x_scaled = (x_num - mean) / std

    mat = df["substrate"].to_numpy()
    pos = df["position"].to_numpy()
    y = df[TARGET].astype(float).to_numpy()
    metrics = _read_json(OUTPUT_ROOT / MODEL_NAME / "metrics.json", {})
    uncertainty = _fit_residual_uncertainty_model(df)

    domain: dict[str, dict[str, float]] = {}
    for col in NUMERIC:
        domain[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }

    return {
        "df": df,
        "x_scaled": x_scaled,
        "mean": mean,
        "std": std,
        "mat": mat,
        "pos": pos,
        "y": y,
        "metrics": metrics,
        "uncertainty": uncertainty,
        "domain": domain,
    }


def _payload_to_recipe(payload: dict[str, Any]) -> dict[str, Any]:
    state = app_state()
    domain = state["domain"]
    return {
        "pressure_Pa": _safe_float(payload.get("pressure", payload.get("pressure_Pa")), domain["pressure_Pa"]["median"]),
        "power_W": _safe_float(payload.get("power", payload.get("power_W")), domain["power_W"]["median"]),
        "time_min": _safe_float(payload.get("time", payload.get("time_min")), domain["time_min"]["median"]),
        "Ar_frac": _as_fraction(payload.get("ar", payload.get("Ar_frac")), domain["Ar_frac"]["median"]),
        "O2_frac": _as_fraction(payload.get("o2", payload.get("O2_frac")), domain["O2_frac"]["median"]),
        "CF4_frac": _as_fraction(payload.get("cf4", payload.get("CF4_frac")), domain["CF4_frac"]["median"]),
        "substrate": _normalize_material(payload.get("material", payload.get("substrate"))),
        "position": _normalize_position(payload.get("position")),
    }


def _recipe_vector(recipe: dict[str, Any]) -> np.ndarray:
    state = app_state()
    raw = np.array([[float(recipe[col]) for col in NUMERIC]], dtype=float)
    return ((raw - state["mean"]) / state["std"])[0]


def _distance(recipe: dict[str, Any]) -> np.ndarray:
    state = app_state()
    q = _recipe_vector(recipe)
    d = np.linalg.norm(state["x_scaled"] - q, axis=1)
    d = d + (state["mat"] != recipe["substrate"]) * 1.25
    d = d + (state["pos"] != recipe["position"]) * 0.35
    return d.astype(float)


def _domain_report(recipe: dict[str, Any], min_distance: float) -> dict[str, Any]:
    state = app_state()
    outside: list[str] = []
    ext = 0.0
    for col in NUMERIC:
        lo = state["domain"][col]["min"]
        hi = state["domain"][col]["max"]
        val = float(recipe[col])
        span = max(hi - lo, 1e-9)
        if val < lo:
            outside.append(col)
            ext += (lo - val) / span
        elif val > hi:
            outside.append(col)
            ext += (val - hi) / span
    gas_sum = recipe["Ar_frac"] + recipe["O2_frac"] + recipe["CF4_frac"]
    if abs(gas_sum - 1.0) > 0.03:
        outside.append("gas_sum")
        ext += abs(gas_sum - 1.0)
    in_domain = not outside and min_distance < 2.2
    return {
        "in_domain": bool(in_domain),
        "outside": outside,
        "gas_sum": float(gas_sum),
        "nearest_distance": float(min_distance),
        "extrapolation": float(ext + max(0.0, min_distance - 2.2) / 2.2),
    }


def _residual_interval(recipe: dict[str, Any], rate: float, domain: dict[str, Any]) -> dict[str, Any]:
    uncertainty = app_state()["uncertainty"]
    if uncertainty.get("available"):
        x = _recipe_feature(recipe, rate)
        pred_abs = float(np.expm1(uncertainty["model"].predict(x)[0]))
        pred_abs = max(pred_abs, 1e-6)
        half_width = pred_abs * float(uncertainty["calibration_q95"])
        half_width *= 1.0 + 0.35 * float(domain.get("extrapolation", 0.0))
        return {
            "available": True,
            "method": "auxiliary residual uncertainty model",
            "target": uncertainty["target"],
            "predicted_abs_residual": pred_abs,
            "calibration_q95": float(uncertainty["calibration_q95"]),
            "rate_half_width": half_width,
            "n_rows": int(uncertainty["n_rows"]),
        }

    material = recipe["substrate"]
    fallback_abs = float(
        uncertainty.get("mean_abs_residual_by_material", {}).get(
            material,
            uncertainty.get("mean_abs_residual", 0.0),
        )
    )
    if fallback_abs <= 0.0:
        fallback_abs = 0.10 * max(rate, 1.0)
    half_width = 1.96 * fallback_abs * (1.0 + float(domain.get("extrapolation", 0.0)))
    return {
        "available": False,
        "method": "fallback mean absolute residual",
        "reason": uncertainty.get("reason", "residual model unavailable"),
        "predicted_abs_residual": fallback_abs,
        "calibration_q95": 1.96,
        "rate_half_width": half_width,
        "n_rows": 0,
    }


def predict_recipe(recipe: dict[str, Any], k: int = 24) -> dict[str, Any]:
    state = app_state()
    d = _distance(recipe)
    k = min(k, len(d))
    idx = np.argpartition(d, k - 1)[:k]
    local_d = d[idx]
    weights = 1.0 / np.square(local_d + 0.12)
    rate = float(np.average(state["y"][idx], weights=weights))

    time_min = max(0.0, float(recipe["time_min"]))
    depth = rate * time_min
    domain = _domain_report(recipe, float(local_d.min()))
    uncertainty = _residual_interval(recipe, rate, domain)
    rate_half_width = float(uncertainty["rate_half_width"])
    rate_lo = max(0.0, rate - rate_half_width)
    rate_hi = rate + rate_half_width
    depth_lo = rate_lo * time_min
    depth_hi = rate_hi * time_min

    material = recipe["substrate"]
    pr_factor = {"Si": 0.85, "SiO2": 1.15, "SiN": 1.0}.get(material, 1.0)
    selectivity = max(0.1, rate / 8.0 * pr_factor)

    neighbors = state["df"].iloc[idx].copy()
    neighbors["_distance"] = local_d
    neighbors = neighbors.sort_values("_distance").head(5)

    return {
        "model": MODEL_NAME,
        "method": "backend-data-calibrated-nearest-neighbors",
        "input": recipe,
        "rate": rate,
        "rate_unit": "nm/min",
        "rate_ci95": [rate_lo, rate_hi],
        "sigma_rate": (rate_hi - rate_lo) / (2.0 * 1.96),
        "depth": depth,
        "depth_unit": "nm",
        "depth_ci95": [depth_lo, depth_hi],
        "selectivity": selectivity,
        "domain": domain,
        "uncertainty": uncertainty,
        "metrics": state["metrics"].get("overall_oof", {}),
        "neighbors": [
            {
                "distance": float(row["_distance"]),
                "substrate": str(row["substrate"]),
                "position": str(row["position"]),
                "pressure_Pa": float(row["pressure_Pa"]),
                "power_W": float(row["power_W"]),
                "time_min": float(row["time_min"]),
                "Ar_frac": float(row["Ar_frac"]),
                "O2_frac": float(row["O2_frac"]),
                "CF4_frac": float(row["CF4_frac"]),
                "etch_rate_nm_per_min": float(row[TARGET]),
            }
            for _, row in neighbors.iterrows()
        ],
    }


def sweep_recipe(payload: dict[str, Any]) -> dict[str, Any]:
    key_map = {
        "pressure": ("pressure_Pa", 1.0),
        "power": ("power_W", 1.0),
        "time": ("time_min", 1.0),
        "ar": ("Ar_frac", 100.0),
        "o2": ("O2_frac", 100.0),
        "cf4": ("CF4_frac", 100.0),
    }
    key = str(payload.get("key", "pressure"))
    col, front_scale = key_map.get(key, key_map["pressure"])
    n = int(min(160, max(8, _safe_float(payload.get("n"), 80))))
    recipe = _payload_to_recipe(payload)
    domain = app_state()["domain"][col]
    xs = np.linspace(domain["min"], domain["max"], n)
    points = []
    for x in xs:
        r = dict(recipe)
        r[col] = float(x)
        p = predict_recipe(r, k=24)
        points.append({
            "x": float(x * front_scale),
            "depth": p["depth"],
            "lo": p["depth_ci95"][0],
            "hi": p["depth_ci95"][1],
            "rate": p["rate"],
            "in_domain": p["domain"]["in_domain"],
        })
    return {"key": key, "points": points}


def domain_payload() -> dict[str, Any]:
    state = app_state()
    domain = {}
    for col, item in state["domain"].items():
        scale = 100.0 if col.endswith("_frac") else 1.0
        domain[col] = {k: v * scale for k, v in item.items()}
    return {
        "model": MODEL_NAME,
        "domain": domain,
        "materials": MATERIALS,
        "positions": POSITIONS,
        "rows": int(len(state["df"])),
        "metrics": state["metrics"].get("overall_oof", {}),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "EtchPIML/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._send(204, b"", "text/plain; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._send(200, FRONTEND_HTML.read_bytes(), "text/html; charset=utf-8")
            elif parsed.path == "/api/health":
                self._json(200, {"ok": True, "service": "etch-piml-api"})
            elif parsed.path == "/api/metrics":
                self._json(200, app_state()["metrics"])
            elif parsed.path == "/api/domain":
                self._json(200, domain_payload())
            elif parsed.path == "/api/predict":
                payload = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
                self._json(200, predict_recipe(_payload_to_recipe(payload)))
            else:
                self._json(404, {"error": f"Not found: {parsed.path}"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._body_json()
            if parsed.path == "/api/predict":
                self._json(200, predict_recipe(_payload_to_recipe(payload)))
            elif parsed.path == "/api/sweep":
                self._json(200, sweep_recipe(payload))
            else:
                self._json(404, {"error": f"Not found: {parsed.path}"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    app_state()
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"PIML etch API serving {FRONTEND_HTML} at {url}")
    print(f"Health check: {url}/api/health")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
