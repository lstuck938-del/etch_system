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
    ci = (
        metrics.get("ci", {})
        .get("quantiles_per_material", {})
    )

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
        "ci": ci,
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


def predict_recipe(recipe: dict[str, Any], k: int = 24) -> dict[str, Any]:
    state = app_state()
    d = _distance(recipe)
    k = min(k, len(d))
    idx = np.argpartition(d, k - 1)[:k]
    local_d = d[idx]
    weights = 1.0 / np.square(local_d + 0.12)
    rate = float(np.average(state["y"][idx], weights=weights))

    material = recipe["substrate"]
    quant = state["ci"].get(material, {})
    if quant:
        rate_lo = max(0.0, rate + float(quant.get("q_lo", 0.0)))
        rate_hi = max(rate_lo, rate + float(quant.get("q_hi", 0.0)))
    else:
        local_std = float(np.sqrt(np.average((state["y"][idx] - rate) ** 2, weights=weights)))
        rate_lo = max(0.0, rate - 1.96 * local_std)
        rate_hi = rate + 1.96 * local_std

    time_min = max(0.0, float(recipe["time_min"]))
    depth = rate * time_min
    depth_lo = rate_lo * time_min
    depth_hi = rate_hi * time_min
    domain = _domain_report(recipe, float(local_d.min()))

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
