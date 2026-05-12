"""Passivation-aware Physics-informed Residual EtchNet.

Implements the design in PILML初步思路.md:
  y_pred = y_physics(x; theta_phys) + y_nn(x; theta_nn)
with physics features (E_ion proxy, ion flux, active fluorine, CFx
passivation, pressure transport) and soft physics constraints
(power monotonicity, pressure unimodality around material-specific peak,
non-negativity).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURES, NUMERIC, RANDOM_SEED, TARGET
from src.data import load_long, make_groups
from src.evaluate import (
    cv_split,
    output_dir,
    regression_metrics,
    save_metrics_json,
    save_oof_csv,
)

MODEL_NAME = "piml"
DEVICE = torch.device("cpu")

MATERIALS = ["Si", "SiO2", "SiN"]
P_PEAK = {"Si": 20.0, "SiO2": 7.5, "SiN": 13.0}
P_SIGMA = {"Si": 8.0, "SiO2": 5.0, "SiN": 8.0}

# Nominal channel thresholds. E_ion is only a W/P proxy, so keep these explicit:
# small-sample CV favored the conservative hierarchy over proxy-scale thresholds.
E_TH = {
    "Si":   {"chem": 10.0, "direct": 50.0, "phys": 150.0},
    "SiO2": {"chem": 50.0, "direct": 70.0, "phys": 100.0},
    "SiN":  {"chem": 30.0, "direct": 60.0, "phys": 130.0},
}

EPOCHS = 450
PHYSICS_PRETRAIN_EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = (64, 64, 32)
LAMBDA_POWER = 0.20
LAMBDA_PRESSURE = 0.30
LAMBDA_NONNEG = 0.10
LAMBDA_RESIDUAL = 0.0
SAMPLE_WEIGHT_POWER = 0.0
N_PERTURB = 96
N_DOMAIN_PRES = 64  # extra synthetic pressure perturbation pairs over full domain

CI_ALPHA = 0.05  # 95% prediction interval from per-material OOF residual quantiles

NUM_RAW = ["pressure_Pa", "power_W", "time_min", "Ar_frac", "O2_frac", "CF4_frac", "electrode_gap_cm"]


def physics_features(
    P: torch.Tensor,
    W: torch.Tensor,
    Ar: torch.Tensor,
    O2: torch.Tensor,
    CF4: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute physics proxies. All inputs in original units, outputs un-normalized."""
    eps = 1.0
    E_ion = W / (P + eps)
    pressure_transport = torch.sqrt(P + eps) / (P + eps)
    Ion_flux = W * pressure_transport * (1.0 + 0.5 * Ar)
    O2_boost = (1.0 + 2.0 * O2) / (1.0 + 4.0 * O2.pow(2))
    Flux_F = W * CF4 * torch.log1p(P) * O2_boost
    Flux_CFx = W * CF4 * P / (P + 1.0) * torch.clamp(1.0 - O2, min=0.0).pow(1.5)
    return {
        "E_ion": E_ion,
        "Flux_F": Flux_F,
        "Flux_CFx": Flux_CFx,
        "Ion_flux": Ion_flux,
        "Pressure_transport": pressure_transport,
        "O2_boost": O2_boost,
    }


def threshold_act(E: torch.Tensor, E_th: float, k: float = 0.08) -> torch.Tensor:
    return torch.sigmoid(k * (E - E_th))


def pressure_shape(P: torch.Tensor, P_peak: float, sigma: float) -> torch.Tensor:
    return torch.exp(-((P - P_peak) ** 2) / (2.0 * sigma ** 2))


class PhysicsBackbone(nn.Module):
    """Per-material semi-empirical finite-time etch-rate backbone.

    The model predicts the measured average rate as:
        rate_avg(t) = R_ss + A * (1 - exp(-t / tau)) / t
    where R_ss is a steady-state competition between chemical etch,
    ion-enhanced etch, sputtering, and CFx passivation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.k_chem_raw = nn.Parameter(torch.tensor([0.6, 0.4, 0.5]))
        self.k_ion_raw = nn.Parameter(torch.tensor([-2.0, -1.8, -1.9]))
        self.k_sputter_raw = nn.Parameter(torch.tensor([-2.5, -2.0, -2.3]))
        self.k_pass_raw = nn.Parameter(torch.tensor([0.1, 0.4, 0.2]))
        self.k_pass_remove_o2_raw = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.k_pass_remove_ion_raw = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0]))
        self.k_o2_promote_raw = nn.Parameter(torch.tensor([1.0, 0.8, 0.9]))
        self.k_o2_quench_raw = nn.Parameter(torch.tensor([3.0, 3.0, 3.0]))
        self.k_ar_boost_raw = nn.Parameter(torch.tensor([0.3, 0.3, 0.3]))

        self.p_alpha_raw = nn.Parameter(torch.tensor([1.3, 1.3, 1.3]))
        self.p_c_raw = nn.Parameter(torch.tensor([8.0, 7.0, 8.0]))
        self.p_loss_raw = nn.Parameter(torch.tensor([28.0, 24.0, 26.0]))

        self.scale_raw = nn.Parameter(torch.tensor([-3.4, -3.3, -3.3]))
        self.bias = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.transient_amp_raw = nn.Parameter(torch.tensor([35.0, 38.0, 36.0]))
        self.tau_raw = nn.Parameter(torch.tensor([2.0, 2.5, 2.2]))

    def forward(self, P: torch.Tensor, W: torch.Tensor, Ar: torch.Tensor,
                O2: torch.Tensor, CF4: torch.Tensor,
                T: torch.Tensor,
                mat_idx: torch.Tensor) -> torch.Tensor:
        feats = physics_features(P, W, Ar, O2, CF4)
        E = feats["E_ion"]
        ion_flux = feats["Ion_flux"]

        E_chem = torch.as_tensor([E_TH[m]["chem"] for m in MATERIALS], dtype=P.dtype, device=P.device)[mat_idx]
        E_dir = torch.as_tensor([E_TH[m]["direct"] for m in MATERIALS], dtype=P.dtype, device=P.device)[mat_idx]
        E_phys = torch.as_tensor([E_TH[m]["phys"] for m in MATERIALS], dtype=P.dtype, device=P.device)[mat_idx]
        A_chem = torch.sigmoid(0.75 * (E - E_chem))
        A_dir = torch.sigmoid(0.75 * (E - E_dir))
        A_phys = torch.sigmoid(0.75 * (E - E_phys))

        alpha_p = F.softplus(self.p_alpha_raw[mat_idx]) + 0.5
        p_c = F.softplus(self.p_c_raw[mat_idx]) + 1.0
        p_loss = F.softplus(self.p_loss_raw[mat_idx]) + 5.0
        p_ratio = (P.clamp_min(1e-3) / p_c).pow(alpha_p)
        pressure_transport = p_ratio / (1.0 + p_ratio) * torch.exp(-P / p_loss)

        o2_promote = 1.0 + F.softplus(self.k_o2_promote_raw[mat_idx]) * O2
        o2_quench = 1.0 + F.softplus(self.k_o2_quench_raw[mat_idx]) * O2.pow(2)
        active_f = W * CF4 * pressure_transport * o2_promote / o2_quench

        cf4_poly = torch.clamp(1.0 - O2, min=0.0).pow(1.5)
        cf_x = W * CF4 * pressure_transport * cf4_poly
        ar_boost = 1.0 + F.softplus(self.k_ar_boost_raw[mat_idx]) * Ar
        ion_eff = ion_flux * pressure_transport * ar_boost
        ion_norm = ion_eff / (ion_eff.detach().mean().clamp_min(1.0))

        r_chem = F.softplus(self.k_chem_raw[mat_idx]) * active_f * A_chem
        r_ion = F.softplus(self.k_ion_raw[mat_idx]) * active_f * ion_eff * A_dir
        r_sputter = F.softplus(self.k_sputter_raw[mat_idx]) * ion_eff * A_phys
        pass_removal = (
            1.0
            + F.softplus(self.k_pass_remove_o2_raw[mat_idx]) * O2
            + F.softplus(self.k_pass_remove_ion_raw[mat_idx]) * ion_norm
        )
        r_pass = F.softplus(self.k_pass_raw[mat_idx]) * cf_x / pass_removal

        balance = r_chem + r_ion + r_sputter - r_pass
        steady = F.softplus(F.softplus(self.scale_raw[mat_idx]) * balance + self.bias[mat_idx])
        amp = F.softplus(self.transient_amp_raw[mat_idx])
        tau = F.softplus(self.tau_raw[mat_idx]) + 0.2
        transient_rate = amp * (1.0 - torch.exp(-T.clamp_min(1e-3) / tau)) / T.clamp_min(1e-3)
        return steady + transient_rate


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        # zero-init the final layer so the model starts close to physics-only
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PIML(nn.Module):
    def __init__(self, raw_dim: int, n_mat: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.physics = PhysicsBackbone()
        # NN sees: standardized raw numerics (raw_dim), physics features (6),
        # material one-hot (n_mat), photoresist & position one-hot (handled outside)
        nn_in = raw_dim + 6 + n_mat
        self.nn = ResidualMLP(nn_in, hidden)
        self.raw_dim = raw_dim
        self.n_mat = n_mat

    def forward(
        self,
        raw_std: torch.Tensor,     # [N, raw_dim] standardized
        P: torch.Tensor, W: torch.Tensor, Ar: torch.Tensor, O2: torch.Tensor,
        CF4: torch.Tensor, T: torch.Tensor,  # original units
        mat_idx: torch.Tensor,     # [N] long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = physics_features(P, W, Ar, O2, CF4)
        # standardize physics features by their own batch-stable stats (log1p + scale)
        phys_in = torch.stack([
            torch.log1p(feats["E_ion"]),
            torch.log1p(feats["Flux_F"]),
            torch.log1p(feats["Flux_CFx"]),
            torch.log1p(feats["Ion_flux"]),
            torch.log1p(feats["Pressure_transport"]),
            torch.log1p(1.0 / T.clamp_min(1e-3)),
        ], dim=-1)
        mat_oh = F.one_hot(mat_idx, num_classes=self.n_mat).float()
        x_nn = torch.cat([raw_std, phys_in, mat_oh], dim=-1)
        y_phys = self.physics(P, W, Ar, O2, CF4, T, mat_idx)
        y_nn = self.nn(x_nn)
        return y_phys, y_nn


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def make_tensors(df: pd.DataFrame) -> dict:
    P = df["pressure_Pa"].astype(float).to_numpy()
    W = df["power_W"].astype(float).to_numpy()
    T = df["time_min"].astype(float).to_numpy()
    Ar = df["Ar_frac"].astype(float).fillna(0.0).to_numpy() if "Ar_frac" in df.columns else np.zeros(len(df))
    O2 = df["O2_frac"].astype(float).fillna(0.0).to_numpy() if "O2_frac" in df.columns else np.zeros(len(df))
    CF4 = df["CF4_frac"].astype(float).fillna(0.0).to_numpy() if "CF4_frac" in df.columns else np.zeros(len(df))
    if "sample_weight" in df.columns:
        weight = df["sample_weight"].astype(float).to_numpy() ** SAMPLE_WEIGHT_POWER
    else:
        weight = np.ones(len(df))

    mat = df["substrate"].astype(str).fillna("Si").to_numpy()
    mat_idx = np.array([MATERIALS.index(m) if m in MATERIALS else 0 for m in mat], dtype=np.int64)

    raw_cols = [c for c in NUM_RAW if c in df.columns]
    raw = df[raw_cols].astype(float).fillna(df[raw_cols].astype(float).median()).to_numpy()

    # categorical (photoresist, position) one-hot
    cat_pieces = []
    cat_cols: list[str] = []
    for c in ["photoresist", "position"]:
        if c in df.columns:
            s = df[c].astype(str).fillna("unknown")
            d = pd.get_dummies(s, prefix=c, dtype=float)
            cat_cols.extend(d.columns.tolist())
            cat_pieces.append(d.to_numpy())
    if cat_pieces:
        raw = np.concatenate([raw] + cat_pieces, axis=1)

    return {
        "P": P, "W": W, "Ar": Ar, "O2": O2, "T": T, "CF4": CF4,
        "mat_idx": mat_idx,
        "sample_weight": weight,
        "raw": raw,
        "raw_cols": raw_cols,
        "cat_cols": cat_cols,
    }


# --------------------------------------------------------------------------- #
# Constraint losses
# --------------------------------------------------------------------------- #

def weighted_mse(y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight / weight.mean().clamp_min(1e-6)
    return (w * (y_pred - y_true).pow(2)).mean()


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight / weight.mean().clamp_min(1e-6)
    return (w * value).mean()


def loss_power_monotone(model: PIML, raw_std: torch.Tensor, P: torch.Tensor, W: torch.Tensor,
                        Ar: torch.Tensor, O2: torch.Tensor, CF4: torch.Tensor,
                        T: torch.Tensor, mat_idx: torch.Tensor,
                        raw_W_idx: int, w_mean: float, w_std: float) -> torch.Tensor:
    """Penalize d(rate)/d(W) < 0 by sampling pairs with small +ΔW perturbation."""
    delta = 0.10 * (W.mean() + 1.0)
    raw_hi = raw_std.clone()
    raw_hi[:, raw_W_idx] = raw_hi[:, raw_W_idx] + delta / max(w_std, 1e-6)
    yp_lo_phys, yp_lo_nn = model(raw_std, P, W, Ar, O2, CF4, T, mat_idx)
    yp_hi_phys, yp_hi_nn = model(raw_hi, P, W + delta, Ar, O2, CF4, T, mat_idx)
    yp_lo = yp_lo_phys + yp_lo_nn
    yp_hi = yp_hi_phys + yp_hi_nn
    return F.relu(yp_lo - yp_hi).mean()


def loss_pressure_unimodal(model: PIML, raw_std: torch.Tensor, P: torch.Tensor, W: torch.Tensor,
                           Ar: torch.Tensor, O2: torch.Tensor, CF4: torch.Tensor,
                           T: torch.Tensor, mat_idx: torch.Tensor,
                           raw_P_idx: int, p_std: float) -> torch.Tensor:
    """Penalize wrong-sign d(rate)/d(P) relative to material-specific peak."""
    delta = 0.3  # Pa
    raw_hi = raw_std.clone()
    raw_hi[:, raw_P_idx] = raw_hi[:, raw_P_idx] + delta / max(p_std, 1e-6)
    yp0_phys, yp0_nn = model(raw_std, P, W, Ar, O2, CF4, T, mat_idx)
    yp1_phys, yp1_nn = model(raw_hi, P + delta, W, Ar, O2, CF4, T, mat_idx)
    yp0 = yp0_phys + yp0_nn
    yp1 = yp1_phys + yp1_nn
    dy = yp1 - yp0
    P_pk = torch.as_tensor([P_PEAK[m] for m in MATERIALS], dtype=P.dtype, device=P.device)[mat_idx]
    # below peak: dy should be ≥ 0 → penalize negative dy
    # above peak: dy should be ≤ 0 → penalize positive dy
    below = (P < P_pk).float()
    above = (P > P_pk).float()
    return (below * F.relu(-dy) + above * F.relu(dy)).mean()


def loss_pressure_domain(model: PIML, bundle_raw_mean: np.ndarray, bundle_raw_std: np.ndarray,
                         raw_dim: int, raw_P_idx: int, raw_W_idx: int, raw_T_idx: int | None,
                         n: int = N_DOMAIN_PRES) -> torch.Tensor:
    """Sample synthetic (P, material) pairs across the full pressure domain
    with all other features at zero (i.e., dataset mean) and enforce
    unimodality. This generalizes the constraint beyond observed samples."""
    p_std = max(float(bundle_raw_std[raw_P_idx]), 1e-6)
    w_std = max(float(bundle_raw_std[raw_W_idx]), 1e-6)
    p_mean = float(bundle_raw_mean[raw_P_idx])
    w_mean = float(bundle_raw_mean[raw_W_idx])
    if raw_T_idx is not None:
        t_std = max(float(bundle_raw_std[raw_T_idx]), 1e-6)
        t_mean = float(bundle_raw_mean[raw_T_idx])

    P_lo, P_hi = 5.0, 30.0
    P0 = torch.empty(n).uniform_(P_lo, P_hi)
    delta = 0.3
    raw0 = torch.zeros(n, raw_dim)
    # set pressure column in standardized space
    raw0[:, raw_P_idx] = (P0 - p_mean) / p_std
    # use a CF4-rich power level (boosts physics signal across pressures)
    W_val = 50.0
    T_val = 5.0
    raw0[:, raw_W_idx] = (W_val - w_mean) / w_std
    if raw_T_idx is not None:
        raw0[:, raw_T_idx] = (T_val - t_mean) / t_std
    raw1 = raw0.clone()
    raw1[:, raw_P_idx] = (P0 + delta - p_mean) / p_std

    P0t = P0
    P1t = P0 + delta
    Wt = torch.full((n,), W_val)
    Tt = torch.full((n,), T_val)
    Art = torch.full((n,), 0.3)
    O2t = torch.full((n,), 0.1)
    CF4t = torch.full((n,), 0.7)  # CF4-dominant
    M = torch.randint(0, len(MATERIALS), (n,), dtype=torch.long)

    yp0_phys, yp0_nn = model(raw0, P0t, Wt, Art, O2t, CF4t, Tt, M)
    yp1_phys, yp1_nn = model(raw1, P1t, Wt, Art, O2t, CF4t, Tt, M)
    dy = (yp1_phys + yp1_nn) - (yp0_phys + yp0_nn)
    P_pk = torch.as_tensor([P_PEAK[m] for m in MATERIALS], dtype=P0t.dtype, device=P0t.device)[M]
    below = (P0t < P_pk).float()
    above = (P0t > P_pk).float()
    return (below * F.relu(-dy) + above * F.relu(dy)).mean()


def loss_nonneg(y: torch.Tensor) -> torch.Tensor:
    return F.relu(-y).mean()


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def train_one_fold(
    bundle: dict, tr: np.ndarray, va: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, PIML]:
    torch.manual_seed(RANDOM_SEED)

    raw_mean = bundle["raw"][tr].mean(axis=0)
    raw_std = bundle["raw"][tr].std(axis=0)
    raw_tr = (bundle["raw"][tr] - raw_mean) / np.where(raw_std > 0, raw_std, 1.0)
    raw_va = (bundle["raw"][va] - raw_mean) / np.where(raw_std > 0, raw_std, 1.0)

    raw_tr_t = torch.from_numpy(raw_tr).float()
    raw_va_t = torch.from_numpy(raw_va).float()

    P_tr = torch.from_numpy(bundle["P"][tr]).float()
    W_tr = torch.from_numpy(bundle["W"][tr]).float()
    Ar_tr = torch.from_numpy(bundle["Ar"][tr]).float()
    O2_tr = torch.from_numpy(bundle["O2"][tr]).float()
    T_tr = torch.from_numpy(bundle["T"][tr]).float()
    CF4_tr = torch.from_numpy(bundle["CF4"][tr]).float()
    M_tr = torch.from_numpy(bundle["mat_idx"][tr]).long()
    y_tr = torch.from_numpy(y[tr]).float()
    weight_tr = torch.from_numpy(bundle["sample_weight"][tr]).float()

    P_va = torch.from_numpy(bundle["P"][va]).float()
    W_va = torch.from_numpy(bundle["W"][va]).float()
    Ar_va = torch.from_numpy(bundle["Ar"][va]).float()
    O2_va = torch.from_numpy(bundle["O2"][va]).float()
    T_va = torch.from_numpy(bundle["T"][va]).float()
    CF4_va = torch.from_numpy(bundle["CF4"][va]).float()
    M_va = torch.from_numpy(bundle["mat_idx"][va]).long()

    model = PIML(raw_dim=raw_tr.shape[1], n_mat=len(MATERIALS), hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    raw_W_idx = bundle["raw_cols"].index("power_W")
    raw_P_idx = bundle["raw_cols"].index("pressure_Pa")
    raw_T_idx = bundle["raw_cols"].index("time_min") if "time_min" in bundle["raw_cols"] else None
    w_std = float(raw_std[raw_W_idx])
    p_std = float(raw_std[raw_P_idx])

    history = {"epoch": [], "train_mse": [], "val_mse": [],
               "L_power": [], "L_pressure": [], "L_nonneg": [], "L_residual": []}

    if PHYSICS_PRETRAIN_EPOCHS:
        phys_opt = torch.optim.Adam(model.physics.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(PHYSICS_PRETRAIN_EPOCHS):
            model.train()
            phys_opt.zero_grad()
            y_phys = model.physics(P_tr, W_tr, Ar_tr, O2_tr, CF4_tr, T_tr, M_tr)
            L_phys = weighted_mse(y_phys, y_tr, weight_tr)
            L_phys.backward()
            phys_opt.step()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        y_phys, y_nn = model(raw_tr_t, P_tr, W_tr, Ar_tr, O2_tr, CF4_tr, T_tr, M_tr)
        y_pred = y_phys + y_nn
        L_data = weighted_mse(y_pred, y_tr, weight_tr)

        # constraint sampling: random subset + perturbation
        idx = torch.randperm(raw_tr_t.size(0))[:N_PERTURB]
        L_pwr = loss_power_monotone(
            model, raw_tr_t[idx], P_tr[idx], W_tr[idx], Ar_tr[idx], O2_tr[idx],
            CF4_tr[idx], T_tr[idx], M_tr[idx],
            raw_W_idx, float(W_tr.mean()), w_std,
        )
        L_pres_obs = loss_pressure_unimodal(
            model, raw_tr_t[idx], P_tr[idx], W_tr[idx], Ar_tr[idx], O2_tr[idx],
            CF4_tr[idx], T_tr[idx], M_tr[idx],
            raw_P_idx, p_std,
        )
        L_pres_dom = loss_pressure_domain(
            model, raw_mean, raw_std, raw_tr.shape[1], raw_P_idx, raw_W_idx, raw_T_idx,
        )
        L_pres = 0.5 * L_pres_obs + 0.5 * L_pres_dom
        L_nn = loss_nonneg(y_pred)
        L_res = weighted_mean(y_nn.pow(2), weight_tr)

        loss = (
            L_data
            + LAMBDA_POWER * L_pwr
            + LAMBDA_PRESSURE * L_pres
            + LAMBDA_NONNEG * L_nn
            + LAMBDA_RESIDUAL * L_res
        )
        loss.backward()
        opt.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                y_phys_v, y_nn_v = model(raw_va_t, P_va, W_va, Ar_va, O2_va, CF4_va, T_va, M_va)
                y_pred_v = y_phys_v + y_nn_v
                vmse = F.mse_loss(y_pred_v, torch.from_numpy(y[va]).float()).item()
            history["epoch"].append(epoch)
            history["train_mse"].append(float(L_data.item()))
            history["val_mse"].append(float(vmse))
            history["L_power"].append(float(L_pwr.item()))
            history["L_pressure"].append(float(L_pres.item()))
            history["L_nonneg"].append(float(L_nn.item()))
            history["L_residual"].append(float(L_res.item()))

    model.eval()
    with torch.no_grad():
        y_phys_v, y_nn_v = model(raw_va_t, P_va, W_va, Ar_va, O2_va, CF4_va, T_va, M_va)
    return (
        (y_phys_v + y_nn_v).numpy(),
        y_phys_v.numpy(),
        y_nn_v.numpy(),
        history,
        model,
    )


def fit_final(bundle: dict, y: np.ndarray, raw_mean: np.ndarray, raw_std: np.ndarray) -> PIML:
    torch.manual_seed(RANDOM_SEED)
    raw_n = (bundle["raw"] - raw_mean) / np.where(raw_std > 0, raw_std, 1.0)
    raw_t = torch.from_numpy(raw_n).float()
    P = torch.from_numpy(bundle["P"]).float()
    W = torch.from_numpy(bundle["W"]).float()
    Ar = torch.from_numpy(bundle["Ar"]).float()
    O2 = torch.from_numpy(bundle["O2"]).float()
    T = torch.from_numpy(bundle["T"]).float()
    CF4 = torch.from_numpy(bundle["CF4"]).float()
    M = torch.from_numpy(bundle["mat_idx"]).long()
    yt = torch.from_numpy(y).float()
    weight = torch.from_numpy(bundle["sample_weight"]).float()

    model = PIML(raw_dim=raw_n.shape[1], n_mat=len(MATERIALS), hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    raw_W_idx = bundle["raw_cols"].index("power_W")
    raw_P_idx = bundle["raw_cols"].index("pressure_Pa")
    raw_T_idx = bundle["raw_cols"].index("time_min") if "time_min" in bundle["raw_cols"] else None
    w_std = float(raw_std[raw_W_idx])
    p_std = float(raw_std[raw_P_idx])

    if PHYSICS_PRETRAIN_EPOCHS:
        phys_opt = torch.optim.Adam(model.physics.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(PHYSICS_PRETRAIN_EPOCHS):
            model.train()
            phys_opt.zero_grad()
            y_phys = model.physics(P, W, Ar, O2, CF4, T, M)
            L_phys = weighted_mse(y_phys, yt, weight)
            L_phys.backward()
            phys_opt.step()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        y_phys, y_nn = model(raw_t, P, W, Ar, O2, CF4, T, M)
        y_pred = y_phys + y_nn
        L_data = weighted_mse(y_pred, yt, weight)
        idx = torch.randperm(raw_t.size(0))[:N_PERTURB]
        L_pwr = loss_power_monotone(model, raw_t[idx], P[idx], W[idx], Ar[idx], O2[idx], CF4[idx], T[idx], M[idx], raw_W_idx, float(W.mean()), w_std)
        L_pres_obs = loss_pressure_unimodal(model, raw_t[idx], P[idx], W[idx], Ar[idx], O2[idx], CF4[idx], T[idx], M[idx], raw_P_idx, p_std)
        L_pres_dom = loss_pressure_domain(model, raw_mean, raw_std, raw_n.shape[1], raw_P_idx, raw_W_idx, raw_T_idx)
        L_pres = 0.5 * L_pres_obs + 0.5 * L_pres_dom
        L_nn = loss_nonneg(y_pred)
        L_res = weighted_mean(y_nn.pow(2), weight)
        loss = (
            L_data
            + LAMBDA_POWER * L_pwr
            + LAMBDA_PRESSURE * L_pres
            + LAMBDA_NONNEG * L_nn
            + LAMBDA_RESIDUAL * L_res
        )
        loss.backward()
        opt.step()
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Prediction interval from OOF residuals
# --------------------------------------------------------------------------- #

def residual_quantiles_per_material(
    y_true: np.ndarray, y_pred: np.ndarray, mat_idx: np.ndarray, alpha: float = CI_ALPHA,
) -> dict[str, dict[str, float]]:
    """Empirical residual quantiles per material, used to build prediction intervals
    that do not require any change to the trained model."""
    qs: dict[str, dict[str, float]] = {}
    for i, m in enumerate(MATERIALS):
        sel = mat_idx == i
        if not sel.any():
            continue
        r = y_true[sel] - y_pred[sel]
        lo, hi = np.quantile(r, [alpha / 2.0, 1.0 - alpha / 2.0])
        qs[m] = {"q_lo": float(lo), "q_hi": float(hi), "n": int(sel.sum())}
    return qs


def apply_ci_band(
    y_pred: np.ndarray, mat_idx: np.ndarray, qs: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Add per-material residual quantile bands; clamp lower bound at 0 (rate ≥ 0)."""
    lo = y_pred.copy().astype(float)
    hi = y_pred.copy().astype(float)
    for i, m in enumerate(MATERIALS):
        sel = mat_idx == i
        if not sel.any() or m not in qs:
            continue
        lo[sel] = y_pred[sel] + qs[m]["q_lo"]
        hi[sel] = y_pred[sel] + qs[m]["q_hi"]
    return np.clip(lo, 0.0, None), hi


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, mat_idx: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"Si": "#1f77b4", "SiO2": "#d62728", "SiN": "#2ca02c"}
    for i, m in enumerate(MATERIALS):
        sel = mat_idx == i
        if not sel.any():
            continue
        ax.scatter(y_true[sel], y_pred[sel], s=22, alpha=0.55, color=colors[m], label=m)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    m = (hi - lo) * 0.05
    ax.plot([lo - m, hi + m], [lo - m, hi + m], "k--", lw=1)
    ax.set_xlabel("Actual etch_rate (nm/min)")
    ax.set_ylabel("Predicted etch_rate (nm/min)")
    ax.set_title("PIML GroupKFold OOF parity (by material)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, mat_idx: np.ndarray, path: Path) -> None:
    res = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Si": "#1f77b4", "SiO2": "#d62728", "SiN": "#2ca02c"}
    for i, m in enumerate(MATERIALS):
        sel = mat_idx == i
        if not sel.any():
            continue
        ax.scatter(y_true[sel], res[sel], s=22, alpha=0.55, color=colors[m], label=m)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Actual etch_rate (nm/min)")
    ax.set_ylabel("Residual (pred - actual)")
    ax.set_title("PIML residuals")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scan_inputs(df: pd.DataFrame, raw_cols: list[str]) -> dict[str, dict]:
    """Median / mode baseline values for the scan."""
    base: dict = {}
    for c in raw_cols:
        base[c] = float(df[c].median()) if c in df.columns else 0.0
    base["CF4_frac"] = float(df["CF4_frac"].median()) if "CF4_frac" in df.columns else 0.0
    return base


def _scan_baseline(df: pd.DataFrame, bundle: dict) -> tuple[np.ndarray, float, float, float, float, float]:
    """Use the CF4-containing experiments as the scan baseline (the .md
    discusses a CF4 plasma; averaging over Ar:O2-only runs would mask the
    expected unimodal pressure shape)."""
    cf4_mask = df["CF4_frac"].fillna(0) > 0
    sub = df[cf4_mask]
    if sub.empty:
        sub = df
    raw_cols = bundle["raw_cols"]
    medians = np.array([float(sub[c].median()) for c in raw_cols])
    P_med = float(sub["pressure_Pa"].median())
    W_med = float(sub["power_W"].median())
    Ar_med = float(sub["Ar_frac"].median()) if "Ar_frac" in sub.columns else 0.0
    O2_med = float(sub["O2_frac"].median()) if "O2_frac" in sub.columns else 0.0
    CF4_med = float(sub["CF4_frac"].median())
    return medians, P_med, W_med, Ar_med, O2_med, CF4_med


def _predict_curve(
    model: PIML,
    bundle: dict,
    df: pd.DataFrame,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    sweep_col: str,
    xs: np.ndarray,
    material: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep one column over xs for a given material, others held at the
    CF4-experiment baseline."""
    n = len(xs)
    raw_cols = bundle["raw_cols"]
    medians, P_med, W_med, Ar_med, O2_med, CF4_med = _scan_baseline(df, bundle)

    base_raw = np.zeros((n, bundle["raw"].shape[1]))
    base_raw[:, : len(raw_cols)] = medians
    # categorical one-hots remain at the column means (already in raw)
    cat_means = bundle["raw"][:, len(raw_cols):].mean(axis=0)
    base_raw[:, len(raw_cols):] = cat_means

    if sweep_col in raw_cols:
        base_raw[:, raw_cols.index(sweep_col)] = xs

    raw_n = (base_raw - raw_mean) / np.where(raw_std > 0, raw_std, 1.0)
    raw_t = torch.from_numpy(raw_n).float()

    P = torch.full((n,), P_med)
    W = torch.full((n,), W_med)
    Ar = torch.full((n,), Ar_med)
    O2 = torch.full((n,), O2_med)
    T_med = float(medians[raw_cols.index("time_min")]) if "time_min" in raw_cols else 5.0
    T = torch.full((n,), T_med)
    CF4 = torch.full((n,), CF4_med)
    if sweep_col == "pressure_Pa":
        P = torch.from_numpy(xs).float()
    elif sweep_col == "power_W":
        W = torch.from_numpy(xs).float()
    elif sweep_col == "time_min":
        T = torch.from_numpy(xs).float()
    elif sweep_col == "Ar_frac":
        Ar = torch.from_numpy(xs).float()
    elif sweep_col == "O2_frac":
        O2 = torch.from_numpy(xs).float()
    elif sweep_col == "CF4_frac":
        CF4 = torch.from_numpy(xs).float()

    M = torch.full((n,), MATERIALS.index(material), dtype=torch.long)

    with torch.no_grad():
        y_phys, y_nn = model(raw_t, P, W, Ar, O2, CF4, T, M)
    return y_phys.numpy(), y_nn.numpy(), (y_phys + y_nn).numpy()


def _draw_ci_band(ax, xs: np.ndarray, y_total: np.ndarray, q: dict[str, float], color: str) -> None:
    lo = np.clip(y_total + q["q_lo"], 0.0, None)
    hi = y_total + q["q_hi"]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)


def plot_pressure_curve(model: PIML, bundle: dict, df: pd.DataFrame,
                        raw_mean: np.ndarray, raw_std: np.ndarray, path: Path,
                        ci_quantiles: dict[str, dict[str, float]] | None = None) -> None:
    xs = np.linspace(float(df["pressure_Pa"].min()), float(df["pressure_Pa"].max()), 120)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"Si": "#1f77b4", "SiO2": "#d62728", "SiN": "#2ca02c"}
    for m in MATERIALS:
        _, _, y_total = _predict_curve(model, bundle, df, raw_mean, raw_std,
                                       "pressure_Pa", xs, m)
        if ci_quantiles and m in ci_quantiles:
            _draw_ci_band(ax, xs, y_total, ci_quantiles[m], colors[m])
        ax.plot(xs, y_total, color=colors[m], lw=2.0, label=f"{m} (peak={P_PEAK[m]} Pa)")
        # actual scatter for this material
        sub = df[(df["substrate"] == m) & (df["CF4_frac"].fillna(0) > 0)]
        ax.scatter(sub["pressure_Pa"], sub[TARGET], s=12, alpha=0.30,
                   color=colors[m], edgecolors="none")
        ax.axvline(P_PEAK[m], color=colors[m], lw=1, ls=":", alpha=0.6)
    ax.set_xlabel("Pressure (Pa)")
    ax.set_ylabel("Predicted etch_rate (nm/min)")
    title = "Pressure → etch_rate (others at median; expected unimodal)"
    if ci_quantiles:
        title += "  |  shaded = 95% PI"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_power_curve(model: PIML, bundle: dict, df: pd.DataFrame,
                     raw_mean: np.ndarray, raw_std: np.ndarray, path: Path,
                     ci_quantiles: dict[str, dict[str, float]] | None = None) -> None:
    xs = np.linspace(float(df["power_W"].min()), float(df["power_W"].max()), 120)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"Si": "#1f77b4", "SiO2": "#d62728", "SiN": "#2ca02c"}
    for m in MATERIALS:
        _, _, y_total = _predict_curve(model, bundle, df, raw_mean, raw_std,
                                       "power_W", xs, m)
        if ci_quantiles and m in ci_quantiles:
            _draw_ci_band(ax, xs, y_total, ci_quantiles[m], colors[m])
        ax.plot(xs, y_total, color=colors[m], lw=2.0, label=m)
        sub = df[(df["substrate"] == m) & (df["CF4_frac"].fillna(0) > 0)]
        ax.scatter(sub["power_W"], sub[TARGET], s=12, alpha=0.30,
                   color=colors[m], edgecolors="none")
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Predicted etch_rate (nm/min)")
    title = "Power → etch_rate (expected monotone increasing)"
    if ci_quantiles:
        title += "  |  shaded = 95% PI"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cf4_curve(model: PIML, bundle: dict, df: pd.DataFrame,
                   raw_mean: np.ndarray, raw_std: np.ndarray, path: Path,
                   ci_quantiles: dict[str, dict[str, float]] | None = None) -> None:
    xs = np.linspace(0.0, float(df["CF4_frac"].max()), 120)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"Si": "#1f77b4", "SiO2": "#d62728", "SiN": "#2ca02c"}
    for m in MATERIALS:
        _, _, y_total = _predict_curve(model, bundle, df, raw_mean, raw_std,
                                       "CF4_frac", xs, m)
        if ci_quantiles and m in ci_quantiles:
            _draw_ci_band(ax, xs, y_total, ci_quantiles[m], colors[m])
        ax.plot(xs, y_total, color=colors[m], lw=2.0, label=m)
        sub = df[(df["substrate"] == m) & (df["CF4_frac"].fillna(0) > 0)]
        ax.scatter(sub["CF4_frac"], sub[TARGET], s=12, alpha=0.30,
                   color=colors[m], edgecolors="none")
    ax.set_xlabel("CF4 fraction")
    ax.set_ylabel("Predicted etch_rate (nm/min)")
    title = "CF4 fraction → etch_rate (expected monotone increasing)"
    if ci_quantiles:
        title += "  |  shaded = 95% PI"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_decomposition(y_true: np.ndarray, y_phys: np.ndarray, y_nn: np.ndarray,
                       mat_idx: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"Si": "#1f77b4", "SiO2": "#d62728", "SiN": "#2ca02c"}

    ax = axes[0]
    for i, m in enumerate(MATERIALS):
        sel = mat_idx == i
        if not sel.any():
            continue
        ax.scatter(y_phys[sel], y_true[sel], s=22, alpha=0.55, color=colors[m], label=m)
    lo = float(min(y_phys.min(), y_true.min()))
    hi = float(max(y_phys.max(), y_true.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("y_physics only")
    ax.set_ylabel("Actual")
    ax.set_title("Physics backbone alone")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, m in enumerate(MATERIALS):
        sel = mat_idx == i
        if not sel.any():
            continue
        ax.scatter(y_phys[sel], y_nn[sel], s=22, alpha=0.55, color=colors[m], label=m)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("y_physics")
    ax.set_ylabel("y_nn (residual correction)")
    ax.set_title("Residual NN correction vs physics rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curves(histories: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for i, h in enumerate(histories, start=1):
        ax.plot(h["epoch"], h["train_mse"], color="#1f77b4", alpha=0.35, lw=1)
        ax.plot(h["epoch"], h["val_mse"], color="#ff7f0e", alpha=0.75, lw=1.2)
    train_mat = np.asarray([h["train_mse"] for h in histories])
    val_mat = np.asarray([h["val_mse"] for h in histories])
    ep = histories[0]["epoch"]
    ax.plot(ep, train_mat.mean(0), color="#1f77b4", lw=2.5, label="train mean")
    ax.plot(ep, val_mat.mean(0), color="#ff7f0e", lw=2.8, label="val mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Data MSE per fold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key, color, label in [("L_power", "#1f77b4", "L_power"),
                               ("L_pressure", "#d62728", "L_pressure"),
                               ("L_nonneg", "#2ca02c", "L_nonneg"),
                               ("L_residual", "#9467bd", "L_residual")]:
        mat = np.asarray([h[key] for h in histories])
        ax.plot(ep, mat.mean(0), color=color, lw=2.0, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Constraint loss (mean over folds)")
    ax.set_title("Physics-constraint losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run() -> None:
    np.random.seed(RANDOM_SEED)
    out = output_dir(MODEL_NAME)
    fig_dir = out / "figures"

    df = load_long()
    groups, n_groups = make_groups(df)
    print(f"rows usable: {len(df)}  |  n_groups: {n_groups}")

    y = df[TARGET].to_numpy().astype(np.float32)
    bundle = make_tensors(df)

    raw_mean = bundle["raw"].mean(axis=0)
    raw_std = bundle["raw"].std(axis=0)

    fold_iter = cv_split(df, y, groups)

    oof = np.full(len(y), np.nan, dtype=np.float32)
    oof_phys = np.full(len(y), np.nan, dtype=np.float32)
    oof_nn = np.full(len(y), np.nan, dtype=np.float32)
    per_fold: list[dict] = []
    histories: list[dict] = []

    for i, (tr, va) in enumerate(fold_iter, start=1):
        pred, pred_phys, pred_nn, hist, _ = train_one_fold(
            bundle, tr, va, y,
        )
        oof[va] = pred
        oof_phys[va] = pred_phys
        oof_nn[va] = pred_nn
        m = regression_metrics(y[va], pred)
        per_fold.append(m)
        histories.append(hist)
        print(f"[{MODEL_NAME}] fold={i} mae={m['mae']:.3f} rmse={m['rmse']:.3f} r2={m['r2']:.3f}")

    overall = regression_metrics(y, oof)
    overall_phys = regression_metrics(y, oof_phys)
    print(f"[{MODEL_NAME}] OOF total: mae={overall['mae']:.3f} rmse={overall['rmse']:.3f} r2={overall['r2']:.3f}")
    print(f"[{MODEL_NAME}] OOF physics-only: mae={overall_phys['mae']:.3f} rmse={overall_phys['rmse']:.3f} r2={overall_phys['r2']:.3f}")

    # 95% prediction interval from per-material OOF residual quantiles.
    # This is post-hoc — the trained model and `oof` predictions are unchanged.
    ci_quantiles = residual_quantiles_per_material(y, oof, bundle["mat_idx"], alpha=CI_ALPHA)
    oof_lo, oof_hi = apply_ci_band(oof, bundle["mat_idx"], ci_quantiles)
    coverage = float(((y >= oof_lo) & (y <= oof_hi)).mean())
    mean_width = float((oof_hi - oof_lo).mean())
    print(f"[{MODEL_NAME}] 95% PI empirical coverage={coverage:.3f}  mean_width={mean_width:.3f} nm/min")
    for m, q in ci_quantiles.items():
        print(f"  {m}: q_lo={q['q_lo']:+.3f}  q_hi={q['q_hi']:+.3f}  n={q['n']}")

    save_oof_csv(
        df, oof, MODEL_NAME, out,
        extra_cols={f"pred_{MODEL_NAME}_lo": oof_lo, f"pred_{MODEL_NAME}_hi": oof_hi},
    )
    pd.DataFrame({"y_true": y, "y_phys": oof_phys, "y_nn": oof_nn,
                  "y_total": oof, "y_lo": oof_lo, "y_hi": oof_hi,
                  "substrate": df["substrate"].to_numpy()}).to_csv(
        out / "oof_decomposition.csv", index=False, encoding="utf-8-sig")

    # figures
    plot_parity(y, oof, bundle["mat_idx"], fig_dir / "parity.png")
    plot_residuals(y, oof, bundle["mat_idx"], fig_dir / "residuals.png")
    plot_decomposition(y, oof_phys, oof_nn, bundle["mat_idx"], fig_dir / "decomposition.png")
    plot_loss_curves(histories, fig_dir / "loss_curves.png")

    # physics scans use the full-data fit
    full_model = fit_final(bundle, y, raw_mean, raw_std)

    # Save a reloadable checkpoint so the API can serve PIML predictions
    # directly instead of falling back to KNN over the training table.
    n_num = len(bundle["raw_cols"])
    cat_means = bundle["raw"][:, n_num:].mean(axis=0) if bundle["raw"].shape[1] > n_num else np.zeros(0)
    torch.save({
        "model_state": full_model.state_dict(),
        "raw_mean": np.asarray(raw_mean, dtype=float),
        "raw_std": np.asarray(raw_std, dtype=float),
        "raw_cols": list(bundle["raw_cols"]),
        "cat_cols": list(bundle["cat_cols"]),
        "cat_means": np.asarray(cat_means, dtype=float),
        "materials": list(MATERIALS),
        "hidden": list(HIDDEN),
        "n_mat": len(MATERIALS),
        "raw_dim": int(bundle["raw"].shape[1]),
        "ci_quantiles": ci_quantiles,
    }, out / "model.pt")
    print(f"saved checkpoint: {out / 'model.pt'}")

    plot_pressure_curve(full_model, bundle, df, raw_mean, raw_std, fig_dir / "scan_pressure.png", ci_quantiles=ci_quantiles)
    plot_power_curve(full_model, bundle, df, raw_mean, raw_std, fig_dir / "scan_power.png", ci_quantiles=ci_quantiles)
    plot_cf4_curve(full_model, bundle, df, raw_mean, raw_std, fig_dir / "scan_cf4.png", ci_quantiles=ci_quantiles)

    save_metrics_json(out, {
        "model": MODEL_NAME,
        "design": "Finite-time passivation-aware residual PIML (y = y_physics + y_nn)",
        "physics_design": "rate_avg(t) = R_ss(P,W,gas,material) + A_material*(1-exp(-t/tau_material))/t; R_ss balances chemical, ion-enhanced, sputter, and CFx passivation terms",
        "n_rows": int(len(df)),
        "n_groups": n_groups,
        "p_peak_per_material": P_PEAK,
        "lambda_power": LAMBDA_POWER,
        "lambda_pressure": LAMBDA_PRESSURE,
        "lambda_nonneg": LAMBDA_NONNEG,
        "lambda_residual": LAMBDA_RESIDUAL,
        "epochs": EPOCHS,
        "physics_pretrain_epochs": PHYSICS_PRETRAIN_EPOCHS,
        "lr": LR,
        "hidden": list(HIDDEN),
        "raw_numeric": NUM_RAW,
        "sample_weight_power": SAMPLE_WEIGHT_POWER,
        "standardization": "per-fold train statistics for OOF; full-data statistics for final scan plots",
        "per_fold": per_fold,
        "overall_oof": overall,
        "overall_oof_physics_only": overall_phys,
        "energy_thresholds": E_TH,
        "physics_terms": ["R_chem", "R_ion_enhanced", "R_sputter", "R_passivation", "time_transient"],
        "ci": {
            "method": "per-material empirical quantiles of OOF residuals (post-hoc; model unchanged)",
            "alpha": CI_ALPHA,
            "level": 1.0 - CI_ALPHA,
            "quantiles_per_material": ci_quantiles,
            "empirical_coverage": coverage,
            "mean_width_nm_per_min": mean_width,
            "lower_clamped_at_zero": True,
        },
    })
    print(f"saved to: {out}")


if __name__ == "__main__":
    run()
