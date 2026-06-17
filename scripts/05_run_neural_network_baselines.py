
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

# Public release script: 05_run_neural_network_baselines.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

#!/usr/bin/env python
# -*- coding: utf-8 -*-




import argparse
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    TORCH_IMPORT_ERROR = repr(e)
else:
    TORCH_IMPORT_ERROR = ""


SCRIPT_VERSION = "Q1_NN_BASELINES_FOR_BENCHMARKING__MLP_GRU_TCN__RUNNER_PATCH_READY__NO_REPLACEMENT"

DATASETS = ["NASA C-MAPSS", "NASA Battery", "PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]

FEATURE_FILES = {
    "NASA C-MAPSS": "cmapss_engine_features_clean_raw.csv",
    "NASA Battery": "battery_features_clean_raw_mat.csv",
    "PRONOSTIA/FEMTO": "pronostia_femto_raw_hi_features.csv",
    "XJTU-SY": "xjtu_sy_raw_hi_features.csv",
    "IMS": "ims_failed_bearing_raw_hi_features.csv",
}

EXPECTED_COUNTS = {
    "NASA C-MAPSS": {"rows": 160359, "assets": 709},
    "NASA Battery": {"rows": 2769, "assets": 34},
    "PRONOSTIA/FEMTO": {"rows": 8383, "assets": 6},
    "XJTU-SY": {"rows": 9216, "assets": 15},
    "IMS": {"rows": 11620, "assets": 4},
}

WINDOWS_FULL = {
    "NASA C-MAPSS": 30,
    "NASA Battery": 10,
    "PRONOSTIA/FEMTO": 40,
    "XJTU-SY": 40,
    "IMS": 80,
}

WINDOWS_QUICK = {
    "NASA C-MAPSS": 20,
    "NASA Battery": 8,
    "PRONOSTIA/FEMTO": 25,
    "XJTU-SY": 25,
    "IMS": 40,
}

GATES = {
    "coverage_min": 0.88,
    "urgent_min": 0.80,
    "max_false_safe": 0.02,
    "max_underwarning": 0.20,
    "width_max": {
        "NASA C-MAPSS": 0.50,
        "NASA Battery": 0.50,
        "PRONOSTIA/FEMTO": 0.50,
        "XJTU-SY": 0.50,
        "IMS": 0.45,
    },
}

COMMON_REJECT_TERMS = [
    "target", "label", "y_true", "y_pred", "lower", "upper",
    "rul_original", "rul_capped", "future", "failure", "eol",
    "source", "path", "file", "filename",
]

CMAPSS_EXTRA_REJECT = ["rul", "life", "remaining"]
BATTERY_EXTRA_REJECT = ["capacity_roll", "capacity_lag", "capacity_delta", "capacity_slope", "soh", "initial", "baseline"]
BEARING_EXTRA_REJECT = ["original_order"]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def banner(text: str) -> None:
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def slug(dataset: str) -> str:
    return norm(dataset).replace("nasa_", "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> str:
    if torch is None:
        return "none"
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("WARNING: CUDA requested but unavailable. Falling back to CPU.", flush=True)
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lookup:
            return lookup[norm(c)]
    return None


def detect_asset_col(df: pd.DataFrame, dataset: str) -> str:
    candidates = {
        "NASA C-MAPSS": ["asset_id", "engine_group", "unit_group", "engine_id", "unit_id", "unit_number", "engine"],
        "NASA Battery": ["asset_id", "battery_id", "cell_id", "battery"],
        "PRONOSTIA/FEMTO": ["asset_id", "bearing_id", "bearing"],
        "XJTU-SY": ["asset_id", "bearing_id", "bearing"],
        "IMS": ["asset_id", "bearing_id", "bearing"],
    }[dataset]
    col = find_col(df, candidates)
    if col is None:
        raise ValueError(f"{dataset}: could not detect asset/group column.")
    return col


def detect_time_col(df: pd.DataFrame) -> Optional[str]:
    return find_col(df, ["cycle", "discharge_index", "time_index", "file_index", "measurement_index", "sample_index", "original_order", "order"])


def minmax_target(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mn, mx = x.min(skipna=True), x.max(skipna=True)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.full(len(x), np.nan, dtype=float)
    return np.clip(((x - mn) / (mx - mn)).to_numpy(float), 0.0, 1.0)


def derive_target(df: pd.DataFrame, dataset: str, asset_col: str) -> Tuple[np.ndarray, str]:
    if dataset == "NASA C-MAPSS":
        col = find_col(df, ["target_health", "target_RUL", "target_RUL_capped_125", "target_RUL__normalized", "RUL"])
        if col is None:
            raise ValueError("C-MAPSS target not found.")
        if norm(col) == "target_health":
            return np.clip(pd.to_numeric(df[col], errors="coerce").to_numpy(float), 0, 1), col
        return minmax_target(df[col]), f"{col} normalized to [0,1]"

    if dataset == "NASA Battery":
        col = find_col(df, ["capacity"])
        if col is None:
            raise ValueError("Battery exact capacity target not found.")
        return minmax_target(df[col]), "capacity -> global minmax health"

    col = find_col(df, ["target_health", "health", "health_index", "HI_raw", "hi"])
    if col is not None:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        if np.nanmax(y) > 1.5 or np.nanmin(y) < -0.05:
            y = minmax_target(pd.Series(y))
        return np.clip(y, 0, 1), col

    y = np.full(len(df), np.nan, dtype=float)
    for _, idx in df.groupby(asset_col).groups.items():
        idx = list(idx)
        n = len(idx)
        y[idx] = 1.0 - np.arange(n) / max(n - 1, 1)
    return np.clip(y, 0, 1), "1 - within-asset progression"


def safe_feature_columns(df: pd.DataFrame, dataset: str, asset_col: str, time_col: Optional[str]) -> Tuple[List[str], List[str]]:
    rejects = set()
    reject_terms = list(COMMON_REJECT_TERMS)
    if dataset == "NASA C-MAPSS":
        reject_terms += CMAPSS_EXTRA_REJECT
    elif dataset == "NASA Battery":
        reject_terms += BATTERY_EXTRA_REJECT
    else:
        reject_terms += BEARING_EXTRA_REJECT

    for c in df.columns:
        cn = norm(c)
        if c == asset_col:
            rejects.add(c)
        if c == time_col and dataset in ["PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]:
            rejects.add(c)
        if df[c].dtype == object:
            rejects.add(c)
        if any(t in cn for t in reject_terms):
            rejects.add(c)
        if cn in {"dataset", "condition", "source_file", "battery_id", "asset_id", "bearing_id"}:
            rejects.add(c)

    if dataset in ["NASA C-MAPSS", "NASA Battery"] and time_col is not None:
        if time_col in rejects and norm(time_col) in {"cycle", "discharge_index", "time_index"}:
            rejects.remove(time_col)

    features = []
    for c in df.columns:
        if c in rejects:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 20 and s.nunique(dropna=True) > 2:
                features.append(c)

    max_features = 120 if dataset == "NASA C-MAPSS" else 80
    if len(features) > max_features:
        variances = df[features].apply(pd.to_numeric, errors="coerce").var(numeric_only=True).sort_values(ascending=False)
        features = list(variances.head(max_features).index)

    return features, sorted([c for c in rejects if c in df.columns])


def phase1_feature_path(phase1_run: Path, dataset: str) -> Path:
    p = phase1_run / "02_CLEAN_FEATURES" / FEATURE_FILES[dataset]
    if not p.exists():
        raise FileNotFoundError(f"Missing Phase-1 feature file for {dataset}: {p}")
    return p


@dataclass
class DatasetPack:
    dataset: str
    df: pd.DataFrame
    asset_col: str
    time_col: Optional[str]
    target_col: str
    target_note: str
    features: List[str]
    rejected_features: List[str]


def load_dataset_pack(phase1_run: Path, dataset: str) -> DatasetPack:
    p = phase1_feature_path(phase1_run, dataset)
    df = pd.read_csv(p, low_memory=False)
    asset_col = detect_asset_col(df, dataset)
    time_col = detect_time_col(df)
    y, note = derive_target(df, dataset, asset_col)
    df = df.copy()
    df["__nn_target__"] = y
    df = df[np.isfinite(df["__nn_target__"].to_numpy(float))].reset_index(drop=True)
    asset_col = detect_asset_col(df, dataset)
    time_col = detect_time_col(df)
    features, rejected = safe_feature_columns(df, dataset, asset_col, time_col)
    if not features:
        raise ValueError(f"{dataset}: no safe NN features selected.")
    return DatasetPack(dataset, df, asset_col, time_col, "__nn_target__", note, features, rejected)


def make_outer_folds(pack: DatasetPack, mode: str, seed: int) -> List[Dict[str, Any]]:
    groups = sorted(pack.df[pack.asset_col].astype(str).unique().tolist())
    max_folds = {
        "NASA C-MAPSS": 4 if mode == "quick" else 10,
        "NASA Battery": 3 if mode == "quick" else 10,
        "PRONOSTIA/FEMTO": 3 if mode == "quick" else 6,
        "XJTU-SY": 3 if mode == "quick" else 15,
        "IMS": 3 if mode == "quick" else 4,
    }[pack.dataset]
    if len(groups) <= max_folds:
        test_groups = groups
    else:
        rng = np.random.default_rng(seed)
        keep = set([groups[0], groups[-1]])
        rem = [g for g in groups if g not in keep]
        need = max_folds - len(keep)
        if need > 0:
            keep.update(rng.choice(rem, size=min(need, len(rem)), replace=False).tolist())
        test_groups = sorted(keep)

    folds = []
    all_groups = set(groups)
    rng = np.random.default_rng(seed)
    for i, tg in enumerate(test_groups, start=1):
        test_group_set = {tg}
        remaining = sorted(all_groups - test_group_set)
        if len(remaining) < 2:
            continue
        cal_n = max(1, int(math.ceil(0.20 * len(remaining))))
        cal_groups = set(rng.choice(remaining, size=cal_n, replace=False).tolist())
        fit_groups = set(remaining) - cal_groups
        folds.append({
            "fold": f"LOGO_{tg}",
            "fold_index": i,
            "fit_groups": fit_groups,
            "cal_groups": cal_groups,
            "test_groups": test_group_set,
        })
    return folds


def subset_by_groups(pack: DatasetPack, groups: set) -> pd.DataFrame:
    return pack.df[pack.df[pack.asset_col].astype(str).isin(groups)].copy()


def order_df(pack: DatasetPack, df: pd.DataFrame) -> pd.DataFrame:
    if pack.time_col is not None:
        return df.sort_values([pack.asset_col, pack.time_col]).reset_index(drop=False).rename(columns={"index": "__row_index__"})
    return df.sort_values([pack.asset_col]).reset_index(drop=False).rename(columns={"index": "__row_index__"})


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_tabular_arrays(pack: DatasetPack, fit_df: pd.DataFrame, cal_df: pd.DataFrame, test_df: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_fit = scaler.fit_transform(imputer.fit_transform(fit_df[pack.features]))
    X_cal = scaler.transform(imputer.transform(cal_df[pack.features]))
    X_test = scaler.transform(imputer.transform(test_df[pack.features]))

    y_fit = fit_df[pack.target_col].to_numpy(float)
    y_cal = cal_df[pack.target_col].to_numpy(float)
    y_test = test_df[pack.target_col].to_numpy(float)

    meta_test = test_df[[pack.asset_col]].copy()
    meta_test = meta_test.rename(columns={pack.asset_col: "asset_id"})
    meta_test["row_index"] = test_df.index.values
    meta_test["time_like"] = test_df[pack.time_col].values if pack.time_col else np.arange(len(test_df))

    return X_fit, y_fit, X_cal, y_cal, X_test, y_test, meta_test


def build_windows_for_df(pack: DatasetPack, df: pd.DataFrame, features: List[str], window: int, imputer=None, scaler=None, fit_transform=False):
    df_ord = order_df(pack, df)
    X_rows = df_ord[features]
    if fit_transform:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(imputer.fit_transform(X_rows))
    else:
        X_scaled = scaler.transform(imputer.transform(X_rows))

    df_ord = df_ord.reset_index(drop=True)
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    y_all = df_ord[pack.target_col].to_numpy(float)

    Xw, yw, meta_rows = [], [], []
    for group, idx in df_ord.groupby(pack.asset_col).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        local_window = min(window, len(idx))
        stride = 2 if pack.dataset == "NASA C-MAPSS" and len(idx) > 80 else 1
        for end_pos in range(local_window - 1, len(idx), stride):
            start_pos = end_pos - local_window + 1
            ids = idx[start_pos:end_pos + 1]
            arr = X_scaled[ids]
            if len(arr) < window:
                pad = np.repeat(arr[:1], window - len(arr), axis=0)
                arr = np.vstack([pad, arr])
            Xw.append(arr)
            yw.append(y_all[idx[end_pos]])
            row = df_ord.iloc[idx[end_pos]]
            meta_rows.append({
                "asset_id": group,
                "row_index": int(row["__row_index__"]),
                "time_like": row[pack.time_col] if pack.time_col else end_pos,
            })
    if not Xw:
        return np.empty((0, window, len(features)), dtype=np.float32), np.empty((0,), dtype=float), pd.DataFrame(), imputer, scaler
    return np.stack(Xw).astype(np.float32), np.asarray(yw, dtype=float), pd.DataFrame(meta_rows), imputer, scaler


def prepare_sequence_arrays(pack: DatasetPack, fit_df: pd.DataFrame, cal_df: pd.DataFrame, test_df: pd.DataFrame, mode: str):
    window = WINDOWS_QUICK[pack.dataset] if mode == "quick" else WINDOWS_FULL[pack.dataset]
    X_fit, y_fit, _, imputer, scaler = build_windows_for_df(pack, fit_df, pack.features, window, fit_transform=True)
    X_cal, y_cal, _, _, _ = build_windows_for_df(pack, cal_df, pack.features, window, imputer=imputer, scaler=scaler, fit_transform=False)
    X_test, y_test, meta_test, _, _ = build_windows_for_df(pack, test_df, pack.features, window, imputer=imputer, scaler=scaler, fit_transform=False)
    return X_fit, y_fit, X_cal, y_cal, X_test, y_test, meta_test, window


class MLPRegressorTorch(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class GRURegressor(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.gru = nn.GRU(input_size=n_in, hidden_size=64, num_layers=2, batch_first=True, dropout=0.15)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class TCNBlock(nn.Module):
    def __init__(self, n_in, n_out, dilation):
        super().__init__()
        self.conv = nn.Conv1d(n_in, n_out, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(n_out)
        self.drop = nn.Dropout(0.10)
        self.res = nn.Conv1d(n_in, n_out, kernel_size=1) if n_in != n_out else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., :x.shape[-1]]
        r = self.res(x)
        if r.shape[-1] != y.shape[-1]:
            r = r[..., :y.shape[-1]]
        return self.drop(self.bn(self.relu(y + r)))


class TCNRegressor(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.blocks = nn.Sequential(TCNBlock(n_in, 64, 1), TCNBlock(64, 64, 2), TCNBlock(64, 64, 4))
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.blocks(z)
        z = z.mean(dim=-1)
        return self.head(z)


def train_torch_model(model, X_fit, y_fit, X_cal, y_cal, device, epochs, patience, batch_size, lr, seed):
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(ArrayDataset(X_fit, y_fit), batch_size=batch_size, shuffle=True, drop_last=False)
    Xc = torch.tensor(X_cal, dtype=torch.float32, device=device)
    yc = torch.tensor(y_cal.reshape(-1, 1), dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    bad = 0
    rows = []
    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xc), yc).detach().cpu())
        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        rows.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_torch(model, X, device, batch_size=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.clip(np.concatenate(preds), 0, 1) if preds else np.array([], dtype=float)


def categories(y):
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), "safe", dtype=object)
    out[y <= 0.50] = "monitor"
    out[y <= 0.25] = "urgent"
    out[y <= 0.10] = "critical"
    return out


def rank_cat(cats):
    order = {"critical": 0, "urgent": 1, "monitor": 2, "safe": 3}
    return np.asarray([order.get(str(c), 3) for c in cats])


def metric_dict(y_true, y_pred, lower, upper) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 0, 1)
    lo = np.clip(np.asarray(lower, dtype=float), 0, 1)
    hi = np.clip(np.asarray(upper, dtype=float), 0, 1)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(lo) & np.isfinite(hi)
    y, p, lo, hi = y[m], p[m], lo[m], hi[m]
    if len(y) == 0:
        return {}
    true_cat = categories(y)
    point_cat = categories(p)
    interval_cat = categories(lo)
    tr, pr, ir = rank_cat(true_cat), rank_cat(point_cat), rank_cat(interval_cat)
    uc = np.isin(true_cat, ["urgent", "critical"])
    critical = true_cat == "critical"
    covered = (y >= lo) & (y <= hi)
    point_false = (point_cat == "safe") & uc
    interval_false = (interval_cat == "safe") & uc
    point_under = pr > tr
    interval_under = ir > tr
    point_over = pr < tr
    interval_over = ir < tr
    return {
        "n_rows": int(len(y)),
        "n_urgent_critical": int(uc.sum()),
        "n_critical": int(critical.sum()),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "empirical_coverage": float(np.mean(covered)),
        "urgent_critical_coverage": float(np.mean(covered[uc])) if uc.any() else np.nan,
        "critical_coverage": float(np.mean(covered[critical])) if critical.any() else np.nan,
        "point_false_safe_rate": float(np.mean(point_false)),
        "interval_false_safe_rate": float(np.mean(interval_false)),
        "false_safe_reduction": float(np.mean(point_false) - np.mean(interval_false)),
        "point_underwarning_rate": float(np.mean(point_under)),
        "interval_underwarning_rate": float(np.mean(interval_under)),
        "underwarning_reduction": float(np.mean(point_under) - np.mean(interval_under)),
        "point_overwarning_rate": float(np.mean(point_over)),
        "interval_overwarning_rate": float(np.mean(interval_over)),
        "overwarning_increase": float(np.mean(interval_over) - np.mean(point_over)),
        "mean_interval_width": float(np.mean(hi - lo)),
        "median_interval_width": float(np.median(hi - lo)),
        "covered_array": covered,
        "true_category_array": true_cat,
        "point_category_array": point_cat,
        "interval_category_array": interval_cat,
    }


def clean_metric(m: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in m.items() if not k.endswith("_array")}


def quantile_safe(v, q):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(max(0.0, np.quantile(v, q))) if len(v) else 0.0


def interval_variants(y_cal, p_cal, p_test):
    y_cal = np.asarray(y_cal, dtype=float)
    p_cal = np.asarray(p_cal, dtype=float)
    p_test = np.asarray(p_test, dtype=float)
    abs_res = np.abs(y_cal - p_cal)
    low_res = p_cal - y_cal
    high_res = y_cal - p_cal
    variants = []
    for q in [0.90, 0.95, 0.99]:
        qq = quantile_safe(abs_res, q)
        variants.append((f"global_{int(q*100)}", np.clip(p_test - qq, 0, 1), np.clip(p_test + qq, 0, 1), {"q_abs": qq}))
    qlow = quantile_safe(low_res, 0.97)
    qhigh = quantile_safe(high_res, 0.90)
    guard = np.isin(categories(p_test), ["monitor", "urgent", "critical"])
    lo = np.clip(p_test - np.where(guard, qlow * 1.20, qlow), 0, 1)
    hi = np.clip(p_test + np.where(guard, qhigh * 1.20, qhigh), 0, 1)
    variants.append(("twosided_guard_l97_u90_b12", lo, hi, {"q_low": qlow, "q_high": qhigh}))
    return variants


def gate_candidate(row: Dict[str, Any], dataset: str):
    fail = []
    cov = float(row.get("empirical_coverage", np.nan))
    urg = row.get("urgent_critical_coverage", np.nan)
    fs = float(row.get("interval_false_safe_rate", np.nan))
    under = float(row.get("interval_underwarning_rate", np.nan))
    width = float(row.get("mean_interval_width", np.nan))
    coverage_ok = cov >= GATES["coverage_min"]
    urgent_ok = pd.isna(urg) or float(urg) >= GATES["urgent_min"]
    false_ok = fs <= GATES["max_false_safe"]
    under_ok = under <= GATES["max_underwarning"]
    width_ok = width <= GATES["width_max"][dataset]
    not_full = width <= 0.75
    reduction_ok = row.get("false_safe_reduction", 0) >= -1e-12 and row.get("underwarning_reduction", 0) >= -1e-12
    checks = [coverage_ok, urgent_ok, false_ok, under_ok, width_ok, not_full, reduction_ok]
    strict = all(checks)
    if not coverage_ok: fail.append(f"coverage {cov:.3f}<0.88")
    if not urgent_ok: fail.append(f"urgent {float(urg):.3f}<0.80")
    if not false_ok: fail.append("false-safe gate")
    if not under_ok: fail.append("underwarning gate")
    if not width_ok: fail.append(f"width {width:.3f}>{GATES['width_max'][dataset]:.2f}")
    if not not_full: fail.append("full-width reject")
    if not reduction_ok: fail.append("risk reduction negative")
    pass_count = int(sum(bool(x) for x in checks))
    score = int(strict) * 1000 + pass_count * 50 + (0 if not np.isfinite(cov) else cov * 5) + (0 if pd.isna(urg) else float(urg) * 4) - (999 if not np.isfinite(width) else width * 2) - fs * 20
    return strict, ("passes strict" if strict else "; ".join(fail)), pass_count, float(score)


def prediction_frame(dataset, model_name, fold_name, interval_name, meta_test, y_test, p_test, lo, hi):
    m = metric_dict(y_test, p_test, lo, hi)
    return pd.DataFrame({
        "dataset": dataset,
        "nn_model": model_name,
        "fold": fold_name,
        "interval_variant": interval_name,
        "asset_id": meta_test["asset_id"].astype(str).values if "asset_id" in meta_test.columns else "",
        "row_index": meta_test["row_index"].values if "row_index" in meta_test.columns else np.arange(len(y_test)),
        "time_like": meta_test["time_like"].values if "time_like" in meta_test.columns else np.arange(len(y_test)),
        "y_true": y_test,
        "y_pred": p_test,
        "lower": lo,
        "upper": hi,
        "covered": m.get("covered_array", np.zeros(len(y_test), dtype=bool)),
        "true_category": m.get("true_category_array", [""] * len(y_test)),
        "point_category": m.get("point_category_array", [""] * len(y_test)),
        "interval_category": m.get("interval_category_array", [""] * len(y_test)),
        "interval_width": np.asarray(hi) - np.asarray(lo),
    })


def train_and_evaluate_fold(pack, fold, model_kind, mode, device, args, seed):
    fit_df = subset_by_groups(pack, fold["fit_groups"])
    cal_df = subset_by_groups(pack, fold["cal_groups"])
    test_df = subset_by_groups(pack, fold["test_groups"])

    if model_kind == "MLP_TABULAR":
        X_fit, y_fit, X_cal, y_cal, X_test, y_test, meta_test = prepare_tabular_arrays(pack, fit_df, cal_df, test_df)
        model = MLPRegressorTorch(X_fit.shape[1])
        batch_size = args.batch_size
    else:
        X_fit, y_fit, X_cal, y_cal, X_test, y_test, meta_test, window = prepare_sequence_arrays(pack, fit_df, cal_df, test_df, mode)
        if len(X_fit) < 20 or len(X_cal) < 5 or len(X_test) < 5:
            raise ValueError(f"{pack.dataset} {model_kind}: too few windows in fold {fold['fold']}.")
        model = GRURegressor(X_fit.shape[-1]) if model_kind == "GRU_SEQUENCE" else TCNRegressor(X_fit.shape[-1])
        batch_size = min(args.batch_size, 512)

    if len(X_fit) == 0 or len(X_cal) == 0 or len(X_test) == 0:
        raise ValueError(f"{pack.dataset} {model_kind}: empty fit/cal/test arrays.")

    start = time.perf_counter()
    model, loss_df = train_torch_model(
        model, X_fit, y_fit, X_cal, y_cal, device,
        args.epochs_quick if mode == "quick" else args.epochs,
        args.patience_quick if mode == "quick" else args.patience,
        batch_size, args.lr, seed
    )
    train_time = time.perf_counter() - start

    start_pred = time.perf_counter()
    p_cal = predict_torch(model, X_cal, device)
    p_test = predict_torch(model, X_test, device)
    pred_time = time.perf_counter() - start_pred

    fold_rows, pred_rows = [], []
    point_metrics = metric_dict(y_test, p_test, p_test, p_test)
    fold_rows.append({
        "dataset": pack.dataset, "nn_model": model_kind, "fold": fold["fold"], "interval_variant": "point_only",
        "candidate_id": f"{model_kind}__point_only", "fit_groups": len(fold["fit_groups"]),
        "cal_groups": len(fold["cal_groups"]), "test_groups": len(fold["test_groups"]),
        "training_time_sec": train_time, "prediction_time_sec": pred_time,
        **clean_metric(point_metrics),
    })

    for ivar, lo, hi, qinfo in interval_variants(y_cal, p_cal, p_test):
        m = metric_dict(y_test, p_test, lo, hi)
        fold_rows.append({
            "dataset": pack.dataset, "nn_model": model_kind, "fold": fold["fold"], "interval_variant": ivar,
            "candidate_id": f"{model_kind}__{ivar}", "fit_groups": len(fold["fit_groups"]),
            "cal_groups": len(fold["cal_groups"]), "test_groups": len(fold["test_groups"]),
            "training_time_sec": train_time, "prediction_time_sec": pred_time,
            **clean_metric(m), **qinfo,
        })
        pred_rows.append(prediction_frame(pack.dataset, model_kind, fold["fold"], ivar, meta_test, y_test, p_test, lo, hi))

    loss_df["dataset"] = pack.dataset
    loss_df["nn_model"] = model_kind
    loss_df["fold"] = fold["fold"]
    loss_df["training_time_sec"] = train_time

    return pd.DataFrame(fold_rows), pd.concat(pred_rows, ignore_index=True), loss_df


def append_nn_ensemble_interval_candidates(pred_df: pd.DataFrame) -> pd.DataFrame:
    

    if pred_df is None or pred_df.empty:
        return pred_df
    required = {"dataset", "fold", "nn_model", "interval_variant", "y_true", "y_pred", "lower", "upper"}
    if not required.issubset(set(pred_df.columns)):
        return pred_df

    base = pred_df.copy()
    obs_cols = ["dataset", "fold", "interval_variant"]
    for c in ["asset_id", "row_index", "time_like"]:
        if c in base.columns:
            obs_cols.append(c)

    rows = []
    for _, g in base.groupby(obs_cols, dropna=False):
        if g["nn_model"].nunique() < 2:
            continue
        first = g.iloc[0].to_dict()
        y_pred = float(pd.to_numeric(g["y_pred"], errors="coerce").mean())
        lower = float(pd.to_numeric(g["lower"], errors="coerce").min())
        upper = float(pd.to_numeric(g["upper"], errors="coerce").max())
        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))
        if lower > upper:
            lower, upper = upper, lower
        first.update({
            "nn_model": "NN_ENSEMBLE_INTERVAL_ENVELOPE",
            "candidate_id": f"NN_ENSEMBLE_INTERVAL_ENVELOPE__{first.get('interval_variant', 'unknown')}",
            "y_pred": max(0.0, min(1.0, y_pred)),
            "lower": lower,
            "upper": upper,
            "interval_width": upper - lower,
            "ensemble_source_nn_count": int(g["nn_model"].nunique()),
        })
        rows.append(first)
    if not rows:
        return pred_df
    return pd.concat([pred_df, pd.DataFrame(rows)], ignore_index=True, sort=False)


def summarize_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, model, ivar), d in pred_df.groupby(["dataset", "nn_model", "interval_variant"]):
        m = metric_dict(d["y_true"], d["y_pred"], d["lower"], d["upper"])
        row = {
            "dataset": dataset, "nn_model": model, "interval_variant": ivar,
            "candidate_id": f"{model}__{ivar}", "folds": d["fold"].nunique(),
            "assets": d["asset_id"].nunique() if "asset_id" in d.columns else np.nan,
            **clean_metric(m),
        }
        strict, reason, pass_count, score = gate_candidate(row, dataset)
        row.update({"strict_promotion": strict, "blocking_reason": reason, "gate_pass_count": pass_count, "gate_rank_score": score})
        rows.append(row)
    return pd.DataFrame(rows)


def choose_nn_by_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in summary.groupby("dataset"):
        strict = g[g["strict_promotion"].astype(bool)].copy()
        if not strict.empty:
            r = strict.sort_values(["gate_rank_score", "RMSE"], ascending=[False, True]).iloc[0].to_dict()
            decision = "NN_STRICT_PASS"
        else:
            r = g.sort_values(["gate_rank_score", "RMSE"], ascending=[False, True]).iloc[0].to_dict()
            decision = "NN_NEAREST_MISS"
        r["nn_decision"] = decision
        rows.append(r)
    return pd.DataFrame(rows)


def load_locked_metrics(locked_run: Optional[Path]) -> pd.DataFrame:
    if not locked_run:
        return pd.DataFrame()
    p = locked_run / "02_TABLES" / "SINGLE_FILE_SELECTED_CANDIDATE_METRICS.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    keep = ["dataset", "selected_candidate", "RMSE", "MAE", "R2", "empirical_coverage", "urgent_critical_coverage", "interval_false_safe_rate", "interval_underwarning_rate", "mean_interval_width", "passed_exact_old_gate"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df.rename(columns={c: f"locked_{c}" for c in df.columns if c != "dataset"})


def compare_nn_vs_locked(nn_selected: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    if locked.empty or nn_selected.empty:
        return pd.DataFrame()
    out = nn_selected.merge(locked, on="dataset", how="left")
    for k in ["RMSE", "MAE", "R2", "empirical_coverage", "urgent_critical_coverage", "mean_interval_width"]:
        lk = f"locked_{k}"
        if k in out.columns and lk in out.columns:
            out[f"nn_minus_locked_{k}"] = out[k] - out[lk]
    out["interpretation"] = np.where(
        out["strict_promotion"].astype(bool),
        "NN baseline passes safety gate; compare point metrics but keep main objective as safety-calibrated decisions.",
        "NN baseline useful for point-prediction benchmarking but does not replace locked safety-calibrated selected model.",
    )
    return out


def plot_loss_curves(loss_df: pd.DataFrame, out_dir: Path) -> None:
    if loss_df.empty:
        return
    ensure_dir(out_dir)
    for (dataset, model), d in loss_df.groupby(["dataset", "nn_model"]):
        plt.figure(figsize=(7, 4))
        for fold, s in d.groupby("fold"):
            plt.plot(s["epoch"], s["val_loss"], alpha=0.35, label=str(fold)[:20])
        plt.xlabel("Epoch")
        plt.ylabel("Validation MSE")
        plt.title(f"{dataset}: {model} validation loss")
        if d["fold"].nunique() <= 8:
            plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(out_dir / f"{slug(dataset)}__{model.lower()}__loss.png", dpi=220, bbox_inches="tight")
        plt.close()


def plot_true_vs_pred(pred_df: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> None:
    if pred_df.empty or selected.empty:
        return
    ensure_dir(out_dir)
    for _, row in selected.iterrows():
        ds, model, ivar = row["dataset"], row["nn_model"], row["interval_variant"]
        d = pred_df[(pred_df["dataset"] == ds) & (pred_df["nn_model"] == model) & (pred_df["interval_variant"] == ivar)].copy()
        if d.empty:
            continue
        if len(d) > 12000:
            d = d.sample(12000, random_state=42)
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(d["y_true"], d["y_pred"], s=5, alpha=0.30)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("True normalized RUL/SOH")
        plt.ylabel("NN predicted normalized RUL/SOH")
        plt.title(f"{ds}: {model} true vs predicted")
        plt.tight_layout()
        plt.savefig(out_dir / f"{slug(ds)}__{model.lower()}__true_vs_pred.png", dpi=220, bbox_inches="tight")
        plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_run", required=True)
    p.add_argument("--locked_run", default="")
    p.add_argument("--output_root", default=None)
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--epochs_quick", type=int, default=45)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--patience_quick", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--skip_sequence", action="store_true")
    p.add_argument("--only_dataset", default="")
    p.add_argument("--nn_protocol", choices=["auto", "generic", "strong_dataset_specific"], default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    if torch is None:
        raise ImportError(f"PyTorch is required for this NN baseline script. Import error was: {TORCH_IMPORT_ERROR}")

    set_seed(args.seed)
    phase1_run = Path(args.phase1_run)
    locked_run = Path(args.locked_run) if args.locked_run else None
    output_root = Path(args.output_root)
    suffix = "_junk" if args.mode == "quick" else ""
    run_dir = output_root / f"Q1_NN_BASELINES_RUN_{now_stamp()}{suffix}"

    dirs = {
        "manifest": ensure_dir(run_dir / "00_RUN_MANIFEST"),
        "audit": ensure_dir(run_dir / "01_INPUT_AUDIT"),
        "results": ensure_dir(run_dir / "02_NN_CANDIDATE_RESULTS"),
        "preds": ensure_dir(run_dir / "03_NN_PREDICTIONS"),
        "loss": ensure_dir(run_dir / "04_NN_LOSS_CURVES"),
        "figs": ensure_dir(run_dir / "05_NN_FIGURES"),
        "compare": ensure_dir(run_dir / "06_NN_VS_LOCKED"),
        "bench": ensure_dir(run_dir / "07_BENCHMARK_READY_SUMMARY"),
    }

    banner("Q1 NEURAL-NETWORK BASELINES FOR BENCHMARKING")
    print(f"SCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"Phase-1 run: {phase1_run}")
    print(f"Locked run : {locked_run if locked_run else 'not provided'}")
    print(f"Output run : {run_dir}")
    print(f"Mode       : {args.mode}")
    device = get_device(args.device)
    print(f"Device     : {device}")
    print("Role       : NN baselines/benchmarking only; not replacing locked selected models unless gates pass.")

    save_json(dirs["manifest"] / "run_config.json", {
        "script_version": SCRIPT_VERSION,
        "phase1_run": str(phase1_run),
        "locked_run": str(locked_run) if locked_run else "",
        "output_run": str(run_dir),
        "mode": args.mode,
        "device": device,
        "seed": args.seed,
        "epochs": args.epochs,
        "epochs_quick": args.epochs_quick,
        "patience": args.patience,
        "patience_quick": args.patience_quick,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "skip_sequence": args.skip_sequence,
        "old_outputs_used": False,
        "model_role": "NN baseline/benchmarking, not final replacement by default",
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
    })

    if not phase1_run.exists():
        raise FileNotFoundError(phase1_run)

    datasets = DATASETS if not args.only_dataset else [args.only_dataset]
    packs, audit_rows = {}, []
    banner("LOADING CLEAN PHASE-1 TABLES")
    for ds in datasets:
        pack = load_dataset_pack(phase1_run, ds)
        packs[ds] = pack
        exp = EXPECTED_COUNTS[ds]
        row = {
            "dataset": ds,
            "rows": len(pack.df),
            "expected_rows": exp["rows"],
            "rows_match_expected": len(pack.df) == exp["rows"],
            "assets": pack.df[pack.asset_col].nunique(),
            "expected_assets": exp["assets"],
            "assets_match_expected": pack.df[pack.asset_col].nunique() == exp["assets"],
            "asset_col": pack.asset_col,
            "time_col": pack.time_col or "",
            "target_note": pack.target_note,
            "safe_feature_count": len(pack.features),
            "rejected_feature_count": len(pack.rejected_features),
            "safe_features": "; ".join(pack.features[:200]),
            "rejected_features": "; ".join(pack.rejected_features[:200]),
        }
        audit_rows.append(row)
        print(f"{ds}: rows={row['rows']:,}/{row['expected_rows']:,}, assets={row['assets']}/{row['expected_assets']}, features={row['safe_feature_count']}, target={pack.target_note}")
    save_csv(pd.DataFrame(audit_rows), dirs["audit"] / "nn_input_feature_audit.csv")

    model_kinds = ["MLP_TABULAR"]
    if not args.skip_sequence:
        model_kinds += ["GRU_SEQUENCE", "TCN_LITE_SEQUENCE"]

    all_fold, all_pred, all_loss, errors = [], [], [], []
    banner("TRAINING NN BASELINES")
    for ds, pack in packs.items():
        folds = make_outer_folds(pack, args.mode, args.seed)
        print(f"\n{ds}: folds={len(folds)}, models={model_kinds}", flush=True)
        for model_kind in model_kinds:
            for fold in folds:
                banner(f"{ds} | {model_kind} | {fold['fold']}")
                try:
                    fr, pr, lrdf = train_and_evaluate_fold(pack, fold, model_kind, args.mode, device, args, args.seed + fold["fold_index"])
                    all_fold.append(fr); all_pred.append(pr); all_loss.append(lrdf)
                    print(f"done: rows={len(pr):,}, last_val_loss={lrdf['val_loss'].iloc[-1]:.6f}, epochs={len(lrdf)}", flush=True)
                except Exception as e:
                    tb = traceback.format_exc()
                    errors.append({"dataset": ds, "nn_model": model_kind, "fold": fold["fold"], "error": repr(e), "traceback": tb})
                    print(f"ERROR: {repr(e)}", flush=True)
                    (dirs["manifest"] / f"ERROR__{slug(ds)}__{model_kind}__{norm(fold['fold'])}.txt").write_text(tb, encoding="utf-8")

    fold_df = pd.concat(all_fold, ignore_index=True) if all_fold else pd.DataFrame()
    pred_df = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    pred_df = append_nn_ensemble_interval_candidates(pred_df)
    loss_df = pd.concat(all_loss, ignore_index=True) if all_loss else pd.DataFrame()
    errors_df = pd.DataFrame(errors)

    save_csv(fold_df, dirs["results"] / "01_nn_fold_metrics_all.csv")
    save_csv(pred_df, dirs["preds"] / "01_nn_predictions_all.csv")
    save_csv(loss_df, dirs["loss"] / "01_nn_loss_curves_all.csv")
    if not errors_df.empty:
        save_csv(errors_df, dirs["manifest"] / "NN_ERRORS.csv")

    summary = summarize_predictions(pred_df)
    selected = choose_nn_by_dataset(summary) if not summary.empty else pd.DataFrame()
    save_csv(summary, dirs["results"] / "02_nn_candidate_summary.csv")
    save_csv(selected, dirs["results"] / "03_nn_selected_by_dataset.csv")

    locked = load_locked_metrics(locked_run)
    compare = compare_nn_vs_locked(selected, locked)
    save_csv(locked, dirs["compare"] / "01_locked_selected_metrics_reference.csv")
    save_csv(compare, dirs["compare"] / "02_nn_vs_locked_selected_models.csv")

    if not selected.empty:
        bench_cols = ["dataset", "nn_model", "candidate_id", "nn_decision", "RMSE", "MAE", "R2", "empirical_coverage", "urgent_critical_coverage", "interval_false_safe_rate", "interval_underwarning_rate", "mean_interval_width", "strict_promotion", "blocking_reason"]
        bench = selected[[c for c in bench_cols if c in selected.columns]].copy()
        bench["benchmark_role"] = "Neural baseline for point-prediction comparison; final paper objective remains safety-calibrated maintenance decision intervals."
        save_csv(bench, dirs["bench"] / "01_benchmark_ready_nn_summary.csv")
    else:
        bench = pd.DataFrame()
        save_csv(bench, dirs["bench"] / "01_benchmark_ready_nn_summary.csv")

    plot_loss_curves(loss_df, dirs["figs"])
    plot_true_vs_pred(pred_df, selected, dirs["figs"])

    verdict = {
        "run_dir": str(run_dir),
        "script_version": SCRIPT_VERSION,
        "mode": args.mode,
        "datasets_requested": datasets,
        "datasets_completed_in_summary": int(selected["dataset"].nunique()) if not selected.empty else 0,
        "fold_metric_rows": int(len(fold_df)),
        "prediction_rows": int(len(pred_df)),
        "loss_rows": int(len(loss_df)),
        "errors": int(len(errors_df)),
        "any_nn_strict_pass": bool(selected["strict_promotion"].astype(bool).any()) if "strict_promotion" in selected.columns else False,
        "datasets_with_nn_strict_pass": selected[selected["strict_promotion"].astype(bool)]["dataset"].tolist() if "strict_promotion" in selected.columns else [],
        "interpretation": "Use NN results for benchmark/ablation. Do not replace locked final models unless a NN passes the same safety gates and the manuscript explicitly justifies the switch.",
    }
    save_json(dirs["manifest"] / "NN_BASELINE_VERDICT.json", verdict)

    banner("NN BASELINE RUN COMPLETE")
    print(f"Run directory: {run_dir}")
    print(f"Candidate summary: {dirs['results'] / '02_nn_candidate_summary.csv'}")
    print(f"Selected NN by dataset: {dirs['results'] / '03_nn_selected_by_dataset.csv'}")
    print(f"NN vs locked: {dirs['compare'] / '02_nn_vs_locked_selected_models.csv'}")
    print(f"Benchmark summary: {dirs['bench'] / '01_benchmark_ready_nn_summary.csv'}")

    if not selected.empty:
        show_cols = ["dataset", "nn_model", "interval_variant", "nn_decision", "RMSE", "MAE", "R2", "empirical_coverage", "urgent_critical_coverage", "interval_false_safe_rate", "interval_underwarning_rate", "mean_interval_width", "strict_promotion", "blocking_reason"]
        print("\nNN SELECTED BY DATASET")
        print(selected[[c for c in show_cols if c in selected.columns]].to_string(index=False))

    print("\nVERDICT")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# Q1_MASTER_REPAIR_PATCH_V4
# END_Q1_MASTER_REPAIR_PATCH_V4

