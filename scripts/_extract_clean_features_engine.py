
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

# Public release script: 01_extract_clean_features.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

# ============================================================
# 00A_Q1_RAW_EXTRACTION_AND_AUDIT.py
# Fresh raw-data extraction and audit only
# ============================================================
#
# Purpose:
#   Build clean feature tables and audit files from the confirmed raw datasets.
#
# Run quick:
# runfile(
#     args=r"--mode quick",
# )
#
# Run full:
# runfile(
#     args=r"--mode full",
# )
# ============================================================


import argparse
import json
import math
import os
import platform
import re
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

OUTPUT_ROOT_DEFAULT = Path(r"pm_interval_safety_workspace/outputs")

CMAPSS_ROOT_DEFAULT = Path(r"pm_interval_safety_workspace")
BATTERY_ROOT_DEFAULT = Path(r"pm_interval_safety_workspace")
PRONOSTIA_ROOT_DEFAULT = Path(r"pm_interval_safety_workspace")
XJTU_BASE_DEFAULT = Path(
    r"pm_interval_safety_workspace"
    r"\raw__Data__XJTU-SY_Bearing_Datasets\XJTU-SY_Bearing_Datasets"
)
IMS_ROOT_DEFAULT = Path(r"pm_interval_safety_workspace")

PRONOSTIA_WHITELIST = [
    "Bearing1_1", "Bearing1_2",
    "Bearing2_1", "Bearing2_2",
    "Bearing3_1", "Bearing3_2",
]

XJTU_EXPECTED = [
    ("35Hz12kN", "Bearing1_1", 123),
    ("35Hz12kN", "Bearing1_2", 161),
    ("35Hz12kN", "Bearing1_3", 158),
    ("35Hz12kN", "Bearing1_4", 122),
    ("35Hz12kN", "Bearing1_5", 52),
    ("37.5Hz11kN", "Bearing2_1", 491),
    ("37.5Hz11kN", "Bearing2_2", 161),
    ("37.5Hz11kN", "Bearing2_3", 533),
    ("37.5Hz11kN", "Bearing2_4", 42),
    ("37.5Hz11kN", "Bearing2_5", 339),
    ("40Hz10kN", "Bearing3_1", 2538),
    ("40Hz10kN", "Bearing3_2", 2496),
    ("40Hz10kN", "Bearing3_3", 371),
    ("40Hz10kN", "Bearing3_4", 1515),
    ("40Hz10kN", "Bearing3_5", 114),
]

IMS_SPECS = [
    {"asset_id": "IMS_1st_Bearing3", "marker": "1st_test", "alt_marker": "", "channels": [4, 5]},
    {"asset_id": "IMS_1st_Bearing4", "marker": "1st_test", "alt_marker": "", "channels": [6, 7]},
    {"asset_id": "IMS_2nd_Bearing1", "marker": "2nd_test", "alt_marker": "", "channels": [0, 1]},
    {"asset_id": "IMS_3rd_Bearing3", "marker": "3rd_test", "alt_marker": "4th_test", "channels": [2, 3]},
]

FORBIDDEN_BY_DATASET = {
    "PRONOSTIA/FEMTO": ["xjtu", "ims", "battery", "cmapss", "c-mapss", "c_mapss"],
    "XJTU-SY": ["femto", "pronostia", "ims", "battery", "cmapss", "c-mapss", "c_mapss"],
    "IMS": ["femto", "pronostia", "xjtu", "battery", "cmapss", "c-mapss", "c_mapss"],
}

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def save_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def print_block(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120, flush=True)

def safe_size_mb(p: Path):
    try:
        return round(p.stat().st_size / 1024 / 1024, 4)
    except Exception:
        return np.nan

def numeric_sort_key(p: Path):
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else p.name

def path_has_any(path: Path, terms: list[str]) -> list[str]:
    s = str(path).lower()
    return [t for t in terms if t in s]

def safe_trapezoid(y, x) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    n = min(len(y), len(x))
    if n < 2:
        return 0.0
    y = y[:n]
    x = x[:n]
    m = np.isfinite(y) & np.isfinite(x)
    y = y[m]
    x = x[m]
    if len(y) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))

def to_float_1d(x):
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([], dtype=float)

def scalar_float(x):
    arr = to_float_1d(x)
    return float(arr[0]) if len(arr) else np.nan

def get_field(obj, name, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        if hasattr(obj, "dtype") and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    except Exception:
        pass
    return default

def as_list(x):
    if isinstance(x, np.ndarray):
        return list(x.reshape(-1))
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None:
        return []
    return [x]

def signal_summary(prefix: str, x):
    arr = to_float_1d(x)
    if len(arr) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_first": np.nan,
            f"{prefix}_last": np.nan,
            f"{prefix}_range": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_first": float(arr[0]),
        f"{prefix}_last": float(arr[-1]),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
    }

def read_numeric_file(path: Path, max_cols=8, min_rows=10):
    attempts = [
        dict(header=None, sep=r"\s+", engine="python"),
        dict(header=None, sep="\t"),
        dict(header=None, sep=";"),
        dict(header=None, sep=","),
        dict(header=None),
    ]
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.empty:
                continue
            num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            if num.shape[0] >= min_rows and num.shape[1] >= 1:
                return num.iloc[:, :max_cols].values.astype(float)
        except Exception:
            continue
    return None

def vibration_features(arr: np.ndarray, max_channels=2):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        channels = [arr]
    else:
        channels = [arr[:, j] for j in range(min(arr.shape[1], max_channels))]
    feats = {}
    for ci, sig in enumerate(channels, start=1):
        sig = np.asarray(sig, dtype=float)
        sig = sig[np.isfinite(sig)]
        if len(sig) == 0:
            sig = np.array([0.0])
        prefix = f"ch{ci}_"
        mean = float(np.mean(sig))
        std = float(np.std(sig))
        rms = float(np.sqrt(np.mean(sig ** 2)))
        absmean = float(np.mean(np.abs(sig)))
        peak = float(np.max(np.abs(sig)))
        feats[prefix + "mean"] = mean
        feats[prefix + "std"] = std
        feats[prefix + "rms"] = rms
        feats[prefix + "absmean"] = absmean
        feats[prefix + "peak"] = peak
        feats[prefix + "crest_factor"] = peak / (rms + 1e-12)
        feats[prefix + "shape_factor"] = rms / (absmean + 1e-12)
        feats[prefix + "kurtosis"] = float(np.mean(((sig - mean) / (std + 1e-12)) ** 4))
        feats[prefix + "skewness"] = float(np.mean(((sig - mean) / (std + 1e-12)) ** 3))
        spec = np.abs(np.fft.rfft(sig[: min(len(sig), 4096)]))
        if len(spec) >= 3:
            feats[prefix + "fft_mean"] = float(np.mean(spec))
            feats[prefix + "fft_std"] = float(np.std(spec))
            feats[prefix + "fft_peak"] = float(np.max(spec))
            feats[prefix + "fft_energy"] = float(np.sum(spec ** 2))
        else:
            feats[prefix + "fft_mean"] = 0.0
            feats[prefix + "fft_std"] = 0.0
            feats[prefix + "fft_peak"] = 0.0
            feats[prefix + "fft_energy"] = 0.0
    return feats

def add_bearing_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["asset_id", "time_index"]).reset_index(drop=True)
    target_values = []
    hi_values = []
    for asset, g in df.groupby("asset_id", sort=False):
        n = len(g)
        idx = np.arange(n, dtype=float)
        target = 1.0 - idx / max(n - 1, 1)
        target_values.extend(target.tolist())
        energy_cols = [c for c in g.columns if ("rms" in c or "peak" in c or "fft_energy" in c)]
        if energy_cols:
            score = g[energy_cols].apply(pd.to_numeric, errors="coerce").median(axis=1).values.astype(float)
            if np.nanmax(score) > np.nanmin(score):
                hi = (score - np.nanmin(score)) / max(np.nanmax(score) - np.nanmin(score), 1e-12)
            else:
                hi = np.zeros_like(score)
        else:
            hi = np.zeros(n)
        hi_values.extend(np.clip(hi, 0, 1).tolist())
    df["target_health"] = np.clip(target_values, 0, 1)
    df["HI_raw"] = np.clip(hi_values, 0, 1)
    return df

def verify_roots(args, dirs):
    rows = []
    def add(dataset, key, path, status, note):
        rows.append({
            "dataset": dataset,
            "key": key,
            "path": str(path),
            "exists": Path(path).exists(),
            "status": status,
            "note": note,
        })
    train_files = sorted(args.cmapss_root.rglob("train_FD*.txt")) if args.cmapss_root.exists() else []
    test_files = sorted(args.cmapss_root.rglob("test_FD*.txt")) if args.cmapss_root.exists() else []
    rul_files = sorted(args.cmapss_root.rglob("RUL_FD*.txt")) if args.cmapss_root.exists() else []
    add("NASA C-MAPSS", "root", args.cmapss_root, "OK" if len(train_files) >= 4 else "MISSING_OR_INCOMPLETE", f"train={len(train_files)}, test={len(test_files)}, RUL={len(rul_files)}")

    mat_files = sorted(args.battery_root.rglob("B*.mat")) if args.battery_root.exists() else []
    add("NASA Battery", "raw_extracted_parent", args.battery_root, "OK" if len(mat_files) >= 4 else "MISSING_OR_INCOMPLETE", f"B*.mat files found recursively={len(mat_files)}")

    p_bad = path_has_any(args.pronostia_root, FORBIDDEN_BY_DATASET["PRONOSTIA/FEMTO"])
    add("PRONOSTIA/FEMTO", "root", args.pronostia_root, "OK" if args.pronostia_root.exists() and not p_bad else "BAD_PATH", f"forbidden_terms={p_bad}")

    found = 0
    exact = 0
    total_files = 0
    for cond, bearing, expected_count in XJTU_EXPECTED:
        folder = args.xjtu_base / cond / bearing
        files = sorted(folder.glob("*.csv")) if folder.exists() else []
        if files:
            found += 1
        if len(files) == expected_count:
            exact += 1
        total_files += len(files)
    x_bad = path_has_any(args.xjtu_base, FORBIDDEN_BY_DATASET["XJTU-SY"])
    add("XJTU-SY", "exact_15_asset_base", args.xjtu_base, "OK" if found == 15 and exact == 15 and not x_bad else "MISSING_OR_INCOMPLETE", f"found={found}/15, exact_file_counts={exact}/15, total_csv={total_files}, forbidden_terms={x_bad}")

    ims_files = []
    if args.ims_root.exists():
        for p in args.ims_root.rglob("*"):
            if p.is_file() and p.stat().st_size > 1000:
                low = str(p).lower()
                if "_archive" not in low and "archive" not in low:
                    if ("1st_test" in low or "2nd_test" in low or "3rd_test" in low or "4th_test" in low):
                        if p.suffix.lower() not in [".rar", ".zip", ".7z"]:
                            ims_files.append(p)
    i_bad = path_has_any(args.ims_root, FORBIDDEN_BY_DATASET["IMS"])
    add("IMS", "root", args.ims_root, "OK" if len(ims_files) > 0 and not i_bad else "MISSING_OR_BAD_PATH", f"canonical non-archive candidate files={len(ims_files)}, forbidden_terms={i_bad}")

    df = pd.DataFrame(rows)
    save_csv(df, dirs["audit"] / "source_path_verification.csv")
    print("\nSOURCE PATH VERIFICATION")
    print(df.to_string(index=False))
    bad = df[df["status"].astype(str) != "OK"]
    if not bad.empty:
        raise RuntimeError("One or more source roots failed verification. See source_path_verification.csv")
    return df

def extract_cmapss(args, dirs):
    print_block("EXTRACTING NASA C-MAPSS FROM RAW TXT")
    col_names = ["unit_number", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    frames = []
    audit_rows = []
    train_files = sorted(args.cmapss_root.rglob("train_FD*.txt"))
    test_files = sorted(args.cmapss_root.rglob("test_FD*.txt"))
    rul_files = sorted(args.cmapss_root.rglob("RUL_FD*.txt"))
    for f in train_files:
        subset_match = re.search(r"FD\d+", f.name, flags=re.IGNORECASE)
        subset = subset_match.group(0).upper() if subset_match else f.stem
        print(f"Reading C-MAPSS {subset}: {f}", flush=True)
        raw = pd.read_csv(f, sep=r"\s+", header=None, engine="python")
        raw = raw.iloc[:, :26].copy()
        raw.columns = col_names[: raw.shape[1]]
        if raw.shape[1] != 26:
            raise RuntimeError(f"C-MAPSS file has unexpected column count: {f}, shape={raw.shape}")
        raw["subset"] = subset
        raw = raw.sort_values(["unit_number", "cycle"]).reset_index(drop=True)
        max_cycle = raw.groupby("unit_number")["cycle"].transform("max")
        raw["target_RUL"] = max_cycle - raw["cycle"]
        raw["target_RUL_capped_125"] = np.minimum(raw["target_RUL"], 125)
        raw["target_health"] = np.clip(raw["target_RUL_capped_125"] / 125.0, 0, 1)
        raw["asset_id"] = raw["subset"].astype(str) + "_unit_" + raw["unit_number"].astype(str)
        sensor_cols = [c for c in raw.columns if c.startswith("sensor_")]
        for c in sensor_cols:
            raw[f"{c}_roll_mean_5"] = raw.groupby("unit_number")[c].transform(lambda s: s.rolling(5, min_periods=1).mean())
            raw[f"{c}_roll_std_5"] = raw.groupby("unit_number")[c].transform(lambda s: s.rolling(5, min_periods=1).std()).fillna(0)
            raw[f"{c}_delta_1"] = raw.groupby("unit_number")[c].diff().fillna(0)
        frames.append(raw)
        audit_rows.append({"dataset": "NASA C-MAPSS", "file": str(f), "subset": subset, "rows": len(raw), "units": raw["unit_number"].nunique(), "target_min": raw["target_RUL"].min(), "target_max": raw["target_RUL"].max()})
    df = pd.concat(frames, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    file_inv = pd.DataFrame({
        "file_type": ["train", "test", "RUL"],
        "count": [len(train_files), len(test_files), len(rul_files)],
        "files": ["; ".join(str(p) for p in train_files), "; ".join(str(p) for p in test_files), "; ".join(str(p) for p in rul_files)],
    })
    save_csv(df, dirs["features"] / "cmapss_engine_features_clean_raw.csv")
    save_csv(audit, dirs["audit"] / "cmapss_raw_extraction_audit.csv")
    save_csv(file_inv, dirs["audit"] / "cmapss_file_inventory.csv")
    print(f"C-MAPSS extracted rows={len(df):,}, assets={df['asset_id'].nunique():,}, columns={df.shape[1]}")
    return df, audit

def choose_unique_battery_mats(args, dirs):
    print_block("DISCOVERING NASA BATTERY RAW MAT FILES")
    all_mats = sorted(args.battery_root.rglob("B*.mat"))
    rows = []
    for p in all_mats:
        m = re.match(r"(B\d+)\\.mat$", p.name, flags=re.IGNORECASE)
        if not m:
            continue
        battery_id = m.group(1).upper()
        rows.append({"battery_id": battery_id, "path": str(p), "parent": str(p.parent), "size_bytes": p.stat().st_size, "size_mb": safe_size_mb(p)})
    inv = pd.DataFrame(rows)
    if inv.empty:
        raise RuntimeError(f"No B*.mat files found under {args.battery_root}")
    inv = inv.sort_values(["battery_id", "size_bytes", "path"], ascending=[True, False, True]).reset_index(drop=True)
    inv["rank_within_battery"] = inv.groupby("battery_id").cumcount() + 1
    inv["selected"] = inv["rank_within_battery"] == 1
    inv["duplicate_status"] = np.where(inv["selected"], "SELECTED_CANONICAL", "REJECT_DUPLICATE")
    selected = inv[inv["selected"]].copy()
    save_csv(inv, dirs["audit"] / "battery_mat_inventory_with_duplicates.csv")
    save_csv(selected, dirs["audit"] / "battery_selected_unique_mats.csv")
    print(f"Battery .mat files found={len(inv)}, unique batteries selected={len(selected)}")
    print(selected[["battery_id", "path", "size_mb"]].to_string(index=False))
    return selected

def extract_battery(args, dirs):
    print_block("EXTRACTING NASA BATTERY FROM RAW MAT FILES")
    if loadmat is None:
        raise ImportError("scipy is required for reading NASA Battery .mat files.")
    selected = choose_unique_battery_mats(args, dirs)
    rows = []
    audit_rows = []
    for _, rec in selected.iterrows():
        battery_id = rec["battery_id"]
        mat_path = Path(rec["path"])
        print(f"Reading Battery {battery_id}: {mat_path}", flush=True)
        try:
            mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            audit_rows.append({"dataset": "NASA Battery", "battery_id": battery_id, "file": str(mat_path), "status": "READ_FAILED", "error": repr(e)})
            continue
        battery_obj = mat.get(battery_id)
        if battery_obj is None:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            battery_obj = mat[keys[0]] if keys else None
        cycles = as_list(get_field(battery_obj, "cycle"))
        discharge_count = 0
        skipped = 0
        for idx, cyc in enumerate(cycles, start=1):
            ctype = str(get_field(cyc, "type", "")).lower()
            if ctype != "discharge":
                skipped += 1
                continue
            data = get_field(cyc, "data")
            capacity = scalar_float(get_field(data, "Capacity"))
            if not np.isfinite(capacity):
                skipped += 1
                continue
            time_arr = to_float_1d(get_field(data, "Time"))
            duration = float(time_arr[-1] - time_arr[0]) if len(time_arr) >= 2 else np.nan
            row = {
                "dataset": "NASA Battery",
                "asset_id": battery_id,
                "battery_id": battery_id,
                "source_mat_path": str(mat_path),
                "cycle": idx,
                "discharge_index": discharge_count + 1,
                "capacity": capacity,
                "ambient_temperature": scalar_float(get_field(cyc, "ambient_temperature")),
                "duration": duration,
            }
            signal_map = {
                "voltage_measured": get_field(data, "Voltage_measured"),
                "current_measured": get_field(data, "Current_measured"),
                "temperature_measured": get_field(data, "Temperature_measured"),
                "current_load": get_field(data, "Current_load"),
                "voltage_load": get_field(data, "Voltage_load"),
                "time": get_field(data, "Time"),
            }
            for name, values in signal_map.items():
                row.update(signal_summary(name, values))
            v = to_float_1d(get_field(data, "Voltage_measured"))
            i = to_float_1d(get_field(data, "Current_measured"))
            t = to_float_1d(get_field(data, "Time"))
            n = min(len(v), len(i), len(t))
            if n >= 2:
                row["energy_proxy"] = safe_trapezoid(np.abs(v[:n] * i[:n]), t[:n])
                row["charge_proxy"] = safe_trapezoid(np.abs(i[:n]), t[:n])
                row["mean_power_proxy"] = float(np.mean(np.abs(v[:n] * i[:n])))
            else:
                row["energy_proxy"] = np.nan
                row["charge_proxy"] = np.nan
                row["mean_power_proxy"] = np.nan
            rows.append(row)
            discharge_count += 1
        audit_rows.append({"dataset": "NASA Battery", "battery_id": battery_id, "file": str(mat_path), "status": "OK", "cycles_found": len(cycles), "discharge_rows": discharge_count, "skipped_cycles": skipped})
    df = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    if df.empty:
        raise RuntimeError("Battery extraction produced zero rows.")
    gmin = df["capacity"].min()
    gmax = df["capacity"].max()
    df["target_health_global_minmax"] = np.clip((df["capacity"] - gmin) / max(gmax - gmin, 1e-12), 0, 1)
    first_cap = df.groupby("battery_id")["capacity"].transform("first")
    df["target_health_per_battery_initial"] = np.clip(df["capacity"] / np.maximum(first_cap, 1e-12), 0, 1)
    save_csv(df, dirs["features"] / "battery_features_clean_raw_mat.csv")
    save_csv(audit, dirs["audit"] / "battery_raw_mat_extraction_audit.csv")
    print(f"Battery extracted rows={len(df):,}, unique batteries={df['battery_id'].nunique():,}, columns={df.shape[1]}")
    return df, audit

def find_pronostia_asset_dir(root: Path, bearing: str):
    candidates = []
    for p in root.rglob(bearing):
        if not p.is_dir():
            continue
        bad = path_has_any(p, FORBIDDEN_BY_DATASET["PRONOSTIA/FEMTO"])
        if bad:
            continue
        files = [f for f in p.iterdir() if f.is_file() and f.stat().st_size > 1000]
        if files:
            candidates.append((p, len(files)))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]

def extract_pronostia(args, dirs):
    print_block("EXTRACTING PRONOSTIA/FEMTO FROM RAW VIBRATION FILES")
    rows = []
    audit_rows = []
    for bearing in PRONOSTIA_WHITELIST:
        asset_dir = find_pronostia_asset_dir(args.pronostia_root, bearing)
        if asset_dir is None:
            audit_rows.append({"dataset": "PRONOSTIA/FEMTO", "asset_id": bearing, "status": "MISSING", "path": "", "files_found": 0, "files_used": 0})
            continue
        files = sorted([f for f in asset_dir.iterdir() if f.is_file() and f.stat().st_size > 1000], key=numeric_sort_key)
        files_found = len(files)
        if args.mode == "quick":
            files = files[: min(files_found, args.quick_files_per_asset)]
        used = 0
        unreadable = 0
        print(f"PRONOSTIA {bearing}: path={asset_dir}, files_found={files_found}, files_attempted={len(files)}", flush=True)
        for i, f in enumerate(files):
            arr = read_numeric_file(f, max_cols=4, min_rows=10)
            if arr is None:
                unreadable += 1
                continue
            row = {"dataset": "PRONOSTIA/FEMTO", "asset_id": bearing, "condition": bearing.split("_")[0], "source_file": str(f), "time_index": i}
            row.update(vibration_features(arr, max_channels=2))
            rows.append(row)
            used += 1
        audit_rows.append({"dataset": "PRONOSTIA/FEMTO", "asset_id": bearing, "status": "FOUND" if used > 0 else "NO_READABLE_FILES", "path": str(asset_dir), "files_found": files_found, "files_attempted": len(files), "files_used": used, "unreadable_files": unreadable})
    df = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    save_csv(audit, dirs["audit"] / "pronostia_femto_whitelist_extraction_audit.csv")
    if df.empty:
        raise RuntimeError("PRONOSTIA extraction produced zero rows.")
    if df["asset_id"].nunique() != len(PRONOSTIA_WHITELIST):
        raise RuntimeError("PRONOSTIA extraction did not produce all whitelisted assets.")
    df = add_bearing_targets(df)
    save_csv(df, dirs["features"] / "pronostia_femto_raw_hi_features.csv")
    print(f"PRONOSTIA extracted rows={len(df):,}, assets={df['asset_id'].nunique()}, columns={df.shape[1]}")
    return df, audit

def extract_xjtu(args, dirs):
    print_block("EXTRACTING XJTU-SY FROM VERIFIED 15 RAW BEARING FOLDERS")
    rows = []
    audit_rows = []
    for condition, bearing, expected_files in XJTU_EXPECTED:
        folder = args.xjtu_base / condition / bearing
        files = sorted([f for f in folder.glob("*.csv") if f.is_file()], key=numeric_sort_key)
        if not folder.exists() or not files:
            audit_rows.append({"dataset": "XJTU-SY", "condition": condition, "asset_id": bearing, "status": "MISSING_OR_EMPTY", "path": str(folder), "expected_files": expected_files, "files_found": 0, "matches_expected_count": False, "files_used": 0})
            continue
        files_found = len(files)
        matches_expected = files_found == expected_files
        files_to_use = files if args.mode == "full" else files[: min(files_found, args.quick_files_per_asset)]
        print(f"XJTU {condition}/{bearing}: found={files_found}, expected={expected_files}, using={len(files_to_use)}", flush=True)
        used = 0
        unreadable = 0
        for i, f in enumerate(files_to_use):
            arr = read_numeric_file(f, max_cols=2, min_rows=10)
            if arr is None:
                unreadable += 1
                continue
            row = {"dataset": "XJTU-SY", "asset_id": bearing, "condition": condition, "source_file": str(f), "time_index": i}
            row.update(vibration_features(arr, max_channels=2))
            rows.append(row)
            used += 1
        audit_rows.append({"dataset": "XJTU-SY", "condition": condition, "asset_id": bearing, "status": "FOUND", "path": str(folder), "expected_files": expected_files, "files_found": files_found, "matches_expected_count": matches_expected, "files_attempted": len(files_to_use), "files_used": used, "unreadable_files": unreadable})
    audit = pd.DataFrame(audit_rows)
    save_csv(audit, dirs["audit"] / "xjtu_15_asset_extraction_audit.csv")
    found_assets = int((audit["status"] == "FOUND").sum()) if not audit.empty else 0
    exact_assets = int(audit["matches_expected_count"].sum()) if "matches_expected_count" in audit.columns else 0
    if found_assets != 15:
        raise RuntimeError(f"XJTU extraction requires all 15 assets. Found={found_assets}/15")
    if exact_assets != 15:
        raise RuntimeError(f"XJTU exact file-count check failed. Exact={exact_assets}/15")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("XJTU extraction produced zero rows.")
    df = add_bearing_targets(df)
    save_csv(df, dirs["features"] / "xjtu_sy_raw_hi_features.csv")
    print(f"XJTU extracted rows={len(df):,}, assets={df['asset_id'].nunique()}, columns={df.shape[1]}")
    return df, audit

def is_rejected_ims_file(p: Path):
    low = str(p).lower()
    name = p.name.lower()
    if not p.is_file() or p.stat().st_size <= 1000:
        return True, "reject_too_small_or_not_file"
    if "_archive" in low or "archive" in low:
        return True, "reject_archive_duplicate"
    if p.suffix.lower() in [".rar", ".zip", ".7z"]:
        return True, "reject_compressed_archive"
    if any(t in name for t in ["log", "metadata", "readme", "extracted_ok"]):
        return True, "reject_metadata_log"
    if any(t in low for t in ["xjtu", "femto", "pronostia", "battery", "cmapss", "c-mapss", "c_mapss"]):
        return True, "reject_cross_dataset_contamination"
    if not ("1st_test" in low or "2nd_test" in low or "3rd_test" in low or "4th_test" in low):
        return True, "reject_not_ims_test_file"
    return False, "accept"

def extract_ims(args, dirs):
    print_block("EXTRACTING IMS FAILED-BEARING CHANNELS FROM RAW TIMESTAMP FILES")
    accepted = []
    audit_rows = []
    rejection_counts = {}
    for p in args.ims_root.rglob("*"):
        reject, reason = is_rejected_ims_file(p)
        if reject:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            if reason in ["reject_archive_duplicate", "reject_compressed_archive", "reject_metadata_log", "reject_cross_dataset_contamination"]:
                audit_rows.append({"dataset": "IMS", "asset_id": "", "file": str(p), "status": "REJECT_FILE", "reason": reason})
            continue
        accepted.append(p)
    rows = []
    print(f"IMS accepted canonical candidate files={len(accepted)}", flush=True)
    for spec in IMS_SPECS:
        marker = spec["marker"].lower()
        alt_marker = spec.get("alt_marker", "").lower()
        files = [f for f in accepted if marker in str(f).lower() or (alt_marker and alt_marker in str(f).lower())]
        files = sorted(files)
        files_available = len(files)
        files_to_use = files if args.mode == "full" else files[: min(files_available, args.quick_files_per_asset)]
        print(f"IMS {spec['asset_id']}: marker={spec['marker']}, available={files_available}, using={len(files_to_use)}", flush=True)
        used = 0
        unreadable = 0
        wrong_channels = 0
        for i, f in enumerate(files_to_use):
            arr = read_numeric_file(f, max_cols=8, min_rows=100)
            if arr is None:
                unreadable += 1
                continue
            cols = [c for c in spec["channels"] if c < arr.shape[1]]
            if not cols:
                wrong_channels += 1
                continue
            arr2 = arr[:, cols]
            row = {"dataset": "IMS", "asset_id": spec["asset_id"], "condition": spec["marker"], "source_file": str(f), "time_index": i}
            row.update(vibration_features(arr2, max_channels=2))
            rows.append(row)
            used += 1
        audit_rows.append({"dataset": "IMS", "asset_id": spec["asset_id"], "status": "FOUND" if used > 0 else "NO_ROWS", "marker": spec["marker"], "alt_marker": spec.get("alt_marker", ""), "channels": str(spec["channels"]), "files_available_before_cap": files_available, "files_attempted": len(files_to_use), "files_used": used, "unreadable_files": unreadable, "wrong_channel_files": wrong_channels})
    for reason, count in sorted(rejection_counts.items()):
        audit_rows.append({"dataset": "IMS", "asset_id": "", "status": "REJECTION_SUMMARY", "reason": reason, "count": count})
    audit = pd.DataFrame(audit_rows)
    save_csv(audit, dirs["audit"] / "ims_failed_bearing_extraction_audit.csv")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("IMS extraction produced zero rows. See ims_failed_bearing_extraction_audit.csv")
    if df["asset_id"].astype(str).str.contains("UNKNOWN", case=False, na=False).any():
        raise RuntimeError("IMS extraction produced UNKNOWN_ASSET, refusing to continue.")
    expected_assets = {s["asset_id"] for s in IMS_SPECS}
    actual_assets = set(df["asset_id"].unique())
    if actual_assets != expected_assets:
        raise RuntimeError(f"IMS asset mismatch. Expected={expected_assets}; actual={actual_assets}")
    df = add_bearing_targets(df)
    save_csv(df, dirs["features"] / "ims_failed_bearing_raw_hi_features.csv")
    print(f"IMS extracted rows={len(df):,}, assets={df['asset_id'].nunique()}, columns={df.shape[1]}")
    return df, audit

def make_dirs(output_root: Path):
    run_dir = output_root / f"RAW_EXTRACTION_AUDIT_RUN_{now_stamp()}"
    dirs = {"run": run_dir, "manifest": run_dir / "00_RUN_MANIFEST", "audit": run_dir / "01_DATA_AUDIT", "features": run_dir / "02_CLEAN_FEATURES"}
    for p in dirs.values():
        ensure_dir(p)
    return dirs

def save_environment(args, dirs):
    save_json({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "output_root": str(args.output_root),
        "old_outputs_used": False,
        "script_purpose": "Raw extraction and audit only. No modeling.",
        "paths": {
            "cmapss_root": str(args.cmapss_root),
            "battery_root": str(args.battery_root),
            "pronostia_root": str(args.pronostia_root),
            "xjtu_base": str(args.xjtu_base),
            "ims_root": str(args.ims_root),
        },
        "quick_files_per_asset": args.quick_files_per_asset,
    }, dirs["manifest"] / "run_config.json")
    save_json({"python": sys.version, "platform": platform.platform(), "executable": sys.executable, "cwd": os.getcwd()}, dirs["manifest"] / "environment.json")

def build_summary(datasets, dirs, elapsed):
    rows = []
    for ds, df in datasets.items():
        rows.append({"dataset": ds, "rows_extracted": len(df), "columns": df.shape[1], "assets": df["asset_id"].nunique() if "asset_id" in df.columns else np.nan})
    summary = pd.DataFrame(rows)
    output_map = {
        "NASA C-MAPSS": "02_CLEAN_FEATURES/cmapss_engine_features_clean_raw.csv",
        "NASA Battery": "02_CLEAN_FEATURES/battery_features_clean_raw_mat.csv",
        "PRONOSTIA/FEMTO": "02_CLEAN_FEATURES/pronostia_femto_raw_hi_features.csv",
        "XJTU-SY": "02_CLEAN_FEATURES/xjtu_sy_raw_hi_features.csv",
        "IMS": "02_CLEAN_FEATURES/ims_failed_bearing_raw_hi_features.csv",
    }
    summary["output_file"] = summary["dataset"].map(output_map)
    save_csv(summary, dirs["audit"] / "data_cleaning_summary.csv")
    pointers = {"run_dir": str(dirs["run"]), "elapsed_seconds": elapsed, "features": {k: str(dirs["run"] / v) for k, v in output_map.items()}, "audit_summary": str(dirs["audit"] / "data_cleaning_summary.csv")}
    save_json(pointers, dirs["manifest"] / "LATEST_RAW_FEATURE_POINTERS.json")
    print("\nDATA CLEANING SUMMARY")
    print(summary.to_string(index=False))
    print("\nFeature pointers saved:")
    print(dirs["manifest"] / "LATEST_RAW_FEATURE_POINTERS.json")
    return summary

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--output_root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--cmapss_root", type=Path, default=CMAPSS_ROOT_DEFAULT)
    p.add_argument("--battery_root", type=Path, default=BATTERY_ROOT_DEFAULT)
    p.add_argument("--pronostia_root", type=Path, default=PRONOSTIA_ROOT_DEFAULT)
    p.add_argument("--xjtu_base", type=Path, default=XJTU_BASE_DEFAULT)
    p.add_argument("--ims_root", type=Path, default=IMS_ROOT_DEFAULT)
    p.add_argument("--quick_files_per_asset", type=int, default=120)
    return p.parse_args()

def main():
    args = parse_args()
    dirs = make_dirs(args.output_root)
    start = time.perf_counter()
    try:
        save_environment(args, dirs)
        verify_roots(args, dirs)
        datasets = {}
        df, aud = extract_cmapss(args, dirs)
        datasets["NASA C-MAPSS"] = df
        df, aud = extract_battery(args, dirs)
        datasets["NASA Battery"] = df
        df, aud = extract_pronostia(args, dirs)
        datasets["PRONOSTIA/FEMTO"] = df
        df, aud = extract_xjtu(args, dirs)
        datasets["XJTU-SY"] = df
        df, aud = extract_ims(args, dirs)
        datasets["IMS"] = df
        elapsed = time.perf_counter() - start
        build_summary(datasets, dirs, elapsed)
        save_json({"status": "SUCCESS", "completed_at": datetime.now().isoformat(timespec="seconds"), "elapsed_seconds": elapsed, "run_dir": str(dirs["run"])}, dirs["manifest"] / "COMPLETION_STATUS.json")
        print_block("RAW EXTRACTION AND AUDIT COMPLETE")
        print("Run directory:", dirs["run"])
        print("Elapsed seconds:", round(elapsed, 1))
    except Exception as e:
        elapsed = time.perf_counter() - start
        err_text = traceback.format_exc()
        save_text(err_text, dirs["manifest"] / "ERROR_TRACEBACK.txt")
        save_json({"status": "FAILED", "failed_at": datetime.now().isoformat(timespec="seconds"), "elapsed_seconds": elapsed, "error": repr(e), "traceback_file": str(dirs["manifest"] / "ERROR_TRACEBACK.txt"), "run_dir": str(dirs["run"])}, dirs["manifest"] / "COMPLETION_STATUS.json")
        print(err_text)
        print("\nFAILED. Run directory:", dirs["run"])
        raise

if __name__ == "__main__":
    main()

