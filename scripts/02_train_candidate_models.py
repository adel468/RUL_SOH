
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

# Public release script: 02_train_candidate_models.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

#!/usr/bin/env python
# -*- coding: utf-8 -*-




import argparse
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE1_FILES = {
    "NASA C-MAPSS": "cmapss_engine_features_clean_raw.csv",
    "NASA Battery": "battery_features_clean_raw_mat.csv",
    "PRONOSTIA/FEMTO": "pronostia_femto_raw_hi_features.csv",
    "XJTU-SY": "xjtu_sy_raw_hi_features.csv",
    "IMS": "ims_failed_bearing_raw_hi_features.csv",
}
DATASETS = list(PHASE1_FILES.keys())

EXPECTED_COUNTS = {
    "NASA C-MAPSS": {"rows": 160359, "assets": 709},
    "NASA Battery": {"rows": 2769, "assets": 34},
    "PRONOSTIA/FEMTO": {"rows": 8383, "assets": 6},
    "XJTU-SY": {"rows": 9216, "assets": 15},
    "IMS": {"rows": 11620, "assets": 4},
}

LOCKED_CANDIDATES = {
    "NASA C-MAPSS": "safe_no_leak_allow_cycle__HGB__global_90",
    "NASA Battery": "capacity__global_minmax_health__safe_measurements_allow_cycle__ENSEMBLE_HGB_ET_RIDGE__twosided_guard_l97_u90_b12",
    "PRONOSTIA/FEMTO": "TRAJECTORY_SIMILARITY__repair_twosided_guard_l97_u90_b12",
    "XJTU-SY": "TRAJECTORY_SIMILARITY__global_90",
    "IMS": "TRAJECTORY_SIMILARITY__global_99",
}

LOCKED_METRICS = {
    "NASA C-MAPSS": {"coverage": 0.913892, "urgent_critical_coverage": 0.970501, "false_safe": 0.0, "underwarning": 0.015212, "mean_width": 0.345245},
    "NASA Battery": {"coverage": 0.887994, "urgent_critical_coverage": 0.936364, "false_safe": 0.0, "underwarning": 0.001612, "mean_width": 0.252979},
    "PRONOSTIA/FEMTO": {"coverage": 0.920174, "urgent_critical_coverage": 0.950957, "false_safe": 0.0, "underwarning": 0.0, "mean_width": 0.057900},
    "XJTU-SY": {"coverage": 0.882735, "urgent_critical_coverage": 0.843776, "false_safe": 0.0, "underwarning": 0.006293, "mean_width": 0.471824},
    "IMS": {"coverage": 0.989500, "urgent_critical_coverage": 0.986000, "false_safe": 0.0, "underwarning": 0.000500, "mean_width": 0.004076},
}

OLD_GATE = {
    "coverage_min": 0.88,
    "urgent_min": 0.80,
    "false_safe_max": 0.02,
    "underwarning_max": 0.20,
    "width_max": {
        "NASA C-MAPSS": 0.50,
        "NASA Battery": 0.50,
        "PRONOSTIA/FEMTO": 0.50,
        "XJTU-SY": 0.50,
        "IMS": 0.45,
    },
}


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def banner(text: str):
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path):
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(path: Path, obj: Any):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def safe_float(x, default=np.nan):
    

    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def dataset_slug(dataset: str) -> str:
    return norm(dataset).replace("nasa_", "")


def load_embedded_module(name: str, source: str) -> dict:
    ns = {"__name__": name, "__file__": f"<embedded:{name}>"}
    exec(compile(source, f"<embedded:{name}>", "exec"), ns)
    return ns


# Embedded validated protocol sources. They are included so this file is standalone.
PROTOCOL52_SOURCE = '# ============================================================\n# 52_CMAPSS_NO_LEAK_LOCK_VALIDATION.py\n# C-MAPSS-only no-leak lock validation + final five-dataset verdict\n# ============================================================\n#\n# Purpose:\n#   Fix the only remaining unresolved dataset after progress lock:\n#     NASA C-MAPSS.\n#\n# Why:\n#   The previous C-MAPSS result from run47 was strict but invalid for final\n#   manuscript evidence because audit script 50 found selected target-leakage\n#   predictors:\n#       RUL_capped\n#       RUL_original\n#\n# This script:\n#   - Loads the C-MAPSS feature table.\n#   - Uses target_RUL/RUL only as target, never as predictor.\n#   - Excludes every RUL/target/life/failure-like predictor.\n#   - Excludes RUL_capped, RUL_original, target_RUL, target_normalized_RUL,\n#     remaining useful life, labels, y_true/y_pred, etc.\n#   - Runs controlled group-wise validation by engine group.\n#   - Tests only leakage-safe feature policies.\n#   - Produces a C-MAPSS lock decision and a final five-dataset verdict by\n#     importing:\n#       Battery controlled lock from script 49\n#       Bearing integrated results from script 47\n#\n# Recommended first run:\n# runfile(\n#     r"scripts\\52_CMAPSS_NO_LEAK_LOCK_VALIDATION.py",\n#     args="--preset quick",\n#     wdir=r"scripts"\n# )\n#\n# Final lock run:\n# runfile(\n#     r"scripts\\52_CMAPSS_NO_LEAK_LOCK_VALIDATION.py",\n#     args="--preset lock",\n#     wdir=r"scripts"\n# )\n#\n# Expected runtime:\n#   quick: ~10-25 minutes\n#   lock : ~30-90 minutes depending on CPU/storage\n#\n# ============================================================\n\nfrom __future__ import annotations\n\nfrom pathlib import Path\nfrom datetime import datetime\nimport argparse\nimport json\nimport math\nimport re\nimport warnings\n\nwarnings.filterwarnings("ignore")\n\nimport numpy as np\nimport pandas as pd\n\nfrom sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor\nfrom sklearn.linear_model import Ridge\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import RobustScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n\ntry:\n    from scipy.stats import spearmanr\nexcept Exception:\n    spearmanr = None\n\n\n# ============================================================\n# Defaults\n# ============================================================\n\nDATA_ROOT_DEFAULT = Path(r"raw_datasets")\nOUTPUT_ROOT_DEFAULT = Path(r"outputs")\n\nRUN47_FOLDER = "ALL_DATASET_FULL_IMPROVEMENT_VERDICT"\nBATTERY_LOCK_FOLDER = "BATTERY_CAPACITY_ONLY_LOCK_VALIDATION"\n\nDATASET = "NASA C-MAPSS"\n\nGATE = {\n    "coverage_min": 0.88,\n    "urgent_min": 0.80,\n    "max_false_safe": 0.02,\n    "max_underwarning": 0.20,\n    "strict_width_limit": 0.50,\n    "relaxed_width_limit": 0.65,\n    "full_width_reject": 0.75,\n    "min_rows": 1000,\n    "min_groups": 20,\n    "min_folds": 8,\n}\n\n# Anything containing these terms is forbidden as a predictor.\nTARGET_LEAK_TERMS = [\n    "target", "rul", "remaining", "useful", "life", "label", "y_true",\n    "y_pred", "ground_truth", "truth", "failure", "ttf", "eol",\n    "end_of_life", "degradation_target", "health_target",\n]\n\n# Time/cycle terms are tested in a separate allow-cycle protocol.\nTIME_TERMS = [\n    "cycle", "time", "timestamp", "order", "index", "age",\n]\n\n# Metadata names are never predictors.\nMETADATA_TERMS = [\n    "source_file", "file", "path", "dataset", "split", "set_name", "subset_name",\n]\n\n# Feature names that are acceptable for C-MAPSS predictors after leakage exclusion.\nSAFE_CMAPSS_TERMS = [\n    "sensor", "s_", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",\n    "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19",\n    "s20", "s21", "setting", "op_setting", "operational",\n    "mean", "std", "min", "max", "median", "slope", "trend", "delta",\n    "lag", "diff", "roll", "rolling", "ewm", "ema", "window",\n]\n\nPREFERRED_TARGETS = [\n    "target_rul",\n    "rul",\n    "target_normalized_rul",\n    "remaining_useful_life",\n]\n\n\n# ============================================================\n# Utilities\n# ============================================================\n\ndef now_stamp() -> str:\n    return datetime.now().strftime("%Y-%m-%d_%H%M%S")\n\n\ndef ensure_dir(path: Path):\n    path.mkdir(parents=True, exist_ok=True)\n\n\ndef print_block(title: str):\n    print("\\n" + "=" * 120)\n    print(title)\n    print("=" * 120, flush=True)\n\n\ndef norm(x) -> str:\n    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")\n\n\ndef contains_any(name: str, terms: list[str]) -> bool:\n    n = norm(name)\n    return any(t in n for t in terms)\n\n\ndef save_csv(df: pd.DataFrame, path: Path):\n    ensure_dir(path.parent)\n    df.to_csv(path, index=False, encoding="utf-8-sig")\n\n\ndef read_csv(path: Path) -> pd.DataFrame:\n    if not path.exists():\n        return pd.DataFrame()\n    try:\n        return pd.read_csv(path)\n    except Exception:\n        return pd.DataFrame()\n\n\ndef write_json(path: Path, obj):\n    ensure_dir(path.parent)\n    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")\n\n\ndef save_excel(path: Path, sheets: dict[str, pd.DataFrame]):\n    ensure_dir(path.parent)\n    with pd.ExcelWriter(path, engine="openpyxl") as writer:\n        for name, df in sheets.items():\n            sheet = str(name)[:31]\n            if df is None or df.empty:\n                pd.DataFrame([{"message": "empty"}]).to_excel(writer, sheet_name=sheet, index=False)\n            else:\n                df.to_excel(writer, sheet_name=sheet, index=False)\n\n        wb = writer.book\n        for ws in wb.worksheets:\n            ws.freeze_panes = "A2"\n            ws.auto_filter.ref = ws.dimensions\n            for cell in ws[1]:\n                cell.font = cell.font.copy(bold=True, color="FFFFFF")\n                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")\n                cell.alignment = cell.alignment.copy(horizontal="center", vertical="center", wrap_text=True)\n            for col in ws.columns:\n                max_len = 0\n                for cell in col[:700]:\n                    value = "" if cell.value is None else str(cell.value)\n                    max_len = max(max_len, min(len(value), 160))\n                ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 105)\n            for row in ws.iter_rows(min_row=2):\n                for cell in row:\n                    cell.alignment = cell.alignment.copy(vertical="top", wrap_text=True)\n\n\ndef newest_run_dir(parent: Path) -> Path | None:\n    if not parent.exists():\n        return None\n    runs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("run_")]\n    if not runs:\n        return None\n    return max(runs, key=lambda p: p.stat().st_mtime)\n\n\ndef safe_float(x, default=np.nan):\n    try:\n        if pd.isna(x):\n            return default\n        return float(x)\n    except Exception:\n        return default\n\n\ndef clip01(x):\n    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)\n\n\ndef robust_spearman(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    m = np.isfinite(a) & np.isfinite(b)\n    if m.sum() < 3:\n        return 0.0\n    if spearmanr is not None:\n        val = spearmanr(a[m], b[m]).correlation\n        return 0.0 if not np.isfinite(val) else float(val)\n    ar = pd.Series(a[m]).rank().values\n    br = pd.Series(b[m]).rank().values\n    val = np.corrcoef(ar, br)[0, 1]\n    return 0.0 if not np.isfinite(val) else float(val)\n\n\ndef safe_r2(y, p):\n    try:\n        y = np.asarray(y, dtype=float)\n        p = np.asarray(p, dtype=float)\n        if len(y) < 2 or len(np.unique(y)) <= 1:\n            return np.nan\n        return float(r2_score(y, p))\n    except Exception:\n        return np.nan\n\n\n# ============================================================\n# Data discovery\n# ============================================================\n\ndef find_cmapss_table(data_root: Path, output_root: Path):\n    exacts = [\n        output_root / "processed_multi_asset" / "feature_tables" / "cmapss_engine_features.csv",\n        output_root / "processed_multi_asset" / "feature_tables" / "CMAPSS_engine_features.csv",\n    ]\n    for p in exacts:\n        if p.exists() and p.stat().st_size > 1000:\n            return p, "exact"\n\n    candidates = []\n    for root in [output_root / "processed_multi_asset" / "feature_tables", output_root, data_root]:\n        if not root.exists():\n            continue\n        for pat in ["*cmapss*feature*.csv", "*CMAPSS*feature*.csv", "*C-MAPSS*feature*.csv", "*c-mapss*feature*.csv"]:\n            try:\n                for p in root.rglob(pat):\n                    if p.is_file() and p.stat().st_size > 1000:\n                        candidates.append(p)\n            except Exception:\n                pass\n\n    if not candidates:\n        return None, "not_found"\n    candidates = sorted(set(candidates), key=lambda p: (\n        0 if "processed_multi_asset" in str(p).lower() else 1,\n        -p.stat().st_size,\n        len(str(p)),\n    ))\n    return candidates[0], "searched"\n\n\ndef identify_target(df: pd.DataFrame):\n    lower = {norm(c): c for c in df.columns}\n    for key in PREFERRED_TARGETS:\n        if key in lower and pd.api.types.is_numeric_dtype(df[lower[key]]):\n            return lower[key]\n    for c in df.columns:\n        if pd.api.types.is_numeric_dtype(df[c]) and contains_any(c, ["rul", "target", "remaining"]):\n            return c\n    return None\n\n\ndef normalize_target(yraw):\n    y = pd.to_numeric(yraw, errors="coerce").astype(float).values\n    finite = np.isfinite(y)\n    if finite.sum() == 0:\n        return y\n    ymin = np.nanmin(y[finite])\n    ymax = np.nanmax(y[finite])\n    if ymax > 1.5:\n        y = (y - ymin) / max(ymax - ymin, 1e-12)\n    else:\n        y = np.clip(y, 0, 1)\n    return np.clip(y, 0, 1)\n\n\ndef identify_group(df: pd.DataFrame):\n    lower = {norm(c): c for c in df.columns}\n    if "subset" in lower and "unit_number" in lower:\n        g = df[lower["subset"]].astype(str) + "_unit_" + df[lower["unit_number"]].astype(str)\n        return "subset+unit_number", g\n    for key in ["engine_id", "unit_number", "unit_id", "asset_id", "engine"]:\n        if key in lower:\n            return lower[key], df[lower[key]].astype(str)\n    block = max(20, len(df) // 100)\n    return "__pseudo_group__", pd.Series((np.arange(len(df)) // block).astype(str), index=df.index)\n\n\n# ============================================================\n# Feature audit/selection\n# ============================================================\n\ndef audit_all_columns(df: pd.DataFrame, target_col: str, y: np.ndarray, group_col: str):\n    rows = []\n    for c in df.columns:\n        n = norm(c)\n        is_numeric = pd.api.types.is_numeric_dtype(df[c])\n        is_target = c == target_col or n == norm(target_col)\n        is_group = c == group_col or n == norm(group_col)\n        target_like = contains_any(c, TARGET_LEAK_TERMS)\n        time_like = contains_any(c, TIME_TERMS)\n        metadata_like = contains_any(c, METADATA_TERMS)\n        corr = np.nan\n        minmax_mae = np.nan\n\n        if is_numeric and not is_target:\n            x = pd.to_numeric(df[c], errors="coerce").values.astype(float)\n            m = np.isfinite(x) & np.isfinite(y)\n            if m.sum() >= 20:\n                corr = robust_spearman(x[m], y[m])\n                xn = x.copy()\n                if np.nanmax(xn[m]) > np.nanmin(xn[m]):\n                    xn = (xn - np.nanmin(xn[m])) / max(np.nanmax(xn[m]) - np.nanmin(xn[m]), 1e-12)\n                    minmax_mae = float(np.nanmean(np.abs(xn[m] - y[m])))\n\n        fatal_leak = bool((target_like and not is_target) or (np.isfinite(minmax_mae) and minmax_mae <= 0.002))\n        rows.append({\n            "column": c,\n            "normalized": n,\n            "is_numeric": is_numeric,\n            "is_target_col": is_target,\n            "is_group_col": is_group,\n            "target_like_name": target_like,\n            "time_like_name": time_like,\n            "metadata_like_name": metadata_like,\n            "spearman_to_target": corr,\n            "abs_spearman_to_target": abs(corr) if np.isfinite(corr) else np.nan,\n            "minmax_mae_to_target": minmax_mae,\n            "fatal_leakage_if_predictor": fatal_leak,\n        })\n    out = pd.DataFrame(rows)\n    if not out.empty:\n        out = out.sort_values(["fatal_leakage_if_predictor", "abs_spearman_to_target"], ascending=[False, False])\n    return out\n\n\ndef select_features(df: pd.DataFrame, y: np.ndarray, target_col: str, group_col: str,\n                    allow_cycle: bool, safe_terms_only: bool, max_features: int):\n    audit = audit_all_columns(df, target_col, y, group_col)\n    fatal = set(audit.loc[audit["fatal_leakage_if_predictor"], "column"].astype(str).tolist())\n\n    rows = []\n    eligible = []\n    for c in df.columns:\n        n = norm(c)\n        status = ""\n        if c == target_col or n == norm(target_col):\n            status = "reject_target_column"\n        elif c == group_col or n == norm(group_col) or c == "__group__":\n            status = "reject_group_column"\n        elif not pd.api.types.is_numeric_dtype(df[c]):\n            status = "reject_non_numeric"\n        elif contains_any(c, TARGET_LEAK_TERMS):\n            status = "reject_target_leak_name"\n        elif c in fatal:\n            status = "reject_fatal_leakage_audit"\n        elif contains_any(c, METADATA_TERMS):\n            status = "reject_metadata_name"\n        elif (not allow_cycle) and contains_any(c, TIME_TERMS):\n            status = "reject_cycle_time_not_allowed"\n        elif safe_terms_only and not contains_any(c, SAFE_CMAPSS_TERMS + (TIME_TERMS if allow_cycle else [])):\n            status = "reject_not_safe_cmapss_term"\n        else:\n            s = pd.to_numeric(df[c], errors="coerce")\n            if s.notna().mean() < 0.70:\n                status = "reject_too_sparse"\n            elif s.nunique(dropna=True) <= 2:\n                status = "reject_constant_or_binary"\n            else:\n                eligible.append(c)\n                status = "eligible"\n        rows.append({"feature": c, "normalized": n, "selection_status": status})\n\n    # Rank by absolute Spearman with target. This is feature ranking only after all leakage exclusions.\n    scores = []\n    for c in eligible:\n        x = pd.to_numeric(df[c], errors="coerce").values\n        scores.append((c, abs(robust_spearman(x, y))))\n    scores = sorted(scores, key=lambda z: z[1], reverse=True)\n    selected = [c for c, _ in scores[:max_features]]\n\n    # Final safety re-audit of selected features.\n    selected_audit = audit[audit["column"].astype(str).isin(selected)].copy()\n    used_fatal = selected_audit[selected_audit["fatal_leakage_if_predictor"]]["column"].astype(str).tolist()\n    used_target_like = selected_audit[selected_audit["target_like_name"]]["column"].astype(str).tolist()\n\n    policy = {\n        "allow_cycle": bool(allow_cycle),\n        "safe_terms_only": bool(safe_terms_only),\n        "max_features": int(max_features),\n        "eligible_before_cap": int(len(eligible)),\n        "feature_count": int(len(selected)),\n        "feature_policy_final_eligible": bool(len(used_fatal) == 0 and len(used_target_like) == 0),\n        "used_fatal_leakage_features": ";".join(used_fatal),\n        "used_target_like_features": ";".join(used_target_like),\n    }\n\n    feature_scores = pd.DataFrame([\n        {"feature": c, "abs_spearman_to_target_after_exclusion": s, "selected": c in selected}\n        for c, s in scores\n    ])\n\n    return selected, policy, audit, selected_audit, pd.DataFrame(rows), feature_scores\n\n\n# ============================================================\n# Metrics / intervals\n# ============================================================\n\ndef categories(y):\n    y = np.asarray(y, dtype=float)\n    out = np.full(len(y), "safe", dtype=object)\n    out[y <= 0.50] = "monitor"\n    out[y <= 0.25] = "urgent"\n    out[y <= 0.10] = "critical"\n    return out\n\n\ndef rank_cat(cats):\n    order = {"critical": 0, "urgent": 1, "monitor": 2, "safe": 3}\n    return np.asarray([order.get(str(c), 3) for c in cats])\n\n\ndef metric_dict(y_true, y_pred, lower, upper):\n    y = np.asarray(y_true, dtype=float)\n    p = clip01(y_pred)\n    lo = clip01(lower)\n    hi = clip01(upper)\n\n    tc = categories(y)\n    pc = categories(p)\n    ic = categories(lo)\n\n    tr = rank_cat(tc)\n    pr = rank_cat(pc)\n    ir = rank_cat(ic)\n\n    uc = np.isin(tc, ["urgent", "critical"])\n    critical = tc == "critical"\n    covered = (y >= lo) & (y <= hi)\n    width = hi - lo\n\n    point_false = float(np.mean((pc == "safe") & uc)) if len(y) else np.nan\n    interval_false = float(np.mean((ic == "safe") & uc)) if len(y) else np.nan\n    point_under = float(np.mean(pr > tr)) if len(y) else np.nan\n    interval_under = float(np.mean(ir > tr)) if len(y) else np.nan\n    point_over = float(np.mean(pr < tr)) if len(y) else np.nan\n    interval_over = float(np.mean(ir < tr)) if len(y) else np.nan\n\n    return {\n        "n_rows": int(len(y)),\n        "n_urgent_critical": int(uc.sum()),\n        "n_critical": int(critical.sum()),\n        "RMSE": float(np.sqrt(mean_squared_error(y, p))) if len(y) else np.nan,\n        "MAE": float(mean_absolute_error(y, p)) if len(y) else np.nan,\n        "R2": safe_r2(y, p),\n        "empirical_coverage": float(np.mean(covered)) if len(y) else np.nan,\n        "urgent_critical_coverage": float(np.mean(covered[uc])) if uc.any() else np.nan,\n        "critical_coverage": float(np.mean(covered[critical])) if critical.any() else np.nan,\n        "point_false_safe_rate": point_false,\n        "interval_false_safe_rate": interval_false,\n        "false_safe_reduction": point_false - interval_false,\n        "point_underwarning_rate": point_under,\n        "interval_underwarning_rate": interval_under,\n        "underwarning_reduction": point_under - interval_under,\n        "point_overwarning_rate": point_over,\n        "interval_overwarning_rate": interval_over,\n        "overwarning_increase": interval_over - point_over,\n        "mean_interval_width": float(np.mean(width)) if len(y) else np.nan,\n        "median_interval_width": float(np.median(width)) if len(y) else np.nan,\n        "covered_array": covered,\n        "true_category_array": tc,\n        "point_category_array": pc,\n        "interval_category_array": ic,\n    }\n\n\ndef clean_metric(m):\n    drop = {"covered_array", "true_category_array", "point_category_array", "interval_category_array"}\n    return {k: v for k, v in m.items() if k not in drop}\n\n\ndef qsafe(v, q):\n    v = np.asarray(v, dtype=float)\n    v = v[np.isfinite(v)]\n    if len(v) == 0:\n        return 0.0\n    return float(max(0.0, np.quantile(v, q)))\n\n\ndef interval_variants(ycal, pcal, ptest):\n    ycal = np.asarray(ycal, dtype=float)\n    pcal = np.asarray(pcal, dtype=float)\n    ptest = np.asarray(ptest, dtype=float)\n\n    abs_res = np.abs(ycal - pcal)\n    low_res = pcal - ycal\n    high_res = ycal - pcal\n    pred_cat = categories(ptest)\n\n    variants = []\n    for q in [0.90, 0.95, 0.97, 0.99]:\n        qq = qsafe(abs_res, q)\n        variants.append((f"global_{int(q*100)}", clip01(ptest - qq), clip01(ptest + qq), {"q_global": qq}))\n\n    specs = [\n        ("asym_l95_u80", 0.95, 0.80, 1.0, 1.0),\n        ("guard_l95_u80_b12", 0.95, 0.80, 1.2, 1.0),\n        ("guard_l97_u80_b15", 0.97, 0.80, 1.5, 1.0),\n        ("twosided_guard_l97_u90_b12", 0.97, 0.90, 1.2, 1.2),\n    ]\n    for name, ql, qh, lb, ub in specs:\n        qlow = qsafe(low_res, ql)\n        qhigh = qsafe(high_res, qh)\n        guard = np.isin(pred_cat, ["monitor", "urgent", "critical"])\n        ql_each = np.where(guard, qlow * lb, qlow)\n        qh_each = np.where(guard, qhigh * ub, qhigh)\n        variants.append((name, clip01(ptest - ql_each), clip01(ptest + qh_each), {"q_low": qlow, "q_high": qhigh}))\n\n    for w in [0.10, 0.20, 0.30, 0.40, 0.50]:\n        variants.append((f"fixed_width_{w:.2f}", clip01(ptest - w/2), clip01(ptest + w/2), {"fixed_width": w}))\n\n    return variants\n\n\ndef gate_candidate(row):\n    r = dict(row)\n    r["coverage_ok"] = safe_float(r.get("empirical_coverage")) >= GATE["coverage_min"]\n    urgent = r.get("urgent_critical_coverage")\n    r["urgent_ok"] = pd.isna(urgent) or safe_float(urgent) >= GATE["urgent_min"]\n    r["false_safe_ok"] = safe_float(r.get("interval_false_safe_rate")) <= GATE["max_false_safe"]\n    r["underwarning_ok"] = safe_float(r.get("interval_underwarning_rate")) <= GATE["max_underwarning"]\n    r["reduction_ok"] = safe_float(r.get("false_safe_reduction"), 0) >= -1e-12 and safe_float(r.get("underwarning_reduction"), 0) >= -1e-12\n    r["strict_width_ok"] = safe_float(r.get("mean_interval_width")) <= GATE["strict_width_limit"]\n    r["relaxed_width_ok"] = safe_float(r.get("mean_interval_width")) <= GATE["relaxed_width_limit"]\n    r["not_full_width"] = safe_float(r.get("mean_interval_width")) <= GATE["full_width_reject"]\n    r["strict_promotion"] = all([r["coverage_ok"], r["urgent_ok"], r["false_safe_ok"], r["underwarning_ok"], r["reduction_ok"], r["strict_width_ok"], r["not_full_width"]])\n    r["relaxed_review"] = all([r["coverage_ok"], r["urgent_ok"], r["false_safe_ok"], r["underwarning_ok"], r["reduction_ok"], r["relaxed_width_ok"], r["not_full_width"]])\n    gates = ["coverage_ok", "urgent_ok", "false_safe_ok", "underwarning_ok", "reduction_ok", "strict_width_ok", "not_full_width"]\n    r["gate_pass_count"] = int(sum(bool(r[c]) for c in gates))\n    r["gate_rank_score"] = int(r["strict_promotion"]) * 1000 + int(r["relaxed_review"]) * 500 + r["gate_pass_count"] * 50 + safe_float(r.get("urgent_critical_coverage"), 0) * 2 - safe_float(r.get("mean_interval_width"), 999) * 2\n    fail = []\n    if not r["coverage_ok"]: fail.append(f"coverage {safe_float(r.get(\'empirical_coverage\')):.3f}<0.88")\n    if not r["urgent_ok"]: fail.append(f"urgent {safe_float(r.get(\'urgent_critical_coverage\')):.3f}<0.80")\n    if not r["false_safe_ok"]: fail.append("false-safe gate")\n    if not r["underwarning_ok"]: fail.append("underwarning gate")\n    if not r["reduction_ok"]: fail.append("risk reduction negative")\n    if not r["strict_width_ok"]: fail.append(f"width {safe_float(r.get(\'mean_interval_width\')):.3f}>0.50")\n    if not r["not_full_width"]: fail.append("full-width reject")\n    r["blocking_reason"] = "; ".join(fail) if fail else "passes strict"\n    return r\n\n\n# ============================================================\n# Folds / models / evaluation\n# ============================================================\n\ndef make_group_folds(df: pd.DataFrame, group_col: str, max_folds: int, seed: int):\n    groups = sorted(df[group_col].astype(str).dropna().unique().tolist())\n    if len(groups) < 3:\n        return []\n    rng = np.random.default_rng(seed)\n\n    if max_folds and len(groups) > max_folds:\n        # deterministic representative set: early, late, and random middle groups.\n        keep = set(groups[:2] + groups[-2:])\n        rem = [g for g in groups if g not in keep]\n        need = max_folds - len(keep)\n        if need > 0 and rem:\n            keep.update(rng.choice(rem, size=min(need, len(rem)), replace=False).tolist())\n        test_groups = sorted(keep)\n    else:\n        test_groups = groups\n\n    folds = []\n    for tg in test_groups:\n        test_mask = df[group_col].astype(str).eq(str(tg)).values\n        rem = sorted(df.loc[~test_mask, group_col].astype(str).unique().tolist())\n        if len(rem) < 2:\n            continue\n        cal_n = max(1, int(math.ceil(0.20 * len(rem))))\n        cal_groups = set(rng.choice(rem, size=cal_n, replace=False))\n        cal_mask = df[group_col].astype(str).isin(cal_groups).values\n        train_mask = (~test_mask) & (~cal_mask)\n        folds.append((f"LOGO_{tg}", train_mask, cal_mask, test_mask))\n    return folds\n\n\ndef model_specs(seed: int, include_et: bool):\n    models = {\n        "HGB": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(\n            max_iter=180,\n            learning_rate=0.045,\n            max_leaf_nodes=23,\n            min_samples_leaf=12,\n            l2_regularization=0.01,\n            random_state=seed,\n            early_stopping=True,\n            validation_fraction=0.15,\n            n_iter_no_change=20,\n        ))]),\n        "RIDGE": Pipeline([\n            ("imputer", SimpleImputer(strategy="median")),\n            ("scaler", RobustScaler()),\n            ("ridge", Ridge(alpha=1.0)),\n        ]),\n    }\n    if include_et:\n        models["ET"] = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(\n            n_estimators=160,\n            min_samples_leaf=4,\n            max_features="sqrt",\n            random_state=seed,\n            n_jobs=-1,\n        ))])\n    return models\n\n\ndef evaluate_protocol(df: pd.DataFrame, y: np.ndarray, group_col: str, feature_cols: list[str],\n                      protocol_name: str, feature_policy: dict, args, out_dir: Path):\n    folds = make_group_folds(df, group_col, args.max_folds, args.seed)\n    models = model_specs(args.seed, args.include_et)\n\n    fold_rows = []\n    point_rows = []\n    pred_rows = []\n\n    for i, (fold_name, tr_mask, cal_mask, te_mask) in enumerate(folds, start=1):\n        train = df.loc[tr_mask]\n        cal = df.loc[cal_mask]\n        test = df.loc[te_mask]\n\n        Xtr = train[feature_cols].values\n        Xcal = cal[feature_cols].values\n        Xte = test[feature_cols].values\n\n        ytr = y[tr_mask].astype(float)\n        ycal = y[cal_mask].astype(float)\n        yte = y[te_mask].astype(float)\n\n        pcal_list = []\n        pte_list = []\n        model_names = []\n\n        for model_name, model in models.items():\n            try:\n                model.fit(Xtr, ytr)\n                pcal = clip01(model.predict(Xcal))\n                pte = clip01(model.predict(Xte))\n                pcal_list.append(pcal)\n                pte_list.append(pte)\n                model_names.append(model_name)\n\n                pm = metric_dict(yte, pte, pte, pte)\n                point_rows.append({\n                    "dataset": DATASET,\n                    "protocol_name": protocol_name,\n                    "fold": fold_name,\n                    "point_model": model_name,\n                    **clean_metric(pm),\n                })\n\n                for ivar, lo, hi, qinfo in interval_variants(ycal, pcal, pte):\n                    m = metric_dict(yte, pte, lo, hi)\n                    cid = f"{protocol_name}__{model_name}__{ivar}"\n                    fold_rows.append({\n                        "dataset": DATASET,\n                        "candidate_id": cid,\n                        "protocol_name": protocol_name,\n                        "point_model": model_name,\n                        "interval_variant": ivar,\n                        "fold": fold_name,\n                        "feature_count": len(feature_cols),\n                        "feature_policy_final_eligible": feature_policy["feature_policy_final_eligible"],\n                        "n_train": len(train),\n                        "n_cal": len(cal),\n                        "n_test": len(test),\n                        "train_groups": train[group_col].nunique(),\n                        "cal_groups": cal[group_col].nunique(),\n                        "test_groups": test[group_col].nunique(),\n                        **clean_metric(m),\n                        **qinfo,\n                    })\n\n                    pred_rows.append(pd.DataFrame({\n                        "dataset": DATASET,\n                        "candidate_id": cid,\n                        "protocol_name": protocol_name,\n                        "point_model": model_name,\n                        "interval_variant": ivar,\n                        "fold": fold_name,\n                        "asset_id": test[group_col].astype(str).values,\n                        "row_index": test.index.values,\n                        "y_true": yte,\n                        "y_pred": pte,\n                        "lower": lo,\n                        "upper": hi,\n                        "true_category": m["true_category_array"],\n                        "point_category": m["point_category_array"],\n                        "interval_category": m["interval_category_array"],\n                        "covered": m["covered_array"],\n                        "interval_width": hi - lo,\n                    }))\n            except Exception as e:\n                point_rows.append({\n                    "dataset": DATASET,\n                    "protocol_name": protocol_name,\n                    "fold": fold_name,\n                    "point_model": model_name,\n                    "error": repr(e),\n                })\n\n        if len(pcal_list) >= 2:\n            pcal = clip01(np.mean(pcal_list, axis=0))\n            pte = clip01(np.mean(pte_list, axis=0))\n            model_name = "ENSEMBLE_" + "_".join(model_names)\n            for ivar, lo, hi, qinfo in interval_variants(ycal, pcal, pte):\n                m = metric_dict(yte, pte, lo, hi)\n                cid = f"{protocol_name}__{model_name}__{ivar}"\n                fold_rows.append({\n                    "dataset": DATASET,\n                    "candidate_id": cid,\n                    "protocol_name": protocol_name,\n                    "point_model": model_name,\n                    "interval_variant": ivar,\n                    "fold": fold_name,\n                    "feature_count": len(feature_cols),\n                    "feature_policy_final_eligible": feature_policy["feature_policy_final_eligible"],\n                    "n_train": len(train),\n                    "n_cal": len(cal),\n                    "n_test": len(test),\n                    "train_groups": train[group_col].nunique(),\n                    "cal_groups": cal[group_col].nunique(),\n                    "test_groups": test[group_col].nunique(),\n                    **clean_metric(m),\n                    **qinfo,\n                })\n\n                pred_rows.append(pd.DataFrame({\n                    "dataset": DATASET,\n                    "candidate_id": cid,\n                    "protocol_name": protocol_name,\n                    "point_model": model_name,\n                    "interval_variant": ivar,\n                    "fold": fold_name,\n                    "asset_id": test[group_col].astype(str).values,\n                    "row_index": test.index.values,\n                    "y_true": yte,\n                    "y_pred": pte,\n                    "lower": lo,\n                    "upper": hi,\n                    "true_category": m["true_category_array"],\n                    "point_category": m["point_category_array"],\n                    "interval_category": m["interval_category_array"],\n                    "covered": m["covered_array"],\n                    "interval_width": hi - lo,\n                }))\n\n        print(f"  C-MAPSS | {protocol_name} | fold {i}/{len(folds)} done", flush=True)\n\n    return (\n        pd.DataFrame(fold_rows),\n        pd.DataFrame(point_rows),\n        pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame(),\n    )\n\n\ndef summarize_candidates(preds: pd.DataFrame, fold_metrics: pd.DataFrame):\n    if preds.empty:\n        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{"dataset": DATASET, "policy_decision": "NO_CANDIDATE"}])\n\n    meta_cols = [\n        "candidate_id", "protocol_name", "feature_count",\n        "feature_policy_final_eligible", "point_model", "interval_variant"\n    ]\n    meta = fold_metrics[[c for c in meta_cols if c in fold_metrics.columns]].drop_duplicates("candidate_id")\n\n    rows = []\n    for cid, g in preds.groupby("candidate_id"):\n        m = metric_dict(g["y_true"], g["y_pred"], g["lower"], g["upper"])\n        rows.append({\n            "dataset": DATASET,\n            "candidate_id": cid,\n            "folds": g["fold"].nunique(),\n            "assets": g["asset_id"].nunique(),\n            **clean_metric(m),\n        })\n    summary = pd.DataFrame(rows).merge(meta, on="candidate_id", how="left")\n    summary["candidate_final_eligible"] = summary["feature_policy_final_eligible"].fillna(False).astype(bool)\n\n    gated_rows = []\n    for _, r in summary.iterrows():\n        gr = gate_candidate(r.to_dict())\n        if not gr.get("candidate_final_eligible", False):\n            gr["strict_promotion"] = False\n            gr["relaxed_review"] = False\n            gr["blocking_reason"] = str(gr.get("blocking_reason", "")) + "; feature policy not final-eligible"\n            gr["gate_rank_score"] -= 1000\n        gated_rows.append(gr)\n\n    gated = pd.DataFrame(gated_rows).sort_values("gate_rank_score", ascending=False).reset_index(drop=True)\n\n    strict = gated[gated["strict_promotion"]]\n    relaxed = gated[gated["relaxed_review"]]\n    if not strict.empty:\n        r = strict.sort_values("gate_rank_score", ascending=False).iloc[0]\n        policy = "STRICT_PROMOTION"\n    elif not relaxed.empty:\n        r = relaxed.sort_values("gate_rank_score", ascending=False).iloc[0]\n        policy = "RELAXED_PROMOTION_REVIEW"\n    else:\n        r = gated.sort_values("gate_rank_score", ascending=False).iloc[0]\n        policy = "NO_PROMOTION_NEAREST_MISS"\n\n    decision = pd.DataFrame([{\n        "dataset": DATASET,\n        "policy_decision": policy,\n        "candidate_id": r.get("candidate_id", ""),\n        "protocol_name": r.get("protocol_name", ""),\n        "feature_count": r.get("feature_count", ""),\n        "candidate_final_eligible": r.get("candidate_final_eligible", False),\n        "point_model": r.get("point_model", ""),\n        "interval_variant": r.get("interval_variant", ""),\n        "empirical_coverage": r.get("empirical_coverage", np.nan),\n        "urgent_critical_coverage": r.get("urgent_critical_coverage", np.nan),\n        "interval_false_safe_rate": r.get("interval_false_safe_rate", np.nan),\n        "interval_underwarning_rate": r.get("interval_underwarning_rate", np.nan),\n        "false_safe_reduction": r.get("false_safe_reduction", np.nan),\n        "underwarning_reduction": r.get("underwarning_reduction", np.nan),\n        "mean_interval_width": r.get("mean_interval_width", np.nan),\n        "gate_pass_count": r.get("gate_pass_count", np.nan),\n        "blocking_reason": r.get("blocking_reason", ""),\n        "gate_rank_score": r.get("gate_rank_score", np.nan),\n    }])\n\n    return summary, gated, decision\n\n\ndef conditionals(preds: pd.DataFrame, decision: pd.DataFrame):\n    if preds.empty or decision.empty:\n        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()\n    cid = decision["candidate_id"].iloc[0]\n    sub = preds[preds["candidate_id"].astype(str).eq(str(cid))].copy()\n    if sub.empty:\n        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()\n    sub["_zone"] = categories(sub["y_true"].astype(float).values)\n\n    def group_table(col):\n        rows = []\n        for val, g in sub.groupby(col, dropna=False):\n            m = metric_dict(g["y_true"], g["y_pred"], g["lower"], g["upper"])\n            rows.append({"dataset": DATASET, "candidate_id": cid, col: val, **clean_metric(m)})\n        return pd.DataFrame(rows)\n\n    return group_table("asset_id"), group_table("fold"), group_table("_zone")\n\n\n# ============================================================\n# Import existing confirmed evidence\n# ============================================================\n\ndef get_row(df: pd.DataFrame, dataset: str):\n    if df.empty or "dataset" not in df.columns:\n        return {}\n    sub = df[df["dataset"].astype(str).eq(dataset)]\n    if sub.empty:\n        return {}\n    return sub.iloc[0].to_dict()\n\n\ndef load_battery_status(output_root: Path):\n    run = newest_run_dir(output_root / BATTERY_LOCK_FOLDER)\n    if run is None:\n        return {\n            "dataset": "NASA Battery",\n            "final_status": "NOT_FINAL_BATTERY_LOCK_MISSING",\n            "usable_for_main_claim_now": False,\n            "failure_reasons": "Battery lock run missing",\n        }\n    decision = read_csv(run / "tables" / "01_DECISION.csv")\n    if decision.empty:\n        return {\n            "dataset": "NASA Battery",\n            "final_status": "NOT_FINAL_BATTERY_DECISION_MISSING",\n            "usable_for_main_claim_now": False,\n            "source_run_dir": str(run),\n            "failure_reasons": "Battery decision missing",\n        }\n    d = decision.iloc[0].to_dict()\n    ok = str(d.get("policy_decision", "")) == "STRICT_PROMOTION" and bool(d.get("candidate_final_eligible", True))\n    return {\n        "dataset": "NASA Battery",\n        "final_status": "FINAL_CONFIRMED_CONTROLLED_LOCK" if ok else "NOT_FINAL_BATTERY_LOCK_FAILED",\n        "usable_for_main_claim_now": bool(ok),\n        "policy_decision": d.get("policy_decision", ""),\n        "candidate_id": d.get("candidate_id", ""),\n        "coverage": d.get("empirical_coverage", np.nan),\n        "urgent_critical": d.get("urgent_critical_coverage", np.nan),\n        "width": d.get("mean_interval_width", np.nan),\n        "source_run_dir": str(run),\n        "failure_reasons": "none" if ok else "Battery lock not strict/final eligible",\n    }\n\n\ndef load_bearing_statuses(output_root: Path):\n    run = newest_run_dir(output_root / RUN47_FOLDER)\n    if run is None:\n        return []\n    decisions = read_csv(run / "tables" / "02_INTEGRATED_DECISIONS.csv")\n    status = read_csv(run / "tables" / "01_DATASET_STATUS.csv")\n    out = []\n    for ds in ["PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]:\n        s = get_row(status, ds)\n        d = get_row(decisions, ds)\n        usable = bool(s.get("usable_for_main_claim_now", False))\n        out.append({\n            "dataset": ds,\n            "final_status": s.get("final_status", "MISSING_RUN47_STATUS"),\n            "usable_for_main_claim_now": usable,\n            "policy_decision": d.get("policy_decision", ""),\n            "candidate_id": d.get("candidate_id", ""),\n            "coverage": d.get("empirical_coverage", s.get("coverage", np.nan)),\n            "urgent_critical": d.get("urgent_critical_coverage", s.get("urgent_critical", np.nan)),\n            "width": d.get("mean_interval_width", s.get("width", np.nan)),\n            "source_run_dir": str(run),\n            "failure_reasons": "none" if usable else str(s.get("blocking_reason", "")),\n        })\n    return out\n\n\ndef build_final_status(cmapss_decision: pd.DataFrame, metadata: pd.DataFrame, output_root: Path, out_dir: Path):\n    d = cmapss_decision.iloc[0].to_dict() if not cmapss_decision.empty else {}\n    m = metadata.iloc[0].to_dict() if not metadata.empty else {}\n    rows = int(safe_float(m.get("rows", 0), 0))\n    groups = int(safe_float(m.get("groups", 0), 0))\n    folds = int(safe_float(m.get("folds", 0), 0))\n\n    size_ok = rows >= GATE["min_rows"] and groups >= GATE["min_groups"] and folds >= GATE["min_folds"]\n    policy_ok = str(d.get("policy_decision", "")) == "STRICT_PROMOTION"\n    candidate_ok = bool(d.get("candidate_final_eligible", False))\n    final_ok = policy_ok and candidate_ok and size_ok\n\n    reasons = []\n    if not policy_ok:\n        reasons.append(f"policy={d.get(\'policy_decision\', \'\')}")\n    if not candidate_ok:\n        reasons.append("candidate not final eligible")\n    if not size_ok:\n        reasons.append(f"size/folds insufficient rows={rows}, groups={groups}, folds={folds}")\n\n    cmapss_status = {\n        "dataset": DATASET,\n        "final_status": "FINAL_CONFIRMED_NO_LEAK_LOCK" if final_ok else "NOT_FINAL_CMAPSS_NO_LEAK_LOCK_FAILED",\n        "usable_for_main_claim_now": bool(final_ok),\n        "policy_decision": d.get("policy_decision", ""),\n        "candidate_id": d.get("candidate_id", ""),\n        "coverage": d.get("empirical_coverage", np.nan),\n        "urgent_critical": d.get("urgent_critical_coverage", np.nan),\n        "width": d.get("mean_interval_width", np.nan),\n        "source_run_dir": str(out_dir),\n        "failure_reasons": "none" if final_ok else "; ".join(reasons),\n    }\n\n    rows_all = [cmapss_status, load_battery_status(output_root)] + load_bearing_statuses(output_root)\n    dataset_status = pd.DataFrame(rows_all)\n    confirmed = int(dataset_status["usable_for_main_claim_now"].sum())\n    global_verdict = pd.DataFrame([{\n        "global_verdict": "ALL_FIVE_DATASETS_FINAL_CONFIRMED" if confirmed == 5 else "NOT_ALL_DATASETS_FINAL_CONFIRMED",\n        "confirmed_count": confirmed,\n        "required_count": 5,\n        "confirmed_datasets": ", ".join(dataset_status.loc[dataset_status["usable_for_main_claim_now"], "dataset"].tolist()),\n        "not_final_datasets": ", ".join(dataset_status.loc[~dataset_status["usable_for_main_claim_now"], "dataset"].tolist()),\n        "interpretation": "C-MAPSS no-leak lock replaces the invalid leaked run47 C-MAPSS result.",\n    }])\n    return global_verdict, dataset_status\n\n\n# ============================================================\n# Main\n# ============================================================\n\ndef parse_args():\n    p = argparse.ArgumentParser()\n    p.add_argument("--data_root", default=str(DATA_ROOT_DEFAULT))\n    p.add_argument("--output_root", default=str(OUTPUT_ROOT_DEFAULT))\n    p.add_argument("--preset", choices=["quick", "lock"], default="lock")\n    p.add_argument("--max_features", type=int, default=None)\n    p.add_argument("--max_folds", type=int, default=None)\n    p.add_argument("--include_et", action="store_true", help="Add ExtraTrees. Slower. Default off.")\n    p.add_argument("--seed", type=int, default=42)\n    args = p.parse_args()\n\n    if args.max_features is None:\n        args.max_features = 80 if args.preset == "quick" else 120\n    if args.max_folds is None:\n        args.max_folds = 6 if args.preset == "quick" else 10\n    return args\n\n\ndef main():\n    args = parse_args()\n    data_root = Path(args.data_root)\n    output_root = Path(args.output_root)\n\n    out_dir = output_root / "CMAPSS_NO_LEAK_LOCK_VALIDATION" / f"run_{now_stamp()}"\n    for sub in ["tables", "predictions", "folds", "metadata", "logs"]:\n        ensure_dir(out_dir / sub)\n\n    print_block("52_CMAPSS_NO_LEAK_LOCK_VALIDATION.py")\n    print(f"Output dir       : {out_dir}")\n    print(f"Preset           : {args.preset}")\n    print(f"Max folds        : {args.max_folds}")\n    print(f"Max features     : {args.max_features}")\n    print(f"Include ET       : {args.include_et}")\n    print("Mode             : C-MAPSS-only no-leak validation")\n    print("Expected runtime : quick ~10-25 min; lock ~30-90 min depending on CPU/storage")\n\n    table, source = find_cmapss_table(data_root, output_root)\n    if table is None:\n        raise FileNotFoundError("C-MAPSS feature table not found.")\n\n    print_block("LOADING C-MAPSS TABLE")\n    df0 = pd.read_csv(table)\n    target_col = identify_target(df0)\n    if target_col is None:\n        raise ValueError("C-MAPSS target column not found.")\n\n    group_col_name, group_series = identify_group(df0)\n    df0 = df0.copy()\n    df0["__group__"] = group_series.astype(str).values\n    y = normalize_target(df0[target_col])\n\n    finite = np.isfinite(y)\n    df = df0.loc[finite].copy().reset_index(drop=True)\n    y = y[finite]\n    df["__group__"] = df["__group__"].astype(str).values\n\n    print(f"Feature table: {table}")\n    print(f"Rows={len(df)} Cols={len(df.columns)} Target={target_col} Groups={df[\'__group__\'].nunique()}")\n\n    protocols = [\n        ("safe_no_leak_allow_cycle", True, False),\n        ("safe_no_leak_no_cycle", False, False),\n        ("sensor_setting_terms_only_no_cycle", False, True),\n    ]\n    if args.preset == "quick":\n        protocols = protocols[:2]\n\n    all_fold = []\n    all_point = []\n    all_pred = []\n    all_policy = []\n    all_audit = []\n    all_selected_audit = []\n    all_selection = []\n    all_scores = []\n\n    for protocol_name, allow_cycle, safe_terms_only in protocols:\n        print_block(f"PROTOCOL: {protocol_name}")\n        features, policy, full_audit, selected_audit, selection_table, feature_scores = select_features(\n            df, y, target_col, "__group__", allow_cycle, safe_terms_only, args.max_features\n        )\n        policy.update({\n            "dataset": DATASET,\n            "protocol_name": protocol_name,\n            "target_col": target_col,\n            "feature_table": str(table),\n        })\n\n        print(f"Selected features={len(features)} | final_eligible={policy[\'feature_policy_final_eligible\']}")\n        if policy["used_fatal_leakage_features"] or policy["used_target_like_features"]:\n            print("Rejected protocol due leakage in selected features.")\n            print("Fatal:", policy["used_fatal_leakage_features"])\n            print("Target-like:", policy["used_target_like_features"])\n\n        policy_df = pd.DataFrame([policy])\n        all_policy.append(policy_df)\n        full_audit["protocol_name"] = protocol_name\n        selected_audit["protocol_name"] = protocol_name\n        selection_table["protocol_name"] = protocol_name\n        feature_scores["protocol_name"] = protocol_name\n        all_audit.append(full_audit)\n        all_selected_audit.append(selected_audit)\n        all_selection.append(selection_table)\n        all_scores.append(feature_scores)\n\n        if len(features) < 3 or not policy["feature_policy_final_eligible"]:\n            continue\n\n        fold_df, point_df, pred_df = evaluate_protocol(df, y, "__group__", features, protocol_name, policy, args, out_dir)\n\n        if not fold_df.empty:\n            all_fold.append(fold_df)\n        if not point_df.empty:\n            all_point.append(point_df)\n        if not pred_df.empty:\n            all_pred.append(pred_df)\n\n        # Checkpoint after each protocol\n        preds_ck = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()\n        folds_ck = pd.concat(all_fold, ignore_index=True) if all_fold else pd.DataFrame()\n        summary_ck, gated_ck, decision_ck = summarize_candidates(preds_ck, folds_ck)\n        save_csv(decision_ck, out_dir / "tables" / "CHECKPOINT_decision.csv")\n        save_csv(gated_ck.head(5000), out_dir / "tables" / "CHECKPOINT_gated_top.csv")\n\n    fold_metrics = pd.concat(all_fold, ignore_index=True) if all_fold else pd.DataFrame()\n    point_metrics = pd.concat(all_point, ignore_index=True) if all_point else pd.DataFrame()\n    preds = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()\n    policy_table = pd.concat(all_policy, ignore_index=True) if all_policy else pd.DataFrame()\n    full_audit_table = pd.concat(all_audit, ignore_index=True) if all_audit else pd.DataFrame()\n    selected_audit_table = pd.concat(all_selected_audit, ignore_index=True) if all_selected_audit else pd.DataFrame()\n    selection_table = pd.concat(all_selection, ignore_index=True) if all_selection else pd.DataFrame()\n    feature_scores_table = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()\n\n    summary, gated, decision = summarize_candidates(preds, fold_metrics)\n    by_asset, by_fold, by_zone = conditionals(preds, decision)\n\n    metadata = pd.DataFrame([{\n        "dataset": DATASET,\n        "feature_table": str(table),\n        "table_source": source,\n        "rows": len(df),\n        "cols": len(df.columns),\n        "groups": df["__group__"].nunique(),\n        "target_col": target_col,\n        "group_col_source": group_col_name,\n        "folds": args.max_folds,\n        "max_features": args.max_features,\n        "include_et": args.include_et,\n        "protocols_tested": len(protocols),\n        "strict_no_leak_rule": "all RUL/target/life/failure-like predictors excluded before training",\n    }])\n\n    global_verdict, dataset_status = build_final_status(decision, metadata, output_root, out_dir)\n\n    final_gate = pd.DataFrame([{\n        "dataset": DATASET,\n        "final_gate": dataset_status.loc[dataset_status["dataset"].eq(DATASET), "final_status"].iloc[0] if not dataset_status.empty else "UNKNOWN",\n        "policy_decision": decision["policy_decision"].iloc[0] if not decision.empty else "NO_DECISION",\n        "candidate_id": decision["candidate_id"].iloc[0] if not decision.empty else "",\n        "coverage": decision["empirical_coverage"].iloc[0] if not decision.empty else np.nan,\n        "urgent_critical": decision["urgent_critical_coverage"].iloc[0] if not decision.empty else np.nan,\n        "width": decision["mean_interval_width"].iloc[0] if not decision.empty else np.nan,\n        "interpretation": "Final only if STRICT_PROMOTION under no-leak feature policy.",\n    }])\n\n    # Save outputs\n    save_csv(global_verdict, out_dir / "tables" / "00_GLOBAL_FINAL_VERDICT.csv")\n    save_csv(dataset_status, out_dir / "tables" / "01_ALL_DATASET_STATUS.csv")\n    save_csv(final_gate, out_dir / "tables" / "02_CMAPSS_FINAL_GATE.csv")\n    save_csv(decision, out_dir / "tables" / "03_CMAPSS_DECISION.csv")\n    save_csv(summary, out_dir / "tables" / "04_CANDIDATE_SUMMARY.csv")\n    save_csv(gated, out_dir / "tables" / "05_GATED_CANDIDATES.csv")\n    save_csv(policy_table, out_dir / "tables" / "06_FEATURE_POLICIES.csv")\n    save_csv(selected_audit_table, out_dir / "tables" / "07_SELECTED_FEATURE_AUDIT.csv")\n    save_csv(full_audit_table, out_dir / "tables" / "08_FULL_COLUMN_LEAKAGE_AUDIT.csv")\n    save_csv(selection_table, out_dir / "tables" / "09_FEATURE_SELECTION_STATUS.csv")\n    save_csv(feature_scores_table, out_dir / "tables" / "10_FEATURE_RANKING_AFTER_EXCLUSION.csv")\n    save_csv(by_asset, out_dir / "tables" / "11_CONDITIONAL_BY_ASSET.csv")\n    save_csv(by_fold, out_dir / "tables" / "12_CONDITIONAL_BY_FOLD.csv")\n    save_csv(by_zone, out_dir / "tables" / "13_CONDITIONAL_BY_ZONE.csv")\n    save_csv(point_metrics, out_dir / "tables" / "14_POINT_MODEL_METRICS.csv")\n    save_csv(fold_metrics, out_dir / "folds" / "15_FOLD_INTERVAL_METRICS.csv")\n    save_csv(metadata, out_dir / "metadata" / "16_METADATA.csv")\n    save_csv(preds, out_dir / "predictions" / "17_ROW_LEVEL_PREDICTIONS.csv")\n\n    workbook = out_dir / "CMAPSS_NO_LEAK_LOCK_VALIDATION.xlsx"\n    save_excel(workbook, {\n        "GLOBAL_VERDICT": global_verdict,\n        "ALL_DATASET_STATUS": dataset_status,\n        "CMAPSS_FINAL_GATE": final_gate,\n        "CMAPSS_DECISION": decision,\n        "CANDIDATE_SUMMARY": summary.head(30000),\n        "GATED": gated.head(30000),\n        "FEATURE_POLICIES": policy_table,\n        "SELECTED_FEATURE_AUDIT": selected_audit_table.head(30000),\n        "BY_ASSET": by_asset.head(30000),\n        "BY_FOLD": by_fold,\n        "BY_ZONE": by_zone,\n        "METADATA": metadata,\n    })\n\n    write_json(out_dir / "manifest_52_cmapss_no_leak_lock.json", {\n        "generated_at": datetime.now().isoformat(timespec="seconds"),\n        "script": "52_CMAPSS_NO_LEAK_LOCK_VALIDATION.py",\n        "output_dir": str(out_dir),\n        "preset": args.preset,\n        "global_verdict": global_verdict.to_dict(orient="records"),\n        "cmapss_decision": decision.to_dict(orient="records"),\n    })\n\n    print_block("C-MAPSS FINAL GATE")\n    print(final_gate.to_string(index=False))\n    print_block("GLOBAL FINAL VERDICT")\n    print(global_verdict.to_string(index=False))\n    print_block("ALL DATASET STATUS")\n    print(dataset_status.to_string(index=False))\n    print_block("C-MAPSS DECISION")\n    print(decision.to_string(index=False) if not decision.empty else "No decision.")\n\n    print("\\nSaved:")\n    print(workbook)\n    print(out_dir / "tables" / "00_GLOBAL_FINAL_VERDICT.csv")\n    print(out_dir / "tables" / "01_ALL_DATASET_STATUS.csv")\n    print(out_dir / "tables" / "03_CMAPSS_DECISION.csv")\n    print("=" * 120)\n\n\nif __name__ == "__main__":\n    main()\n'
PROTOCOL49_SOURCE = '# ============================================================\n# 49_BATTERY_CAPACITY_ONLY_LOCK_VALIDATION.py\n# Controlled Battery-only validation with explicit target whitelist\n# ============================================================\n# This script deliberately rejects engineered capacity-derived columns as targets.\n# Allowed targets: exact capacity / capacity_ah / measured_capacity / discharge_capacity\n#                  exact soh / target_soh / state_of_health if present.\n# Rejected targets: capacity_roll_*, capacity_lag_*, capacity_delta_*, capacity_slope_*,\n#                   initial_discharge_capacity, any engineered capacity derivative.\n# ============================================================\n\nfrom __future__ import annotations\n\nfrom pathlib import Path\nfrom datetime import datetime\nimport argparse\nimport json\nimport math\nimport re\nimport warnings\n\nwarnings.filterwarnings("ignore")\n\nimport numpy as np\nimport pandas as pd\nfrom sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor\nfrom sklearn.linear_model import Ridge\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import RobustScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n\ntry:\n    from scipy.stats import spearmanr\nexcept Exception:\n    spearmanr = None\n\nDATA_ROOT_DEFAULT = Path(r"raw_datasets")\nOUTPUT_ROOT_DEFAULT = Path(r"outputs")\n\nGATE = dict(\n    coverage_min=0.88,\n    urgent_min=0.80,\n    max_false_safe=0.02,\n    max_underwarning=0.20,\n    strict_width_limit=0.50,\n    full_width_reject=0.75,\n)\n\nTARGET_EXACT_ALLOWED = {\n    "capacity", "capacity_ah", "measured_capacity", "discharge_capacity", "target_capacity",\n    "soh", "target_soh", "state_of_health",\n}\nTARGET_PREFERENCE = [\n    "soh", "target_soh", "state_of_health", "capacity", "measured_capacity",\n    "discharge_capacity", "capacity_ah", "target_capacity",\n]\nTARGET_REJECT_PATTERNS = [\n    "roll", "rolling", "lag", "delta", "diff", "slope", "gradient", "mean", "std",\n    "min", "max", "initial", "first", "baseline", "ema", "ewm", "window",\n]\nPREDICTOR_REJECT_TERMS = [\n    "target", "label", "y_true", "y_pred", "rul", "soh", "state_of_health",\n    "capacity", "eol", "end_of_life", "failure", "future",\n]\nSAFE_MEASUREMENT_TERMS = [\n    "voltage", "current", "temperature", "temp", "resistance", "impedance",\n    "charge", "discharge", "duration", "time", "energy", "power", "area",\n    "integral", "cc", "cv",\n]\nCYCLE_TERMS = ["cycle", "order", "index", "age"]\n\n\ndef now_stamp() -> str:\n    return datetime.now().strftime("%Y-%m-%d_%H%M%S")\n\n\ndef ensure_dir(path: Path):\n    path.mkdir(parents=True, exist_ok=True)\n\n\ndef print_block(title: str):\n    print("\\n" + "=" * 120)\n    print(title)\n    print("=" * 120, flush=True)\n\n\ndef norm(name: str) -> str:\n    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")\n\n\ndef has_any(name: str, terms: list[str]) -> bool:\n    n = norm(name)\n    return any(t in n for t in terms)\n\n\ndef save_csv(df: pd.DataFrame, path: Path):\n    ensure_dir(path.parent)\n    df.to_csv(path, index=False, encoding="utf-8-sig")\n\n\ndef save_excel(path: Path, sheets: dict[str, pd.DataFrame]):\n    ensure_dir(path.parent)\n    with pd.ExcelWriter(path, engine="openpyxl") as writer:\n        for name, df in sheets.items():\n            sheet = str(name)[:31]\n            if df is None or df.empty:\n                pd.DataFrame([{"message": "empty"}]).to_excel(writer, sheet_name=sheet, index=False)\n            else:\n                df.to_excel(writer, sheet_name=sheet, index=False)\n        wb = writer.book\n        for ws in wb.worksheets:\n            ws.freeze_panes = "A2"\n            ws.auto_filter.ref = ws.dimensions\n            for cell in ws[1]:\n                cell.font = cell.font.copy(bold=True, color="FFFFFF")\n                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")\n                cell.alignment = cell.alignment.copy(horizontal="center", vertical="center", wrap_text=True)\n            for col in ws.columns:\n                max_len = 0\n                for cell in col[:700]:\n                    val = "" if cell.value is None else str(cell.value)\n                    max_len = max(max_len, min(len(val), 130))\n                ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 95)\n            for row in ws.iter_rows(min_row=2):\n                for cell in row:\n                    cell.alignment = cell.alignment.copy(vertical="top", wrap_text=True)\n\n\ndef write_json(path: Path, obj):\n    ensure_dir(path.parent)\n    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")\n\n\ndef safe_float(x, default=np.nan):\n    try:\n        if pd.isna(x):\n            return default\n        return float(x)\n    except Exception:\n        return default\n\n\ndef clip01(x):\n    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)\n\n\ndef robust_spearman(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    m = np.isfinite(a) & np.isfinite(b)\n    if m.sum() < 3:\n        return 0.0\n    if spearmanr is not None:\n        val = spearmanr(a[m], b[m]).correlation\n        return 0.0 if not np.isfinite(val) else float(val)\n    ar = pd.Series(a[m]).rank().values\n    br = pd.Series(b[m]).rank().values\n    val = np.corrcoef(ar, br)[0, 1]\n    return 0.0 if not np.isfinite(val) else float(val)\n\n\ndef safe_r2(y, p):\n    try:\n        if len(y) < 2 or len(np.unique(y)) <= 1:\n            return np.nan\n        return float(r2_score(y, p))\n    except Exception:\n        return np.nan\n\n\ndef find_battery_table(data_root: Path, output_root: Path):\n    exacts = [\n        output_root / "processed_multi_asset" / "feature_tables" / "nasa_battery_features.csv",\n        output_root / "processed_multi_asset" / "feature_tables" / "battery_features.csv",\n        output_root / "processed_multi_asset" / "feature_tables" / "battery_cycle_features.csv",\n    ]\n    for p in exacts:\n        if p.exists() and p.stat().st_size > 1000:\n            return p, "exact"\n    candidates = []\n    for root in [output_root / "processed_multi_asset" / "feature_tables", output_root, data_root]:\n        if not root.exists():\n            continue\n        for pat in ["*battery*feature*.csv", "*Battery*feature*.csv", "*NASA*Battery*.csv"]:\n            for p in root.rglob(pat):\n                if p.is_file() and p.stat().st_size > 1000:\n                    candidates.append(p)\n    if not candidates:\n        return None, "not_found"\n    candidates = sorted(set(candidates), key=lambda p: (-p.stat().st_size, len(str(p))))\n    return candidates[0], "searched"\n\n\ndef identify_group(df: pd.DataFrame):\n    lower = {norm(c): c for c in df.columns}\n    for key in ["battery_id", "cell_id", "asset_id", "unit_id", "battery", "cell"]:\n        if key in lower:\n            return lower[key], df[lower[key]].astype(str)\n    n = len(df)\n    block = max(10, n // 20)\n    return "__pseudo_group__", pd.Series((np.arange(n) // block).astype(str), index=df.index)\n\n\ndef audit_targets(df: pd.DataFrame):\n    rows = []\n    accepted = []\n    for c in df.columns:\n        n = norm(c)\n        numeric = pd.api.types.is_numeric_dtype(df[c])\n        exact_ok = n in TARGET_EXACT_ALLOWED\n        engineered = any(p in n for p in TARGET_REJECT_PATTERNS)\n        accept = bool(numeric and exact_ok and not engineered)\n        if accept:\n            accepted.append(c)\n        rows.append(dict(\n            column=c,\n            normalized=n,\n            is_numeric=numeric,\n            exact_allowed_name=exact_ok,\n            rejected_engineered_pattern=engineered,\n            accepted_as_target=accept,\n        ))\n    accepted = sorted(\n        accepted,\n        key=lambda c: TARGET_PREFERENCE.index(norm(c)) if norm(c) in TARGET_PREFERENCE else 999,\n    )\n    return accepted, pd.DataFrame(rows)\n\n\ndef build_target_variants(df: pd.DataFrame, target_col: str, group: pd.Series):\n    n = norm(target_col)\n    yraw = pd.to_numeric(df[target_col], errors="coerce").astype(float)\n    variants = []\n    if n in ["soh", "target_soh", "state_of_health"]:\n        y = yraw.copy()\n        if y.max() > 1.5:\n            y = y / 100.0\n        variants.append((f"{target_col}__as_soh", np.clip(y.values.astype(float), 0, 1), "clean SOH target"))\n        return variants\n\n    finite = yraw[np.isfinite(yraw)]\n    if finite.size and finite.max() > finite.min():\n        y_global = (yraw - finite.min()) / max(finite.max() - finite.min(), 1e-12)\n        variants.append((f"{target_col}__global_minmax_health", np.clip(y_global.values.astype(float), 0, 1), "exact capacity target with global target scaling"))\n\n    tmp = pd.DataFrame({"g": group.astype(str).values, "cap": yraw.values})\n    first = tmp.groupby("g")["cap"].transform(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)\n    soh = yraw.values / np.maximum(first.values.astype(float), 1e-12)\n    variants.append((f"{target_col}__per_battery_initial_soh", np.clip(soh, 0, 1), "exact capacity divided by first observed capacity per battery"))\n    return variants\n\n\ndef leakage_audit(df: pd.DataFrame, y: np.ndarray, target_col: str, group_col: str):\n    rows = []\n    for c in df.columns:\n        if c == target_col or c == group_col or c.startswith("__"):\n            continue\n        if not pd.api.types.is_numeric_dtype(df[c]):\n            continue\n        x = pd.to_numeric(df[c], errors="coerce").values.astype(float)\n        m = np.isfinite(x) & np.isfinite(y)\n        if m.sum() < 20:\n            continue\n        corr = robust_spearman(x[m], y[m])\n        target_like = has_any(c, PREDICTOR_REJECT_TERMS)\n        high_corr = abs(corr) >= 0.995\n        rows.append(dict(\n            feature=c,\n            normalized=norm(c),\n            target_like_name=target_like,\n            spearman_to_target=corr,\n            abs_spearman_to_target=abs(corr),\n            high_corr_suspect=high_corr,\n            direct_leakage_suspect=bool(target_like or high_corr),\n        ))\n    out = pd.DataFrame(rows)\n    if not out.empty:\n        out = out.sort_values(["direct_leakage_suspect", "abs_spearman_to_target"], ascending=[False, False])\n    return out\n\n\ndef select_features(df: pd.DataFrame, y: np.ndarray, target_col: str, group_col: str, allow_cycle: bool, max_features: int):\n    audit = leakage_audit(df, y, target_col, group_col)\n    leak = set(audit.loc[audit["direct_leakage_suspect"], "feature"].astype(str).tolist()) if not audit.empty else set()\n    selected = []\n    rows = []\n    for c in df.columns:\n        n = norm(c)\n        if c == target_col or c == group_col or c.startswith("__"):\n            reason = "target/group/internal"\n        elif not pd.api.types.is_numeric_dtype(df[c]):\n            reason = "non_numeric"\n        elif has_any(c, PREDICTOR_REJECT_TERMS):\n            reason = "target_like_predictor_rejected"\n        elif c in leak:\n            reason = "high_corr_leakage_suspect"\n        elif not any(t in n for t in SAFE_MEASUREMENT_TERMS + (CYCLE_TERMS if allow_cycle else [])):\n            reason = "not_safe_measurement_term"\n        elif (not allow_cycle) and any(t in n for t in CYCLE_TERMS):\n            reason = "cycle_or_age_excluded"\n        else:\n            s = pd.to_numeric(df[c], errors="coerce")\n            if s.notna().mean() < 0.70 or s.nunique(dropna=True) <= 2:\n                reason = "sparse_or_constant"\n            else:\n                selected.append(c)\n                reason = "selected"\n        rows.append(dict(feature=c, normalized=n, status=reason))\n    scored = []\n    for c in selected:\n        x = pd.to_numeric(df[c], errors="coerce").values\n        scored.append((c, abs(robust_spearman(x, y))))\n    scored = sorted(scored, key=lambda z: z[1], reverse=True)\n    final = [c for c, _ in scored[:max_features]]\n    used_leaks = sorted(set(final).intersection(leak))\n    policy = dict(\n        allow_cycle=allow_cycle,\n        feature_count=len(final),\n        candidate_feature_count_before_cap=len(selected),\n        max_features=max_features,\n        feature_policy_final_eligible=len(used_leaks) == 0,\n        used_leakage_suspects=";".join(used_leaks),\n    )\n    return final, policy, audit, pd.DataFrame(rows)\n\n\ndef make_folds(df: pd.DataFrame, group_col: str, max_folds: int, seed: int):\n    groups = sorted(df[group_col].astype(str).dropna().unique().tolist())\n    if len(groups) < 3:\n        return []\n    rng = np.random.default_rng(seed)\n    if max_folds and len(groups) > max_folds:\n        keep = set(groups[:1] + groups[-1:])\n        rem = [g for g in groups if g not in keep]\n        need = max_folds - len(keep)\n        if need > 0 and rem:\n            keep.update(rng.choice(rem, size=min(need, len(rem)), replace=False).tolist())\n        test_groups = sorted(keep)\n    else:\n        test_groups = groups\n    folds = []\n    for tg in test_groups:\n        test_mask = df[group_col].astype(str).eq(str(tg)).values\n        rem = sorted(df.loc[~test_mask, group_col].astype(str).unique().tolist())\n        if len(rem) < 2:\n            continue\n        cal_n = max(1, int(math.ceil(0.25 * len(rem))))\n        cal_groups = set(rng.choice(rem, size=cal_n, replace=False))\n        cal_mask = df[group_col].astype(str).isin(cal_groups).values\n        train_mask = (~test_mask) & (~cal_mask)\n        folds.append((f"LOGO_{tg}", train_mask, cal_mask, test_mask))\n    return folds\n\n\ndef model_specs(seed: int, preset: str):\n    if preset == "quick":\n        return {\n            "HGB": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(\n                max_iter=120, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8,\n                l2_regularization=0.01, random_state=seed, early_stopping=True,\n            ))]),\n            "RIDGE": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("ridge", Ridge(alpha=1.0))]),\n        }\n    return {\n        "HGB": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(\n            max_iter=220, learning_rate=0.04, max_leaf_nodes=23, min_samples_leaf=8,\n            l2_regularization=0.01, random_state=seed, early_stopping=True,\n            validation_fraction=0.15, n_iter_no_change=20,\n        ))]),\n        "ET": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(\n            n_estimators=220, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=-1,\n        ))]),\n        "RIDGE": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("ridge", Ridge(alpha=1.0))]),\n    }\n\n\ndef categories(y):\n    y = np.asarray(y, dtype=float)\n    out = np.full(len(y), "safe", dtype=object)\n    out[y <= 0.50] = "monitor"\n    out[y <= 0.25] = "urgent"\n    out[y <= 0.10] = "critical"\n    return out\n\n\ndef rank_cat(cats):\n    order = {"critical": 0, "urgent": 1, "monitor": 2, "safe": 3}\n    return np.asarray([order.get(str(c), 3) for c in cats])\n\n\ndef metric_dict(y_true, y_pred, lower, upper):\n    y = np.asarray(y_true, dtype=float)\n    p = clip01(y_pred)\n    lo = clip01(lower)\n    hi = clip01(upper)\n    tc = categories(y)\n    pc = categories(p)\n    ic = categories(lo)\n    tr = rank_cat(tc)\n    pr = rank_cat(pc)\n    ir = rank_cat(ic)\n    uc = np.isin(tc, ["urgent", "critical"])\n    critical = tc == "critical"\n    covered = (y >= lo) & (y <= hi)\n    width = hi - lo\n    point_false = float(np.mean((pc == "safe") & uc)) if len(y) else np.nan\n    interval_false = float(np.mean((ic == "safe") & uc)) if len(y) else np.nan\n    point_under = float(np.mean(pr > tr)) if len(y) else np.nan\n    interval_under = float(np.mean(ir > tr)) if len(y) else np.nan\n    point_over = float(np.mean(pr < tr)) if len(y) else np.nan\n    interval_over = float(np.mean(ir < tr)) if len(y) else np.nan\n    return dict(\n        n_rows=int(len(y)),\n        n_urgent_critical=int(uc.sum()),\n        n_critical=int(critical.sum()),\n        RMSE=float(np.sqrt(mean_squared_error(y, p))) if len(y) else np.nan,\n        MAE=float(mean_absolute_error(y, p)) if len(y) else np.nan,\n        R2=safe_r2(y, p),\n        empirical_coverage=float(np.mean(covered)) if len(y) else np.nan,\n        urgent_critical_coverage=float(np.mean(covered[uc])) if uc.any() else np.nan,\n        critical_coverage=float(np.mean(covered[critical])) if critical.any() else np.nan,\n        point_false_safe_rate=point_false,\n        interval_false_safe_rate=interval_false,\n        false_safe_reduction=point_false - interval_false,\n        point_underwarning_rate=point_under,\n        interval_underwarning_rate=interval_under,\n        underwarning_reduction=point_under - interval_under,\n        point_overwarning_rate=point_over,\n        interval_overwarning_rate=interval_over,\n        overwarning_increase=interval_over - point_over,\n        mean_interval_width=float(np.mean(width)) if len(y) else np.nan,\n        median_interval_width=float(np.median(width)) if len(y) else np.nan,\n        covered_array=covered,\n        true_category_array=tc,\n        point_category_array=pc,\n        interval_category_array=ic,\n    )\n\n\ndef clean_metric(m):\n    return {k: v for k, v in m.items() if not k.endswith("_array")}\n\n\ndef qsafe(v, q):\n    v = np.asarray(v, dtype=float)\n    v = v[np.isfinite(v)]\n    if len(v) == 0:\n        return 0.0\n    return float(max(0.0, np.quantile(v, q)))\n\n\ndef interval_variants(ycal, pcal, ptest):\n    ycal = np.asarray(ycal, dtype=float)\n    pcal = np.asarray(pcal, dtype=float)\n    ptest = np.asarray(ptest, dtype=float)\n    abs_res = np.abs(ycal - pcal)\n    low_res = pcal - ycal\n    high_res = ycal - pcal\n    pred_cat = categories(ptest)\n    variants = []\n    for q in [0.90, 0.95, 0.97, 0.99]:\n        qq = qsafe(abs_res, q)\n        variants.append((f"global_{int(q*100)}", clip01(ptest - qq), clip01(ptest + qq), {"q_global": qq}))\n    for name, ql, qh, lb, ub in [\n        ("asym_l95_u80", 0.95, 0.80, 1.0, 1.0),\n        ("guard_l95_u80_b12", 0.95, 0.80, 1.2, 1.0),\n        ("guard_l97_u80_b15", 0.97, 0.80, 1.5, 1.0),\n        ("twosided_guard_l97_u90_b12", 0.97, 0.90, 1.2, 1.2),\n    ]:\n        qlow = qsafe(low_res, ql)\n        qhigh = qsafe(high_res, qh)\n        guard = np.isin(pred_cat, ["monitor", "urgent", "critical"])\n        ql_each = np.where(guard, qlow * lb, qlow)\n        qh_each = np.where(guard, qhigh * ub, qhigh)\n        variants.append((name, clip01(ptest - ql_each), clip01(ptest + qh_each), {"q_low": qlow, "q_high": qhigh}))\n    for w in [0.05, 0.10, 0.20, 0.30, 0.50]:\n        variants.append((f"fixed_width_{w:.2f}", clip01(ptest - w/2), clip01(ptest + w/2), {"fixed_width": w}))\n    return variants\n\n\ndef gate_candidate(row):\n    r = dict(row)\n    r["coverage_ok"] = safe_float(r.get("empirical_coverage")) >= GATE["coverage_min"]\n    urgent = r.get("urgent_critical_coverage")\n    r["urgent_ok"] = pd.isna(urgent) or safe_float(urgent) >= GATE["urgent_min"]\n    r["false_safe_ok"] = safe_float(r.get("interval_false_safe_rate")) <= GATE["max_false_safe"]\n    r["underwarning_ok"] = safe_float(r.get("interval_underwarning_rate")) <= GATE["max_underwarning"]\n    r["reduction_ok"] = safe_float(r.get("false_safe_reduction"), 0) >= -1e-12 and safe_float(r.get("underwarning_reduction"), 0) >= -1e-12\n    r["strict_width_ok"] = safe_float(r.get("mean_interval_width")) <= GATE["strict_width_limit"]\n    r["not_full_width"] = safe_float(r.get("mean_interval_width")) <= GATE["full_width_reject"]\n    r["strict_promotion"] = all([r["coverage_ok"], r["urgent_ok"], r["false_safe_ok"], r["underwarning_ok"], r["reduction_ok"], r["strict_width_ok"], r["not_full_width"]])\n    gates = ["coverage_ok", "urgent_ok", "false_safe_ok", "underwarning_ok", "reduction_ok", "strict_width_ok", "not_full_width"]\n    r["gate_pass_count"] = int(sum(bool(r[c]) for c in gates))\n    r["gate_rank_score"] = int(r["strict_promotion"]) * 1000 + r["gate_pass_count"] * 50 + safe_float(r.get("urgent_critical_coverage"), 0) * 2 - safe_float(r.get("mean_interval_width"), 999) * 2\n    fail = []\n    if not r["coverage_ok"]: fail.append(f"coverage {safe_float(r.get(\'empirical_coverage\')):.3f}<0.88")\n    if not r["urgent_ok"]: fail.append(f"urgent {safe_float(r.get(\'urgent_critical_coverage\')):.3f}<0.80")\n    if not r["false_safe_ok"]: fail.append("false-safe gate")\n    if not r["underwarning_ok"]: fail.append("underwarning gate")\n    if not r["reduction_ok"]: fail.append("risk reduction negative")\n    if not r["strict_width_ok"]: fail.append(f"width {safe_float(r.get(\'mean_interval_width\')):.3f}>0.50")\n    if not r["not_full_width"]: fail.append("full-width reject")\n    r["blocking_reason"] = "; ".join(fail) if fail else "passes strict"\n    return r\n\n\ndef evaluate(df, y, target_variant, target_col, group_col, features, policy, args):\n    folds = make_folds(df, group_col, args.max_folds, args.seed)\n    models = model_specs(args.seed, args.preset)\n    fold_rows, point_rows, pred_rows = [], [], []\n    for i, (fold_name, tr_mask, cal_mask, te_mask) in enumerate(folds, start=1):\n        train, cal, test = df.loc[tr_mask], df.loc[cal_mask], df.loc[te_mask]\n        Xtr, Xcal, Xte = train[features].values, cal[features].values, test[features].values\n        ytr, ycal, yte = y[tr_mask].astype(float), y[cal_mask].astype(float), y[te_mask].astype(float)\n        pcal_list, pte_list, model_names = [], [], []\n        for model_name, model in models.items():\n            try:\n                model.fit(Xtr, ytr)\n                pcal, pte = clip01(model.predict(Xcal)), clip01(model.predict(Xte))\n                pcal_list.append(pcal); pte_list.append(pte); model_names.append(model_name)\n                pm = metric_dict(yte, pte, pte, pte)\n                point_rows.append({"target_variant": target_variant, "protocol_name": policy["protocol_name"], "fold": fold_name, "point_model": model_name, **clean_metric(pm)})\n                for ivar, lo, hi, qinfo in interval_variants(ycal, pcal, pte):\n                    m = metric_dict(yte, pte, lo, hi)\n                    cid = f"{target_variant}__{policy[\'protocol_name\']}__{model_name}__{ivar}"\n                    fold_rows.append({\n                        "dataset": "NASA Battery", "candidate_id": cid, "target_variant": target_variant,\n                        "target_col": target_col, "protocol_name": policy["protocol_name"],\n                        "allow_cycle": policy["allow_cycle"], "feature_count": len(features),\n                        "feature_policy_final_eligible": policy["feature_policy_final_eligible"],\n                        "point_model": model_name, "interval_variant": ivar, "fold": fold_name,\n                        "n_train": len(train), "n_cal": len(cal), "n_test": len(test),\n                        "train_groups": train[group_col].nunique(), "cal_groups": cal[group_col].nunique(), "test_groups": test[group_col].nunique(),\n                        **clean_metric(m), **qinfo,\n                    })\n                    pred_rows.append(pd.DataFrame({\n                        "dataset": "NASA Battery", "candidate_id": cid, "target_variant": target_variant,\n                        "protocol_name": policy["protocol_name"], "point_model": model_name,\n                        "interval_variant": ivar, "fold": fold_name,\n                        "asset_id": test[group_col].astype(str).values,\n                        "row_index": test.index.values,\n                        "y_true": yte, "y_pred": pte, "lower": lo, "upper": hi,\n                        "true_category": m["true_category_array"], "point_category": m["point_category_array"],\n                        "interval_category": m["interval_category_array"], "covered": m["covered_array"],\n                        "interval_width": hi - lo,\n                    }))\n            except Exception as e:\n                point_rows.append({"target_variant": target_variant, "protocol_name": policy["protocol_name"], "fold": fold_name, "point_model": model_name, "error": repr(e)})\n        if len(pcal_list) >= 2:\n            pcal, pte = clip01(np.mean(pcal_list, axis=0)), clip01(np.mean(pte_list, axis=0))\n            model_name = "ENSEMBLE_" + "_".join(model_names)\n            for ivar, lo, hi, qinfo in interval_variants(ycal, pcal, pte):\n                m = metric_dict(yte, pte, lo, hi)\n                cid = f"{target_variant}__{policy[\'protocol_name\']}__{model_name}__{ivar}"\n                fold_rows.append({\n                    "dataset": "NASA Battery", "candidate_id": cid, "target_variant": target_variant,\n                    "target_col": target_col, "protocol_name": policy["protocol_name"],\n                    "allow_cycle": policy["allow_cycle"], "feature_count": len(features),\n                    "feature_policy_final_eligible": policy["feature_policy_final_eligible"],\n                    "point_model": model_name, "interval_variant": ivar, "fold": fold_name,\n                    "n_train": len(train), "n_cal": len(cal), "n_test": len(test),\n                    "train_groups": train[group_col].nunique(), "cal_groups": cal[group_col].nunique(), "test_groups": test[group_col].nunique(),\n                    **clean_metric(m), **qinfo,\n                })\n                pred_rows.append(pd.DataFrame({\n                    "dataset": "NASA Battery", "candidate_id": cid, "target_variant": target_variant,\n                    "protocol_name": policy["protocol_name"], "point_model": model_name,\n                    "interval_variant": ivar, "fold": fold_name,\n                    "asset_id": test[group_col].astype(str).values,\n                    "row_index": test.index.values,\n                    "y_true": yte, "y_pred": pte, "lower": lo, "upper": hi,\n                    "true_category": m["true_category_array"], "point_category": m["point_category_array"],\n                    "interval_category": m["interval_category_array"], "covered": m["covered_array"],\n                    "interval_width": hi - lo,\n                }))\n        print(f"  Battery | {target_variant} | {policy[\'protocol_name\']} | fold {i}/{len(folds)} done", flush=True)\n    return pd.DataFrame(fold_rows), pd.DataFrame(point_rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()\n\n\ndef summarize(preds: pd.DataFrame, fold_metrics: pd.DataFrame):\n    if preds.empty:\n        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{"dataset": "NASA Battery", "policy_decision": "NO_CANDIDATE"}])\n    meta_cols = ["candidate_id", "target_variant", "target_col", "protocol_name", "allow_cycle", "feature_count", "feature_policy_final_eligible", "point_model", "interval_variant"]\n    meta = fold_metrics[[c for c in meta_cols if c in fold_metrics.columns]].drop_duplicates("candidate_id")\n    rows = []\n    for cid, g in preds.groupby("candidate_id"):\n        m = metric_dict(g["y_true"], g["y_pred"], g["lower"], g["upper"])\n        rows.append({"dataset": "NASA Battery", "candidate_id": cid, "folds": g["fold"].nunique(), "assets": g["asset_id"].nunique(), **clean_metric(m)})\n    summary = pd.DataFrame(rows).merge(meta, on="candidate_id", how="left")\n    summary["candidate_final_eligible"] = summary["feature_policy_final_eligible"].fillna(False).astype(bool)\n    gated = pd.DataFrame([gate_candidate(r.to_dict()) for _, r in summary.iterrows()])\n    bad = ~gated["candidate_final_eligible"].fillna(False).astype(bool)\n    gated.loc[bad, "strict_promotion"] = False\n    gated.loc[bad, "blocking_reason"] = gated.loc[bad, "blocking_reason"].astype(str) + "; non-final feature policy"\n    gated = gated.sort_values("gate_rank_score", ascending=False).reset_index(drop=True)\n    strict = gated[gated["strict_promotion"]]\n    if not strict.empty:\n        r = strict.sort_values("gate_rank_score", ascending=False).iloc[0]\n        policy = "STRICT_PROMOTION"\n    else:\n        r = gated.sort_values("gate_rank_score", ascending=False).iloc[0]\n        policy = "NO_PROMOTION_NEAREST_MISS"\n    decision = pd.DataFrame([{\n        "dataset": "NASA Battery", "policy_decision": policy,\n        "candidate_id": r.get("candidate_id"), "target_variant": r.get("target_variant"),\n        "target_col": r.get("target_col"), "protocol_name": r.get("protocol_name"),\n        "allow_cycle": r.get("allow_cycle"), "feature_count": r.get("feature_count"),\n        "candidate_final_eligible": r.get("candidate_final_eligible"),\n        "point_model": r.get("point_model"), "interval_variant": r.get("interval_variant"),\n        "empirical_coverage": r.get("empirical_coverage"),\n        "urgent_critical_coverage": r.get("urgent_critical_coverage"),\n        "interval_false_safe_rate": r.get("interval_false_safe_rate"),\n        "interval_underwarning_rate": r.get("interval_underwarning_rate"),\n        "false_safe_reduction": r.get("false_safe_reduction"),\n        "underwarning_reduction": r.get("underwarning_reduction"),\n        "mean_interval_width": r.get("mean_interval_width"),\n        "gate_pass_count": r.get("gate_pass_count"),\n        "blocking_reason": r.get("blocking_reason"),\n        "gate_rank_score": r.get("gate_rank_score"),\n    }])\n    return summary, gated, decision\n\n\ndef conditionals(preds: pd.DataFrame, decision: pd.DataFrame):\n    if preds.empty or decision.empty or "candidate_id" not in decision.columns:\n        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()\n    cid = decision["candidate_id"].iloc[0]\n    sub = preds[preds["candidate_id"].astype(str).eq(str(cid))].copy()\n    if sub.empty:\n        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()\n    sub["_zone"] = categories(sub["y_true"].astype(float).values)\n    def gtab(col):\n        rows = []\n        for val, g in sub.groupby(col, dropna=False):\n            m = metric_dict(g["y_true"], g["y_pred"], g["lower"], g["upper"])\n            rows.append({"dataset": "NASA Battery", "candidate_id": cid, col: val, **clean_metric(m)})\n        return pd.DataFrame(rows)\n    return gtab("asset_id"), gtab("fold"), gtab("_zone")\n\n\ndef parse_args():\n    p = argparse.ArgumentParser()\n    p.add_argument("--data_root", default=str(DATA_ROOT_DEFAULT))\n    p.add_argument("--output_root", default=str(OUTPUT_ROOT_DEFAULT))\n    p.add_argument("--preset", choices=["quick", "lock"], default="lock")\n    p.add_argument("--max_folds", type=int, default=None)\n    p.add_argument("--max_features", type=int, default=80)\n    p.add_argument("--seed", type=int, default=42)\n    args = p.parse_args()\n    if args.max_folds is None:\n        args.max_folds = 6 if args.preset == "quick" else 14\n    if args.preset == "quick":\n        args.max_features = min(args.max_features, 50)\n    return args\n\n\ndef main():\n    args = parse_args()\n    data_root = Path(args.data_root)\n    output_root = Path(args.output_root)\n    out_dir = output_root / "BATTERY_CAPACITY_ONLY_LOCK_VALIDATION" / f"run_{now_stamp()}"\n    for sub in ["tables", "predictions", "folds", "metadata"]:\n        ensure_dir(out_dir / sub)\n\n    print_block("49_BATTERY_CAPACITY_ONLY_LOCK_VALIDATION.py")\n    print(f"Output dir : {out_dir}")\n    print(f"Preset     : {args.preset}")\n    print("Target rule: exact clean capacity/SOH only. Engineered capacity columns are rejected as targets.")\n\n    table, source = find_battery_table(data_root, output_root)\n    if table is None:\n        raise FileNotFoundError("Battery feature table not found.")\n    df0 = pd.read_csv(table)\n    group_name, group = identify_group(df0)\n    df0 = df0.copy()\n    df0["__group__"] = group.astype(str).values\n\n    accepted_targets, target_audit = audit_targets(df0)\n    print(f"Battery table: {table}")\n    print(f"Rows={len(df0)} Cols={len(df0.columns)} Groups={df0[\'__group__\'].nunique()}")\n    print("Accepted target columns:", accepted_targets)\n    if not accepted_targets:\n        raise ValueError("No exact clean target found. Refusing to use engineered capacity-derived targets.")\n\n    fold_all, point_all, pred_all = [], [], []\n    policy_all, leakage_all, reject_all = [], [], []\n\n    for target_col in accepted_targets:\n        for target_variant, y, target_note in build_target_variants(df0, target_col, df0["__group__"]):\n            finite = np.isfinite(y)\n            df = df0.loc[finite].copy().reset_index(drop=True)\n            y_work = y[finite]\n            df["__group__"] = df["__group__"].astype(str).values\n            print_block(f"TARGET: {target_variant}")\n            print(f"Rows={len(df)} Groups={df[\'__group__\'].nunique()} Note={target_note}")\n            for allow_cycle in [True, False]:\n                protocol_name = "safe_measurements_allow_cycle" if allow_cycle else "safe_measurements_no_cycle"\n                features, policy, audit, rejects = select_features(df, y_work, target_col, "__group__", allow_cycle, args.max_features)\n                policy.update({"dataset": "NASA Battery", "target_col": target_col, "target_variant": target_variant, "protocol_name": protocol_name, "target_note": target_note})\n                policy_all.append(pd.DataFrame([policy]))\n                if not audit.empty:\n                    audit["target_variant"] = target_variant; audit["protocol_name"] = protocol_name\n                    leakage_all.append(audit)\n                if not rejects.empty:\n                    rejects["target_variant"] = target_variant; rejects["protocol_name"] = protocol_name\n                    reject_all.append(rejects)\n                print(f"Protocol {protocol_name}: selected_features={len(features)} final_eligible={policy[\'feature_policy_final_eligible\']}")\n                if len(features) < 3:\n                    continue\n                fold_df, point_df, pred_df = evaluate(df, y_work, target_variant, target_col, "__group__", features, policy, args)\n                if not fold_df.empty: fold_all.append(fold_df)\n                if not point_df.empty: point_all.append(point_df)\n                if not pred_df.empty: pred_all.append(pred_df)\n                # Checkpoint after each protocol.\n                ck_pred = pd.concat(pred_all, ignore_index=True) if pred_all else pd.DataFrame()\n                ck_fold = pd.concat(fold_all, ignore_index=True) if fold_all else pd.DataFrame()\n                ck_summary, ck_gated, ck_decision = summarize(ck_pred, ck_fold)\n                save_csv(ck_decision, out_dir / "tables" / "CHECKPOINT_DECISION.csv")\n                save_csv(ck_gated.head(5000), out_dir / "tables" / "CHECKPOINT_GATED_TOP.csv")\n\n    fold_metrics = pd.concat(fold_all, ignore_index=True) if fold_all else pd.DataFrame()\n    point_metrics = pd.concat(point_all, ignore_index=True) if point_all else pd.DataFrame()\n    preds = pd.concat(pred_all, ignore_index=True) if pred_all else pd.DataFrame()\n    policy_df = pd.concat(policy_all, ignore_index=True) if policy_all else pd.DataFrame()\n    leakage_df = pd.concat(leakage_all, ignore_index=True) if leakage_all else pd.DataFrame()\n    reject_df = pd.concat(reject_all, ignore_index=True) if reject_all else pd.DataFrame()\n\n    summary, gated, decision = summarize(preds, fold_metrics)\n    by_asset, by_fold, by_zone = conditionals(preds, decision)\n\n    final_gate = pd.DataFrame([{\n        "dataset": "NASA Battery",\n        "final_gate": "FINAL_CONFIRMED" if decision["policy_decision"].iloc[0] == "STRICT_PROMOTION" else "NOT_FINAL",\n        "policy_decision": decision["policy_decision"].iloc[0],\n        "candidate_id": decision["candidate_id"].iloc[0],\n        "coverage": decision["empirical_coverage"].iloc[0],\n        "urgent_critical": decision["urgent_critical_coverage"].iloc[0],\n        "width": decision["mean_interval_width"].iloc[0],\n        "interpretation": "Final only if STRICT_PROMOTION using exact clean target and leakage-safe features.",\n    }])\n    metadata = pd.DataFrame([{\n        "dataset": "NASA Battery", "table": str(table), "source": source,\n        "rows": len(df0), "cols": len(df0.columns), "groups": df0["__group__"].nunique(),\n        "group_col_source": group_name, "accepted_targets": ";".join(accepted_targets),\n        "target_rule": "exact clean capacity/SOH only; engineered capacity-derived columns rejected",\n        "max_folds": args.max_folds, "max_features": args.max_features,\n    }])\n\n    save_csv(final_gate, out_dir / "tables" / "00_FINAL_GATE.csv")\n    save_csv(decision, out_dir / "tables" / "01_DECISION.csv")\n    save_csv(summary, out_dir / "tables" / "02_CANDIDATE_SUMMARY.csv")\n    save_csv(gated, out_dir / "tables" / "03_GATED_CANDIDATES.csv")\n    save_csv(target_audit, out_dir / "tables" / "04_TARGET_COLUMN_AUDIT.csv")\n    save_csv(policy_df, out_dir / "tables" / "05_FEATURE_POLICIES.csv")\n    save_csv(leakage_df, out_dir / "tables" / "06_LEAKAGE_AUDIT.csv")\n    save_csv(reject_df, out_dir / "tables" / "07_FEATURE_SELECTION_REJECTIONS.csv")\n    save_csv(by_asset, out_dir / "tables" / "08_CONDITIONAL_BY_ASSET.csv")\n    save_csv(by_fold, out_dir / "tables" / "09_CONDITIONAL_BY_FOLD.csv")\n    save_csv(by_zone, out_dir / "tables" / "10_CONDITIONAL_BY_ZONE.csv")\n    save_csv(point_metrics, out_dir / "tables" / "11_POINT_MODEL_METRICS.csv")\n    save_csv(fold_metrics, out_dir / "folds" / "12_FOLD_INTERVAL_METRICS.csv")\n    save_csv(preds, out_dir / "predictions" / "13_ROW_LEVEL_PREDICTIONS.csv")\n    save_csv(metadata, out_dir / "metadata" / "14_METADATA.csv")\n\n    workbook = out_dir / "BATTERY_CAPACITY_ONLY_LOCK_VALIDATION.xlsx"\n    save_excel(workbook, {\n        "FINAL_GATE": final_gate,\n        "DECISION": decision,\n        "CANDIDATE_SUMMARY": summary.head(30000),\n        "GATED": gated.head(30000),\n        "TARGET_AUDIT": target_audit,\n        "FEATURE_POLICIES": policy_df,\n        "LEAKAGE_AUDIT": leakage_df.head(30000),\n        "BY_ASSET": by_asset,\n        "BY_FOLD": by_fold,\n        "BY_ZONE": by_zone,\n        "METADATA": metadata,\n    })\n    write_json(out_dir / "manifest_49_battery_capacity_only.json", {\n        "generated_at": datetime.now().isoformat(timespec="seconds"),\n        "script": "49_BATTERY_CAPACITY_ONLY_LOCK_VALIDATION.py",\n        "output_dir": str(out_dir),\n        "accepted_targets": accepted_targets,\n        "final_gate": final_gate.to_dict(orient="records"),\n    })\n\n    print_block("FINAL GATE")\n    print(final_gate.to_string(index=False))\n    print_block("DECISION")\n    print(decision.to_string(index=False))\n    print("\\nSaved:")\n    print(workbook)\n    print(out_dir / "tables" / "01_DECISION.csv")\n    print(out_dir / "tables" / "04_TARGET_COLUMN_AUDIT.csv")\n    print("=" * 120)\n\n\nif __name__ == "__main__":\n    main()\n'
PROTOCOL47_SOURCE = '# ============================================================\n# 47_ALL_DATASET_FULL_IMPROVEMENT_VERDICT_ENGINE.py\n# Full integrated improvement + verification + verdict for all five datasets\n# ============================================================\n#\n# Scope:\n#   1. NASA C-MAPSS\n#   2. NASA Battery\n#   3. PRONOSTIA/FEMTO\n#   4. XJTU-SY\n#   5. IMS\n#\n# Purpose:\n#   Run one integrated, conservative evidence engine that:\n#     - uses the correct protocol per dataset family,\n#     - tests multiple point models and interval/calibration variants,\n#     - integrates XJTU/IMS repair variants inside fold evaluation,\n#     - produces one final verdict workbook.\n#\n# Scientific discipline:\n#   - Bearing datasets are evaluated as trajectories:\n#       raw vibration feature rows -> fold-specific HI -> trajectory/similarity RUL\n#       -> integrated conformal/guard intervals -> decision metrics.\n#   - Generic row-wise bearing regression is NOT used as the main evidence.\n#   - XJTU/IMS diagnostic repair variants are integrated into the fold evaluation\n#     using the fold calibration split, not counted as post-hoc final evidence.\n#   - C-MAPSS and Battery are evaluated as feature-table datasets if their\n#     feature tables are found.\n#\n# Recommended run:\n# runfile(\n#     r"scripts\\47_ALL_DATASET_FULL_IMPROVEMENT_VERDICT_ENGINE.py",\n#     args="",\n#     wdir=r"scripts"\n# )\n#\n# Faster debugging:\n# runfile(\n#     r"scripts\\47_ALL_DATASET_FULL_IMPROVEMENT_VERDICT_ENGINE.py",\n#     args="--quick",\n#     wdir=r"scripts"\n# )\n#\n# If bearing raw-feature files are missing, you can force creation by rerunning\n# the known dataset-specific scripts first:\n#   --force_rerun_bearing_extractors\n#\n# Output:\n#   outputs\\ALL_DATASET_FULL_IMPROVEMENT_VERDICT\\run_<timestamp>\n# ============================================================\n\nfrom __future__ import annotations\n\nfrom pathlib import Path\nfrom datetime import datetime\nimport argparse\nimport json\nimport math\nimport os\nimport re\nimport subprocess\nimport sys\nimport traceback\nimport warnings\n\nwarnings.filterwarnings("ignore")\n\nimport numpy as np\nimport pandas as pd\n\nfrom sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor\nfrom sklearn.linear_model import Ridge\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import RobustScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n\ntry:\n    from scipy.stats import spearmanr\nexcept Exception:\n    spearmanr = None\n\n\n# ============================================================\n# Defaults and gates\n# ============================================================\n\nDATA_ROOT_DEFAULT = Path(r"raw_datasets")\nOUTPUT_ROOT_DEFAULT = Path(r"outputs")\nCODE_ROOT_DEFAULT = Path(r"scripts")\n\nPRONOSTIA_ROOT = Path(r"raw_datasets\\FEMTO_PRONOSTIA")\nXJTU_ROOT = Path(r"raw_datasets\\XJTU_SY")\nIMS_ROOT = Path(r"raw_datasets\\IMS_Bearing_optional")\n\nBEARING_OUTPUT_FOLDERS = {\n    "PRONOSTIA/FEMTO": "PRONOSTIA_LITERATURE_HI_SIMILARITY_REPRODUCTION",\n    "XJTU-SY": "XJTU_LITERATURE_HI_SIMILARITY_QUICKTEST",\n    "IMS": "IMS_CLEAN_FAILED_BEARING_VALIDATION",\n}\n\nBEARING_EXTRACTOR_SCRIPTS = {\n    "PRONOSTIA/FEMTO": (\n        "41_PRONOSTIA_LITERATURE_HI_SIMILARITY_REPRODUCTION.py",\n        f"--preset reproduction --data_root {PRONOSTIA_ROOT}",\n    ),\n    "XJTU-SY": (\n        "42_XJTU_LITERATURE_HI_SIMILARITY_QUICKTEST.py",\n        f"--preset audit --data_root {XJTU_ROOT}",\n    ),\n    "IMS": (\n        "43B_IMS_CLEAN_FAILED_BEARING_VALIDATION.py",\n        f"--preset quick --data_root {IMS_ROOT}",\n    ),\n}\n\nDATASET_GATES = {\n    "NASA C-MAPSS": {\n        "coverage_min": 0.88, "urgent_min": 0.80, "max_false_safe": 0.02,\n        "max_underwarning": 0.20, "strict_width_limit": 0.50,\n        "relaxed_width_limit": 0.65, "full_width_reject": 0.75,\n        "min_rows": 1000, "min_assets": 20, "min_folds": 4,\n    },\n    "NASA Battery": {\n        "coverage_min": 0.88, "urgent_min": 0.80, "max_false_safe": 0.02,\n        "max_underwarning": 0.20, "strict_width_limit": 0.50,\n        "relaxed_width_limit": 0.65, "full_width_reject": 0.75,\n        "min_rows": 100, "min_assets": 3, "min_folds": 3,\n    },\n    "PRONOSTIA/FEMTO": {\n        "coverage_min": 0.88, "urgent_min": 0.80, "max_false_safe": 0.02,\n        "max_underwarning": 0.20, "strict_width_limit": 0.50,\n        "relaxed_width_limit": 0.65, "full_width_reject": 0.75,\n        "min_rows": 1000, "min_assets": 6, "min_folds": 5,\n    },\n    "XJTU-SY": {\n        "coverage_min": 0.88, "urgent_min": 0.80, "max_false_safe": 0.02,\n        "max_underwarning": 0.20, "strict_width_limit": 0.50,\n        "relaxed_width_limit": 0.58, "full_width_reject": 0.75,\n        "min_rows": 1000, "min_assets": 8, "min_folds": 6,\n    },\n    "IMS": {\n        "coverage_min": 0.88, "urgent_min": 0.80, "max_false_safe": 0.02,\n        "max_underwarning": 0.20, "strict_width_limit": 0.45,\n        "relaxed_width_limit": 0.60, "full_width_reject": 0.75,\n        "min_rows": 1000, "min_assets": 4, "min_folds": 4,\n    },\n}\n\nSAFE_NAME = {\n    "NASA C-MAPSS": "NASA_CMAPSS",\n    "NASA Battery": "NASA_Battery",\n    "PRONOSTIA/FEMTO": "PRONOSTIA_FEMTO",\n    "XJTU-SY": "XJTU_SY",\n    "IMS": "IMS",\n}\n\n\n# ============================================================\n# Basic utilities\n# ============================================================\n\ndef now_stamp():\n    return datetime.now().strftime("%Y-%m-%d_%H%M%S")\n\n\ndef ensure_dir(path: Path):\n    path.mkdir(parents=True, exist_ok=True)\n\n\ndef print_block(title: str):\n    print("\\n" + "=" * 120)\n    print(title)\n    print("=" * 120, flush=True)\n\n\ndef save_csv(df: pd.DataFrame, path: Path):\n    ensure_dir(path.parent)\n    df.to_csv(path, index=False, encoding="utf-8-sig")\n\n\ndef write_json(path: Path, obj):\n    ensure_dir(path.parent)\n    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")\n\n\ndef save_excel(path: Path, sheets: dict[str, pd.DataFrame]):\n    ensure_dir(path.parent)\n    with pd.ExcelWriter(path, engine="openpyxl") as writer:\n        for name, df in sheets.items():\n            sheet = str(name)[:31]\n            if df is None or df.empty:\n                pd.DataFrame([{"message": "empty"}]).to_excel(writer, sheet_name=sheet, index=False)\n            else:\n                df.to_excel(writer, sheet_name=sheet, index=False)\n\n        wb = writer.book\n        for ws in wb.worksheets:\n            ws.freeze_panes = "A2"\n            ws.auto_filter.ref = ws.dimensions\n            for cell in ws[1]:\n                cell.font = cell.font.copy(bold=True, color="FFFFFF")\n                cell.fill = cell.fill.copy(fill_type="solid", fgColor="1F4E78")\n                cell.alignment = cell.alignment.copy(horizontal="center", vertical="center", wrap_text=True)\n            for col in ws.columns:\n                max_len = 0\n                for cell in col[:700]:\n                    value = "" if cell.value is None else str(cell.value)\n                    max_len = max(max_len, min(len(value), 160))\n                ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 105)\n            for row in ws.iter_rows(min_row=2):\n                for cell in row:\n                    cell.alignment = cell.alignment.copy(vertical="top", wrap_text=True)\n\n\ndef read_csv(path: Path) -> pd.DataFrame:\n    if not path.exists():\n        return pd.DataFrame()\n    try:\n        return pd.read_csv(path)\n    except Exception:\n        return pd.DataFrame()\n\n\ndef newest_run_dir(parent: Path) -> Path | None:\n    if not parent.exists():\n        return None\n    runs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("run_")]\n    if not runs:\n        return None\n    return max(runs, key=lambda p: p.stat().st_mtime)\n\n\ndef safe_float(x, default=np.nan):\n    try:\n        if pd.isna(x):\n            return default\n        return float(x)\n    except Exception:\n        return default\n\n\ndef clip01(x):\n    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)\n\n\ndef robust_spearman(a, b):\n    a = np.asarray(a, dtype=float)\n    b = np.asarray(b, dtype=float)\n    m = np.isfinite(a) & np.isfinite(b)\n    if m.sum() < 3:\n        return 0.0\n    if spearmanr is not None:\n        val = spearmanr(a[m], b[m]).correlation\n        return 0.0 if not np.isfinite(val) else float(val)\n    ar = pd.Series(a[m]).rank().values\n    br = pd.Series(b[m]).rank().values\n    val = np.corrcoef(ar, br)[0, 1]\n    return 0.0 if not np.isfinite(val) else float(val)\n\n\ndef safe_r2(y, p):\n    try:\n        y = np.asarray(y, dtype=float)\n        p = np.asarray(p, dtype=float)\n        if len(y) < 2 or len(np.unique(y)) <= 1:\n            return np.nan\n        return float(r2_score(y, p))\n    except Exception:\n        return np.nan\n\n\ndef run_external_script(code_root: Path, script_name: str, arg_string: str, out_log: Path):\n    script = code_root / script_name\n    if not script.exists():\n        raise FileNotFoundError(f"Missing required script: {script}")\n    cmd = [sys.executable, str(script)] + arg_string.split()\n    ensure_dir(out_log.parent)\n    with out_log.open("w", encoding="utf-8") as f:\n        f.write("COMMAND:\\n" + " ".join(cmd) + "\\n\\n")\n        f.flush()\n        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)\n    if proc.returncode != 0:\n        raise RuntimeError(f"{script_name} failed with return code {proc.returncode}. See {out_log}")\n\n\n# ============================================================\n# Metrics / decision gates\n# ============================================================\n\ndef categories(y):\n    y = np.asarray(y, dtype=float)\n    out = np.full(len(y), "safe", dtype=object)\n    out[y <= 0.50] = "monitor"\n    out[y <= 0.25] = "urgent"\n    out[y <= 0.10] = "critical"\n    return out\n\n\ndef rank_cat(cats):\n    order = {"critical": 0, "urgent": 1, "monitor": 2, "safe": 3}\n    return np.asarray([order.get(str(c), 3) for c in cats])\n\n\ndef metric_dict(y_true, y_pred, lower, upper):\n    y = np.asarray(y_true, dtype=float)\n    p = clip01(y_pred)\n    lo = clip01(lower)\n    hi = clip01(upper)\n\n    tc = categories(y)\n    pc = categories(p)\n    ic = categories(lo)\n\n    tr = rank_cat(tc)\n    pr = rank_cat(pc)\n    ir = rank_cat(ic)\n\n    uc = np.isin(tc, ["urgent", "critical"])\n    critical = tc == "critical"\n    covered = (y >= lo) & (y <= hi)\n    width = hi - lo\n\n    point_false = float(np.mean((pc == "safe") & uc)) if len(y) else np.nan\n    interval_false = float(np.mean((ic == "safe") & uc)) if len(y) else np.nan\n    point_under = float(np.mean(pr > tr)) if len(y) else np.nan\n    interval_under = float(np.mean(ir > tr)) if len(y) else np.nan\n    point_over = float(np.mean(pr < tr)) if len(y) else np.nan\n    interval_over = float(np.mean(ir < tr)) if len(y) else np.nan\n\n    return {\n        "n_rows": int(len(y)),\n        "n_urgent_critical": int(uc.sum()),\n        "n_critical": int(critical.sum()),\n        "RMSE": float(np.sqrt(mean_squared_error(y, p))) if len(y) else np.nan,\n        "MAE": float(mean_absolute_error(y, p)) if len(y) else np.nan,\n        "R2": safe_r2(y, p),\n        "empirical_coverage": float(np.mean(covered)) if len(y) else np.nan,\n        "urgent_critical_coverage": float(np.mean(covered[uc])) if uc.any() else np.nan,\n        "critical_coverage": float(np.mean(covered[critical])) if critical.any() else np.nan,\n        "point_false_safe_rate": point_false,\n        "interval_false_safe_rate": interval_false,\n        "false_safe_reduction": point_false - interval_false,\n        "point_underwarning_rate": point_under,\n        "interval_underwarning_rate": interval_under,\n        "underwarning_reduction": point_under - interval_under,\n        "point_overwarning_rate": point_over,\n        "interval_overwarning_rate": interval_over,\n        "overwarning_increase": interval_over - point_over,\n        "mean_interval_width": float(np.mean(width)) if len(y) else np.nan,\n        "median_interval_width": float(np.median(width)) if len(y) else np.nan,\n        "covered_array": covered,\n        "true_category_array": tc,\n        "point_category_array": pc,\n        "interval_category_array": ic,\n    }\n\n\ndef clean_metric(m):\n    drop = {"covered_array", "true_category_array", "point_category_array", "interval_category_array"}\n    return {k: v for k, v in m.items() if k not in drop}\n\n\ndef gate_candidate(row: dict, dataset: str):\n    gate = DATASET_GATES[dataset]\n    r = dict(row)\n    r["coverage_ok"] = safe_float(r.get("empirical_coverage")) >= gate["coverage_min"]\n    urgent = r.get("urgent_critical_coverage")\n    r["urgent_ok"] = pd.isna(urgent) or safe_float(urgent) >= gate["urgent_min"]\n    r["false_safe_ok"] = safe_float(r.get("interval_false_safe_rate")) <= gate["max_false_safe"]\n    r["underwarning_ok"] = safe_float(r.get("interval_underwarning_rate")) <= gate["max_underwarning"]\n    r["reduction_ok"] = (\n        safe_float(r.get("false_safe_reduction"), 0) >= -1e-12\n        and safe_float(r.get("underwarning_reduction"), 0) >= -1e-12\n    )\n    r["strict_width_ok"] = safe_float(r.get("mean_interval_width")) <= gate["strict_width_limit"]\n    r["relaxed_width_ok"] = safe_float(r.get("mean_interval_width")) <= gate["relaxed_width_limit"]\n    r["not_full_width"] = safe_float(r.get("mean_interval_width")) <= gate["full_width_reject"]\n    r["strict_promotion"] = all([\n        r["coverage_ok"], r["urgent_ok"], r["false_safe_ok"], r["underwarning_ok"],\n        r["reduction_ok"], r["strict_width_ok"], r["not_full_width"],\n    ])\n    r["relaxed_review"] = all([\n        r["coverage_ok"], r["urgent_ok"], r["false_safe_ok"], r["underwarning_ok"],\n        r["reduction_ok"], r["relaxed_width_ok"], r["not_full_width"],\n    ])\n    gate_cols = ["coverage_ok", "urgent_ok", "false_safe_ok", "underwarning_ok", "reduction_ok", "strict_width_ok", "not_full_width"]\n    r["gate_pass_count"] = int(sum(bool(r[c]) for c in gate_cols))\n    r["gate_rank_score"] = (\n        int(r["strict_promotion"]) * 1000\n        + int(r["relaxed_review"]) * 500\n        + r["gate_pass_count"] * 50\n        + safe_float(r.get("urgent_critical_coverage"), 0) * 2\n        - abs(safe_float(r.get("empirical_coverage"), 0) - 0.90)\n        - safe_float(r.get("mean_interval_width"), 999) * 2\n    )\n    fail = []\n    if not r["coverage_ok"]:\n        fail.append(f"coverage {safe_float(r.get(\'empirical_coverage\')):.3f}<{gate[\'coverage_min\']:.2f}")\n    if not r["urgent_ok"]:\n        fail.append(f"urgent {safe_float(r.get(\'urgent_critical_coverage\')):.3f}<{gate[\'urgent_min\']:.2f}")\n    if not r["false_safe_ok"]:\n        fail.append("false-safe gate")\n    if not r["underwarning_ok"]:\n        fail.append("underwarning gate")\n    if not r["reduction_ok"]:\n        fail.append("risk reduction negative")\n    if not r["strict_width_ok"]:\n        fail.append(f"width {safe_float(r.get(\'mean_interval_width\')):.3f}>{gate[\'strict_width_limit\']:.3f}")\n    if not r["not_full_width"]:\n        fail.append("full-width reject")\n    r["blocking_reason"] = "; ".join(fail) if fail else "passes strict"\n    return r\n\n\ndef choose_decision(gated: pd.DataFrame, dataset: str):\n    if gated.empty:\n        return pd.DataFrame([{"dataset": dataset, "policy_decision": "NO_CANDIDATE"}])\n    strict = gated[gated["strict_promotion"]]\n    relaxed = gated[gated["relaxed_review"]]\n    if not strict.empty:\n        r = strict.sort_values("gate_rank_score", ascending=False).iloc[0]\n        policy = "STRICT_PROMOTION"\n    elif not relaxed.empty:\n        r = relaxed.sort_values("gate_rank_score", ascending=False).iloc[0]\n        policy = "RELAXED_PROMOTION_REVIEW"\n    else:\n        r = gated.sort_values("gate_rank_score", ascending=False).iloc[0]\n        policy = "NO_PROMOTION_NEAREST_MISS"\n    return pd.DataFrame([{\n        "dataset": dataset,\n        "policy_decision": policy,\n        "candidate_id": r.get("candidate_id"),\n        "point_model": r.get("point_model"),\n        "interval_variant": r.get("interval_variant"),\n        "empirical_coverage": r.get("empirical_coverage"),\n        "urgent_critical_coverage": r.get("urgent_critical_coverage"),\n        "interval_false_safe_rate": r.get("interval_false_safe_rate"),\n        "interval_underwarning_rate": r.get("interval_underwarning_rate"),\n        "false_safe_reduction": r.get("false_safe_reduction"),\n        "underwarning_reduction": r.get("underwarning_reduction"),\n        "mean_interval_width": r.get("mean_interval_width"),\n        "strict_width_limit": DATASET_GATES[dataset]["strict_width_limit"],\n        "relaxed_width_limit": DATASET_GATES[dataset]["relaxed_width_limit"],\n        "gate_pass_count": r.get("gate_pass_count"),\n        "blocking_reason": r.get("blocking_reason"),\n        "gate_rank_score": r.get("gate_rank_score"),\n    }])\n\n\n# ============================================================\n# Intervals\n# ============================================================\n\ndef quantile_safe(v, q):\n    v = np.asarray(v, dtype=float)\n    v = v[np.isfinite(v)]\n    if len(v) == 0:\n        return 0.0\n    return float(max(0.0, np.quantile(v, q)))\n\n\ndef interval_variants(y_cal, p_cal, p_test, dataset):\n    y_cal = np.asarray(y_cal, dtype=float)\n    p_cal = np.asarray(p_cal, dtype=float)\n    p_test = np.asarray(p_test, dtype=float)\n    abs_res = np.abs(y_cal - p_cal)\n    low_res = p_cal - y_cal\n    high_res = y_cal - p_cal\n    pred_cat = categories(p_test)\n\n    variants = []\n\n    # Core global conformal variants\n    for q in [0.90, 0.95, 0.97, 0.99]:\n        qq = quantile_safe(abs_res, q)\n        variants.append((f"global_{int(q*100)}", clip01(p_test - qq), clip01(p_test + qq), {"q_global": qq}))\n\n    # Integrated repair variants previously found diagnostically\n    qq = quantile_safe(abs_res, 0.90)\n    variants.append(("repair_sym_cal_q90", clip01(p_test - qq), clip01(p_test + qq), {"q_global": qq}))\n\n    # Asymmetric/guard variants\n    specs = [\n        ("asym_l95_u80", 0.95, 0.80, 1.00, 1.00),\n        ("asym_l97_u80", 0.97, 0.80, 1.00, 1.00),\n        ("guard_l95_u80_b12", 0.95, 0.80, 1.20, 1.00),\n        ("guard_l97_u80_b15", 0.97, 0.80, 1.50, 1.00),\n        ("guard_l99_u80_b18", 0.99, 0.80, 1.80, 1.00),\n        ("repair_twosided_guard_l97_u90_b12", 0.97, 0.90, 1.20, 1.20),\n    ]\n    for name, ql, qh, lb, ub in specs:\n        qlow = quantile_safe(low_res, ql)\n        qhigh = quantile_safe(high_res, qh)\n        guard = np.isin(pred_cat, ["monitor", "urgent", "critical"])\n        ql_each = np.where(guard, qlow * lb, qlow)\n        qh_each = np.where(guard, qhigh * ub, qhigh)\n        variants.append((name, clip01(p_test - ql_each), clip01(p_test + qh_each), {"q_low": qlow, "q_high": qhigh, "lower_boost": lb, "upper_boost": ub}))\n\n    # Fixed-width diagnostic variants within budgets, useful to see if calibration-only can solve.\n    gate = DATASET_GATES[dataset]\n    widths = sorted(set([\n        0.05, 0.10, 0.15, 0.20, 0.25, 0.30,\n        min(0.35, gate["strict_width_limit"]),\n        gate["strict_width_limit"],\n        gate["relaxed_width_limit"],\n    ]))\n    for w in widths:\n        variants.append((f"fixed_width_{w:.3f}", clip01(p_test - w/2), clip01(p_test + w/2), {"fixed_width": w}))\n\n    return variants\n\n\n# ============================================================\n# Fold splitting\n# ============================================================\n\ndef make_group_folds(df, dataset, group_col, max_test_groups=8, cal_fraction=0.25, seed=42):\n    groups = sorted(df[group_col].astype(str).dropna().unique().tolist())\n    if len(groups) < 3:\n        return []\n    rng = np.random.default_rng(seed)\n    if max_test_groups and len(groups) > max_test_groups:\n        keep = set(groups[:1] + groups[-1:])\n        rem = [g for g in groups if g not in keep]\n        need = max_test_groups - len(keep)\n        if need > 0 and rem:\n            keep.update(rng.choice(rem, size=min(need, len(rem)), replace=False).tolist())\n        test_groups = sorted(keep)\n    else:\n        test_groups = groups\n\n    folds = []\n    for test_group in test_groups:\n        test_mask = df[group_col].astype(str).eq(str(test_group)).values\n        rem = sorted(df.loc[~test_mask, group_col].astype(str).unique().tolist())\n        if len(rem) < 2:\n            continue\n        cal_n = max(1, int(math.ceil(cal_fraction * len(rem))))\n        cal_groups = set(rng.choice(rem, size=cal_n, replace=False))\n        cal_mask = df[group_col].astype(str).isin(cal_groups).values\n        train_mask = (~test_mask) & (~cal_mask)\n        folds.append((f"LOGO_{test_group}", train_mask, cal_mask, test_mask))\n    return folds\n\n\n# ============================================================\n# Feature-table dataset evaluation: C-MAPSS and Battery\n# ============================================================\n\ndef find_feature_table(data_root: Path, output_root: Path, dataset: str):\n    candidates = []\n    if dataset == "NASA C-MAPSS":\n        exact = output_root / "processed_multi_asset" / "feature_tables" / "cmapss_engine_features.csv"\n        if exact.exists():\n            return exact, "exact_processed_multi_asset"\n        patterns = ["*cmapss*feature*.csv", "*CMAPSS*feature*.csv", "*c-mapss*feature*.csv", "*C-MAPSS*feature*.csv"]\n    else:\n        exacts = [\n            output_root / "processed_multi_asset" / "feature_tables" / "battery_features.csv",\n            output_root / "processed_multi_asset" / "feature_tables" / "nasa_battery_features.csv",\n        ]\n        for e in exacts:\n            if e.exists():\n                return e, "exact_processed_multi_asset"\n        patterns = ["*battery*feature*.csv", "*Battery*feature*.csv", "*NASA*Battery*.csv", "*capacity*feature*.csv"]\n\n    search_roots = [\n        output_root / "processed_multi_asset" / "feature_tables",\n        output_root,\n        data_root,\n    ]\n    for root in search_roots:\n        if not root.exists():\n            continue\n        for pat in patterns:\n            try:\n                for p in root.rglob(pat):\n                    if p.is_file() and p.stat().st_size > 1000:\n                        candidates.append(p)\n            except Exception:\n                pass\n\n    if not candidates:\n        return None, "not_found"\n    # prefer output tables and larger files\n    candidates = sorted(set(candidates), key=lambda p: (0 if str(output_root).lower() in str(p).lower() else 1, -p.stat().st_size, len(str(p))))\n    return candidates[0], "searched"\n\n\ndef choose_target_and_y(df: pd.DataFrame, dataset: str):\n    cols = list(df.columns)\n    lower_map = {c.lower(): c for c in cols}\n\n    if dataset == "NASA C-MAPSS":\n        prefs = ["target_normalized_rul", "target_rul", "rul", "remaining_useful_life"]\n    else:\n        prefs = ["target_soh", "soh", "state_of_health", "capacity_normalized", "capacity", "capacity_ah", "target_capacity"]\n\n    target = None\n    for p in prefs:\n        if p in lower_map:\n            target = lower_map[p]\n            break\n    if target is None:\n        # fallback: choose any numeric column with target/soh/rul/capacity\n        for c in cols:\n            lc = c.lower()\n            if any(k in lc for k in ["target", "soh", "rul", "capacity"]):\n                if pd.api.types.is_numeric_dtype(df[c]):\n                    target = c\n                    break\n    if target is None:\n        return None, None, "no_target_column"\n\n    yraw = pd.to_numeric(df[target], errors="coerce").astype(float)\n    if yraw.notna().sum() < 20:\n        return target, None, "target_too_sparse"\n\n    y = yraw.copy()\n    # Normalize and orient: high = safe/far from failure, low = urgent.\n    if y.max() > 1.5:\n        if "soh" in target.lower() and y.max() > 10:\n            y = y / 100.0\n        else:\n            # RUL/capacity in original units: min-max / max.\n            ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))\n            if ymax > ymin:\n                y = (y - ymin) / (ymax - ymin)\n            else:\n                y = y / max(abs(ymax), 1.0)\n    else:\n        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))\n        # Capacity/SOH may be 0.7..1.0; keep if already health-like.\n        if ymin < -0.05 or ymax > 1.05:\n            y = (y - ymin) / max(ymax - ymin, 1e-12)\n\n    y = np.clip(y.values.astype(float), 0, 1)\n    return target, y, "ok"\n\n\ndef choose_group_col(df: pd.DataFrame, dataset: str):\n    cols = list(df.columns)\n    lower = {c.lower(): c for c in cols}\n\n    if dataset == "NASA C-MAPSS":\n        if "subset" in lower and "unit_number" in lower:\n            return "__group__", df[lower["subset"]].astype(str) + "_unit_" + df[lower["unit_number"]].astype(str)\n        for key in ["engine_id", "unit_number", "unit_id", "asset_id", "engine"]:\n            if key in lower:\n                return lower[key], df[lower[key]].astype(str)\n    else:\n        for key in ["battery_id", "cell_id", "asset_id", "unit_id", "battery", "cell"]:\n            if key in lower:\n                return lower[key], df[lower[key]].astype(str)\n\n    # fallback: create pseudo groups by row blocks, not final-quality but prevents crash\n    n = len(df)\n    g = pd.Series(np.floor(np.arange(n) / max(20, n // 10)).astype(int).astype(str), index=df.index)\n    return "__pseudo_group__", g\n\n\ndef select_feature_columns(df: pd.DataFrame, target_col: str, group_col: str, max_features: int):\n    exclude = {\n        target_col, group_col, "__group__", "__pseudo_group__",\n        "rul", "target_rul", "target_normalized_rul", "soh", "target_soh",\n        "capacity", "capacity_ah", "state_of_health",\n    }\n    cols = []\n    for c in df.columns:\n        if c in exclude or c.lower() in exclude:\n            continue\n        if any(k in c.lower() for k in ["source_file", "file", "path", "split", "set", "dataset"]):\n            continue\n        if pd.api.types.is_numeric_dtype(df[c]):\n            s = pd.to_numeric(df[c], errors="coerce")\n            if s.notna().mean() >= 0.70 and s.nunique(dropna=True) > 2:\n                cols.append(c)\n\n    if len(cols) <= max_features:\n        return cols\n\n    # Select features with strongest absolute Spearman to target if possible.\n    scores = []\n    y = pd.to_numeric(df[target_col], errors="coerce").values\n    for c in cols:\n        try:\n            x = pd.to_numeric(df[c], errors="coerce").values\n            scores.append((c, abs(robust_spearman(x, y))))\n        except Exception:\n            scores.append((c, 0.0))\n    scores = sorted(scores, key=lambda z: z[1], reverse=True)\n    return [c for c, _ in scores[:max_features]]\n\n\ndef feature_table_models(seed=42):\n    return {\n        "HGB": HistGradientBoostingRegressor(\n            max_iter=220, learning_rate=0.04, max_leaf_nodes=23,\n            min_samples_leaf=12, l2_regularization=0.01,\n            random_state=seed, early_stopping=True, validation_fraction=0.15,\n            n_iter_no_change=20,\n        ),\n        "ET": ExtraTreesRegressor(\n            n_estimators=250, min_samples_leaf=3, max_features="sqrt",\n            random_state=seed, n_jobs=-1,\n        ),\n        "RF": RandomForestRegressor(\n            n_estimators=200, min_samples_leaf=4, max_features="sqrt",\n            random_state=seed, n_jobs=-1,\n        ),\n        "RIDGE": Pipeline([\n            ("imputer", SimpleImputer(strategy="median")),\n            ("scaler", RobustScaler()),\n            ("ridge", Ridge(alpha=1.0)),\n        ]),\n    }\n\n\ndef evaluate_feature_table_dataset(dataset: str, df: pd.DataFrame, args, out_dir: Path):\n    target_col, y, target_status = choose_target_and_y(df, dataset)\n    if y is None:\n        raise ValueError(f"{dataset}: target not usable ({target_status})")\n\n    df = df.copy()\n    df["__y__"] = y\n    group_col, groups = choose_group_col(df, dataset)\n    df["__group__"] = groups.values\n\n    max_features = args.max_features_quick if args.quick else args.max_features\n    feature_cols = select_feature_columns(df, target_col, "__group__", max_features)\n    if not feature_cols:\n        raise ValueError(f"{dataset}: no numeric feature columns selected")\n\n    max_test_groups = args.max_test_groups_quick if args.quick else args.max_test_groups\n    folds = make_group_folds(df, dataset, "__group__", max_test_groups=max_test_groups, cal_fraction=0.25, seed=args.seed)\n\n    print(f"{dataset}: rows={len(df)} groups={df[\'__group__\'].nunique()} features={len(feature_cols)} folds={len(folds)} target={target_col}", flush=True)\n\n    fold_rows, pred_rows, point_rows = [], [], []\n    for i, (fold_name, tr_mask, cal_mask, te_mask) in enumerate(folds, start=1):\n        train, cal, test = df.loc[tr_mask], df.loc[cal_mask], df.loc[te_mask]\n        Xtr, ytr = train[feature_cols].values, train["__y__"].values.astype(float)\n        Xcal, ycal = cal[feature_cols].values, cal["__y__"].values.astype(float)\n        Xte, yte = test[feature_cols].values, test["__y__"].values.astype(float)\n\n        model_preds_cal, model_preds_test = [], []\n        for name, model in feature_table_models(args.seed).items():\n            try:\n                # Tree models need imputation too when data have NaN.\n                if name in ["HGB"]:\n                    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])\n                elif name in ["ET", "RF"]:\n                    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])\n                else:\n                    pipe = model\n                pipe.fit(Xtr, ytr)\n                pcal = clip01(pipe.predict(Xcal))\n                pte = clip01(pipe.predict(Xte))\n                model_preds_cal.append(pcal)\n                model_preds_test.append(pte)\n                point_sets = [(name, pcal, pte)]\n            except Exception:\n                continue\n\n            if model_preds_cal:\n                pass\n\n            for point_name, pcal, pte in point_sets:\n                pm = metric_dict(yte, pte, pte, pte)\n                point_rows.append({"dataset": dataset, "fold": fold_name, "point_model": point_name, **clean_metric(pm)})\n                for ivar, lo, hi, qinfo in interval_variants(ycal, pcal, pte, dataset):\n                    m = metric_dict(yte, pte, lo, hi)\n                    cid = f"{point_name}__{ivar}"\n                    row = {\n                        "dataset": dataset, "candidate_id": cid, "point_model": point_name,\n                        "interval_variant": ivar, "fold": fold_name,\n                        "n_train": len(train), "n_cal": len(cal), "n_test": len(test),\n                        "train_groups": train["__group__"].nunique(),\n                        "cal_groups": cal["__group__"].nunique(),\n                        "test_groups": test["__group__"].nunique(),\n                        **clean_metric(m), **qinfo,\n                    }\n                    fold_rows.append(row)\n                    pred_rows.append(pd.DataFrame({\n                        "dataset": dataset, "candidate_id": cid, "point_model": point_name,\n                        "interval_variant": ivar, "fold": fold_name,\n                        "asset_id": test["__group__"].astype(str).values,\n                        "original_order": np.arange(len(test)),\n                        "y_true": yte, "y_pred": pte, "lower": lo, "upper": hi,\n                        "true_category": m["true_category_array"],\n                        "point_category": m["point_category_array"],\n                        "interval_category": m["interval_category_array"],\n                        "covered": m["covered_array"],\n                        "interval_width": hi - lo,\n                    }))\n\n        # Ensemble over available models\n        if len(model_preds_cal) >= 2:\n            pcal = clip01(np.mean(model_preds_cal, axis=0))\n            pte = clip01(np.mean(model_preds_test, axis=0))\n            point_name = "ENSEMBLE_ALL"\n            pm = metric_dict(yte, pte, pte, pte)\n            point_rows.append({"dataset": dataset, "fold": fold_name, "point_model": point_name, **clean_metric(pm)})\n            for ivar, lo, hi, qinfo in interval_variants(ycal, pcal, pte, dataset):\n                m = metric_dict(yte, pte, lo, hi)\n                cid = f"{point_name}__{ivar}"\n                fold_rows.append({\n                    "dataset": dataset, "candidate_id": cid, "point_model": point_name,\n                    "interval_variant": ivar, "fold": fold_name,\n                    "n_train": len(train), "n_cal": len(cal), "n_test": len(test),\n                    "train_groups": train["__group__"].nunique(),\n                    "cal_groups": cal["__group__"].nunique(),\n                    "test_groups": test["__group__"].nunique(),\n                    **clean_metric(m), **qinfo,\n                })\n                pred_rows.append(pd.DataFrame({\n                    "dataset": dataset, "candidate_id": cid, "point_model": point_name,\n                    "interval_variant": ivar, "fold": fold_name,\n                    "asset_id": test["__group__"].astype(str).values,\n                    "original_order": np.arange(len(test)),\n                    "y_true": yte, "y_pred": pte, "lower": lo, "upper": hi,\n                    "true_category": m["true_category_array"],\n                    "point_category": m["point_category_array"],\n                    "interval_category": m["interval_category_array"],\n                    "covered": m["covered_array"],\n                    "interval_width": hi - lo,\n                }))\n\n        print(f"  {dataset} fold {i}/{len(folds)} {fold_name}: done", flush=True)\n\n    fold_df = pd.DataFrame(fold_rows)\n    point_df = pd.DataFrame(point_rows)\n    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()\n    metadata = pd.DataFrame([{\n        "dataset": dataset, "evidence_type": "FEATURE_TABLE_GROUP_VALIDATION",\n        "feature_table_rows": len(df), "groups": df["__group__"].nunique(),\n        "target_col": target_col, "feature_count": len(feature_cols),\n        "folds": len(folds), "feature_table_protocol_note": "Group-aware feature-table validation; verify against official benchmark split before final paper if required.",\n    }])\n    return summarize_predictions(pred_df, dataset), choose_decision(gate_summary(summarize_predictions(pred_df, dataset), dataset), dataset), fold_df, point_df, pred_df, metadata\n\n\n# ============================================================\n# Bearing raw-HI integrated evaluation\n# ============================================================\n\nBEARING_META = {\n    "dataset", "asset_id", "condition_id", "test_key", "failure_note", "source_file",\n    "channel_indices", "channel_columns", "target_normalized_rul", "life_fraction_consumed",\n    "selected_order", "total_files", "total_files_in_test", "total_files_in_asset",\n}\n\n\ndef find_latest_bearing_features(output_root: Path, dataset: str):\n    folder = output_root / BEARING_OUTPUT_FOLDERS[dataset]\n    run = newest_run_dir(folder)\n    if run is None:\n        return None, None\n    raw_dir = run / "raw_features"\n    if not raw_dir.exists():\n        return run, None\n    csvs = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)\n    for p in csvs:\n        if p.stat().st_size > 1000:\n            return run, p\n    return run, None\n\n\ndef validate_bearing_raw_paths(df: pd.DataFrame, dataset: str):\n    issues = []\n    if "asset_id" not in df.columns:\n        issues.append("missing asset_id")\n    elif df["asset_id"].astype(str).str.contains("UNKNOWN", case=False, na=False).any():\n        issues.append("UNKNOWN_ASSET present")\n\n    if "source_file" in df.columns:\n        s = df["source_file"].dropna().astype(str).str.lower()\n        if dataset == "PRONOSTIA/FEMTO":\n            if not s.str.contains("femto|pronostia").any():\n                issues.append("source_file does not show FEMTO/PRONOSTIA root")\n            bad = s.str.contains("xjtu|ims|battery|cmapss|c-mapss")\n        elif dataset == "XJTU-SY":\n            if not s.str.contains("xjtu").any():\n                issues.append("source_file does not show XJTU root")\n            bad = s.str.contains("femto|pronostia|ims|battery|cmapss|c-mapss")\n        else:\n            if not s.str.contains("ims|bearing").any():\n                issues.append("source_file does not show IMS root")\n            bad = s.str.contains("xjtu|femto|pronostia|battery|cmapss|c-mapss")\n        if bad.any():\n            issues.append("foreign dataset source_file paths detected")\n    return issues\n\n\ndef bearing_feature_cols(df: pd.DataFrame):\n    cols = []\n    for c in df.columns:\n        if c in BEARING_META or c.lower() in {x.lower() for x in BEARING_META}:\n            continue\n        if c in ["original_order"]:\n            continue  # not used as ML feature; trajectory only\n        if c.startswith("__"):\n            continue\n        if pd.api.types.is_numeric_dtype(df[c]):\n            s = pd.to_numeric(df[c], errors="coerce")\n            if s.notna().mean() >= 0.70 and s.nunique(dropna=True) > 2:\n                cols.append(c)\n    return cols\n\n\ndef learn_direction(train, feature):\n    corrs = []\n    for _, g in train.groupby("asset_id"):\n        x = pd.to_numeric(g[feature], errors="coerce").values\n        t = pd.to_numeric(g["life_fraction_consumed"], errors="coerce").values\n        c = robust_spearman(x, t)\n        if np.isfinite(c):\n            corrs.append(c)\n    mean_corr = float(np.nanmean(corrs)) if corrs else 0.0\n    return 1.0 if mean_corr >= 0 else -1.0\n\n\ndef transform_feature(df, feature, direction, baseline_n=20):\n    out = pd.Series(index=df.index, dtype=float)\n    for asset, g in df.groupby("asset_id", sort=False):\n        idx = g.index\n        x = direction * pd.to_numeric(g[feature], errors="coerce").astype(float)\n        base = x.head(min(baseline_n, max(3, len(x)//5)))\n        med = base.median()\n        iqr = base.quantile(0.75) - base.quantile(0.25)\n        if not np.isfinite(iqr) or abs(iqr) < 1e-12:\n            iqr = base.std()\n        if not np.isfinite(iqr) or abs(iqr) < 1e-12:\n            iqr = 1.0\n        z = (x - med) / iqr\n        out.loc[idx] = 1.0 / (1.0 + np.exp(-z / 3.0))\n    return out.astype(float)\n\n\ndef score_hi(train, hi):\n    scores = []\n    robs = []\n    ranges = []\n    for _, g in train.groupby("asset_id"):\n        x = pd.to_numeric(hi.loc[g.index], errors="coerce").values\n        t = pd.to_numeric(g["life_fraction_consumed"], errors="coerce").values\n        if len(x) < 5:\n            continue\n        corr = abs(robust_spearman(x, t))\n        dx = np.diff(x)\n        scores.append(corr)\n        robs.append(1.0 / (1.0 + np.nanstd(dx)))\n        ranges.append(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))\n    if not scores:\n        return {"correlation": 0.0, "robustness": 0.0, "range": 0.0, "score": 0.0}\n    final = 0.70*np.nanmean(scores) + 0.15*np.nanmean(robs) + 0.15*np.nanmean(ranges)\n    return {"correlation": float(np.nanmean(scores)), "robustness": float(np.nanmean(robs)), "range": float(np.nanmean(ranges)), "score": float(final)}\n\n\ndef build_hi(train, cal, test, top_n=8, ewm_span=5):\n    feats = bearing_feature_cols(train)\n    rows = []\n    directions = {}\n    for f in feats:\n        try:\n            d = learn_direction(train, f)\n            directions[f] = d\n            h = transform_feature(train, f, d)\n            sc = score_hi(train, h)\n            rows.append({"feature": f, "direction": d, **sc})\n        except Exception:\n            continue\n    cand = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)\n    if cand.empty:\n        raise ValueError("No HI candidates.")\n    top = cand.head(top_n).copy()\n    w = top["score"].clip(lower=0).values.astype(float)\n    if w.sum() <= 0:\n        w = np.ones(len(top))\n    w = w / w.sum()\n    top["fusion_weight"] = w\n\n    def add(part):\n        out = part.copy()\n        fused = np.zeros(len(out), dtype=float)\n        for weight, feat in zip(w, top["feature"]):\n            h = transform_feature(out, feat, directions[feat]).fillna(0).values\n            fused += weight * h\n        out["HI_fused_raw"] = fused\n        out["HI_fused"] = out.groupby("asset_id")["HI_fused_raw"].transform(\n            lambda s: pd.to_numeric(s, errors="coerce").ewm(span=ewm_span, adjust=False).mean()\n        )\n        out["HI_slope_5"] = out.groupby("asset_id")["HI_fused"].diff().rolling(5, min_periods=1).mean().fillna(0.0).values\n        return out\n    return add(train), add(cal), add(test), cand, top\n\n\ndef train_hi_mapping(train):\n    x = pd.to_numeric(train["HI_fused"], errors="coerce").values\n    y = pd.to_numeric(train["life_fraction_consumed"], errors="coerce").values\n    m = np.isfinite(x) & np.isfinite(y)\n    if m.sum() < 10:\n        return None\n    bins = np.unique(np.quantile(x[m], np.linspace(0, 1, 31)))\n    rows = []\n    if len(bins) > 2:\n        for lo, hi in zip(bins[:-1], bins[1:]):\n            mask = m & (x >= lo) & (x <= hi)\n            if mask.sum() >= 3:\n                rows.append((float(np.nanmedian(x[mask])), float(np.nanmedian(y[mask]))))\n    if len(rows) < 3:\n        rows = sorted(zip(x[m], y[m]), key=lambda z: z[0])\n    else:\n        rows = sorted(rows, key=lambda z: z[0])\n    xs = np.asarray([r[0] for r in rows])\n    ys = np.asarray([r[1] for r in rows])\n    ys = np.maximum.accumulate(ys)\n    return xs, ys\n\n\ndef predict_hi_mapping(model, df):\n    if model is None:\n        return np.full(len(df), 0.5)\n    xs, ys = model\n    hi = pd.to_numeric(df["HI_fused"], errors="coerce").values\n    frac = np.interp(hi, xs, ys, left=ys[0], right=ys[-1])\n    return clip01(1.0 - frac)\n\n\ndef resample_prefix(values, length):\n    values = np.asarray(values, dtype=float)\n    if len(values) == 0:\n        return np.zeros(length)\n    if len(values) == 1:\n        return np.full(length, values[0])\n    xp = np.linspace(0, len(values)-1, length)\n    return np.interp(xp, np.arange(len(values)), values)\n\n\ndef trajectory_similarity_predict(train, test, prefix_len=30):\n    groups = {a: g.sort_values("original_order").reset_index(drop=True) for a, g in train.groupby("asset_id", sort=False)}\n    pieces = []\n    for asset, tg in test.groupby("asset_id", sort=False):\n        tg = tg.sort_values("original_order").copy()\n        hi_test = pd.to_numeric(tg["HI_fused"], errors="coerce").values\n        preds = []\n        for i in range(len(tg)):\n            if i < 3:\n                preds.append(1.0)\n                continue\n            prefix = hi_test[:i+1]\n            L = min(prefix_len, len(prefix))\n            pref = resample_prefix(prefix, L)\n            frac_seen = i / max(len(tg)-1, 1)\n            best_dist, best_rul = np.inf, 0.5\n            for _, tr in groups.items():\n                hi_tr = pd.to_numeric(tr["HI_fused"], errors="coerce").values\n                if len(hi_tr) < L + 2:\n                    continue\n                center = int(frac_seen * max(len(hi_tr)-1, 1))\n                offsets = [-30, -20, -12, -6, 0, 6, 12, 20, 30]\n                cand_idx = sorted(set([min(max(L-1, center+d), len(hi_tr)-1) for d in offsets]))\n                for j in cand_idx:\n                    seg = resample_prefix(hi_tr[:j+1], L)\n                    dist = float(np.nanmean((pref - seg) ** 2))\n                    if dist < best_dist:\n                        best_dist = dist\n                        best_rul = float(tr["target_normalized_rul"].iloc[j])\n            preds.append(best_rul)\n        tg["_p"] = clip01(preds)\n        pieces.append(tg[["_p"]])\n    return pd.concat(pieces).sort_index()["_p"].values\n\n\ndef ml_feature_cols(top, max_top=12):\n    cols = ["HI_fused", "HI_fused_raw", "HI_slope_5"]\n    cols += list(top["feature"].head(max_top))\n    return list(dict.fromkeys(cols))\n\n\ndef bearing_point_predictions(train, cal, test, top, seed=42):\n    out = []\n    hm = train_hi_mapping(train)\n    out.append(("HI_MAPPING", predict_hi_mapping(hm, cal), predict_hi_mapping(hm, test)))\n\n    try:\n        out.append(("TRAJECTORY_SIMILARITY", trajectory_similarity_predict(train, cal), trajectory_similarity_predict(train, test)))\n    except Exception:\n        pass\n\n    cols = [c for c in ml_feature_cols(top) if c in train.columns]\n    if cols:\n        for name, model in [\n            ("HGB_HI_FEATURES_NO_ORDER", HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=6, l2_regularization=0.01, random_state=seed, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20)),\n            ("RIDGE_HI_FEATURES_NO_ORDER", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("ridge", Ridge(alpha=1.0))])),\n        ]:\n            try:\n                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)]) if name.startswith("HGB") else model\n                pipe.fit(train[cols].values, train["target_normalized_rul"].astype(float).values)\n                out.append((name, clip01(pipe.predict(cal[cols].values)), clip01(pipe.predict(test[cols].values))))\n            except Exception:\n                pass\n\n    if len(out) >= 2:\n        out.append(("ENSEMBLE_ALL_NO_ORDER_ML", clip01(np.mean([x[1] for x in out], axis=0)), clip01(np.mean([x[2] for x in out], axis=0))))\n    return out\n\n\ndef evaluate_bearing_dataset(dataset: str, raw: pd.DataFrame, args, out_dir: Path):\n    raw = raw.copy()\n    if "target_normalized_rul" not in raw.columns:\n        raise ValueError(f"{dataset}: missing target_normalized_rul")\n    if "life_fraction_consumed" not in raw.columns:\n        raw["life_fraction_consumed"] = 1.0 - pd.to_numeric(raw["target_normalized_rul"], errors="coerce").astype(float)\n    if "original_order" not in raw.columns:\n        raw["original_order"] = raw.groupby("asset_id").cumcount()\n\n    issues = validate_bearing_raw_paths(raw, dataset)\n    if issues:\n        raise ValueError(f"{dataset}: raw path validation failed: {issues}")\n\n    max_test = args.max_bearing_test_assets_quick if args.quick else args.max_bearing_test_assets\n    folds = make_group_folds(raw, dataset, "asset_id", max_test_groups=max_test, cal_fraction=0.25, seed=args.seed)\n    print(f"{dataset}: rows={len(raw)} assets={raw[\'asset_id\'].nunique()} folds={len(folds)}", flush=True)\n\n    fold_rows, point_rows, pred_rows, hi_top_rows, hi_candidate_rows = [], [], [], [], []\n    for i, (fold_name, tr_mask, cal_mask, te_mask) in enumerate(folds, start=1):\n        train0 = raw.loc[tr_mask].copy()\n        cal0 = raw.loc[cal_mask].copy()\n        test0 = raw.loc[te_mask].copy()\n        train, cal, test, hi_cand, hi_top = build_hi(train0, cal0, test0)\n        hi_cand["dataset"] = dataset; hi_cand["fold"] = fold_name\n        hi_top["dataset"] = dataset; hi_top["fold"] = fold_name\n        hi_candidate_rows.append(hi_cand); hi_top_rows.append(hi_top)\n\n        ycal = cal["target_normalized_rul"].astype(float).values\n        yte = test["target_normalized_rul"].astype(float).values\n\n        psets = bearing_point_predictions(train, cal, test, hi_top, seed=args.seed)\n        for point_name, pcal, pte in psets:\n            pm = metric_dict(yte, pte, pte, pte)\n            point_rows.append({"dataset": dataset, "fold": fold_name, "point_model": point_name, **clean_metric(pm)})\n\n            for ivar, lo, hi, qinfo in interval_variants(ycal, pcal, pte, dataset):\n                m = metric_dict(yte, pte, lo, hi)\n                cid = f"{point_name}__{ivar}"\n                fold_rows.append({\n                    "dataset": dataset, "candidate_id": cid, "point_model": point_name,\n                    "interval_variant": ivar, "fold": fold_name,\n                    "n_train": len(train), "n_cal": len(cal), "n_test": len(test),\n                    "train_assets": train["asset_id"].nunique(), "cal_assets": cal["asset_id"].nunique(), "test_assets": test["asset_id"].nunique(),\n                    **clean_metric(m), **qinfo,\n                })\n                pred_rows.append(pd.DataFrame({\n                    "dataset": dataset, "candidate_id": cid, "point_model": point_name, "interval_variant": ivar,\n                    "fold": fold_name, "asset_id": test["asset_id"].astype(str).values,\n                    "original_order": test["original_order"].values,\n                    "y_true": yte, "y_pred": pte, "lower": lo, "upper": hi,\n                    "true_category": m["true_category_array"],\n                    "point_category": m["point_category_array"],\n                    "interval_category": m["interval_category_array"],\n                    "covered": m["covered_array"],\n                    "interval_width": hi - lo,\n                }))\n\n        print(f"  {dataset} fold {i}/{len(folds)} {fold_name}: point_models={len(psets)}", flush=True)\n\n    fold_df = pd.DataFrame(fold_rows)\n    point_df = pd.DataFrame(point_rows)\n    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()\n    hi_top_df = pd.concat(hi_top_rows, ignore_index=True) if hi_top_rows else pd.DataFrame()\n    hi_candidate_df = pd.concat(hi_candidate_rows, ignore_index=True) if hi_candidate_rows else pd.DataFrame()\n    metadata = pd.DataFrame([{\n        "dataset": dataset, "evidence_type": "RAW_HI_INTEGRATED_VALIDATION",\n        "raw_rows": len(raw), "assets": raw["asset_id"].nunique(), "folds": len(folds),\n        "foreign_path_validation": "PASS", "original_order_in_ml_features": False,\n    }])\n    return summarize_predictions(pred_df, dataset), choose_decision(gate_summary(summarize_predictions(pred_df, dataset), dataset), dataset), fold_df, point_df, pred_df, metadata, hi_top_df, hi_candidate_df\n\n\n# ============================================================\n# Candidate summaries and conditional inspection\n# ============================================================\n\ndef summarize_predictions(preds: pd.DataFrame, dataset: str):\n    if preds.empty:\n        return pd.DataFrame()\n    rows = []\n    for cid, g in preds.groupby("candidate_id"):\n        m = metric_dict(g["y_true"], g["y_pred"], g["lower"], g["upper"])\n        rows.append({\n            "dataset": dataset,\n            "candidate_id": cid,\n            "point_model": g["point_model"].iloc[0],\n            "interval_variant": g["interval_variant"].iloc[0],\n            "folds": g["fold"].nunique(),\n            "assets": g["asset_id"].nunique(),\n            **clean_metric(m),\n        })\n    out = pd.DataFrame(rows)\n    out["coverage_gap"] = (out["empirical_coverage"] - 0.90).abs()\n    out["urgent_gap"] = (out["urgent_critical_coverage"] - 0.90).abs()\n    out["score"] = (\n        out["false_safe_reduction"].fillna(0)*6\n        + out["underwarning_reduction"].fillna(0)*3\n        + out["urgent_critical_coverage"].fillna(0)*2\n        - out["coverage_gap"].fillna(1)*1.5\n        - out["urgent_gap"].fillna(1)*1.0\n        - out["mean_interval_width"].fillna(1)*2\n        - np.maximum(out["overwarning_increase"].fillna(0), 0)*0.25\n        - out["RMSE"].fillna(1)*0.02\n    )\n    return out.sort_values("score", ascending=False).reset_index(drop=True)\n\n\ndef gate_summary(summary: pd.DataFrame, dataset: str):\n    if summary.empty:\n        return summary\n    rows = [gate_candidate(r.to_dict(), dataset) for _, r in summary.iterrows()]\n    return pd.DataFrame(rows).sort_values("gate_rank_score", ascending=False).reset_index(drop=True)\n\n\ndef conditional_tables(preds: pd.DataFrame, decisions: pd.DataFrame):\n    if preds.empty or decisions.empty:\n        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()\n    asset_rows, fold_rows, zone_rows = [], [], []\n    for _, d in decisions.iterrows():\n        dataset = d["dataset"]\n        cid = d.get("candidate_id")\n        sub = preds[(preds["dataset"].astype(str).eq(str(dataset))) & (preds["candidate_id"].astype(str).eq(str(cid)))]\n        if sub.empty:\n            continue\n        sub = sub.copy()\n        sub["_zone"] = categories(sub["y_true"].astype(float).values)\n        for key, group_cols, rows in [\n            ("asset", ["asset_id"], asset_rows),\n            ("fold", ["fold"], fold_rows),\n            ("zone", ["_zone"], zone_rows),\n        ]:\n            for vals, g in sub.groupby(group_cols, dropna=False):\n                if not isinstance(vals, tuple):\n                    vals = (vals,)\n                m = metric_dict(g["y_true"], g["y_pred"], g["lower"], g["upper"])\n                row = {"dataset": dataset, "candidate_id": cid}\n                row.update({c: v for c, v in zip(group_cols, vals)})\n                row.update(clean_metric(m))\n                rows.append(row)\n    return pd.DataFrame(asset_rows), pd.DataFrame(fold_rows), pd.DataFrame(zone_rows)\n\n\ndef evidence_sufficiency(decisions: pd.DataFrame, metadata: pd.DataFrame):\n    rows = []\n    for _, d in decisions.iterrows():\n        dataset = d["dataset"]\n        gate = DATASET_GATES[dataset]\n        md = metadata[metadata["dataset"].astype(str).eq(str(dataset))]\n        if md.empty:\n            size_ok = False\n            meta_reason = "missing metadata"\n        else:\n            m = md.iloc[0].to_dict()\n            rows_count = int(safe_float(m.get("raw_rows", m.get("feature_table_rows", 0)), 0))\n            assets = int(safe_float(m.get("assets", m.get("groups", 0)), 0))\n            folds = int(safe_float(m.get("folds", 0), 0))\n            size_ok = rows_count >= gate["min_rows"] and assets >= gate["min_assets"] and folds >= gate["min_folds"]\n            meta_reason = f"rows={rows_count}, assets/groups={assets}, folds={folds}"\n\n        final_ok = d["policy_decision"] == "STRICT_PROMOTION" and size_ok\n        if final_ok:\n            status = "FINAL_CONFIRMED_INTEGRATED"\n        elif d["policy_decision"] in ["STRICT_PROMOTION", "RELAXED_PROMOTION_REVIEW"] and not size_ok:\n            status = "PROMISING_BUT_INSUFFICIENT_RUN_SIZE"\n        elif d["policy_decision"] == "RELAXED_PROMOTION_REVIEW":\n            status = "RELAXED_REVIEW_NOT_FINAL"\n        else:\n            status = "NOT_FINAL_NEEDS_REPAIR_OR_MORE_VALIDATION"\n\n        rows.append({\n            "dataset": dataset,\n            "final_status": status,\n            "usable_for_main_claim_now": bool(final_ok),\n            "policy_decision": d["policy_decision"],\n            "size_ok": bool(size_ok),\n            "size_reason": meta_reason,\n            "coverage": d.get("empirical_coverage"),\n            "urgent_critical": d.get("urgent_critical_coverage"),\n            "width": d.get("mean_interval_width"),\n            "blocking_reason": d.get("blocking_reason"),\n        })\n    out = pd.DataFrame(rows)\n    global_final = "ALL_DATASETS_FINAL_CONFIRMED" if (not out.empty and out["usable_for_main_claim_now"].all()) else "NOT_ALL_DATASETS_FINAL_CONFIRMED"\n    global_df = pd.DataFrame([{\n        "global_verdict": global_final,\n        "dataset_count": len(out),\n        "confirmed_count": int(out["usable_for_main_claim_now"].sum()) if not out.empty else 0,\n        "required_count": len(out),\n        "interpretation": "Use only FINAL_CONFIRMED_INTEGRATED datasets for main manuscript claims; others need targeted repair/rerun.",\n    }])\n    return global_df, out\n\n\n# ============================================================\n# Main\n# ============================================================\n\ndef parse_args():\n    p = argparse.ArgumentParser()\n    p.add_argument("--data_root", default=str(DATA_ROOT_DEFAULT))\n    p.add_argument("--output_root", default=str(OUTPUT_ROOT_DEFAULT))\n    p.add_argument("--code_root", default=str(CODE_ROOT_DEFAULT))\n    p.add_argument("--quick", action="store_true")\n    p.add_argument("--force_rerun_bearing_extractors", action="store_true")\n    p.add_argument("--datasets", nargs="*", default=["NASA C-MAPSS", "NASA Battery", "PRONOSTIA/FEMTO", "XJTU-SY", "IMS"])\n    p.add_argument("--max_features", type=int, default=140)\n    p.add_argument("--max_features_quick", type=int, default=80)\n    p.add_argument("--max_test_groups", type=int, default=10)\n    p.add_argument("--max_test_groups_quick", type=int, default=6)\n    p.add_argument("--max_bearing_test_assets", type=int, default=999)\n    p.add_argument("--max_bearing_test_assets_quick", type=int, default=5)\n    p.add_argument("--seed", type=int, default=42)\n    return p.parse_args()\n\n\ndef main():\n    args = parse_args()\n    data_root = Path(args.data_root)\n    output_root = Path(args.output_root)\n    code_root = Path(args.code_root)\n\n    out_dir = output_root / "ALL_DATASET_FULL_IMPROVEMENT_VERDICT" / f"run_{now_stamp()}"\n    for sub in ["tables", "predictions", "folds", "metadata", "logs"]:\n        ensure_dir(out_dir / sub)\n\n    print_block("47_ALL_DATASET_FULL_IMPROVEMENT_VERDICT_ENGINE.py")\n    print(f"Output dir: {out_dir}")\n    print(f"Datasets  : {args.datasets}")\n    print(f"Quick     : {args.quick}")\n    print(f"Purpose   : integrated improvement + verification + verdict for all five datasets")\n\n    all_summaries, all_gated, all_decisions = [], [], []\n    all_fold_metrics, all_point_metrics, all_predictions = [], [], []\n    all_metadata, all_hi_top, all_hi_cand = [], [], []\n    input_detection_rows, error_rows = [], []\n\n    # Optional rerun bearing extractors to refresh raw feature files.\n    if args.force_rerun_bearing_extractors:\n        for dataset, (script, script_args) in BEARING_EXTRACTOR_SCRIPTS.items():\n            if dataset not in args.datasets:\n                continue\n            print_block(f"FORCE RERUN EXTRACTOR/EVALUATOR: {dataset}")\n            try:\n                run_external_script(code_root, script, script_args, out_dir / "logs" / f"rerun_{SAFE_NAME[dataset]}.log")\n                input_detection_rows.append({"dataset": dataset, "input_action": "force_rerun_completed", "script": script})\n            except Exception as e:\n                error_rows.append({"dataset": dataset, "stage": "force_rerun_bearing_extractor", "error": repr(e)})\n\n    # 1) C-MAPSS and Battery feature-table validations\n    for dataset in ["NASA C-MAPSS", "NASA Battery"]:\n        if dataset not in args.datasets:\n            continue\n        print_block(f"FEATURE-TABLE VALIDATION: {dataset}")\n        try:\n            fpath, source = find_feature_table(data_root, output_root, dataset)\n            input_detection_rows.append({"dataset": dataset, "input_type": "feature_table", "path": str(fpath) if fpath else "", "source": source})\n            if fpath is None:\n                raise FileNotFoundError(f"No feature table found for {dataset}")\n            df = pd.read_csv(fpath)\n            summary, decision, fold_df, point_df, pred_df, metadata = evaluate_feature_table_dataset(dataset, df, args, out_dir)\n            gated = gate_summary(summary, dataset)\n\n            all_summaries.append(summary); all_gated.append(gated); all_decisions.append(decision)\n            all_fold_metrics.append(fold_df); all_point_metrics.append(point_df); all_predictions.append(pred_df); all_metadata.append(metadata)\n\n            save_csv(pred_df, out_dir / "predictions" / f"{SAFE_NAME[dataset]}_predictions.csv")\n            print(decision.to_string(index=False))\n        except Exception as e:\n            error_rows.append({"dataset": dataset, "stage": "feature_table_validation", "error": repr(e)})\n            print(f"ERROR {dataset}: {repr(e)}")\n\n    # 2) Bearing raw-HI integrated validations\n    for dataset in ["PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]:\n        if dataset not in args.datasets:\n            continue\n        print_block(f"RAW-HI INTEGRATED VALIDATION: {dataset}")\n        try:\n            run_dir, raw_path = find_latest_bearing_features(output_root, dataset)\n            input_detection_rows.append({"dataset": dataset, "input_type": "bearing_raw_features", "run_dir": str(run_dir) if run_dir else "", "path": str(raw_path) if raw_path else ""})\n            if raw_path is None:\n                raise FileNotFoundError(f"No raw feature file found for {dataset}. Use --force_rerun_bearing_extractors.")\n            raw = pd.read_csv(raw_path)\n            summary, decision, fold_df, point_df, pred_df, metadata, hi_top_df, hi_cand_df = evaluate_bearing_dataset(dataset, raw, args, out_dir)\n            gated = gate_summary(summary, dataset)\n\n            all_summaries.append(summary); all_gated.append(gated); all_decisions.append(decision)\n            all_fold_metrics.append(fold_df); all_point_metrics.append(point_df); all_predictions.append(pred_df); all_metadata.append(metadata)\n            all_hi_top.append(hi_top_df); all_hi_cand.append(hi_cand_df)\n\n            save_csv(pred_df, out_dir / "predictions" / f"{SAFE_NAME[dataset]}_predictions.csv")\n            print(decision.to_string(index=False))\n        except Exception as e:\n            error_rows.append({"dataset": dataset, "stage": "bearing_raw_hi_validation", "error": repr(e)})\n            print(f"ERROR {dataset}: {repr(e)}")\n\n    summaries = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()\n    gated = pd.concat(all_gated, ignore_index=True) if all_gated else pd.DataFrame()\n    decisions = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()\n    folds = pd.concat(all_fold_metrics, ignore_index=True) if all_fold_metrics else pd.DataFrame()\n    points = pd.concat(all_point_metrics, ignore_index=True) if all_point_metrics else pd.DataFrame()\n    preds = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()\n    metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else pd.DataFrame()\n    hi_top = pd.concat(all_hi_top, ignore_index=True) if all_hi_top else pd.DataFrame()\n    hi_cand = pd.concat(all_hi_cand, ignore_index=True) if all_hi_cand else pd.DataFrame()\n    input_detection = pd.DataFrame(input_detection_rows)\n    errors = pd.DataFrame(error_rows)\n\n    asset_cond, fold_cond, zone_cond = conditional_tables(preds, decisions)\n    global_verdict, dataset_status = evidence_sufficiency(decisions, metadata)\n\n    # Save outputs\n    save_csv(global_verdict, out_dir / "tables" / "00_GLOBAL_VERDICT.csv")\n    save_csv(dataset_status, out_dir / "tables" / "01_DATASET_STATUS.csv")\n    save_csv(decisions, out_dir / "tables" / "02_INTEGRATED_DECISIONS.csv")\n    save_csv(summaries, out_dir / "tables" / "03_CANDIDATE_SUMMARIES.csv")\n    save_csv(gated, out_dir / "tables" / "04_GATED_CANDIDATES.csv")\n    save_csv(asset_cond, out_dir / "tables" / "05_CONDITIONAL_BY_ASSET.csv")\n    save_csv(fold_cond, out_dir / "tables" / "06_CONDITIONAL_BY_FOLD.csv")\n    save_csv(zone_cond, out_dir / "tables" / "07_CONDITIONAL_BY_ZONE.csv")\n    save_csv(points, out_dir / "tables" / "08_POINT_MODEL_METRICS.csv")\n    save_csv(folds, out_dir / "folds" / "09_FOLD_INTERVAL_METRICS.csv")\n    save_csv(metadata, out_dir / "metadata" / "10_EVIDENCE_METADATA.csv")\n    save_csv(input_detection, out_dir / "metadata" / "11_INPUT_DETECTION.csv")\n    save_csv(hi_top, out_dir / "tables" / "12_BEARING_HI_TOP_FEATURES.csv")\n    save_csv(hi_cand, out_dir / "tables" / "13_BEARING_HI_CANDIDATES.csv")\n    save_csv(errors, out_dir / "tables" / "99_ERRORS.csv")\n\n    workbook = out_dir / "ALL_DATASET_FULL_IMPROVEMENT_VERDICT.xlsx"\n    save_excel(workbook, {\n        "GLOBAL_VERDICT": global_verdict,\n        "DATASET_STATUS": dataset_status,\n        "INTEGRATED_DECISIONS": decisions,\n        "CANDIDATE_SUMMARIES": summaries.head(30000),\n        "GATED_CANDIDATES": gated.head(30000),\n        "BY_ASSET": asset_cond.head(30000),\n        "BY_FOLD": fold_cond.head(30000),\n        "BY_ZONE": zone_cond,\n        "POINT_METRICS": points.head(30000),\n        "METADATA": metadata,\n        "INPUT_DETECTION": input_detection,\n        "ERRORS": errors,\n    })\n\n    write_json(out_dir / "manifest_47_all_dataset_verdict.json", {\n        "generated_at": datetime.now().isoformat(timespec="seconds"),\n        "script": "47_ALL_DATASET_FULL_IMPROVEMENT_VERDICT_ENGINE.py",\n        "output_dir": str(out_dir),\n        "datasets": args.datasets,\n        "quick": args.quick,\n        "global_verdict": global_verdict.to_dict(orient="records"),\n        "note": "Bearing evidence is integrated raw-HI validation. Feature-table datasets still require checking against paper-specific benchmark protocol.",\n    })\n\n    print_block("GLOBAL VERDICT")\n    print(global_verdict.to_string(index=False))\n    print_block("DATASET STATUS")\n    print(dataset_status.to_string(index=False) if not dataset_status.empty else "No dataset status.")\n    print_block("INTEGRATED DECISIONS")\n    print(decisions.to_string(index=False) if not decisions.empty else "No decisions.")\n    if not errors.empty:\n        print_block("ERRORS")\n        print(errors.to_string(index=False))\n\n    print("\\nSaved:")\n    print(workbook)\n    print(out_dir / "tables" / "00_GLOBAL_VERDICT.csv")\n    print(out_dir / "tables" / "01_DATASET_STATUS.csv")\n    print(out_dir / "tables" / "02_INTEGRATED_DECISIONS.csv")\n    print("=" * 120)\n\n\nif __name__ == "__main__":\n    main()\n'


def phase1_feature_path(phase1_run: Path, dataset: str) -> Path:
    p = phase1_run / "02_CLEAN_FEATURES" / PHASE1_FILES[dataset]
    if not p.exists():
        raise FileNotFoundError(f"Missing clean feature table for {dataset}: {p}")
    return p


def detect_asset_col(df: pd.DataFrame, dataset: str) -> str | None:
    candidates = {
        "NASA C-MAPSS": ["engine_group", "unit_group", "engine_id", "unit_id", "asset_id", "unit_number", "engine"],
        "NASA Battery": ["battery_id", "cell_id", "asset_id", "battery"],
        "PRONOSTIA/FEMTO": ["asset_id", "bearing_id", "bearing"],
        "XJTU-SY": ["asset_id", "bearing_id", "bearing"],
        "IMS": ["asset_id", "bearing_id", "bearing"],
    }[dataset]
    lookup = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lookup:
            return lookup[norm(c)]
    return None


def input_inventory(phase1_run: Path, out_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    dfs = {}
    # DATASETS is filtered in main() when --only_dataset is supplied.
    # Do not read the parsed command-line object here.
    datasets_to_run = list(DATASETS)
    for dataset in datasets_to_run:
        p = phase1_feature_path(phase1_run, dataset)
        df = pd.read_csv(p, low_memory=False)
        dfs[dataset] = df
        asset_col = detect_asset_col(df, dataset)
        assets = int(df[asset_col].nunique()) if asset_col and asset_col in df.columns else np.nan
        exp = EXPECTED_COUNTS[dataset]
        rows.append({
            "dataset": dataset,
            "feature_file": str(p),
            "rows": len(df),
            "expected_rows": exp["rows"],
            "rows_match_expected": len(df) == exp["rows"],
            "asset_col": asset_col or "",
            "assets": assets,
            "expected_assets": exp["assets"],
            "assets_match_expected": (assets == exp["assets"]) if pd.notna(assets) else False,
            "columns": len(df.columns),
        })
        print(f"{dataset}: rows={len(df):,}/{exp['rows']:,}, assets={assets}/{exp['assets']}, columns={len(df.columns)}")
    inv = pd.DataFrame(rows)
    save_csv(inv, out_dir / "phase1_input_inventory.csv")
    return inv, dfs


def standardize_cmapss_for_52(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "asset_id" not in df.columns:
        for c in ["engine_group", "unit_group", "engine_id", "unit_id", "unit_number"]:
            if c in df.columns:
                df["asset_id"] = df[c].astype(str)
                break
    return df


def standardize_bearing_for_47(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = df.copy()
    if "target_normalized_rul" not in df.columns:
        for c in ["target_health", "health", "health_index", "hi", "target"]:
            if c in df.columns:
                df["target_normalized_rul"] = pd.to_numeric(df[c], errors="coerce").astype(float).clip(0, 1)
                break
    if "target_normalized_rul" not in df.columns:
        raise ValueError(f"{dataset}: cannot create target_normalized_rul from Phase-1 columns.")
    if "life_fraction_consumed" not in df.columns:
        df["life_fraction_consumed"] = 1.0 - pd.to_numeric(df["target_normalized_rul"], errors="coerce").astype(float)
    if "original_order" not in df.columns:
        if "asset_id" not in df.columns:
            raise ValueError(f"{dataset}: missing asset_id.")
        df["original_order"] = df.groupby("asset_id").cumcount()
    return df


def metric_gate(dataset: str, row: dict) -> tuple[int, str]:
    coverage = float(row.get("empirical_coverage", np.nan))
    urgent = float(row.get("urgent_critical_coverage", np.nan))
    false_safe = float(row.get("interval_false_safe_rate", np.nan))
    under = float(row.get("interval_underwarning_rate", np.nan))
    width = float(row.get("mean_interval_width", np.nan))
    reasons = []
    if not np.isfinite(coverage) or coverage < OLD_GATE["coverage_min"]:
        reasons.append(f"coverage {coverage:.4f}<0.88")
    if np.isfinite(urgent) and urgent < OLD_GATE["urgent_min"]:
        reasons.append(f"urgent {urgent:.4f}<0.80")
    if np.isfinite(false_safe) and false_safe > OLD_GATE["false_safe_max"]:
        reasons.append(f"false_safe {false_safe:.4f}>0.02")
    if np.isfinite(under) and under > OLD_GATE["underwarning_max"]:
        reasons.append(f"underwarning {under:.4f}>0.20")
    if not np.isfinite(width) or width > OLD_GATE["width_max"][dataset]:
        reasons.append(f"width {width:.4f}>{OLD_GATE['width_max'][dataset]}")
    return (0, "; ".join(reasons)) if reasons else (1, "passes exact-script gate")


def old_floor_dominance_gate(dataset: str, row: dict) -> tuple[int, str]:
    

    locked = LOCKED_METRICS[dataset]
    coverage = safe_float(row.get("empirical_coverage"))
    urgent = safe_float(row.get("urgent_critical_coverage"))
    false_safe = safe_float(row.get("interval_false_safe_rate"))
    under = safe_float(row.get("interval_underwarning_rate"))
    width = safe_float(row.get("mean_interval_width"))

    reasons = []
    if not np.isfinite(coverage) or coverage < locked["coverage"] - 1e-12:
        reasons.append(f"coverage below locked floor: {coverage}<{locked['coverage']}")
    if not np.isfinite(urgent) or urgent < locked["urgent_critical_coverage"] - 1e-12:
        reasons.append(f"urgent below locked floor: {urgent}<{locked['urgent_critical_coverage']}")
    if not np.isfinite(false_safe) or false_safe > locked["false_safe"] + 1e-12:
        reasons.append(f"false_safe worse than locked floor: {false_safe}>{locked['false_safe']}")
    if not np.isfinite(under) or under > locked["underwarning"] + 1e-12:
        reasons.append(f"underwarning worse than locked floor: {under}>{locked['underwarning']}")

    width_ok = (
        np.isfinite(width)
        and (
            width <= locked["mean_width"] + 1e-12
            or (
                width <= OLD_GATE["width_max"][dataset]
                and np.isfinite(coverage)
                and np.isfinite(urgent)
                and coverage >= locked["coverage"] + 0.02
                and urgent >= locked["urgent_critical_coverage"] + 0.02
            )
        )
    )
    if not width_ok:
        reasons.append(f"width not acceptable vs locked floor: {width}>{locked['mean_width']}")

    return (0, "; ".join(reasons)) if reasons else (1, "passes old-floor dominance gate")


def _copy_candidate_id_to_selected_candidate(df: pd.DataFrame) -> pd.DataFrame:
    

    if df is None or df.empty:
        return df
    out = df.copy()
    if "candidate_id" in out.columns:
        cid = out["candidate_id"].astype(str)
        if "selected_candidate" in out.columns:
            out["selected_candidate_original_stale"] = out["selected_candidate"]
        out["selected_candidate"] = cid
    elif "candidate" in out.columns:
        cid = out["candidate"].astype(str)
        if "selected_candidate" in out.columns:
            out["selected_candidate_original_stale"] = out["selected_candidate"]
        out["candidate_id"] = cid
        out["selected_candidate"] = cid
    return out


def harmonize_selected_artifacts(dataset: str, artifacts: dict) -> dict:
    

    for key in ["selected_summary", "selected_gated", "selected_decision"]:
        if isinstance(artifacts.get(key), pd.DataFrame) and not artifacts[key].empty:
            artifacts[key] = _copy_candidate_id_to_selected_candidate(artifacts[key])

    # Add explicit gate labels to selected metric tables when metric columns exist.
    for key in ["selected_summary", "selected_gated"]:
        df = artifacts.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        rows = []
        for _, rr in df.iterrows():
            r = rr.to_dict()
            script_pass, script_reason = metric_gate(dataset, r)
            floor_pass, floor_reason = old_floor_dominance_gate(dataset, r)
            r["passed_exact_script_gate"] = int(script_pass)
            r["exact_script_gate_reason"] = script_reason
            r["passed_old_floor_dominance_gate"] = int(floor_pass)
            r["old_floor_dominance_reason"] = floor_reason
            rows.append(r)
        artifacts[key] = pd.DataFrame(rows)

    return artifacts



def normalize_result_row(dataset: str, summary: pd.DataFrame, decision: pd.DataFrame) -> dict:
    locked_cid = LOCKED_CANDIDATES[dataset]
    row = {
        "dataset": dataset,
        "locked_candidate": locked_cid,
        "selected_candidate": locked_cid,
    }

    if summary is None or summary.empty:
        row.update({"status": "SELECTED_CANDIDATE_NOT_EVALUATED"})
    else:
        row.update(summary.iloc[0].to_dict())
        row["status"] = "SELECTED_CANDIDATE_EVALUATED"

    # Critical reporting fix: candidate_id is the actual selected candidate.
    # selected_candidate was previously initialized to the locked candidate and
    # could remain stale for Battery robustness selections.
    actual_cid = str(row.get("candidate_id", row.get("candidate", row.get("selected_candidate", locked_cid))))
    if "selected_candidate" in row and str(row.get("selected_candidate", "")) != actual_cid:
        row["selected_candidate_original_stale"] = row.get("selected_candidate", "")
    row["candidate_id"] = actual_cid
    row["selected_candidate"] = actual_cid

    script_pass, script_reason = metric_gate(dataset, row)
    floor_pass, floor_reason = old_floor_dominance_gate(dataset, row)

    # Keep both gates explicit.
    row["passed_exact_script_gate"] = int(script_pass)
    row["exact_script_gate_reason"] = script_reason
    row["passed_old_floor_dominance_gate"] = int(floor_pass)
    row["old_floor_dominance_reason"] = floor_reason

    row["passed_exact_old_gate"] = int(floor_pass)
    row["gate_reason"] = floor_reason

    locked = LOCKED_METRICS[dataset]
    mapping = {
        "coverage": "empirical_coverage",
        "urgent_critical_coverage": "urgent_critical_coverage",
        "false_safe": "interval_false_safe_rate",
        "underwarning": "interval_underwarning_rate",
        "mean_width": "mean_interval_width",
    }
    for short, col in mapping.items():
        row[f"locked_{short}"] = locked[short]
        row[f"delta_{short}"] = row.get(col, np.nan) - locked[short]

    if decision is not None and not decision.empty:
        # Decision table may also have candidate_id; make it visible.
        d0 = decision.iloc[0].to_dict()
        row["decision_candidate_id"] = d0.get("candidate_id", "")
        row["policy_decision_for_selected_subset"] = d0.get("policy_decision", "")
        row["blocking_reason_for_selected_subset"] = d0.get("blocking_reason", "")

    return row

def choose_gate_first_candidate_for_robustness(dataset: str, artifacts: dict, seed: int, selection_policy: str = "primary_locked_or_robustness_gate_first"):
    

    all_summary = artifacts.get("all_gated", pd.DataFrame())
    if all_summary is None or all_summary.empty:
        all_summary = artifacts.get("all_summary", pd.DataFrame())
    if all_summary is None or all_summary.empty:
        return artifacts.get("selected_summary", pd.DataFrame()), artifacts.get("selected_decision", pd.DataFrame()), artifacts.get("selected_pred", pd.DataFrame()), "fallback_locked_no_all_summary"

    cand = all_summary.copy()
    if "candidate_id" not in cand.columns and "candidate" in cand.columns:
        cand["candidate_id"] = cand["candidate"]
    if "dataset" not in cand.columns:
        cand["dataset"] = dataset

    script_passes, script_reasons, floor_passes, floor_reasons = [], [], [], []
    for _, r in cand.iterrows():
        p_script, reason_script = metric_gate(dataset, r.to_dict())
        p_floor, reason_floor = old_floor_dominance_gate(dataset, r.to_dict())
        script_passes.append(int(p_script))
        script_reasons.append(reason_script)
        floor_passes.append(int(p_floor))
        floor_reasons.append(reason_floor)

    cand["_exact_script_gate_pass"] = script_passes
    cand["_exact_script_gate_reason"] = script_reasons
    cand["_old_floor_dominance_pass"] = floor_passes
    cand["_old_floor_dominance_reason"] = floor_reasons

    for c in [
        "empirical_coverage", "urgent_critical_coverage", "interval_false_safe_rate",
        "interval_underwarning_rate", "mean_interval_width", "RMSE", "MAE",
    ]:
        if c not in cand.columns:
            cand[c] = np.nan
        cand[c] = pd.to_numeric(cand[c], errors="coerce")

    policy = str(selection_policy or "").strip().lower()
    v3_policy = policy in {
        "battery_v3_benchmark_informed_old_floor_first",
        "v3_benchmark_informed_old_floor_first",
        "v2_old_floor_dominance_first",
    }

    def _simple_rank(candidate_id: str) -> int:
        s = str(candidate_id)
        order = [
            ("RIDGE", 1), ("ELASTICNET", 2), ("HUBER", 3),
            ("KERNELRIDGE", 4), ("SVR_RBF", 5),
            ("HGB", 6), ("GBR", 7), ("RF", 8), ("ET", 9),
            ("ENSEMBLE_MEDIAN", 10), ("ENSEMBLE_TRIMMED_MEAN", 11),
            ("ENSEMBLE_MEAN", 12), ("STACK_RIDGE", 13),
        ]
        for key, val in order:
            if key in s:
                return val
        return 99

    cand["_simplicity_rank"] = cand["candidate_id"].astype(str).map(_simple_rank)

    if v3_policy and dataset == "NASA Battery":
        eligible = cand[cand["_old_floor_dominance_pass"].eq(1)].copy()
        selected_policy = "battery_v3_benchmark_informed_old_floor_first"

        if not eligible.empty:
            pool = eligible.copy()
            policy_decision = "OLD_FLOOR_DOMINANCE_PROMOTION"
            pool = pool.sort_values(
                [
                    "urgent_critical_coverage",
                    "interval_underwarning_rate",
                    "empirical_coverage",
                    "mean_interval_width",
                    "_simplicity_rank",
                    "RMSE",
                    "MAE",
                ],
                ascending=[False, True, False, True, True, True, True],
            )
        else:
            pool = cand[cand["_exact_script_gate_pass"].eq(1)].copy()
            if pool.empty:
                pool = cand.copy()
            locked = LOCKED_METRICS[dataset]
            pool["_miss_score"] = (
                10000.0 * pool["_exact_script_gate_pass"].fillna(0)
                - 5000.0 * np.maximum(0.0, locked["urgent_critical_coverage"] - pool["urgent_critical_coverage"].fillna(-1.0))
                - 2000.0 * np.maximum(0.0, pool["interval_underwarning_rate"].fillna(1.0) - locked["underwarning"])
                - 5000.0 * np.maximum(0.0, pool["interval_false_safe_rate"].fillna(1.0) - locked["false_safe"])
                - 100.0 * np.maximum(0.0, pool["mean_interval_width"].fillna(999.0) - locked["mean_width"])
                + 100.0 * pool["empirical_coverage"].fillna(0.0)
                - 1.0 * pool["_simplicity_rank"].fillna(99)
            )
            pool = pool.sort_values("_miss_score", ascending=False)
            policy_decision = "NO_OLD_FLOOR_DOMINANCE_NEAREST_MISS"

        best = pool.iloc[[0]].copy()

    else:
        strict = cand[cand["_exact_script_gate_pass"].eq(1)].copy()
        if strict.empty:
            return artifacts.get("selected_summary", pd.DataFrame()), artifacts.get("selected_decision", pd.DataFrame()), artifacts.get("selected_pred", pd.DataFrame()), "fallback_locked_no_strict_candidate"

        strict["_robust_score"] = (
            1000.0
            + 6.0 * strict["urgent_critical_coverage"].fillna(0.0)
            + 4.0 * strict["empirical_coverage"].fillna(0.0)
            - 60.0 * strict["interval_false_safe_rate"].fillna(1.0)
            - 20.0 * strict["interval_underwarning_rate"].fillna(1.0)
            - 2.0 * strict["mean_interval_width"].fillna(1.0)
            - 0.20 * strict["RMSE"].fillna(1.0)
            - 0.10 * strict["MAE"].fillna(1.0)
        )
        best = strict.sort_values("_robust_score", ascending=False).iloc[[0]].copy()
        selected_policy = "predeclared_gate_first_candidate_selection"
        policy_decision = "STRICT_PROMOTION"

    best_id = str(best.iloc[0].get("candidate_id", ""))
    best["selected_candidate"] = best_id
    best["robustness_selection_policy"] = selected_policy
    best["selection_policy_requested"] = selection_policy
    best["robustness_seed"] = int(seed)

    pred = artifacts.get("pred_df", pd.DataFrame())
    best_pred = pd.DataFrame()
    if isinstance(pred, pd.DataFrame) and not pred.empty and "candidate_id" in pred.columns:
        best_pred = pred[pred["candidate_id"].astype(str).eq(best_id)].copy()

    p_floor, reason_floor = old_floor_dominance_gate(dataset, best.iloc[0].to_dict())
    p_script, reason_script = metric_gate(dataset, best.iloc[0].to_dict())

    decision = pd.DataFrame([{
        "dataset": dataset,
        "candidate_id": best_id,
        "selected_candidate": best_id,
        "policy_decision": policy_decision if int(p_floor) else "NO_OLD_FLOOR_DOMINANCE_NEAREST_MISS",
        "passed_exact_script_gate": int(p_script),
        "exact_script_gate_reason": reason_script,
        "passed_old_floor_dominance_gate": int(p_floor),
        "old_floor_dominance_reason": reason_floor,
        "robustness_selection_policy": selected_policy,
        "selection_policy_requested": selection_policy,
        "robustness_seed": int(seed),
    }])
    return best, decision, best_pred, selected_policy


def run_cmapss(mod52: dict, df0: pd.DataFrame, out_dir: Path, mode: str, seed: int) -> dict:
    df0 = standardize_cmapss_for_52(df0)
    target_col = mod52["identify_target"](df0)
    if target_col is None:
        raise ValueError("C-MAPSS target not found.")
    group_name, group = mod52["identify_group"](df0)
    df0 = df0.copy()
    df0["__group__"] = group.astype(str).values
    y = mod52["normalize_target"](df0[target_col])
    finite = np.isfinite(y)
    df = df0.loc[finite].copy().reset_index(drop=True)
    y = y[finite]
    args = SimpleNamespace(max_folds=3 if mode == "quick" else 10, max_features=120, include_et=False, seed=seed, preset="lock")
    protocol_name = "safe_no_leak_allow_cycle"
    features, policy, full_audit, selected_audit, selection_table, feature_scores = mod52["select_features"](
        df, y, target_col, "__group__", True, False, args.max_features
    )
    policy.update({"dataset": "NASA C-MAPSS", "protocol_name": protocol_name, "target_col": target_col})
    fold_df, point_df, pred_df = mod52["evaluate_protocol"](df, y, "__group__", features, protocol_name, policy, args, out_dir)
    summary, gated, decision = mod52["summarize_candidates"](pred_df, fold_df)
    selected = pred_df[pred_df["candidate_id"].astype(str).eq(LOCKED_CANDIDATES["NASA C-MAPSS"])].copy()
    selected_summary, selected_gated, selected_decision = mod52["summarize_candidates"](selected, fold_df)
    return {
        "metadata": pd.DataFrame([{"dataset": "NASA C-MAPSS", "source": "embedded 52 logic", "rows": len(df), "groups": df["__group__"].nunique(), "features": len(features), "target_col": target_col, "group_source": group_name}]),
        "selected_summary": selected_summary, "selected_gated": selected_gated, "selected_decision": selected_decision,
        "all_summary": summary, "all_gated": gated, "all_decision": decision,
        "fold_df": fold_df, "point_df": point_df, "pred_df": pred_df, "selected_pred": selected,
        "feature_policy": pd.DataFrame([policy]), "full_audit": full_audit, "selected_audit": selected_audit,
        "selection_table": selection_table, "feature_scores": feature_scores,
    }



# ---------------------------------------------------------------------
# Top-level metric helpers for Battery V3
# ---------------------------------------------------------------------
# These must exist in the outer reproducer namespace. Do not rely on metric
# helpers inside embedded protocol source strings; those are executed in
# separate namespaces and are not visible to run_battery().
def categories(y):
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), "safe", dtype=object)
    out[y <= 0.50] = "monitor"
    out[y <= 0.25] = "urgent"
    out[y <= 0.10] = "critical"
    return out


def _rank_cat(cats):
    order = {"critical": 0, "urgent": 1, "monitor": 2, "safe": 3}
    return np.asarray([order.get(str(c), 3) for c in cats], dtype=int)


def _safe_r2_np(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) < 2:
        return np.nan
    yy = y[m]
    pp = p[m]
    ss_res = float(np.sum((yy - pp) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    if ss_tot <= 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def metric_dict(y_true, y_pred, lower, upper):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    lo = np.clip(np.asarray(lower, dtype=float), 0.0, 1.0)
    hi = np.clip(np.asarray(upper, dtype=float), 0.0, 1.0)

    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(lo) & np.isfinite(hi)
    y = y[m]
    p = p[m]
    lo = lo[m]
    hi = hi[m]

    if len(y) == 0:
        return {
            "n_rows": 0,
            "n_urgent_critical": 0,
            "n_critical": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "empirical_coverage": np.nan,
            "urgent_critical_coverage": np.nan,
            "critical_coverage": np.nan,
            "point_false_safe_rate": np.nan,
            "interval_false_safe_rate": np.nan,
            "false_safe_reduction": np.nan,
            "point_underwarning_rate": np.nan,
            "interval_underwarning_rate": np.nan,
            "underwarning_reduction": np.nan,
            "point_overwarning_rate": np.nan,
            "interval_overwarning_rate": np.nan,
            "overwarning_increase": np.nan,
            "mean_interval_width": np.nan,
            "median_interval_width": np.nan,
            "covered_array": np.asarray([], dtype=bool),
            "true_category_array": np.asarray([], dtype=object),
            "point_category_array": np.asarray([], dtype=object),
            "interval_category_array": np.asarray([], dtype=object),
        }

    tc = categories(y)
    pc = categories(p)
    ic = categories(lo)

    tr = _rank_cat(tc)
    pr = _rank_cat(pc)
    ir = _rank_cat(ic)

    uc = np.isin(tc, ["urgent", "critical"])
    critical = tc == "critical"
    covered = (y >= lo) & (y <= hi)
    width = hi - lo

    point_false = float(np.mean((pc == "safe") & uc))
    interval_false = float(np.mean((ic == "safe") & uc))
    point_under = float(np.mean(pr > tr))
    interval_under = float(np.mean(ir > tr))
    point_over = float(np.mean(pr < tr))
    interval_over = float(np.mean(ir < tr))

    return {
        "n_rows": int(len(y)),
        "n_urgent_critical": int(uc.sum()),
        "n_critical": int(critical.sum()),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "R2": _safe_r2_np(y, p),
        "empirical_coverage": float(np.mean(covered)),
        "urgent_critical_coverage": float(np.mean(covered[uc])) if uc.any() else np.nan,
        "critical_coverage": float(np.mean(covered[critical])) if critical.any() else np.nan,
        "point_false_safe_rate": point_false,
        "interval_false_safe_rate": interval_false,
        "false_safe_reduction": point_false - interval_false,
        "point_underwarning_rate": point_under,
        "interval_underwarning_rate": interval_under,
        "underwarning_reduction": point_under - interval_under,
        "point_overwarning_rate": point_over,
        "interval_overwarning_rate": interval_over,
        "overwarning_increase": interval_over - point_over,
        "mean_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "covered_array": covered,
        "true_category_array": tc,
        "point_category_array": pc,
        "interval_category_array": ic,
        "interval_category_array": ic,
    }


def clean_metric(m):
    drop = {"covered_array", "true_category_array", "point_category_array", "interval_category_array"}
    return {k: v for k, v in dict(m).items() if k not in drop}


def _battery_v3_model_specs(seed: int) -> dict:
    

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, BayesianRidge
    from sklearn.svm import SVR
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

    return {
        "RIDGE": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("model", Ridge(alpha=1.0))]),
        "ELASTICNET": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", ElasticNet(alpha=0.0005, l1_ratio=0.15, max_iter=10000, random_state=seed))]),
        "HUBER": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("model", HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=500))]),
        "BAYESIAN_RIDGE": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", BayesianRidge())]),
        "SVR_RBF": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=12.0, epsilon=0.015, gamma="scale", cache_size=400, max_iter=30000))]),
        "KERNELRIDGE_RBF": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", KernelRidge(kernel="rbf", alpha=0.01, gamma=0.05))]),
        "HGB": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=260, learning_rate=0.035, max_leaf_nodes=23, min_samples_leaf=8, l2_regularization=0.01, random_state=seed, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20))]),
        "GBR": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(n_estimators=260, learning_rate=0.035, max_depth=3, min_samples_leaf=5, random_state=seed))]),
        "RF": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=260, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=-1))]),
        "ET": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=260, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=-1))]),
    }


def _battery_v3_fit_predict(model_name: str, model, Xfit, yfit, Xcal, Xtest, seed: int, fold_number: int):
    

    audit = {
        "model_name": model_name,
        "fit_rows_available": int(len(yfit)),
        "fit_rows_used": int(len(yfit)),
        "subsampled_for_feasibility": False,
        "status": "ok",
        "error": "",
    }
    try:
        Xf, yf = Xfit, yfit
        if model_name in {"SVR_RBF", "KERNELRIDGE_RBF"} and len(yfit) > 1200:
            rng = np.random.default_rng(int(seed) + 1009 * int(fold_number))
            idx = rng.choice(np.arange(len(yfit)), size=1200, replace=False)
            Xf, yf = Xfit[idx], yfit[idx]
            audit["fit_rows_used"] = int(len(yf))
            audit["subsampled_for_feasibility"] = True
        model.fit(Xf, yf)
        pcal = np.clip(model.predict(Xcal), 0, 1)
        ptest = np.clip(model.predict(Xtest), 0, 1)
        return pcal, ptest, audit
    except Exception as e:
        audit["status"] = "failed"
        audit["error"] = repr(e)
        return None, None, audit


def _battery_v3_interval_variants(ycal, pcal, ptest):
    

    ycal = np.asarray(ycal, dtype=float)
    pcal = np.asarray(pcal, dtype=float)
    ptest = np.asarray(ptest, dtype=float)

    m = np.isfinite(ycal) & np.isfinite(pcal)
    ycal = ycal[m]
    pcal = pcal[m]
    if len(ycal) == 0:
        ycal = np.array([0.5])
        pcal = np.array([0.5])

    abs_res = np.abs(ycal - pcal)
    low_res = pcal - ycal
    high_res = ycal - pcal

    def qsafe_local(v, q):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return 0.0
        return float(max(0.0, np.quantile(v, q)))

    pred_cat = categories(ptest)
    guarded = np.isin(pred_cat, ["monitor", "urgent", "critical"])

    variants = []
    for q in [0.90, 0.95, 0.97, 0.99]:
        qq = qsafe_local(abs_res, q)
        variants.append((f"global_{int(q*100)}", np.clip(ptest - qq, 0, 1), np.clip(ptest + qq, 0, 1), {"q_abs": qq}))

    two_sided_specs = [
        ("twosided_l97_u90", 0.97, 0.90, 1.0, 1.0),
        ("twosided_guard_l97_u90_b12", 0.97, 0.90, 1.2, 1.2),
        ("twosided_guard_l99_u95_b12", 0.99, 0.95, 1.2, 1.2),
        ("twosided_guard_l99_u99_b15", 0.99, 0.99, 1.5, 1.5),
    ]
    for name, ql, qh, lb, ub in two_sided_specs:
        qlow = qsafe_local(low_res, ql)
        qhigh = qsafe_local(high_res, qh)
        ql_each = np.where(guarded, qlow * lb, qlow)
        qh_each = np.where(guarded, qhigh * ub, qhigh)
        variants.append((name, np.clip(ptest - ql_each, 0, 1), np.clip(ptest + qh_each, 0, 1), {"q_low": qlow, "q_high": qhigh, "guard_lb": lb, "guard_ub": ub}))

    urgent_cal = (ycal <= 0.50) | (pcal <= 0.50)
    urgent_abs = abs_res[urgent_cal]
    if len(urgent_abs) < 10:
        urgent_abs = abs_res
    for q, mult in [(0.95, 1.2), (0.97, 1.3), (0.99, 1.5)]:
        qg = qsafe_local(abs_res, q)
        qu = qsafe_local(urgent_abs, q)
        hw = np.where(guarded, max(qg, qu) * mult, qg)
        variants.append((f"urgent_stratified_global_{int(q*100)}_m{str(mult).replace('.', '')}", np.clip(ptest - hw, 0, 1), np.clip(ptest + hw, 0, 1), {"q_global": qg, "q_urgent": qu, "urgent_mult": mult}))

    for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        variants.append((f"fixed_width_{w:.2f}", np.clip(ptest - w / 2, 0, 1), np.clip(ptest + w / 2, 0, 1), {"fixed_width": w}))
    return variants


def _battery_v3_summarize(preds: pd.DataFrame, fold_metrics: pd.DataFrame):
    if preds.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{"dataset": "NASA Battery", "policy_decision": "NO_CANDIDATE"}])

    meta_cols = [
        "candidate_id", "target_variant", "target_col", "protocol_name", "allow_cycle",
        "feature_count", "feature_policy_final_eligible", "point_model", "interval_variant",
        "v3_model_family", "v3_interval_family"
    ]
    meta = fold_metrics[[c for c in meta_cols if c in fold_metrics.columns]].drop_duplicates("candidate_id")

    rows = []
    for cid, g in preds.groupby("candidate_id"):
        m = metric_dict(g["y_true"], g["y_pred"], g["lower"], g["upper"])
        rows.append({"dataset": "NASA Battery", "candidate_id": cid, "folds": g["fold"].nunique(), "assets": g["asset_id"].nunique(), **clean_metric(m)})

    summary = pd.DataFrame(rows).merge(meta, on="candidate_id", how="left")
    summary["candidate_final_eligible"] = summary["feature_policy_final_eligible"].fillna(False).astype(bool)

    gated_rows = []
    for _, rr in summary.iterrows():
        r = rr.to_dict()
        p_script, reason_script = metric_gate("NASA Battery", r)
        p_floor, reason_floor = old_floor_dominance_gate("NASA Battery", r)
        r["passed_exact_script_gate"] = int(p_script)
        r["exact_script_gate_reason"] = reason_script
        r["passed_old_floor_dominance_gate"] = int(p_floor)
        r["old_floor_dominance_reason"] = reason_floor
        if not bool(r.get("candidate_final_eligible", False)):
            r["passed_exact_script_gate"] = 0
            r["passed_old_floor_dominance_gate"] = 0
            r["old_floor_dominance_reason"] = str(r.get("old_floor_dominance_reason", "")) + "; feature policy not final-eligible"
        gated_rows.append(r)

    gated = pd.DataFrame(gated_rows)

    def simple_rank(candidate_id: str) -> int:
        s = str(candidate_id)
        order = [
            ("RIDGE", 1), ("ELASTICNET", 2), ("HUBER", 3), ("BAYESIAN_RIDGE", 4),
            ("KERNELRIDGE", 5), ("SVR_RBF", 6),
            ("HGB", 7), ("GBR", 8), ("RF", 9), ("ET", 10),
            ("ENSEMBLE_MEDIAN", 11), ("ENSEMBLE_TRIMMED_MEAN", 12),
            ("ENSEMBLE_MEAN", 13), ("STACK_RIDGE", 14),
        ]
        for k, v in order:
            if k in s:
                return v
        return 99

    gated["_simplicity_rank"] = gated["candidate_id"].astype(str).map(simple_rank)
    eligible = gated[gated["passed_old_floor_dominance_gate"].eq(1)].copy()

    if not eligible.empty:
        selected = eligible.sort_values(
            [
                "urgent_critical_coverage",
                "interval_underwarning_rate",
                "empirical_coverage",
                "mean_interval_width",
                "_simplicity_rank",
                "RMSE",
                "MAE",
            ],
            ascending=[False, True, False, True, True, True, True],
        ).iloc[[0]].copy()
        decision_label = "OLD_FLOOR_DOMINANCE_PROMOTION"
    else:
        # nearest miss, explicitly non-promotion
        pool = gated[gated["passed_exact_script_gate"].eq(1)].copy()
        if pool.empty:
            pool = gated.copy()
        locked = LOCKED_METRICS["NASA Battery"]
        pool["_miss_score"] = (
            10000.0 * pool["passed_exact_script_gate"].fillna(0)
            - 5000.0 * np.maximum(0.0, locked["urgent_critical_coverage"] - pool["urgent_critical_coverage"].fillna(-1.0))
            - 2000.0 * np.maximum(0.0, pool["interval_underwarning_rate"].fillna(1.0) - locked["underwarning"])
            - 5000.0 * np.maximum(0.0, pool["interval_false_safe_rate"].fillna(1.0) - locked["false_safe"])
            - 100.0 * np.maximum(0.0, pool["mean_interval_width"].fillna(999.0) - locked["mean_width"])
            + 100.0 * pool["empirical_coverage"].fillna(0.0)
            - 1.0 * pool["_simplicity_rank"].fillna(99)
        )
        selected = pool.sort_values("_miss_score", ascending=False).iloc[[0]].copy()
        decision_label = "NO_OLD_FLOOR_DOMINANCE_NEAREST_MISS"

    r = selected.iloc[0].to_dict()
    decision = pd.DataFrame([{
        "dataset": "NASA Battery",
        "policy_decision": decision_label,
        "candidate_id": r.get("candidate_id", ""),
        "selected_candidate": r.get("candidate_id", ""),
        "target_variant": r.get("target_variant", ""),
        "target_col": r.get("target_col", ""),
        "protocol_name": r.get("protocol_name", ""),
        "allow_cycle": r.get("allow_cycle", ""),
        "feature_count": r.get("feature_count", ""),
        "candidate_final_eligible": r.get("candidate_final_eligible", ""),
        "point_model": r.get("point_model", ""),
        "interval_variant": r.get("interval_variant", ""),
        "empirical_coverage": r.get("empirical_coverage", np.nan),
        "urgent_critical_coverage": r.get("urgent_critical_coverage", np.nan),
        "interval_false_safe_rate": r.get("interval_false_safe_rate", np.nan),
        "interval_underwarning_rate": r.get("interval_underwarning_rate", np.nan),
        "mean_interval_width": r.get("mean_interval_width", np.nan),
        "passed_exact_script_gate": r.get("passed_exact_script_gate", 0),
        "passed_old_floor_dominance_gate": r.get("passed_old_floor_dominance_gate", 0),
    }])
    gated = gated.sort_values(
        ["passed_old_floor_dominance_gate", "urgent_critical_coverage", "interval_underwarning_rate", "empirical_coverage", "mean_interval_width"],
        ascending=[False, False, True, False, True],
    ).reset_index(drop=True)
    return summary, gated, decision


def run_battery(mod49: dict, df0: pd.DataFrame, out_dir: Path, mode: str, seed: int) -> dict:
    

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    group_name, group = mod49["identify_group"](df0)
    df0 = df0.copy()
    df0["__group__"] = group.astype(str).values

    accepted, target_audit = mod49["audit_targets"](df0)
    if not accepted:
        raise ValueError("Battery exact clean target not found.")
    target_col = next((c for c in accepted if mod49["norm"](c) == "capacity"), accepted[0])

    variants = mod49["build_target_variants"](df0, target_col, df0["__group__"])
    chosen = next((v for v in variants if "global_minmax_health" in v[0]), variants[0])
    target_variant, y, target_note = chosen

    finite = np.isfinite(y)
    df = df0.loc[finite].copy().reset_index(drop=True)
    y = y[finite].astype(float)
    df["__group__"] = df["__group__"].astype(str)

    max_folds = 3 if mode == "quick" else 10
    feature_cap = 45
    feature_specs = [
        ("safe_measurements_allow_cycle", True, feature_cap),
        ("safe_measurements_no_cycle", False, feature_cap),
    ]

    all_fold_rows, all_point_rows, all_pred_frames, model_audit_rows = [], [], [], []
    registry_rows = []

    for protocol_name, allow_cycle, max_features in feature_specs:
        features, policy, audit, rejects = mod49["select_features"](df, y, target_col, "__group__", allow_cycle, max_features)
        if len(features) < 3 or not bool(policy.get("feature_policy_final_eligible", False)):
            registry_rows.append({"protocol_name": protocol_name, "allow_cycle": allow_cycle, "status": "feature_policy_not_eligible_or_too_few_features", "feature_count": len(features)})
            continue

        policy.update({
            "dataset": "NASA Battery",
            "protocol_name": protocol_name,
            "target_col": target_col,
            "target_variant": target_variant,
            "target_note": target_note,
            "v3_feature_cap": max_features,
        })

        Xdf = df[features].replace([np.inf, -np.inf], np.nan)
        Xdf = Xdf.fillna(Xdf.median(numeric_only=True)).fillna(0.0)
        X = Xdf.to_numpy(float)
        groups = df["__group__"].astype(str).to_numpy()

        folds = mod49["make_folds"](df, "__group__", max_folds, seed)
        models = _battery_v3_model_specs(int(seed))

        for i, (fold_name, tr_mask, cal_mask, te_mask) in enumerate(folds, start=1):
            train_idx = np.where(tr_mask)[0]
            cal_idx_all = np.where(cal_mask)[0]
            test_idx = np.where(te_mask)[0]

            # Split calibration groups deterministically for stack training vs interval calibration.
            cal_groups = sorted(pd.unique(groups[cal_idx_all]).tolist())
            if len(cal_groups) >= 4:
                rng = np.random.default_rng(int(seed) + 7919 * i)
                meta_n = max(1, int(round(0.30 * len(cal_groups))))
                meta_groups = set(rng.choice(cal_groups, size=meta_n, replace=False).tolist())
                meta_idx = np.array([ix for ix in cal_idx_all if groups[ix] in meta_groups], dtype=int)
                interval_cal_idx = np.array([ix for ix in cal_idx_all if groups[ix] not in meta_groups], dtype=int)
                if len(interval_cal_idx) < 5:
                    interval_cal_idx = cal_idx_all
                    meta_idx = np.array([], dtype=int)
            else:
                meta_idx = np.array([], dtype=int)
                interval_cal_idx = cal_idx_all

            Xfit, yfit = X[train_idx], y[train_idx]
            Xcal, ycal = X[interval_cal_idx], y[interval_cal_idx]
            Xtest, ytest = X[test_idx], y[test_idx]

            model_cal_preds, model_test_preds, model_meta_preds = {}, {}, {}

            for model_name, model in models.items():
                pcal, ptest, audit_row = _battery_v3_fit_predict(model_name, model, Xfit, yfit, Xcal, Xtest, int(seed), i)
                audit_row.update({"fold": fold_name, "protocol_name": protocol_name, "feature_count": len(features)})
                model_audit_rows.append(audit_row)
                if pcal is None:
                    continue

                model_cal_preds[model_name] = pcal
                model_test_preds[model_name] = ptest

                if len(meta_idx) > 0:
                    try:
                        model_meta_preds[model_name] = np.clip(model.predict(X[meta_idx]), 0, 1)
                    except Exception:
                        pass

            # Robust ensembles.
            base_names = list(model_test_preds.keys())
            if len(base_names) >= 2:
                cal_stack = np.vstack([model_cal_preds[k] for k in base_names])
                test_stack = np.vstack([model_test_preds[k] for k in base_names])

                model_cal_preds["ENSEMBLE_MEAN"] = np.mean(cal_stack, axis=0)
                model_test_preds["ENSEMBLE_MEAN"] = np.mean(test_stack, axis=0)

                model_cal_preds["ENSEMBLE_MEDIAN"] = np.median(cal_stack, axis=0)
                model_test_preds["ENSEMBLE_MEDIAN"] = np.median(test_stack, axis=0)

                if len(base_names) >= 4:
                    model_cal_preds["ENSEMBLE_TRIMMED_MEAN"] = np.mean(np.sort(cal_stack, axis=0)[1:-1], axis=0)
                    model_test_preds["ENSEMBLE_TRIMMED_MEAN"] = np.mean(np.sort(test_stack, axis=0)[1:-1], axis=0)

            # Leakage-safe stacked ridge: train meta model on held-out meta calibration groups, calibrate on interval_cal groups.
            if len(meta_idx) > 5 and len(model_meta_preds) >= 2:
                common = [k for k in base_names if k in model_meta_preds]
                if len(common) >= 2:
                    try:
                        Zmeta = np.vstack([model_meta_preds[k] for k in common]).T
                        Zcal = np.vstack([model_cal_preds[k] for k in common]).T
                        Ztest = np.vstack([model_test_preds[k] for k in common]).T
                        stack = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
                        stack.fit(Zmeta, y[meta_idx])
                        model_cal_preds["STACK_RIDGE"] = np.clip(stack.predict(Zcal), 0, 1)
                        model_test_preds["STACK_RIDGE"] = np.clip(stack.predict(Ztest), 0, 1)
                    except Exception as e:
                        model_audit_rows.append({"model_name": "STACK_RIDGE", "fold": fold_name, "protocol_name": protocol_name, "status": "failed", "error": repr(e), "fit_rows_available": len(meta_idx), "fit_rows_used": len(meta_idx)})

            for model_name, ptest in model_test_preds.items():
                pcal = model_cal_preds[model_name]
                pm = metric_dict(ytest, ptest, ptest, ptest)
                all_point_rows.append({
                    "dataset": "NASA Battery",
                    "target_variant": target_variant,
                    "protocol_name": protocol_name,
                    "fold": fold_name,
                    "point_model": model_name,
                    **clean_metric(pm),
                })

                for interval_name, lower, upper, qinfo in _battery_v3_interval_variants(ycal, pcal, ptest):
                    cid = f"{target_variant}__{protocol_name}__{model_name}__{interval_name}"
                    m = metric_dict(ytest, ptest, lower, upper)
                    row = {
                        "dataset": "NASA Battery",
                        "candidate_id": cid,
                        "target_variant": target_variant,
                        "target_col": target_col,
                        "protocol_name": protocol_name,
                        "allow_cycle": bool(allow_cycle),
                        "feature_count": len(features),
                        "feature_policy_final_eligible": bool(policy.get("feature_policy_final_eligible", False)),
                        "point_model": model_name,
                        "v3_model_family": model_name,
                        "interval_variant": interval_name,
                        "v3_interval_family": interval_name,
                        "fold": fold_name,
                        "n_train": int(len(train_idx)),
                        "n_cal_total": int(len(cal_idx_all)),
                        "n_stack_meta": int(len(meta_idx)),
                        "n_interval_cal": int(len(interval_cal_idx)),
                        "n_test": int(len(test_idx)),
                        "train_groups": int(len(pd.unique(groups[train_idx]))),
                        "cal_groups": int(len(pd.unique(groups[cal_idx_all]))),
                        "test_groups": int(len(pd.unique(groups[test_idx]))),
                        **clean_metric(m),
                        **qinfo,
                    }
                    all_fold_rows.append(row)

                    all_pred_frames.append(pd.DataFrame({
                        "dataset": "NASA Battery",
                        "candidate_id": cid,
                        "target_variant": target_variant,
                        "protocol_name": protocol_name,
                        "point_model": model_name,
                        "interval_variant": interval_name,
                        "fold": fold_name,
                        "asset_id": df.loc[test_idx, "__group__"].astype(str).values,
                        "row_index": df.loc[test_idx].index.values,
                        "y_true": ytest,
                        "y_pred": ptest,
                        "lower": lower,
                        "upper": upper,
                        "true_category": m["true_category_array"],
                        "point_category": m["point_category_array"],
                        "interval_category": m["interval_category_array"],
                        "covered": m["covered_array"],
                        "interval_width": upper - lower,
                    }))
            print(f"  Battery V3 | {protocol_name} | fold {i}/{len(folds)} done", flush=True)

        registry_rows.append({
            "protocol_name": protocol_name,
            "allow_cycle": allow_cycle,
            "status": "evaluated",
            "feature_count": len(features),
            "models": ";".join(_battery_v3_model_specs(int(seed)).keys()),
            "intervals": "global_90/95/97/99; twosided; urgent_stratified; fixed_width",
        })

    fold_df = pd.DataFrame(all_fold_rows)
    point_df = pd.DataFrame(all_point_rows)
    pred_df = pd.concat(all_pred_frames, ignore_index=True) if all_pred_frames else pd.DataFrame()
    summary, gated, decision = _battery_v3_summarize(pred_df, fold_df)

    selected_id = str(decision.iloc[0].get("candidate_id", "")) if not decision.empty else ""
    selected_pred = pred_df[pred_df["candidate_id"].astype(str).eq(selected_id)].copy() if selected_id and not pred_df.empty else pd.DataFrame()
    selected_summary = summary[summary["candidate_id"].astype(str).eq(selected_id)].copy() if selected_id and not summary.empty else pd.DataFrame()
    selected_gated = gated[gated["candidate_id"].astype(str).eq(selected_id)].copy() if selected_id and not gated.empty else pd.DataFrame()

    return {
        "metadata": pd.DataFrame([{
            "dataset": "NASA Battery",
            "source": "Battery V3 benchmark-informed final try",
            "rows": len(df),
            "groups": df["__group__"].nunique(),
            "target_col": target_col,
            "target_variant": target_variant,
            "group_source": group_name,
            "v3_protocol": "battery_v3_benchmark_informed_old_floor_first",
            "feature_cap": feature_cap,
        }]),
        "selected_summary": selected_summary,
        "selected_gated": selected_gated,
        "selected_decision": decision,
        "all_summary": summary,
        "all_gated": gated,
        "all_decision": decision,
        "fold_df": fold_df,
        "point_df": point_df,
        "pred_df": pred_df,
        "selected_pred": selected_pred,
        "feature_policy": pd.DataFrame(registry_rows),
        "target_audit": target_audit,
        "model_audit": pd.DataFrame(model_audit_rows),
        "v3_candidate_registry": pd.DataFrame(registry_rows),
    }


def run_bearing(mod47: dict, dataset: str, df0: pd.DataFrame, out_dir: Path, mode: str, seed: int) -> dict:
    raw = standardize_bearing_for_47(df0, dataset)
    args = SimpleNamespace(quick=(mode == "quick"), seed=seed, max_bearing_test_assets_quick=3 if mode == "quick" else 999, max_bearing_test_assets=999)
    summary, decision, fold_df, point_df, pred_df, metadata, hi_top, hi_cand = mod47["evaluate_bearing_dataset"](dataset, raw, args, out_dir)
    gated = mod47["gate_summary"](summary, dataset)
    selected_cid = LOCKED_CANDIDATES[dataset]
    selected = pred_df[pred_df["candidate_id"].astype(str).eq(selected_cid)].copy()
    selected_summary = mod47["summarize_predictions"](selected, dataset) if not selected.empty else pd.DataFrame()
    selected_gated = mod47["gate_summary"](selected_summary, dataset) if not selected_summary.empty else pd.DataFrame()
    selected_decision = mod47["choose_decision"](selected_gated, dataset) if not selected_gated.empty else pd.DataFrame([{"dataset": dataset, "policy_decision": "SELECTED_CANDIDATE_NOT_FOUND", "candidate_id": selected_cid}])
    return {
        "metadata": metadata, "selected_summary": selected_summary, "selected_gated": selected_gated, "selected_decision": selected_decision,
        "all_summary": summary, "all_gated": gated, "all_decision": decision,
        "fold_df": fold_df, "point_df": point_df, "pred_df": pred_df, "selected_pred": selected,
        "hi_top": hi_top, "hi_candidates": hi_cand,
    }


def save_artifacts(dataset: str, artifacts: dict, dirs: dict):
    slug = dataset_slug(dataset)
    for key, value in artifacts.items():
        if isinstance(value, pd.DataFrame):
            save_csv(value, dirs["tables"] / f"{slug}__{key}.csv")
    if isinstance(artifacts.get("selected_pred"), pd.DataFrame):
        save_csv(artifacts["selected_pred"], dirs["predictions"] / f"{slug}__selected_predictions.csv")


def plot_diagnostics(dataset: str, pred: pd.DataFrame, fig_dir: Path):
    if pred is None or pred.empty:
        return
    slug = dataset_slug(dataset)
    d = pred.copy()
    if len(d) > 12000:
        d_plot = d.sample(12000, random_state=42)
    else:
        d_plot = d
    # true vs predicted
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(d_plot["y_true"], d_plot["y_pred"], s=6, alpha=0.35)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("True normalized RUL/SOH")
    plt.ylabel("Predicted normalized RUL/SOH")
    plt.title(f"{dataset}: selected candidate true vs predicted")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{slug}__true_vs_pred.png", dpi=220, bbox_inches="tight")
    plt.close()
    # residuals
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(d["y_true"] - d["y_pred"], bins=50)
    plt.xlabel("Residual (true - predicted)")
    plt.ylabel("Count")
    plt.title(f"{dataset}: selected candidate residuals")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{slug}__residuals.png", dpi=220, bbox_inches="tight")
    plt.close()
    # heatmap
    asset_col = "asset_id" if "asset_id" in d.columns else None
    if asset_col and "fold" in d.columns:
        piv = d.groupby([asset_col, "fold"])["covered"].mean().unstack(fill_value=np.nan)
        plt.figure(figsize=(max(6, piv.shape[1] * 0.7), max(4, piv.shape[0] * 0.3)))
        plt.imshow(piv.values, aspect="auto", vmin=0, vmax=1)
        plt.colorbar(label="Coverage")
        plt.yticks(np.arange(len(piv.index)), piv.index)
        plt.xticks(np.arange(len(piv.columns)), piv.columns)
        plt.xlabel("Fold")
        plt.ylabel("Asset/group")
        plt.title(f"{dataset}: selected candidate coverage heatmap")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{slug}__coverage_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_run", required=True, help="Existing clean RAW_EXTRACTION_AUDIT_RUN_<timestamp>. Extraction is skipped.")
    p.add_argument("--output_root", default=None)
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only_dataset", default="", help="Dataset filter added by Q1 protocol router master runner.")
    p.add_argument("--selection_policy", default="primary_locked_or_robustness_gate_first")
    p.add_argument("--export_all_candidates", action="store_true")
    p.add_argument("--export_fold_level_candidates", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # >>> Q1_PROTOCOL_ROUTER_DATASET_FILTER
    _q1_only_dataset = getattr(args, "only_dataset", "")
    if _q1_only_dataset:
        _q1_aliases = {
            "cmapss": "NASA C-MAPSS", "nasa cmapss": "NASA C-MAPSS", "nasa c-mapss": "NASA C-MAPSS", "c-mapss": "NASA C-MAPSS",
            "battery": "NASA Battery", "nasa battery": "NASA Battery",
            "pronostia": "PRONOSTIA/FEMTO", "femto": "PRONOSTIA/FEMTO", "pronostia/femto": "PRONOSTIA/FEMTO",
            "xjtu": "XJTU-SY", "xjtu-sy": "XJTU-SY", "ims": "IMS",
        }
        _q1_wanted = _q1_aliases.get(str(_q1_only_dataset).strip().lower(), str(_q1_only_dataset).strip())
        if _q1_wanted not in ["NASA C-MAPSS", "NASA Battery", "PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]:
            raise ValueError(f"Unknown --only_dataset: {_q1_only_dataset}")
        if "DATASETS" in globals():
            globals()["DATASETS"] = [_q1_wanted]
        if hasattr(args, "datasets"):
            args.datasets = [_q1_wanted]
    # <<< Q1_PROTOCOL_ROUTER_DATASET_FILTER
    phase1_run = Path(args.phase1_run)
    if not phase1_run.exists():
        raise FileNotFoundError(phase1_run)
    suffix = "_junk" if args.mode == "quick" else ""
    run_dir = Path(args.output_root) / f"Q1_SINGLE_FILE_REPRODUCTION_RUN_{stamp()}{suffix}"
    dirs = {
        "manifest": ensure_dir(run_dir / "00_RUN_MANIFEST"),
        "input": ensure_dir(run_dir / "01_INPUT_AUDIT"),
        "tables": ensure_dir(run_dir / "02_TABLES"),
        "figures": ensure_dir(run_dir / "04_DIAGNOSTIC_FIGURES"),
        "runtime": ensure_dir(run_dir / "05_RUNTIME"),
        "errors": ensure_dir(run_dir / "06_ERRORS"),
    }
    banner("Q1 CLEAN SINGLE-FILE REPRODUCER")
    print("SCRIPT_VERSION:", SCRIPT_VERSION)
    print("Extraction: SKIPPED because --phase1_run was provided")
    print("Phase-1 run:", phase1_run)
    print("Output run:", run_dir)
    print("Mode:", args.mode)
    save_json(dirs["manifest"] / "run_config.json", {
        "script_version": SCRIPT_VERSION,
        "phase1_run": str(phase1_run),
        "output_run": str(run_dir),
        "mode": args.mode,
        "only_dataset": getattr(args, "only_dataset", ""),
        "selection_policy": getattr(args, "selection_policy", ""),
        "datasets_to_run": list(DATASETS),
        "extraction_skipped": True,
        "old_result_folders_used": False,
        "external_old_scripts_imported": False,
        "protocol_logic": "embedded in this single file",
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    })
    # Copy Phase-1 audit if available for provenance.
    for sub in ["00_RUN_MANIFEST", "01_DATA_AUDIT"]:
        src = phase1_run / sub
        dst = dirs["input"] / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    banner("LOADING EMBEDDED PROTOCOL LOGIC")
    mod52 = load_embedded_module("embedded_protocol52", PROTOCOL52_SOURCE)
    mod49 = load_embedded_module("embedded_protocol49", PROTOCOL49_SOURCE)
    mod47 = load_embedded_module("embedded_protocol47", PROTOCOL47_SOURCE)
    print("Embedded protocol logic loaded: 52, 49, 47")

    banner("VERIFYING CLEAN PHASE-1 FEATURE TABLES")
    inventory, dfs = input_inventory(phase1_run, dirs["input"])

    final_rows = []
    runtime_rows = []
    all_selected_predictions = []
    errors = []

    def run_dataset(dataset: str, func):
        banner(f"TRAINING SELECTED LOCKED PROTOCOL: {dataset}")
        t0 = time.perf_counter()
        try:
            artifacts = func()
            elapsed = time.perf_counter() - t0
            selected_summary = artifacts.get("selected_summary", pd.DataFrame())
            selected_decision = artifacts.get("selected_decision", pd.DataFrame())
            selected_pred = artifacts.get("selected_pred", pd.DataFrame())
            selection_policy = "primary_seed42_locked_candidate"

            if int(args.seed) != 42:
                selected_summary, selected_decision, selected_pred, selection_policy = choose_gate_first_candidate_for_robustness(dataset, artifacts, args.seed, getattr(args, "selection_policy", "primary_locked_or_robustness_gate_first"))
                artifacts["selected_summary"] = selected_summary
                artifacts["selected_decision"] = selected_decision
                artifacts["selected_pred"] = selected_pred

            artifacts = harmonize_selected_artifacts(dataset, artifacts)
            selected_summary = artifacts.get("selected_summary", pd.DataFrame())
            selected_decision = artifacts.get("selected_decision", pd.DataFrame())
            selected_pred = artifacts.get("selected_pred", pd.DataFrame())

            save_artifacts(dataset, artifacts, dirs)
            row = normalize_result_row(dataset, selected_summary, selected_decision)
            row["selection_policy"] = selection_policy
            row["seed"] = int(args.seed)
            row["runtime_sec"] = elapsed
            final_rows.append(row)
            if isinstance(selected_pred, pd.DataFrame) and not selected_pred.empty:
                all_selected_predictions.append(selected_pred.assign(dataset=dataset))
                plot_diagnostics(dataset, selected_pred, dirs["figures"])
            runtime_rows.append({"dataset": dataset, "candidate": row.get("selected_candidate", LOCKED_CANDIDATES[dataset]), "locked_candidate": LOCKED_CANDIDATES[dataset], "runtime_sec": elapsed})
            print(
                f"{dataset} done: coverage={row.get('empirical_coverage', np.nan):.6f}, "
                f"urgent={row.get('urgent_critical_coverage', np.nan):.6f}, "
                f"false_safe={row.get('interval_false_safe_rate', np.nan):.6f}, "
                f"underwarning={row.get('interval_underwarning_rate', np.nan):.6f}, "
                f"width={row.get('mean_interval_width', np.nan):.6f}, "
                f"gate={row.get('passed_exact_old_gate')} {row.get('gate_reason')}"
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            errors.append({"dataset": dataset, "error": repr(e), "traceback": tb})
            runtime_rows.append({"dataset": dataset, "candidate": LOCKED_CANDIDATES[dataset], "runtime_sec": elapsed, "error": repr(e)})
            (dirs["errors"] / f"ERROR__{dataset_slug(dataset)}.txt").write_text(tb, encoding="utf-8")
            print(f"ERROR {dataset}: {repr(e)}")

    # Run only the filtered DATASETS list. This prevents --only_dataset runs from
    # trying to access feature tables that were intentionally not loaded.
    if "NASA C-MAPSS" in DATASETS:
        run_dataset("NASA C-MAPSS", lambda: run_cmapss(mod52, dfs["NASA C-MAPSS"], dirs["tables"], args.mode, args.seed))
    if "NASA Battery" in DATASETS:
        run_dataset("NASA Battery", lambda: run_battery(mod49, dfs["NASA Battery"], dirs["tables"], args.mode, args.seed))
    for ds in ["PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]:
        if ds in DATASETS:
            run_dataset(ds, lambda d=ds: run_bearing(mod47, d, dfs[d], dirs["tables"], args.mode, args.seed))

    final_df = pd.DataFrame(final_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    save_csv(final_df, dirs["tables"] / "SINGLE_FILE_SELECTED_CANDIDATE_METRICS.csv")
    save_csv(runtime_df, dirs["runtime"] / "runtime.csv")
    if errors:
        save_csv(pd.DataFrame(errors), dirs["errors"] / "errors.csv")
    if all_selected_predictions:
        all_pred = pd.concat(all_selected_predictions, ignore_index=True)
        save_csv(all_pred, dirs["predictions"] / "ALL_SELECTED_CANDIDATE_ROW_LEVEL_PREDICTIONS.csv")
    fig_audit = pd.DataFrame([{"figure": str(p), "exists": p.exists()} for p in sorted(dirs["figures"].glob("*.png"))])
    save_csv(fig_audit, dirs["figures"] / "figure_generation_audit.csv")

    verdict = {
        "run_dir": str(run_dir),
        "script_version": SCRIPT_VERSION,
        "mode": args.mode,
        "extraction_skipped": True,
        "phase1_counts_all_match": bool(inventory["rows_match_expected"].all() and inventory["assets_match_expected"].all()),
        "datasets_completed": int(len(final_df)),
        "datasets_expected": int(len(DATASETS)),
        "errors": len(errors),
        "all_selected_candidates_pass_gate": bool(final_df["passed_exact_old_gate"].all()) if len(final_df) == len(DATASETS) else False,
        "all_selected_candidates_pass_old_floor_dominance_gate": bool(final_df["passed_old_floor_dominance_gate"].all()) if "passed_old_floor_dominance_gate" in final_df.columns and len(final_df) == len(DATASETS) else False,
    }
    save_json(dirs["manifest"] / "SINGLE_FILE_REPRODUCTION_VERDICT.json", verdict)

    banner("SINGLE-FILE REPRODUCTION COMPLETE")
    print("Run directory:", run_dir)
    print("Metrics table:", dirs["tables"] / "SINGLE_FILE_SELECTED_CANDIDATE_METRICS.csv")
    if errors:
        print("Errors table:", dirs["errors"] / "errors.csv")
    if not final_df.empty:
        show = ["dataset", "selected_candidate", "candidate_id", "locked_candidate", "empirical_coverage", "urgent_critical_coverage", "interval_false_safe_rate", "interval_underwarning_rate", "mean_interval_width", "passed_exact_script_gate", "passed_old_floor_dominance_gate", "passed_exact_old_gate", "gate_reason", "delta_coverage", "delta_urgent_critical_coverage", "delta_mean_width"]
        print("\nSINGLE-FILE TRAINED SELECTED CANDIDATE METRICS")
        print(final_df[[c for c in show if c in final_df.columns]].to_string(index=False))
    print("\nVERDICT")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# Q1_MASTER_REPAIR_PATCH_V4
# END_Q1_MASTER_REPAIR_PATCH_V4

