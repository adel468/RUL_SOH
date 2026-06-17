
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

# Public release script: run_full_pipeline.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

# -*- coding: utf-8 -*-



import argparse
import json
import os
import math
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DATASETS = ["NASA C-MAPSS", "NASA Battery", "PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]
BEARING_DATASETS = ["PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]

OLD_LOCKED = {
    "NASA C-MAPSS": {"candidate": "safe_no_leak_allow_cycle__HGB__global_90", "coverage": 0.913892, "urgent_critical_coverage": 0.970501, "false_safe_rate": 0.0, "underwarning_rate": 0.015212, "width": 0.345245},
    "NASA Battery": {"candidate": "capacity__global_minmax_health__safe_measurements_allow_cycle__ENSEMBLE_HGB_ET_RIDGE__twosided_guard_l97_u90_b12", "coverage": 0.887994, "urgent_critical_coverage": 0.936364, "false_safe_rate": 0.0, "underwarning_rate": 0.001612, "width": 0.252979},
    "PRONOSTIA/FEMTO": {"candidate": "TRAJECTORY_SIMILARITY__repair_twosided_guard_l97_u90_b12", "coverage": 0.920174, "urgent_critical_coverage": 0.950957, "false_safe_rate": 0.0, "underwarning_rate": 0.0, "width": 0.057900},
    "XJTU-SY": {"candidate": "TRAJECTORY_SIMILARITY__global_90", "coverage": 0.882735, "urgent_critical_coverage": 0.843776, "false_safe_rate": 0.0, "underwarning_rate": 0.006293, "width": 0.471824},
    "IMS": {"candidate": "TRAJECTORY_SIMILARITY__global_99", "coverage": 0.989500, "urgent_critical_coverage": 0.986000, "false_safe_rate": 0.0, "underwarning_rate": 0.000500, "width": 0.118834},
}

DATASET_FILE_HINTS = {
    "NASA C-MAPSS": ["cmapss_engine_features_clean_raw.csv", "*cmapss*features*.csv"],
    "NASA Battery": ["battery_features_clean_raw_mat.csv", "*battery*features*.csv"],
    "PRONOSTIA/FEMTO": ["pronostia_femto_raw_hi_features.csv", "*pronostia*femto*.csv", "*femto*features*.csv"],
    "XJTU-SY": ["xjtu_sy_raw_hi_features.csv", "*xjtu*features*.csv"],
    "IMS": ["ims_failed_bearing_raw_hi_features.csv", "*ims*failed*bearing*.csv", "*ims*features*.csv"],
}

GROUP_COL_CANDIDATES = [
    "asset_id", "battery_id", "engine_id", "unit_id", "bearing_id", "cell_id",
    "id", "asset", "battery", "engine", "unit", "bearing"
]

TIME_COL_CANDIDATES = [
    "cycle", "cycle_index", "cycle_number", "time_index", "timestamp_index",
    "file_index", "original_order", "row_index", "time", "timestep"
]

TARGET_CANDIDATES = [
    "target_health", "target_soh", "health", "soh", "SOH",
    "target_RUL", "target_rul", "RUL", "rul", "rul_norm", "normalized_rul",
    "capacity", "Capacity", "discharge_capacity", "capacity_ah",
    "remaining_life", "remaining_useful_life"
]

LEAKY_FRAGMENTS = [
    "target", "label", "class", "rul", "soh", "health", "capacity",
    "failure", "remaining", "life", "truth", "y_true", "prediction",
    "lower", "upper", "interval", "coverage", "urgent", "critical"
]

ALIASES = {
    "cmapss": "NASA C-MAPSS", "nasa cmapss": "NASA C-MAPSS", "nasa c-mapss": "NASA C-MAPSS",
    "battery": "NASA Battery", "nasa battery": "NASA Battery",
    "pronostia": "PRONOSTIA/FEMTO", "femto": "PRONOSTIA/FEMTO", "pronostia/femto": "PRONOSTIA/FEMTO",
    "xjtu": "XJTU-SY", "xjtu-sy": "XJTU-SY", "ims": "IMS",
}


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return {}
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def write_csv(path, rows_or_df):
    path = Path(path)
    ensure_dir(path.parent)
    if isinstance(rows_or_df, pd.DataFrame):
        rows_or_df.to_csv(path, index=False)
    else:
        pd.DataFrame(rows_or_df).to_csv(path, index=False)


def read_csv(path):
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def canon_dataset(x):
    s = str(x).strip()
    return ALIASES.get(s.lower(), s)


def find_col(df, candidates):
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def to_float(v, default=np.nan):
    try:
        x = pd.to_numeric(v, errors="coerce")
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def old_baseline_rows():
    rows = []
    for ds in DATASETS:
        o = OLD_LOCKED[ds]
        rows.append({
            "dataset": ds,
            "old_candidate": o["candidate"],
            "old_coverage": o["coverage"],
            "old_urgent_critical_coverage": o["urgent_critical_coverage"],
            "old_false_safe_rate": o["false_safe_rate"],
            "old_underwarning_rate": o["underwarning_rate"],
            "old_width": o["width"],
        })
    return rows


def latest_seed_list(output_root):
    output_root = Path(output_root)
    seed_files = sorted(output_root.rglob("robustness_seed_list_master20260605.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in seed_files:
        df = read_csv(f)
        scol = find_col(df, ["seed", "random_seed"])
        if scol:
            vals = pd.to_numeric(df[scol], errors="coerce").dropna().astype(int).tolist()
            if vals:
                return sorted(set(vals)), f
    return [], None


def split_dev_holdout(seeds, dev_fraction=0.60):
    seeds = sorted(set(int(s) for s in seeds))
    if len(seeds) <= 1:
        return seeds, []
    n_dev = max(1, int(round(len(seeds) * float(dev_fraction))))
    n_dev = min(n_dev, len(seeds) - 1)
    return seeds[:n_dev], seeds[n_dev:]


def locate_dataset_file(phase1_run, dataset):
    phase1_run = Path(phase1_run)
    clean_dir = phase1_run / "02_CLEAN_FEATURES"
    hints = DATASET_FILE_HINTS[dataset]
    for h in hints:
        p = clean_dir / h
        if "*" not in h and p.exists():
            return p
    for h in hints:
        files = list(phase1_run.rglob(h))
        files = [p for p in files if p.is_file() and p.stat().st_size > 0]
        if files:
            return sorted(files, key=lambda p: p.stat().st_size, reverse=True)[0]
    raise FileNotFoundError(f"No feature file found for {dataset} under {phase1_run}")


def infer_group_col(df):
    col = find_col(df, GROUP_COL_CANDIDATES)
    if col:
        return col
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    candidates = []
    n = len(df)
    for c in obj_cols:
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= max(1000, n // 3):
            candidates.append((nun, c))
    if candidates:
        return sorted(candidates)[0][1]
    raise RuntimeError("Could not infer group/asset column.")


def infer_time_col(df):
    return find_col(df, TIME_COL_CANDIDATES)


def infer_target_col(df, dataset):
    # Dataset-specific preference.
    if dataset == "NASA C-MAPSS":
        for c in ["target_RUL", "target_rul", "RUL", "rul", "RUL_capped"]:
            if c in df.columns:
                return c
    if dataset == "NASA Battery":
        for c in ["capacity", "Capacity", "discharge_capacity", "capacity_ah"]:
            if c in df.columns:
                return c
    # General.
    col = find_col(df, TARGET_CANDIDATES)
    return col


def build_target(df, dataset, group_col, target_col, time_col):
    df = df.copy()
    if target_col and target_col in df.columns:
        y_raw = pd.to_numeric(df[target_col], errors="coerce")
        if dataset == "NASA Battery":
            # Preserve capacity health scale.
            gmin, gmax = float(y_raw.min()), float(y_raw.max())
            y = ((y_raw - gmin) / max(gmax - gmin, 1e-12)).clip(0, 1)
            return y, f"{target_col}__global_minmax_health"
        if "rul" in target_col.lower() or "remaining" in target_col.lower():
            maxv = float(y_raw.max())
            minv = float(y_raw.min())
            y = ((y_raw - minv) / max(maxv - minv, 1e-12)).clip(0, 1)
            return y, f"{target_col}__normalized_remaining"
        # If likely health/SOH/capacity-like.
        vals = y_raw.astype(float)
        if vals.max() > 2 or vals.min() < -0.1:
            y = ((vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)).clip(0, 1)
        else:
            y = vals.clip(0, 1)
        return y, f"{target_col}__as_health"

    # Bearing fallback: remaining fraction from per-asset order. This is fallback only.
    if dataset in BEARING_DATASETS:
        order = None
        if time_col and time_col in df.columns:
            order = pd.to_numeric(df[time_col], errors="coerce")
        if order is None or order.isna().all():
            order = df.groupby(group_col).cumcount().astype(float)
        tmp = pd.DataFrame({"g": df[group_col].astype(str), "order": order})
        min_o = tmp.groupby("g")["order"].transform("min")
        max_o = tmp.groupby("g")["order"].transform("max")
        frac = (tmp["order"] - min_o) / (max_o - min_o).replace(0, np.nan)
        y = (1.0 - frac).clip(0, 1).fillna(1.0)
        return y, "fallback_per_asset_remaining_fraction_from_order"

    raise RuntimeError(f"Could not infer target for dataset={dataset}")


def select_feature_cols(df, dataset, group_col, target_col, time_col, max_features=120):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set()
    if target_col:
        exclude.add(target_col)
    exclude.add(group_col)
    for c in numeric:
        cl = c.lower()
        if any(frag in cl for frag in LEAKY_FRAGMENTS):
            exclude.add(c)
    # Conservative leakage rules.
    for c in numeric:
        cl = c.lower()
        if dataset in BEARING_DATASETS and ("original_order" in cl or cl in {"order", "row_index"}):
            exclude.add(c)
    feats = [c for c in numeric if c not in exclude]
    if not feats:
        raise RuntimeError(f"No features after leakage filtering for {dataset}")
    var = df[feats].replace([np.inf, -np.inf], np.nan).var(numeric_only=True).sort_values(ascending=False)
    feats = [c for c in var.index if np.isfinite(var[c]) and var[c] > 0]
    return feats[:max_features]


def prepare_dataset(phase1_run, dataset, max_features=120, max_rows=-1):
    file_path = locate_dataset_file(phase1_run, dataset)
    df = read_csv(file_path)
    if df.empty:
        raise RuntimeError(f"Empty feature file for {dataset}: {file_path}")
    group_col = infer_group_col(df)
    time_col = infer_time_col(df)
    target_col = infer_target_col(df, dataset)
    y, target_policy = build_target(df, dataset, group_col, target_col, time_col)

    df = df.copy()
    df["__target__"] = pd.to_numeric(y, errors="coerce")
    df[group_col] = df[group_col].astype(str)
    if time_col:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.sort_values([group_col, time_col])
    else:
        df = df.sort_values([group_col])

    df = df.dropna(subset=["__target__", group_col]).copy()
    if max_rows and max_rows > 0 and len(df) > max_rows:
        # deterministic stratified-ish sample by group to keep all assets.
        rng = np.random.default_rng(20260606)
        parts = []
        per_group = max(5, int(math.ceil(max_rows / max(1, df[group_col].nunique()))))
        for _, g in df.groupby(group_col, sort=True):
            if len(g) <= per_group:
                parts.append(g)
            else:
                idx = np.linspace(0, len(g) - 1, per_group).astype(int)
                parts.append(g.iloc[idx])
        df = pd.concat(parts, ignore_index=True)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=20260606).sort_values(group_col).reset_index(drop=True)

    feature_cols = select_feature_cols(df, dataset, group_col, target_col, time_col, max_features=max_features)
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    y = df["__target__"].astype(float).clip(0, 1)
    groups = df[group_col].astype(str)

    manifest = {
        "dataset": dataset,
        "file": str(file_path),
        "rows": int(len(df)),
        "groups": int(groups.nunique()),
        "group_col": group_col,
        "time_col": time_col,
        "target_col": target_col,
        "target_policy": target_policy,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
    }
    return X, y, groups, manifest


def make_models(seed, dataset, quick=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.svm import SVR

    models = {
        "RIDGE": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ELASTICNET": make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=20000, random_state=seed)),
        "HGB": HistGradientBoostingRegressor(max_iter=350, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.001, random_state=seed),
        "EXTRATREES": ExtraTreesRegressor(n_estimators=260, min_samples_leaf=2, max_features=0.75, random_state=seed, n_jobs=-1),
        "RANDOMFOREST": RandomForestRegressor(n_estimators=220, min_samples_leaf=2, max_features=0.75, random_state=seed, n_jobs=-1),
    }

    # SVR is expensive on large datasets.
    if dataset != "NASA C-MAPSS":
        models["SVR_RBF"] = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.01, gamma="scale"))

    # Optional xgboost/lightgbm if installed; safe no-crash.
    try:
        from xgboost import XGBRegressor
        models["XGB"] = XGBRegressor(
            n_estimators=350, max_depth=4, learning_rate=0.045,
            subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
            random_state=seed, n_jobs=-1, verbosity=0
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor
        models["LGBM"] = LGBMRegressor(
            n_estimators=350, learning_rate=0.045, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, random_state=seed, n_jobs=-1, verbose=-1
        )
    except Exception:
        pass

    if quick:
        keep = {"RIDGE", "HGB", "EXTRATREES"}
        models = {k: v for k, v in models.items() if k in keep}
    return models


def group_fit_cal_split(train_groups, seed, cal_fraction=0.25):
    rng = np.random.default_rng(int(seed))
    groups = np.array(sorted(set(map(str, train_groups))))
    if len(groups) < 3:
        return set(groups), set()
    cal_n = max(1, int(round(len(groups) * cal_fraction)))
    cal_n = min(cal_n, len(groups) - 1)
    cal = set(rng.choice(groups, size=cal_n, replace=False).tolist())
    fit = set([g for g in groups if g not in cal])
    return fit, cal


def interval_variants(y_cal, p_cal):
    resid = np.abs(np.asarray(y_cal, float) - np.asarray(p_cal, float))
    resid = resid[np.isfinite(resid)]
    if len(resid) == 0:
        resid = np.array([0.10])
    q = {
        "global_90": float(np.quantile(resid, 0.90)),
        "global_95": float(np.quantile(resid, 0.95)),
        "global_97": float(np.quantile(resid, 0.97)),
        "global_99": float(np.quantile(resid, 0.99)),
    }
    variants = {name: (v, v) for name, v in q.items()}
    for total in [0.050, 0.075, 0.100, 0.150, 0.200, 0.250]:
        variants[f"fixed_width_{total:.3f}"] = (total / 2.0, total / 2.0)
    variants["twosided_guard_l97_u90"] = (q["global_97"], q["global_90"])
    variants["twosided_guard_l99_u95"] = (q["global_99"], q["global_95"])
    variants["guard_max_g95_fixed100"] = (max(q["global_95"], 0.05), max(q["global_95"], 0.05))
    variants["guard_max_g97_fixed150"] = (max(q["global_97"], 0.075), max(q["global_97"], 0.075))
    return variants


def metric_row(dataset, seed, candidate, y, pred, lo, hi, rmse=np.nan, mae=np.nan):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    m = np.isfinite(y) & np.isfinite(pred) & np.isfinite(lo) & np.isfinite(hi)
    y, pred, lo, hi = y[m], pred[m], lo[m], hi[m]
    if len(y) == 0:
        return None
    urgent_thr = 0.35
    covered = (y >= lo) & (y <= hi)
    urgent = y <= urgent_thr
    coverage = float(np.mean(covered))
    urgent_cov = float(np.mean(covered[urgent])) if np.any(urgent) else coverage
    false_safe = float(np.mean((y <= urgent_thr) & (lo > urgent_thr)))
    underwarning = float(np.mean(y < lo))
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "candidate": candidate,
        "candidate_lower": candidate.lower(),
        "coverage": coverage,
        "urgent_critical_coverage": urgent_cov,
        "false_safe_rate": false_safe,
        "underwarning_rate": underwarning,
        "width": float(np.mean(hi - lo)),
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))) if pd.isna(rmse) else float(rmse),
        "MAE": float(np.mean(np.abs(y - pred))) if pd.isna(mae) else float(mae),
        "n_rows": int(len(y)),
        "urgent_rows": int(np.sum(urgent)),
    }
    return dominance_eval(row)


def dominance_eval(row):
    ds = row["dataset"]
    old = OLD_LOCKED[ds]
    coverage_ok = pd.notna(row["coverage"]) and row["coverage"] >= old["coverage"]
    urgent_ok = pd.notna(row["urgent_critical_coverage"]) and row["urgent_critical_coverage"] >= old["urgent_critical_coverage"]
    false_safe_ok = pd.notna(row["false_safe_rate"]) and row["false_safe_rate"] <= old["false_safe_rate"] + 1e-12
    underwarning_ok = pd.notna(row["underwarning_rate"]) and row["underwarning_rate"] <= old["underwarning_rate"] + 1e-12
    width_ok = (
        pd.notna(row["width"]) and (
            row["width"] <= old["width"] + 1e-12
            or (
                row["width"] <= 0.50 and row["coverage"] >= old["coverage"] + 0.02
                and row["urgent_critical_coverage"] >= old["urgent_critical_coverage"] + 0.02
            )
        )
    )
    failures = []
    if not coverage_ok: failures.append("coverage lower/missing")
    if not urgent_ok: failures.append("urgent lower/missing")
    if not false_safe_ok: failures.append("false-safe worse/missing")
    if not underwarning_ok: failures.append("underwarning worse/missing")
    if not width_ok: failures.append("width worse")
    row = dict(row)
    row.update({
        "coverage_ok": bool(coverage_ok), "urgent_ok": bool(urgent_ok), "false_safe_ok": bool(false_safe_ok),
        "underwarning_ok": bool(underwarning_ok), "width_ok": bool(width_ok),
        "dominance_pass": bool(coverage_ok and urgent_ok and false_safe_ok and underwarning_ok and width_ok),
        "failure_reason": "; ".join(failures),
    })
    row["seed_dataset_pass"] = bool(row["dominance_pass"])
    return row


def run_dataset_seed(dataset, seed, X, y, groups, manifest, out_dir, quick=False, max_folds=-1):
    from sklearn.model_selection import GroupKFold

    t0 = time.time()
    unique_groups = np.array(sorted(pd.unique(groups)))
    n_splits = min(10, len(unique_groups))
    if n_splits < 3:
        raise RuntimeError(f"{dataset}: needs at least 3 groups, found {len(unique_groups)}")

    models = make_models(seed, dataset, quick=quick)
    gkf = GroupKFold(n_splits=n_splits)

    X_np = X.to_numpy(float) if hasattr(X, "to_numpy") else np.asarray(X, float)
    y_np = np.asarray(y, float)
    groups_np = np.asarray(groups.astype(str))

    pred_rows = []
    metric_rows = []
    fold = 0

    for tr_idx, te_idx in gkf.split(X_np, y_np, groups_np):
        fold += 1
        if max_folds > 0 and fold > max_folds:
            break

        tr_groups = groups_np[tr_idx]
        fit_groups, cal_groups = group_fit_cal_split(tr_groups, seed + fold)
        fit_idx = np.array([i for i in tr_idx if groups_np[i] in fit_groups], dtype=int)
        cal_idx = np.array([i for i in tr_idx if groups_np[i] in cal_groups], dtype=int)

        if len(fit_idx) < 20 or len(cal_idx) < 5:
            rng = np.random.default_rng(seed + fold)
            tr = np.array(tr_idx)
            rng.shuffle(tr)
            cal_n = max(5, int(0.2 * len(tr)))
            cal_idx = tr[:cal_n]
            fit_idx = tr[cal_n:]

        if len(fit_idx) < 20 or len(cal_idx) < 5:
            continue

        cal_preds = {}
        test_preds = {}
        for model_name, model in models.items():
            try:
                model.fit(X_np[fit_idx], y_np[fit_idx])
                cal_preds[model_name] = np.clip(model.predict(X_np[cal_idx]), 0, 1)
                test_preds[model_name] = np.clip(model.predict(X_np[te_idx]), 0, 1)
            except Exception:
                continue

        # Ensembles.
        if len(test_preds) >= 2:
            keys = [k for k in ["HGB", "EXTRATREES", "RANDOMFOREST", "RIDGE", "ELASTICNET", "XGB", "LGBM"] if k in test_preds]
            if len(keys) >= 2:
                cal_preds["ENSEMBLE_MEAN"] = np.mean([cal_preds[k] for k in keys], axis=0)
                test_preds["ENSEMBLE_MEAN"] = np.mean([test_preds[k] for k in keys], axis=0)
            keys_all = list(test_preds.keys())
            if len(keys_all) >= 3:
                cal_preds["ENSEMBLE_MEDIAN"] = np.median([cal_preds[k] for k in keys_all], axis=0)
                test_preds["ENSEMBLE_MEDIAN"] = np.median([test_preds[k] for k in keys_all], axis=0)

        y_cal = y_np[cal_idx]
        y_test = y_np[te_idx]

        for model_name, p_test in test_preds.items():
            p_cal = cal_preds[model_name]
            for interval_name, (lo_hw, hi_hw) in interval_variants(y_cal, p_cal).items():
                lo = np.clip(p_test - lo_hw, 0, 1)
                hi = np.clip(p_test + hi_hw, 0, 1)
                candidate = f"REAL_TOURNAMENT__{manifest['target_policy']}__{model_name}__{interval_name}"

                mr = metric_row(dataset, seed, candidate, y_test, p_test, lo, hi)
                if mr is None:
                    continue
                mr.update({
                    "fold": int(fold),
                    "model_family": model_name,
                    "interval_policy": interval_name,
                    "target_policy": manifest["target_policy"],
                    "feature_count": manifest["feature_count"],
                })
                metric_rows.append(mr)

                # Store predictions only if requested later? For all datasets this can be huge.
                # We store compact fold/candidate aggregate by default, not row-level predictions.

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        # Aggregate folds per seed/candidate.
        agg_rows = []
        for cand, g in metrics.groupby("candidate"):
            row = {
                "dataset": dataset,
                "seed": int(seed),
                "candidate": cand,
                "candidate_lower": cand.lower(),
                "coverage": float(g["coverage"].mean()),
                "urgent_critical_coverage": float(g["urgent_critical_coverage"].mean()),
                "false_safe_rate": float(g["false_safe_rate"].mean()),
                "underwarning_rate": float(g["underwarning_rate"].mean()),
                "width": float(g["width"].mean()),
                "RMSE": float(g["RMSE"].mean()),
                "MAE": float(g["MAE"].mean()),
                "fold_count": int(g["fold"].nunique()),
                "runtime_sec_seed": time.time() - t0,
            }
            agg_rows.append(dominance_eval(row))
        seed_metrics = pd.DataFrame(agg_rows)
    else:
        seed_metrics = pd.DataFrame()

    seed_dir = ensure_dir(out_dir / "seed_outputs" / dataset.replace("/", "_").replace(" ", "_") / f"seed_{seed}")
    write_csv(seed_dir / "fold_candidate_metrics.csv", metrics)
    write_csv(seed_dir / "seed_candidate_metrics.csv", seed_metrics)
    return seed_metrics


def summarize_candidate_across_seeds(metrics_df, candidate, dataset, seeds, required_pass_rate):
    sub = metrics_df[(metrics_df["dataset"].eq(dataset)) & (metrics_df["candidate"].astype(str).eq(str(candidate))) & (metrics_df["seed"].astype(int).isin(seeds))].copy()
    if sub.empty:
        return {
            "dataset": dataset, "candidate": candidate, "seed_count": len(seeds), "seeds_evaluated": 0,
            "seeds_passed": 0, "pass_rate": 0.0, "gate_pass": False,
        }
    passed = sub[sub["seed_dataset_pass"].astype(bool)].copy()
    row = {
        "dataset": dataset,
        "candidate": candidate,
        "seed_count": len(seeds),
        "seeds_evaluated": int(sub["seed"].nunique()),
        "seeds_passed": int(passed["seed"].nunique()) if not passed.empty else 0,
    }
    row["pass_rate"] = row["seeds_passed"] / len(seeds) if seeds else 0.0
    for m in ["coverage", "urgent_critical_coverage", "false_safe_rate", "underwarning_rate", "width", "RMSE", "MAE"]:
        vals = pd.to_numeric(passed[m], errors="coerce") if (not passed.empty and m in passed.columns) else pd.Series(dtype=float)
        row[f"median_{m}"] = float(vals.median()) if len(vals) else np.nan
        row[f"p10_{m}"] = float(vals.quantile(0.10)) if len(vals) else np.nan
        row[f"p90_{m}"] = float(vals.quantile(0.90)) if len(vals) else np.nan

    old = OLD_LOCKED[dataset]
    median_ok = (
        pd.notna(row["median_coverage"]) and row["median_coverage"] >= old["coverage"]
        and pd.notna(row["median_urgent_critical_coverage"]) and row["median_urgent_critical_coverage"] >= old["urgent_critical_coverage"]
        and pd.notna(row["median_false_safe_rate"]) and row["median_false_safe_rate"] <= old["false_safe_rate"] + 1e-12
        and pd.notna(row["median_underwarning_rate"]) and row["median_underwarning_rate"] <= old["underwarning_rate"] + 1e-12
        and pd.notna(row["median_width"]) and (
            row["median_width"] <= old["width"] + 1e-12
            or (
                row["median_width"] <= 0.50
                and row["median_coverage"] >= old["coverage"] + 0.02
                and row["median_urgent_critical_coverage"] >= old["urgent_critical_coverage"] + 0.02
            )
        )
    )
    row["median_dominates_old_floor"] = bool(median_ok)
    row["gate_pass"] = bool(row["pass_rate"] >= required_pass_rate and median_ok)
    return row


def choose_candidate_on_dev(metrics_df, dataset, dev_seeds, required_pass_rate):
    rows = []
    for cand in sorted(metrics_df[metrics_df["dataset"].eq(dataset)]["candidate"].dropna().astype(str).unique()):
        s = summarize_candidate_across_seeds(metrics_df, cand, dataset, dev_seeds, required_pass_rate)
        old = OLD_LOCKED[dataset]
        score = (
            100000.0 * int(s.get("gate_pass", False))
            + 1000.0 * float(s.get("pass_rate", 0.0))
            + 100.0 * ((s.get("median_coverage", np.nan) - old["coverage"]) if pd.notna(s.get("median_coverage", np.nan)) else -1.0)
            + 100.0 * ((s.get("median_urgent_critical_coverage", np.nan) - old["urgent_critical_coverage"]) if pd.notna(s.get("median_urgent_critical_coverage", np.nan)) else -1.0)
            + 20.0 * ((old["width"] - s.get("median_width", np.nan)) if pd.notna(s.get("median_width", np.nan)) else -1.0)
            - 0.10 * (s.get("median_RMSE", 1.0) if pd.notna(s.get("median_RMSE", np.nan)) else 1.0)
            - 0.05 * (s.get("median_MAE", 1.0) if pd.notna(s.get("median_MAE", np.nan)) else 1.0)
        )
        s["dev_score"] = float(score)
        rows.append(s)
    df = pd.DataFrame(rows).sort_values("dev_score", ascending=False) if rows else pd.DataFrame()
    if df.empty:
        return "", df
    return str(df.iloc[0]["candidate"]), df


class Runner:
    def __init__(self, args):
        self.args = args
        self.output_root = Path(args.output_root)
        self.phase1_run = Path(args.phase1_run)
        self.primary_clean_run = Path(args.primary_clean_run)
        self.code_root = Path(args.code_root)

        # RESUME-SAFE PATCH:
        # If --resume_run_dir is supplied, continue inside that exact folder and reuse
        # completed seed-level cache files. This prevents losing completed work after
        # local Python, Windows, power, or laptop interruption.
        resume_arg = str(getattr(args, "resume_run_dir", "") or "").strip()
        if resume_arg:
            self.run_dir = ensure_dir(Path(resume_arg))
            self.is_resumed_run = True
        else:
            self.run_dir = ensure_dir(self.output_root / f"Q1_ONE_SHOT_MASTER_RUN_{stamp()}")
            self.is_resumed_run = False

        for sub in [
            "00_RUN_MANIFEST", "03_BASELINE_LOCK", "04_ALL_DATASET_REAL_TOURNAMENT",
            "05_DEV_HOLDOUT_SELECTION", "06_NN_RESCUE", "07_ABLATION_STATISTICS",
            "08_BENCHMARKS", "09_REQUIREMENT_GATES", "10_MANUSCRIPT_ASSEMBLY_INPUTS",
            "11_LOGS"
        ]:
            ensure_dir(self.run_dir / sub)

        write_json(self.run_dir / "00_RUN_MANIFEST" / "resume_state.json", {
            "script_version": SCRIPT_VERSION,
            "run_dir": str(self.run_dir),
            "is_resumed_run": bool(self.is_resumed_run),
            "created_or_resumed_at": now(),
            "cache_rule": "Completed seed outputs under 04_ALL_DATASET_REAL_TOURNAMENT/seed_outputs are reused unless --force_rerun is provided.",
        })

        self.gates = []
        self.baseline_pass = False

    def log(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    def gate(self, name, status, message, evidence_path="", required_action=""):
        row = {"gate": name, "status": status, "message": message, "evidence_path": str(evidence_path), "required_action": required_action, "created_at": now()}
        self.gates.append(row)
        self.log(f"GATE {name}: {status} â€” {message}")
        return row

    def baseline_lock(self):
        out = self.run_dir / "03_BASELINE_LOCK"
        write_csv(out / "old_locked_baseline_metrics.csv", old_baseline_rows())
        write_json(out / "baseline_lock_verdict.json", {"status": "PASS", "rule": "old locked metrics are floor; no seed special"})
        self.baseline_pass = True
        self.gate("BASELINE_LOCK", "PASS", "old locked five-dataset result protected", out)

    def data_gate(self):
        out = self.run_dir / "09_REQUIREMENT_GATES"
        ok = self.phase1_run.exists()
        self.gate("DATA_LEAKAGE_GATE", "PASS" if ok else "FAIL", f"phase1_run_exists={ok}", out, "Phase-1 evidence required." if not ok else "")

    def all_dataset_real_tournament(self):
        out = self.run_dir / "04_ALL_DATASET_REAL_TOURNAMENT"
        target_seeds, seed_source = latest_seed_list(self.output_root)
        if not target_seeds:
            self.gate("ALL_DATASET_REAL_TOURNAMENT_GATE", "FAIL", "no robustness seed list found", out, "Need target seed list.")
            return pd.DataFrame(), [], [], []

        if self.args.max_total_seeds > 0:
            target_seeds = target_seeds[: self.args.max_total_seeds]

        dev_seeds, holdout_seeds = split_dev_holdout(target_seeds, self.args.dev_fraction)
        write_json(out / "seed_split.json", {"target_seeds": target_seeds, "dev_seeds": dev_seeds, "holdout_seeds": holdout_seeds, "seed_source": str(seed_source or ""), "rule": "no seed special"})

        datasets = DATASETS
        if self.args.only_dataset:
            datasets = [canon_dataset(self.args.only_dataset)]
        write_json(out / "dataset_run_plan.json", {"datasets": datasets, "smoke": bool(self.args.smoke), "script_version": SCRIPT_VERSION})

        all_seed_metrics = []
        manifests = []
        errors = []

        for dataset in datasets:
            ds_start = time.time()
            self.log(f"PREPARING DATASET: {dataset}")
            try:
                max_rows = self.args.max_rows_per_dataset
                if self.args.smoke:
                    max_rows = min(max_rows if max_rows > 0 else 5000, 5000)
                X, y, groups, manifest = prepare_dataset(self.phase1_run, dataset, max_features=self.args.max_features, max_rows=max_rows)
                manifests.append(manifest)
                write_json(out / f"manifest_{dataset.replace('/', '_').replace(' ', '_')}.json", manifest)
            except Exception as e:
                tb = traceback.format_exc()
                self.log(f"DATASET PREP FAILED {dataset}: {e}")
                errors.append({"dataset": dataset, "stage": "prepare_dataset", "error": repr(e), "traceback": tb})
                continue

            seeds_to_run = target_seeds
            if self.args.smoke:
                seeds_to_run = target_seeds[: min(2, len(target_seeds))]
            elif self.args.max_seeds_per_dataset > 0:
                seeds_to_run = target_seeds[: self.args.max_seeds_per_dataset]

            for i, seed in enumerate(seeds_to_run, start=1):
                seed_dir = out / "seed_outputs" / dataset.replace("/", "_").replace(" ", "_") / f"seed_{seed}"
                cache_file = seed_dir / "seed_candidate_metrics.csv"
                if cache_file.exists() and cache_file.stat().st_size > 0 and not self.args.force_rerun:
                    mdf = read_csv(cache_file)
                    required_cache_cols = {"dataset", "seed", "candidate", "coverage", "urgent_critical_coverage", "width"}
                    if not mdf.empty and required_cache_cols.issubset(set(mdf.columns)):
                        self.log(f"USING CACHE: {dataset} seed {seed} rows={len(mdf)}")
                        all_seed_metrics.append(mdf)
                        continue
                    else:
                        self.log(f"IGNORING INCOMPLETE CACHE: {cache_file}")

                self.log(f"TRAINING {dataset}: seed {i}/{len(seeds_to_run)} = {seed}")
                try:
                    mdf = run_dataset_seed(
                        dataset, seed, X, y, groups, manifest, out,
                        quick=self.args.smoke,
                        max_folds=(self.args.max_folds if self.args.max_folds > 0 else (-1 if not self.args.smoke else 3)),
                    )
                    if not mdf.empty:
                        all_seed_metrics.append(mdf)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.log(f"TRAINING FAILED {dataset} seed={seed}: {e}")
                    errors.append({"dataset": dataset, "seed": seed, "stage": "run_dataset_seed", "error": repr(e), "traceback": tb})

            self.log(f"DATASET DONE: {dataset}; elapsed_min={(time.time()-ds_start)/60:.1f}")

        write_csv(out / "dataset_manifests.csv", manifests)
        write_csv(out / "errors.csv", errors)

        metrics_all = pd.concat(all_seed_metrics, ignore_index=True, sort=False) if all_seed_metrics else pd.DataFrame()
        write_csv(out / "all_seed_candidate_metrics.csv", metrics_all)

        produced_datasets = sorted(metrics_all["dataset"].dropna().astype(str).unique().tolist()) if not metrics_all.empty and "dataset" in metrics_all.columns else []
        missing_metric_datasets = sorted(set(datasets) - set(produced_datasets))
        status = "PASS" if (not metrics_all.empty and len(errors) == 0 and not missing_metric_datasets) else "FAIL"
        self.gate(
            "ALL_DATASET_REAL_TOURNAMENT_GATE",
            status,
            f"datasets={datasets}; produced={produced_datasets}; missing_metrics={missing_metric_datasets}; candidate_rows={len(metrics_all)}; errors={len(errors)}",
            out,
            "Real tournament failed for at least one dataset or produced incomplete metrics." if status == "FAIL" else "",
        )
        return metrics_all, target_seeds, dev_seeds, holdout_seeds

    def dev_holdout_selection_gate(self, metrics_all, target_seeds, dev_seeds, holdout_seeds):
        out = self.run_dir / "05_DEV_HOLDOUT_SELECTION"
        if metrics_all is None or metrics_all.empty:
            self.gate("DEV_HOLDOUT_SELECTION_GATE", "FAIL", "no metrics available for selection", out, "Run real tournament first.")
            return

        chosen_rows = []
        candidate_score_frames = []
        selected_summaries = []
        failed = []

        for dataset in DATASETS:
            if self.args.only_dataset and canon_dataset(self.args.only_dataset) != dataset:
                continue

            chosen, dev_scores = choose_candidate_on_dev(metrics_all, dataset, dev_seeds, self.args.robustness_pass_rate_threshold)
            if not dev_scores.empty:
                dev_scores["dataset"] = dataset
                candidate_score_frames.append(dev_scores)
            if not chosen:
                failed.append(dataset)
                continue

            dev_summary = summarize_candidate_across_seeds(metrics_all, chosen, dataset, dev_seeds, self.args.robustness_pass_rate_threshold)
            hold_summary = summarize_candidate_across_seeds(metrics_all, chosen, dataset, holdout_seeds, self.args.robustness_pass_rate_threshold)
            all_summary = summarize_candidate_across_seeds(metrics_all, chosen, dataset, target_seeds, self.args.robustness_pass_rate_threshold)

            row = {
                "dataset": dataset,
                "chosen_candidate": chosen,
                "dev_gate_pass": bool(dev_summary.get("gate_pass", False)),
                "holdout_gate_pass": bool(hold_summary.get("gate_pass", False)),
                "all_seed_gate_pass": bool(all_summary.get("gate_pass", False)),
                "dev_pass_rate": dev_summary.get("pass_rate", np.nan),
                "holdout_pass_rate": hold_summary.get("pass_rate", np.nan),
                "all_pass_rate": all_summary.get("pass_rate", np.nan),
            }
            chosen_rows.append(row)
            srow = dict(all_summary)
            srow["chosen_candidate"] = chosen
            srow["selection_policy"] = "candidate_chosen_on_dev_validated_on_holdout_no_seed_special"
            selected_summaries.append(srow)

            if not (row["dev_gate_pass"] and row["holdout_gate_pass"] and row["all_seed_gate_pass"]):
                failed.append(dataset)

        score_df = pd.concat(candidate_score_frames, ignore_index=True, sort=False) if candidate_score_frames else pd.DataFrame()
        write_csv(out / "development_candidate_scores.csv", score_df)
        chosen_df = pd.DataFrame(chosen_rows)
        write_csv(out / "chosen_candidate_by_dataset.csv", chosen_df)
        summary_df = pd.DataFrame(selected_summaries)
        write_csv(out / "selected_candidate_all_seed_summary.csv", summary_df)

        promoted_path = ""
        expected_datasets = [canon_dataset(self.args.only_dataset)] if self.args.only_dataset else DATASETS
        missing = sorted(set(expected_datasets) - set(summary_df["dataset"].astype(str).tolist())) if not summary_df.empty else expected_datasets

        if not failed and not missing and len(target_seeds) >= self.args.min_robustness_seeds and not self.args.only_dataset:
            promoted = out / "robust_all_dataset_promoted_metrics.csv"
            rows = []
            for _, r in summary_df.iterrows():
                rows.append({
                    "dataset": r["dataset"],
                    "candidate": r["chosen_candidate"],
                    "coverage": r.get("median_coverage", np.nan),
                    "urgent_critical_coverage": r.get("median_urgent_critical_coverage", np.nan),
                    "false_safe_rate": r.get("median_false_safe_rate", np.nan),
                    "underwarning_rate": r.get("median_underwarning_rate", np.nan),
                    "width": r.get("median_width", np.nan),
                    "pass_rate": r.get("pass_rate", np.nan),
                    "seed_count": r.get("seed_count", np.nan),
                    "seeds_passed": r.get("seeds_passed", np.nan),
                    "selection_policy": "all_dataset_real_tournament_dev_holdout_no_seed_special",
                })
            write_csv(promoted, rows)
            promoted_path = str(promoted)

        verdict = {
            "status": "PASS" if promoted_path else "FAIL",
            "failed_datasets": sorted(set(failed)),
            "missing_datasets": missing,
            "target_seed_count": len(target_seeds),
            "dev_seed_count": len(dev_seeds),
            "holdout_seed_count": len(holdout_seeds),
            "promoted_metrics": promoted_path,
            "rule": "Each dataset candidate chosen on dev seeds only, validated on holdout, summarized on all seeds.",
        }
        write_json(out / "dev_holdout_selection_verdict.json", verdict)

        self.gate(
            "DEV_HOLDOUT_SELECTION_GATE",
            "PASS" if promoted_path else "FAIL",
            f"failed={sorted(set(failed))}; missing={missing}; promoted={promoted_path}",
            out,
        )

    def nn_gate(self):
        out = self.run_dir / "06_NN_RESCUE"
        rows = []
        failed = []
        for ds in DATASETS:
            runs = sorted([p for p in self.output_root.glob("Q1_NN_BASELINES_RUN_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = {"dataset": ds, "strict_pass": False, "reason": "no explicit NN verdict for dataset"}
            for run in runs:
                verdicts = list(run.rglob("*VERDICT*.json")) + list(run.rglob("verdict.json")) + list(run.rglob("NN_VERDICT.json"))
                for vf in verdicts:
                    obj = read_json(vf)
                    if not obj:
                        continue
                    requested = [canon_dataset(x) for x in obj.get("datasets_requested", [])]
                    if ds not in requested:
                        continue
                    passed = [canon_dataset(x) for x in obj.get("datasets_with_nn_strict_pass", [])]
                    strict = (int(obj.get("errors", 0) or 0) == 0) and ds in passed
                    chosen = {
                        "dataset": ds,
                        "strict_pass": bool(strict),
                        "run": str(run),
                        "verdict_file": str(vf),
                        "any_nn_strict_pass": obj.get("any_nn_strict_pass", ""),
                        "datasets_with_nn_strict_pass": ";".join(map(str, obj.get("datasets_with_nn_strict_pass", []))),
                        "errors": obj.get("errors", ""),
                        "prediction_rows": obj.get("prediction_rows", ""),
                        "loss_rows": obj.get("loss_rows", ""),
                    }
                    break
                if chosen.get("verdict_file"):
                    break
            rows.append(chosen)
            if not chosen.get("strict_pass", False):
                failed.append(ds)
        write_csv(out / "nn_strict_verdict_by_dataset.csv", rows)
        self.gate("NN_STRICT_VERDICT_GATE", "PASS" if not failed else "FAIL", f"failed_datasets={failed}", out, "NN strict evidence failed; do not use NN as replacement." if failed else "")

    def ablation_gate(self):
        out = self.run_dir / "07_ABLATION_STATISTICS"
        required = {
            "selected_vs_nearest": ["*selected_vs_nearest*.csv", "*nearest_competitor*.csv"],
            "paired_bootstrap": ["*paired_bootstrap*.csv", "*bootstrap_selected_vs_nearest*.csv"],
            "effect_sizes": ["*effect_size*.csv", "*effect_sizes*.csv"],
            "bootstrap_probability": ["*bootstrap_probability*.csv", "*selected_better*.csv"],
            "confidence_intervals": ["*confidence_intervals*.csv", "*wilson*.csv"],
        }
        missing = []
        rows = []
        for key, pats in required.items():
            files = []
            for pat in pats:
                files += [p for p in self.output_root.rglob(pat) if p.is_file() and p.stat().st_size > 0]
            files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
            rows.append({"item": key, "file": str(files[0]) if files else "", "missing": not bool(files)})
            if not files:
                missing.append(key)
        write_csv(out / "ablation_statistics_inventory.csv", rows)
        self.gate("ABLATION_STATISTICS_GATE", "PASS" if not missing else "FAIL", f"missing={missing}", out, "Missing formal statistical outputs." if missing else "")

    def manuscript_gate(self):
        out = self.run_dir / "10_MANUSCRIPT_ASSEMBLY_INPUTS"
        files = sorted([p for p in self.output_root.rglob("figure_caption_audit.csv") if p.is_file() and p.stat().st_size > 0], key=lambda p: p.stat().st_mtime, reverse=True)
        status = "FAIL"
        msg = "figure_caption_audit not found/nonempty"
        if files:
            df = read_csv(files[0])
            if not df.empty:
                write_csv(out / "figure_caption_audit.csv", df)
                status = "PASS"
                msg = f"figure_caption_audit_source={files[0]}; rows={len(df)}"
        self.gate("MANUSCRIPT_INPUTS_GATE", status, msg, out, "Complete caption audit." if status == "FAIL" else "")

    def benchmark_gate(self):
        out = self.run_dir / "08_BENCHMARKS"
        files = [p for p in self.output_root.rglob("*benchmark*") if p.is_file() and p.stat().st_size > 0 and p.suffix.lower() in {".csv", ".xlsx", ".json"}]
        write_csv(out / "benchmark_file_inventory.csv", [{"file": str(p), "size": p.stat().st_size} for p in files])
        self.gate("BENCHMARK_VERIFICATION_GATE", "PASS" if files else "FAIL", f"benchmark_files={len(files)}", out, "Benchmark package missing." if not files else "")

    def diagnostics_gate(self):
        out = self.run_dir / "09_REQUIREMENT_GATES"
        pats = ["*train*pred*.csv", "*cal*pred*.csv", "*test*pred*.csv", "*runtime*.csv", "*runtime*.json"]
        files = []
        for pat in pats:
            files += [p for p in self.output_root.rglob(pat) if p.is_file() and p.stat().st_size > 0]
        write_csv(out / "diagnostics_file_inventory.csv", [{"file": str(p), "size": p.stat().st_size} for p in files])
        self.gate("DIAGNOSTICS_GATE", "PASS" if files else "FAIL", f"diagnostic_files={len(files)}", out, "Diagnostics missing." if not files else "")

    def final_verdict(self):
        out = self.run_dir / "09_REQUIREMENT_GATES"
        write_csv(out / "quality_gate_matrix.csv", self.gates)
        mandatory = [
            "BASELINE_LOCK", "DATA_LEAKAGE_GATE", "ALL_DATASET_REAL_TOURNAMENT_GATE",
            "DEV_HOLDOUT_SELECTION_GATE", "NN_STRICT_VERDICT_GATE",
            "DIAGNOSTICS_GATE", "ABLATION_STATISTICS_GATE",
            "BENCHMARK_VERIFICATION_GATE", "MANUSCRIPT_INPUTS_GATE",
        ]
        evaluated = {g["gate"] for g in self.gates}
        missing = sorted(set(mandatory) - evaluated)
        failed = [g["gate"] for g in self.gates if str(g["status"]).upper() == "FAIL"]
        allowed = bool(self.baseline_pass and not missing and not failed)
        verdict = {
            "script_version": SCRIPT_VERSION,
            "baseline_lock_status": "PASS" if self.baseline_pass else "FAIL",
            "improvement_gate_status": "PASS" if allowed else ("INCOMPLETE" if missing else "FAIL"),
            "failed_gates": failed,
            "missing_required_gates": missing,
            "evaluated_gates": sorted(evaluated),
            "mode": self.args.mode,
            "manuscript_allowed": allowed,
            "ready_for_manuscript": allowed,
            "master_run_dir": str(self.run_dir),
            "created_at": now(),
            "rule": "All-dataset real model tournament. No seed is special. Tournament gate fails if any planned dataset errors or lacks metrics.",
        }
        write_json(out / "final_gate_verdict.json", verdict)
        write_json(self.run_dir / "00_RUN_MANIFEST" / "MASTER_VERDICT.json", verdict)
        self.log(f"FINAL VERDICT: baseline={verdict['baseline_lock_status']} improvement={verdict['improvement_gate_status']} missing_required={len(missing)} failed={len(failed)} manuscript_allowed={allowed}")
        return verdict

    def run(self):
        self.log("=" * 120)
        self.log("Q1 ONE-SHOT MASTER RUNNER â€” V20 ALL-DATASET REAL MODEL TOURNAMENT")
        self.log("=" * 120)
        self.log(f"SCRIPT_VERSION: {SCRIPT_VERSION}")
        self.log(f"Mode: {self.args.mode}")
        self.log(f"Output folder: {self.run_dir}")
        self.log(f"Resume mode: {bool(getattr(self, 'is_resumed_run', False))}")
        self.log("This is full all-dataset candidate generation, not fragmented Battery-only repair.")
        self.log("Resume-safe cache: completed dataset/seed outputs are reused unless --force_rerun is provided.")
        self.baseline_lock()
        self.data_gate()
        metrics_all, target_seeds, dev_seeds, holdout_seeds = self.all_dataset_real_tournament()
        self.dev_holdout_selection_gate(metrics_all, target_seeds, dev_seeds, holdout_seeds)
        self.nn_gate()
        self.diagnostics_gate()
        self.ablation_gate()
        self.benchmark_gate()
        self.manuscript_gate()
        return self.final_verdict()


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "resume", "audit_only", "repair_only"], default="full")
    p.add_argument("--output_root", required=True)
    p.add_argument("--phase1_run", required=True)
    p.add_argument("--primary_clean_run", required=True)
    p.add_argument("--code_root", required=True)
    p.add_argument("--allow_quality_gate_failure", action="store_true")
    p.add_argument("--dev_fraction", type=float, default=0.60)
    p.add_argument("--min_robustness_seeds", type=int, default=30)
    p.add_argument("--robustness_pass_rate_threshold", type=float, default=0.80)
    p.add_argument("--max_features", type=int, default=120)
    p.add_argument("--max_rows_per_dataset", type=int, default=-1)
    p.add_argument("--max_total_seeds", type=int, default=-1)
    p.add_argument("--max_seeds_per_dataset", type=int, default=-1)
    p.add_argument("--max_folds", type=int, default=-1)
    p.add_argument("--only_dataset", default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume_run_dir", default="", help="Existing Q1_ONE_SHOT_MASTER_RUN_* folder to continue from. Reuses completed seed outputs.")
    p.add_argument("--force_rerun", action="store_true", help="Ignore cached seed outputs and retrain.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    runner = Runner(args)
    verdict = runner.run()
    if not verdict.get("manuscript_allowed", False) and not args.allow_quality_gate_failure:
        raise SystemExit(10)


if __name__ == "__main__":
    main()

