
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

# Public release script: 03_generate_candidate_diagnostics.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

#!/usr/bin/env python
# -*- coding: utf-8 -*-




import argparse
import json
import math
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_VERSION = "Q1_FINAL_CANDIDATE_DIAGNOSTICS_NEW_RESULTS_ONLY__VERIFIED_NO_FAKE_TRAIN_CURVES"

DATASETS = ["NASA C-MAPSS", "NASA Battery", "PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]

PHASE1_FEATURES = {
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


def banner(text: str) -> None:
    print("\n" + "=" * 120)
    print(text)
    print("=" * 120, flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, p: Path) -> None:
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding="utf-8-sig")


def save_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def slug(dataset: str) -> str:
    return norm(dataset).replace("nasa_", "")


def find_asset_col(df: pd.DataFrame, dataset: str) -> Optional[str]:
    candidates = {
        "NASA C-MAPSS": ["asset_id", "engine_group", "unit_group", "engine_id", "unit_id", "unit_number", "engine"],
        "NASA Battery": ["asset_id", "battery_id", "cell_id", "battery"],
        "PRONOSTIA/FEMTO": ["asset_id", "bearing_id", "bearing"],
        "XJTU-SY": ["asset_id", "bearing_id", "bearing"],
        "IMS": ["asset_id", "bearing_id", "bearing"],
    }[dataset]
    lookup = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lookup:
            return lookup[norm(c)]
    return None


def read_final_metrics(clean_run: Path) -> pd.DataFrame:
    p = clean_run / "02_TABLES" / "SINGLE_FILE_SELECTED_CANDIDATE_METRICS.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def find_prediction_files(clean_run: Path) -> List[Path]:
    patterns = [
        "*selected*prediction*.csv",
        "*SELECTED*PREDICTION*.csv",
        "*prediction*.csv",
        "*PREDICTION*.csv",
    ]
    files = []
    for pat in patterns:
        files.extend([p for p in clean_run.rglob(pat) if p.is_file() and p.stat().st_size > 1000])
    unique = []
    seen = set()
    for p in sorted(files):
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            unique.append(p)
    return unique


def infer_dataset_from_path(p: Path) -> Optional[str]:
    s = str(p).lower()
    for ds in DATASETS:
        ds_s = slug(ds)
        if ds_s in s or norm(ds).replace("_", "") in s.replace("_", ""):
            return ds
    return None


def load_prediction_files(clean_run: Path) -> pd.DataFrame:
    frames = []
    for p in find_prediction_files(clean_run):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        lower = {c.lower(): c for c in df.columns}
        if "y_true" not in lower or "y_pred" not in lower:
            continue
        # standardize key columns
        if lower["y_true"] != "y_true":
            df["y_true"] = df[lower["y_true"]]
        if lower["y_pred"] != "y_pred":
            df["y_pred"] = df[lower["y_pred"]]
        for old, new in [("lower", "lower"), ("upper", "upper"), ("lo", "lower"), ("hi", "upper")]:
            if old in lower and new not in df.columns:
                df[new] = df[lower[old]]
        if "lower" not in df.columns:
            df["lower"] = df["y_pred"]
        if "upper" not in df.columns:
            df["upper"] = df["y_pred"]
        if "dataset" not in df.columns:
            df["dataset"] = infer_dataset_from_path(p) or "UNKNOWN"
        if "source_prediction_file" not in df.columns:
            df["source_prediction_file"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    pred = pd.concat(frames, ignore_index=True)
    # Keep only known datasets if possible.
    if "dataset" in pred.columns:
        pred = pred[pred["dataset"].isin(DATASETS)].copy()
    return pred.reset_index(drop=True)


def categorise(y: np.ndarray) -> np.ndarray:
    out = np.full(len(y), "safe", dtype=object)
    out[y <= 0.50] = "monitor"
    out[y <= 0.25] = "urgent"
    out[y <= 0.10] = "critical"
    return out


def cat_rank(cats: np.ndarray) -> np.ndarray:
    order = {"critical": 0, "urgent": 1, "monitor": 2, "safe": 3}
    return np.asarray([order.get(str(c), 3) for c in cats])


def metric_dict(d: pd.DataFrame) -> Dict[str, Any]:
    y = pd.to_numeric(d["y_true"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(d["y_pred"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(d["lower"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(d["upper"], errors="coerce").to_numpy(float)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    m = np.isfinite(y) & np.isfinite(p) & np.isfinite(lo) & np.isfinite(hi)
    y, p, lo, hi = y[m], p[m], lo[m], hi[m]
    if len(y) == 0:
        return {}
    covered = (y >= lo) & (y <= hi)
    true_cat = categorise(y)
    point_cat = categorise(p)
    interval_cat = categorise(lo)
    tr, pr, ir = cat_rank(true_cat), cat_rank(point_cat), cat_rank(interval_cat)
    uc = np.isin(true_cat, ["urgent", "critical"])
    crit = true_cat == "critical"

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - p) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "n_rows": int(len(y)),
        "n_urgent_critical": int(uc.sum()),
        "n_critical": int(crit.sum()),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "R2": float(r2),
        "empirical_coverage": float(np.mean(covered)),
        "urgent_critical_coverage": float(np.mean(covered[uc])) if uc.any() else np.nan,
        "critical_coverage": float(np.mean(covered[crit])) if crit.any() else np.nan,
        "interval_false_safe_rate": float(np.mean((interval_cat == "safe") & uc)),
        "interval_underwarning_rate": float(np.mean(ir > tr)),
        "interval_overwarning_rate": float(np.mean(ir < tr)),
        "mean_interval_width": float(np.mean(hi - lo)),
        "median_interval_width": float(np.median(hi - lo)),
        "mean_abs_error": float(np.mean(np.abs(y - p))),
        "mean_signed_error": float(np.mean(y - p)),
    }


def gate_status(row: Dict[str, Any], dataset: str) -> tuple[int, str]:
    reasons = []
    cov = row.get("empirical_coverage", np.nan)
    urg = row.get("urgent_critical_coverage", np.nan)
    fs = row.get("interval_false_safe_rate", np.nan)
    under = row.get("interval_underwarning_rate", np.nan)
    width = row.get("mean_interval_width", np.nan)
    if not np.isfinite(cov) or cov < GATES["coverage_min"]:
        reasons.append(f"coverage {cov:.3f}<0.88")
    if np.isfinite(urg) and urg < GATES["urgent_min"]:
        reasons.append(f"urgent {urg:.3f}<0.80")
    if np.isfinite(fs) and fs > GATES["max_false_safe"]:
        reasons.append(f"false_safe {fs:.3f}>0.02")
    if np.isfinite(under) and under > GATES["max_underwarning"]:
        reasons.append(f"underwarning {under:.3f}>0.20")
    if not np.isfinite(width) or width > GATES["width_max"][dataset]:
        reasons.append(f"width {width:.3f}>{GATES['width_max'][dataset]}")
    return (0, "; ".join(reasons)) if reasons else (1, "passes gate")


def split_manifest_from_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for dataset, d in pred.groupby("dataset"):
        fold_col = "fold" if "fold" in d.columns else None
        asset_col = "asset_id" if "asset_id" in d.columns else "group" if "group" in d.columns else None
        if fold_col is None:
            rows.append({"dataset": dataset, "fold": "UNKNOWN", "test_rows": len(d), "test_assets": np.nan})
            continue
        for fold, f in d.groupby(fold_col):
            rows.append({
                "dataset": dataset,
                "fold": fold,
                "split_role_available": "test_only_from_clean_reproduction_predictions",
                "test_rows": len(f),
                "test_assets": f[asset_col].nunique() if asset_col and asset_col in f.columns else np.nan,
                "test_asset_list": "; ".join(sorted(f[asset_col].astype(str).unique())[:50]) if asset_col and asset_col in f.columns else "",
            })
    return pd.DataFrame(rows)


def fold_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    fold_col = "fold" if "fold" in pred.columns else None
    if fold_col is None:
        return pd.DataFrame()
    for (dataset, fold), d in pred.groupby(["dataset", fold_col]):
        m = metric_dict(d)
        passed, reason = gate_status(m, dataset)
        rows.append({
            "dataset": dataset,
            "fold": fold,
            **m,
            "gate_passed": passed,
            "gate_reason": reason,
        })
    return pd.DataFrame(rows)


def risk_zone_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for dataset, d in pred.groupby("dataset"):
        y = pd.to_numeric(d["y_true"], errors="coerce").to_numpy(float)
        zones = {
            "critical": y <= 0.10,
            "urgent": (y > 0.10) & (y <= 0.25),
            "monitor": (y > 0.25) & (y <= 0.50),
            "safe": y > 0.50,
        }
        for zone, mask in zones.items():
            sub = d[mask].copy()
            rows.append({"dataset": dataset, "risk_zone": zone, **metric_dict(sub)} if len(sub) else {"dataset": dataset, "risk_zone": zone, "n_rows": 0})
    return pd.DataFrame(rows)


def error_width_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for dataset, d in pred.groupby("dataset"):
        dd = d.copy()
        dd["abs_error"] = (pd.to_numeric(dd["y_true"], errors="coerce") - pd.to_numeric(dd["y_pred"], errors="coerce")).abs()
        dd["interval_width"] = pd.to_numeric(dd["upper"], errors="coerce") - pd.to_numeric(dd["lower"], errors="coerce")
        dd = dd[np.isfinite(dd["abs_error"]) & np.isfinite(dd["interval_width"])]
        if dd.empty:
            continue
        corr = dd["abs_error"].corr(dd["interval_width"]) if dd["interval_width"].nunique() > 1 else np.nan
        rows.append({
            "dataset": dataset,
            "n_rows": len(dd),
            "mean_abs_error": float(dd["abs_error"].mean()),
            "median_abs_error": float(dd["abs_error"].median()),
            "mean_interval_width": float(dd["interval_width"].mean()),
            "error_width_correlation": float(corr) if pd.notna(corr) else np.nan,
        })
    return pd.DataFrame(rows)


def runtime_by_stage(clean_run: Path, final_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    runtime_cols = [c for c in final_metrics.columns if "runtime" in c.lower() or "time" in c.lower()]
    for _, r in final_metrics.iterrows():
        rows.append({
            "dataset": r.get("dataset", ""),
            "selected_candidate": r.get("selected_candidate", ""),
            "rows": r.get("n_rows", np.nan),
            "folds": r.get("folds", np.nan),
            "total_runtime_sec": r.get("runtime_sec", np.nan),
            "training_runtime_sec": np.nan,
            "calibration_runtime_sec": np.nan,
            "prediction_runtime_sec": np.nan,
            "stage_breakdown_status": "TOTAL_ONLY_AVAILABLE_FROM_CLEAN_RUN",
            "note": "Stage-specific times require a retrain script instrumented by stage; total runtime is available from the clean reproduction.",
        })
    return pd.DataFrame(rows)


def phase1_audit(phase1_run: Path) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        p = phase1_run / "02_CLEAN_FEATURES" / PHASE1_FEATURES[ds]
        if not p.exists():
            rows.append({"dataset": ds, "status": "MISSING", "feature_file": str(p)})
            continue
        df = pd.read_csv(p, low_memory=False)
        asset_col = find_asset_col(df, ds)
        rows.append({
            "dataset": ds,
            "status": "PASS" if len(df) == EXPECTED_COUNTS[ds]["rows"] and asset_col and df[asset_col].nunique() == EXPECTED_COUNTS[ds]["assets"] else "CHECK",
            "rows": len(df),
            "expected_rows": EXPECTED_COUNTS[ds]["rows"],
            "asset_col": asset_col,
            "assets": int(df[asset_col].nunique()) if asset_col else np.nan,
            "expected_assets": EXPECTED_COUNTS[ds]["assets"],
            "columns": len(df.columns),
            "feature_file": str(p),
        })
    return pd.DataFrame(rows)


def safe_sample(d: pd.DataFrame, n=15000) -> pd.DataFrame:
    if len(d) > n:
        return d.sample(n, random_state=42)
    return d


def save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_true_pred(pred: pd.DataFrame, fig_dir: Path) -> None:
    for dataset, d in pred.groupby("dataset"):
        dd = safe_sample(d)
        plt.figure(figsize=(5.5, 5.5))
        plt.scatter(dd["y_true"], dd["y_pred"], s=5, alpha=0.35)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("True normalized RUL/SOH")
        plt.ylabel("Predicted normalized RUL/SOH")
        plt.title(f"{dataset}: held-out true vs predicted")
        save_fig(fig_dir / f"{slug(dataset)}__test_true_vs_pred.png")


def plot_residuals(pred: pd.DataFrame, fig_dir: Path) -> None:
    for dataset, d in pred.groupby("dataset"):
        residual = pd.to_numeric(d["y_true"], errors="coerce") - pd.to_numeric(d["y_pred"], errors="coerce")
        abs_error = residual.abs()
        plt.figure(figsize=(6, 4))
        plt.hist(residual.dropna(), bins=50)
        plt.xlabel("Residual (true - predicted)")
        plt.ylabel("Count")
        plt.title(f"{dataset}: held-out residual distribution")
        save_fig(fig_dir / f"{slug(dataset)}__test_residual_distribution.png")

        plt.figure(figsize=(6, 4))
        plt.hist(abs_error.dropna(), bins=50)
        plt.xlabel("Absolute error")
        plt.ylabel("Count")
        plt.title(f"{dataset}: held-out absolute-error distribution")
        save_fig(fig_dir / f"{slug(dataset)}__test_absolute_error_distribution.png")


def plot_error_width(pred: pd.DataFrame, fig_dir: Path) -> None:
    for dataset, d in pred.groupby("dataset"):
        dd = d.copy()
        dd["abs_error"] = (pd.to_numeric(dd["y_true"], errors="coerce") - pd.to_numeric(dd["y_pred"], errors="coerce")).abs()
        dd["interval_width"] = pd.to_numeric(dd["upper"], errors="coerce") - pd.to_numeric(dd["lower"], errors="coerce")
        dd = safe_sample(dd[np.isfinite(dd["abs_error"]) & np.isfinite(dd["interval_width"])])
        if dd.empty:
            continue
        plt.figure(figsize=(6, 4))
        plt.scatter(dd["interval_width"], dd["abs_error"], s=5, alpha=0.35)
        plt.xlabel("Interval width")
        plt.ylabel("Absolute error")
        plt.title(f"{dataset}: held-out error vs interval width")
        save_fig(fig_dir / f"{slug(dataset)}__test_error_vs_interval_width.png")


def plot_interval_trajectories(pred: pd.DataFrame, fig_dir: Path) -> None:
    for dataset, d in pred.groupby("dataset"):
        asset_col = "asset_id" if "asset_id" in d.columns else "group" if "group" in d.columns else None
        if asset_col is None:
            continue
        asset = d[asset_col].astype(str).value_counts().index[0]
        s = d[d[asset_col].astype(str).eq(asset)].copy()
        order_col = "time_like" if "time_like" in s.columns else "row_index" if "row_index" in s.columns else None
        if order_col:
            s = s.sort_values(order_col)
        if len(s) > 900:
            s = s.iloc[np.linspace(0, len(s) - 1, 900).astype(int)]
        x = np.arange(len(s))
        plt.figure(figsize=(10, 4.5))
        plt.plot(x, s["y_true"], label="True")
        plt.plot(x, s["y_pred"], label="Predicted")
        plt.fill_between(x, s["lower"], s["upper"], alpha=0.25, label="Interval")
        plt.axhline(0.25, linestyle="--", label="Urgent threshold")
        plt.axhline(0.10, linestyle=":", label="Critical threshold")
        plt.xlabel("Ordered held-out observation")
        plt.ylabel("Normalized RUL/SOH")
        plt.title(f"{dataset}: interval trajectory example ({asset})")
        plt.legend(fontsize=8)
        save_fig(fig_dir / f"{slug(dataset)}__test_interval_trajectory.png")


def plot_coverage_heatmaps(pred: pd.DataFrame, fig_dir: Path) -> None:
    if "fold" not in pred.columns:
        return
    for dataset, d in pred.groupby("dataset"):
        asset_col = "asset_id" if "asset_id" in d.columns else "group" if "group" in d.columns else None
        if asset_col is None or "covered" not in d.columns:
            # reconstruct covered
            dd = d.copy()
            dd["covered"] = (
                (pd.to_numeric(dd["y_true"], errors="coerce") >= pd.to_numeric(dd["lower"], errors="coerce")) &
                (pd.to_numeric(dd["y_true"], errors="coerce") <= pd.to_numeric(dd["upper"], errors="coerce"))
            ).astype(float)
            d = dd
        piv = d.groupby([asset_col, "fold"])["covered"].mean().unstack(fill_value=np.nan)
        if piv.empty:
            continue
        plt.figure(figsize=(max(6, 0.7 * len(piv.columns)), max(4, 0.3 * len(piv.index))))
        plt.imshow(piv.values.astype(float), aspect="auto", vmin=0, vmax=1)
        plt.colorbar(label="Coverage")
        plt.yticks(np.arange(len(piv.index)), piv.index)
        plt.xticks(np.arange(len(piv.columns)), piv.columns, rotation=45, ha="right")
        plt.xlabel("Fold")
        plt.ylabel("Asset/group")
        plt.title(f"{dataset}: held-out coverage by asset/fold")
        save_fig(fig_dir / f"{slug(dataset)}__coverage_heatmap.png")


def caption_table(fig_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(fig_dir.glob("*.png")):
        name = p.name
        if "true_vs_pred" in name:
            cap = "Held-out true-versus-predicted normalized RUL/SOH. The dashed diagonal marks perfect prediction."
        elif "residual" in name:
            cap = "Held-out residual distribution; residual is true minus predicted normalized RUL/SOH."
        elif "absolute_error" in name:
            cap = "Held-out absolute-error distribution."
        elif "error_vs_interval_width" in name:
            cap = "Held-out absolute error versus calibrated interval width; wider intervals should reflect higher uncertainty."
        elif "interval_trajectory" in name:
            cap = "Example held-out degradation trajectory with true value, prediction, calibrated interval, and urgent/critical thresholds."
        elif "coverage_heatmap" in name:
            cap = "Coverage heatmap by held-out asset/group and fold; values near one indicate interval containment."
        else:
            cap = "Diagnostic figure generated from clean reproduced held-out predictions."
        rows.append({"figure_file": str(p), "caption_ready_text": cap})
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_run", required=True)
    p.add_argument("--clean_run", required=True)
    p.add_argument("--output_root", default="")
    return p.parse_args()


def main():
    args = parse_args()
    phase1_run = Path(args.phase1_run)
    clean_run = Path(args.clean_run)
    if not phase1_run.exists():
        raise FileNotFoundError(phase1_run)
    if not clean_run.exists():
        raise FileNotFoundError(clean_run)

    out_root = Path(args.output_root) if args.output_root else clean_run
    out_dir = out_root / f"Q1_FINAL_CANDIDATE_DIAGNOSTICS_NEW_RESULTS_ONLY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dirs = {
        "manifest": ensure_dir(out_dir / "00_RUN_MANIFEST"),
        "split": ensure_dir(out_dir / "01_SPLIT_AND_FOLD_AUDIT"),
        "pred": ensure_dir(out_dir / "02_TRAIN_CAL_TEST_PREDICTIONS"),
        "metrics": ensure_dir(out_dir / "03_TRAIN_TEST_METRICS"),
        "figs": ensure_dir(out_dir / "04_FIGURES_TRUE_PRED_RESIDUALS"),
        "interval": ensure_dir(out_dir / "05_INTERVAL_AND_WIDTH_DIAGNOSTICS"),
        "heatmaps": ensure_dir(out_dir / "06_COVERAGE_HEATMAPS"),
        "runtime": ensure_dir(out_dir / "07_RUNTIME_BY_STAGE"),
        "summary": ensure_dir(out_dir / "08_REVIEWER_SUMMARY"),
    }

    banner("Q1 FINAL CANDIDATE DIAGNOSTICS â€” NEW RESULTS ONLY")
    print(f"SCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"Phase1 run: {phase1_run}")
    print(f"Clean run : {clean_run}")
    print(f"Output    : {out_dir}")
    print("Verified mode: no fake train/cal predictions.")

    save_json(dirs["manifest"] / "run_config.json", {
        "script_version": SCRIPT_VERSION,
        "phase1_run": str(phase1_run),
        "clean_run": str(clean_run),
        "output_dir": str(out_dir),
        "old_outputs_used": False,
        "obsolete_results_used": False,
        "training_performed": False,
        "note": "This diagnostics run post-processes new clean reproduced outputs. Train/cal predictions are only included if already exported by the clean run.",
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    })

    final = read_final_metrics(clean_run)
    pred = load_prediction_files(clean_run)
    phase = phase1_audit(phase1_run)

    save_csv(phase, dirs["manifest"] / "phase1_clean_feature_inventory.csv")
    save_csv(final, dirs["metrics"] / "final_selected_candidate_metrics_from_clean_run.csv")
    if not pred.empty:
        save_csv(pred, dirs["pred"] / "heldout_test_predictions_combined.csv")

    split = split_manifest_from_predictions(pred)
    folds = fold_metrics(pred)
    risks = risk_zone_metrics(pred)
    ewidth = error_width_metrics(pred)
    runtime = runtime_by_stage(clean_run, final)

    save_csv(split, dirs["split"] / "test_fold_manifest_from_clean_predictions.csv")
    save_csv(folds, dirs["split"] / "test_fold_level_metrics.csv")
    save_csv(risks, dirs["metrics"] / "test_risk_zone_metrics.csv")
    save_csv(ewidth, dirs["interval"] / "test_error_vs_interval_width_metrics.csv")
    save_csv(runtime, dirs["runtime"] / "runtime_by_stage_audit.csv")

    if not pred.empty:
        plot_true_pred(pred, dirs["figs"])
        plot_residuals(pred, dirs["figs"])
        plot_error_width(pred, dirs["interval"])
        plot_interval_trajectories(pred, dirs["interval"])
        plot_coverage_heatmaps(pred, dirs["heatmaps"])

    caps = pd.concat([
        caption_table(dirs["figs"]),
        caption_table(dirs["interval"]),
        caption_table(dirs["heatmaps"]),
    ], ignore_index=True)
    save_csv(caps, dirs["summary"] / "figure_caption_audit.csv")

    gap_rows = []
    train_cal_patterns = ["*train*prediction*.csv", "*cal*prediction*.csv", "*calibration*prediction*.csv"]
    train_cal_files = []
    for pat in train_cal_patterns:
        train_cal_files.extend([p for p in clean_run.rglob(pat) if p.is_file() and p.stat().st_size > 1000])
    if not train_cal_files:
        gap_rows.append({
            "gap_item": "train_and_calibration_predictions",
            "status": "NOT_AVAILABLE_IN_EXISTING_CLEAN_RUN",
            "required_next_action": "Only if required by reviewers, run an instrumented retraining script that exports train/cal/test predictions separately.",
            "manuscript_rule": "Do not claim train/cal prediction diagnostics unless those files are generated."
        })
    if runtime["stage_breakdown_status"].eq("TOTAL_ONLY_AVAILABLE_FROM_CLEAN_RUN").any():
        gap_rows.append({
            "gap_item": "runtime_stage_breakdown",
            "status": "PARTIAL_TOTAL_RUNTIME_ONLY",
            "required_next_action": "Instrument final reproducer by stage if exact training/calibration/prediction times are required.",
            "manuscript_rule": "Use total runtime now; do not claim exact stage times unless generated."
        })
    gap = pd.DataFrame(gap_rows)
    save_csv(gap, dirs["summary"] / "diagnostic_evidence_gaps_do_not_invent.csv")

    verdict = {
        "output_dir": str(out_dir),
        "script_version": SCRIPT_VERSION,
        "prediction_files_loaded": int(pred["source_prediction_file"].nunique()) if not pred.empty and "source_prediction_file" in pred.columns else 0,
        "prediction_rows_loaded": int(len(pred)),
        "datasets_in_predictions": sorted(pred["dataset"].dropna().unique().tolist()) if not pred.empty else [],
        "fold_metrics_available": bool(not folds.empty),
        "risk_zone_metrics_available": bool(not risks.empty),
        "figures_generated": int(len(list(out_dir.rglob("*.png")))),
        "train_cal_predictions_available": bool(train_cal_files),
        "stage_runtime_breakdown_available": False,
        "verified_mode_note": "No train/cal diagnostics were fabricated. Existing clean output supports held-out/test diagnostics; gaps are explicitly reported.",
        "status": "PASS_WITH_EXPLICIT_GAPS" if len(pred) else "NO_PREDICTIONS_FOUND",
    }
    save_json(dirs["manifest"] / "FINAL_CANDIDATE_DIAGNOSTICS_VERDICT.json", verdict)

    banner("FINAL CANDIDATE DIAGNOSTICS COMPLETE")
    print(f"Output folder: {out_dir}")
    print()
    print("KEY OUTPUTS")
    print(f"- {dirs['pred'] / 'heldout_test_predictions_combined.csv'}")
    print(f"- {dirs['split'] / 'test_fold_level_metrics.csv'}")
    print(f"- {dirs['metrics'] / 'test_risk_zone_metrics.csv'}")
    print(f"- {dirs['summary'] / 'figure_caption_audit.csv'}")
    print(f"- {dirs['summary'] / 'diagnostic_evidence_gaps_do_not_invent.csv'}")
    print()
    print("VERDICT")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

