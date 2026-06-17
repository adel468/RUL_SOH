
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

import os
# Public release script: 10_reconcile_metric_definitions.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

# -*- coding: utf-8 -*-




import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple


ROOT = Path(os.environ.get("Q1_OUTPUT_ROOT", "outputs"))
EXACT_AUDIT_DIR = ROOT / "Q1_LITERAL_GATE_EXACT_AUDIT_LOCAL"
STRICT_AUDIT_DIR = ROOT / "Q1_strict-interval_LITERAL_SELECTED_ROW_AUDIT_LOCAL"

SELECTED_FILE = EXACT_AUDIT_DIR / "04_reconstructed_deterministic_selected_current_gate.csv"
SEED_MAP_FILE = STRICT_AUDIT_DIR / "01_seed_to_run_folder_map.csv"

OUT = ROOT / "Q1_RECONCILE_LITERAL_GATE_METRIC_DEFINITIONS"
OUT.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 250_000


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def norm_value(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def find_col(cols: List[str], names: List[str], required: bool = False) -> Optional[str]:
    mapping = {norm_col(c): c for c in cols}
    for name in names:
        key = norm_col(name)
        if key in mapping:
            return mapping[key]
    for c in cols:
        nc = norm_col(c)
        for name in names:
            key = norm_col(name)
            if key and key in nc:
                return c
    if required:
        raise KeyError(f"Could not find any of {names} in columns:\n{cols}")
    return None


def dataset_to_pred_stem(dataset: str) -> str:
    d = str(dataset).lower()
    if "c-mapss" in d or "c_mapss" in d or "cmapss" in d:
        return "c_mapss"
    if "battery" in d:
        return "battery"
    if "pronostia" in d or "femto" in d:
        return "pronostia_femto"
    if "xjtu" in d:
        return "xjtu_sy"
    if "ims" in d:
        return "ims"
    return re.sub(r"[^a-z0-9]+", "_", d).strip("_")


def infer_candidate_col(cols: List[str]) -> Optional[str]:
    col = find_col(cols, ["_candidate_id", "candidate_id", "candidate", "selected_candidate", "protocol_name", "protocol"])
    if col:
        return col
    return find_col(cols, ["model_name", "model", "config"])


def infer_y_cols(cols: List[str]) -> Tuple[str, str, str, str]:
    y_col = find_col(cols, ["y_true", "true", "target_true", "actual", "y", "target", "target_normalized_rul", "target_health", "capacity"], required=True)
    yp_col = find_col(cols, ["y_pred", "pred", "prediction", "yhat", "y_hat", "point_pred", "point_prediction"], required=True)
    l_col = find_col(cols, ["lower", "lo", "l", "lower_bound", "pred_lower", "interval_lower", "q_lower"], required=True)
    u_col = find_col(cols, ["upper", "hi", "u", "upper_bound", "pred_upper", "interval_upper", "q_upper"], required=True)
    return y_col, yp_col, l_col, u_col


def locate_pred_file(run_dir: Path, dataset: str) -> Optional[Path]:
    stem = dataset_to_pred_stem(dataset)
    candidates = [
        run_dir / "02_TABLES" / f"{stem}__pred_df.csv",
        run_dir / "02_TABLES" / f"{stem}_pred_df.csv",
        run_dir / f"{stem}__pred_df.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in sorted((run_dir / "02_TABLES").glob("*pred_df.csv")):
        if stem in p.name.lower():
            return p
    for p in sorted(run_dir.rglob("*pred_df.csv")):
        if stem in p.name.lower():
            return p
    return None


def extract_candidate_rows(pred_file: Path, candidate_id: str) -> Tuple[pd.DataFrame, dict]:
    head = pd.read_csv(pred_file, nrows=5)
    cols = list(head.columns)

    cand_col = infer_candidate_col(cols)
    if cand_col is None:
        raise KeyError(f"No candidate column in {pred_file}. Columns:\n{cols}")

    y_col, yp_col, l_col, u_col = infer_y_cols(cols)
    group_col = find_col(cols, ["asset_id", "unit", "unit_id", "battery_id", "bearing_id", "engine_id", "group", "group_id"])

    matched = []
    cid = norm_value(candidate_id)

    for chunk in pd.read_csv(pred_file, chunksize=CHUNKSIZE, low_memory=False):
        cvals = chunk[cand_col].astype(str).map(norm_value)
        mask = cvals.eq(cid)
        if not mask.any():
            mask = cvals.map(lambda x: re.sub(r"\s+", "", x)).eq(re.sub(r"\s+", "", cid))
        if mask.any():
            keep = [cand_col, y_col, yp_col, l_col, u_col]
            if group_col:
                keep.append(group_col)
            matched.append(chunk.loc[mask, list(dict.fromkeys(keep))].copy())

    df = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame()
    return df, {"candidate_col": cand_col, "y": y_col, "pred": yp_col, "lower": l_col, "upper": u_col, "group": group_col}


def rates_from_rows(df: pd.DataFrame, y_col: str, pred_col: str, lower_col: str, upper_col: str) -> dict:
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    yp = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df[lower_col], errors="coerce").to_numpy(float)
    up = pd.to_numeric(df[upper_col], errors="coerce").to_numpy(float)

    valid = np.isfinite(y) & np.isfinite(yp) & np.isfinite(lo) & np.isfinite(up)
    y, yp, lo, up = y[valid], yp[valid], lo[valid], up[valid]

    urgent = y <= 0.25
    critical = y <= 0.10
    covered = (y >= lo) & (y <= up)
    lower_miss = y < lo
    upper_miss = y > up

    def rate(mask, denom=None):
        if denom is None:
            return float(mask.mean()) if len(mask) else np.nan
        d = int(denom.sum())
        if d == 0:
            return np.nan
        return float((mask & denom).sum() / d)

    return {
        "n": int(len(y)),
        "coverage": rate(covered),
        "riskcov_y_le_025": rate(covered, urgent),
        "critcov_y_le_010": rate(covered, critical),
        "urgent_n": int(urgent.sum()),
        "critical_n": int(critical.sum()),

        # Safe-zone false-safe decompositions, denominator = urgent samples.
        "fsr050_point_only": rate(urgent & (yp > 0.50), urgent),
        "fsr050_interval_lower_only": rate(urgent & (lo > 0.50), urgent),
        "fsr050_point_or_lower": rate(urgent & ((yp > 0.50) | (lo > 0.50)), urgent),

        # Urgent-boundary diagnostics, denominator = urgent samples.
        "miss025_point_only": rate(urgent & (yp > 0.25), urgent),
        "miss025_interval_lower_only": rate(urgent & (lo > 0.25), urgent),
        "miss025_point_or_lower": rate(urgent & ((yp > 0.25) | (lo > 0.25)), urgent),

        # Underwarning/lower-bound miss alternatives.
        "under_all_rows": rate(lower_miss),
        "under_urgent_only": rate(lower_miss & urgent, urgent),
        "under_critical_only": rate(lower_miss & critical, critical),
        "under_nonurgent_only": rate(lower_miss & (~urgent), ~urgent),

        # Upper-side miss alternatives, for completeness.
        "upper_all_rows": rate(upper_miss),
        "upper_urgent_only": rate(upper_miss & urgent, urgent),
        "upper_critical_only": rate(upper_miss & critical, critical),
        "width_mean": float(np.mean(up - lo)) if len(y) else np.nan,
    }


def print_block(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


print_block("RECONCILING LITERAL-GATE METRIC DEFINITIONS")
print("SELECTED_FILE:", SELECTED_FILE)
print("SEED_MAP_FILE:", SEED_MAP_FILE)
print("OUT:", OUT)

if not SELECTED_FILE.exists():
    raise FileNotFoundError(SELECTED_FILE)
if not SEED_MAP_FILE.exists():
    raise FileNotFoundError("Missing seed map. Run Q1_strict-interval_LITERAL_SELECTED_ROW_AUDIT.py first.")

selected = pd.read_csv(SELECTED_FILE)
seed_map = pd.read_csv(SEED_MAP_FILE)
seed_to_run = {int(r["seed"]): Path(str(r["run_dir"])) for _, r in seed_map.iterrows()}

dataset_col = "_dataset" if "_dataset" in selected.columns else find_col(list(selected.columns), ["dataset"], required=True)
seed_col = "_seed" if "_seed" in selected.columns else find_col(list(selected.columns), ["seed", "seed_from_folder"], required=True)
candidate_col = "_candidate_id" if "_candidate_id" in selected.columns else find_col(list(selected.columns), ["candidate_id", "candidate"], required=True)

rows = []
missing = []

for _, srow in selected.iterrows():
    dataset = str(srow[dataset_col])
    seed = int(srow[seed_col])
    cid = norm_value(srow[candidate_col])
    run_dir = seed_to_run.get(seed)

    if run_dir is None:
        missing.append({"dataset": dataset, "seed": seed, "candidate_id": cid, "reason": "no run dir"})
        continue

    pred_file = locate_pred_file(run_dir, dataset)
    if pred_file is None:
        missing.append({"dataset": dataset, "seed": seed, "candidate_id": cid, "reason": "no pred_df"})
        continue

    try:
        rdf, colinfo = extract_candidate_rows(pred_file, cid)
        if rdf.empty:
            missing.append({"dataset": dataset, "seed": seed, "candidate_id": cid, "reason": "candidate not found", "pred_file": str(pred_file)})
            continue

        rates = rates_from_rows(rdf, colinfo["y"], colinfo["pred"], colinfo["lower"], colinfo["upper"])
        rates.update({
            "dataset": dataset,
            "seed": seed,
            "candidate_id": cid,
            "pred_file": str(pred_file),
            "stored_false_safe": float(srow["_false_safe"]) if "_false_safe" in selected.columns else np.nan,
            "stored_underwarning": float(srow["_underwarning"]) if "_underwarning" in selected.columns else np.nan,
            "stored_coverage": float(srow["_coverage"]) if "_coverage" in selected.columns else np.nan,
            "stored_riskcov": float(srow["_risk_coverage"]) if "_risk_coverage" in selected.columns else np.nan,
            "stored_critcov": float(srow["_critical_coverage"]) if "_critical_coverage" in selected.columns else np.nan,
        })
        rows.append(rates)

    except Exception as e:
        missing.append({"dataset": dataset, "seed": seed, "candidate_id": cid, "reason": "error", "error": str(e), "pred_file": str(pred_file)})

recon = pd.DataFrame(rows)
missing_df = pd.DataFrame(missing)

recon_path = OUT / "01_row_level_decomposed_metric_definitions.csv"
missing_path = OUT / "02_missing_reconciliation_rows.csv"
recon.to_csv(recon_path, index=False)
missing_df.to_csv(missing_path, index=False)

print("Rows reconciled:", len(recon))
print("Missing:", len(missing_df))
print("Saved:", recon_path)
print("Saved:", missing_path)

if recon.empty:
    raise RuntimeError("No rows reconciled.")

# Compare stored metrics against row-level alternatives.
compare_rows = []
metric_pairs = [
    ("stored_false_safe", "fsr050_point_only"),
    ("stored_false_safe", "fsr050_interval_lower_only"),
    ("stored_false_safe", "fsr050_point_or_lower"),
    ("stored_underwarning", "under_all_rows"),
    ("stored_underwarning", "under_urgent_only"),
    ("stored_underwarning", "under_critical_only"),
    ("stored_underwarning", "under_nonurgent_only"),
    ("stored_coverage", "coverage"),
    ("stored_riskcov", "riskcov_y_le_025"),
    ("stored_critcov", "critcov_y_le_010"),
]

for stored, calc in metric_pairs:
    if stored in recon.columns and calc in recon.columns:
        diff = (pd.to_numeric(recon[stored], errors="coerce") - pd.to_numeric(recon[calc], errors="coerce")).abs()
        compare_rows.append({
            "stored_metric": stored,
            "candidate_row_level_definition": calc,
            "max_abs_diff": float(diff.max()),
            "median_abs_diff": float(diff.median()),
            "mean_abs_diff": float(diff.mean()),
            "n_compared": int(diff.notna().sum()),
        })

comparison = pd.DataFrame(compare_rows).sort_values(["stored_metric", "max_abs_diff"])
comparison_path = OUT / "03_stored_metric_definition_match_report.csv"
comparison.to_csv(comparison_path, index=False)

print_block("STORED METRIC DEFINITION MATCH REPORT")
print(comparison.to_string(index=False))
print("Saved:", comparison_path)

# Dataset summary of decomposed strict metrics.
summary = (
    recon.groupby("dataset")
    .agg(
        seeds=("seed", "nunique"),
        rows=("seed", "count"),
        coverage_min=("coverage", "min"),
        riskcov_min=("riskcov_y_le_025", "min"),
        critcov_min=("critcov_y_le_010", "min"),
        stored_false_safe_max=("stored_false_safe", "max"),
        fsr050_point_only_max=("fsr050_point_only", "max"),
        fsr050_interval_lower_only_max=("fsr050_interval_lower_only", "max"),
        fsr050_point_or_lower_max=("fsr050_point_or_lower", "max"),
        miss025_point_only_max=("miss025_point_only", "max"),
        miss025_interval_lower_only_max=("miss025_interval_lower_only", "max"),
        miss025_point_or_lower_max=("miss025_point_or_lower", "max"),
        stored_underwarning_max=("stored_underwarning", "max"),
        under_all_rows_max=("under_all_rows", "max"),
        under_urgent_only_max=("under_urgent_only", "max"),
        under_critical_only_max=("under_critical_only", "max"),
        width_median=("width_mean", "median"),
    )
    .reset_index()
)

summary_path = OUT / "04_decomposed_definition_summary_by_dataset.csv"
summary.to_csv(summary_path, index=False)

print_block("DECOMPOSED DEFINITION SUMMARY BY DATASET")
print(summary.to_string(index=False))
print("Saved:", summary_path)

# status.
print_block("RECONCILIATION status")

best_false = comparison[comparison["stored_metric"].eq("stored_false_safe")].sort_values("max_abs_diff").head(1)
best_under = comparison[comparison["stored_metric"].eq("stored_underwarning")].sort_values("max_abs_diff").head(1)

if not best_false.empty:
    print("Best match for stored _false_safe:", best_false.iloc[0]["candidate_row_level_definition"], "max_abs_diff=", best_false.iloc[0]["max_abs_diff"])
if not best_under.empty:
    print("Best match for stored _underwarning:", best_under.iloc[0]["candidate_row_level_definition"], "max_abs_diff=", best_under.iloc[0]["max_abs_diff"])

interval025_fails = int((recon["miss025_interval_lower_only"] > 0).sum())
combined025_fails = int((recon["miss025_point_or_lower"] > 0).sum())
point025_fails = int((recon["miss025_point_only"] > 0).sum())

print("Rows with interval lower miss at 0.25:", interval025_fails, "of", len(recon))
print("Rows with point miss at 0.25:", point025_fails, "of", len(recon))
print("Rows with combined point-or-lower miss at 0.25:", combined025_fails, "of", len(recon))

print("\nInterpretation:")
print("- If stored _false_safe matches fsr050_interval_lower_only, then the implemented gate is interval-only.")
print("- If manuscript equation says point OR lower, the code/results do not support that wording.")
print("- Strict interval 0.25 failures are real only if miss025_interval_lower_only > 0 after correct extraction.")
print("- Point-only 0.25 failures should be reported as diagnostic, not necessarily interval-safety failure.")
print("=" * 90)

