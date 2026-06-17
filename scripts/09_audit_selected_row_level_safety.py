
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

import os
# Public release script: 09_audit_selected_row_level_safety.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

# -*- coding: utf-8 -*-




import json
import math
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_ROOT = Path(os.environ.get("Q1_OUTPUT_ROOT", "outputs"))
LITERAL_DIR = (
    OUTPUT_ROOT
    / "Q1_LITERAL_GATE_ALL_DATASETS_repeated-seed_RESELECT_AUDIT_PACKAGE_EXTRACTED"
    / "Q1_LITERAL_GATE_ALL_DATASETS_repeated-seed_RESELECT_AUDIT"
)

EXACT_AUDIT_DIR = OUTPUT_ROOT / "Q1_LITERAL_GATE_EXACT_AUDIT_LOCAL"

SELECTED_FILE = EXACT_AUDIT_DIR / "04_reconstructed_deterministic_selected_current_gate.csv"


OUT = OUTPUT_ROOT / "Q1_strict-interval_LITERAL_SELECTED_ROW_AUDIT_LOCAL"
OUT.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 250_000

# Set True only if you want to save row-level extracted selected predictions.
# These can be large. The metrics are enough for first decision.
SAVE_EXTRACTED_ROW_LEVEL = False


# =============================================================================
# HELPERS
# =============================================================================

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

    # Fuzzy contains; avoid overly broad candidates unless no exact.
    for c in cols:
        nc = norm_col(c)
        for name in names:
            key = norm_col(name)
            if key and key in nc:
                return c

    if required:
        raise KeyError(f"Could not find any of {names} in columns:\n{cols}")
    return None


def read_csv_safe(p: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(p, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="latin1", **kwargs)


def dataset_to_pred_stem(dataset: str) -> str:
    d = str(dataset).strip().lower()
    if "c-mapss" in d or "c_mapss" in d or "cmapss" in d or "mapss" in d:
        return "c_mapss"
    if "battery" in d:
        return "battery"
    if "pronostia" in d or "femto" in d:
        return "pronostia_femto"
    if "xjtu" in d:
        return "xjtu_sy"
    if d == "ims" or "ims" in d:
        return "ims"
    return re.sub(r"[^a-z0-9]+", "_", d).strip("_")


def infer_dataset_col(cols: List[str]) -> Optional[str]:
    return find_col(cols, ["dataset", "dataset_name", "_dataset"])


def infer_seed_col(cols: List[str]) -> Optional[str]:
    return find_col(cols, ["seed", "_seed", "random_seed", "split_seed", "seed_from_folder"])


def infer_candidate_col(cols: List[str]) -> Optional[str]:
    # Prefer exact candidate id fields. Avoid generic "model" unless needed.
    exact_names = ["_candidate_id", "candidate_id", "candidate", "selected_candidate", "protocol_name", "protocol"]
    col = find_col(cols, exact_names, required=False)
    if col:
        return col
    return find_col(cols, ["model_name", "model", "config"], required=False)


def infer_y_cols(cols: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    y_col = find_col(cols, ["y_true", "true", "target_true", "actual", "y", "target", "target_normalized_rul", "target_health", "capacity"])
    yp_col = find_col(cols, ["y_pred", "pred", "prediction", "yhat", "y_hat", "point_pred", "point_prediction"])
    l_col = find_col(cols, ["lower", "lo", "l", "lower_bound", "pred_lower", "interval_lower", "q_lower"])
    u_col = find_col(cols, ["upper", "hi", "u", "upper_bound", "pred_upper", "interval_upper", "q_upper"])
    return y_col, yp_col, l_col, u_col


def metric_block(y_true, y_pred, lower, upper) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(lower) & np.isfinite(upper)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    lower = lower[valid]
    upper = upper[valid]

    n = len(y_true)
    if n == 0:
        return {"n": 0}

    urgent = y_true <= 0.25
    critical = y_true <= 0.10

    covered = (y_true >= lower) & (y_true <= upper)
    underwarn = y_true < lower

    fsr_safe_zone_050 = urgent & ((y_pred > 0.50) | (lower > 0.50))
    point_intervention_miss_025 = urgent & (y_pred > 0.25)
    interval_intervention_miss_025 = urgent & (lower > 0.25)
    combined_intervention_miss_025 = urgent & ((y_pred > 0.25) | (lower > 0.25))

    def rate(mask, denom=None):
        if denom is None:
            return float(np.mean(mask)) if len(mask) else np.nan
        d = int(np.sum(denom))
        if d == 0:
            return np.nan
        return float(np.sum(mask & denom) / d)

    out = {
        "n": int(n),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if n >= 2 and np.nanstd(y_true) > 0 else np.nan,
        "coverage_picp": rate(covered),
        "riskcov_y_le_025": rate(covered, urgent),
        "critcov_y_le_010": rate(covered, critical),
        "critical_n": int(np.sum(critical)),
        "urgentcritical_n": int(np.sum(urgent)),
        "fsr_safe_zone_050": rate(fsr_safe_zone_050, urgent),
        "point_intervention_miss_025": rate(point_intervention_miss_025, urgent),
        "interval_intervention_miss_025": rate(interval_intervention_miss_025, urgent),
        "combined_intervention_miss_025": rate(combined_intervention_miss_025, urgent),
        "underwarning_rate": rate(underwarn),
        "width_mean": float(np.mean(upper - lower)),
        "width_median": float(np.median(upper - lower)),
    }
    return out


def print_block(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# =============================================================================
# 1. LOAD FINAL 150 SELECTED CANDIDATES
# =============================================================================

print_block("STRICT 0.25 AUDIT FOR LITERAL-GATE SELECTED CANDIDATES")
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("LITERAL_DIR:", LITERAL_DIR)
print("EXACT_AUDIT_DIR:", EXACT_AUDIT_DIR)
print("SELECTED_FILE:", SELECTED_FILE)
print("RERUN_ROOT:", RERUN_ROOT)
print("OUT:", OUT)

if not SELECTED_FILE.exists():
    raise FileNotFoundError(
        f"Missing selected file:\n{SELECTED_FILE}\n\n"
        "Run Q1_LITERAL_GATE_EXACT_AUDIT_LOCAL.py first."
    )

selected = read_csv_safe(SELECTED_FILE)

dataset_col = "_dataset" if "_dataset" in selected.columns else find_col(list(selected.columns), ["dataset"], required=True)
seed_col = "_seed" if "_seed" in selected.columns else find_col(list(selected.columns), ["seed", "seed_from_folder"], required=True)
candidate_col = "_candidate_id" if "_candidate_id" in selected.columns else infer_candidate_col(list(selected.columns))

if candidate_col is None:
    raise KeyError(f"Could not infer candidate column in selected file. Columns:\n{list(selected.columns)}")

selected["_dataset_key"] = selected[dataset_col].astype(str)
selected["_seed_key"] = pd.to_numeric(selected[seed_col], errors="coerce").astype("Int64")
selected["_candidate_key"] = selected[candidate_col].astype(str).map(norm_value)

print("Selected rows loaded:", len(selected))
print("Datasets:")
print(selected["_dataset_key"].value_counts().to_string())
print("Candidate column:", candidate_col)

selected.to_csv(OUT / "00_selected_150_literal_gate_candidates_loaded.csv", index=False)


# =============================================================================
# 2. MAP SEEDS TO EXISTING RUN FOLDERS
# =============================================================================

seed_run_map: Dict[int, Path] = {}

disc_path = LITERAL_DIR / "01_AUDIT" / "discovered_seed_runs.csv"
if disc_path.exists():
    disc = read_csv_safe(disc_path)
    dcols = list(disc.columns)
    dseed_col = find_col(dcols, ["seed", "random_seed", "split_seed"], required=False)
    path_col = find_col(dcols, ["run_dir", "run_path", "path", "folder", "run_folder", "output_dir", "root"], required=False)

    if dseed_col and path_col:
        for _, r in disc.iterrows():
            try:
                s = int(r[dseed_col])
                p = Path(str(r[path_col]))
                if p.exists():
                    seed_run_map[s] = p
            except Exception:
                pass

    print("Seed-run mappings from discovered_seed_runs.csv:", len(seed_run_map))
else:
    print("No discovered_seed_runs.csv found:", disc_path)

# Fallback: scan run_config.json under RERUN_ROOT.
for run_dir in sorted(RERUN_ROOT.glob("Q1_SINGLE_FILE_REPRODUCTION_RUN_*")):
    if not run_dir.is_dir():
        continue

    possible_jsons = list(run_dir.glob("*.json")) + list(run_dir.rglob("run_config.json"))
    seed_found = None

    for jp in possible_jsons:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            # Try common keys.
            for key in ["seed", "random_seed", "split_seed", "master_seed"]:
                if key in obj:
                    val = obj[key]
                    if isinstance(val, (int, float, str)) and str(val).isdigit():
                        seed_found = int(val)
                        break
            if seed_found is not None:
                break
        except Exception:
            continue

    # Last fallback: if a file in the run folder contains a seed column with a single value.
    if seed_found is None:
        # Do not expensive-scan all CSVs; use likely summary files only.
        for csv in list(run_dir.glob("*.csv")) + list((run_dir / "02_TABLES").glob("*summary*.csv")):
            try:
                head = pd.read_csv(csv, nrows=20)
                sc = find_col(list(head.columns), ["seed", "random_seed", "split_seed"])
                if sc:
                    vals = pd.to_numeric(head[sc], errors="coerce").dropna().astype(int).unique()
                    if len(vals) == 1:
                        seed_found = int(vals[0])
                        break
            except Exception:
                pass

    if seed_found is not None and seed_found not in seed_run_map:
        seed_run_map[seed_found] = run_dir

seed_map_df = pd.DataFrame(
    [{"seed": k, "run_dir": str(v)} for k, v in sorted(seed_run_map.items())]
)
seed_map_df.to_csv(OUT / "01_seed_to_run_folder_map.csv", index=False)

print("Total seed-run mappings:", len(seed_run_map))
print(seed_map_df.head(50).to_string(index=False))


# =============================================================================
# 3. EXTRACT SELECTED CANDIDATE ROW PREDICTIONS FROM pred_df FILES
# =============================================================================

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

    # Fallback glob.
    for p in sorted((run_dir / "02_TABLES").glob("*pred_df.csv")):
        if stem in p.name.lower():
            return p
    for p in sorted(run_dir.rglob("*pred_df.csv")):
        if stem in p.name.lower():
            return p
    return None


def extract_candidate_predictions(pred_file: Path, candidate_id: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    

    head = pd.read_csv(pred_file, nrows=5)
    cols = list(head.columns)

    cand_col = infer_candidate_col(cols)
    y_col, yp_col, l_col, u_col = infer_y_cols(cols)
    group_col = find_col(cols, ["asset_id", "unit", "unit_id", "battery_id", "bearing_id", "engine_id", "group", "group_id"])

    if cand_col is None:
        raise KeyError(f"No candidate column found in {pred_file}. Columns:\n{cols}")
    if not all([y_col, yp_col, l_col, u_col]):
        raise KeyError(
            f"Missing y/pred/lower/upper columns in {pred_file}\n"
            f"y={y_col}, pred={yp_col}, lower={l_col}, upper={u_col}\n"
            f"Columns:\n{cols}"
        )

    matched_chunks = []
    candidate_id_norm = norm_value(candidate_id)

    for chunk in pd.read_csv(pred_file, chunksize=CHUNKSIZE, low_memory=False):
        # Exact string match first.
        cvals = chunk[cand_col].astype(str).map(norm_value)
        mask = cvals.eq(candidate_id_norm)

        # Fallback normalized-column-like comparison if exact fails in this chunk.
        if not mask.any():
            cvals2 = cvals.map(lambda x: re.sub(r"\s+", "", x))
            cid2 = re.sub(r"\s+", "", candidate_id_norm)
            mask = cvals2.eq(cid2)

        if mask.any():
            keep_cols = [cand_col, y_col, yp_col, l_col, u_col]
            if group_col:
                keep_cols.append(group_col)
            keep_cols = list(dict.fromkeys(keep_cols))
            matched_chunks.append(chunk.loc[mask, keep_cols].copy())

    if matched_chunks:
        out = pd.concat(matched_chunks, ignore_index=True)
    else:
        out = pd.DataFrame()

    colinfo = {
        "candidate_col": cand_col,
        "y_col": y_col,
        "y_pred_col": yp_col,
        "lower_col": l_col,
        "upper_col": u_col,
        "group_col": group_col if group_col else "",
    }
    return out, colinfo


audit_rows = []
missing_rows = []
per_group_rows = []

for i, row in selected.iterrows():
    dataset = row["_dataset_key"]
    seed = int(row["_seed_key"])
    candidate_id = row["_candidate_key"]

    run_dir = seed_run_map.get(seed)
    if run_dir is None or not Path(run_dir).exists():
        missing_rows.append({
            "dataset": dataset,
            "seed": seed,
            "candidate_id": candidate_id,
            "reason": "NO_RUN_DIR_FOR_SEED",
        })
        continue

    pred_file = locate_pred_file(Path(run_dir), dataset)
    if pred_file is None:
        missing_rows.append({
            "dataset": dataset,
            "seed": seed,
            "candidate_id": candidate_id,
            "run_dir": str(run_dir),
            "reason": "NO_PRED_DF_FILE_FOUND",
        })
        continue

    try:
        pred_df, colinfo = extract_candidate_predictions(pred_file, candidate_id)

        if pred_df.empty:
            # Save candidate values sample for debugging.
            try:
                head = pd.read_csv(pred_file, nrows=5000)
                cand_col_tmp = infer_candidate_col(list(head.columns))
                candidate_sample = " || ".join(
                    sorted(head[cand_col_tmp].astype(str).dropna().unique())[:30]
                ) if cand_col_tmp else ""
            except Exception:
                candidate_sample = ""

            missing_rows.append({
                "dataset": dataset,
                "seed": seed,
                "candidate_id": candidate_id,
                "run_dir": str(run_dir),
                "pred_file": str(pred_file),
                "reason": "CANDIDATE_NOT_FOUND_IN_PRED_DF",
                "candidate_sample_first_5000": candidate_sample,
            })
            continue

        y_col = colinfo["y_col"]
        yp_col = colinfo["y_pred_col"]
        l_col = colinfo["lower_col"]
        u_col = colinfo["upper_col"]
        group_col = colinfo["group_col"] or None

        m = metric_block(
            pred_df[y_col].to_numpy(),
            pred_df[yp_col].to_numpy(),
            pred_df[l_col].to_numpy(),
            pred_df[u_col].to_numpy(),
        )

        m.update({
            "dataset": dataset,
            "seed": seed,
            "candidate_id": candidate_id,
            "run_dir": str(run_dir),
            "pred_file": str(pred_file),
            "candidate_col": colinfo["candidate_col"],
            "y_col": y_col,
            "y_pred_col": yp_col,
            "lower_col": l_col,
            "upper_col": u_col,
            "group_col": group_col if group_col else "",
            "rows_extracted": int(len(pred_df)),
        })

        audit_rows.append(m)

        if group_col:
            for gv, sub in pred_df.groupby(group_col):
                gm = metric_block(
                    sub[y_col].to_numpy(),
                    sub[yp_col].to_numpy(),
                    sub[l_col].to_numpy(),
                    sub[u_col].to_numpy(),
                )
                gm.update({
                    "dataset": dataset,
                    "seed": seed,
                    "candidate_id": candidate_id,
                    "group_col": group_col,
                    "group_value": str(gv),
                    "rows_extracted": int(len(sub)),
                    "pred_file": str(pred_file),
                })
                per_group_rows.append(gm)

        if SAVE_EXTRACTED_ROW_LEVEL:
            ds_slug = dataset_to_pred_stem(dataset)
            out_pred = OUT / "extracted_row_predictions"
            out_pred.mkdir(exist_ok=True)
            pred_df.to_csv(out_pred / f"{ds_slug}__seed_{seed}__selected_candidate_predictions.csv", index=False)

        print(f"[OK] {dataset} seed={seed} rows={len(pred_df)} candidate={candidate_id[:80]}")

    except Exception as e:
        missing_rows.append({
            "dataset": dataset,
            "seed": seed,
            "candidate_id": candidate_id,
            "run_dir": str(run_dir),
            "pred_file": str(pred_file),
            "reason": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        print(f"[ERROR] {dataset} seed={seed}: {e}")


audit_df = pd.DataFrame(audit_rows)
missing_df = pd.DataFrame(missing_rows)
per_group_df = pd.DataFrame(per_group_rows)

audit_path = OUT / "02_strict-interval_literal_selected_row_metrics.csv"
missing_path = OUT / "03_missing_or_unmatched_literal_selected_predictions.csv"
group_path = OUT / "04_strict-interval_literal_selected_per_group_metrics.csv"

audit_df.to_csv(audit_path, index=False)
missing_df.to_csv(missing_path, index=False)
per_group_df.to_csv(group_path, index=False)

print_block("ROW-LEVEL STRICT 0.25 EXTRACTION RESULTS")
print("Audit rows:", len(audit_df))
print("Missing/unmatched rows:", len(missing_df))
print("Per-group rows:", len(per_group_df))
print("Saved:", audit_path)
print("Saved:", missing_path)
print("Saved:", group_path)


# =============================================================================
# 4. SUMMARIZE AND FLAG FAILURES
# =============================================================================

if not audit_df.empty:
    audit_df["fails_current_gate_from_rows"] = (
        (audit_df["coverage_picp"] < 0.90)
        | (audit_df["riskcov_y_le_025"] < 0.90)
        | (audit_df["fsr_safe_zone_050"] > 0)
        | (audit_df["underwarning_rate"] > 0.05)
    )

    audit_df["fails_interval_operational_gate_025"] = (
        (audit_df["coverage_picp"] < 0.90)
        | (audit_df["riskcov_y_le_025"] < 0.90)
        | (audit_df["interval_intervention_miss_025"] > 0)
        | (audit_df["underwarning_rate"] > 0.05)
    )

    audit_df["fails_combined_combined_gate_025"] = (
        (audit_df["coverage_picp"] < 0.90)
        | (audit_df["riskcov_y_le_025"] < 0.90)
        | (audit_df["combined_intervention_miss_025"] > 0)
        | (audit_df["underwarning_rate"] > 0.05)
    )

    audit_df.to_csv(OUT / "02A_strict-interval_literal_selected_row_metrics_with_fail_flags.csv", index=False)

    summary = (
        audit_df.groupby("dataset")
        .agg(
            seeds=("seed", "nunique"),
            rows=("seed", "count"),
            coverage_min=("coverage_picp", "min"),
            coverage_median=("coverage_picp", "median"),
            riskcov_min=("riskcov_y_le_025", "min"),
            riskcov_median=("riskcov_y_le_025", "median"),
            critcov_min=("critcov_y_le_010", "min"),
            critcov_median=("critcov_y_le_010", "median"),
            critical_n_min=("critical_n", "min"),
            critical_n_median=("critical_n", "median"),
            fsr_safe_zone_050_max=("fsr_safe_zone_050", "max"),
            point_intervention_miss_025_max=("point_intervention_miss_025", "max"),
            point_intervention_miss_025_median=("point_intervention_miss_025", "median"),
            interval_intervention_miss_025_max=("interval_intervention_miss_025", "max"),
            interval_intervention_miss_025_median=("interval_intervention_miss_025", "median"),
            combined_intervention_miss_025_max=("combined_intervention_miss_025", "max"),
            combined_intervention_miss_025_median=("combined_intervention_miss_025", "median"),
            underwarning_max=("underwarning_rate", "max"),
            width_median=("width_mean", "median"),
            current_gate_failures=("fails_current_gate_from_rows", "sum"),
            interval025_gate_failures=("fails_interval_operational_gate_025", "sum"),
            combined025_gate_failures=("fails_combined_combined_gate_025", "sum"),
        )
        .reset_index()
    )

    summary_path = OUT / "05_strict-interval_literal_selected_summary_by_dataset.csv"
    summary.to_csv(summary_path, index=False)

    current_fails = audit_df[audit_df["fails_current_gate_from_rows"]].copy()
    interval_fails = audit_df[audit_df["fails_interval_operational_gate_025"]].copy()
    combined_fails = audit_df[audit_df["fails_combined_combined_gate_025"]].copy()

    current_fails.to_csv(OUT / "06_current_gate_failures_from_row_level_predictions.csv", index=False)
    interval_fails.to_csv(OUT / "07_interval025_gate_failures_from_row_level_predictions.csv", index=False)
    combined_fails.to_csv(OUT / "08_combined025_gate_failures_from_row_level_predictions.csv", index=False)

    print_block("STRICT 0.25 SUMMARY BY DATASET")
    print(summary.to_string(index=False))
    print("Saved:", summary_path)

    print_block("FAIL COUNTS")
    print("Current gate failures from row-level predictions:", int(current_fails.shape[0]))
    print("Interval operational 0.25 gate failures:", int(interval_fails.shape[0]))
    print("combined combined 0.25 gate failures:", int(combined_fails.shape[0]))

    if not interval_fails.empty:
        cols_show = [
            "dataset", "seed", "coverage_picp", "riskcov_y_le_025", "fsr_safe_zone_050",
            "point_intervention_miss_025", "interval_intervention_miss_025",
            "combined_intervention_miss_025", "underwarning_rate", "candidate_id"
        ]
        print("\nInterval 0.25 failures:")
        print(interval_fails[cols_show].to_string(index=False))

    print_block("status")
    expected = 150
    print("Expected literal selected candidates:", expected)
    print("Audited row-level selected candidates:", len(audit_df))
    print("Missing/unmatched selected candidates:", len(missing_df))

    if len(audit_df) == expected and len(missing_df) == 0:
        print("ROW-LEVEL SOURCE MATCH: GREEN")
    else:
        print("ROW-LEVEL SOURCE MATCH: RED/YELLOW â€” some selected candidates could not be audited.")

    if len(audit_df) == expected and len(missing_df) == 0 and int(current_fails.shape[0]) == 0:
        print("CURRENT WRITTEN GATE FROM ROW-LEVEL PREDICTIONS: GREEN")
    else:
        print("CURRENT WRITTEN GATE FROM ROW-LEVEL PREDICTIONS: RED/YELLOW")

    if len(audit_df) == expected and len(missing_df) == 0 and int(interval_fails.shape[0]) == 0:
        print("STRICT INTERVAL 0.25 OPERATIONAL GATE: GREEN")
    else:
        print("STRICT INTERVAL 0.25 OPERATIONAL GATE: RED/YELLOW")

    if len(audit_df) == expected and len(missing_df) == 0 and int(combined_fails.shape[0]) == 0:
        print("combined COMBINED POINT-OR-INTERVAL 0.25 GATE: GREEN")
    else:
        print("combined COMBINED POINT-OR-INTERVAL 0.25 GATE: RED/YELLOW")

else:
    print_block("status")
    print("No row-level selected predictions were audited. Check missing/unmatched file.")

