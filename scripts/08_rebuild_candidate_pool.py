
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

import os
# Public release script: 08_rebuild_candidate_pool.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

# -*- coding: utf-8 -*-




from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(os.environ.get("Q1_OUTPUT_ROOT", "outputs"))
LITERAL_DIR = (
    ROOT
    / "Q1_LITERAL_GATE_ALL_DATASETS_repeated-seed_RESELECT_AUDIT_PACKAGE_EXTRACTED"
    / "Q1_LITERAL_GATE_ALL_DATASETS_repeated-seed_RESELECT_AUDIT"
)

OUT = ROOT / "Q1_LITERAL_GATE_EXACT_AUDIT_LOCAL"
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def find_col(cols, names, required=False):
    mapping = {norm_col(c): c for c in cols}

    for name in names:
        key = norm_col(name)
        if key in mapping:
            return mapping[key]

    for c in cols:
        nc = norm_col(c)
        for name in names:
            if norm_col(name) in nc:
                return c

    if required:
        raise KeyError(f"Could not find any of {names} in columns:\n{cols}")
    return None


def read_csv_safe(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="latin1")


def print_block(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# =============================================================================
# 0. BASIC CHECKS
# =============================================================================

print_block("Q1 LITERAL-GATE EXACT AUDIT")
print("ROOT:", ROOT)
print("LITERAL_DIR:", LITERAL_DIR)
print("LITERAL_DIR exists:", LITERAL_DIR.exists())
print("OUT:", OUT)

if not LITERAL_DIR.exists():
    raise FileNotFoundError(
        "Literal-gate audit directory not found. Expected:\n"
        f"{LITERAL_DIR}\n\n"
        "First extract Q1_LITERAL_GATE_ALL_DATASETS_repeated-seed_RESELECT_AUDIT_PACKAGE.zip."
    )


# =============================================================================
# 1. READ FINAL_STATUS / README
# =============================================================================

status_path = LITERAL_DIR / "FINAL_STATUS.json"
readme_path = LITERAL_DIR / "README.txt"

if status_path.exists():
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)
        print_block("FINAL_STATUS.json")
        print(json.dumps(status, indent=2))
        with open(OUT / "00_FINAL_STATUS_COPY.json", "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print("Could not read FINAL_STATUS.json:", e)
else:
    print("No FINAL_STATUS.json found.")

if readme_path.exists():
    txt = readme_path.read_text(encoding="utf-8", errors="ignore")
    (OUT / "00_README_COPY.txt").write_text(txt, encoding="utf-8")
    print_block("README first 2000 chars")
    print(txt[:2000])
else:
    print("No README.txt found.")


# =============================================================================
# 2. INVENTORY CSV FILES AND COLUMNS
# =============================================================================

inventory_rows = []
for p in sorted(LITERAL_DIR.rglob("*.csv")):
    try:
        head = pd.read_csv(p, nrows=3)
        inventory_rows.append({
            "file": str(p),
            "relative_file": str(p.relative_to(LITERAL_DIR)),
            "name": p.name,
            "columns": " | ".join(map(str, head.columns)),
        })
    except Exception as e:
        inventory_rows.append({
            "file": str(p),
            "relative_file": str(p.relative_to(LITERAL_DIR)),
            "name": p.name,
            "columns": f"ERROR: {e}",
        })

inventory = pd.DataFrame(inventory_rows)
inventory_path = OUT / "00_literal_gate_file_inventory_with_columns.csv"
inventory.to_csv(inventory_path, index=False)

print_block("CSV INVENTORY")
print("CSV files:", len(inventory))
print("Saved:", inventory_path)
print(inventory[["relative_file", "name"]].head(20).to_string(index=False))


# =============================================================================
# 3. LOAD PER-SEED CANDIDATE POOLS AND ORIGINAL SELECTED METRICS
# =============================================================================

per_seed_dir = LITERAL_DIR / "02_PER_SEED"
if not per_seed_dir.exists():
    raise FileNotFoundError(f"Missing per-seed directory: {per_seed_dir}")

candidate_frames = []
original_frames = []

for seed_dir in sorted(per_seed_dir.glob("seed_*")):
    if not seed_dir.is_dir():
        continue

    try:
        seed = int(seed_dir.name.replace("seed_", ""))
    except Exception:
        seed = seed_dir.name

    cand_file = seed_dir / "candidate_pool_normalized_metrics.csv"
    orig_file = seed_dir / "original_selected_metrics_normalized.csv"

    if cand_file.exists():
        df = read_csv_safe(cand_file)
        df["seed_from_folder"] = seed
        df["source_file"] = str(cand_file)
        candidate_frames.append(df)

    if orig_file.exists():
        df = read_csv_safe(orig_file)
        df["seed_from_folder"] = seed
        df["source_file"] = str(orig_file)
        original_frames.append(df)

candidate_pool = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
original_selected = pd.concat(original_frames, ignore_index=True) if original_frames else pd.DataFrame()

candidate_pool_path = OUT / "01_all_candidate_pool_normalized_metrics_combined.csv"
original_selected_path = OUT / "02_all_original_selected_metrics_normalized_combined.csv"

candidate_pool.to_csv(candidate_pool_path, index=False)
original_selected.to_csv(original_selected_path, index=False)

print_block("LOADED TABLES")
print("Candidate pool rows:", len(candidate_pool))
print("Original selected rows:", len(original_selected))
print("Candidate pool saved:", candidate_pool_path)
print("Original selected saved:", original_selected_path)

if candidate_pool.empty:
    raise RuntimeError("No candidate_pool_normalized_metrics.csv files were loaded.")


# =============================================================================
# 4. INFER COLUMNS
# =============================================================================

cols = list(candidate_pool.columns)

dataset_col = find_col(cols, ["dataset", "dataset_name"], required=True)
seed_col = find_col(cols, ["seed", "random_seed", "split_seed", "seed_from_folder"], required=False)
candidate_col = find_col(cols, ["candidate", "candidate_id", "model", "model_name", "protocol", "config"], required=False)

coverage_col = find_col(cols, ["coverage", "picp", "coverage_picp", "empirical_coverage"], required=True)
riskcov_col = find_col(cols, ["riskcov", "risk_coverage", "riskcov_y_le_025", "urgent_critical_coverage"], required=True)
fsr_col = find_col(cols, ["false_safe", "false_safe_rate", "fsr"], required=True)
under_col = find_col(cols, ["underwarning", "underwarn", "underwarning_rate"], required=True)

width_col = find_col(cols, ["width", "mean_width", "width_mean", "interval_width"], required=False)
rmse_col = find_col(cols, ["rmse"], required=False)
mae_col = find_col(cols, ["mae"], required=False)
r2_col = find_col(cols, ["r2", "r_squared"], required=False)
critcov_col = find_col(cols, ["critcov", "critical_coverage", "critcov_y_le_010"], required=False)
critical_n_col = find_col(cols, ["critical_n", "n_critical", "critical_count"], required=False)

print_block("INFERRED COLUMNS")
for k, v in {
    "dataset_col": dataset_col,
    "seed_col": seed_col,
    "candidate_col": candidate_col,
    "coverage_col": coverage_col,
    "riskcov_col": riskcov_col,
    "fsr_col": fsr_col,
    "under_col": under_col,
    "width_col": width_col,
    "rmse_col": rmse_col,
    "mae_col": mae_col,
    "r2_col": r2_col,
    "critcov_col": critcov_col,
    "critical_n_col": critical_n_col,
}.items():
    print(f"{k}: {v}")


# =============================================================================
# 5. CURRENT WRITTEN-GATE DETERMINISTIC SELECTOR
# =============================================================================

cp = candidate_pool.copy()

cp["_dataset"] = cp[dataset_col].astype(str)

if seed_col and seed_col in cp.columns:
    cp["_seed"] = cp[seed_col]
else:
    cp["_seed"] = cp["seed_from_folder"]

cp["_coverage"] = pd.to_numeric(cp[coverage_col], errors="coerce")
cp["_riskcov"] = pd.to_numeric(cp[riskcov_col], errors="coerce")
cp["_fsr"] = pd.to_numeric(cp[fsr_col], errors="coerce")
cp["_under"] = pd.to_numeric(cp[under_col], errors="coerce")
cp["_width"] = pd.to_numeric(cp[width_col], errors="coerce") if width_col else np.nan
cp["_rmse"] = pd.to_numeric(cp[rmse_col], errors="coerce") if rmse_col else np.nan
cp["_mae"] = pd.to_numeric(cp[mae_col], errors="coerce") if mae_col else np.nan
cp["_r2"] = pd.to_numeric(cp[r2_col], errors="coerce") if r2_col else np.nan
cp["_critcov"] = pd.to_numeric(cp[critcov_col], errors="coerce") if critcov_col else np.nan
cp["_critical_n"] = pd.to_numeric(cp[critical_n_col], errors="coerce") if critical_n_col else np.nan

cp["_eligible_current_gate"] = (
    (cp["_coverage"] >= 0.90)
    & (cp["_riskcov"] >= 0.90)
    & (cp["_fsr"] == 0)
    & (cp["_under"] <= 0.05)
)

cp_flag_path = OUT / "03_candidate_pool_with_current_gate_flags.csv"
cp.to_csv(cp_flag_path, index=False)

selected_rows = []
unresolved_rows = []

for (dataset, seed), sub in cp.groupby(["_dataset", "_seed"], dropna=False):
    eligible = sub[sub["_eligible_current_gate"]].copy()

    if eligible.empty:
        unresolved_rows.append({
            "dataset": dataset,
            "seed": seed,
            "n_candidates": len(sub),
            "status": "NO_ELIGIBLE_CURRENT_GATE",
            "coverage_max": sub["_coverage"].max(),
            "riskcov_max": sub["_riskcov"].max(),
            "fsr_min": sub["_fsr"].min(),
            "under_min": sub["_under"].min(),
        })
        continue

    eligible["_rank_width"] = eligible["_width"].fillna(np.inf)
    eligible["_rank_rmse"] = eligible["_rmse"].fillna(np.inf)
    eligible["_rank_mae"] = eligible["_mae"].fillna(np.inf)

    # Hard constraints already applied. Then sharpness/width. Then RMSE. Then MAE.
    eligible = eligible.sort_values(
        ["_rank_width", "_rank_rmse", "_rank_mae"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    row = eligible.iloc[0].to_dict()
    row["_selection_status"] = "SELECTED_BY_RECONSTRUCTED_CURRENT_GATE"
    row["_n_candidates"] = len(sub)
    row["_n_eligible"] = len(eligible)
    selected_rows.append(row)

selected = pd.DataFrame(selected_rows)
unresolved = pd.DataFrame(unresolved_rows)

selected_path = OUT / "04_reconstructed_deterministic_selected_current_gate.csv"
unresolved_path = OUT / "05_unresolved_seed_dataset_current_gate.csv"

selected.to_csv(selected_path, index=False)
unresolved.to_csv(unresolved_path, index=False)

print_block("RECONSTRUCTED CURRENT-GATE SELECTION")
print("Selected rows:", len(selected))
print("Unresolved seed-dataset rows:", len(unresolved))
print("Selected saved:", selected_path)
print("Unresolved saved:", unresolved_path)

if not unresolved.empty:
    print("\nUNRESOLVED ROWS:")
    print(unresolved.to_string(index=False))


# =============================================================================
# 6. DATASET SUMMARY
# =============================================================================

if selected.empty:
    summary = pd.DataFrame()
else:
    summary = (
        selected.groupby("_dataset")
        .agg(
            seeds=("_seed", "nunique"),
            selected_rows=("_seed", "count"),
            coverage_min=("_coverage", "min"),
            coverage_median=("_coverage", "median"),
            riskcov_min=("_riskcov", "min"),
            riskcov_median=("_riskcov", "median"),
            fsr_max=("_fsr", "max"),
            underwarning_max=("_under", "max"),
            width_median=("_width", "median"),
            rmse_median=("_rmse", "median"),
            mae_median=("_mae", "median"),
            r2_median=("_r2", "median"),
            critcov_min=("_critcov", "min"),
            critcov_median=("_critcov", "median"),
            critical_n_min=("_critical_n", "min"),
            critical_n_median=("_critical_n", "median"),
        )
        .reset_index()
    )

summary_path = OUT / "06_reconstructed_current_gate_summary_by_dataset.csv"
summary.to_csv(summary_path, index=False)

print_block("RECONSTRUCTED CURRENT-GATE SUMMARY")
if summary.empty:
    print("No summary because selected is empty.")
else:
    print(summary.to_string(index=False))
print("Summary saved:", summary_path)


# =============================================================================
# 7. POSSIBLE STRICT 0.25 COLUMNS
# =============================================================================

strict_keywords = [
    "intervention",
    "miss",
    "strict",
    "025",
    "0_25",
    "lower_gt",
    "l_gt",
    "point",
    "yhat",
    "false_safe_025",
    "fsr_025",
    "urgent_miss",
]

strict_candidates = [
    c for c in candidate_pool.columns
    if any(k in norm_col(c) for k in strict_keywords)
]

strict_cols_path = OUT / "07_possible_strict_025_columns_in_candidate_pool.csv"
pd.DataFrame({"possible_strict_025_column": strict_candidates}).to_csv(strict_cols_path, index=False)

print_block("COLUMNS THAT MAY ALREADY CONTAIN STRICT 0.25 DIAGNOSTICS")
if strict_candidates:
    for c in strict_candidates:
        print(" -", c)
else:
    print("NONE FOUND")
print("Saved:", strict_cols_path)


# =============================================================================
# 8. ORIGINAL VS RECONSTRUCTED COMPARISON
# =============================================================================

comparison_path = OUT / "08_original_vs_reconstructed_selector_comparison.csv"

if len(original_selected) > 0 and len(selected) > 0:
    orig = original_selected.copy()
    orig_cols = list(orig.columns)

    odataset_col = find_col(orig_cols, ["dataset", "dataset_name"], required=True)
    oseed_col = find_col(orig_cols, ["seed", "random_seed", "split_seed", "seed_from_folder"], required=False)
    ocandidate_col = find_col(orig_cols, ["candidate", "candidate_id", "model", "model_name", "protocol", "config"], required=False)

    orig["_dataset"] = orig[odataset_col].astype(str)
    orig["_seed"] = orig[oseed_col] if oseed_col and oseed_col in orig.columns else orig["seed_from_folder"]

    if ocandidate_col and candidate_col and candidate_col in selected.columns:
        orig["_candidate_key"] = orig[ocandidate_col].astype(str)
        selected["_candidate_key"] = selected[candidate_col].astype(str)

        comp = orig[["_dataset", "_seed", "_candidate_key"]].merge(
            selected[["_dataset", "_seed", "_candidate_key"]],
            on=["_dataset", "_seed"],
            how="outer",
            suffixes=("_original", "_reconstructed"),
        )
        comp["same_candidate"] = comp["_candidate_key_original"] == comp["_candidate_key_reconstructed"]
    else:
        comp = orig[["_dataset", "_seed"]].drop_duplicates().merge(
            selected[["_dataset", "_seed"]].drop_duplicates(),
            on=["_dataset", "_seed"],
            how="outer",
            indicator=True,
        )

    comp.to_csv(comparison_path, index=False)
    print_block("ORIGINAL VS RECONSTRUCTED COMPARISON")
    print("Saved:", comparison_path)
    print(comp.head(50).to_string(index=False))
else:
    pd.DataFrame().to_csv(comparison_path, index=False)
    print_block("ORIGINAL VS RECONSTRUCTED COMPARISON")
    print("Skipped because original_selected or selected is empty.")
    print("Saved empty:", comparison_path)


# =============================================================================
# 9. status
# =============================================================================

print_block("LITERAL-GATE AUDIT status")

expected_rows = 5 * 30
observed_selected = len(selected)
observed_unresolved = len(unresolved)

print("Expected selected rows:", expected_rows)
print("Observed reconstructed selected rows:", observed_selected)
print("Observed unresolved rows:", observed_unresolved)

if observed_selected == expected_rows and observed_unresolved == 0:
    print("CURRENT WRITTEN GATE: GREEN at metric-summary level.")
else:
    print("CURRENT WRITTEN GATE: RED/YELLOW â€” missing or unresolved rows.")

if not summary.empty:
    print("\nDataset pass summary:")
    for _, r in summary.iterrows():
        ok = (
            r["seeds"] == 30
            and r["coverage_min"] >= 0.90
            and r["riskcov_min"] >= 0.90
            and r["fsr_max"] == 0
            and r["underwarning_max"] <= 0.05
        )
        print(f"{r['_dataset']}: {'PASS' if ok else 'FAIL'}")

print("\nImportant:")
print("This package verifies the current written gate from summary metrics only.")
print("It does NOT verify strict intervention miss at 0.25 unless strict columns were found above.")
print("If no strict columns were found, the next script must compute strict 0.25 from candidate-level row predictions.")
print("=" * 90)

