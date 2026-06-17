
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

import os
# Public release script: 11_reselect_strict_interval_candidates.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

# -*- coding: utf-8 -*-




import json
import math
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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
STRICT_SELECTED_AUDIT_DIR = OUTPUT_ROOT / "Q1_strict-interval_LITERAL_SELECTED_ROW_AUDIT_LOCAL"


CANDIDATE_POOL_FILE = EXACT_AUDIT_DIR / "01_all_candidate_pool_normalized_metrics_combined.csv"
SEED_MAP_FILE = STRICT_SELECTED_AUDIT_DIR / "01_seed_to_run_folder_map.csv"

OUT = OUTPUT_ROOT / "Q1_strict-interval_CANDIDATE_POOL_FEASIBILITY_RESELECTOR"
OUT.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUT / "CACHE_PER_SEED_DATASET_ROW_METRICS"
CACHE_DIR.mkdir(exist_ok=True)

CHUNKSIZE = 250_000
USE_CACHE = True

COVERAGE_GATE = 0.90
RISKCOV_GATE = 0.90
UNDERWARNING_GATE = 0.05
EPS = 1e-12


# =============================================================================
# HELPERS
# =============================================================================

def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def norm_value(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


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


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


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
    return slug(d)


def infer_candidate_col(cols: List[str]) -> Optional[str]:
    col = find_col(cols, ["_candidate_id", "candidate_id", "candidate", "selected_candidate", "protocol_name", "protocol"], required=False)
    if col:
        return col
    return find_col(cols, ["model_name", "model", "config"], required=False)


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


def print_block(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def build_seed_map() -> Dict[int, Path]:
    seed_run_map: Dict[int, Path] = {}

    if SEED_MAP_FILE.exists():
        df = read_csv_safe(SEED_MAP_FILE)
        if {"seed", "run_dir"}.issubset(df.columns):
            for _, r in df.iterrows():
                try:
                    s = int(r["seed"])
                    p = Path(str(r["run_dir"]))
                    if p.exists():
                        seed_run_map[s] = p
                except Exception:
                    pass

    # Fallback via discovered_seed_runs.csv
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

    # Fallback scan run_config.json
    for run_dir in sorted(RERUN_ROOT.glob("Q1_SINGLE_FILE_REPRODUCTION_RUN_*")):
        if not run_dir.is_dir():
            continue

        seed_found = None
        jsons = list(run_dir.glob("*.json")) + list(run_dir.rglob("run_config.json"))

        for jp in jsons:
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                for key in ["seed", "random_seed", "split_seed", "master_seed"]:
                    if key in obj and str(obj[key]).isdigit():
                        seed_found = int(obj[key])
                        break
                if seed_found is not None:
                    break
            except Exception:
                continue

        if seed_found is not None and seed_found not in seed_run_map:
            seed_run_map[seed_found] = run_dir

    return seed_run_map


# =============================================================================
# ROW-LEVEL CANDIDATE METRIC SCAN
# =============================================================================

def scan_pred_df_for_candidate_metrics(
    pred_file: Path,
    candidate_ids: List[str],
    dataset: str,
    seed: int,
) -> pd.DataFrame:
    

    wanted = set(norm_value(x) for x in candidate_ids)
    wanted_no_space = {re.sub(r"\s+", "", x) for x in wanted}

    head = pd.read_csv(pred_file, nrows=5)
    cols = list(head.columns)

    cand_col = infer_candidate_col(cols)
    if cand_col is None:
        raise KeyError(f"No candidate column found in {pred_file}. Columns:\n{cols}")

    y_col, yp_col, lo_col, up_col = infer_y_cols(cols)

    acc: Dict[str, Dict[str, float]] = {}
    found_raw_values: Dict[str, str] = {}

    chunk_i = 0
    for chunk in pd.read_csv(pred_file, chunksize=CHUNKSIZE, low_memory=False):
        chunk_i += 1

        cvals_raw = chunk[cand_col].astype(str).map(norm_value)
        cvals_nospace = cvals_raw.map(lambda x: re.sub(r"\s+", "", x))

        mask = cvals_raw.isin(wanted)
        if not mask.any():
            mask = cvals_nospace.isin(wanted_no_space)

        if not mask.any():
            continue

        sub = chunk.loc[mask, [cand_col, y_col, yp_col, lo_col, up_col]].copy()
        sub["_candidate_key"] = sub[cand_col].astype(str).map(norm_value)

        # If exact-normalized candidate key not in wanted, use no-space mapping back.
        not_wanted = ~sub["_candidate_key"].isin(wanted)
        if not_wanted.any():
            ns = sub.loc[not_wanted, "_candidate_key"].map(lambda x: re.sub(r"\s+", "", x))
            # map no-space target to original wanted key
            ns_to_wanted = {re.sub(r"\s+", "", w): w for w in wanted}
            sub.loc[not_wanted, "_candidate_key"] = ns.map(ns_to_wanted).fillna(sub.loc[not_wanted, "_candidate_key"])

        y = pd.to_numeric(sub[y_col], errors="coerce")
        yp = pd.to_numeric(sub[yp_col], errors="coerce")
        lo = pd.to_numeric(sub[lo_col], errors="coerce")
        up = pd.to_numeric(sub[up_col], errors="coerce")
        valid = y.notna() & yp.notna() & lo.notna() & up.notna()

        if not valid.any():
            continue

        tmp = pd.DataFrame({
            "candidate": sub.loc[valid, "_candidate_key"].astype(str).values,
            "y": y.loc[valid].astype(float).values,
            "yp": yp.loc[valid].astype(float).values,
            "lo": lo.loc[valid].astype(float).values,
            "up": up.loc[valid].astype(float).values,
        })

        tmp["err"] = tmp["yp"] - tmp["y"]
        tmp["abs_err"] = tmp["err"].abs()
        tmp["sq_err"] = tmp["err"] ** 2
        tmp["covered"] = (tmp["y"] >= tmp["lo"]) & (tmp["y"] <= tmp["up"])
        tmp["urgent"] = tmp["y"] <= 0.25
        tmp["critical"] = tmp["y"] <= 0.10
        tmp["lower_miss"] = tmp["y"] < tmp["lo"]
        tmp["upper_miss"] = tmp["y"] > tmp["up"]
        tmp["width"] = tmp["up"] - tmp["lo"]

        tmp["urgent_covered"] = tmp["urgent"] & tmp["covered"]
        tmp["critical_covered"] = tmp["critical"] & tmp["covered"]

        tmp["fsr050_interval"] = tmp["urgent"] & (tmp["lo"] > 0.50)
        tmp["fsr050_point"] = tmp["urgent"] & (tmp["yp"] > 0.50)
        tmp["fsr050_point_or_lower"] = tmp["urgent"] & ((tmp["yp"] > 0.50) | (tmp["lo"] > 0.50))

        tmp["miss025_interval"] = tmp["urgent"] & (tmp["lo"] > 0.25)
        tmp["miss025_point"] = tmp["urgent"] & (tmp["yp"] > 0.25)
        tmp["miss025_point_or_lower"] = tmp["urgent"] & ((tmp["yp"] > 0.25) | (tmp["lo"] > 0.25))

        tmp["under_urgent"] = tmp["lower_miss"] & tmp["urgent"]
        tmp["under_critical"] = tmp["lower_miss"] & tmp["critical"]

        g = tmp.groupby("candidate", dropna=False)

        ag = g.agg(
            n=("y", "size"),
            sum_y=("y", "sum"),
            sum_y2=("y", lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2))),
            sum_sq_err=("sq_err", "sum"),
            sum_abs_err=("abs_err", "sum"),
            covered=("covered", "sum"),
            urgent_n=("urgent", "sum"),
            critical_n=("critical", "sum"),
            urgent_covered=("urgent_covered", "sum"),
            critical_covered=("critical_covered", "sum"),
            fsr050_interval_n=("fsr050_interval", "sum"),
            fsr050_point_n=("fsr050_point", "sum"),
            fsr050_point_or_lower_n=("fsr050_point_or_lower", "sum"),
            miss025_interval_n=("miss025_interval", "sum"),
            miss025_point_n=("miss025_point", "sum"),
            miss025_point_or_lower_n=("miss025_point_or_lower", "sum"),
            lower_miss_n=("lower_miss", "sum"),
            upper_miss_n=("upper_miss", "sum"),
            under_urgent_n=("under_urgent", "sum"),
            under_critical_n=("under_critical", "sum"),
            width_sum=("width", "sum"),
        )

        for cand, r in ag.iterrows():
            d = acc.setdefault(str(cand), {
                "n": 0,
                "sum_y": 0.0,
                "sum_y2": 0.0,
                "sum_sq_err": 0.0,
                "sum_abs_err": 0.0,
                "covered": 0.0,
                "urgent_n": 0.0,
                "critical_n": 0.0,
                "urgent_covered": 0.0,
                "critical_covered": 0.0,
                "fsr050_interval_n": 0.0,
                "fsr050_point_n": 0.0,
                "fsr050_point_or_lower_n": 0.0,
                "miss025_interval_n": 0.0,
                "miss025_point_n": 0.0,
                "miss025_point_or_lower_n": 0.0,
                "lower_miss_n": 0.0,
                "upper_miss_n": 0.0,
                "under_urgent_n": 0.0,
                "under_critical_n": 0.0,
                "width_sum": 0.0,
            })
            for k in d.keys():
                d[k] += float(r[k])

        if chunk_i % 10 == 0:
            print(f"  scanned chunk {chunk_i} of {pred_file.name}; candidates found so far={len(acc)}")

    rows = []
    for cand, d in acc.items():
        n = d["n"]
        urgent_n = d["urgent_n"]
        critical_n = d["critical_n"]

        sst = d["sum_y2"] - (d["sum_y"] ** 2 / n) if n > 0 else np.nan
        r2 = 1.0 - d["sum_sq_err"] / sst if n > 1 and sst and sst > 0 else np.nan

        def div(num, den):
            if den is None or den == 0 or np.isnan(den):
                return np.nan
            return float(num / den)

        rows.append({
            "dataset": dataset,
            "seed": int(seed),
            "_candidate_key": cand,
            "row_n": int(n),
            "row_rmse": float(np.sqrt(d["sum_sq_err"] / n)) if n else np.nan,
            "row_mae": div(d["sum_abs_err"], n),
            "row_r2": float(r2) if np.isfinite(r2) else np.nan,
            "row_coverage": div(d["covered"], n),
            "row_riskcov_y_le_025": div(d["urgent_covered"], urgent_n),
            "row_critcov_y_le_010": div(d["critical_covered"], critical_n),
            "row_urgent_n": int(urgent_n),
            "row_critical_n": int(critical_n),
            "row_fsr050_interval_lower_only": div(d["fsr050_interval_n"], urgent_n),
            "row_fsr050_point_only": div(d["fsr050_point_n"], urgent_n),
            "row_fsr050_point_or_lower": div(d["fsr050_point_or_lower_n"], urgent_n),
            "row_miss025_interval_lower_only": div(d["miss025_interval_n"], urgent_n),
            "row_miss025_point_only": div(d["miss025_point_n"], urgent_n),
            "row_miss025_point_or_lower": div(d["miss025_point_or_lower_n"], urgent_n),
            "row_under_all": div(d["lower_miss_n"], n),
            "row_under_urgent_only": div(d["under_urgent_n"], urgent_n),
            "row_under_critical_only": div(d["under_critical_n"], critical_n),
            "row_upper_all": div(d["upper_miss_n"], n),
            "row_width_mean": div(d["width_sum"], n),
            "pred_file": str(pred_file),
        })

    out = pd.DataFrame(rows)
    return out


# =============================================================================
# MAIN
# =============================================================================

print_block("STRICT 0.25 CANDIDATE-POOL FEASIBILITY RESELECTOR")
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("LITERAL_DIR:", LITERAL_DIR)
print("CANDIDATE_POOL_FILE:", CANDIDATE_POOL_FILE)
print("SEED_MAP_FILE:", SEED_MAP_FILE)
print("RERUN_ROOT:", RERUN_ROOT)
print("OUT:", OUT)

if not CANDIDATE_POOL_FILE.exists():
    raise FileNotFoundError(
        f"Missing candidate pool file:\n{CANDIDATE_POOL_FILE}\n"
        "Run Q1_LITERAL_GATE_EXACT_AUDIT_LOCAL.py first."
    )

candidate_pool = read_csv_safe(CANDIDATE_POOL_FILE)
cols = list(candidate_pool.columns)

dataset_col = find_col(cols, ["dataset", "_dataset"], required=True)
seed_col = find_col(cols, ["seed", "_seed", "seed_from_folder"], required=True)
cand_col = find_col(cols, ["_candidate_id", "candidate_id", "candidate"], required=True)

coverage_col = find_col(cols, ["_coverage", "coverage"], required=True)
riskcov_col = find_col(cols, ["_risk_coverage", "riskcov", "risk_coverage"], required=True)
fsr_col = find_col(cols, ["_false_safe", "false_safe", "fsr"], required=True)
under_col = find_col(cols, ["_underwarning", "underwarning"], required=True)
width_col = find_col(cols, ["_mean_width", "mean_width", "width"], required=False)
rmse_col = find_col(cols, ["_rmse", "rmse"], required=False)
mae_col = find_col(cols, ["_mae", "mae"], required=False)

candidate_pool["_dataset_key"] = candidate_pool[dataset_col].astype(str)
candidate_pool["_seed_key"] = pd.to_numeric(candidate_pool[seed_col], errors="coerce").astype("Int64")
candidate_pool["_candidate_key"] = candidate_pool[cand_col].astype(str).map(norm_value)

candidate_pool["_stored_coverage"] = pd.to_numeric(candidate_pool[coverage_col], errors="coerce")
candidate_pool["_stored_riskcov"] = pd.to_numeric(candidate_pool[riskcov_col], errors="coerce")
candidate_pool["_stored_false_safe"] = pd.to_numeric(candidate_pool[fsr_col], errors="coerce")
candidate_pool["_stored_underwarning"] = pd.to_numeric(candidate_pool[under_col], errors="coerce")
candidate_pool["_stored_width"] = pd.to_numeric(candidate_pool[width_col], errors="coerce") if width_col else np.nan
candidate_pool["_stored_rmse"] = pd.to_numeric(candidate_pool[rmse_col], errors="coerce") if rmse_col else np.nan
candidate_pool["_stored_mae"] = pd.to_numeric(candidate_pool[mae_col], errors="coerce") if mae_col else np.nan

candidate_pool["_stored_current_gate"] = (
    (candidate_pool["_stored_coverage"] >= COVERAGE_GATE)
    & (candidate_pool["_stored_riskcov"] >= RISKCOV_GATE)
    & (candidate_pool["_stored_false_safe"].abs() <= EPS)
    & (candidate_pool["_stored_underwarning"] <= UNDERWARNING_GATE)
)

print("Candidate pool rows:", len(candidate_pool))
print("Stored current-gate candidates:", int(candidate_pool["_stored_current_gate"].sum()))

seed_run_map = build_seed_map()
print("Seed-run map rows:", len(seed_run_map))
pd.DataFrame([{"seed": k, "run_dir": str(v)} for k, v in sorted(seed_run_map.items())]).to_csv(
    OUT / "00_seed_to_run_folder_map_used.csv",
    index=False
)

# For definitive feasibility, scan all candidate IDs in the candidate pool.
# If this is too slow, it can be changed later, but first run should be definitive.
metric_frames = []
scan_log = []

groups = list(candidate_pool.groupby(["_dataset_key", "_seed_key"], dropna=False))
print("Seed-dataset groups to scan:", len(groups))

t0 = time.time()

for gi, ((dataset, seed_val), sub) in enumerate(groups, start=1):
    if pd.isna(seed_val):
        scan_log.append({
            "dataset": dataset,
            "seed": "",
            "status": "SKIP_NO_SEED",
            "n_candidate_ids": len(sub),
        })
        continue

    seed = int(seed_val)
    ds_slug = dataset_to_pred_stem(dataset)
    cache_file = CACHE_DIR / f"{ds_slug}__seed_{seed}__row_candidate_metrics.csv"

    print_block(f"[{gi}/{len(groups)}] {dataset} seed={seed}")
    print("candidate IDs:", sub["_candidate_key"].nunique())

    if USE_CACHE and cache_file.exists():
        print("Using cache:", cache_file)
        metrics = read_csv_safe(cache_file)
        metric_frames.append(metrics)
        scan_log.append({
            "dataset": dataset,
            "seed": seed,
            "status": "CACHE_USED",
            "n_candidate_ids": sub["_candidate_key"].nunique(),
            "n_metrics": len(metrics),
            "cache_file": str(cache_file),
        })
        continue

    run_dir = seed_run_map.get(seed)
    if run_dir is None or not Path(run_dir).exists():
        print("No run dir for seed.")
        scan_log.append({
            "dataset": dataset,
            "seed": seed,
            "status": "NO_RUN_DIR",
            "n_candidate_ids": sub["_candidate_key"].nunique(),
        })
        continue

    pred_file = locate_pred_file(Path(run_dir), dataset)
    if pred_file is None:
        print("No pred_df file found.")
        scan_log.append({
            "dataset": dataset,
            "seed": seed,
            "status": "NO_PRED_DF",
            "run_dir": str(run_dir),
            "n_candidate_ids": sub["_candidate_key"].nunique(),
        })
        continue

    print("pred_file:", pred_file)

    try:
        metrics = scan_pred_df_for_candidate_metrics(
            pred_file=pred_file,
            candidate_ids=sorted(sub["_candidate_key"].dropna().unique().tolist()),
            dataset=dataset,
            seed=seed,
        )
        metrics.to_csv(cache_file, index=False)
        metric_frames.append(metrics)

        print("metrics rows:", len(metrics))
        scan_log.append({
            "dataset": dataset,
            "seed": seed,
            "status": "SCANNED",
            "run_dir": str(run_dir),
            "pred_file": str(pred_file),
            "n_candidate_ids": sub["_candidate_key"].nunique(),
            "n_metrics": len(metrics),
            "cache_file": str(cache_file),
        })

    except Exception as e:
        print("ERROR:", e)
        scan_log.append({
            "dataset": dataset,
            "seed": seed,
            "status": "ERROR",
            "run_dir": str(run_dir),
            "pred_file": str(pred_file),
            "n_candidate_ids": sub["_candidate_key"].nunique(),
            "error": str(e),
            "traceback": traceback.format_exc(),
        })

scan_log_df = pd.DataFrame(scan_log)
scan_log_df.to_csv(OUT / "01_scan_log.csv", index=False)

if metric_frames:
    row_metrics = pd.concat(metric_frames, ignore_index=True)
else:
    row_metrics = pd.DataFrame()

row_metrics_path = OUT / "02_all_candidate_row_level_strict-interval_metrics.csv"
row_metrics.to_csv(row_metrics_path, index=False)

print_block("ROW-LEVEL CANDIDATE METRICS COMPLETE")
print("Metrics rows:", len(row_metrics))
print("Saved:", row_metrics_path)
print("Elapsed minutes:", round((time.time() - t0) / 60, 2))

if row_metrics.empty:
    raise RuntimeError("No row-level metrics were computed. Check 01_scan_log.csv.")

# Merge with candidate pool.
merged = candidate_pool.merge(
    row_metrics,
    left_on=["_dataset_key", "_seed_key", "_candidate_key"],
    right_on=["dataset", "seed", "_candidate_key"],
    how="left",
    suffixes=("", "_row"),
)

merged["_row_metric_found"] = merged["row_n"].notna()
merged_path = OUT / "03_candidate_pool_merged_with_row_strict-interval_metrics.csv"
merged.to_csv(merged_path, index=False)

print("Merged rows:", len(merged))
print("Rows with row metrics:", int(merged["_row_metric_found"].sum()))
print("Merged saved:", merged_path)


# =============================================================================
# GATES
# =============================================================================

m = merged.copy()

def is_zero(s):
    return pd.to_numeric(s, errors="coerce").fillna(np.inf).abs() <= EPS

m["_row_base_coverage_risk"] = (
    (m["row_coverage"] >= COVERAGE_GATE)
    & (m["row_riskcov_y_le_025"] >= RISKCOV_GATE)
)

m["gate_current_literal_row_under"] = (
    m["_row_base_coverage_risk"]
    & is_zero(m["row_fsr050_interval_lower_only"])
    & (m["row_under_all"] <= UNDERWARNING_GATE)
)

m["gate_current_literal_stored_under"] = (
    m["_row_base_coverage_risk"]
    & is_zero(m["row_fsr050_interval_lower_only"])
    & (m["_stored_underwarning"] <= UNDERWARNING_GATE)
)

m["gate_strict-interval_interval_stored_under"] = (
    m["_row_base_coverage_risk"]
    & is_zero(m["row_fsr050_interval_lower_only"])
    & is_zero(m["row_miss025_interval_lower_only"])
    & (m["_stored_underwarning"] <= UNDERWARNING_GATE)
)

m["gate_strict-interval_interval_row_under"] = (
    m["_row_base_coverage_risk"]
    & is_zero(m["row_fsr050_interval_lower_only"])
    & is_zero(m["row_miss025_interval_lower_only"])
    & (m["row_under_all"] <= UNDERWARNING_GATE)
)

m["gate_strict-interval_interval_no_under"] = (
    m["_row_base_coverage_risk"]
    & is_zero(m["row_fsr050_interval_lower_only"])
    & is_zero(m["row_miss025_interval_lower_only"])
)

m["gate_combined_combined025_row_under"] = (
    m["_row_base_coverage_risk"]
    & is_zero(m["row_fsr050_interval_lower_only"])
    & is_zero(m["row_miss025_point_or_lower"])
    & (m["row_under_all"] <= UNDERWARNING_GATE)
)

m["gate_combined_combined025_stored_under"] = (
    m["_row_base_coverage_risk"]
    & is_zero(m["row_fsr050_interval_lower_only"])
    & is_zero(m["row_miss025_point_or_lower"])
    & (m["_stored_underwarning"] <= UNDERWARNING_GATE)
)

flag_path = OUT / "04_candidate_pool_all_gate_flags.csv"
m.to_csv(flag_path, index=False)
print("Gate flags saved:", flag_path)


# =============================================================================
# DETERMINISTIC SELECTION PER GATE
# =============================================================================

gate_cols = [
    "gate_current_literal_stored_under",
    "gate_current_literal_row_under",
    "gate_strict-interval_interval_stored_under",
    "gate_strict-interval_interval_row_under",
    "gate_strict-interval_interval_no_under",
    "gate_combined_combined025_stored_under",
    "gate_combined_combined025_row_under",
]

selection_summaries = []

def deterministic_select(df: pd.DataFrame, gate_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    unresolved_rows = []
    availability_rows = []

    for (dataset, seed), sub in df.groupby(["_dataset_key", "_seed_key"], dropna=False):
        eligible = sub[sub[gate_col].fillna(False)].copy()

        availability_rows.append({
            "gate": gate_col,
            "dataset": dataset,
            "seed": seed,
            "n_candidates": len(sub),
            "n_row_metrics_found": int(sub["_row_metric_found"].sum()),
            "n_eligible": len(eligible),
        })

        if eligible.empty:
            unresolved_rows.append({
                "gate": gate_col,
                "dataset": dataset,
                "seed": seed,
                "n_candidates": len(sub),
                "n_row_metrics_found": int(sub["_row_metric_found"].sum()),
                "n_eligible": 0,
                "best_row_coverage": sub["row_coverage"].max(),
                "best_row_riskcov": sub["row_riskcov_y_le_025"].max(),
                "min_row_fsr050_interval": sub["row_fsr050_interval_lower_only"].min(),
                "min_row_miss025_interval": sub["row_miss025_interval_lower_only"].min(),
                "min_row_under_all": sub["row_under_all"].min(),
                "min_stored_underwarning": sub["_stored_underwarning"].min(),
            })
            continue

        eligible["_rank_width"] = eligible["row_width_mean"].fillna(eligible["_stored_width"]).fillna(np.inf)
        eligible["_rank_rmse"] = eligible["row_rmse"].fillna(eligible["_stored_rmse"]).fillna(np.inf)
        eligible["_rank_mae"] = eligible["row_mae"].fillna(eligible["_stored_mae"]).fillna(np.inf)

        eligible = eligible.sort_values(
            ["_rank_width", "_rank_rmse", "_rank_mae", "_candidate_key"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )

        row = eligible.iloc[0].to_dict()
        row["_selected_gate"] = gate_col
        row["_n_eligible_for_gate"] = len(eligible)
        selected_rows.append(row)

    return pd.DataFrame(selected_rows), pd.DataFrame(unresolved_rows), pd.DataFrame(availability_rows)


all_availability = []

for gate in gate_cols:
    sel, unresolved, avail = deterministic_select(m, gate)
    all_availability.append(avail)

    sel_path = OUT / f"05_SELECTED__{gate}.csv"
    unres_path = OUT / f"06_UNRESOLVED__{gate}.csv"
    summary_path = OUT / f"07_SUMMARY__{gate}.csv"

    sel.to_csv(sel_path, index=False)
    unresolved.to_csv(unres_path, index=False)

    if not sel.empty:
        ds_summary = (
            sel.groupby("_dataset_key")
            .agg(
                seeds=("_seed_key", "nunique"),
                selected_rows=("_seed_key", "count"),
                coverage_min=("row_coverage", "min"),
                coverage_median=("row_coverage", "median"),
                riskcov_min=("row_riskcov_y_le_025", "min"),
                riskcov_median=("row_riskcov_y_le_025", "median"),
                critcov_min=("row_critcov_y_le_010", "min"),
                critcov_median=("row_critcov_y_le_010", "median"),
                critical_n_min=("row_critical_n", "min"),
                critical_n_median=("row_critical_n", "median"),
                fsr050_interval_max=("row_fsr050_interval_lower_only", "max"),
                miss025_interval_max=("row_miss025_interval_lower_only", "max"),
                miss025_point_max=("row_miss025_point_only", "max"),
                miss025_combined_max=("row_miss025_point_or_lower", "max"),
                row_under_all_max=("row_under_all", "max"),
                stored_underwarning_max=("_stored_underwarning", "max"),
                width_median=("row_width_mean", "median"),
                rmse_median=("row_rmse", "median"),
                mae_median=("row_mae", "median"),
                r2_median=("row_r2", "median"),
            )
            .reset_index()
        )
    else:
        ds_summary = pd.DataFrame()

    ds_summary.to_csv(summary_path, index=False)

    selection_summaries.append({
        "gate": gate,
        "selected_rows": len(sel),
        "unresolved_rows": len(unresolved),
        "all_150_selected": len(sel) == 150 and len(unresolved) == 0,
        "selected_file": str(sel_path),
        "unresolved_file": str(unres_path),
        "summary_file": str(summary_path),
    })

availability_df = pd.concat(all_availability, ignore_index=True) if all_availability else pd.DataFrame()
availability_df.to_csv(OUT / "08_gate_candidate_availability_by_seed_dataset.csv", index=False)

selection_summary_df = pd.DataFrame(selection_summaries)
selection_summary_path = OUT / "09_gate_selection_summary.csv"
selection_summary_df.to_csv(selection_summary_path, index=False)


# =============================================================================
# PRINT FINAL DECISION REPORT
# =============================================================================

print_block("GATE SELECTION SUMMARY")
print(selection_summary_df.to_string(index=False))

print_block("DATASET SUMMARIES FOR KEY GATES")

for gate in [
    "gate_strict-interval_interval_stored_under",
    "gate_strict-interval_interval_row_under",
    "gate_strict-interval_interval_no_under",
    "gate_combined_combined025_stored_under",
]:
    p = OUT / f"07_SUMMARY__{gate}.csv"
    print("\n---", gate, "---")
    if p.exists():
        df = read_csv_safe(p)
        if df.empty:
            print("EMPTY")
        else:
            print(df.to_string(index=False))
    else:
        print("MISSING")

print_block("UNRESOLVED COUNTS BY DATASET FOR KEY GATES")
for gate in [
    "gate_strict-interval_interval_stored_under",
    "gate_strict-interval_interval_row_under",
    "gate_strict-interval_interval_no_under",
    "gate_combined_combined025_stored_under",
]:
    p = OUT / f"06_UNRESOLVED__{gate}.csv"
    print("\n---", gate, "---")
    if p.exists():
        df = read_csv_safe(p)
        if df.empty:
            print("NONE")
        else:
            print(df.groupby("dataset").size().to_string())
    else:
        print("MISSING")

print_block("status")
def status_for(gate):
    row = selection_summary_df[selection_summary_df["gate"].eq(gate)]
    if row.empty:
        return "MISSING"
    r = row.iloc[0]
    return "GREEN" if bool(r["all_150_selected"]) else f"RED/YELLOW selected={int(r['selected_rows'])}, unresolved={int(r['unresolved_rows'])}"

print("Strict interval 0.25 + stored underwarning:", status_for("gate_strict-interval_interval_stored_under"))
print("Strict interval 0.25 + row under_all:", status_for("gate_strict-interval_interval_row_under"))
print("Strict interval 0.25 without underwarning:", status_for("gate_strict-interval_interval_no_under"))
print("combined combined 0.25 + stored underwarning:", status_for("gate_combined_combined025_stored_under"))
print("combined combined 0.25 + row under_all:", status_for("gate_combined_combined025_row_under"))

print("\nInterpretation:")
print("1) If strict interval 0.25 + stored underwarning is GREEN, the paper is strongly salvageable with interval-based operational safety.")
print("2) If only strict interval 0.25 without underwarning is GREEN, underwarning definition/gate is the remaining blocker.")
print("3) If strict interval 0.25 is RED/YELLOW, existing candidates are insufficient and we need targeted recalibration/rerun.")
print("4) combined combined point-or-interval is expected to be hard to satisfy; failure there does not automatically kill interval-based safety.")
print("=" * 90)

