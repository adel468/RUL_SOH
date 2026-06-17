
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

# Public release script: 07_consolidate_repeated_seed_evidence.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

#!/usr/bin/env python
# -*- coding: utf-8 -*-




import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DATASETS = ["NASA C-MAPSS", "NASA Battery", "PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]
PRIMARY_SEED = 42
ROBUSTNESS_MASTER_SEED = 20260605
ROBUSTNESS_N_SEEDS = 30
FINAL_METRICS_REL = Path("02_TABLES") / "SINGLE_FILE_SELECTED_CANDIDATE_METRICS.csv"

NN_REQUIRED_RUNS = {
}

METRIC_COLS = [
    "empirical_coverage",
    "urgent_critical_coverage",
    "interval_false_safe_rate",
    "interval_underwarning_rate",
    "mean_interval_width",
    "RMSE",
    "MAE",
    "R2",
]


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


def slug(x: str) -> str:
    return norm(x).replace("nasa_", "")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv_safe(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


def generate_robustness_seeds(master_seed: int, n: int) -> List[int]:
    rng = np.random.default_rng(master_seed)
    seeds = set()
    while len(seeds) < n:
        seeds.update(int(x) for x in rng.integers(1000, 2_147_000_000, size=n * 3))
    return sorted(list(seeds))[:n]


def final_metrics_path(run_dir: Path) -> Path:
    return run_dir / FINAL_METRICS_REL


def has_final_metrics(run_dir: Path) -> bool:
    p = final_metrics_path(run_dir)
    return p.exists() and p.stat().st_size > 1000


def read_run_config(run_dir: Path) -> Dict[str, Any]:
    for p in [run_dir / "00_RUN_MANIFEST" / "run_config.json", run_dir / "run_config.json"]:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def detect_seed(run_dir: Path) -> Optional[int]:
    cfg = read_run_config(run_dir)
    for key in ["seed", "random_seed"]:
        if key in cfg:
            try:
                return int(cfg[key])
            except Exception:
                pass
    m = re.search(r"seed[_\-]?(\d+)", run_dir.name.lower())
    if m:
        return int(m.group(1))
    if run_dir.name == PRIMARY_CLEAN_RUN_NAME:
        return PRIMARY_SEED
    return None


def add_seed_metadata(run_dir: Path, seed: int, note: str) -> None:
    cfg_path = run_dir / "00_RUN_MANIFEST" / "run_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    cfg["seed"] = int(seed)
    cfg["seed_metadata_verified_by_consolidated_controller"] = True
    cfg["seed_metadata_note"] = note
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_runs(output_root: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(output_root.glob("Q1_SINGLE_FILE_REPRODUCTION_RUN_*")):
        if not run.is_dir() or "_junk" in run.name.lower():
            continue
        rows.append({
            "run_dir": str(run),
            "run_name": run.name,
            "seed": detect_seed(run),
            "has_final_metrics": has_final_metrics(run),
            "is_primary_locked_run": run.name == PRIMARY_CLEAN_RUN_NAME,
            "modified_time": datetime.fromtimestamp(run.stat().st_mtime).isoformat(),
        })
    return pd.DataFrame(rows)


def load_or_create_seed_list(out_dir: Path, master_seed: int, n: int) -> List[int]:
    p = out_dir / "00_RUN_MANIFEST" / "robustness_seed_list_master20260605.csv"
    if p.exists():
        df = pd.read_csv(p)
        return [int(x) for x in df["seed"].tolist()]
    seeds = generate_robustness_seeds(master_seed, n)
    save_csv(pd.DataFrame({
        "seed_index": range(1, len(seeds) + 1),
        "seed": seeds,
        "master_seed": master_seed,
        "pre_specified_before_run": True,
        "role": "robustness_distribution_seed_not_primary_locked_seed",
    }), p)
    return seeds


def make_missing_seed_plan(output_root: Path, out_dir: Path, phase1_run: Path,
                           reproducer_script: Path, code_root: Path, seeds: List[int]) -> pd.DataFrame:
    runs = discover_runs(output_root)
    observed = set(int(x) for x in runs["seed"].dropna().tolist()) if not runs.empty and "seed" in runs else set()
    missing = [s for s in seeds if s not in observed]
    rows = []
    for s in missing:
        stage_command_text = (
            f' {reproducer_script.as_posix()} --work_dir {code_root.as_posix()} '
            f'--args "--phase1_run {phase1_run.as_posix()} --mode full --seed {s}"'
        )
        cli = f'python "{str(reproducer_script)}" --phase1_run "{str(phase1_run)}" --mode full --seed {s}'
        rows.append({
            "seed": s,
            "status": "MISSING",
            "stage_command_text": stage_command_text,
            "cli_command": cli,
            "note": "Run this seed once. Do not delete or hide failures. Rerun controller afterward.",
        })
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "02_ROBUSTNESS_SEED_PLAN" / "missing_robustness_seed_commands.csv")
    return df


def run_missing(seed_plan: pd.DataFrame, out_dir: Path, max_to_run: int) -> pd.DataFrame:
    if seed_plan.empty:
        df = pd.DataFrame([{"status": "NO_MISSING_SEEDS"}])
        save_csv(df, out_dir / "07_EXECUTION_LOGS" / "run_missing_seeds_execution_log.csv")
        return df
    pending_runs = seed_plan.copy()
    if max_to_run and max_to_run > 0:
        pending_runs = pending_runs.head(max_to_run)
    rows = []
    for _, r in pending_runs.iterrows():
        seed = int(r["seed"])
        cmd = str(r["cli_command"])
        start = datetime.now()
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            rows.append({
                "seed": seed,
                "status": "COMPLETED_RETURN_0" if proc.returncode == 0 else "FAILED_NONZERO_RETURN",
                "returncode": proc.returncode,
                "start_time": start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "command": cmd,
            })
        except Exception as e:
            rows.append({"seed": seed, "status": "EXCEPTION", "error": repr(e), "command": cmd})
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "07_EXECUTION_LOGS" / "run_missing_seeds_execution_log.csv")
    return df


def aggregate_seed_metrics(output_root: Path, out_dir: Path, robustness_seeds: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = discover_runs(output_root)
    rows = []
    if not runs.empty:
        for _, rr in runs.iterrows():
            if not bool(rr.get("has_final_metrics", False)):
                continue
            seed = rr.get("seed")
            if pd.isna(seed):
                continue
            seed = int(seed)
            if seed != PRIMARY_SEED and seed not in robustness_seeds:
                continue
            df = read_csv_safe(final_metrics_path(Path(rr["run_dir"])))
            for _, r in df.iterrows():
                item = r.to_dict()
                item["seed"] = seed
                item["run_dir"] = rr["run_dir"]
                item["seed_role"] = "primary_locked_seed42" if bool(rr.get("is_primary_locked_run", False)) else "robustness_seed"
                item["pre_specified_robustness_seed"] = seed in robustness_seeds
                rows.append(item)
    long = pd.DataFrame(rows)
    save_csv(long, out_dir / "03_ROBUSTNESS_AGGREGATION" / "all_seed_final_metrics_long.csv")
    summaries = []
    if not long.empty:
        rob = long[long["pre_specified_robustness_seed"].astype(bool)].copy()
        for ds, g in rob.groupby("dataset"):
            row = {"dataset": ds, "robustness_seed_count_completed": int(g["seed"].nunique())}
            pass_col = "passed_exact_old_gate" if "passed_exact_old_gate" in g.columns else "strict_promotion" if "strict_promotion" in g.columns else None
            if pass_col:
                vals = pd.to_numeric(g[pass_col], errors="coerce")
                row["pass_count"] = int(vals.sum())
                row["pass_rate"] = float(vals.mean())
            for m in METRIC_COLS:
                if m in g.columns:
                    vals = pd.to_numeric(g[m], errors="coerce")
                    row[f"{m}_mean"] = vals.mean()
                    row[f"{m}_std"] = vals.std(ddof=1)
                    row[f"{m}_min"] = vals.min()
                    row[f"{m}_median"] = vals.median()
                    row[f"{m}_max"] = vals.max()
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    save_csv(summary, out_dir / "03_ROBUSTNESS_AGGREGATION" / "robustness30_dataset_summary.csv")
    return long, summary


def find_prediction_files(run_dir: Path) -> List[Path]:
    files = []
    for pat in ["*prediction*.csv", "*PREDICTION*.csv"]:
        files.extend([p for p in run_dir.rglob(pat) if p.is_file() and p.stat().st_size > 1000])
    out, seen = [], set()
    for p in sorted(files):
        if str(p) not in seen:
            seen.add(str(p)); out.append(p)
    return out


def infer_dataset_from_path(p: Path) -> Optional[str]:
    s = str(p).lower()
    for ds in DATASETS:
        if slug(ds) in s or norm(ds).replace("_", "") in s.replace("_", ""):
            return ds
    return None


def load_predictions(run_dir: Path) -> pd.DataFrame:
    frames = []
    for p in find_prediction_files(run_dir):
        df = read_csv_safe(p)
        if df.empty:
            continue
        lower = {c.lower(): c for c in df.columns}
        if "y_true" not in lower or "y_pred" not in lower:
            continue
        if lower["y_true"] != "y_true": df["y_true"] = df[lower["y_true"]]
        if lower["y_pred"] != "y_pred": df["y_pred"] = df[lower["y_pred"]]
        if "lower" not in df.columns: df["lower"] = df[lower["lower"]] if "lower" in lower else df["y_pred"]
        if "upper" not in df.columns: df["upper"] = df[lower["upper"]] if "upper" in lower else df["y_pred"]
        if "dataset" not in df.columns: df["dataset"] = infer_dataset_from_path(p) or "UNKNOWN"
        df["source_prediction_file"] = str(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out[out["dataset"].isin(DATASETS)].copy()


def cat(y: np.ndarray) -> np.ndarray:
    out = np.full(len(y), "safe", dtype=object)
    out[y <= 0.50] = "monitor"
    out[y <= 0.25] = "urgent"
    out[y <= 0.10] = "critical"
    return out


def fold_failure_analysis(output_root: Path, out_dir: Path, robustness_seeds: List[int]) -> pd.DataFrame:
    runs = discover_runs(output_root)
    rows = []
    if runs.empty:
        df = pd.DataFrame()
        save_csv(df, out_dir / "03_ROBUSTNESS_AGGREGATION" / "fold_asset_failure_analysis_from_predictions.csv")
        return df
    for _, rr in runs.iterrows():
        seed = rr.get("seed")
        if pd.isna(seed):
            continue
        seed = int(seed)
        if seed != PRIMARY_SEED and seed not in robustness_seeds:
            continue
        pred = load_predictions(Path(rr["run_dir"]))
        if pred.empty or "fold" not in pred.columns:
            continue
        for (ds, fold), d in pred.groupby(["dataset", "fold"]):
            y = pd.to_numeric(d["y_true"], errors="coerce").to_numpy(float)
            p = pd.to_numeric(d["y_pred"], errors="coerce").to_numpy(float)
            lo = pd.to_numeric(d["lower"], errors="coerce").to_numpy(float)
            hi = pd.to_numeric(d["upper"], errors="coerce").to_numpy(float)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(lo) & np.isfinite(hi)
            y, p, lo, hi = y[mask], p[mask], lo[mask], hi[mask]
            if len(y) == 0:
                continue
            uc = np.isin(cat(y), ["urgent", "critical"])
            covered = (y >= lo) & (y <= hi)
            rows.append({
                "seed": seed,
                "seed_role": "primary_locked_seed42" if bool(rr.get("is_primary_locked_run", False)) else "robustness_seed",
                "dataset": ds,
                "fold": fold,
                "rows": int(len(y)),
                "urgent_critical_count": int(uc.sum()),
                "coverage": float(np.mean(covered)),
                "urgent_critical_coverage": float(np.mean(covered[uc])) if uc.any() else np.nan,
                "mean_width": float(np.mean(hi - lo)),
                "mean_abs_error": float(np.mean(np.abs(y - p))),
                "run_dir": rr["run_dir"],
            })
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "03_ROBUSTNESS_AGGREGATION" / "fold_asset_failure_analysis_from_predictions.csv")
    return df


def latest_dir(root: Path, prefix: str) -> Optional[Path]:
    c = [p for p in root.glob(prefix + "*") if p.is_dir()]
    return sorted(c, key=lambda p: p.stat().st_mtime)[-1] if c else None


def collect_nn(output_root: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inv_rows, selected_frames, err_frames = [], [], []
    for ds, folder in NN_REQUIRED_RUNS.items():
        run = output_root / folder
        sel_p = run / "02_NN_CANDIDATE_RESULTS" / "03_nn_selected_by_dataset.csv"
        sum_p = run / "02_NN_CANDIDATE_RESULTS" / "02_nn_candidate_summary.csv"
        err_p = run / "00_RUN_MANIFEST" / "NN_ERRORS.csv"
        verdict_p = run / "00_RUN_MANIFEST" / "NN_BASELINE_VERDICT.json"
        inv = {"dataset": ds, "run_dir": str(run), "run_exists": run.exists(), "selected_exists": sel_p.exists(), "summary_exists": sum_p.exists(), "errors_exists": err_p.exists(), "verdict_exists": verdict_p.exists()}
        if verdict_p.exists():
            try:
                obj = json.loads(verdict_p.read_text(encoding="utf-8"))
                inv.update({"errors": obj.get("errors"), "any_nn_strict_pass": obj.get("any_nn_strict_pass"), "prediction_rows": obj.get("prediction_rows"), "loss_rows": obj.get("loss_rows")})
            except Exception:
                pass
        inv_rows.append(inv)
        if sel_p.exists():
            df = read_csv_safe(sel_p); df["source_nn_run"] = str(run); selected_frames.append(df)
        if err_p.exists():
            df = read_csv_safe(err_p); df["source_nn_run"] = str(run); err_frames.append(df)
    inv = pd.DataFrame(inv_rows)
    selected = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    errors = pd.concat(err_frames, ignore_index=True, sort=False) if err_frames else pd.DataFrame()
    save_csv(inv, out_dir / "04_NN_EVIDENCE" / "nn_evidence_inventory.csv")
    save_csv(selected, out_dir / "04_NN_EVIDENCE" / "nn_selected_all_five.csv")
    save_csv(errors, out_dir / "04_NN_EVIDENCE" / "nn_errors_all_five.csv")
    return inv, selected, errors


def bearing_nn_triage(nn_selected: pd.DataFrame, nn_errors: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for ds in ["PRONOSTIA/FEMTO", "XJTU-SY", "IMS"]:
        d = nn_selected[nn_selected["dataset"].eq(ds)].copy() if not nn_selected.empty and "dataset" in nn_selected.columns else pd.DataFrame()
        e = nn_errors[nn_errors["dataset"].eq(ds)].copy() if not nn_errors.empty and "dataset" in nn_errors.columns else pd.DataFrame()
        if d.empty:
            rows.append({"dataset": ds, "status": "MISSING_NN_SELECTED_ROW", "action": "Run/inspect full NN baseline."})
            continue
        r = d.iloc[0]
        strict = bool(r.get("strict_promotion", False))
        rows.append({
            "dataset": ds,
            "selected_nn_model": r.get("nn_model", ""),
            "selected_interval_variant": r.get("interval_variant", ""),
            "status": "NN_STRICT_PASS" if strict else "NN_DID_NOT_PASS_SAFETY_GATE",
            "RMSE": r.get("RMSE", np.nan),
            "MAE": r.get("MAE", np.nan),
            "R2": r.get("R2", np.nan),
            "coverage": r.get("empirical_coverage", np.nan),
            "urgent_critical_coverage": r.get("urgent_critical_coverage", np.nan),
            "mean_width": r.get("mean_interval_width", np.nan),
            "strict_promotion": strict,
            "blocking_reason": r.get("blocking_reason", ""),
            "nn_error_rows": int(len(e)),
            "interpretation": "Bearing NN evidence is not a replacement if it fails safety gates.",
            "action": "Use as negative/nearest-miss ablation or run targeted bearing-NN rescue before claiming strong all-five NN success." if not strict else "Can report as NN safety-pass baseline.",
        })
    triage = pd.DataFrame(rows)
    save_csv(triage, out_dir / "04_NN_EVIDENCE" / "bearing_nn_triage_and_action.csv")
    return triage


def collect_inventories(output_root: Path, primary_clean_run: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Diagnostics inventory
    if not diag.exists(): diag = latest_dir(primary_clean_run, "Q1_FINAL_CANDIDATE_DIAGNOSTICS_NEW_RESULTS_ONLY_") or diag
    diag_files = {
        "verdict": diag / "00_RUN_MANIFEST" / "FINAL_CANDIDATE_DIAGNOSTICS_VERDICT.json",
        "fold_metrics": diag / "01_SPLIT_AND_FOLD_AUDIT" / "test_fold_level_metrics.csv",
        "risk_zone": diag / "03_TRAIN_TEST_METRICS" / "test_risk_zone_metrics.csv",
        "figure_captions": diag / "08_REVIEWER_SUMMARY" / "figure_caption_audit.csv",
        "gaps": diag / "08_REVIEWER_SUMMARY" / "diagnostic_evidence_gaps_do_not_invent.csv",
    }
    diag_df = pd.DataFrame([{"item": k, "path": str(v), "exists": v.exists(), "rows": len(read_csv_safe(v)) if v.suffix == ".csv" and v.exists() else np.nan} for k, v in diag_files.items()])
    save_csv(diag_df, out_dir / "05_OTHER_EVIDENCE" / "diagnostics_inventory.csv")

    # Ablation inventory
    abl = latest_dir(primary_clean_run, "Q1_ABLATION_AND_STATISTICAL_TESTS_NEW_RESULTS_ONLY_")
    if abl:
        abl_files = {
            "verdict": abl / "00_RUN_MANIFEST" / "ABLATION_AND_STATISTICAL_TESTS_VERDICT.json",
            "model_family": abl / "02_MODEL_FAMILY_ABLATION" / "model_family_ablation_summary.csv",
            "interval_policy": abl / "03_INTERVAL_POLICY_ABLATION" / "interval_policy_ablation_summary.csv",
            "feature_policy": abl / "04_FEATURE_POLICY_ABLATION" / "feature_policy_ablation_summary.csv",
            "target_formulation": abl / "05_TARGET_FORMULATION_ABLATION" / "target_formulation_ablation_summary.csv",
            "selected_vs_nearest": abl / "06_SELECTED_VS_NEAREST_COMPETITOR" / "selected_vs_nearest_competitor.csv",
            "bootstrap": abl / "07_STATISTICAL_TESTS" / "paired_bootstrap_selected_vs_nearest_competitor.csv",
            "gaps": abl / "08_GAPS_AND_REVIEWER_SUMMARY" / "ablation_and_statistical_gaps_do_not_invent.csv",
        }
        abl_df = pd.DataFrame([{"item": k, "path": str(v), "exists": v.exists(), "rows": len(read_csv_safe(v)) if v.suffix == ".csv" and v.exists() else np.nan} for k, v in abl_files.items()])
    else:
        abl_df = pd.DataFrame([{"item": "ablation_package", "path": "", "exists": False, "rows": np.nan}])
    save_csv(abl_df, out_dir / "05_OTHER_EVIDENCE" / "ablation_inventory.csv")

    # Benchmark inventory
    if not bench.exists(): bench = latest_dir(output_root, "Q1_PUBLISHED_BENCHMARK_COMPARISON_PACKAGE_") or bench
    bench_files = {
        "verdict": bench / "BENCHMARK_PACKAGE_VERDICT.json",
        "validated_inputs": bench / "01_validated_benchmark_inputs.csv",
        "main_table": bench / "02_table_A_main_published_point_prediction_comparison.csv",
        "context_rows": bench / "03_context_only_or_needs_verification_rows.csv",
        "safety_validity": bench / "04_table_B_safety_validity_comparison.csv",
        "issues": bench / "06_benchmark_validation_issues.csv",
    }
    bench_df = pd.DataFrame([{"item": k, "path": str(v), "exists": v.exists(), "rows": len(read_csv_safe(v)) if v.suffix == ".csv" and v.exists() else np.nan} for k, v in bench_files.items()])
    save_csv(bench_df, out_dir / "05_OTHER_EVIDENCE" / "benchmark_inventory.csv")
    return diag_df, abl_df, bench_df


def final_matrix(out_dir: Path, seed_plan: pd.DataFrame, rob_summary: pd.DataFrame, nn_inv: pd.DataFrame, triage: pd.DataFrame, diag: pd.DataFrame, abl: pd.DataFrame, bench: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    def add(item, status, can_write, blocker, note):
        rows.append({"plan_item": item, "status": status, "can_write_now": can_write, "blocker": bool(blocker), "note": note})
    missing = len(seed_plan)
    robustness_ready = missing == 0 and not rob_summary.empty and rob_summary["robustness_seed_count_completed"].min() >= ROBUSTNESS_N_SEEDS
    add("Primary locked seed-42 clean result", "AVAILABLE", "YES", False, "Main result; old results remain obsolete.")
    add("30-seed robustness distribution", "AVAILABLE" if robustness_ready else "INCOMPLETE", "YES" if robustness_ready else "NO", missing > 0, f"Missing robustness seeds: {missing}.")
    nn_ready = not nn_inv.empty and bool(nn_inv["selected_exists"].all())
    add("Five-dataset NN baselines", "AVAILABLE" if nn_ready else "INCOMPLETE", "YES" if nn_ready else "NO", not nn_ready, "Bearing NN failures must be discussed honestly.")
    bearing_failed = not triage.empty and triage["status"].astype(str).str.contains("DID_NOT_PASS").any()
    add("Bearing NN triage", "NEGATIVE_EVIDENCE_PRESENT" if bearing_failed else "AVAILABLE", "YES_WITH_LIMITED_CLAIM", False, "Do not claim strong all-five NN success unless targeted rescue succeeds.")
    add("Held-out diagnostics and figures", "AVAILABLE_WITH_EXPLICIT_GAPS" if diag["exists"].any() else "INCOMPLETE", "YES_WITH_GAPS" if diag["exists"].any() else "NO", not diag["exists"].any(), "Train/cal and stage-runtime gaps must not be invented.")
    add("Ablation and statistical comparison", "AVAILABLE_WITHOUT_FORMAL_SIGNIFICANCE" if abl["exists"].any() else "INCOMPLETE", "YES_WITH_LIMITED_CLAIM" if abl["exists"].any() else "NO", not abl["exists"].any(), "Do not claim formal significance unless paired bootstrap completed.")
    add("Published benchmark package", "AVAILABLE_WITH_FAIRNESS_NOTES" if bench["exists"].any() else "INCOMPLETE", "YES_WITH_FAIRNESS_NOTES" if bench["exists"].any() else "NO", not bench["exists"].any(), "Use as contextual comparison, not apples-to-apples SOTA unless protocols match.")
    add("Correct block diagram and figure explanations", "WRITING_REQUIRED", "NO_MANUSCRIPT_YET", False, "After evidence closure, diagram must match actual clean workflow.")
    df = pd.DataFrame(rows)
    save_csv(df, out_dir / "06_FINAL_EVIDENCE_MATRIX" / "final_evidence_matrix.csv")
    blockers = df[df["blocker"].astype(bool)]
    verdict = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().isoformat(),
        "ready_for_manuscript": bool(blockers.empty),
        "blocker_count": int(len(blockers)),
        "blockers": blockers["plan_item"].tolist(),
        "missing_robustness_seed_count": int(missing),
        "bearing_nn_failure_present": bool(bearing_failed),
        "next_action": "Run missing robustness seeds, then rerun controller." if missing else "Evidence can move to manuscript planning with explicit limitations unless targeted bearing-NN rescue is required.",
        "required_interpretation": [
            "Primary seed 42 is the locked main result.",
            "30-seed robustness must include failures, not best seed selection.",
            "Bearing NN did not pass safety gates unless triage says otherwise.",
            "Do not invent train/cal diagnostics, stage runtime, or formal significance.",
        ],
    }
    save_json(out_dir / "00_RUN_MANIFEST" / "CONSOLIDATED_EVIDENCE_VERDICT.json", verdict)
    return df, verdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["all_no_run", "seed_plan", "run_missing_seeds", "aggregate", "final_matrix"], default="all_no_run")
    p.add_argument("--output_root", required=True)
    p.add_argument("--phase1_run", required=True)
    p.add_argument("--primary_clean_run", required=True)
    p.add_argument("--reproducer_script", required=True)
    p.add_argument("--master_seed", type=int, default=ROBUSTNESS_MASTER_SEED)
    p.add_argument("--n_robustness_seeds", type=int, default=ROBUSTNESS_N_SEEDS)
    p.add_argument("--max_seeds_to_run", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    phase1_run = Path(args.phase1_run)
    primary_clean_run = Path(args.primary_clean_run)
    reproducer_script = Path(args.reproducer_script)
    code_root = Path(args.code_root)
    if not output_root.exists(): raise FileNotFoundError(output_root)
    if not phase1_run.exists(): raise FileNotFoundError(phase1_run)
    if not primary_clean_run.exists(): raise FileNotFoundError(primary_clean_run)
    out_dir = output_root / f"Q1_CONSOLIDATED_EVIDENCE_CONTROLLER_{now_stamp()}"
    for d in ["00_RUN_MANIFEST", "01_RUN_DISCOVERY", "02_ROBUSTNESS_SEED_PLAN", "03_ROBUSTNESS_AGGREGATION", "04_NN_EVIDENCE", "05_OTHER_EVIDENCE", "06_FINAL_EVIDENCE_MATRIX", "07_EXECUTION_LOGS"]:
        ensure_dir(out_dir / d)
    banner("Q1 CONSOLIDATED EVIDENCE CONTROLLER â€” NEW RESULTS ONLY")
    print(f"SCRIPT_VERSION: {SCRIPT_VERSION}")
    print(f"Mode: {args.mode}")
    print(f"Output: {out_dir}")
    print("No manuscript generation. No obsolete results. No best-seed selection.")
    add_seed_metadata(primary_clean_run, PRIMARY_SEED, "Primary locked clean run.")
    seeds = load_or_create_seed_list(out_dir, args.master_seed, args.n_robustness_seeds)
    save_json(out_dir / "00_RUN_MANIFEST" / "controller_config.json", vars(args) | {"script_version": SCRIPT_VERSION, "primary_seed": PRIMARY_SEED, "robustness_seeds": seeds, "created_at": datetime.now().isoformat(), "python": sys.version, "platform": platform.platform()})
    discovery = discover_runs(output_root)
    save_csv(discovery, out_dir / "01_RUN_DISCOVERY" / "single_file_run_discovery_and_seed_metadata.csv")
    seed_plan = make_missing_seed_plan(output_root, out_dir, phase1_run, reproducer_script, code_root, seeds)
    if args.mode == "run_missing_seeds":
        exec_log = run_missing(seed_plan, out_dir, args.max_seeds_to_run)
        # Refresh after running.
        discovery = discover_runs(output_root)
        save_csv(discovery, out_dir / "01_RUN_DISCOVERY" / "single_file_run_discovery_after_execution.csv")
        seed_plan = make_missing_seed_plan(output_root, out_dir, phase1_run, reproducer_script, code_root, seeds)
    if args.mode == "seed_plan":
        metrics = pd.DataFrame(); rob_summary = pd.DataFrame(); nn_inv = pd.DataFrame(); nn_sel = pd.DataFrame(); nn_err = pd.DataFrame(); triage = pd.DataFrame(); diag = pd.DataFrame(); abl = pd.DataFrame(); bench = pd.DataFrame()
    else:
        metrics, rob_summary = aggregate_seed_metrics(output_root, out_dir, seeds)
        fold_failure_analysis(output_root, out_dir, seeds)
        nn_inv, nn_sel, nn_err = collect_nn(output_root, out_dir)
        triage = bearing_nn_triage(nn_sel, nn_err, out_dir)
        diag, abl, bench = collect_inventories(output_root, primary_clean_run, out_dir)
    matrix, verdict = final_matrix(out_dir, seed_plan, rob_summary, nn_inv, triage, diag, abl, bench)
    banner("CONSOLIDATED EVIDENCE CONTROLLER COMPLETE")
    print(f"Output folder: {out_dir}")
    print(f"Seed list: {out_dir / '00_RUN_MANIFEST' / 'robustness_seed_list_master20260605.csv'}")
    print(f"Missing seed commands: {out_dir / '02_ROBUSTNESS_SEED_PLAN' / 'missing_robustness_seed_commands.csv'}")
    print(f"Final evidence matrix: {out_dir / '06_FINAL_EVIDENCE_MATRIX' / 'final_evidence_matrix.csv'}")
    print("\nVERDICT")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

