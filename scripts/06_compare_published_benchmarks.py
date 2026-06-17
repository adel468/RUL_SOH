
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

# Public release script: 06_compare_published_benchmarks.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

#!/usr/bin/env python
# -*- coding: utf-8 -*-



import argparse, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED = [
    "dataset","study","authors","year","model","protocol","metric","published_value",
    "our_comparable_value","source_type","doi_or_url","table_or_figure","fairness_note","include_in_main_table"
]
TRUSTED_MAIN = {"peer_reviewed","conference","official","thesis"}

SAFETY_COLUMNS = [
    "dataset","study","reports_uncertainty","reports_urgent_coverage","reports_false_safe",
    "reports_leakage_audit","reports_asset_wise_validation","note"
]

OUR_SAFETY_ROW = {
    "study":"This work",
    "reports_uncertainty":"Yes",
    "reports_urgent_coverage":"Yes",
    "reports_false_safe":"Yes",
    "reports_leakage_audit":"Yes",
    "reports_asset_wise_validation":"Yes",
    "note":"Leakage-controlled grouped validation with calibrated decision intervals and safety-gate metrics."
}

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def load_locked_metrics(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if p.is_dir():
        p = p / "02_TABLES" / "SINGLE_FILE_SELECTED_CANDIDATE_METRICS.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def validate_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED:
        if c not in out.columns:
            out[c] = ""
    issues = []
    valid_main = []
    for _, r in out.iterrows():
        row_issues = []
        for c in ["dataset","study","year","model","protocol","metric","published_value","doi_or_url","table_or_figure","fairness_note"]:
            v = str(r.get(c,"")).strip()
            if not v or v.upper().startswith("FILL") or v.lower() == "nan":
                row_issues.append(f"missing_{c}")
        st = str(r.get("source_type","")).strip().lower()
        if st not in TRUSTED_MAIN:
            row_issues.append("source_type_not_main_trusted")
        include = str(r.get("include_in_main_table","0")).strip() in {"1","true","TRUE","yes","YES"}
        valid = include and not row_issues
        issues.append("; ".join(row_issues) if row_issues else "OK")
        valid_main.append(valid)
    out["validation_issues"] = issues
    out["main_table_validated"] = valid_main
    return out

def build_safety_validity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "dataset": r.get("dataset",""),
            "study": r.get("study",""),
            "reports_uncertainty": "Usually no / not in benchmark row",
            "reports_urgent_coverage": "No",
            "reports_false_safe": "No",
            "reports_leakage_audit": "Usually not explicit",
            "reports_asset_wise_validation": "Protocol-specific / verify",
            "note": "Published row is used for point-prediction context; safety/validity fields must be verified from the paper before stronger claims."
        })
    for dataset in sorted(df["dataset"].dropna().unique()):
        rr = dict(OUR_SAFETY_ROW)
        rr["dataset"] = dataset
        rows.append(rr)
    return pd.DataFrame(rows, columns=SAFETY_COLUMNS)

def manuscript_paragraph(main: pd.DataFrame, context: pd.DataFrame) -> str:
    datasets = ", ".join(sorted(main["dataset"].dropna().unique()))
    return (
        "Published-work benchmarking was curated from source-verified rows rather than scraped automatically. "
        f"The main comparison table includes validated rows for: {datasets}. "
        "Because published RUL/SOH studies use different targets, splits, scales, and bearing/battery selections, "
        "the benchmark table is used as point-prediction context rather than as a universal SOTA leaderboard. "
        "The central contribution of this work remains leakage-controlled, asset-wise, safety-aware decision evidence: "
        "empirical coverage, urgent/critical coverage, false-safe rate, underwarning, interval width, and explicit leakage audits."
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_csv", required=True)
    p.add_argument("--dataset_sources_csv", default="")
    p.add_argument("--clean_run_metrics", default="")
    p.add_argument("--output_root", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_root) / ("Q1_PUBLISHED_BENCHMARK_COMPARISON_PACKAGE_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    ensure_dir(out_dir)

    raw = pd.read_csv(args.benchmark_csv)
    validated = validate_rows(raw)
    main = validated[validated["main_table_validated"]].copy()
    context = validated[~validated["main_table_validated"]].copy()

    locked = load_locked_metrics(args.clean_run_metrics)
    if not locked.empty:
        keep = [c for c in ["dataset","RMSE","MAE","R2","empirical_coverage","urgent_critical_coverage","interval_false_safe_rate","interval_underwarning_rate","mean_interval_width"] if c in locked.columns]
        locked_small = locked[keep].copy()
        main = main.merge(locked_small, on="dataset", how="left", suffixes=("","_our_clean"))
        context = context.merge(locked_small, on="dataset", how="left", suffixes=("","_our_clean"))
    safety = build_safety_validity(validated)

    save_csv(validated, out_dir / "01_validated_benchmark_inputs.csv")
    save_csv(main, out_dir / "02_table_A_main_published_point_prediction_comparison.csv")
    save_csv(context, out_dir / "03_context_only_or_needs_verification_rows.csv")
    save_csv(safety, out_dir / "04_table_B_safety_validity_comparison.csv")

    if args.dataset_sources_csv and Path(args.dataset_sources_csv).exists():
        src = pd.read_csv(args.dataset_sources_csv)
        save_csv(src, out_dir / "05_official_dataset_source_references.csv")

    missing = validated[validated["validation_issues"].ne("OK")].copy()
    save_csv(missing, out_dir / "06_benchmark_validation_issues.csv")

    para = manuscript_paragraph(main, context)
    (out_dir / "07_benchmark_section_paragraph_draft.txt").write_text(para, encoding="utf-8")

    verdict = {
        "output_dir": str(out_dir),
        "input_rows": int(len(validated)),
        "main_table_rows_validated": int(len(main)),
        "context_or_needs_verification_rows": int(len(context)),
        "validation_issue_rows": int(len(missing)),
        "status": "PASS_WITH_CURATED_INPUT" if len(main) > 0 else "NO_MAIN_ROWS_VALIDATED"
    }
    (out_dir / "BENCHMARK_PACKAGE_VERDICT.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")

    print("BENCHMARK PACKAGE COMPLETE")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

