from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
try:
    from _public_config import load_config, apply_config
except ImportError:
    from scripts._public_config import load_config, apply_config

STAGES = [
    "02_train_candidate_models.py",
    "03_generate_candidate_diagnostics.py",
    "04_run_ablation_and_statistics.py",
    "05_run_neural_network_baselines.py",
    "06_compare_published_benchmarks.py",
    "07_consolidate_repeated_seed_evidence.py",
    "08_rebuild_candidate_pool.py",
    "09_audit_selected_row_level_safety.py",
    "10_reconcile_metric_definitions.py",
    "11_reselect_strict_interval_candidates.py",
    "12_finalize_strict_interval_evidence.py",
    "13_generate_manuscript_tables_and_figures.py",
    "14_generate_supporting_audits.py",
]

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public reproduction pipeline using run_config.json.")
    parser.add_argument("--config", required=True, help="Path to run_config.json produced by 01_extract_clean_features.py.")
    parser.add_argument("--start_at", default="", help="Optional stage filename to start from.")
    parser.add_argument("--stop_after", default="", help="Optional stage filename to stop after.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    apply_config(cfg)
    script_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["Q1_RUN_CONFIG"] = str(Path(args.config).expanduser().resolve())
    started = not bool(args.start_at)
    for stage in STAGES:
        if args.start_at and stage == args.start_at:
            started = True
        if not started:
            continue
        stage_path = script_dir / stage
        if not stage_path.exists():
            print(f"Skipping missing stage: {stage}")
            continue
        print(f"\n=== Running {stage} ===")
        result = subprocess.run([sys.executable, str(stage_path), "--config", str(args.config)], env=env)
        if result.returncode != 0:
            print(f"Stage failed: {stage}")
            return int(result.returncode)
        if args.stop_after and stage == args.stop_after:
            break
    print("\nPipeline completed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

