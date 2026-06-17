from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description='Verify the public supporting result package and checksum manifest.')
    parser.add_argument("--config", default=None, help="Path to run_config.json created by 01_extract_clean_features.py.")
    parser.add_argument("--expected_results", default=None, help="Optional expected-results directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    expected_root = Path(args.expected_results).expanduser().resolve() if args.expected_results else repo_root / "expected_results"
    output_root = Path(cfg.get("output_root") or os.environ.get("Q1_OUTPUT_ROOT") or (repo_root / "_public_stage_outputs")).expanduser().resolve()
    status_dir = output_root / "public_stage_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    required = [
    "paper_results_manifest.csv",
    "sha256_manifest.csv"
]
    missing = [name for name in required if not (expected_root / name).exists()]
    status = {
        "stage": Path(__file__).name,
        "description": 'Verify the public supporting result package and checksum manifest.',
        "expected_results": str(expected_root),
        "required_files": required,
        "missing_files": missing,
        "status": "PASS" if not missing else "FAIL",
    }
    status_path = status_dir / '14_supporting_audits_status.json'
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if missing:
        print("Stage verification failed. Missing files:")
        for name in missing:
            print(f"- {name}")
        print(f"Status written to: {status_path}")
        return 1

    print(f"Stage verification passed. Status written to: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

