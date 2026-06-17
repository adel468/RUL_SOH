from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
try:
    from _public_config import write_config, apply_config, load_config
except ImportError:
    from scripts._public_config import write_config, apply_config, load_config

EXPECTED_DATASET_FOLDERS = {
    "CMAPSS": "NASA C-MAPSS / turbofan data",
    "NASA_Battery": "NASA battery capacity data",
    "PRONOSTIA": "PRONOSTIA/FEMTO bearing data",
    "XJTU_SY": "XJTU-SY bearing data",
    "IMS": "IMS bearing data"
}

def validate_dataset_layout(data_root: Path) -> list[str]:
    return [f"{folder} ({label})" for folder, label in EXPECTED_DATASET_FOLDERS.items() if not (data_root / folder).exists()]

def main() -> int:
    parser = argparse.ArgumentParser(description="Configure data/output paths, create run_config.json, and run feature extraction.")
    parser.add_argument("--data_root", required=True, help="Directory containing raw dataset folders.")
    parser.add_argument("--output_root", required=True, help="Directory where all generated outputs will be written.")
    parser.add_argument("--setup_only", action="store_true", help="Create run_config.json only; do not run feature extraction.")
    parser.add_argument("--allow_missing", action="store_true", help="Continue even if expected dataset folders are not found.")
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    missing = validate_dataset_layout(data_root)
    if missing and not args.allow_missing:
        raise FileNotFoundError(f"Missing expected dataset folders under {data_root}: {missing}")
    config_path = write_config(data_root, output_root)
    cfg = load_config(config_path)
    apply_config(cfg)
    print(f"Configuration written to: {config_path}")
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    if args.setup_only:
        print("Setup completed. Feature extraction was not executed because --setup_only was used.")
        return 0
    engine = Path(__file__).with_name("_extract_clean_features_engine.py")
    if not engine.exists():
        raise FileNotFoundError(f"Feature-extraction engine not found: {engine}")
    env = os.environ.copy()
    env["Q1_RUN_CONFIG"] = str(config_path)
    env["Q1_DATA_ROOT"] = str(data_root)
    env["Q1_OUTPUT_ROOT"] = str(output_root)
    print("Running feature extraction engine...")
    result = subprocess.run([sys.executable, str(engine), "--config", str(config_path)], env=env)
    return int(result.returncode)

if __name__ == "__main__":
    raise SystemExit(main())

