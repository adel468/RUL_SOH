from __future__ import annotations
import json
import os
import sys
from pathlib import Path


def load_config(path: str | Path) -> dict:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(p)
    return cfg


def write_config(data_root: str | Path, output_root: str | Path) -> Path:
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "workspace_convention": "pm_interval_safety_workspace",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "features_dir": str(output_root / "01_clean_features"),
        "results_dir": str(output_root / "02_results"),
        "tables_dir": str(output_root / "03_tables"),
        "figures_dir": str(output_root / "04_figures"),
        "logs_dir": str(output_root / "logs"),
        "expected_dataset_folders": {
            "cmapss": "CMAPSS",
            "battery": "NASA_Battery",
            "pronostia": "PRONOSTIA",
            "xjtu_sy": "XJTU_SY",
            "ims": "IMS"
        }
    }
    for key in ["features_dir", "results_dir", "tables_dir", "figures_dir", "logs_dir"]:
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    config_path = output_root / "run_config.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return config_path


def apply_config(config: dict) -> None:
    os.environ["Q1_DATA_ROOT"] = str(config["data_root"])
    os.environ["Q1_OUTPUT_ROOT"] = str(config["output_root"])
    os.environ["Q1_FEATURES_DIR"] = str(config["features_dir"])
    os.environ["Q1_RESULTS_DIR"] = str(config["results_dir"])
    os.environ["Q1_TABLES_DIR"] = str(config["tables_dir"])
    os.environ["Q1_FIGURES_DIR"] = str(config["figures_dir"])
    os.environ["Q1_RUN_CONFIG"] = str(config.get("_config_path", ""))


def apply_config_from_argv() -> dict | None:
    config_path = None
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
            continue
        new_argv.append(arg)
        i += 1
    if config_path:
        cfg = load_config(config_path)
        apply_config(cfg)
        sys.argv[:] = new_argv
        return cfg
    env_cfg = os.environ.get("Q1_RUN_CONFIG")
    if env_cfg and Path(env_cfg).exists():
        cfg = load_config(env_cfg)
        apply_config(cfg)
        return cfg
    return None


def public_output_root() -> Path:
    return Path(os.environ.get("Q1_OUTPUT_ROOT", "outputs")).expanduser().resolve()


def public_data_root() -> Path:
    return Path(os.environ.get("Q1_DATA_ROOT", "raw_datasets")).expanduser().resolve()

