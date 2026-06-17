# Predictive Maintenance Interval-Safety Reproducibility Package

## Overview

This repository contains public code and small reference outputs needed to reproduce and verify the reported multi-asset predictive-maintenance results. The study evaluates remaining useful life and state-of-health modelling with interval-based operational safety checks across turbofan, battery, and bearing datasets.

Raw datasets are not included in this repository.

## Repository contents

```text
README.md
DATA_AVAILABILITY.md
requirements.txt
run_reproduction.md
configs/
scripts/
expected_results/
```

## Public workspace convention

Before running the code, create an external workspace outside the repository:

```text
pm_interval_safety_workspace/
  raw_datasets/
    CMAPSS/
    NASA_Battery/
    PRONOSTIA/
    XJTU_SY/
    IMS/
  outputs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Step 1: configure paths and extract clean features

```bash
python scripts/01_extract_clean_features.py \
  --data_root "/path/to/pm_interval_safety_workspace/raw_datasets" \
  --output_root "/path/to/pm_interval_safety_workspace/outputs"
```

This writes `/path/to/pm_interval_safety_workspace/outputs/run_config.json`.

## Step 2: run later stages using the saved configuration

```bash
python scripts/run_full_pipeline.py \
  --config "/path/to/pm_interval_safety_workspace/outputs/run_config.json"
```

Individual later scripts also accept the same configuration file.

## Paper-result reference files

The `expected_results/` directory contains small CSV/JSON/TXT files related to the reported paper results. It does not contain raw data, large logs, full prediction dumps, or the complete run archive.

```bash
python scripts/verify_paper_results.py --expected_results expected_results
```

## Data

See `DATA_AVAILABILITY.md` for dataset acquisition and local folder layout.
