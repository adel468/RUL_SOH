# Reproduction Guide

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare the external workspace

Create this folder outside the repository:

```text
pm_interval_safety_workspace/
  raw_datasets/
  outputs/
```

Place raw datasets under `raw_datasets/` using the folder names described in `DATA_AVAILABILITY.md`.

## 3. Run feature extraction and create the shared configuration

```bash
python scripts/01_extract_clean_features.py \
  --data_root "/path/to/pm_interval_safety_workspace/raw_datasets" \
  --output_root "/path/to/pm_interval_safety_workspace/outputs"
```

## 4. Run the full pipeline

```bash
python scripts/run_full_pipeline.py \
  --config "/path/to/pm_interval_safety_workspace/outputs/run_config.json"
```

## 5. Verify the reference paper-result files

```bash
python scripts/verify_paper_results.py --expected_results expected_results
```
