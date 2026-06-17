
# Public configuration bootstrap. This allows --config to be accepted by every
# stage script without requiring the original parser to define it explicitly.
try:
    from _public_config import apply_config_from_argv
except ImportError:
    from scripts._public_config import apply_config_from_argv
apply_config_from_argv()

# Public release script: 13_generate_manuscript_tables_and_figures.py
# Update local raw-data/output paths at the top of the script if your directory layout differs.

from pathlib import Path
import os

p = Path(r"pm_interval_safety_workspace/outputs")

print("Exists:", p.exists())
print("Path:", p)

if p.exists():
    print("\nFiles:")
    for f in p.iterdir():
        print(" -", f.name)
    os.startfile(p)
else:
    pass

