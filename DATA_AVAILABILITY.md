# Data Availability

Raw datasets are intentionally not included in this repository.

Users must obtain each dataset from its official source and comply with the applicable license, terms of use, and citation requirements. Public availability of a dataset does not automatically imply permission to redistribute a copy through this repository.

Use this local workspace layout outside the repository:

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

Expected dataset folders:

| Folder | Dataset |
|---|---|
| `CMAPSS/` | NASA C-MAPSS turbofan data |
| `NASA_Battery/` | NASA battery capacity data |
| `PRONOSTIA/` | PRONOSTIA/FEMTO bearing data |
| `XJTU_SY/` | XJTU-SY bearing data |
| `IMS/` | IMS bearing data |

The first script writes a configuration file that all later scripts reuse.
