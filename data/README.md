# Data Directory

Place the following files here before running the notebooks:

| File | Source | Size |
|------|--------|------|
| `final_master_panel.csv` | WRDS (CRSP + Compustat merged panel) | ~100 MB |
| `price.csv` | WRDS CRSP daily stock file | ~200 MB |

These files are not included in the repository due to WRDS licensing restrictions.

## How to Obtain

1. Log in to [WRDS](https://wrds-www.wharton.upenn.edu/)
2. **Compustat**: Fundamentals Quarterly → select all ratio variables
3. **CRSP**: Monthly/Daily Stock File → select PERMNO, date, PRC, RET, VOL, SHROUT
4. Merge on PERMNO + date, save as `final_master_panel.csv`

## Generated Files

After running `01_random_forest.ipynb` (Stage 0), the following file is generated:

| File | Description |
|------|-------------|
| `final_master_panel_large_caps.csv` | Panel with 51 features, bottom 30% small-caps removed |

This file is used by all three notebooks.
