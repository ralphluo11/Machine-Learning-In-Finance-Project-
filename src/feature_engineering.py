"""
Feature Engineering: Construct price-based factors from CRSP daily data.

This script reads the raw CRSP price file and the Compustat fundamental panel,
constructs 6 price/volume-based factors, merges them into the panel, and
filters out small-cap stocks (bottom 30% by market cap each month).

Input:
    - data/final_master_panel.csv   (45 fundamental features)
    - data/price.csv                (CRSP daily: PERMNO, date, PRC, RET, VOL, SHROUT)

Output:
    - data/final_master_panel_large_caps.csv  (51 features, large-cap only)

New factors constructed:
    1. mom_12_1      : Momentum (cumulative return months t-12 to t-2)
    2. reversal_1m   : Short-term reversal (prior month return)
    3. vol_12m       : 12-month realized volatility (std of monthly returns)
    4. turnover      : Share turnover (avg daily volume / shares outstanding)
    5. amihud_illiq  : Amihud illiquidity (|return| / dollar volume)
    6. log_mcap      : Log market capitalization
"""

import pandas as pd
import numpy as np
import os

def build_price_factors(panel_path, price_path, output_path, filter_pct=0.30):
    """
    Main pipeline: load data → construct factors → merge → filter → save.
    
    Parameters
    ----------
    panel_path : str
        Path to the fundamental feature panel CSV.
    price_path : str
        Path to the CRSP daily price CSV.
    output_path : str
        Path to save the final output CSV.
    filter_pct : float
        Bottom percentile of market cap to filter out each month (default 0.30).
    """
    
    print(">>> 1. Loading raw data...")
    df_panel = pd.read_csv(panel_path)
    df_price = pd.read_csv(price_path, low_memory=False)

    # Standardize column names
    df_price.rename(columns={'PERMNO': 'permno'}, inplace=True)
    df_price['date'] = pd.to_datetime(df_price['date'])
    df_price['PRC'] = pd.to_numeric(df_price['PRC'], errors='coerce').abs()
    df_price['SHROUT'] = pd.to_numeric(df_price['SHROUT'], errors='coerce')
    df_price['RET'] = pd.to_numeric(df_price['RET'], errors='coerce')
    df_price['VOL'] = pd.to_numeric(df_price['VOL'], errors='coerce')

    df_price['market_cap'] = df_price['PRC'] * df_price['SHROUT']
    df_price['year_month'] = df_price['date'].dt.to_period('M')
    df_price = df_price.sort_values(['permno', 'date']).reset_index(drop=True)

    # ── Monthly aggregation ──────────────────────────────────────
    print(">>> 2. Aggregating daily data to monthly...")
    monthly = df_price.groupby(['permno', 'year_month']).agg(
        monthly_ret   = ('RET', lambda x: (1 + x.dropna()).prod() - 1),
        monthly_vol   = ('RET', lambda x: x.dropna().std()),
        avg_volume    = ('VOL', 'mean'),
        avg_shrout    = ('SHROUT', 'mean'),
        month_end_cap = ('market_cap', 'last'),
        trading_days  = ('RET', 'count'),
    ).reset_index()

    monthly = monthly.sort_values(['permno', 'year_month']).reset_index(drop=True)

    # ── Rolling factor construction ──────────────────────────────
    print(">>> 3. Constructing rolling factors per stock...")
    results = []
    for permno, g in monthly.groupby('permno'):
        g = g.copy()

        # Momentum (t-12 to t-2 cumulative return)
        g['mom_12_1'] = g['monthly_ret'].shift(1).rolling(11, min_periods=9).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )

        # Short-term reversal (prior month return)
        g['reversal_1m'] = g['monthly_ret'].shift(1)

        # 12-month volatility
        g['vol_12m'] = g['monthly_ret'].shift(1).rolling(12, min_periods=9).std()

        # Turnover
        g['turnover'] = (g['avg_volume'] / g['avg_shrout'].replace(0, np.nan)).shift(1)

        # Amihud illiquidity
        g['amihud_illiq'] = (
            g['monthly_ret'].abs() / (g['avg_volume'] * g['month_end_cap']).replace(0, np.nan)
        ).shift(1)

        # Log market cap
        g['log_mcap'] = np.log(g['month_end_cap'].replace(0, np.nan)).shift(1)

        results.append(g)

    monthly = pd.concat(results, ignore_index=True)

    # ── Merge with fundamental panel ─────────────────────────────
    print(">>> 4. Merging factors into panel...")
    new_factor_cols = ['mom_12_1', 'reversal_1m', 'vol_12m', 'turnover', 'amihud_illiq', 'log_mcap']
    monthly_factors = monthly[['permno', 'year_month', 'month_end_cap'] + new_factor_cols].copy()

    df_panel['public_date'] = pd.to_datetime(df_panel['public_date'])
    df_panel['year_month'] = df_panel['public_date'].dt.to_period('M')

    df_merged = pd.merge(df_panel, monthly_factors, on=['permno', 'year_month'], how='inner')

    # ── Market cap filter ────────────────────────────────────────
    print(f">>> 5. Filtering bottom {filter_pct:.0%} by market cap each month...")
    def filter_small(group):
        threshold = group['month_end_cap'].quantile(filter_pct)
        return group[group['month_end_cap'] >= threshold]

    df_filtered = df_merged.groupby('year_month', group_keys=False).apply(filter_small)
    df_filtered = df_filtered.drop(columns=['year_month', 'month_end_cap'], errors='ignore')

    # ── Save ─────────────────────────────────────────────────────
    df_filtered.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"Original panel:  {len(df_panel):,} rows")
    print(f"After merge:     {len(df_merged):,} rows")
    print(f"After filter:    {len(df_filtered):,} rows")
    print(f"New factors:     {new_factor_cols}")
    n_orig = len([c for c in df_panel.columns if c not in ['permno','adate','qdate','public_date','year_month','next_month_ret','year']])
    print(f"Total features:  {n_orig} (original) + {len(new_factor_cols)} (new) = {n_orig + len(new_factor_cols)}")
    print(f"Saved to:        {output_path}")
    print(f"{'='*50}")

    return df_filtered


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    build_price_factors(
        panel_path=os.path.join(data_dir, "final_master_panel.csv"),
        price_path=os.path.join(data_dir, "price.csv"),
        output_path=os.path.join(data_dir, "final_master_panel_large_caps.csv"),
        filter_pct=0.30,
    )
