# ML-Driven Stock Selection: Comparing Optimization Objectives Across Three Model Families

## Overview

This project investigates whether the choice of **optimization objective** matters more than the **model itself** for ML-based stock selection. Three team members independently build models using the same dataset, each optimizing for three different objectives:

| Objective | What It Optimizes | Metric |
|-----------|-------------------|--------|
| **Accuracy** | Classification correctness | Cross-entropy / Accuracy |
| **Profit** | Top-decile portfolio return | Mean return of Long Top 10% |
| **Sharpe** | Risk-adjusted return | Sharpe ratio on validation set |

### Models

| Model | Type | Frequency | Tuning Method |
|-------|------|-----------|---------------|
| **Random Forest** | Non-linear ensemble (Bagging) | Monthly | Grid Search |
| **Gradient Boosting** | Non-linear ensemble (Boosting) | Monthly | Optuna (Bayesian) |
| **Linear Models** | Logistic Regression / Ridge | Quarterly | Grid Search + Nelder-Mead |

## Key Findings

1. **Optimization objective matters more than model choice** — Profit-driven and Sharpe-driven objectives consistently outperform accuracy-based optimization across all three models.

2. **Price-based factors dominate** — Momentum (12-1 month), short-term reversal, and volatility rank as the top 3 most important features, ahead of all 45 fundamental factors.

3. **Gradient Boosting achieves the best performance** — Long-Short Sharpe of 0.77 (Sharpe-optimized), compared to RF's best of 0.38 and Linear's results.

4. **Transaction costs erode most alpha** — After accounting for slippage, turnover, and taxes, excess returns shrink significantly. Lower-turnover strategies (RF at 32% vs GB at 50%) show more resilience.

5. **Beating the equal-weight benchmark is hard** — No Long-Only strategy consistently beats the benchmark Sharpe (0.53), highlighting market efficiency in the large-cap universe.

## Data

- **Source**: WRDS (CRSP + Compustat)
- **Universe**: ~5,000 US large-cap stocks (bottom 30% by market cap filtered out)
- **Period**: January 2012 – June 2022
- **Features**: 51 total
  - 45 fundamental factors from Compustat (valuation, profitability, leverage, liquidity, efficiency)
  - 6 price-based factors constructed from CRSP daily data (momentum, reversal, volatility, turnover, Amihud illiquidity, log market cap)
- **Target**: Next-month stock return (`next_month_ret`)

### Data Files (not included — place in `data/`)

| File | Description |
|------|-------------|
| `final_master_panel.csv` | Merged CRSP-Compustat panel with 45 fundamental features |
| `price.csv` | CRSP daily stock data (price, volume, returns, shares outstanding) |
| `final_master_panel_large_caps.csv` | Generated after running feature engineering + market cap filter |

## Methodology

### Walk-Forward Backtest Design

All models use a strict out-of-sample rolling window approach to prevent lookahead bias:

```
├── Train (36 months) ──► Gap (3 mo) ──► Valid (12 mo) ──► Gap (3 mo) ──► Test (12 mo)
│                                                                              │
└── Slide forward 12 months and repeat ◄───────────────────────────────────────┘
```

- **Test period**: July 2016 – June 2022 (6 rolling windows)
- **Portfolio**: Long Top 10% (equal-weight), rebalanced monthly
- **Transaction cost**: 10 bps double-sided (baseline)

### Portfolio Construction

For each test month:
1. Model predicts scores for all stocks in the universe
2. Stocks are ranked by predicted score and divided into deciles
3. **Long-Only**: Buy equal-weight Top 10% (Decile 10)
4. **Long-Short**: Buy Top 10%, Short Bottom 10% (zero-cost)

## Project Structure

```
ml-stock-selection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_random_forest.ipynb          # Model 1: Random Forest (RF)
│   ├── 02_gradient_boosting.ipynb      # Model 2: Gradient Boosting (GB)
│   └── 03_linear_models.ipynb          # Model 3: Linear Models (LR/Ridge)
├── src/
│   └── feature_engineering.py          # CRSP price factor construction
├── data/                               # Raw data files (git-ignored)
└── results/                            # Output CSVs and plots (git-ignored)
```

## Notebooks

### 01 — Random Forest (`01_random_forest.ipynb`)

**Stages:**
| Stage | Description | Sharpe |
|-------|-------------|--------|
| 0 | Data prep: construct 6 price factors from CRSP + market cap filter | — |
| 1 | Baseline regression (51 features) | 0.25 |
| 2 | Feature selection (Top 20 by importance) | 0.29 |
| 3 | Grid Search (MSE-optimized) | 0.38 |
| 4 | Accuracy-Driven classification | 0.32 |
| 5 | Long-Short analysis | -0.17 |
| 6 | Profit-Driven (Alpha-weighted + Grid Search) | **0.38** |
| 7 | Sharpe-Driven (Sharpe-optimized Grid Search) | 0.32 |
| 8 | Transaction cost sensitivity analysis | — |

**Key insight**: Momentum (12-1m) is the #1 feature with importance score 0.084, followed by short-term reversal (0.069) and volatility (0.048). Five of the top 6 features are newly constructed price-based factors.

### 02 — Gradient Boosting (`02_gradient_boosting.ipynb`)

**Stages:**
| Stage | Description | Best Sharpe |
|-------|-------------|-------------|
| Feature importance | Baseline GB on rolling windows | — |
| Baseline backtest | All 52 features, default params | — |
| Prediction line | Optuna-tuned, MSE objective | LO: 0.42, LS: 0.59 |
| Profit line | Optuna-tuned, profit objective | LO: 0.40, LS: 0.61 |
| Sharpe line | Optuna-tuned, Sharpe objective | LO: 0.43, **LS: 0.77** |
| Financial frictions | Commission + slippage + turnover + tax | — |
| Capital Gains Overhang | Disposition effect robustness test | — |

**Key insight**: Long-Short Sharpe of 0.77 is the highest across all models. Capital Gains Overhang analysis shows strategy returns are partially explained by investor reluctance to realize gains (disposition effect).

### 03 — Linear Models (`03_linear_models.ipynb`)

**Models:**
| Model | Objective | Method |
|-------|-----------|--------|
| Logistic Regression | Accuracy | Multinomial with balanced class weights |
| Cost-Sensitive LR | Profit | Sample weights ∝ |excess return| |
| Ridge + Threshold | Sharpe | Ridge regression → Nelder-Mead threshold optimization |

**Key difference**: Quarterly frequency (vs monthly for tree models). Labels based on ±2% excess return threshold over S&P 500 proxy.

## Cross-Model Results Summary

### Long-Only Top 10% (Sharpe Ratio)

| Objective | Random Forest | Gradient Boosting | Benchmark |
|-----------|--------------|-------------------|-----------|
| Accuracy | 0.32 | 0.42 | 0.53 |
| Profit | **0.38** | 0.40 | 0.53 |
| Sharpe | 0.32 | **0.43** | 0.53 |

### Transaction Cost Sensitivity (RF Profit-Driven)

| Scenario | Sharpe | Ann. Return |
|----------|--------|-------------|
| A: Commission only (10 bps) | 0.38 | 11.39% |
| B: + Slippage (20 bps) | 0.22 | 6.59% |
| C: + Real turnover (32%) | 0.38 | 11.44% |
| D: + Tax (25%) | 0.02 | 0.49% |

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Running the Notebooks

1. Place `final_master_panel.csv` and `price.csv` in `data/`
2. Run notebooks in order (01 → 02 → 03)
3. Notebook 01 generates `final_master_panel_large_caps.csv` which is used by all three

```bash
jupyter notebook notebooks/01_random_forest.ipynb
```

### Hardware Notes

- Notebooks were developed on machines with 16GB+ RAM
- Random Forest and Gradient Boosting training use `n_jobs=-1` (all CPU cores)
- Optuna tuning (Notebook 02) takes ~1 hour with 50 trials
- Grid Search (Notebook 01) takes ~30 minutes per objective

## References

- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency.
- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning.
- Frazzini, A. (2006). The disposition effect and underreaction to news.

## License

This project is for academic purposes only. Not financial advice.
