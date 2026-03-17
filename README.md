# ASX 200 Pairs Trading Strategy

A statistical arbitrage strategy based on cointegration analysis for ASX 200 listed stocks. The strategy identifies cointegrated stock pairs, models the mean-reverting spread, and generates long/short signals using z-score thresholds.

---

## Backtest Results (2023-01-01 to 2025-11-26)

| Metric | Value |
|--------|-------|
| Total Return | 0.39% |
| Annual Return | 0.09% |
| Sharpe Ratio | 0.35 |
| Max Drawdown | -0.55% |
| Win Rate | 0.00% |
| Total Trades | 118 |
| Final Equity | $100,388.90 |

> Initial Capital: $100,000 — Transaction Cost: 0.1% per leg

### Cointegrated Pairs Identified

| Pair | Beta | Trace Stat | Correlation | Sector |
|------|------|------------|-------------|--------|
| CMM.AX / EVN.AX | 0.860 | 21.71 | 0.750 | Gold Mining |
| VCX.AX / MGR.AX | 1.413 | 17.71 | 0.700 | A-REITs |
| RIO.AX / BHP.AX | 0.695 | 16.22 | 0.843 | Diversified Mining |
| CMM.AX / GMD.AX | 0.696 | 16.21 | 0.788 | Gold Mining |

Johansen trace statistic critical value at 95% confidence: **15.494**

Universe screened: **200 ASX tickers → 194 valid stocks → 41 correlated pairs (ρ > 0.7) → 4 cointegrated pairs**

---

## Strategy Overview

### Pipeline

```
main.py
├── Step 1: Data Preparation      (src/data_prep.py)
│   ├── Download OHLCV via yfinance
│   ├── Filter stocks with < 200 trading days
│   └── Screen pairs by return correlation (ρ > 0.7)
│
├── Step 2: Cointegration Analysis (src/cointegration_analysis.py)
│   ├── Johansen test on log price pairs
│   ├── OLS hedge ratio (beta) estimation
│   └── Rolling z-score spread computation
│
├── Step 3: Backtesting            (src/backtest_engine.py)
│   ├── Event-driven bar-by-bar simulation
│   ├── Beta-neutral position sizing
│   └── Entry / exit / stop-loss signal logic
│
└── Step 4: Reporting              (src/visualization.py)
    ├── Equity curve & drawdown chart
    ├── Monthly returns bar chart
    └── P&L distribution histogram
```

### Signal Logic

| Condition | Action |
|-----------|--------|
| Z-score < -2.0 | Enter long spread (buy A, sell B) |
| Z-score > +2.0 | Enter short spread (sell A, buy B) |
| \|Z-score\| < 0.5 | Exit — mean reversion complete |
| \|Z-score\| > 3.0 | Exit — stop loss triggered |

---

## Project Structure

```
.
├── main.py                  # Full pipeline entry point
├── config.py                # All parameters and settings
├── requirements.txt
├── src/
│   ├── data_prep.py         # Data download and feature engineering
│   ├── cointegration_analysis.py  # Johansen test, beta, z-score
│   ├── backtest_engine.py   # Event-driven backtester
│   └── visualization.py     # Charts and text report
├── data/
│   ├── raw/                 # Downloaded OHLCV (git-ignored)
│   └── processed/           # Intermediate outputs (git-ignored)
└── reports/                 # Generated charts and reports (git-ignored)
```

---

## Installation & Usage

```bash
pip install -r requirements.txt
python main.py
```

Outputs saved to `reports/`:
- `backtest_results.png` — performance chart
- `strategy_report.txt` — full metrics report
- `trades.csv` — trade-by-trade log

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_START_DATE` | 2023-01-01 | Backtest start |
| `DATA_END_DATE` | 2025-11-26 | Backtest end |
| `MIN_CORRELATION` | 0.7 | Pair screening threshold |
| `COINTEGRATION_PVALUE_THRESHOLD` | 0.05 | Johansen test significance |
| `ZSCORE_WINDOW` | 20 | Rolling window for spread stats |
| `ZSCORE_ENTRY_LONG` | -2.0 | Long entry threshold |
| `ZSCORE_ENTRY_SHORT` | +2.0 | Short entry threshold |
| `ZSCORE_EXIT` | 0.5 | Mean reversion exit |
| `ZSCORE_STOP_LOSS` | 3.0 | Stop loss |
| `INITIAL_CASH` | 100,000 | Starting capital (AUD) |
| `TRANSACTION_COST` | 0.001 | Cost per trade (0.1%) |
| `POSITION_SIZE_PCT` | 0.01 | Capital per pair (1%) |

---

## Dependencies

```
yfinance
pandas
numpy
statsmodels
matplotlib
```

---

## Notes

- The strategy found only 4 cointegrated pairs from the ASX 200 universe over this period. The limited number of tradeable pairs and the small per-pair position size (1% of capital) keeps total return modest.
- Win Rate appears 0% in the report because the single open position at end of backtest (unrealized gain of ~$389) is not counted as a closed winning trade — all 117 closed trades had small negative PnL due to transaction costs exceeding spread reversion gains.
- Potential improvements: extend the lookback window, lower the correlation screening threshold, add sector-neutral constraints, or use rolling cointegration windows.
