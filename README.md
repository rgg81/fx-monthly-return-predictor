# 📈 FX Monthly Return Predictor

[![Build Status](https://github.com/rgg81/fx-monthly-return-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/rgg81/fx-monthly-return-predictor/actions)

A machine learning system that **predicts monthly returns** for FX pairs and gold (XAUUSD) using **OHLC price data**, technical indicators, mean reversion signals, and regime detection features. Employs ensemble ML strategies with Optuna hyperparameter optimization.

## 🚀 Features
✅ Predict **monthly returns** from OHLC price data for multiple FX pairs  
✅ Engineer features across three categories: **technical indicators, mean reversion, and regime detection**  
✅ Train **ensemble ML models** (LightGBM, XGBoost, CatBoost, Random Forest, Neural Networks, and more)  
✅ Optimize hyperparameters with **Optuna** using two-step bitmap feature selection  
✅ Backtest strategies with **walk-forward validation** and adaptive portfolio construction  
✅ Build **multi-currency portfolios** with dynamic pair weighting  

## 📊 Supported Instruments
EURUSD, USDJPY, EURJPY, GBPUSD, AUDUSD, XAUUSD

## 🛠 Installation
```bash
git clone https://github.com/rgg81/fx-monthly-return-predictor.git
cd fx-monthly-return-predictor
pip install -r requirements.txt
```

You also need Node.js installed for downloading price data:
```bash
npm install -g dukascopy-node
```

## 📋 Complete Analysis Workflow

Run a full analysis in three steps:

### Step 1: Download Price Data

Fetch historical OHLC data for all supported FX pairs using the Dukascopy API:

```bash
cd src/data_fetch
python duskacopy_api.py
```

This downloads ~25 years of hourly data and aggregates it to monthly OHLC bars, saving CSV files to the project root (e.g., `EURUSD.csv`, `USDJPY.csv`).

### Step 2: Generate Features

Run all three feature engineering pipelines:

```bash
cd src/features
python run_all_features.py
```

This generates feature sets for each currency pair:
- **Technical Indicators**: Trend, momentum, and volatility oscillators (`technical_indicators_features_{PAIR}.csv`)
- **Mean Reversion**: Bollinger Bands, support/resistance, momentum exhaustion (`mean_reversion_features_{PAIR}.csv`)
- **Regime Detection**: Market states, structural breaks, cycle detection (`regime_detection_features_{PAIR}.csv`)

### Step 3: Run Performance Analysis

Execute the ensemble backtesting and generate performance reports:

```bash
cd src/signals
python performance_analysis.py
```

This runs walk-forward backtesting with ensemble ML models and generates QuantStats HTML reports:

| Report | Description |
|--------|-------------|
| `stats_{PAIR}_model1.html` | Wide frequency range model (3-18 months) |
| `stats_{PAIR}_model2.html` | Short frequency range model (3-4 months) |
| `stats_{PAIR}_combined.html` | Average of both models per pair |
| `stats_simple_average.html` | Weighted average across all pairs |
| `stats_accounting_for_costs.html` | Portfolio with 5% transaction cost penalty |
| `stats_adaptive_portfolio.html` | Adaptive portfolio (excludes losing pairs dynamically) |

Each HTML report includes:
- Cumulative returns chart
- Drawdown analysis
- Monthly/yearly returns heatmap
- Key metrics: CAGR, Sharpe ratio, max drawdown, win rate

