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
