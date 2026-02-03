# ASX 200 Pairs Trading Strategy

A statistical arbitrage strategy based on cointegration for ASX-listed stocks.

## 📋 Overview

This project implements a pairs trading strategy that:
1. Identifies correlated stock pairs from ASX 200
2. Tests for cointegration using Johansen test
3. Generates trading signals based on z-score of spread
4. Backtests the strategy with proper risk management

## How to Start use it?

### Installation

pip install -r requirements.txt

python main.pyThis will:
- Download ASX 200 stock data
- Find cointegrated pairs
- Run backtest
- Generate visualizations and reports

## 📁 Project Structure
