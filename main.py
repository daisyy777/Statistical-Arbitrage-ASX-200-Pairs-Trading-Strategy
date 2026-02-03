"""
Main script to run the complete pairs trading pipeline
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from data_prep import main as data_prep_main
from cointegration_analysis import main as coint_main
from backtest_engine import PairsBacktester
from visualization import plot_backtest_results, generate_report


def main():
    """Main pipeline"""
    print("="*70)
    print("ASX 200 PAIRS TRADING STRATEGY - COMPLETE PIPELINE")
    print("="*70)
    print()
    
    # Step 1: Data preparation
    print("STEP 1: Data Preparation")
    print("-" * 70)
    try:
        data_prep_main()
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return
    
    print()
    
    # Step 2: Cointegration analysis
    print("STEP 2: Cointegration Analysis")
    print("-" * 70)
    try:
        coint_main()
    except Exception as e:
        print(f"Cointegration analysis failed: {e}")
        return
    
    print()
    
    # Step 3: Backtesting
    print("STEP 3: Backtesting")
    print("-" * 70)
    
    # Load data
    data_path = os.path.join(config.PROCESSED_DATA_DIR, 'data_features.pkl')
    coint_path = os.path.join(config.PROCESSED_DATA_DIR, 'cointegrated_pairs.csv')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    if not os.path.exists(coint_path):
        print(f"Cointegrated pairs file not found: {coint_path}")
        return
    
    data = pd.read_pickle(data_path)
    coint_pairs = pd.read_csv(coint_path)
    
    if coint_pairs.empty:
        print("No cointegrated pairs found. Cannot run backtest.")
        return
    
    print(f"Found {len(coint_pairs)} cointegrated pairs for backtesting")
    
    # Initialize backtester
    backtester = PairsBacktester(
        initial_cash=config.INITIAL_CASH,
        transaction_cost=config.TRANSACTION_COST
    )
    
    # Run backtest
    date_range = (config.DATA_START_DATE, config.DATA_END_DATE)
    backtester.run_backtest(data, coint_pairs, date_range)
    
    # Get results
    results = backtester.get_results()
    
    if results is None:
        print("Backtest failed. No results.")
        return
    
    print()
    
    # Step 4: Visualization and reporting
    print("STEP 4: Visualization and Reporting")
    print("-" * 70)
    
    # Plot results
    plot_save_path = os.path.join(config.REPORTS_DIR, 'backtest_results.png')
    plot_backtest_results(results, save_path=plot_save_path)
    
    # Generate report
    report_path = os.path.join(config.REPORTS_DIR, 'strategy_report.txt')
    generate_report(results, output_file=report_path)
    
    # Save trades to CSV
    trades_path = os.path.join(config.REPORTS_DIR, 'trades.csv')
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    results['Trades'].to_csv(trades_path, index=False)
    print(f"Trades saved to: {trades_path}")
    
    print()
    print("="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()