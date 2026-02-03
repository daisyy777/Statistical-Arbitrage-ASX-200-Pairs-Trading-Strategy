"""
Visualization module for backtest results
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import config


def plot_backtest_results(results, save_path=None):
    """
    Generate comprehensive backtest visualization
    
    Parameters:
    -----------
    results : dict
        Results from backtester.get_results()
    save_path : str, optional
        Path to save figure
    """
    if results is None:
        print("No results to plot")
        return
    
    equity_df = results['Equity_Curve']
    trades_df = results['Trades']
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Equity curve
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(equity_df.index, equity_df['Equity'], linewidth=2, color='#2E86AB')
    ax1.axhline(y=config.INITIAL_CASH, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = plt.subplot(3, 2, 2)
    cumulative = equity_df['Cumulative_Returns']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    ax2.fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(equity_df.index, drawdown, color='darkred', linewidth=1)
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Monthly returns
    ax3 = plt.subplot(3, 2, 3)
    equity_df['Month'] = equity_df.index.to_period('M')
    monthly = equity_df.groupby('Month')['Equity'].apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
    colors = ['green' if x > 0 else 'red' for x in monthly.values]
    ax3.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
    ax3.set_title('Monthly Returns', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Return (%)', fontsize=12)
    ax3.set_xlabel('Month', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Trade frequency
    ax4 = plt.subplot(3, 2, 4)
    entry_trades = trades_df[trades_df['Type'] == 'ENTRY'].copy()
    if not entry_trades.empty and 'Date' in entry_trades.columns:
        entry_trades['Date'] = pd.to_datetime(entry_trades['Date'])
        entry_trades['Month'] = entry_trades['Date'].dt.month
        trade_counts = entry_trades['Month'].value_counts().sort_index()
        ax4.bar(trade_counts.index, trade_counts.values, color='steelblue', alpha=0.7)
        ax4.set_title('Trade Frequency by Month', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Trades', fontsize=12)
        ax4.set_xlabel('Month', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. P&L distribution
    ax5 = plt.subplot(3, 2, 5)
    exit_trades = trades_df[trades_df['Type'] == 'EXIT']
    if not exit_trades.empty and 'PnL' in exit_trades.columns:
        pnl = exit_trades['PnL'].dropna()
        if len(pnl) > 0:
            ax5.hist(pnl, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
            ax5.set_title('P&L Distribution', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Frequency', fontsize=12)
            ax5.set_xlabel('P&L ($)', fontsize=12)
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Performance metrics text
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    metrics_text = f"""
    PERFORMANCE METRICS
    
    Total Return: {results['Total_Return']:.2%}
    Annual Return: {results['Annual_Return']:.2%}
    Sharpe Ratio: {results['Sharpe_Ratio']:.2f}
    Max Drawdown: {results['Max_Drawdown']:.2%}
    Win Rate: {results['Win_Rate']:.2%}
    
    TRADE STATISTICS
    
    Total Trades: {results['Num_Trades']}
    Winning Trades: {results['Num_Winning_Trades']}
    Losing Trades: {results['Num_Losing_Trades']}
    Final Equity: ${results['Final_Equity']:,.2f}
    """
    
    ax6.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(save_path)
        if dir_path:  # Only create if dir_path is not empty
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()


def generate_report(results, output_file=None):
    """
    Generate text report
    
    Parameters:
    -----------
    results : dict
        Results from backtester.get_results()
    output_file : str, optional
        Path to save report
    """
    if results is None:
        print("No results to report")
        return
    
    report = f"""
{'='*60}
PAIRS TRADING STRATEGY REPORT
{'='*60}

Strategy: Statistical Arbitrage (Cointegration-based)
Data: ASX 200
Time Period: {config.DATA_START_DATE} to {config.DATA_END_DATE}
Initial Capital: ${config.INITIAL_CASH:,.2f}

{'='*60}
PERFORMANCE METRICS
{'='*60}

Total Return:        {results['Total_Return']:.2%}
Annual Return:       {results['Annual_Return']:.2%}
Sharpe Ratio:        {results['Sharpe_Ratio']:.2f}
Max Drawdown:        {results['Max_Drawdown']:.2%}
Win Rate:            {results['Win_Rate']:.2%}

{'='*60}
TRADE STATISTICS
{'='*60}

Total Trades:        {results['Num_Trades']}
Winning Trades:      {results['Num_Winning_Trades']}
Losing Trades:       {results['Num_Losing_Trades']}
Final Equity:        ${results['Final_Equity']:,.2f}

{'='*60}
STRATEGY PARAMETERS
{'='*60}

Entry Threshold (Long):   Z-score < {config.ZSCORE_ENTRY_LONG}
Entry Threshold (Short):  Z-score > {config.ZSCORE_ENTRY_SHORT}
Exit Threshold:           |Z-score| < {config.ZSCORE_EXIT}
Stop Loss:                |Z-score| > {config.ZSCORE_STOP_LOSS}
Position Size:            {config.POSITION_SIZE_PCT:.1%} per pair
Transaction Cost:         {config.TRANSACTION_COST:.3%}

{'='*60}
KEY FINDINGS
{'='*60}

- The strategy identified {results['Num_Trades']} cointegrated pairs
- Entry signals triggered when z-score exceeds ±2.0
- Exit when z-score mean-reverts to ±0.5 or risk threshold ±3.0
- Transaction costs accounted for ({config.TRANSACTION_COST:.3%} per trade)

{'='*60}
RECOMMENDATIONS
{'='*60}

1. Walk-forward testing on out-of-sample data (2024-2025)
2. Compare with buy-and-hold benchmark
3. Optimize window size and z-score thresholds
4. Consider dynamic position sizing based on volatility
5. Add additional filters (volume, liquidity)

{'='*60}
"""
    
    print(report)
    
    if output_file:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(output_file)
        if dir_path:  # Only create if dir_path is not empty
            os.makedirs(dir_path, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")


if __name__ == '__main__':
    # This would be called from main.py after backtest
    pass