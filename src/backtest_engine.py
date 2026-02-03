"""
Event-driven backtesting engine for pairs trading
"""
import pandas as pd
import numpy as np
from datetime import datetime
import config


class PairsBacktester:
    """
    Event-driven backtesting engine for pairs trading strategy
    """
    
    def __init__(self, initial_cash=100000, transaction_cost=0.001):
        """
        Initialize backtester
        
        Parameters:
        -----------
        initial_cash : float
            Initial capital
        transaction_cost : float
            Transaction cost as fraction (e.g., 0.001 = 0.1%)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.transaction_cost = transaction_cost
        
        # Position tracking
        self.positions = {}  # {pair_id: {'stock_a': qty, 'stock_b': qty, 'entry_date': date, 'entry_zscore': zscore}}
        self.trades = []     # All trades
        self.equity_curve = []  # Daily equity
        
    def calculate_portfolio_value(self, price_snapshot):
        """
        Calculate current portfolio value
        
        Parameters:
        -----------
        price_snapshot : dict
            {ticker: price} for current date
        
        Returns:
        --------
        float
            Total portfolio value
        """
        equity = self.cash
        
        for pair_id, pos in self.positions.items():
            stock_a, stock_b = pair_id
            price_a = price_snapshot.get(stock_a, 0)
            price_b = price_snapshot.get(stock_b, 0)
            
            if price_a > 0 and price_b > 0:
                equity += pos['stock_a'] * price_a + pos['stock_b'] * price_b
        
        return equity
    
    def enter_trade(self, pair_id, stock_a, stock_b, signal, prices, current_date, beta, zscore):
        """
        Enter a pairs trade
        
        Parameters:
        -----------
        pair_id : tuple
            (stock_a, stock_b)
        stock_a, stock_b : str
            Ticker symbols
        signal : str
            'long' (buy A, sell B) or 'short' (sell A, buy B)
        prices : dict
            Current prices {ticker: price}
        current_date : datetime
            Current date
        beta : float
            Hedge ratio
        zscore : float
            Current z-score
        """
        if pair_id in self.positions:
            return  # Already in position
        
        # Capital allocation per trade
        capital_per_trade = self.initial_cash * config.POSITION_SIZE_PCT
        
        price_a = prices.get(stock_a, 0)
        price_b = prices.get(stock_b, 0)
        
        if price_a <= 0 or price_b <= 0:
            return
        
        # Calculate quantities based on beta-hedged spread
        # For long spread: buy A, sell B (in beta ratio)
        # For short spread: sell A, buy B (in beta ratio)
        
        if signal == 'long':
            # Buy spread: buy A, sell B
            # Allocate capital equally between long and short legs
            qty_a = capital_per_trade / (2 * price_a)
            qty_b = -capital_per_trade / (2 * price_b) * beta  # Beta-adjusted
        else:  # short
            # Sell spread: sell A, buy B
            qty_a = -capital_per_trade / (2 * price_a)
            qty_b = capital_per_trade / (2 * price_b) * beta  # Beta-adjusted
        
        # Calculate transaction cost
        cost = abs(qty_a * price_a) + abs(qty_b * price_b)
        trading_fee = cost * self.transaction_cost
        
        if self.cash < trading_fee:
            return  # Insufficient cash
        
        self.cash -= trading_fee
        
        # Record position
        self.positions[pair_id] = {
            'stock_a': qty_a,
            'stock_b': qty_b,
            'entry_date': current_date,
            'entry_zscore': zscore,
            'beta': beta
        }
        
        # Record trade
        self.trades.append({
            'Date': current_date,
            'Type': 'ENTRY',
            'Pair': f"{stock_a}-{stock_b}",
            'Stock_A': stock_a,
            'Stock_B': stock_b,
            'Signal': signal,
            'Zscore': zscore,
            'Price_A': price_a,
            'Price_B': price_b,
            'Qty_A': qty_a,
            'Qty_B': qty_b,
            'Fee': trading_fee
        })
    
    def exit_trade(self, pair_id, prices, current_date, reason='signal'):
        """
        Exit a pairs trade
        
        Parameters:
        -----------
        pair_id : tuple
            (stock_a, stock_b)
        prices : dict
            Current prices {ticker: price}
        current_date : datetime
            Current date
        reason : str
            Exit reason
        """
        if pair_id not in self.positions:
            return
        
        pos = self.positions[pair_id]
        stock_a, stock_b = pair_id
        
        price_a = prices.get(stock_a, 0)
        price_b = prices.get(stock_b, 0)
        
        if price_a <= 0 or price_b <= 0:
            return
        
        # Calculate proceeds
        proceeds_a = pos['stock_a'] * price_a
        proceeds_b = pos['stock_b'] * price_b
        total_proceeds = proceeds_a + proceeds_b
        
        # Transaction cost
        cost = abs(pos['stock_a'] * price_a) + abs(pos['stock_b'] * price_b)
        trading_fee = cost * self.transaction_cost
        
        # Update cash
        self.cash += total_proceeds - trading_fee
        
        # Calculate P&L
        entry_date = pos['entry_date']
        entry_trade = next((t for t in self.trades if t['Type'] == 'ENTRY' and t['Pair'] == f"{stock_a}-{stock_b}" and t['Date'] == entry_date), None)
        
        if entry_trade:
            entry_cost = abs(entry_trade['Qty_A'] * entry_trade['Price_A']) + abs(entry_trade['Qty_B'] * entry_trade['Price_B'])
            pnl = total_proceeds - entry_cost - trading_fee - entry_trade['Fee']
            pnl_pct = pnl / entry_cost if entry_cost > 0 else 0
        else:
            pnl = 0
            pnl_pct = 0
        
        # Record trade
        self.trades.append({
            'Date': current_date,
            'Type': 'EXIT',
            'Pair': f"{stock_a}-{stock_b}",
            'Stock_A': stock_a,
            'Stock_B': stock_b,
            'Reason': reason,
            'Price_A': price_a,
            'Price_B': price_b,
            'PnL': pnl,
            'PnL_Pct': pnl_pct,
            'Fee': trading_fee
        })
        
        # Remove position
        del self.positions[pair_id]
    
    def run_backtest(self, data, coint_pairs, date_range):
        """
        Run backtest
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with features
        coint_pairs : pd.DataFrame
            Cointegrated pairs with beta
        date_range : tuple
            (start_date, end_date)
        """
        start_date, end_date = date_range
        
        print("=" * 50)
        print("Starting Backtest...")
        print(f"Date range: {start_date} to {end_date}")
        print("=" * 50)
        
        # Get date range
        if isinstance(data.index, pd.DatetimeIndex):
            date_mask = (data.index >= start_date) & (data.index <= end_date)
            dates = data.index[date_mask]
        else:
            dates = pd.date_range(start_date, end_date, freq='D')
        
        total_dates = len(dates)
        
        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"Progress: {i}/{total_dates} ({i/total_dates*100:.1f}%)")
            
            # Get current prices
            prices = {}
            for ticker in coint_pairs['Stock_A'].unique():
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        prices[ticker] = data[ticker]['Close'].loc[date]
                    else:
                        prices[ticker] = data['Close'].loc[date]
                except:
                    pass
            
            for ticker in coint_pairs['Stock_B'].unique():
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        prices[ticker] = data[ticker]['Close'].loc[date]
                    else:
                        prices[ticker] = data['Close'].loc[date]
                except:
                    pass
            
            if not prices:
                continue
            
            # Check signals for each pair
            for _, pair_row in coint_pairs.iterrows():
                stock_a = pair_row['Stock_A']
                stock_b = pair_row['Stock_B']
                beta = pair_row['Beta']
                
                if stock_a not in prices or stock_b not in prices:
                    continue
                
                pair_id = (stock_a, stock_b)
                
                try:
                    # Get historical data up to current date
                    if isinstance(data.columns, pd.MultiIndex):
                        hist_a = data[stock_a]['Close'][:date]
                        hist_b = data[stock_b]['Close'][:date]
                    else:
                        hist_a = data['Close'][:date]
                        hist_b = data['Close'][:date]
                    
                    if len(hist_a) < config.ZSCORE_WINDOW + 10:
                        continue
                    
                    # Calculate log prices
                    log_a = np.log(hist_a)
                    log_b = np.log(hist_b)
                    
                    # Align indices
                    common_idx = log_a.index.intersection(log_b.index)
                    if len(common_idx) < config.ZSCORE_WINDOW:
                        continue
                    
                    log_a_aligned = log_a[common_idx]
                    log_b_aligned = log_b[common_idx]
                    
                    # Calculate spread
                    spread = log_a_aligned - beta * log_b_aligned
                    
                    # Rolling statistics
                    ma = spread.rolling(config.ZSCORE_WINDOW).mean()
                    std = spread.rolling(config.ZSCORE_WINDOW).std()
                    
                    if len(spread) == 0 or std.iloc[-1] < 1e-8:
                        continue
                    
                    # Z-score
                    current_spread = spread.iloc[-1]
                    current_ma = ma.iloc[-1]
                    current_std = std.iloc[-1]
                    zscore = (current_spread - current_ma) / (current_std + 1e-8)
                    
                    # Signal logic
                    if pair_id not in self.positions:
                        # Entry signals
                        if zscore < config.ZSCORE_ENTRY_LONG:
                            self.enter_trade(pair_id, stock_a, stock_b, 'long', prices, date, beta, zscore)
                        elif zscore > config.ZSCORE_ENTRY_SHORT:
                            self.enter_trade(pair_id, stock_a, stock_b, 'short', prices, date, beta, zscore)
                    else:
                        # Exit signals
                        pos = self.positions[pair_id]
                        entry_zscore = pos['entry_zscore']
                        
                        # Mean reversion exit
                        if abs(zscore) < config.ZSCORE_EXIT:
                            self.exit_trade(pair_id, prices, date, reason='mean_reversion')
                        # Stop loss
                        elif abs(zscore) > config.ZSCORE_STOP_LOSS:
                            self.exit_trade(pair_id, prices, date, reason='stop_loss')
                        # Opposite signal (take profit)
                        elif (entry_zscore < 0 and zscore > 0) or (entry_zscore > 0 and zscore < 0):
                            self.exit_trade(pair_id, prices, date, reason='opposite_signal')
                
                except Exception as e:
                    # Skip this pair for this date
                    continue
            
            # Record daily equity
            equity = self.calculate_portfolio_value(prices)
            self.equity_curve.append({
                'Date': date,
                'Equity': equity,
                'Cash': self.cash,
                'Num_Positions': len(self.positions)
            })
        
        print("Backtest Complete!")
    
    def get_results(self):
        """
        Generate backtest results
        
        Returns:
        --------
        dict
            Dictionary with performance metrics and data
        """
        if not self.equity_curve:
            return None
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['Date'] = pd.to_datetime(equity_df['Date'])
        equity_df = equity_df.set_index('Date').sort_index()
        
        # Calculate returns
        equity_df['Returns'] = equity_df['Equity'].pct_change()
        equity_df['Cumulative_Returns'] = (1 + equity_df['Returns']).cumprod()
        
        # Performance metrics
        total_return = (equity_df['Equity'].iloc[-1] - self.initial_cash) / self.initial_cash
        
        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (252 / days) - 1
        else:
            annual_return = 0
        
        # Sharpe ratio
        returns = equity_df['Returns'].dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Maximum drawdown
        cumulative = equity_df['Cumulative_Returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        exit_trades = [t for t in self.trades if t['Type'] == 'EXIT']
        if exit_trades:
            winning_trades = len([t for t in exit_trades if t.get('PnL', 0) > 0])
            win_rate = winning_trades / len(exit_trades)
        else:
            win_rate = 0
        
        # Trade statistics
        num_trades = len([t for t in self.trades if t['Type'] == 'ENTRY'])
        
        results = {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Num_Trades': num_trades,
            'Num_Winning_Trades': len([t for t in exit_trades if t.get('PnL', 0) > 0]),
            'Num_Losing_Trades': len([t for t in exit_trades if t.get('PnL', 0) <= 0]),
            'Equity_Curve': equity_df,
            'Trades': pd.DataFrame(self.trades),
            'Final_Equity': equity_df['Equity'].iloc[-1]
        }
        
        return results