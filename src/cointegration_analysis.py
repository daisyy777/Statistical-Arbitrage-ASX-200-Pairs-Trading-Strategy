"""
Cointegration analysis for pairs trading
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import os
import config


def johansen_test(y1, y2, det_order=0, k_ar_diff=1):
    """
    Perform Johansen cointegration test
    
    Parameters:
    -----------
    y1, y2 : array-like
        Two time series
    det_order : int
        Deterministic order (0: no constant, 1: constant)
    k_ar_diff : int
        Number of lags
    
    Returns:
    --------
    tuple
        (trace_statistic, critical_value_95, is_cointegrated)
    """
    try:
        # Ensure arrays
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)
        
        # Remove NaN
        mask = ~(np.isnan(y1) | np.isnan(y2))
        y1_clean = y1[mask]
        y2_clean = y2[mask]
        
        if len(y1_clean) < 50:  # Increased minimum length
            return None, None, False
        
        # Check for constant series
        if np.std(y1_clean) < 1e-8 or np.std(y2_clean) < 1e-8:
            return None, None, False
        
        data = np.column_stack([y1_clean, y2_clean])
        
        # Johansen test requires at least k_ar_diff+1 observations
        if len(data) < k_ar_diff + 10:
            return None, None, False
        
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
        
        # Trace statistic for r=0 (no cointegration)
        trace_stat = result.lr1[0]
        # Critical value at 95% confidence
        critical_value = result.cvt[0, 1]
        
        # Test: H0: no cointegration, H1: cointegration exists
        is_cointegrated = trace_stat > critical_value
        
        return trace_stat, critical_value, is_cointegrated
    except Exception as e:
        # Only print error in debug mode, otherwise silent
        return None, None, False


def find_beta(y1, y2):
    """
    Find hedge ratio using OLS: y1 = alpha + beta * y2
    
    Parameters:
    -----------
    y1, y2 : array-like
        Two time series
    
    Returns:
    --------
    float
        Beta (hedge ratio)
    """
    try:
        # Align data
        mask = ~(np.isnan(y1) | np.isnan(y2))
        y1_clean = y1[mask]
        y2_clean = y2[mask]
        
        if len(y1_clean) < 10:
            return None
        
        # OLS: y1 = alpha + beta * y2
        X = np.column_stack([np.ones(len(y2_clean)), y2_clean])
        params = np.linalg.lstsq(X, y1_clean, rcond=None)[0]
        beta = params[1]
        
        return beta
    except:
        return None


def compute_spread_zscore(y1, y2, beta, window=20):
    """
    Compute spread and z-score
    
    Parameters:
    -----------
    y1, y2 : array-like
        Two time series (log prices)
    beta : float
        Hedge ratio
    window : int
        Rolling window for z-score calculation
    
    Returns:
    --------
    tuple
        (zscore_series, spread_series)
    """
    try:
        # Align data
        mask = ~(np.isnan(y1) | np.isnan(y2))
        y1_clean = y1[mask]
        y2_clean = y2[mask]
        
        if len(y1_clean) < window:
            return None, None
        
        # Convert to pandas Series for consistent operations
        spread_series = pd.Series(y1_clean) - beta * pd.Series(y2_clean)
        
        # Rolling mean and std
        ma = spread_series.rolling(window).mean()
        std = spread_series.rolling(window).std()
        
        # Z-score (all operations on Series, result is Series)
        zscore_series = (spread_series - ma) / (std + 1e-8)
        
        # Return as numpy arrays
        return zscore_series.values, spread_series.values
    except Exception as e:
        print(f"Z-score calculation error: {e}")
        return None, None


def analyze_cointegration_pairs(data, candidates_df, save_path=None):
    """
    Analyze all candidate pairs for cointegration
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data with features
    candidates_df : pd.DataFrame
        DataFrame with candidate pairs
    save_path : str, optional
        Path to save results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with cointegrated pairs and statistics
    """
    print("=" * 50)
    print("Starting Cointegration Analysis...")
    print("=" * 50)
    
    coint_pairs = []
    total_pairs = len(candidates_df)
    stats = {
        'total': total_pairs,
        'insufficient_data': 0,
        'johansen_failed': 0,
        'no_cointegration': 0,
        'beta_failed': 0,
        'zscore_failed': 0,
        'success': 0
    }
    print(f"Total candidate pairs to test: {total_pairs}\n")
    
    for idx, row in candidates_df.iterrows():
        stock_a = row['Stock_A']
        stock_b = row['Stock_B']
        
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{total_pairs} ({stock_a}-{stock_b})")
        
        try:
            # Get log prices
            if isinstance(data.columns, pd.MultiIndex):
                close_a = data[stock_a]['Close']
                close_b = data[stock_b]['Close']
            else:
                close_a = data['Close']
                close_b = data['Close']
            
            # Convert to log prices
            log_price_a = np.log(close_a).dropna()
            log_price_b = np.log(close_b).dropna()
            
            # Align indices
            common_idx = log_price_a.index.intersection(log_price_b.index)
            if len(common_idx) < 50:  # Reduced from 100 to allow more pairs
                stats['insufficient_data'] += 1
                continue
            
            y1 = log_price_a[common_idx].values
            y2 = log_price_b[common_idx].values
            
            # Johansen test
            trace_stat, critical_value, is_cointegrated = johansen_test(y1, y2)
            
            if trace_stat is None:
                stats['johansen_failed'] += 1
                continue
            
            if not is_cointegrated:
                stats['no_cointegration'] += 1
                continue
            
            # Calculate beta
            beta = find_beta(y1, y2)
            if beta is None:
                stats['beta_failed'] += 1
                continue
            
            # Calculate spread statistics
            zscore, spread = compute_spread_zscore(y1, y2, beta, window=config.ZSCORE_WINDOW)
            
            if zscore is None:
                stats['zscore_failed'] += 1
                continue
            
            stats['success'] += 1
            
            # Spread statistics
            mean_spread = np.mean(spread)
            std_spread = np.std(spread)
            mean_abs_zscore = np.mean(np.abs(zscore))
            
            coint_pairs.append({
                'Stock_A': stock_a,
                'Stock_B': stock_b,
                'Beta': beta,
                'Trace_Statistic': trace_stat,
                'Critical_Value_95': critical_value,
                'Is_Cointegrated': is_cointegrated,
                'Mean_Spread': mean_spread,
                'Std_Spread': std_spread,
                'Mean_Abs_Zscore': mean_abs_zscore,
                'Correlation': row.get('Correlation', np.nan)
            })
            
            if len(coint_pairs) % 10 == 0:
                print(f"Processed {len(coint_pairs)} cointegrated pairs...")
                
        except Exception as e:
            print(f"Error processing {stock_a}-{stock_b}: {e}")
            continue
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Cointegration Test Statistics:")
    print("=" * 50)
    print(f"Total candidate pairs: {stats['total']}")
    print(f"Insufficient data: {stats['insufficient_data']}")
    print(f"Johansen test failed: {stats['johansen_failed']}")
    print(f"Failed cointegration test: {stats['no_cointegration']}")
    print(f"Beta calculation failed: {stats['beta_failed']}")
    print(f"Z-score calculation failed: {stats['zscore_failed']}")
    print(f"Successfully found cointegrated pairs: {stats['success']}")
    print("=" * 50)
    
    if not coint_pairs:
        print("\nNo cointegrated pairs found")
        print("Suggestions:")
        print("  1. Lower correlation threshold (MIN_CORRELATION)")
        print("  2. Check data quality and time range")
        print("  3. Adjust cointegration test parameters")
        print("  4. Increase data time range")
        return pd.DataFrame()
    
    coint_df = pd.DataFrame(coint_pairs)
    coint_df = coint_df.sort_values('Trace_Statistic', ascending=False)
    
    print(f"\nFound {len(coint_df)} cointegrated pairs")
    print("\nTop 5 cointegrated pairs:")
    print(coint_df.head())
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        coint_df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
    
    return coint_df


def main():
    """Main cointegration analysis pipeline"""
    # Load data
    data_path = os.path.join(config.PROCESSED_DATA_DIR, 'data_features.pkl')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run data_prep.py first")
        return
    
    data = pd.read_pickle(data_path)
    
    # Load candidates
    candidates_path = os.path.join(config.PROCESSED_DATA_DIR, 'pair_candidates.csv')
    if not os.path.exists(candidates_path):
        print(f"Candidate pairs file not found: {candidates_path}")
        print("Please run data_prep.py first")
        return
    
    candidates_df = pd.read_csv(candidates_path)
    
    # Analyze cointegration
    coint_pairs = analyze_cointegration_pairs(
        data,
        candidates_df,
        save_path=os.path.join(config.PROCESSED_DATA_DIR, 'cointegrated_pairs.csv')
    )
    
    print("=" * 50)
    print("Cointegration Analysis Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()