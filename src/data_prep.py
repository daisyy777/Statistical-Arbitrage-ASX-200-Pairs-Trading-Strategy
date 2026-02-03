"""
Data preparation module for ASX 200 pairs trading
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
import config


def download_asx_data(tickers, start_date, end_date, save_path=None):
    """
    Download ASX stock data from Yahoo Finance
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    save_path : str, optional
        Path to save raw data
    
    Returns:
    --------
    pd.DataFrame
        Multi-index DataFrame with OHLCV data
    """
    print(f"Downloading data for {len(tickers)} stocks...")
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date,
            group_by='ticker',
            progress=True,
            auto_adjust=True,
            prepost=False
        )
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data.to_csv(save_path)
            print(f"Data saved to: {save_path}")
        
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def clean_data(raw_data, min_data_points=200):
    """
    Clean raw data: remove stocks with insufficient data
    
    Parameters:
    -----------
    raw_data : pd.DataFrame
        Raw OHLCV data
    min_data_points : int
        Minimum number of data points required
    
    Returns:
    --------
    tuple
        (cleaned_data, valid_tickers)
    """
    print("Cleaning data...")
    
    if raw_data.empty:
        return None, []
    
    # Get unique tickers from multi-index columns
    if isinstance(raw_data.columns, pd.MultiIndex):
        tickers = raw_data.columns.get_level_values(0).unique()
    else:
        # Single ticker case
        return raw_data, [raw_data.columns.name] if raw_data.columns.name else []
    
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            ticker_data = raw_data[ticker] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
            
            # Check if enough data points
            if len(ticker_data.dropna()) >= min_data_points:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except:
            invalid_tickers.append(ticker)
    
    print(f"Valid stocks: {len(valid_tickers)}")
    print(f"Invalid stocks: {len(invalid_tickers)}")
    
    if valid_tickers:
        cleaned_data = raw_data[valid_tickers] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
        return cleaned_data, valid_tickers
    else:
        return None, []


def add_features(data, tickers):
    """
    Add technical features: log returns, volatility
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    tickers : list
        List of valid tickers
    
    Returns:
    --------
    pd.DataFrame
        Data with added features
    """
    print("Calculating technical indicators...")
    
    data_features = data.copy()
    
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[ticker]['Close']
            else:
                close = data['Close']
            
            # Log returns
            log_ret = np.log(close / close.shift(1))
            
            # Volatility (20-day rolling)
            volatility = log_ret.rolling(20).std() * np.sqrt(252)
            
            # Add to dataframe
            if isinstance(data_features.columns, pd.MultiIndex):
                data_features[(ticker, 'log_returns')] = log_ret
                data_features[(ticker, 'volatility')] = volatility
            else:
                data_features['log_returns'] = log_ret
                data_features['volatility'] = volatility
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    return data_features


def find_pair_candidates(data, tickers, min_corr=0.7, save_path=None):
    """
    Find potential pairs based on correlation
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data with features
    tickers : list
        List of valid tickers
    min_corr : float
        Minimum correlation threshold
    save_path : str, optional
        Path to save candidates
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Stock_A, Stock_B, Correlation
    """
    print(f"Finding pairs with correlation > {min_corr}...")
    
    # Extract close prices
    close_prices = pd.DataFrame()
    
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close_prices[ticker] = data[ticker]['Close']
            else:
                close_prices[ticker] = data['Close']
        except:
            continue
    
    if close_prices.empty:
        return pd.DataFrame(columns=['Stock_A', 'Stock_B', 'Correlation'])
    
    # Calculate returns
    returns = close_prices.pct_change().dropna()
    
    if returns.empty:
        return pd.DataFrame(columns=['Stock_A', 'Stock_B', 'Correlation'])
    
    # Correlation matrix
    corr_matrix = returns.corr()
    
    # Find pairs
    pairs = []
    ticker_list = corr_matrix.index.tolist()
    
    for i in range(len(ticker_list)):
        for j in range(i+1, len(ticker_list)):
            ticker_a = ticker_list[i]
            ticker_b = ticker_list[j]
            corr = corr_matrix.iloc[i, j]
            
            if not np.isnan(corr) and corr > min_corr:
                pairs.append({
                    'Stock_A': ticker_a,
                    'Stock_B': ticker_b,
                    'Correlation': corr
                })
    
    candidates_df = pd.DataFrame(pairs)
    candidates_df = candidates_df.sort_values('Correlation', ascending=False)
    
    print(f"Found {len(candidates_df)} candidate pairs")
    
    if save_path and not candidates_df.empty:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        candidates_df.to_csv(save_path, index=False)
        print(f"Candidate pairs saved to: {save_path}")
    
    return candidates_df


def main():
    """Main data preparation pipeline"""
    print("=" * 50)
    print("Data Preparation Pipeline Started")
    print("=" * 50)
    
    # Step 1: Download data
    raw_data = download_asx_data(
        config.ASX200_TICKERS,
        config.DATA_START_DATE,
        config.DATA_END_DATE,
        save_path=os.path.join(config.RAW_DATA_DIR, 'asx200_raw.csv')
    )
    
    if raw_data is None or raw_data.empty:
        print("Data download failed. Please check network connection and ticker list")
        return
    
    # Step 2: Clean data
    cleaned_data, valid_tickers = clean_data(raw_data, config.MIN_DATA_POINTS)
    
    if cleaned_data is None:
        print("Data cleaning failed")
        return
    
    # Step 3: Add features
    data_features = add_features(cleaned_data, valid_tickers)
    
    # Step 4: Find pair candidates
    candidates = find_pair_candidates(
        data_features,
        valid_tickers,
        min_corr=config.MIN_CORRELATION,
        save_path=os.path.join(config.PROCESSED_DATA_DIR, 'pair_candidates.csv')
    )
    
    # Save processed data
    processed_path = os.path.join(config.PROCESSED_DATA_DIR, 'data_features.pkl')
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    data_features.to_pickle(processed_path)
    print(f"Processed data saved to: {processed_path}")
    
    print("=" * 50)
    print("Data Preparation Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()