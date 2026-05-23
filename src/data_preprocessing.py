"""Data loading, date normalization, resampling, and merge utilities.

This module houses the logic for cleaning FRED time-series data, standardizing 
datetime indices, and handling asynchronous weekends/holidays via resampling.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

def process_time_series(df: pd.DataFrame, value_col: str, resample_method: str = 'last') -> pd.DataFrame:
    """
    Standardize DatetimeIndex and downsample to monthly frequency.
    Uses 'ME' (Month End) instead of deprecated 'M'.
    Forward fills missing dates (e.g. weekends for stock index) before resampling.
    """
    df = df.copy()
    
    # Ensure column is DATE or gracefully rename observation_date
    if 'observation_date' in df.columns and 'DATE' not in df.columns:
        df.rename(columns={'observation_date': 'DATE'}, inplace=True)
    
    # Parse dates and setup index
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    
    # Coerce values to float
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Forward fill gaps before resampling (critical for daily stock prices)
    df = df.ffill()
    
    # Resample
    if resample_method == 'last':
        df_monthly = df.resample('ME').last()
    elif resample_method == 'mean':
        df_monthly = df.resample('ME').mean()
    else:
        raise ValueError(f"Unknown resample_method: {resample_method}")
        
    return df_monthly

def merge_macro_data(ndx_monthly: pd.DataFrame, cpi_monthly: pd.DataFrame, ffr_monthly: pd.DataFrame) -> pd.DataFrame:
    """Join the three datasets using an inner join strictly on matching month ends."""
    # Outer layers passed must already have ME frequencies
    master_df = ndx_monthly.join([cpi_monthly, ffr_monthly], how='inner')
    master_df.dropna(inplace=True)
    
    master_df.rename(columns={
        'NASDAQ100': 'NASDAQ',
        'CPIAUCSL': 'CPI',
        'FEDFUNDS': 'FFR'
    }, inplace=True)
    
    return master_df

def check_missing_months(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Check for chronological month discontinuity in the time series."""
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='ME')
    missing_months = expected_index.difference(df.index)
    return missing_months
