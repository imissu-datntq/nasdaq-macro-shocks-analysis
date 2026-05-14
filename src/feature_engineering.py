"""Feature engineering utilities with leakage-safe time split handling."""

from __future__ import annotations
import pandas as pd
import numpy as np

def chronological_split(df, train_ratio=0.70, val_ratio=0.15):
    """Cắt dữ liệu chuỗi thời gian không xáo trộn (No shuffling)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

def apply_fe_pipeline(current_df, lookback_df=None, max_lag=3):
    """
    Pipeline biến đổi đặc trưng có cơ chế mượn dữ liệu quá khứ (lookback)
    để không sinh ra giá trị NaN ở đầu tập Validation và Test.
    """
    # Nối đuôi dữ liệu quá khứ vào đầu (nếu có)
    if lookback_df is not None:
        temp_df = pd.concat([lookback_df.tail(max_lag), current_df])
    else:
        temp_df = current_df.copy()
        
    # 1. Tính Tỷ suất sinh lợi Logarit cho NASDAQ
    temp_df['NDX_log_ret'] = np.log(temp_df['NASDAQ']) - np.log(temp_df['NASDAQ'].shift(1))
    
    # 2. Tính Tỷ lệ Lạm phát từ CPI (Log difference)
    temp_df['CPI_inflation'] = np.log(temp_df['CPI']) - np.log(temp_df['CPI'].shift(1))
    
    # 3. Tính độ thay đổi của Lãi suất (First Difference)
    temp_df['FFR_diff'] = temp_df['FFR'] - temp_df['FFR'].shift(1)
    
    # 4. Tạo các biến trễ (Lagged features) cho SVAR/SARIMAX
    cols_to_lag = ['NDX_log_ret', 'CPI_inflation', 'FFR_diff']
    for col in cols_to_lag:
        for i in range(1, max_lag + 1):
            temp_df[f'{col}_lag{i}'] = temp_df[col].shift(i)
            
    # Cắt bỏ phần "mỏ neo" mượn từ lookback, hoặc drop NaN nếu là tập Train
    if lookback_df is not None:
        result_df = temp_df.iloc[max_lag:]
    else:
        result_df = temp_df.dropna()
        
    return result_df