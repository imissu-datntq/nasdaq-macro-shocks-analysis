import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust imports to find src
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from feature_engineering import apply_fe_pipeline, chronological_split

def test_chronological_split():
    # Setup dummy data
    dates = pd.date_range("2020-01-01", periods=100, freq='M')
    df = pd.DataFrame({'val': np.arange(100)}, index=dates)
    
    train, val, test = chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    
    # Check lengths
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    
    # Check no overlap
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()

def test_fe_pipeline_no_lookback():
    dates = pd.date_range("2020-01-01", periods=10, freq='M')
    df = pd.DataFrame({
        'NASDAQ': [100.0] * 10,
        'CPI': [100.0] * 10,
        'FFR': [1.0] * 10
    }, index=dates)
    
    max_lag = 3
    # Without lookback, the result drops `max_lag` rows
    res = apply_fe_pipeline(df, lookback_df=None, max_lag=max_lag)
    assert len(res) == len(df) - max_lag
    assert 'NDX_log_ret_lag3' in res.columns

def test_fe_pipeline_with_lookback():
    dates = pd.date_range("2020-01-01", periods=20, freq='M')
    df = pd.DataFrame({
        'NASDAQ': np.random.rand(20) * 100 + 100,
        'CPI': np.random.rand(20) * 10 + 100,
        'FFR': np.random.rand(20) + 1.0
    }, index=dates)
    
    train = df.iloc[:10]
    val = df.iloc[10:]
    
    max_lag = 3
    res_val = apply_fe_pipeline(val, lookback_df=train, max_lag=max_lag)
    
    # With lookback, validation length should be preserved exactly!
    assert len(res_val) == len(val)
    # Check no NaNs
    assert not res_val.isnull().values.any()

if __name__ == "__main__":
    print("Running FE pipeline tests...")
    test_chronological_split()
    test_fe_pipeline_no_lookback()
    test_fe_pipeline_with_lookback()
    print("✓ All tests passed successfully!")
