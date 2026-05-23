"""Forecast evaluation metrics and comparison helpers."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and MAPE for forecasts."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid zero division for MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.sum(mask) > 0 else np.nan
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

def print_metrics(metrics: dict[str, float], model_name: str = "Model"):
    """Pretty print the metrics."""
    print(f"--- {model_name} Performance ---")
    print(f"MAE  : {metrics['MAE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"MAPE : {metrics['MAPE']:.4f}%")
    print("-" * 30)
