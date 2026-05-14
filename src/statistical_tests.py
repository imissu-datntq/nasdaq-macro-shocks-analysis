"""Statistical test helpers (ADF, KPSS, Ljung-Box)."""

from __future__ import annotations
from statsmodels.tsa.stattools import adfuller

def run_adf_test(series, signif_level=0.05):
    """Chạy kiểm định Augmented Dickey-Fuller và format kết quả gọn gàng."""
    # Bỏ qua các giá trị NaN nếu có
    series = series.dropna()
    result = adfuller(series, autolag='AIC')
    
    p_value = result[1]
    is_stationary = p_value <= signif_level
    
    return {
        'ADF_Statistic': round(result[0], 4),
        'p_value': round(p_value, 4),
        'Num_Lags': result[2],
        'Num_Obs': result[3],
        'Is_Stationary': is_stationary
    }