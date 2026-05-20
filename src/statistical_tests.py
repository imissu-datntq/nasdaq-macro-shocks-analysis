"""Statistical test helpers (ADF, KPSS, Ljung-Box, ACF/PACF helpers)."""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


def run_adf_test(series: pd.Series, signif_level: float = 0.05) -> Dict[str, Any]:
    """Run ADF test and return formatted results."""
    series = series.dropna()
    res = adfuller(series, autolag='AIC')
    p_value = res[1]
    return {
        'ADF_Statistic': round(res[0], 6),
        'p_value': round(p_value, 6),
        'Num_Lags': res[2],
        'Num_Obs': res[3],
        'Is_Stationary': p_value <= signif_level
    }


def run_kpss_test(series: pd.Series, regression: str = 'c', signif_level: float = 0.05) -> Dict[str, Any]:
    """Run KPSS test and return formatted results. regression='c' or 'ct'."""
    series = series.dropna()
    try:
        stat, p_value, lags, crit = kpss(series, regression=regression, nlags='auto')
    except Exception:
        return {'KPSS_Statistic': np.nan, 'p_value': np.nan, 'Lags': None, 'Is_Stationary': False}
    return {
        'KPSS_Statistic': round(stat, 6),
        'p_value': round(p_value, 6),
        'Lags': lags,
        'Is_Stationary': p_value > signif_level
    }


def run_ljung_box(series: pd.Series, lags: int = 12) -> pd.DataFrame:
    """Run Ljung-Box test for `lags` and return DataFrame of stat and p-value."""
    series = series.dropna()
    lb = acorr_ljungbox(series, lags=lags, return_df=True)
    return lb


def plot_acf_pacf(series: pd.Series, lags: int = 24, figsize=(10,6), savepath: str | None = None):
    """Plot ACF and PACF side by side. If savepath provided, save the figure."""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method='ywm')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')
    return fig


def acf_pacf_values(series: pd.Series, nlags: int = 24) -> Dict[str, Any]:
    """Return numeric ACF and PACF values for programmatic checks."""
    series = series.dropna()
    return {'acf': acf(series, nlags=nlags), 'pacf': pacf(series, nlags=nlags)}