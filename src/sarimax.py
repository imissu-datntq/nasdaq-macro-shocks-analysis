"""SARIMA Box-Jenkins utilities.

Provides functionality for:
- ACF and PACF plotting
- Auto ARIMA grid search wrappers
- Diagnostics plotting and testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import pmdarima as pm

def plot_acf_pacf(series: pd.Series, lags: int = 40, title: str = "ACF & PACF", savepath: str | None = None) -> plt.Figure:
    """Plot ACF and PACF side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], zero=False)
    axes[0].set_title(f"ACF - {title}")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], zero=False, method='ywm')
    axes[1].set_title(f"PACF - {title}")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig

def optimize_sarima_pmdarima(
    series: pd.Series, 
    m: int = 12, 
    seasonal: bool = True,
    d: int | None = None,
    D: int | None = None,
    max_p: int = 3,
    max_q: int = 3,
    max_P: int = 2,
    max_Q: int = 2
):
    """Run pmdarima's auto_arima for optimal parameter selection."""
    print("Running auto_arima to find optimal parameters...")
    model = pm.auto_arima(
        series.dropna(),
        m=m,
        seasonal=seasonal,
        d=d,
        D=D,
        max_p=max_p,
        max_q=max_q,
        max_P=max_P,
        max_Q=max_Q,
        information_criterion='aic',
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    return model

def fit_sarimax_model(
    endog: pd.Series, 
    order: tuple[int, int, int], 
    seasonal_order: tuple[int, int, int, int],
    exog: pd.DataFrame | None = None
) -> SARIMAXResults:
    """Fit a SARIMAX model."""
    model = SARIMAX(
        endog=endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

def plot_diagnostics(results: SARIMAXResults, title: str = "SARIMA Diagnostics", savepath: str | None = None) -> plt.Figure:
    """Plot standard Box-Jenkins diagnostics for the residuals."""
    fig = results.plot_diagnostics(figsize=(15, 12))
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig