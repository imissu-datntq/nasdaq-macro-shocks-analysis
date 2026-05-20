"""Structural Vector Autoregression (SVAR) modeling and IRF utilities.

Provides functionality for:
- Lag order selection (AIC/BIC)
- Reduced-form VAR estimation
- Structural VAR identification via Cholesky decomposition
- Orthogonalized Impulse Response Functions (IRFs) and variance decomposition.
"""

from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults


def check_var_stability(var_results: VARResults) -> pd.DataFrame:
    """Check VAR stability by analyzing the modulus of the eigenvalues."""
    # statsmodels VARResults.roots gives the inverse of the characteristic roots
    # OR we can just use the internal is_stable method, which checks eigenvalues of companion matrix.
    is_stable = var_results.is_stable(verbose=False)
    # We retrieve the actual roots statsmodels computed (which it checks against 1)
    # In some versions of statsmodels, roots are outside unit circle (modulus > 1).
    # It's better to just rely on is_stable and report the roots as they are.
    roots = var_results.roots
    moduli = np.abs(roots)
    out_df = pd.DataFrame({'Root': [str(r) for r in roots], 'Modulus': moduli})
    out_df['Is_Stable_System'] = is_stable
    return out_df


def run_granger_causality(var_results: VARResults, caused: str, causing: str | list[str]) -> Any:
    """Run Granger Causality test on the fitted VAR model."""
    return var_results.test_causality(caused, causing, kind='f')


def select_var_order(df: pd.DataFrame, maxlags: int = 12) -> pd.DataFrame:
    """Select optimal lag order based on information criteria."""
    model = VAR(df)
    res = model.select_order(maxlags=maxlags)
    return res.summary()


def fit_var_model(df: pd.DataFrame, lags: int) -> VARResults:
    """Fit a reduced-form VAR model with specific lags."""
    model = VAR(df)
    results = model.fit(maxlags=lags)
    return results


def run_cholesky_irf(
    var_results: VARResults, 
    periods: int = 24,
    seed: int = 42
) -> Any:
    """
    Compute orthogonalized IRFs using Cholesky decomposition.
    
    The variable ordering in the DataFrame passed to VAR must follow:
    [CPI_inflation, FFR_diff, NDX_log_ret] to respect the identification:
    CPI -> FFR -> NDX
    """
    np.random.seed(seed)
    # The `irf()` method in statsmodels implicitly uses Cholesky orthogonalization
    # based on the column order when using `var_results.irf(periods)`.
    irf = var_results.irf(periods)
    return irf


def run_fevd(var_results: VARResults, periods: int = 24) -> Any:
    """Compute Forecast Error Variance Decomposition."""
    return var_results.fevd(periods)


def plot_sv_irf(irf_results, impulse: str, response: str, periods: int = 24, savepath: str | None = None) -> plt.Figure:
    """Plot Impulse Response of a specific response to a specific structural shock.
    
    Since statsmodels irf() with orthogonalization uses Cholesky, we can
    directly plot the orthogonalized shock.
    """
    fig = irf_results.plot(impulse=impulse, response=response, plot_stderr=True, stderr_type='mc', repl=1000)
    fig.suptitle(f"SVAR IRF: Response of {response} to {impulse} Shock", fontsize=14, y=1.05)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig

def plot_sv_cumulative_irf(irf_results, impulse: str, response: str, periods: int = 24, savepath: str | None = None) -> plt.Figure:
    """Plot Cumulative Impulse Response of a specific response to a specific structural shock."""
    fig = irf_results.plot_cum_effects(impulse=impulse, response=response, plot_stderr=True, stderr_type='mc', repl=1000)
    fig.suptitle(f"Cumulative IRF: Permanent impact of {impulse} on {response}", fontsize=14, y=1.05)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig
