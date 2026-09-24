"""
risk_metrics.py
----------------
Standard quantitative risk/performance metrics used by real asset managers
to evaluate a strategy, all computed from a daily returns series.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    growth = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    return growth ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free=0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free=0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return dd.min()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return annualized_return(returns) / abs(mdd)


def historical_var(returns: pd.Series, alpha=0.05) -> float:
    """Historical (non-parametric) Value at Risk at the given confidence level."""
    return np.percentile(returns, 100 * alpha)


def historical_cvar(returns: pd.Series, alpha=0.05) -> float:
    """Conditional VaR / Expected Shortfall: average loss beyond the VaR threshold."""
    var = historical_var(returns, alpha)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) else var


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def summarize(returns: pd.Series, name="Strategy") -> dict:
    return {
        "name": name,
        "Annualized Return": annualized_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Sortino Ratio": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Calmar Ratio": calmar_ratio(returns),
        "95% Daily VaR": historical_var(returns, 0.05),
        "95% Daily CVaR": historical_cvar(returns, 0.05),
    }


def summary_table(list_of_returns: dict) -> pd.DataFrame:
    """list_of_returns: {"Strategy A": returns_series, "Benchmark": returns_series}"""
    rows = [summarize(r, name) for name, r in list_of_returns.items()]
    df = pd.DataFrame(rows).set_index("name")
    return df
