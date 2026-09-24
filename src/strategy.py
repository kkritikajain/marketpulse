"""
strategy.py
-----------
Backtests a simple, transparent regime-aware allocation rule against a
static buy-and-hold benchmark, using ONLY the regime detected up to and
including the previous trading day (no look-ahead bias: today's allocation
decision never uses today's return).

Allocation rule (equity sleeve vs. cash/bond sleeve):
    Calm Bull        -> 100% equity basket
    Choppy/Neutral    -> 60% equity / 40% bond
    Crisis/Bear       -> 20% equity / 80% bond

This mirrors a real "vol-targeting" / "risk-off" overlay used by many
multi-asset funds. It is intentionally simple so it's auditable -- the goal
of this project is to demonstrate the regime-detection + backtesting
*framework*, which can be swapped for far more sophisticated allocation
rules (see README "Future Features").
"""

import numpy as np
import pandas as pd

ALLOCATION_RULE = {
    "Calm Bull": 1.00,
    "Choppy/Neutral": 0.60,
    "Crisis/Bear": 0.20,
}

TRANSACTION_COST_BPS = 5  # 5 basis points per unit of turnover, a realistic retail/ETF cost


def backtest_regime_strategy(prices: pd.DataFrame, regime_labels: pd.Series, bond_ticker="BOND_ETF"):
    """
    Parameters
    ----------
    prices : DataFrame of asset prices, must include `bond_ticker`
    regime_labels : Series of detected regimes, indexed by date (may start
        later than `prices` because of the rolling window used to detect them)

    Returns
    -------
    dict with 'strategy_returns', 'benchmark_returns', 'equity_weight' (the
    daily target weight the strategy held), all as pd.Series aligned on date.
    """
    equity_tickers = [c for c in prices.columns if c != bond_ticker]
    returns = prices.pct_change().dropna()

    equity_basket_returns = returns[equity_tickers].mean(axis=1)
    bond_returns = returns[bond_ticker]

    # Shift regime labels by 1 day: you can only act on YESTERDAY's confirmed regime.
    target_weight = regime_labels.map(ALLOCATION_RULE).shift(1)
    target_weight = target_weight.reindex(returns.index).ffill().fillna(1.0)

    strategy_gross = target_weight * equity_basket_returns + (1 - target_weight) * bond_returns

    # Transaction cost: charged only when target weight changes (rebalancing)
    turnover = target_weight.diff().abs().fillna(0)
    costs = turnover * (TRANSACTION_COST_BPS / 10000)
    strategy_net = strategy_gross - costs

    benchmark_returns = equity_basket_returns  # 100% equity, buy & hold

    return {
        "strategy_returns": strategy_net.dropna(),
        "benchmark_returns": benchmark_returns.dropna(),
        "equity_weight": target_weight,
    }


def rolling_sharpe(returns: pd.Series, window=63) -> pd.Series:
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std) * np.sqrt(252)
