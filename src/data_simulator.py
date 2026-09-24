"""
data_simulator.py
------------------
Generates realistic, regime-switching multi-asset price data.

Why simulate instead of downloading?
This project is designed to run fully offline and reproducibly (important for
a portfolio project graders / recruiters will re-run). The simulator uses a
Markov-switching Geometric Brownian Motion model, which is the same class of
model researchers use to STRESS-TEST real strategies before deploying them
on live data.

Swapping to real data is a one-line change: `load_real_data()` shows how to
pull actual prices with yfinance if the user has an internet connection.
Everything downstream (regime detection, backtesting, risk metrics) works
identically on either source, because both return the same schema:
    DataFrame indexed by date, one column per ticker, values = adjusted close.
"""

import numpy as np
import pandas as pd


def _regime_path(n_days, seed, avg_durations=(190, 65, 35)):
    """
    Generate a hidden Markov chain of regimes with realistic, state-specific
    persistence (bull markets run for years, crises burn out in weeks/months)
    and realistic transitions (markets usually pass through a "choppy" state
    on the way in or out of a crisis, rather than jumping straight from
    Calm Bull to Crisis).
    """
    rng = np.random.default_rng(seed)
    n_states = len(avg_durations)
    p_stay = [1 - 1 / d for d in avg_durations]

    # Hand-specified transition preferences (rows = FROM, cols = TO), off-diagonal
    # split according to realistic regime adjacency: Calm Bull rarely jumps
    # straight to Crisis; it typically transitions through Choppy first.
    trans = np.array([
        [p_stay[0], (1 - p_stay[0]) * 0.85, (1 - p_stay[0]) * 0.15],  # from Calm Bull
        [(1 - p_stay[1]) * 0.5, p_stay[1], (1 - p_stay[1]) * 0.5],     # from Choppy
        [(1 - p_stay[2]) * 0.15, (1 - p_stay[2]) * 0.85, p_stay[2]],   # from Crisis
    ])

    path = np.zeros(n_days, dtype=int)
    state = 0  # markets start calm
    for t in range(n_days):
        path[t] = state
        state = rng.choice(n_states, p=trans[state])
    return path


def simulate_market(
    tickers=("TECH_CO", "BANK_CO", "ENERGY_CO", "HEALTH_CO", "BOND_ETF"),
    start="2015-01-01",
    n_days=2500,
    seed=7,
):
    """
    Simulate daily adjusted-close prices for a small multi-asset universe
    under a 3-regime Markov-switching model:
        0 = Calm Bull   (low vol, positive drift)
        1 = Choppy/Neutral (medium vol, near-zero drift)
        2 = Crisis/Bear (high vol, negative drift, fat tails)

    Returns
    -------
    prices : pd.DataFrame  (date index, one column per ticker)
    true_regimes : pd.Series (date index) -- ground truth, used only to
        validate the unsupervised regime detector later (in real life you
        would never have this column -- it's our "answer key").
    """
    dates = pd.bdate_range(start=start, periods=n_days)
    regimes = {
        0: {"mu": 0.00055, "sigma": 0.0078, "label": "Calm Bull"},
        1: {"mu": 0.00010, "sigma": 0.0125, "label": "Choppy/Neutral"},
        2: {"mu": -0.00045, "sigma": 0.0210, "label": "Crisis/Bear"},
    }
    regime_path = _regime_path(n_days, seed)

    rng = np.random.default_rng(seed)
    n_assets = len(tickers)

    # Each asset has its own beta to the "market regime factor" + idiosyncratic noise
    betas = rng.uniform(0.7, 1.3, size=n_assets)
    idio_vol = rng.uniform(0.004, 0.010, size=n_assets)
    base_drift = rng.uniform(-0.00005, 0.00015, size=n_assets)

    log_returns = np.zeros((n_days, n_assets))
    for t in range(n_days):
        r = regimes[regime_path[t]]
        # occasional fat-tail jump during crisis regime (mimics real crashes),
        # sized so a full crisis episode produces a realistic 25-45% drawdown
        # rather than an implausible near-total wipeout.
        jump = 0.0
        if regime_path[t] == 2 and rng.random() < 0.05:
            jump = rng.normal(-0.02, 0.010)
        market_factor = rng.normal(r["mu"], r["sigma"])
        for i in range(n_assets):
            idio = rng.normal(0, idio_vol[i])
            log_returns[t, i] = base_drift[i] + betas[i] * (market_factor + jump) + idio

    # Bond ETF behaves differently: lower vol, mild negative correlation to crisis regime
    bond_idx = n_assets - 1 if "BOND" in tickers[-1] else None
    if bond_idx is not None:
        for t in range(n_days):
            flight_to_safety = 0.0015 if regime_path[t] == 2 else 0.0
            log_returns[t, bond_idx] = rng.normal(0.00015 + flight_to_safety, 0.0035)

    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    price_df = pd.DataFrame(prices, index=dates, columns=tickers)
    true_regimes = pd.Series(
        [regimes[s]["label"] for s in regime_path], index=dates, name="true_regime"
    )
    return price_df, true_regimes


def load_real_data(tickers, start="2015-01-01", end=None):
    """
    OPTIONAL: pulls real adjusted-close prices with yfinance.
    Requires internet access and `pip install yfinance`.
    Not used by default so the project runs offline/reproducibly, but this
    is the drop-in replacement for simulate_market() when you want to run
    the exact same pipeline on live market data.
    """
    import yfinance as yf

    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna(how="all")


if __name__ == "__main__":
    prices, regimes = simulate_market()
    print(prices.head())
    print(regimes.value_counts())
