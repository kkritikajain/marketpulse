"""
regime_detection.py
--------------------
Unsupervised detection of market regimes (Calm Bull / Choppy / Crisis) from
a basket of asset prices, using only information available in real time
(no look-ahead bias).

Method
------
1. Build an equal-weight market index from the asset universe.
2. Engineer features that are known in the finance literature to separate
   regimes well:
      - rolling mean return (momentum / drift)
      - rolling realized volatility
      - rolling skewness (crisis regimes are left-skewed)
      - drawdown from the trailing peak
3. Fit a Gaussian Mixture Model (3 components) on the standardized features.
   GMM is used instead of a full Hidden Markov Model to keep the dependency
   footprint small (scikit-learn only); the SKILL.md "future work" section
   in the README notes upgrading to `hmmlearn` for explicit transition
   probabilities.
4. Label the 3 discovered clusters (low/med/high volatility) so labels are
   interpretable regardless of random initialization.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def build_market_index(prices: pd.DataFrame) -> pd.Series:
    """Equal-weighted index of daily returns, then rebuilt into a price level."""
    returns = prices.pct_change().dropna()
    index_returns = returns.mean(axis=1)
    index_level = 100 * (1 + index_returns).cumprod()
    return index_level


def engineer_features(index_level: pd.Series, window=21) -> pd.DataFrame:
    ret = index_level.pct_change()
    roll_mean = ret.rolling(window).mean()
    roll_vol = ret.rolling(window).std()
    roll_skew = ret.rolling(window).skew()
    running_max = index_level.cummax()
    drawdown = index_level / running_max - 1

    feats = pd.DataFrame(
        {
            "roll_mean": roll_mean,
            "roll_vol": roll_vol,
            "roll_skew": roll_skew,
            "drawdown": drawdown,
        }
    ).dropna()
    return feats


def detect_regimes(prices: pd.DataFrame, window=21, n_regimes=3, random_state=7):
    """
    Returns
    -------
    regime_labels : pd.Series of str ("Calm Bull" / "Choppy/Neutral" / "Crisis/Bear")
    features      : pd.DataFrame of the engineered features (for inspection/plots)
    model         : fitted GaussianMixture
    """
    index_level = build_market_index(prices)
    feats = engineer_features(index_level, window=window)

    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=random_state, n_init=5)
    raw_labels = gmm.fit_predict(X)

    # Order clusters by mean rolling volatility (ascending) so labels are stable
    # and interpretable: cluster with lowest vol = Calm Bull, highest = Crisis.
    vol_by_cluster = feats.groupby(raw_labels)["roll_vol"].mean().sort_values()
    order = {old: rank for rank, old in enumerate(vol_by_cluster.index)}
    label_names = {0: "Calm Bull", 1: "Choppy/Neutral", 2: "Crisis/Bear"}
    mapped = pd.Series(raw_labels, index=feats.index).map(order).map(label_names)

    return mapped.rename("detected_regime"), feats, gmm, index_level


def regime_transition_matrix(regime_labels: pd.Series) -> pd.DataFrame:
    """Empirical transition probability matrix between detected regimes."""
    labels = regime_labels.dropna()
    states = sorted(labels.unique())
    counts = pd.DataFrame(0, index=states, columns=states, dtype=float)
    prev = labels.iloc[:-1].values
    nxt = labels.iloc[1:].values
    for p, n in zip(prev, nxt):
        counts.loc[p, n] += 1
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return probs


if __name__ == "__main__":
    from data_simulator import simulate_market

    prices, true_regimes = simulate_market()
    detected, feats, model, idx = detect_regimes(prices)
    print(detected.value_counts())
    print(regime_transition_matrix(detected))
