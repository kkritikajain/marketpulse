"""
visualize.py
------------
All chart-generation for the project. Every function saves a PNG to the
given output directory and returns the filepath, so main.py can collect
paths and hand them to the report builder.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid", context="talk")
REGIME_COLORS = {
    "Calm Bull": "#2E7D32",
    "Choppy/Neutral": "#F9A825",
    "Crisis/Bear": "#C62828",
}


def _save(fig, outdir, name):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_regime_overlay(index_level: pd.Series, regime_labels: pd.Series, outdir, name="01_regime_overlay.png"):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(index_level.index, index_level.values, color="black", linewidth=1.1, zorder=3)

    aligned = regime_labels.reindex(index_level.index).ffill()
    start = aligned.index[0]
    cur = aligned.iloc[0]
    for i in range(1, len(aligned)):
        if aligned.iloc[i] != cur or i == len(aligned) - 1:
            end = aligned.index[i]
            ax.axvspan(start, end, color=REGIME_COLORS.get(cur, "grey"), alpha=0.18, zorder=0)
            start = end
            cur = aligned.iloc[i]

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for c in REGIME_COLORS.values()]
    ax.legend(handles, REGIME_COLORS.keys(), loc="upper left", frameon=True, title="Detected Regime")
    ax.set_title("Market Index with Detected Regimes", fontsize=17, weight="bold")
    ax.set_ylabel("Index Level (Base = 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_cumulative_performance(strategy_returns, benchmark_returns, outdir, name="02_cumulative_performance.png"):
    strat_wealth = (1 + strategy_returns).cumprod() * 100
    bench_wealth = (1 + benchmark_returns).cumprod() * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(strat_wealth.index, strat_wealth.values, label="Regime-Aware Strategy", color="#1565C0", linewidth=2)
    ax.plot(bench_wealth.index, bench_wealth.values, label="Buy & Hold Benchmark", color="#9E9E9E", linewidth=2, linestyle="--")
    ax.set_title("Growth of $100: Strategy vs. Benchmark", fontsize=17, weight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_drawdowns(strategy_returns, benchmark_returns, outdir, name="03_drawdowns.png"):
    from risk_metrics import drawdown_series

    strat_dd = drawdown_series(strategy_returns) * 100
    bench_dd = drawdown_series(benchmark_returns) * 100

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(strat_dd.index, strat_dd.values, 0, color="#1565C0", alpha=0.4, label="Strategy Drawdown")
    ax.fill_between(bench_dd.index, bench_dd.values, 0, color="#9E9E9E", alpha=0.3, label="Benchmark Drawdown")
    ax.set_title("Drawdown Comparison", fontsize=17, weight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_rolling_sharpe(strategy_returns, benchmark_returns, outdir, name="04_rolling_sharpe.png"):
    from strategy import rolling_sharpe

    strat_rs = rolling_sharpe(strategy_returns)
    bench_rs = rolling_sharpe(benchmark_returns)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(strat_rs.index, strat_rs.values, label="Regime-Aware Strategy", color="#1565C0")
    ax.plot(bench_rs.index, bench_rs.values, label="Buy & Hold Benchmark", color="#9E9E9E", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling 3-Month Sharpe Ratio", fontsize=17, weight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_correlation_heatmap(prices, outdir, name="05_correlation_heatmap.png"):
    returns = prices.pct_change().dropna()
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Asset Return Correlation Matrix", fontsize=15, weight="bold")
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_regime_feature_scatter(features, regime_labels, outdir, name="06_regime_clusters.png"):
    aligned = regime_labels.reindex(features.index)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for regime, color in REGIME_COLORS.items():
        mask = aligned == regime
        ax.scatter(features.loc[mask, "roll_vol"], features.loc[mask, "roll_mean"],
                   s=14, alpha=0.55, color=color, label=regime)
    ax.set_xlabel("21-Day Rolling Volatility")
    ax.set_ylabel("21-Day Rolling Mean Return")
    ax.set_title("Regime Clusters in Feature Space", fontsize=15, weight="bold")
    ax.legend(frameon=True)
    fig.tight_layout()
    return _save(fig, outdir, name)


def plot_confusion_vs_truth(true_regimes, detected_regimes, outdir, name="07_detection_accuracy.png"):
    """Compares detected regimes to the simulator's ground truth (validation only)."""
    aligned_true = true_regimes.reindex(detected_regimes.index)
    ct = pd.crosstab(aligned_true, detected_regimes, normalize="index") * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={"label": "% of days"})
    ax.set_xlabel("Detected Regime (unsupervised)")
    ax.set_ylabel("True Simulated Regime")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_title("Regime Detection Validation", fontsize=15, weight="bold")
    fig.tight_layout()
    return _save(fig, outdir, name)
