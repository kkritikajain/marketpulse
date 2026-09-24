"""
main.py
-------
End-to-end pipeline for MarketPulse:
  1. Simulate (or load) multi-asset market data
  2. Detect market regimes with an unsupervised Gaussian Mixture Model
  3. Backtest a regime-aware allocation strategy vs. a buy & hold benchmark
  4. Compute institutional-grade risk metrics for both
  5. Generate all charts and a metrics summary (CSV + JSON) into /outputs

Run with:  python main.py
"""

import json
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_simulator import simulate_market
from regime_detection import detect_regimes, regime_transition_matrix
from strategy import backtest_regime_strategy
from risk_metrics import summary_table
import visualize as viz

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
CHARTDIR = os.path.join(os.path.dirname(__file__), "charts")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CHARTDIR, exist_ok=True)

    print("Step 1/5: Simulating multi-asset market data...")
    prices, true_regimes = simulate_market()
    prices.to_csv(os.path.join(OUTDIR, "simulated_prices.csv"))

    print("Step 2/5: Detecting market regimes (Gaussian Mixture Model)...")
    detected_regimes, features, gmm_model, index_level = detect_regimes(prices)
    detected_regimes.to_csv(os.path.join(OUTDIR, "detected_regimes.csv"))
    trans_matrix = regime_transition_matrix(detected_regimes)
    trans_matrix.to_csv(os.path.join(OUTDIR, "regime_transition_matrix.csv"))

    print("Step 3/5: Backtesting regime-aware strategy vs. buy & hold...")
    bt = backtest_regime_strategy(prices, detected_regimes)
    strat_returns = bt["strategy_returns"]
    bench_returns = bt["benchmark_returns"]

    print("Step 4/5: Computing risk & performance metrics...")
    metrics = summary_table({
        "Regime-Aware Strategy": strat_returns,
        "Buy & Hold Benchmark": bench_returns,
    })
    metrics.to_csv(os.path.join(OUTDIR, "performance_metrics.csv"))
    print(metrics.round(4))

    print("Step 5/5: Generating charts...")
    chart_paths = []
    chart_paths.append(viz.plot_regime_overlay(index_level, detected_regimes, CHARTDIR))
    chart_paths.append(viz.plot_cumulative_performance(strat_returns, bench_returns, CHARTDIR))
    chart_paths.append(viz.plot_drawdowns(strat_returns, bench_returns, CHARTDIR))
    chart_paths.append(viz.plot_rolling_sharpe(strat_returns, bench_returns, CHARTDIR))
    chart_paths.append(viz.plot_correlation_heatmap(prices, CHARTDIR))
    chart_paths.append(viz.plot_regime_feature_scatter(features, detected_regimes, CHARTDIR))
    chart_paths.append(viz.plot_confusion_vs_truth(true_regimes, detected_regimes, CHARTDIR))

    # Save a small run summary used later by the PDF report generator
    run_summary = {
        "n_days": len(prices),
        "date_range": [str(prices.index.min().date()), str(prices.index.max().date())],
        "tickers": list(prices.columns),
        "metrics": json.loads(metrics.round(4).to_json(orient="index")),
        "regime_day_counts": detected_regimes.value_counts().to_dict(),
        "chart_paths": chart_paths,
    }
    with open(os.path.join(OUTDIR, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    print("\nDone. Charts saved to:", CHARTDIR)
    print("Metrics & data saved to:", OUTDIR)


if __name__ == "__main__":
    main()
