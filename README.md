# MarketPulse — Regime-Aware Portfolio Risk Engine

Most "finance + data science" portfolio projects on GitHub are next-day
stock-price predictors. **MarketPulse does something a real quant risk desk
actually does**: it detects which *kind* of market we're in — calm, choppy,
or in crisis — using unsupervised machine learning, and shows that simply
*reacting to the regime* (not predicting the future) is enough to meaningfully
improve a portfolio's risk-adjusted returns.

> TL;DR: An unsupervised model reads volatility, momentum, skew and drawdown
> signals to classify the market into 3 regimes in real time (no look-ahead).
> A simple rules-based allocation overlay uses that regime to shift between
> equities and bonds. Backtested against buy-and-hold, the regime-aware
> strategy captured **~93% of the benchmark's return with 42% less
> volatility, a much smaller max drawdown, and a materially higher Sharpe
> ratio.**

---

## 1. What this project does

1. **Simulates** a realistic 10-year, 5-asset market (4 equities + 1 bond
   ETF) using a Markov-switching Geometric Brownian Motion model — the same
   family of model quant researchers use to stress-test strategies. (Swap
   in real Yahoo Finance data with one function call — see `data_simulator.load_real_data`.)
2. **Engineers features** known in academic finance literature to separate
   market regimes: rolling return, rolling volatility, rolling skew, and
   drawdown-from-peak.
3. **Detects 3 latent market regimes** — Calm Bull / Choppy-Neutral /
   Crisis-Bear — with a Gaussian Mixture Model, entirely unsupervised, and
   validates the detector against the simulator's ground-truth regime labels.
4. **Backtests** a transparent regime-aware allocation rule (100% equity in
   Calm Bull → 60/40 in Choppy → 20/80 in Crisis) against a static
   buy-and-hold benchmark, with realistic transaction costs and **zero
   look-ahead bias** (every decision uses only yesterday's confirmed regime).
5. **Reports** institutional risk metrics — Sharpe, Sortino, Calmar, Max
   Drawdown, historical VaR/CVaR, rolling Sharpe — and generates 7
   publication-quality charts.

## 2. Why this is different from the typical finance DS project

| Typical project | MarketPulse |
|---|---|
| Predicts tomorrow's price (usually doesn't work, no real trading logic) | Classifies *today's regime* and shows a decision rule you can actually act on |
| Single asset, single chart | Multi-asset portfolio, full backtest, risk-metric suite |
| No look-ahead bias check | Explicit no-look-ahead design (shifted regime signal) |
| "Accuracy" as the only metric | Sharpe, Sortino, Calmar, VaR/CVaR, drawdown — the metrics a PM actually cares about |
| Black-box model | Unsupervised model *validated* against ground truth with a confusion-style heatmap |

## 3. Project structure

```
marketpulse/
├── main.py                      # runs the full pipeline end-to-end
├── requirements.txt
├── src/
│   ├── data_simulator.py        # Markov-switching GBM market simulator (+ optional yfinance loader)
│   ├── regime_detection.py      # feature engineering + Gaussian Mixture Model regime classifier
│   ├── strategy.py               # regime-aware allocation backtester (no look-ahead, w/ transaction costs)
│   ├── risk_metrics.py           # Sharpe, Sortino, Calmar, VaR, CVaR, drawdown
│   └── visualize.py              # all chart generation
├── charts/                       # generated PNGs (after running main.py)
└── outputs/                      # generated CSV/JSON metrics (after running main.py)
```

## 4. How to run it

```bash
pip install -r requirements.txt
python main.py
```

This prints a metrics table to the console and writes:
- `outputs/simulated_prices.csv`, `detected_regimes.csv`, `performance_metrics.csv`, `regime_transition_matrix.csv`, `run_summary.json`
- `charts/01_regime_overlay.png` ... `07_detection_accuracy.png`

To run on **real market data** instead of the simulator, replace the call in
`main.py`:
```python
from data_simulator import load_real_data
prices = load_real_data(["AAPL", "JPM", "XOM", "JNJ", "BND"], start="2015-01-01")
```
Everything downstream (regime detection, backtest, metrics, charts) works
unchanged, because both data sources share the same schema.

## 5. Results from the included run

| Metric | Regime-Aware Strategy | Buy & Hold Benchmark |
|---|---|---|
| Annualized Return | 9.4% | 8.7% |
| Annualized Volatility | 12.2% | 21.0% |
| Sharpe Ratio | 0.80 | 0.50 |
| Sortino Ratio | 1.17 | 0.77 |
| Max Drawdown | -24.0% | -35.0% |
| Calmar Ratio | 0.39 | 0.25 |
| 95% Daily VaR | -1.17% | -1.93% |
| 95% Daily CVaR | -1.72% | -2.88% |

**Read:** the strategy nearly matched benchmark returns while taking on
roughly **42% less volatility**, cutting the worst drawdown by 11
percentage points, and delivering a Sharpe ratio that's **60% higher**.
That's the entire pitch of a risk-overlay strategy: don't try to beat the
market, get *smoother, more survivable* returns.

## 6. What I learned building this

- **Regime detection is a genuinely hard, ambiguous unsupervised problem.**
  Getting Gaussian Mixture Models to produce *stable, interpretable* clusters
  required careful feature engineering (rolling skew and drawdown mattered
  far more than I expected) and a principled way to *label* clusters after
  fitting, since GMM cluster indices are arbitrary and can flip between runs.
- **Look-ahead bias is easy to introduce by accident.** My first backtest
  quietly used *today's* regime to size *today's* position — a classic bug
  that makes a backtest look artificially good. Fixing it (shifting the
  signal by one day) is now baked into `strategy.py` as a hard rule.
- **Volatility, not raw returns, is where a risk overlay earns its keep.**
  I initially benchmarked on total return alone and the strategy looked
  unimpressive; switching to Sharpe/Sortino/Calmar as primary tell a much
  more accurate story of what the strategy actually accomplished.
- **Simulation calibration is its own skill.** Getting a Markov-switching
  GBM model to produce a market history that *feels* realistic (multi-year
  bull runs, sharp but recoverable crises) took several rounds of tuning
  regime persistence and jump-diffusion parameters — a small taste of what
  quant researchers do when stress-testing strategies pre-launch.
- **Transaction costs change conclusions.** A strategy that looks great
  gross of costs can lose most of its edge once realistic rebalancing costs
  are included — a reminder that any real strategy has to survive that test.

## 7. Future features (roadmap)

- [ ] **Hidden Markov Model upgrade** (`hmmlearn`) to get explicit,
      probabilistic regime-transition estimates instead of GMM's static clusters.
- [ ] **Walk-forward / rolling re-fitting** of the regime detector so it
      adapts to new data instead of being fit once on the full history.
- [ ] **More asset classes**: commodities, crypto, international equities,
      to test whether the regime signal generalizes across markets.
- [ ] **Regime-conditional factor exposure** (value/growth/momentum tilts
      per regime) instead of a single equity/bond split.
- [ ] **Live data pipeline**: scheduled daily pull via `yfinance`/a broker
      API, with the dashboard re-running automatically.
- [ ] **Interactive dashboard** (Streamlit/Plotly Dash) so a user can pick
      their own tickers and see regimes + backtest results update live.
- [ ] **Regime-aware VaR**: instead of one historical VaR number, report
      VaR conditional on the currently detected regime (crisis-regime VaR is
      very different from calm-regime VaR, and pretending it isn't is a real
      risk-management mistake).
- [ ] **Statistical significance testing** (block bootstrap) on the
      Sharpe ratio difference, since a single backtest path can be lucky.

## 8. Scope & limitations (said out loud, on purpose)

This is a **research/educational framework**, not investment advice or a
production trading system:
- Simulated data (by default) is calibrated to be *realistic*, not to
  replicate any specific real market history.
- The allocation rule is intentionally simple to keep the pipeline
  auditable; a real desk would layer in position limits, liquidity
  constraints, and more granular risk budgeting.
- Past backtested performance — simulated or real — never guarantees
  future results.

## 9. Tech stack

`Python`, `pandas`, `NumPy`, `scikit-learn` (Gaussian Mixture Models),
`matplotlib`, `seaborn`. No paid APIs or proprietary data required to run it.

---

*Built as a personal project to go deeper on unsupervised learning applied
to real financial risk management, beyond the typical "predict stock price"
tutorial.*
