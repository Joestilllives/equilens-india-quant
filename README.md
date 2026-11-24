# EquiLens India Quant (`V6`) — NSE Quantitative ML Shortlister, Risk Engine & Paper-Trading Workstation

EquiLens India Quant is a full-stack Indian stock market (**NSE / NIFTY 50 / INR `₹`**) quantitative analysis, walk-forward machine learning, stress-testing, and forward-proof paper-trading platform built with **FastAPI**, **SQLite (WAL)**, **yfinance**, **scikit-learn**, and a zero-build **Vanilla HTML/CSS/JS** canvas UI.

---

## Core Architecture (`V6` Institutional India Edition)

### 1. Live NSE Data & Global Intermarket Macro (`backend/config.py`, `backend/data_fetcher.py`)
- **131 Liquid NSE Leaders + `^NSEI` (NIFTY 50)** × up to 10 years of daily OHLCV (`261,801` labeled OOS-eligible observations), quarterly earnings surprises, promoter/insider disclosures, and recency-weighted Indian market catalyst news (`backend/catalyst_nlp.py`).
- **7 Global Intermarket & India Macro Series**: India VIX (`^INDIAVIX`), USD/INR (`INR=X`), Brent Crude (`BZ=F`), Gold (`GC=F`), S&P 500 (`^GSPC`), US 10Y Yield (`^TNX`), and DXY (`DX-Y.NYB`).

### 2. Walk-Forward `V6` Sentiment & Intermarket Macro ML Engine (`backend/ml_predictor.py`)
- **19 Strictly Causal Features**:
  - Official-style NSE 6M-1M & 12M-1M volatility-adjusted momentum, Frog-in-the-Pan (FIP) smoothness, 52W high proximity, 1Y & downside volatility, sector-neutral relative alpha.
  - **Stock-Specific Rolling Macro Sensitivities** (`126D` stock beta × `63D` trend for Brent Crude, USD/INR, and S&P 500).
  - **Behavioral Sentiment**: `MAX` 21D retail-lottery hype avoidance (Bali, Cakici & Whitelaw, *JFE* 2011) and `63D` Overnight Institutional Gap Sentiment (Lou, Polk & Skouras, *JFE* 2019).
- **Hybrid Ranking Architecture**: `50%` NSE Institutional Multi-Factor Prior + `50%` Regularized 40D Cross-Sectional Rank-Target ML (`Ridge(alpha=500)` + shallow `HistGradientBoostingRegressor`), validated across **5 expanding chronological folds with a strict 40-day embargo gap**.
- **Calibrated Symmetric Payoff Classifier**: Binary classifiers trained strictly on cross-sectional stock features (`C = 0.05`) predicting symmetric 40-day `+2.0×ATR` vs. `-2.0×ATR` payoff odds + 60-day log-normal probability cones.

### 3. `V6` 4-Tranche Correlation-Penalized Portfolio Engine (`backend/portfolio_backtester.py`)
- **Top-10 NSE Portfolio (`Max 2 per NSE Sector`, ≥ 5 sectors)** with `2×K` (`Top-20`) rank-buffer hysteresis to minimize Indian STT & 20% STCG tax churn.
- **4-Tranche Staggered Overlapping 40D Sleeves**: Splits capital into 4 equal 25% sleeves staggered 10 trading days apart, eliminating rebalance-date calendar luck with zero extra turnover per rupee.
- **Correlation-Penalized Risk Weighting**: `w_i ∝ 1 / (vol_i × (1 + avg_pos_corr_i))` over trailing 126D pairwise correlations.
- **6-Gauge India Sentiment & Macro Risk Gate + Ex-Ante `18%` Volatility Target**: Dynamically scales equity exposure (`100% / 85% / 55% / 25%`) based on India VIX, NIFTY 50 vs. 200-DMA, NSE Breadth, NSE 52W Hi-Lo Fear Sentiment, Gold/NIFTY 3M flight-to-safety ratio, and trailing 60D portfolio covariance. Idle cash uses a 70% `LIQUIDCASE` (Growth Liquid ETF, ~6.2% p.a. yield) / 30% `GOLDBEES` defensive reserve split.

### 4. Live Execution Ticket, Earnings Blackout Guard & Paper-Trading Ledger (`backend/execution_engine.py`)
- Generates exact integer NSE share quantities, rupee allocations (`₹`), `LIQUIDCASE` (70%) / `GOLDBEES` (30%) cash reserve, `±2.0×ATR` target/stop prices, Earnings Blackout flags (`≤ 5` trading days to quarterly results), and Indian 15 bps STT + 20% STCG tax estimates.
- **SQLite Forward-Proof Paper Ledger**: Locks timestamped live NSE fills in `data/equilens_india.db` and marks daily P&L forward against **NIFTY 50** (zero survivorship bias).

---

## Out-of-Sample Research & Audit Scripts (`scripts/`)
- `scripts/run_monte_carlo.py` — 10,000-path paired 10-day block-bootstrap Monte Carlo (`0%`, `-4%/yr`, `-8%/yr` real-world haircuts).
- `scripts/audit_overfitting_v6.py` — 6-test institutional audit: Deflated Sharpe Ratio (DSR), 131-stock survivorship decomposition, year-by-year consistency, 100-trial random stock-selection permutation `p-value`, synthetic `-70%` blowup stress, and 12-cell parameter plateau.
- `scripts/research_india_round*.py` — Full ablation history from `R1` through `V6`.

---

## Running Locally

`data/equilens_india.db` is pre-populated with all 131 NSE stocks, 7 global macro series, the trained `V6` model, and the locked ₹10,00,000 paper-trading ledger.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

## Running Tests & Stress Suite

```bash
.venv/bin/pytest -v -s
```

> **Disclaimer**: Quantitative research & educational tool only — not SEBI-registered investment advice. Equity markets carry risk, historical backtests contain survivorship bias, and past out-of-sample results do not guarantee future returns.
