# EquiLens India Quant (`V7` Unified) — NSE Quantitative ML Shortlister, HRP Portfolio Engine & Paper-Trading Workstation

EquiLens India Quant is a full-stack Indian stock market (**NSE / NIFTY 50 TRI / INR `₹`**) quantitative research, walk-forward machine learning, institutional audit, and forward-proof paper-trading platform built with **FastAPI**, **SQLite (WAL)**, **yfinance**, **scikit-learn**, **SciPy**, and a zero-build **Vanilla HTML/CSS/JS** UI.

> **Branch Map:**
> - **`main` (`5f303af`):** Original `V6` strategy + live safety fixes & ETF-leakage hotfix (untouched V6 alpha).
> - **`feat/v7-book-research` (Active Live Production Engine):** **Unified Frozen `V7`** — combines the best of `V6` (risk shields), `V6.1` (hardened accounting, TRI, next-open chart backtester, sector-normalized shortlist, UI freshness badges), `V7` (canonical finance-book alpha + HRP + 100% free Day-1 PIT pipelines), and `V8`'s non-alpha institutional toolkit (IIMA 4-Factor Indian Alpha, Indian Forensic/QMJ shield, RBI DBIE telemetry, opt-in `₹1 Cr+` Boyd Convex Optimizer).
> - **`feat/v8-frontier-research`:** Experimental `V8` ML/regime ablations (tested against `V7`; see `PLAYBOOK.md` Section 7.6 & Section 8).

---

## Core Architecture (`V7` Unified Institutional India Edition)

### 1. Live NSE Data, Non-Equity Hygiene & 100% Free PIT Pipelines (`backend/config.py`, `backend/data_fetcher.py`, `backend/pit_fundamentals.py`)
- **130 Liquid NSE Leaders + `^NSEI` (NIFTY 50) + `NIFTYBEES.NS` (TRI Proxy) + `GOLDBEES.NS` (30% Gold Reserve)** × 10 years of daily OHLCV, quarterly earnings, Yahoo insider proxy, and recency-weighted Indian catalyst news (`backend/catalyst_nlp.py`).
- **Strict Non-Equity Exclusion (`NON_EQUITY_SYMBOLS`):** `^NSEI`, `NIFTYBEES.NS`, and `GOLDBEES.NS` are excluded from ML training, predictions, breadth gauges, shortlist, and equity holdings while still auto-refreshing on **Sync & Retrain**.
- **100% Free Phase-2 Point-In-Time (PIT) Pipelines:** Auto-records immutable daily `pit_fundamentals_history` snapshots on every sync, plus parsers/importers for NSE Bhavcopy delivery archives (`nsearchives.nseindia.com`), BSE/NSE quarterly XBRL (`+2` NSE trading-day lag), SEBI shareholding/promoter-pledge patterns (`backend/india_structural_signals.py`), Indian Forensic Governance & QMJ (`backend/forensic_quality_sleeve.py`), and RBI DBIE domestic liquidity CSVs (`backend/india_liquidity_regime.py`).

### 2. Walk-Forward `V7` Canonical Book ML Engine (`backend/ml_predictor.py`)
- **Canonical Finance & Behavioral Signals (Zero Lookahead):**
  - **CAPM Residual Momentum (`12-1M`, `6-1M`)** (*Blitz, Huij & Martens 2011; Ilmanen*), **Intermediate Momentum (`12-7M`)** (*Novy-Marx 2012*), **Capital Gains Overhang (`CGO 126D` via VWAP)** (*Grinblatt & Han 2005; Kahneman & Thaler*), **Betting Against Beta (`-Vasicek β`)** (*Frazzini & Pedersen 2014*), **Daniel–Moskowitz (2016) Bear-Rebound Crash Damper** (`NIFTY 252D < 0 & VIX ≥ 19`), and **Pollet–Wilson (2010) `avg_corr_63`** telemetry.
  - Preserves proven `V6` survivors (FIP smoothness, 1Y & downside vol, 3M sector-relative alpha, MAX lottery avoidance, overnight institutional sentiment, FX/S&P 500 tailwinds); prunes noisy/collinear features (`sec_rel_126d`, `vol_ratio_5_20`, `macro_tailwind_brent`).
- **Hybrid Ranking Architecture:** `50%` Canonical Multi-Factor Prior + `50%` Regularized 40D **Vasicek Beta-Adjusted Residual Sharpe** ML (`Ridge` on cross-sectional features only + shallow `HistGradientBoostingRegressor` with macro interactions), evaluated across **5 expanding chronological folds with a strict 40-day embargo**.

### 3. `V7` 4-Tranche `HRP` Portfolio & Hardened Indian Accounting (`backend/hrp_allocator.py`, `backend/portfolio_backtester.py`, `backend/indian_costs.py`)
- **López de Prado (2016) Hierarchical Risk Parity (`HRP`)** on 126D Ledoit-Wolf shrunk covariance (`weighting_mode="hrp"` default; opt-in Boyd `weighting_mode="convex"` available in `backend/convex_optimizer.py` for `₹1 Cr+` scale).
- **Preserved `V6` Risk Shields:** 4 staggered 25% sleeves (10 trading days apart), Top-10 / Hold Top-30 buffer, max 2 stocks/sector, hard `≤12%` stock / `≤20%` sector caps, 6-Gauge Stress Gate (`100%/85%/55%/25%`), `18%` Vol Target (60D sample cov), one-way intra-cycle de-risking, Peak-NAV `-10%/-6%` Drawdown Brake, and `70/30` `LIQUIDCASE`/`GOLDBEES` reserve.
- **Hardened Indian Accounting & Multi-Factor Attribution:** Exact side-explicit NSE statutory delivery fees + square-root ADV impact, transaction-level FIFO STCG (`20.8%`) / LTCG (`13.0%` above `₹1.25L/FY`) tax ledger, `NIFTYBEES.NS` TRI benchmark, Excess Sharpe above `6.2%` India cash, 2-Factor Pure Alpha, and **IIMA 4-Factor Indian Alpha (`MKT + SMB + HML + WML`)**.

### 4. Live Execution Ticket, Security & 4-Tranche Paper Ledger (`backend/execution_engine.py`, `backend/main.py`)
- Exact integer NSE share quantities, `70/30` `LIQUIDCASE`/`GOLDBEES` reserve, Earnings Blackout guard (`≤5d`), `<35%` NSE Delivery Shield, `>2%` 20D ADV liquidity cap warning, data-freshness fail-safe (`max_ffill_days=2`), and localhost API mutation auth (`X-EquiLens-Token`) + restricted CORS + explicit reset confirmation.
- **SQLite Forward-Proof Paper Ledger:** Tracks persistent FIFO `paper_lots`, 4 staggered `paper_sleeves`, `peak_nav_inr` drawdown brake state, and `NIFTYBEES.NS` TRI comparison.

---

## Verified Frozen `V7` Scorecard (See `PLAYBOOK.md` Section 7 for Full History)
- **Core OOS (`2021–2026`, `NIFTYBEES.NS` TRI `+9.09%` / Post-Tax `+8.25%`):** **`+26.48%` Pre-Tax** | **`+23.71%` FIFO Post-Tax** | **`SR(6.2%) 1.48`** (`0% RF = 1.99`) | **`MaxDD -12.18%`** | **2F Pure Alpha `+8.24%/yr` (`t=+2.18`)** | **IIMA 4F Alpha `+7.05%` to `+8.53%/yr`**.
- **Stress & Overfitting Proofs:** **`50/50` permutation null wins (`p = 0.0196`)**, **`95.49%` Corrected DSR (`N=50, 6.2% RF`)**, **`+24.57%` Post-Tax (`SR 1.59`) on 17 real NSE Fallen Angels**, **`9/9`** wins over 8.8Y, **`19Y` (`2007–2026`)** multi-cycle win, and **`+22.10%` Post-Tax** even at `50 bps/side` extreme friction.

---

## Out-of-Sample Research & Audit Scripts (`scripts/`)
- `scripts/audit_overfitting_v6.py` — Hardened `V7` audit: TRI/EqW decomposition, calendar-year consistency, 12-cell plateau + CPCV/PBO, matched 4-tranche Top-10 permutation null test, and corrected Bailey–López de Prado DSR.
- `scripts/validate_v7_multicycle.py` — Full 5-Gate zero-overfitting promotion harness (`V6` vs. `V7` across OOS, 8.8Y, 19Y `2007–2026`, 9-cell plateau, 50bps stress, and execution checks).
- `scripts/test_survivorship_fallen_angels.py` — 17 real Indian Fallen Angels survivorship stress test + Amihud diagnostic.
- `scripts/run_monte_carlo.py` — 10,000-path block-bootstrap Monte Carlo (`V7` median 5Y projected CAGR: `+26.2%` pre-tax / `+23.4%` post-tax; conservative real-world cycle expectation: `~+16% to +20%` pre-tax / `~+14% to +17%` post-tax).

---

## Running Locally

`data/equilens_india.db` is pre-populated with 10Y NSE equity + TRI/Gold/macro history, the trained `V7` model, Day-1 PIT fundamental snapshots, and the live ₹10,00,000 4-tranche paper-trading ledger.

```bash
python3 -m venv .venv
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

## Running Tests & Stress Suite

```bash
.venv/bin/pytest -v -s
```

> **Disclaimer**: Quantitative research & educational tool only — not SEBI-registered investment advice. Equity markets carry risk, historical backtests contain survivorship bias, and past out-of-sample results do not guarantee future returns.
