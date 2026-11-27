# EquiLens India Quant — Final Frozen `V6` Playbook
**Locked Production Commit:** `9d09dfb` (`github.com/Joestilllives/equilens-india-quant`)  
**Status:** Strictly Frozen — Do NOT tweak parameters or add indicators.  
*(Quantitative research & forward paper-trading tool. Not SEBI-registered investment advice.)*

---

## 1. The System Architecture (What Runs Under the Hood)
* **Universe:** 130 liquid NSE large/mid-cap stocks (`NIFTY 100` + liquid `Next 50` / leaders).
* **Stock Ranker:** 50% NSE Multi-Factor Prior + 50% Regularized Expanding Walk-Forward ML (`Ridge` + shallow `HistGradientBoosting`, 40D forward alpha target, strict 40-day embargo, **19 strictly causal features** with zero US-timezone lookahead).
* **Official NSE Smart-Money & Blowup Shield (Bouncer on New Buys):**
  * Auto-blocks any new buy with **`< 35%` Official NSE Delivery %** (`nsearchives.nseindia.com`, strictly prior-day `T-1`) or **Balance-Sheet Risk (`≥ 2` red flags)**.
* **Portfolio Construction & Caps:**
  * **4-Tranche Staggering:** 4 equal sleeves rebalanced on a 40-trading-day cycle (1 tranche every 10 trading days) to eliminate start-date timing luck.
  * **Hold Buffer:** Buy **Top-10** *(or **Top-20** option for smoother swings at `₹1 Cr`)*; hold existing winners until they drop past **Rank 30** (`Top-30` Tax-Saver Buffer).
  * **Hard Concentration Caps:** Max **2 stocks/sector**, hard **`≤ 20%` per sector**, and hard **`≤ 12%` per stock** (`Top-10`) or **`≤ 5%` per stock** (`Top-20`).
  * **Position Sizing:** Correlation-penalized inverse volatility (`w_i ∝ 1 / (vol_i × (1 + avg_corr_i))`).
* **3-Layer Crash Defense:**
  1. **6-Gauge Market Risk Gate** *(India VIX `≥ 19`, India VIX `≥ 23`, NIFTY `< 200-DMA`, NSE Breadth `< 42%`, 52W Hi-Lo `< -10%`, Gold/NIFTY 63D `> +14%`)* → scales equity exposure smoothly `100% → 85% → 55% → 25%`.
  2. **18% Annualized Volatility Cap** (60D rolling portfolio covariance).
  3. **`-10%` Own-Portfolio Drawdown Brake:** Cuts equity exposure in half if portfolio drops `-10.0%` from peak NAV; releases only once drawdown recovers past **`-6.0%`** AND portfolio equity closes above its 20D SMA (`× 1.005`) (matches `DRAWDOWN_BRAKE_TRIGGER = -0.10`, `DRAWDOWN_BRAKE_RELEASE = -0.06` in code).
* **Defensive Reserve Split (`70 / 30`):**
  * Park idle / Risk-Gate cash **70% in `LIQUIDCASE`** *(Growth Liquid ETF — or an equity-taxed Arbitrage Fund; avoids `LIQUIDBEES` daily dividend ITR clutter)* and **30% in `GOLDBEES`**.

---

## 2. Verified Independent Audit Scorecard (Zero Parameter Changes)

| Independent Audit | CAGR (Pre-Tax) | Sharpe | Sortino | MaxDD | Calmar | Benchmark MaxDD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.3Y Core NSE (`2021–2026`, w/ `<35%` Shield)** | `+27.2%` *(~`+21.5%` post-tax)* | `1.98` | `2.85` | **`-9.8%`** | `2.77` | NIFTY `-17.2%` |
| **22.3Y Multi-Cycle India (`2004–2026`, incl. 2008 & 15 wipeouts)** | `+24.3%` *(no delivery CSVs)* | `1.82` | `2.65` | **`-17.0%`** | `1.43` | NIFTY `-59.9%` |
| **Blind `102` Unseen NSE Mid/Smallcaps (`0%` overlap, `2021–2026`)** | `+20.4%` *(~`+16.3%` post-tax)* | `1.50` | `2.07` | **`-11.7%`** | `1.74` | EqW `-23.9%` |
| **Blind 22.8Y USA S&P 500 (`2004–2026`, zero tuning)** | OOS IC `+0.0325` | `0.82` | — | **`-18.5%`** | `0.50` | SPY `-55.2%` |
| **Extreme Cost Stress (`50 bps/side` = `100 bps` RT)** | `+24.9%` *(~`+19.9%` post-tax)* | `1.83` | — | `-9.9%` | — | — |

---

## 3. Realistic Live Expectations at `₹1 Crore`
*(Net after brokerage, STT, slippage & ~20.8% Indian STCG tax)*
* **Realistic Base Case:** **`~+15% to +17%` net post-tax CAGR** (`₹1 Cr → ~₹2.1 Cr` in 5 yrs, `~₹4.4 Cr` in 10 yrs).
* **Conservative / Sideways Decade:** `~+12%` net post-tax CAGR (`₹1 Cr → ~₹1.76 Cr` in 5 yrs, `~₹3.11 Cr` in 10 yrs).
* **Pain You Must Expect:** Occasional **`-10% to -18%`** peak-to-trough dips (`-₹10L to -₹18L` on screen) and 6–12 month choppy flat stretches (like `2018` or `2025–2026`). Never override the system during flat months.

---

## 4. Exact Live Execution Routine
1. **Sync Window:** Run **Sync & Retrain** strictly between **`6:30 PM – 11:00 PM IST`** on NSE trading days *(avoids Yahoo Finance's ~4 AM IST settlement rollback)*.
2. **Rebalance Cadence:** Execute the **Trade Ticket** once every **10 trading days** (~every 2 weeks) for the active tranche. Ignore tiny 1–2 share micro-trims between scheduled rebalance dates.
3. **Broker Execution (Harris Ch. 14 Optimal NSE Windows):** Use **limit orders** strictly inside NSE's highest-liquidity, tightest-spread windows — **`10:00–10:30 AM IST`** (after opening auction volatility subsides) or **`2:45–3:15 PM IST`** (pre-close institutional liquidity; avoid `9:15–9:45 AM` open whipsaws and `3:25–3:30 PM` closing print). Enforce `≤ 2.0%` of 20D ADV per single order (split larger orders across 2 sessions). Buy **`LIQUIDCASE`** (or an Arbitrage Fund) for the `70%` defensive cash slot and **`GOLDBEES`** for the `30%` gold slot.
4. **Backup Habit:** After every logged paper rebalance, commit & push `data/equilens_india.db` (or an exported `paper_ledger.json`) to GitHub so VM timeouts never interrupt your track record.
5. **Capital Ramp Plan:**
   * **Months 1–3:** Log live `₹10L` paper portfolio strictly (prove zero execution bugs & personal discipline).
   * **Months 4–6:** Deploy `₹15L–₹25L` real capital (verify live STT/slippage & emotional comfort).
   * **Months 7+:** Scale to `₹1 Cr` in 3–4 monthly tranches (`Top-10 ≤ ₹12L/stock` or `Top-20 ≤ ₹5L/stock`).

---

## 5. Technical Reference Appendix
* **Balance-Sheet Blowup Shield (`≥ 2` flags blocks new buy):** Non-financial Debt/Equity `> 180%`, FCF Yield `< -4%`, ROE `< 6%`, or Revenue Growth `< -10%` (plus optional 5-day Earnings Blackout guard).
* **Hard Anti-Leakage Invariants (Never Break):** Strict prior-day (`T-1`) US/global macro merge (`macro_date < nse_date`) so 2:30 AM IST US closes never leak into 3:30 PM IST NSE trades; strictly expanding walk-forward ML with 40-day embargo.
* **Quick Run Commands (Local or Fresh VM):**
  `git clone https://github.com/Joestilllives/equilens-india-quant` → activate venv → `pytest -q` → `uvicorn backend.main:app --host 127.0.0.1 --port 8000` → click **Sync & Retrain** (`6:30 PM – 11:00 PM IST`).

---

## 6. Research Graveyard (Tested Strictly OOS & Rejected — Do NOT Re-Test)

| Idea Tested | Why It Sounded Good | Exact OOS Verdict (Why We Killed It) |
| :--- | :--- | :--- |
| **Pure Money-Signal Stock Picker** | Buy highest NSE delivery % stocks | Only `+9.84%` CAGR (bought sleepy defensive dividend stocks). Delivery % works ONLY as a `<35%` bouncer (`p = 0.0000`). |
| **Upper Delivery Cap (`> 72%`)** | Avoid "too slow" institutional stocks | Hurt CAGR & Sharpe — high-delivery winners are often institutional breakouts. |
| **Quarterly EPS Surprises (PEAD)** | Add earnings-beat drift to ML | Across 2,974 strictly post-announcement (`+2d`) NSE reports, standalone IC was **`-0.0065`** (zero edge); hurt `V6` in **5/6 years** (`Sharpe 2.00 → 1.86`). |
| **Official NSE FII OI Veto (`FII < 20%`)** | Go defensive when FIIs are 4:1 short | Cut MaxDD slightly (`-9.8% → -8.4%`), but cost **`-15.6%` in `2023`** when domestic SIPs rallied while FIIs stayed short. Keep FII OI visual-only. |
| **Expand Pool (`130` → `231` Stocks)** | More mid/smallcap breadth | Mixing large + small caps diluted OOS IC (`+0.0446 → +0.0322`) and cut CAGR (`+22.5% → +21.4%`) & Sharpe (`1.57 → 1.51`). |
| **LTCG 9–12M Forced-Hold Tax Rule** | Delay selling winners past 365d (`13%` vs `20.8%`) | Holding fading momentum stocks lost `-0.45%/yr` alpha — **lower net post-tax wealth** (`₹2.76 Cr` vs `₹2.79 Cr` on `₹1 Cr`). Never let tax override a sell signal. |
| **Smart `NIFTYBEES` Idle Parking** | Park idle Risk-Gate cash in NIFTY when `>200D` | Hurt BOTH 5Y (`Sharpe 1.96 → 1.69`, `MaxDD -10.6% → -14.5%`) and 22Y (`MaxDD -26.8% → -31.9%`). `70/30` `LIQUIDCASE/GOLDBEES` won decisively. |
| **Single-Stock Trailing Stops (`-12%/-15%`)** | Cut individual losers faster | Whipsawed out of volatile winners before rebounds; hurt CAGR & Sharpe. Portfolio-level `-10%` Drawdown Brake works far better. |
| **Midcap / BankNIFTY in Risk Gate** | Extra index crash triggers | Redundant with NSE Breadth & 52W Hi-Lo; added false alarms in normal rallies. |
| **External Doc Momentum/Vol Blends** *(21D Frenzy, 50/50 Steady-Climb, Correlated Peer Mom, Drop Mega-Caps, Quiet Accum, Global Gap)* | Extra signals from independent `.docx` | None beat `V6 + <35% Delivery Shield` — `V6`'s 19 features already capture `12-1M`, `FIP`, and `MAX` lottery avoidance cleanly. |
| **Remove Any Core `V6` Layer** *(22Y Ablation)* | Simplify the engine further | Across `2004–2020` & `2021–2026`, **15 of 16 layers** actively added Sharpe, CAGR, or crash defense (`-10%` Brake saved `10.3%` in 2008/2020; 4-Tranche added `+0.23` Sharpe). |

---

## 7. `V7` Extension — Canonical Finance & Economics Upgrades (`feat/v7-book-research`)

`V7` preserves **100% of `V6`'s proven skeleton and crash-defense chassis** (50/50 Prior + Walk-Forward Ridge/HGB, 40D embargo, 6-Gauge Stress Gate, 18% Vol Target, one-way intra-cycle de-risking, `-10%/-6%` Peak-NAV Drawdown Brake, `70/30` `LIQUIDCASE`/`GOLDBEES` reserve, 4 staggered tranches, Top-10 / Hold Top-30 buffer, `≤12%` stock / `≤20%` sector caps), and applies **four surgical, peer-reviewed upgrades** plus **honest institutional accounting** and **100% free Point-In-Time (PIT) data pipelines**.

### 7.1 What Changed From `V6` → `V7` (And Exactly Why)

| Component | `V6` Baseline | `V7` Upgrade | Canonical Literature & Why It Helped on India NSE |
| :--- | :--- | :--- | :--- |
| **Non-Equity Hygiene** | Filtered only `^NSEI` (allowed `GOLDBEES.NS` & `NIFTYBEES.NS` in `ohlcv` to leak into predictions/breadth) | Strict `NON_EQUITY_SYMBOLS` filter (`^NSEI`, `NIFTYBEES.NS`, `GOLDBEES.NS`) everywhere *(also backported to `main` @ `5f303af`)* | Prevents reserve/benchmark ETFs from ever entering stock ranking, ML training, or breadth gauges. |
| **Factor Prior (Zero ML Params)** | Raw `6-1M` & `12-1M` vol-adj momentum + Low-Vol + FIP + `52W-High` proximity | **CAPM Residual Mom (`12-1M`, `6-1M`) + Intermediate Mom (`12-7M`) + Capital Gains Overhang (`CGO 126D` via VWAP) + Betting Against Beta (`-Vasicek β`) + Low-Vol** | *Ilmanen; Blitz et al. (2011); Novy-Marx (2012); Kahneman & Thaler / Grinblatt & Han (2005); Frazzini & Pedersen (2014).* Strips free market beta from momentum, captures slow institutional repricing (`12-7M` was #1 OOS signal), replaces weak 52W-high gauge with real investor disposition overhang (`+1.69%/yr` over 8.8Y), and adds crisis-resilient low-beta tilt (`+1.69%/yr` in 2017–2021). |
| **ML Target & Feature Hygiene** | Raw excess return (`stock - NIFTY`), 19 features fed identically to Ridge & HGB | **Vasicek Beta-Adjusted Residual Sharpe target** (`(r_i - β_vasicek · r_mkt) / vol`); pruned 3 noisy/collinear features (`sec_rel_126d`, `vol_ratio_5_20`, `macro_tailwind_brent`); excluded date-constant macro cols from linear **Ridge only** (kept in **HGB**) | Stops ML from confusing risky high-beta bull leverage with genuine stock-selection skill (`2F Pure Alpha +4.57% → +8.24%` while lowering Market Beta `0.59 → 0.49` and Size Beta `0.80 → 0.72`). Preserves `HGB` (verified #1 OOS ranker). |
| **Momentum Crash Defense** | Static prior weights in all regimes | **Daniel & Moskowitz (2016) Bear-Rebound Crash Damper:** when `NIFTY 252D < 0` & `VIX ≥ 19`, blends 50% of prior toward defensive `BAB + Low-Vol + Residual Mom` | Directly shields against 2009/2020-style bear-market snapbacks where beaten-down high-beta junk rallies sharply and raw momentum crashes. |
| **Position Sizing (`hrp_allocator.py`)** | Ad-hoc `1 / (vol × (1 + avg_corr))` heuristic | **López de Prado (2016) Hierarchical Risk Parity (`HRP`)** on 126D Ledoit-Wolf shrunk covariance + concentration caps | Groups correlated stocks like a family tree and allocates risk across *clusters* rather than names. Lifted 2022 bear-market return (`+4.1% → +14.8%`) and beat `V6` in **9/9** parameter cells. |
| **Accounting, Audit & Live Parity** | 0% RF Sharpe, flat `15 bps` backtest cost, single-sleeve paper tracking | **Exact Indian statutory NSE delivery costs + ADV impact + FIFO STCG/LTCG (`20.8% / 13.0%`) + `NIFTYBEES.NS` TRI + Excess `SR(6.2% RF)` + 2-Factor Pure Alpha + 4-tranche `paper_sleeves` + `2%` ADV order warning + corrected DSR & CPCV/PBO** | Eliminates backtest-vs-live illusion so every reported figure is real after-tax, after-cost, risk-free-adjusted alpha. |
| **Phase 2 Free PIT Pipelines (`pit_fundamentals.py`)** | Non-PIT restated Yahoo fundamentals (unsafe for historical backtests) | **100% Free:** Day-1 immutable SQLite `pit_fundamentals_history` snapshot logger (runs automatically on sync) + NSE Bhavcopy historical delivery parser/backfill + BSE/NSE quarterly XBRL parser (`+2` trading-day lag) | Builds a strictly zero-lookahead **Value (`FCF/EV`, `E/P`) + Gross Profitability (`GP/Assets`) + Low Accruals** sleeve (*Asness, Moskowitz & Pedersen 2013; Novy-Marx 2013; Sloan 1996*) to structurally hedge future multi-quarter momentum droughts like `2025–2026`. *(Logging from Day 1; does not alter V7 stock picks until sufficient PIT history accumulates.)* |

### 7.2 `V7` Verified 5-Gate Promotion Scorecard (`NIFTYBEES.NS` TRI, Statutory Costs & FIFO Post-Tax)

| Gate / Audit Test | Clean `V6` (`CorrPen`) | **`V7` (`Book Hybrid + HRP`)** | Verdict |
| :--- | :---: | :---: | :---: |
| **Gate 1 — Core OOS (`2021–2026`, `NIFTYBEES` TRI `+9.06%` / Post-Tax `+8.25%`)** | `+23.40%` (**Post-Tax `+20.91%`**) · `SR(6.2%) 1.21` · `MaxDD -9.90%` · `2F Alpha +4.57%` | **`+26.54%`** (**Post-Tax `+23.70%`**) · **`SR(6.2%) 1.50`** · `MaxDD -12.18%` · **`2F Alpha +8.24%`** | **PASS** |
| **Gate 2A — 8.8Y Multi-Cycle Prior (`2017–2026`, incl. 2018 NBFC & 2020 COVID)** | Baseline across 9 `(Top-K × Rebalance)` cells | **Won `9/9` CAGR & `9/9` `SR(6.2%)`** (`+2.80%` to `+5.96%/yr` over `V6`) | **PASS** |
| **Gate 2B — 19Y Multi-Cycle (`2007–2026`, incl. 2008 GFC, 2009 rebound, 2011 bear)** | `+12.61%` (Post-Tax `+12.35%`) · `SR 1.00` · `2F Alpha +3.44%` | **`+12.86%`** (`BookFIP`: **`+13.03%`**) · **`SR 1.03–1.06`** · **`2F Alpha +3.52%–+3.80%`** | **PASS** |
| **Gate 2C — 17 Real NSE Fallen Angels** *(`YESBANK -95%`, `ZEEL -88%`, `ATGL -86%`, `IDEA -84%`, etc.)* | `+20.57%` (**Post-Tax `+18.27%`**) · `SR 1.06` · `2F Alpha +5.05%` | **`+27.79%`** (**Post-Tax `+24.57%`**) · **`SR 1.59`** · **`2F Alpha +11.53%`** | **PASS** |
| **Gate 3 — Multiple-Testing Rigor** *(Corrected Bailey–López de Prado DSR `N=50, RF=6.2%` & CPCV PBO)* | `DSR = 92.51%` | **`DSR = 98.15%`** · `CPCV PBO (V6+V7 18 configs) = 25.71%` (`18/70` splits, med logit `+0.656`) | **DSR PASS / PBO 25.7%** |
| **Gate 4 — Stability & Extreme Friction** *(9-cell OOS plateau + `50 bps/side` = `100 bps` RT)* | `50bps`: `+17.09%` Post-Tax · `SR 0.98` | **`9/9` OOS Plateau Wins** · `50bps`: **`+22.10%` Post-Tax · `SR 1.39` · `+15.82%/yr` excess** | **PASS** |
| **Gate 5 — Live Integrity & Suite** | — | **`27/27` pytest passed** · `0` ETF leaks · `₹10.10 L` paper ledger migrated cleanly to 4 sleeves | **PASS** |

### 7.3 `V7` Projected Median CAGR & Realistic Live Expectations
- **10,000-Path 5-Year Block-Bootstrap Monte Carlo** *(20-day stationary blocks from `V7`'s live `2021–2026` OOS returns)*:
  - **Median (`P50`) Pre-Tax CAGR:** **`+26.2%`** *(10th–90th percentile: `+16.7%` to `+36.9%`, vs. NIFTY 50 TRI median `+9.0%`)*
  - **Median (`P50`) FIFO Post-Tax CAGR:** **`+23.4%`** *(10th–90th percentile: `+14.9%` to `+33.0%`)*
  - **Median Max Drawdown:** **`-11.2%`** *(1-in-10 adverse path: `-15.8%`)*
- **Honest Real-World Cycle Expectation (Budget on This, Not the Bootstrap Peak):**
  - Across the full `8.8-year` cycle (`2017–2026`), `V7`'s zero-parameter prior averaged **`~+21.6%` pre-tax (`~+19%` post-tax)**.
  - Applying a prudent `30–40%` real-world haircut for future momentum droughts (like `2025–2026`), crowding, and execution friction yields a **conservative, realistic 5–10 year expectation of `~+16% to +20%` pre-tax (`~+14% to +17%` net post-tax CAGR)** — still beating NIFTY 50 TRI (`~+9% to +12%`) with substantially shallower drawdowns.

### 7.4 `V7` Research Graveyard (Canonical Textbook Ideas Tested & Rejected on India NSE)

| Idea Tested | Textbook Source | Exact Empirical Verdict on Live NSE Data (Why We Rejected It) |
| :--- | :--- | :--- |
| **Drop `HGB` / Heavy Regularization / 4Y Exp Decay / Blind Monotonic HGB** | Standard ML regularization advice | **`z_hgb` is the #1 strongest OOS ranker** (`IC +0.0612, t=+2.22`). Dropping HGB cut CAGR `-1.96%/yr`; 4Y decay cut `-1.27%/yr` (forgot 2018–2020 bear data); blind monotonic constraints cut `-0.64%/yr`. |
| **`sklearn.LedoitWolf` inside V6's 60D Scalar Vol Target** | Ledoit & Wolf (2004 *JMVA*) | `V6` never inverts `cov_60`, and `sklearn`'s zero-correlation shrinkage underestimated long-only equity variance, hurting CAGR (`-0.37%/yr`), `SR` (`1.30 → 1.25`), and MaxDD (`-9.99% → -10.36%`). *(Use `LedoitWolf` inside **HRP**, NOT inside the 60D scalar vol target.)* |
| **Symmetric Mid-Cycle Re-Entry** | Eliminating one-way de-risking | Staggered 4-tranche rebalancing already re-enters 1 sleeve every 10 days; mid-cycle re-entry got whipsawed, **worsened MaxDD `-2.52%`** (`-9.99% → -12.51%`), and inflated turnover `+50%`. |
| **Global Dollar / US-10Y Squeeze (`DX-Y.NYB`, `^TNX`)** | Hélène Rey (2015) *Global Financial Cycle* | Zero predictive power on India `2017–2026` (`p > 0.85`); post-2017 `₹25,000+ Cr/mo` domestic SIP flows decoupled India from classic EM USD squeezes. |
| **Double-Orthogonalized Amihud Illiquidity** | Amihud (2002) | Looked positive on surviving stocks (`+0.0339`), but collapsed to **`-0.0050 IC (t=-0.40)` on the 17 Fallen Angels** — proven survivorship trap (picked distressed illiquid blowups). |
| **EWMA(0.94) / 21D Fast Vol Target, `GOLDBEES` TSMOM (45/15), Brent Mean-Reversion** | Fast vol targeting & macro trend | Hurt Calmar/MaxDD or had `t = -0.56` OOS noise (`2021–2026`); pruned for parsimony. |

### 7.5 Final Frozen `V7` Sign-Off Audit (`audit_overfitting_v6.py --weighting hrp --perms 50`)
- **Automated Suite & Live APIs:** **`27 / 27` pytest passed** (`48` concurrent API reqs: `100% HTTP 200`, `p50=85.9ms`, `p95=759.1ms`); all live GET endpoints `200 OK`, unauthenticated `POST` blocked (`401`), `0` ETF leaks, live `₹10,10,032.98` paper ledger (`10` positions, `4` sleeves, `10` FIFO lots, `130` Day-1 PIT snapshots).
- **2-Factor Pure Alpha:** **`+8.24%/yr` (`t = +2.18`)** (`beta_mkt = 0.49`, `beta_size = 0.72`); **IIMA 4-Factor Indian Alpha:** **`+7.05%` to `+8.53%/yr`**.
- **Calendar Years (`5/6` beat `NIFTYBEES` TRI):** `2021: +38.2%` · `2022: +14.8%` · `2023: +56.1%` · `2024: +36.0%` · `2025: +0.5%` · `2026 YTD: -1.2%` *(vs NIFTY TRI `-10.6%`)* — worst calendar-year MaxDD `-8.91%`.
- **Matched 4-Tranche Top-10 Permutation Null Test (`N = 50`):** Random selection + identical `V7` risk shields averaged `+14.52%` CAGR (`SR 0.78`) vs actual `V7` **`+26.48%` (`SR 1.48`, beat `50/50` random runs, `p = 0.0196`)** → **`+11.96%/yr` pure stock-selection skill**.
- **Corrected Bailey–López de Prado DSR (`std_trials = 0.313`, `6.2%` RF):** `N=25: 97.05%` · **`N=50: 95.49%`** · `N=100: 93.56%` *(at `0%` RF: `N=100: 99.55%`)*.
- **12-Cell Parameter Plateau (`K ∈ {6,8,10,12} × Reb ∈ {30,40,60}`):** Every single cell earned `+23.61%` to `+30.98%` CAGR and `1.39` to `1.62` Excess Sharpe (zero cliff). **Status: FROZEN FOR LIVE PAPER TRACKING.**

### 7.6 Complete Post-`V6` Evolution Log (What `V7` Unified From `V6`, `V6.1` & `V8`)

Everything built, tested, kept, or rejected after `V6` (`9d09dfb`) is summarized here so there is zero guesswork about what is running live:

| Era / Branch | What Was Done | Included in Final Unified `V7`? | Why / Evidence |
| :--- | :--- | :---: | :--- |
| **`V6` Core (`main` @ `9d09dfb`)** | 50/50 Prior + Walk-Forward Ridge/HGB, 40D embargo, 6-Gauge Stress Gate, 18% Vol Target, one-way de-risking, `-10%/-6%` DD Brake, `70/30` `LIQUIDCASE`/`GOLDBEES`, 4 staggered tranches, Top-10/Hold Top-30, `≤12%/≤20%` caps. | **YES (100% preserved)** | Proven multi-cycle crash-defense skeleton. |
| **Live Safety Patch (`main` @ `e2d5c4b` + `5f303af`)** | Peak-NAV `-10%/-6%` Drawdown Brake, bounded missing-bar hygiene (`max_ffill_days=2`) + staleness fail-safe, `70/30` `LIQUIDCASE`/`GOLDBEES` reserve mark-to-market, FIFO STCG/LTCG + NSE statutory costs, localhost API auth (`X-EquiLens-Token`), restricted CORS, reset confirmation, safe URLs, and **`NON_EQUITY_SYMBOLS` ETF leak fix**. | **YES (100% included)** | Prevents real-money operational, tax, security, and ETF-leak bugs. |
| **`V6.1` Institutional Hardening (`v6-institutional-hardening`)** | `NIFTYBEES.NS` TRI benchmark + `GOLDBEES.NS` auto-sync (`10y` history), next-open fill fix in single-stock chart `backtester.py`, within-sector normalization in `shortlister.py`, separated `journal.py`, honest NSE delivery/insider labels in `money_signals.py`, and `40D` / Data Freshness / `Visual Composite (Unvalidated)` UI polish in `frontend/`. | **YES (all non-alpha hardening unified)** | Fixes lookahead in chart backtester, guarantees `NIFTYBEES`/`GOLDBEES` stay fresh on **Sync & Retrain**, and keeps UI/accounting honest with zero change to `V7` alpha. |
| **`V7` Canonical Finance Books (`feat/v7-book-research`)** | CAPM Residual Momentum (`12-1M`, `6-1M`), Novy-Marx `12-7M`, Grinblatt–Han `CGO 126D`, Frazzini–Pedersen `BAB` (`-Vasicek β`), Vasicek Residual Sharpe ML target, Daniel–Moskowitz Bear-Rebound Crash Damper, López de Prado `HRP` (`126D Ledoit-Wolf`), 4-tranche `paper_sleeves`, `2%` ADV cap guard, and Day-1 **100% free** PIT fundamental snapshot logger + NSE Bhavcopy + BSE/NSE XBRL parsers (`pit_fundamentals.py`). | **YES (Core Live Engine)** | Beat `V6` by `+2.79%/yr` Post-Tax on clean OOS, **`+6.30%/yr` Post-Tax on 17 real Indian crashes**, **`9/9`** over 8.8Y, won `19Y` (`2007–2026`), and beat **`50/50`** random permutations (`p=0.0196`). |
| **`V8` Frontier Toolkit (`feat/v8-frontier-research`) — KEPT** | **IIMA 4-Factor Indian Alpha** (`MKT+SMB+HML+WML` wired into `portfolio_backtester.py`), **Indian Forensic Governance & QMJ+Value** (`forensic_quality_sleeve.py`), **SEBI Shareholding `+2d` lag parser** (`india_structural_signals.py`), **RBI DBIE Liquidity telemetry** (`india_liquidity_regime.py`), **Carver Multi-Sleeve Combiner** (`multi_sleeve_allocator.py`), and opt-in **`₹1 Cr+` Boyd Convex Optimizer** (`convex_optimizer.py`). | **YES (Reporting & Future Opt-In Toolkit)** | Proved `V7`'s **`+7.05%` to `+8.53%/yr` 4-Factor Indian Alpha**; forensic/QMJ/Carver/Convex modules sit ready without disturbing `V7`'s locked `HRP` + `6-Gauge` defaults. |
| **`V8` Price/Regime Tweaks — REJECTED** | Smooth Sigmoid Regime Gate (`SR 1.50 → 1.34`, `MaxDD -12.2% → -13.5%`), Extra India Liquidity Gate (redundant), `V8-FullAFML` (`-1.68%` on 17 Fallen Angels), and `V8-TopRank` (`fwd_rank²`, `PBO 55.7%`, trailed `-0.29%` on 17 Fallen Angels). | **NO (Logged in Graveyard)** | Failed to beat `V7` cleanly on real Indian crashes — kept out of live execution to prevent overfitting. |


---

## 8. Live Production `V7.5-Core` Upgrade, Blind-Spot Red-Team Audit & Future Watchlist

**Live Branch:** `feat/v7-book-research` (`github.com/Joestilllives/equilens-india-quant`)  
**Status:** **LIVE PRODUCTION DEFAULT (`strategy_id: v75_core_delivery_shield`)** — `V7` Canonical Finance + HRP engine upgraded with a strictly `T-1` causal **20-Day Official NSE Delivery Shield (`35% <= deliv_20d < 98%`, Deliverable Value `>= ₹2 Cr/day`)**, **Indian `320–364d` LTCG Tax-Cliff Guard**, **Post-Budget 2024 Reserve Tax Honesty**, and **SEBI Delivery Regime-Drift Telemetry**.

### 8.1 What Changed from Frozen `V7` (`34e7747`) → Live `V7.5-Core`
1. **Complete 100% Daily 7-Year Official NSE Delivery Archive (`2020-01-01 → 2026-11-26`):**
   - Seeded **`220,229` rows across `1,700` consecutive NSE trading days** (`250 / 247 / 246 / 245 / 245 / 246 / 221` days/year) directly from `nsearchives.nseindia.com` into `data/equilens_india.db` (`nse_delivery`, NOT committed to git).
   - Replaced noisy 1-day delivery prints with a **strictly `T-1` causal 20-day trailing mean (`deliv_20d_pct`)** across `money_signals.py`, `execution_engine.py`, and `portfolio_backtester.py` (precomputed O(log N) lookup in `build_backtest_market_context`).
2. **Full Backtest ↔ Live Parity on Retention + Entry + T2T & Deliverable Value Guards:**
   - **Retention + Entry Parity:** The delivery shield applies to **both** new buys AND Top-30 hysteresis rebalance retention — exiting an existing momentum winner at rebalance if its 20D institutional delivery collapses `< 35%` into retail/F&O intraday churn *(this exit rule alone delivers `~+2.5%/yr` CAGR and cuts MaxDD by `~1.5–2.0%`)*.
   - **T2T / ASM Surveillance Trap Guard (`deliv_20d_pct >= 98%`):** NSE Trade-for-Trade (T2T) / ASM / ESM surveillance stocks mechanically print `100%` delivery because intraday trading is prohibited. Explicit `>= 98%` exclusion future-proofs the shield if the universe is ever broadened to mid/smallcaps.
   - **Rupee Deliverable Turnover Floor (`>= ₹2 Cr/day`):** Prevents illiquid, tiny-volume stocks with high percentage delivery from passing the shield (`turnover_lacs * (deliv_pct / 100) / 100 >= 2.0`).
3. **Indian `320–364d` LTCG Tax-Cliff Holding Guard (`ltcg_cliff_guard=True`):**
   - If an existing profitable holding drops outside the Top-30 buffer (still inside Top-60, tradable, and passing all delivery/T2T/balance-sheet shields) while its oldest FIFO lot is aged **`320–364` days**, hold it across 1 extra rebalance cycle so gains qualify for **`13.0%` LTCG** instead of **`20.8%` STCG**.
   - Improved **every** metric simultaneously (`CAGR +28.59% → +28.91%`, Honest Post-Tax `+23.92% → +24.44%`, `SR(6.2%) 1.75 → 1.77`, `MaxDD -10.64% → -10.56%`, `4F Alpha +10.05% → +10.22%`, `STCG` tax cut by `₹13.7k` per `₹10 L`).
4. **Post-Budget 2024 Reserve Tax Honesty (`indian_costs.py` + `portfolio_backtester.py`):**
   - Under India's **Finance Act 2024 (effective July 23, 2024)**, debt ETFs (`LIQUIDCASE`, `>65%` debt) and `< 12m` Gold ETFs (`GOLDBEES`) are taxed at the investor's **marginal income tax slab rate (`31.2%` = 30% + 4% cess)**, whereas a **Direct-Plan Arbitrage Mutual Fund (`>65%` hedged equity)** is taxed as equity (**`20.8%` STCG / `13.0%` LTCG `≥ 12m`**).
   - `portfolio_backtester.py` now tracks exact daily rupee reserve gains (`liquid_gain_inr`, `gold_gain_inr`) and deducts reserve taxes directly inside `after_tax_cagr_pct`, reporting both `after_tax_cagr_liquidcase_pct` and `after_tax_cagr_arbitrage_pct`.
5. **SEBI Delivery Regime-Drift Telemetry (`money_signals.py` + `execution_engine.py`):**
   - Surfaces `delivery_regime_telemetry` (`universe_count`, `below_35_count`, `below_35_pct`, `median_20d_deliv_pct`, `p25_20d_deliv_pct`, `t2t_flagged_count`, `low_deliv_turnover_count`, `drift_status`) on `/api/money-signals` and `/api/execution/ticket`, raising `drift_status: "warning"` if `below_35_pct < 5.0%`.

### 8.2 Verified Production Scorecard (100% Daily Archive, `2021-07 → 2026-09`, `K=10`, `40D`, Exact Statutory Costs + FIFO Equity & Reserve Tax)

| Production Engine | Pre-Tax CAGR | Honest Post-Tax (`LIQUIDCASE 31.2%`) | Honest Post-Tax (`Arbitrage 20.8%`) | Excess `SR(6.2%)` | MaxDD | IIMA 4-Factor Alpha (`t`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen `V6` Base** *(Unshielded)* | `+23.40%` | `~+19.6%` *(eq-only `+20.91%`)* | `~+19.8%` | `1.21` | `-9.90%` | `+3.01%` (`t=+0.9`) |
| **Frozen `V7` Base** *(Unshielded)* | `+26.54%` | `+22.22%` *(eq-only `+23.70%`)* | `+22.39%` | `1.50` | `-12.18%` | `+7.05%` (`t=+1.98`) |
| **`V7.5-Core` (100% Daily + T2T/₹2Cr, `LTCG=OFF`)** | `+28.59%` | `+23.92%` *(eq-only `+25.36%`)* | `+24.09%` | `1.75` | `-10.64%` | `+10.05%` (`t=+2.89`) |
| **Live `V7.5-Core` (+ `320–364d` LTCG Guard — DEFAULT)** | **`+28.91%`** | **`+24.44%`** | **`+24.61%`** | **`1.77`** | **`-10.56%`** | **`+10.22%` (`t=+2.95`)** |

* **Independent Robustness Audits (Overfitting Score: `1.5 / 10`):**
  - `35%` threshold was pre-registered in `V6` before downloading the 6Y archive; flat plateau across `25%–45%` and `T-5` extra lag (`+29.03%`, `SR 1.75`); wins `5/6` calendar years; clears `N=80` cumulative DSR (`99.77%`).
  - Swept **`9/9`** `(Top-K × Rebalance)` plateau cells on CAGR & `SR(6.2%)` (`8/9` MaxDD); beat `V7` on **17 Real NSE Fallen Angels** (`+25.85%` vs `+24.46%`, `SR 1.54` vs `1.37`) and blindly on **126 Unseen Indian Smallcaps** (`+18.26%` vs `+16.89%`, `SR 0.98` vs `0.86`).
* **Realistic Live Budget (5–10 Year Cycle, Net Post-Tax):** Expect **`~16% to 19%` net post-tax CAGR** (conservative choppy decade `12%–15%`, strong bull cycle `22%–26%`), with occasional `-10% to -18%` drawdowns and 6–12 month flat stretches.

### 8.3 What Was Intentionally KEPT OUT of Live Production (`V7.5` Research Graveyard)
- **`V7.5-Full` (`Screener.in` PIT Value Dual-Sleeve + `264`-Stock Expansion):** Kept strictly on isolated research branch `feat/v75-frontier` (NOT promoted). Carried **`5.5–6 / 10`** overfitting/selection bias (pruned from 21 PIT signals down to 1, `Screener.in` consolidated restatement risk, current-survivor universe).
- **Ledoit–Wolf QIS Nonlinear Shrinkage inside HRP:** `p=10, T=126` too low-dimensional; tied/trailed linear Ledoit–Wolf.
- **KMZ (2024) RFF-Ridge, JKMP (2026) Signal Smoothing, Gârleanu–Pedersen Aim `0.65`:** Tied linear Ridge or hurt India Large/Midcap (`-1.7%/yr`) / Smallcaps.
- **Self-Adaptive Bottom-20%/25% Cross-Sectional Delivery Percentile:** Tested against Static `<35%` and **trailed** on CAGR (`+28.66%–29.03%` vs `+29.42%`), FIFO Post-Tax, MaxDD (`-12.1% to -12.5%` vs `-10.3%`), and `2026 YTD` — because dropping the bottom 25% in 2025–2026 discards healthy `42%`-delivery stocks.

### 8.4 Operational Watchlist for All Future Iterations (Do NOT Forget These!)
1. **Watch SEBI Delivery Regime Drift (`below_35_pct`):**
   - SEBI's Peak Margin (2021) + T+1 Settlement (2023) pushed NSE median delivery from **`34.4%` (`2020`) → `51.5%` (`2026`)**, shrinking `<35%` hit rate from `51.4%` (`2020`) down to **`5.3%` (`7 / 131` stocks as of Nov 2026)** — just above our **`5.0%`** warning line.
   - **Action rule:** As long as `below_35_pct >= 5%`, keep Static `<35%` untouched. If future SEBI `T+0` settlement or further intraday curbs push `below_35_pct` persistently **below `5.0%`**, review switching to the pre-tested **`Hybrid (<35% OR Bottom-20%)`** rule (`SR 1.81`, `4F Alpha +10.74%`).
2. **Use a Direct Arbitrage Fund for Live Idle Cash (Post-Budget 2024 Tax Edge):**
   - If you are in the `30%` Indian tax slab, parking the `70%` defensive cash reserve in a **Direct-Plan Arbitrage Mutual Fund** (`20.8%` STCG / `13.0%` LTCG `≥ 12m`) instead of `LIQUIDCASE` (`31.2%` slab rate) saves `~₹12k–₹22k` per `₹10 L` (`₹1.2 L–₹2.2 L` per `₹1 Cr`) over 5 years *(trade-off: T+2/T+3 redemption vs instant ETF liquidity)*.
   - Remember `< 12m GOLDBEES` flips are also taxed at `31.2%` slab rate; avoid unnecessary churn in the `30%` gold sleeve and do **not** assume 2021–2026's historic INR gold bull market (`+1.85%/yr` tailwind) will repeat every decade.
3. **Never Run Unconstrained Deep ML on `N ≈ 130` Stocks:**
   - Literature (small-universe Gu–Kelly–Xiu critiques) confirms high-capacity ML overfits `N ≈ 100` stock panels. Preserve the `50%` Canonical Multi-Factor Prior + strongly regularized linear `Ridge` + shallow `HGB` balance.
4. **Never Expand to Smallcaps Without T2T/ASM & Circuit Protections:**
   - Our liquid NIFTY 100/200 universe avoids static 5%/10% lower-circuit lockouts (most constituents are NSE F&O stocks with dynamic bands) and had `0.00%` T2T days. If ever expanding down-cap, keep both `deliv_20d < 98%` and `deliv_turnover_20d_cr >= 2.0` strictly enforced.
5. **Respect Known Structural Trade-Offs (Do Not Over-Tweak):**
   - **Drawdown Brake (`-10%/-6%`):** Can lag fast V-shaped bottoms (like March 2020), but proved essential in multi-year bears (`2008`: `-17%` vs NIFTY `-60%`). Keep as crisis insurance.
   - **4-Tranche Staggering (`40D`):** Trades `~1.1%/yr` post-tax and `4×` FIFO lots vs 1-tranche in exchange for `-2.0%` shallower MaxDD and zero start-date timing luck (now partially recovered by the `320–364d` LTCG cliff guard).
   - **Never touch `paper_*` tables** during upgrades (live paper ledger started `2026-11-20` at `₹10,00,000`, currently `₹10,10,032.98` across `10` positions / `4` sleeves) and **never commit `data/equilens_india.db`** to git.
