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
