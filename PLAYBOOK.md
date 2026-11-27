# EquiLens India Quant — The Complete `V7.5-Core` Master Playbook

> **Active Production Engine:** `EquiLens India V7.5-Core` (`strategy_id: v75_core_delivery_shield`)  
> **Repository & Branches:** `github.com/Joestilllives/equilens-india-quant` — live on `main` (synced with `feat/v7-book-research`; historical `V6` playbook preserved in git history at commit `7614ab0`)  
> **Live Status:** **STRICTLY FROZEN FOR FORWARD PAPER TRACKING** — Do NOT tweak parameters, add indicators, or touch the live `paper_*` ledger without passing the 5-Gate Promotion Standard in Part II.  
> **Disclaimer:** *Quantitative research & personal forward paper-trading guide. Not SEBI-registered investment advice.*

---

## How to Use This Guidebook

This document is written for **two very different readers** — and you should always know which part you are in:

1. **Part I (Sections 1–10) — For Joe (Kitchen-Table Plain English):** Written assuming **zero background in stocks, finance, or math**. Every market word is explained like we are talking over chai at the kitchen table, using everyday examples and real Indian Rupee (`₹10 Lakh` and `₹1 Crore`) numbers. Read this whenever you want to understand *how* your money machine works, *why* it picks or sells a stock, *what* to expect emotionally, or *what to do* when something scary or unexpected happens.
2. **Part II (Sections 11–16) — For the Next AI Iteration (Engineering & Quant Bible):** Written with exact file paths, Python module names, formulas, hyperparameters, SQLite schemas, API endpoints, verified empirical numbers, hard anti-leakage invariants, and strict governance gates so another AI can step in months or years later without breaking a single rule or re-testing dead ends.

---

# PART I — THE KITCHEN-TABLE GUIDEBOOK (PLAIN ENGLISH, ZERO FINANCE BACKGROUND REQUIRED)

---

## 1. Start Here: The 3-Minute Zero-Jargon Dictionary

If you don't know what stocks are or how the stock market works, read these 12 plain-English translations first. Everything else in this book builds on them.

| Market Word | Kitchen-Table Meaning |
| :--- | :--- |
| **Stock / Share** | A tiny slice of ownership in a real Indian company (like HDFC Bank, Reliance, or Marico). If the company grows profits and becomes more valuable, your slice becomes worth more. |
| **NSE & NIFTY 50** | **NSE** (National Stock Exchange) is India's main supermarket where shares are bought and sold every weekday (`9:15 AM – 3:30 PM IST`). **NIFTY 50** is the average scorecard of India's 50 biggest companies — the yardstick we try to beat. |
| **NIFTYBEES (TRI)** | **TRI** stands for *Total Return Index* — meaning NIFTY 50 *plus* the cash dividends companies pay out. Comparing ourselves to NIFTY without dividends would be cheating, so we always measure against `NIFTYBEES.NS` (an ETF that automatically includes dividends). |
| **ETF** | *Exchange-Traded Fund* — a ready-made basket you can buy like a single stock. We use **`GOLDBEES`** (a basket holding real physical gold in India) and **`LIQUIDCASE`** (a basket holding ultra-safe government/bank overnight loans that pay ~`6.2%/yr` interest). |
| **CAGR** | *Compound Annual Growth Rate* — your average yearly speed of growing wealth. At `15%` CAGR, `₹10 Lakh` doubles roughly every 5 years (`₹10L → ₹20L → ₹40L`). At `25%` CAGR, it triples in 5 years (`₹10L → ₹30.5L`). |
| **Volatility** | How bumpy the ride is. A fixed deposit has `0%` bumpiness (smooth, slow). Stocks bounce up and down every day (`12%–25%` bumpiness). |
| **Sharpe Ratio (`SR 6.2%`)** | **Bang-for-your-bumpiness** *above* what a safe Indian savings/liquid fund pays (`6.2%/yr`). Below `0.5` is poor; `0.8–1.0` is good; `1.5+` is elite; **`V7.5-Core` achieved `1.84`**. |
| **Max Drawdown (`MaxDD`)** | The worst peak-to-valley drop you had to stomach before recovering. If your account hit `₹10,00,000` and later fell to `₹8,97,700`, your drawdown was `-10.23%`. NIFTY 50 dropped `-59.9%` in 2008 and `-16.1%` (TRI) / `-17.2%` (price) in 2021–2026. |
| **Alpha (`4-Factor Alpha`)** | Pure skill *after* stripping out luck. If the whole Indian market rose `+20%` and you made `+20%`, your alpha is `0%` (you just rode the wave). **`+10.96%` 4-Factor Alpha** means our system made nearly `11%/yr` of genuine stock-picking profit that cannot be explained by just buying the market, small companies, cheap companies, or dumb momentum. |
| **Rebalance** | Every **10 trading days** (~every 2 weeks), we check one quarter (`25%`) of our portfolio, sell stocks that have lost their spark, and buy the fresh leaders. |
| **Delivery % (NSE)** | On NSE, day-traders buy at 10 AM and sell at 2 PM without ever taking shares home (like flipping lottery coupons). **Delivery %** is the percentage of shares that buyers actually *paid full cash for and took home into their demat locker*. High delivery (`≥ 35%`) = serious investors; low delivery (`< 35%`) = speculative casino churn. |
| **STCG / LTCG & FIFO** | Indian capital gains tax. Sell a stock in **under 365 days** = **STCG** (Short-Term tax = `20.8%` of profit). Hold **365+ days** = **LTCG** (Long-Term tax = `13.0%` of profit, and the first `₹1.25 Lakh` per year is tax-free!). **FIFO** (*First-In, First-Out*) means Indian tax law assumes you always sell your oldest shares first. |

---

## 2. The Big Picture: What `V7.5-Core` Actually Does (The 60-Second Story)

Imagine you want to pick a winning **10-player cricket squad** from India's **130 biggest, most actively traded companies** — and you want to protect your money whenever a storm hits the stadium.

Here is the exact 5-step assembly line `EquiLens India V7.5-Core` runs automatically:

1. **Step 1 — Score All 130 Companies (Half Common-Sense Rules, Half Machine Learning):**
   Every evening, the system scores all 130 NSE stocks from `#1` (strongest) to `#130` (weakest).
   - **50% of the score** comes from **6 timeless, textbook financial rules** discovered by Nobel-level finance professors over the last 30 years (zero machine-learning guesswork).
   - **50% of the score** comes from **two conservative, heavily-restrained Machine Learning models** (`Ridge` + shallow `HistGradientBoosting`) looking at **19 clean signals** to predict which stocks will quietly outperform over the next **40 trading days (~2 months)**.
2. **Step 2 — Run the Bouncers at the Door (NSE Delivery Shield + Balance-Sheet Shield):**
   Even if a stock scores `#1`, it gets **thrown out** (both on new buys AND if we already own it!) if:
   - Its 20-day official NSE **Delivery % drops below `35%`** (meaning serious buyers vanished and day-traders took over the casino), OR
   - Its 20-day Delivery % is **`≥ 98%`** (a trap where NSE's surveillance office banned intraday trading on a shady stock — called T2T/ASM), OR
   - Fewer than **`₹2 Crore/day`** of real delivery money changes hands (too thinly traded), OR
   - Its financial health has **`≥ 2` red flags** (crushing debt, burning cash, terrible profitability, or shrinking sales).
3. **Step 3 — Pick the Squad & Avoid Unnecessary Taxes (Hysteresis + Tax-Cliff Guard):**
   - We buy the **Top 10** stocks that pass the bouncers (never more than **2 stocks from the same industry**, like Banking or IT).
   - Once we own a winner, we **don't sell it just because it slips to `#11`**. We keep holding it as long as it stays inside the **Top 30** (saving huge brokerage and tax bills).
   - **Smart Indian Tax Guard:** If a profitable stock slips outside Top 30 (say `#35`, still inside Top 60 and still healthy) AND we have already held it for **`320 to 364 days`**, we hold it one extra cycle so it crosses the 365-day finish line and pays **`13.0%` LTCG tax instead of `20.8%` STCG tax**.
4. **Step 4 — Size Each Bet Intelligently (`HRP` + Hard Caps + 4 Staggered Sleeves):**
   - Instead of putting `10%` blindly in every stock, **Hierarchical Risk Parity (`HRP`)** groups stocks that move together like a family tree and gives calmer, independent stocks slightly larger weights — while enforcing hard walls of **`≤ 12%` in any single stock** and **`≤ 20%` in any single sector**.
   - Instead of investing 100% on one lucky or unlucky Tuesday, your money is split into **4 equal sleeves (`25%` each)**, staggered **10 trading days apart** across a **40-day cycle**.
5. **Step 5 — The 3-Layer Storm Shelter (Crash Defense + 70/30 Reserve):**
   - **Layer A (6-Gauge Market Weather Radar):** Watches India VIX (fear meter), NIFTY trend, how many stocks are healthy, highs vs. lows, and whether people are fleeing into Gold. As storm clouds gather, it automatically dials stocks down **`100% → 85% → 55% → 25%`**.
   - **Layer B (`18%` Speed Limiter):** If the 10 chosen stocks suddenly become too wild, it dials back exposure so the portfolio never exceeds `18%` annualized bumpiness.
   - **Layer C (`-10%` Emergency Airbag):** If *your own account* ever drops `-10%` from its all-time high, it cuts remaining stock exposure in half until your account heals past `-6%` and trends back up.
   - **Where does the sidelined cash go?** Never sitting dead in a `0%` broker wallet! It parks **`70%` in safe Indian cash (`LIQUIDCASE` or a Direct Arbitrage Fund)** and **`30%` in Indian Gold (`GOLDBEES`)** — which often *rises* when Indian stocks crash!

---

## 3. Every Single One of the 19 Factors in `V7.5-Core` (What It Means, Why It Works, How Much It Matters & Will It Last?)

We ran a brutal **33-run Leave-One-Out (LOO) experiment** (`2021–2026`, exact Indian statutory brokerage/STT/stamp duty, FIFO taxes, Post-Budget 2024 reserve taxes). For every single factor below, we literally **ripped it out of the engine, retrained the whole AI from scratch, and measured what broke**.

> **How to read the "How Much Difference It Makes" column:**  
> Numbers show what happens when you **REMOVE** that factor from the engine. A **negative** number (like `-4.59%` Post-Tax CAGR) means **removing it HURT badly — so the factor is a huge positive contributor!**

### Group A — The 6 "Core Engine" Factors (Inside BOTH the 50% Canonical Prior AND the 50% ML)

These 6 factors are the **top 6 MVPs of the entire system**. Every single one comes from a famous, peer-reviewed finance textbook, has zero ML curve-fitting in the prior, and made a massive measurable difference when removed.

#### 1. Betting Against Beta (`z_neg_beta`) — **#1 MVP OF THE ENTIRE SYSTEM**
* **Prior Weight:** `+10%` (jumps to `+40%` defensive weight during bear-rebound storms)
* **Kitchen-Table Meaning:** Imagine two delivery scooters. Scooter A zooms wildly every time traffic speeds up and crashes hard every time traffic brakes (`Beta = 1.8`). Scooter B cruises steadily regardless of traffic (`Beta = 0.6`). We measure each stock's sensitivity to NIFTY over the past year (gently nudged toward `1.0` using the Vasicek formula so freak months don't fool us) and **prefer the calmer, lower-beta stocks**.
* **Why It Works:** *Frazzini & Pedersen (2014).* Regular investors and mutual funds can't easily borrow money to juice returns, so they crowd into flashy, roller-coaster high-beta stocks hoping to get rich fast — overpaying for them. Steady, "boring" low-beta compounders get underpriced and quietly deliver far better risk-adjusted returns, especially when the market stumbles.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR crashes **`-4.18%/yr`**, Honest Post-Tax crashes **`-4.59%/yr`**, Sharpe collapses **`-0.40`** (`1.77 → 1.36`). Removing this single factor does more damage than removing any other feature in the entire codebase.
* **Will It Stand the Test of Time?** **YES — 9.5 / 10 Durability.** Rooted in permanent institutional leverage constraints and human thrill-seeking. Has worked across 90+ years of US data, 20+ global markets, and multiple Indian market cycles.

#### 2. CAPM Residual Momentum 12–1 Month (`z_resid_12m1m`) — **#2 MVP**
* **Prior Weight:** `+25%` (retained at `+25%` even during bear-rebound storms)
* **Kitchen-Table Meaning:** Suppose Company X rose `+30%` over the past year (skipping the noisy last month). Did it rise because the company is genuinely winning — or only because the entire Indian market rose `+25%` and Company X is just a jittery stock that moves `1.5×` with the market? Residual momentum **subtracts out the "free market ride"** and rewards only the stock's *own* company-specific winning streak, divided by its own bumpiness.
* **Why It Works:** *Blitz, Huij & Martens (2011).* Regular momentum accidentally loads up on high-beta stocks during bull markets and gets crushed when the market snaps back. Stripping out market beta isolates genuine business/earnings leadership that mutual funds and analysts are slow to fully price in.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR drops **`-1.08%/yr`**, Honest Post-Tax drops **`-0.99%/yr`**, Sharpe drops **`-0.07`**.
* **Will It Stand the Test of Time?** **YES — 9 / 10 Durability.** One of the cleanest, least crash-prone momentum signals in institutional quant finance (core pillar of Robeco's multi-billion-dollar equity funds).

#### 3. CAPM Residual Momentum 6–1 Month (`z_resid_6m1m`) — **#3 MVP**
* **Prior Weight:** `+15%`
* **Kitchen-Table Meaning:** The exact same "strip out the free market ride" test as `#2`, but over the last **6 months** (skipping the last 1 month). It catches newer emerging winners twice as fast as the 12-month version.
* **Why It Works:** *Blitz et al. (2011).* Indian sector rotations (Defence, PSU Banks, Pharma, Power, Auto) often turn in 3–6 month waves. Pairing 6–1M with 12–1M catches fresh institutional accumulation early while filtering out market-beta noise.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR drops **`-0.87%/yr`**, Honest Post-Tax drops **`-0.98%/yr`**, Sharpe drops **`-0.07`**, and **MaxDD worsens by `-0.74%`** (`-10.56% → -11.30%`).
* **Will It Stand the Test of Time?** **YES — 8.5 / 10 Durability.** Structural institutional under-reaction to medium-term earnings turns.

#### 4. Intermediate-Horizon Momentum 12–7 Month (`z_mom_12_7`) — **#4 MVP**
* **Prior Weight:** `+25%`
* **Kitchen-Table Meaning:** Instead of looking at what a stock did last week or last quarter, this measures how strongly the stock climbed **between 12 months ago and 6 months ago** (adjusted for volatility).
* **Why It Works:** *Novy-Marx (2012).* Surprisingly, academic research discovered that momentum isn't mostly about "last month's hype" — stocks that began a real fundamental turnaround 7–12 months ago keep drifting higher because big institutions take *quarters* to finish researching, getting committee approval, and building full positions. It completely avoids short-term retail chasing.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR drops **`-1.14%/yr`**, Honest Post-Tax drops **`-0.91%/yr`**, Sharpe drops **`-0.09`**. Also ranked `#1` standalone OOS signal over the 8.8-year `2017–2026` test.
* **Will It Stand the Test of Time?** **YES — 9 / 10 Durability.** Slow institutional capital deployment and multi-quarter earnings cycles are permanent features of equity markets.

#### 5. Capital Gains Overhang 126-Day (`z_cgo_126`) — **#5 MVP**
* **Prior Weight:** `+15%`
* **Kitchen-Table Meaning:** Estimates the **average price paid by everyone who bought this stock over the last 6 months** (weighted by trading volume, called VWAP) and checks whether today's price is *above* or *below* their purchase price.
  * If today's price is **above** their average purchase price (`CGO > 0`), most owners are sitting on happy profits — they are *not* desperate to dump the stock at break-even, and good news pushes it smoothly higher.
  * If today's price is **below** their average cost (`CGO < 0`), thousands of trapped investors are praying *"please just get back to my buy price so I can sell and get out!"* — creating an invisible ceiling that crushes rallies.
* **Why It Works:** *Grinblatt & Han (2005); Kahneman & Tversky's Prospect Theory.* Humans hate realizing a loss (`disposition effect`). Trapped bag-holders sell every bounce, while stocks where everyone is comfortably in profit face far less "get-me-out-at-breakeven" selling pressure. Replaced `V6`'s crude 52-week-high proximity gauge and added **`+1.69%/yr`** across `2017–2026`.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR drops **`-0.49%/yr`**, Honest Post-Tax drops **`-0.63%/yr`**, and **MaxDD worsens by `-0.92%`** (`-10.56% → -11.48%`).
* **Will It Stand the Test of Time?** **YES — 9 / 10 Durability.** Hard-wired human loss aversion does not evolve away in 10 years (or 100 years).

#### 6. 1-Year Annualized Volatility (`z_vol_252d`) — **#6 MVP (Inverted / Low-Vol)**
* **Prior Weight:** `-10%` (minus sign = *lower* volatility is rewarded; jumps to `-35%` defensive weight in bear-rebound storms)
* **Kitchen-Table Meaning:** Simply measures how wildly a stock's daily price swung up and down over the past year. Calm, steady stocks get rewarded; chaotic roller-coaster stocks get penalized.
* **Why It Works:** *Ilmanen; Baker, Bradley & Wurgler (2011).* Compounding math ("volatility drag") punishes wild stocks (`-50%` requires `+100%` just to break even), and retail traders systematically overpay for exciting high-volatility names.
* **How Much Difference It Makes (LOO Impact when removed):** Honest Post-Tax drops **`-0.63%/yr`**, Sharpe drops **`-0.02`** (it also cuts turnover/tax churn by favoring steadier holdings).
* **Will It Stand the Test of Time?** **YES — 9 / 10 Durability.** Mathematical compounding advantage + permanent human preference for exciting "lottery" stocks.

---

### Group B — The 13 Supporting Factors (Used Inside the 50% Machine Learning Sleeve)

These 13 signals give the regularized ML models (`Ridge` + shallow `HistGradientBoosting`) nuance about trend smoothness, lottery traps, downside risk, short-term pullbacks, institutional overnight buying, and overall market weather.

| # | Factor Name (Code) | Kitchen-Table Meaning | Why It Works | LOO Impact When Removed | Will It Stand the Test of Time? |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **7** | **Global S&P 500 Beta Tailwind** (`z_macro_tailwind_sp500`) | Checks how tightly an Indian stock tracks the US S&P 500 (e.g., IT exporters, global pharma/auto parts) multiplied by the S&P 500's 3-month trend (strictly `T-1` prior US close!). | Global risk-on/off flows and US tech demand genuinely drive Indian exporters and FII risk appetite. | **`-0.67%` CAGR · `-0.62%` PT · `-0.04` SR · MaxDD worsens `-1.32%`** (`-11.88%`) — **ESSENTIAL** | **7.5 / 10 (High).** US remains ~60% of global equity capital; Indian IT & FII flows stay linked to S&P 500 cycles. |
| **8** | **Frog-in-the-Pan Smoothness** (`z_fip_smooth_6m`) | Over the last 6 months, did the stock climb via *many quiet green days* (like slowly warming water) or via *one violent news spike* followed by chop? (`% green days − % red days`). | *Da, Gurun & Warachka (2014).* Humans ignore slow, boring daily gains (under-reaction → trend keeps going) but overreact to dramatic 1-day news spikes (which fizzle out). | **`-0.48%` CAGR · `-0.55%` PT · `-0.03` SR** — **ESSENTIAL** | **9 / 10 (Very High).** Pure human attention psychology; works identically in US, Europe, and India. |
| **9** | **Retail Lottery Spike — MAX 21D** (`z_max_ret_21d`) | The stock's single biggest 1-day percentage jump in the past month. ML uses this with a *negative* sign to **avoid** stocks that just had a crazy casino spike. | *Bali, Cakici & Whitelaw (2011).* Retail traders pile into stocks that just jumped `+12%` in a day hoping for a lottery ticket — pushing them temporarily too high so they underperform next month. | **`-0.38%` CAGR · `-0.54%` PT · `-0.03` SR** — **ESSENTIAL** | **9 / 10 (Very High).** Retail FOMO on India's booming demat apps (Zerodha/Groww) makes this *stronger*, not weaker. |
| **10** | **6-Month Downside Volatility** (`z_down_vol_126d`) | Like overall volatility, but measures bumpiness **only on red (losing) days**. Doesn't punish a stock for having big *up* days! | Separates healthy rising stocks from fragile stocks that suffer sudden air-pocket drops. | **`-0.45%` CAGR · `-0.52%` PT · `-0.03` SR · MaxDD `-0.27%`** — **ESSENTIAL** | **8.5 / 10 (High).** Downside jump risk is a structural signal of fragile balance sheets or crowded positioning. |
| **11** | **Sector-Neutral 3M Alpha** (`z_sec_rel_63d`) | How much a stock beat (or trailed) the **median company in its own industry** over the last 3 months (e.g., HDFC Bank vs. other Banks, not vs. Pharma). | Separates the best *company* inside a sector from mere sector luck, allowing ML to find leaders even in quiet sectors. | **`-0.30%` CAGR · `-0.35%` PT · `-0.02` SR** — **HELPFUL** | **8.5 / 10 (High).** Within-industry competitive advantage (market-share gainers vs. laggards) is timeless. |
| **12** | **NIFTY 50 vs. 200-DMA** (`bench_dist_200`) *(HGB tree only)* | How far above or below its 1-year average price NIFTY 50 is sitting today (e.g., `+8%` bull vs. `-6%` bear). | Allows the shallow decision tree (`HGB`) to subtly adjust how aggressively it trusts momentum vs. low-vol depending on whether India is in a bull or bear market. | **`-0.25%` CAGR · `-0.31%` PT · `-0.02` SR · MaxDD `-0.11%`** — **HELPFUL** | **8 / 10 (High).** Market regime conditioning is a foundational macro fact. Excluded from linear Ridge so it cannot corrupt cross-sectional ranking. |
| **13** | **NSE 12–1M Vol-Adj Momentum** (`z_mom_12m1m_voladj`) | Classic 12-to-1 month total return divided by 1-year volatility (before stripping out NIFTY beta). | Complements CAPM residual momentum when strong broad-market leaders legitimately outperform alongside a rising market. | **`-0.37%` CAGR · `-0.23%` PT · `-0.03` SR** — **HELPFUL** | **8 / 10 (High).** 200-year empirical anomaly; kept strictly inside regularized ML (not raw prior) so beta stays controlled. |
| **14** | **Cross-Sectional RSI (14-Day)** (`z_rsi14`) | A 2-week speedometer (`0–100`) comparing a stock's recent up-days vs. down-days relative to the other 129 stocks. | Helps ML detect short-term overheated exhaustion vs. healthy dip-buying pullbacks inside an uptrend. | **`-0.36%` CAGR · `-0.22%` PT · `0.00` SR · MaxDD `-0.14%`** — **HELPFUL** | **7.5 / 10 (Medium-High).** Short-term mean-reversion inside longer uptrends is persistent due to liquidity provision. |
| **15** | **NSE 6–1M Vol-Adj Momentum** (`z_mom_6m1m_voladj`) | Classic 6-to-1 month return divided by volatility. | Captures medium-term trend persistence alongside `z_resid_6m1m`. | **`-0.29%` CAGR · `-0.19%` PT · `-0.02` SR** — **HELPFUL** | **8 / 10 (High).** Pairs stably with residual momentum under Ridge L2 regularization (`alpha=500`). |
| **16** | **1-Month Short-Term Reversal** (`z_ret_21d`) | How much the stock jumped or dropped over the *most recent 21 trading days* (1 month). | *Jegadeesh (1990).* Stocks that surged too fast in the very last month often take a breather; stocks that dipped slightly in a strong 12M trend offer cleaner entries. | **`-0.05%` CAGR · `-0.14%` PT · `-0.01` SR** — **MILD POSITIVE / NEUTRAL** | **7.5 / 10 (Medium-High).** Classic liquidity-provision reversal effect. |
| **17** | **NSE Breadth `> 50-DMA`** (`nse_breadth_50d`) *(HGB tree only)* | What percentage of our 130 Indian stocks are trading above their 50-day average (e.g., `75%` healthy rally vs. `25%` broad selloff). | Gives the shallow `HGB` tree context on whether rallies are broad or narrow. (Note: primary crash defense comes from the 6-Gauge Risk Gate, not here.) | **`0.00%` CAGR · `0.00%` PT · `0.00` SR** — **NEUTRAL IN ML** *(ESSENTIAL inside the 6-Gauge Risk Gate!)* | **8 / 10 (High).** Harmless in HGB (`min_samples_leaf=100`); kept for regime interaction symmetry. |
| **18** | **India VIX Regime** (`vix_level`) *(HGB tree only)* | India's official "Fear Index" derived from NIFTY options (`~12–15` calm; `19+` nervous; `23+` panic). | Lets `HGB` dampen aggressive stock bets during volatility spikes. (Primary defense is in the 6-Gauge Gate & DM Damper.) | **`-0.03%` CAGR · `+0.07%` PT · `-0.01` SR · MaxDD `-0.13%`** — **NEUTRAL / SLIGHT RISK BUFFER** | **8 / 10 (High).** Implied volatility spikes reliably mark stress regimes. |
| **19** | **Overnight Institutional Sentiment** (`z_overnight_sent_63d`) | Sum of the stock's **overnight price gaps** (today's 9:15 AM open vs. yesterday's 3:30 PM close) over 3 months. | *Lou, Polk & Skouras (2019).* Institutions and mutual funds digest corporate filings/global cues overnight and place opening-auction orders, whereas retail noise dominates midday trading. | **`-0.02%` CAGR · `+0.10%` PT · `-0.03` SR · MaxDD worsens `-0.48%`** (`-11.04%`) — **HELPFUL RISK/SHARPE BUFFER** | **7 / 10 (Medium-High).** Structural open-vs-intraday clientele split; worth monitoring if NSE ever introduces extended/evening equity hours. |

> **What happened to Feature #20 (`z_macro_tailwind_fx` — USD/INR stock beta)?**  
> In our 33-run Leave-One-Out audit, **`z_macro_tailwind_fx` was the ONLY feature that actively hurt the system** — removing it **increased** Pre-Tax CAGR by `+1.06%/yr`, **increased** Honest Post-Tax CAGR by `+1.23%/yr`, **boosted** Sharpe by `+0.07` (`1.77 → 1.84`), and **improved** MaxDD (`-10.56% → -10.23%`)! Why? Because **the RBI heavily manages and intervenes in USD/INR**, keeping the exchange rate artificially flat for long stretches so rolling 126-day FX betas are mostly noise. We verified pruning it across a full 9-cell plateau (`7/9` PT wins, `3/3` at `40D`, won `2024`, `2025`, and `2026 YTD`) and **permanently removed it** (`20 → 19` features).

---

## 4. Every Single One of the 13 System Layers (The Car's Engine, Steering, Brakes & Airbags)

Factors only tell you *which* stocks look good. **How you build the portfolio and protect it during crashes** is just as important — often more important. Here are all **13 architectural layers** of `V7.5-Core`, with their exact Leave-One-Out proof from `task-1981.log`:

| # | System Layer | Kitchen-Table Explanation | What Breaks If You Remove It (LOO Proof) | Will It Last? |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **6-Gauge India Risk Gate** *(VIX ≥19, VIX ≥23, NIFTY <200D, Breadth <42%, Hi-Lo <-10%, Gold/NIFTY >+14% → 100/85/55/25% equity)* | The **storm radar**. When 1 gauge flashes red, we ease to `85%` stocks; 2 gauges → `55%`; 3+ gauges → `25%` stocks (`75%` safe cash + gold). | **MaxDD EXPLODES from `-10.56%` to `-25.22%` (`-14.66%` worse!)** and Sharpe crashes `-0.32` (`1.77 → 1.45`). Gives up some bull-market raw CAGR (`+4.24%`) in exchange for sleeping peacefully and surviving crashes! | **9 / 10.** Multi-gauge consensus (vol + trend + breadth + flight-to-gold) avoids single-indicator failure. |
| **2** | **20-Day Official NSE Delivery Shield** *(`35% ≤ deliv_20d < 98%`, `≥ ₹2 Cr/day`, Entry + Exit parity)* | The **casino bouncer**. Uses official NSE exchange files (`T-1` lagged) to refuse buying — and to *exit at rebalance* — any stock where real demat delivery collapses below `35%` (speculative intraday churn), hits `≥98%` (T2T surveillance trap), or trades `<₹2 Cr/day` delivery. | **Pre-Tax drops `-2.06%/yr`, Post-Tax drops `-1.78%/yr`, Sharpe crashes `-0.24`, MaxDD worsens `-1.91%`** (`-12.47%`). Our #1 India-specific structural edge! | **7 / 10 (Monitor!).** Works brilliantly now, *but* SEBI Peak Margin & T+1 pushed median delivery up (`34.4% → 51.5%`). Watch `below_35_pct < 5%` telemetry! |
| **3** | **50% Walk-Forward ML Sleeve** *(5 expanding folds, 40D embargo, Vasicek Residual-Sharpe target)* | The **pattern-recognition co-pilot**. Learns subtle combinations across all 19 signals strictly from past data (with a 40-day blindfold/embargo so it can never peek ahead). | Removing ML (100% Prior) cuts **Post-Tax `-1.56%/yr`, Sharpe `-0.10`, MaxDD `-0.45%`**. | **8 / 10.** Heavy regularization (`Ridge alpha=500`, `HGB depth=2`) prevents overfitting. |
| **4** | **50% Canonical Multi-Factor Prior** *(Zero ML parameters — 6 textbook factors)* | The **steady anchor**. Guarantees the system never goes crazy or chases a weird historical fluke, even if market regimes flip overnight. | Removing the Prior (100% ML) cuts **Post-Tax `-1.42%/yr`, Sharpe `-0.15`, MaxDD `-0.17%`**. The 50/50 marriage beats *both* halves alone! | **9.5 / 10.** Zero fitted parameters; pure academic finance physics. |
| **5** | **30% `GOLDBEES` Defensive Reserve** *(Idle cash split `70%` Cash / `30%` Gold)* | When the storm radar pulls money out of stocks, `30%` of that parked money buys Indian Gold (`GOLDBEES`). Gold often rallies during crises AND when the Indian Rupee weakens. | Removing Gold cuts **Pre-Tax `-1.47%/yr`, Post-Tax `-1.17%/yr`, and worsens MaxDD `-1.58%`** (`-12.14%`). *(Note: 2021–2026 was a great gold bull run; don't expect gold to add +1.5%/yr every decade, treat it as crisis insurance.)* | **8.5 / 10.** Multi-century INR/crisis hedge, even when gold goes sideways for stretches. |
| **6** | **Shallow `HistGradientBoosting` Trees** *(`40%` of ML ranker; `depth=2, leaf=100, L2=10`)* | Allows simple "if-then" interactions (e.g., *"high momentum works great when VIX is calm, be careful when VIX is high"*) without memorizing noise. | Removing HGB (Ridge-only ML) cuts **Post-Tax `-1.13%/yr`, Sharpe `-0.09`, MaxDD `-0.63%`**. | **8 / 10.** Ultra-shallow depth-2 trees cannot memorize individual stocks. |
| **7** | **`-10% / -6%` Own-Portfolio Drawdown Brake** | The **personal emergency airbag**. If *your* portfolio ever falls `-10%` from its peak, cut equity exposure in half until it recovers past `-6%` and closes above its 20-day average (`× 1.005`). | Removing it cuts **Post-Tax `-0.65%/yr`, Sharpe `-0.07`, and worsens MaxDD `-1.37%`** (`-11.93%`) — and across 2008/2020 it saved **`10.3%`** of peak-to-trough pain! | **9 / 10.** Model-agnostic safety net — works even if every other signal breaks simultaneously. |
| **8** | **Indian `320–364d` LTCG Tax-Cliff Guard** | The **smart CA on your shoulder**. If a profitable stock slips outside Top-30 (still inside Top-60 and passing all shields) while aged `320–364` days, hold it one extra cycle to pay `13.0%` LTCG instead of `20.8%` STCG. | Removing it cuts **Pre-Tax `-0.32%/yr`, Honest Post-Tax `-0.53%/yr`, Sharpe `-0.01`, MaxDD `-0.08%`** (improved *every* metric simultaneously!). | **8.5 / 10.** Valid as long as India maintains a meaningful gap between STCG (`20.8%`) and LTCG (`13.0%`) at 365 days. |
| **9** | **López de Prado `HRP` Position Sizing** *(126D Ledoit–Wolf shrunk covariance + `≤12%` stock / `≤20%` sector caps)* | Groups stocks that behave like twins (e.g., two PSU lenders) into branches of a family tree so you don't accidentally bet double on the same risk. | Replacing HRP with Equal Weight cuts **Post-Tax `-0.50%/yr`, Sharpe `-0.08`, MaxDD `-0.52%`** (and HRP lifted 2022 bear-market return from `+4.1%` to `+14.8%`!). | **9.5 / 10.** Pure mathematical diversification with zero return-forecasting fragility. |
| **10** | **4-Tranche Staggering** *(4 equal 25% sleeves, 10 trading days apart, 40D cycle)* | Instead of betting 100% of your timing luck on one day every 2 months, you refresh 25% of the portfolio every 2 weeks. Eliminates "unlucky start-date" lottery. | Going to 1-tranche increases return slightly (`+1.05%` PT in this 5Y window) **at the cost of `-1.58%` worse MaxDD (`-12.14%`) and huge timing-luck dispersion** (across 22Y, 4-tranche added `+0.23` Sharpe). | **9.5 / 10.** Core institutional "rebalance timing luck" defense (Corey Hoffstein / Newfound Research). |
| **11** | **Regularized Linear `Ridge` Ranker** *(`60%` of ML ranker; `alpha=500`, cross-sectional features only)* | A heavily damped straight-line scorecard that keeps ML smooth, stable, and immune to tree-splitting quirks. | Removing Ridge (HGB-only) cuts **Post-Tax `-0.23%/yr`, Sharpe `-0.04`, MaxDD `-0.59%`**. | **9 / 10.** Textbook small-panel stability anchor. |
| **12** | **`18%` Ex-Ante Volatility Cap** *(60D portfolio covariance)* | The **cruise-control speed limiter**. If the chosen stocks become unusually jumpy (`>18%` annualized vol), gently trims equity sizing until calm returns. | Removing it adds a tiny `+0.12%` PT but **worsens Sharpe (`-0.02`) and MaxDD (`-0.78%`, to `-11.34%`)**. | **9 / 10.** Realized volatility clusters reliably in every market on earth. |
| **13** | **Daniel–Moskowitz Bear-Rebound Crash Damper** *(`NIFTY 252D < 0 & VIX ≥ 19`)* | **Rarely fires — pure catastrophe insurance.** Only activates after a true 1-year bear market while fear is still high (like March 2009 or April 2020), blending 50% of the prior toward defensive `BAB + Low-Vol + Residual Mom` so beaten-down junk rebounds don't blow up momentum. | Neutral in `2021–2026` (`0.00%` PT impact — because `2021–2026` had no prolonged `-20%` bear market!). Proven value is in `2009` and `2020` snapbacks. | **9 / 10.** Costs `0%` during normal years and protects against momentum's #1 historical Achilles heel. |

**Bottom line of the 33-run audit:** **12 of 13 layers** actively earned their keep in `2021–2026`, and Layer #13 is free catastrophe insurance for the next 2009/2020-style bear-market snapback. There is **zero dead weight** left in `V7.5-Core`.

---

## 5. The Scorecard: How Good Is `V7.5-Core`, How Does It Compare to the Pros, and What Should You Realistically Expect?

### 5.1 Official Verified Scorecard (`2021-07 → 2026-09`, `K=10`, `40D`, Exact Statutory NSE Costs + FIFO Equity & Post-Budget 2024 Reserve Taxes)

| Engine / Benchmark | Pre-Tax CAGR | Honest Post-Tax (`LIQUIDCASE 31.2%`) | Honest Post-Tax (`Arbitrage 20.8%`) | Excess Sharpe (`6.2%` RF) | Max Drawdown | IIMA 4-Factor Pure Alpha (`t-stat`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NIFTY 50 TRI (`NIFTYBEES.NS`)** | `+9.06%` | `+8.25%` | `+8.25%` | `0.26` | `-16.11%` | `0.00%` (Benchmark) |
| **Old Frozen `V6` Base** *(Unshielded)* | `+23.40%` | `~+19.6%` | `~+19.8%` | `1.21` | `-9.90%` | `+3.01%` (`t=+0.90`) |
| **Frozen `V7` Base** *(Unshielded)* | `+26.54%` | `+22.22%` | `+22.39%` | `1.50` | `-12.18%` | `+7.05%` (`t=+1.98`) |
| **`V7.5-Core` (`20` feats, Pre-FX Prune)** | `+28.91%` | `+24.45%` | `+24.61%` | `1.77` | `-10.56%` | `+10.22%` (`t=+2.95`) |
| **LIVE `V7.5-Core` (`19` feats — `main` DEFAULT)** | **`+29.97%`** | **`+25.67%`** | **`+25.84%`** | **`1.84`** | **`-10.23%`** | **`+10.96%` (`t=+3.14`)** |

* **Year-by-Year Reality Check (Live 19-Feature `V7.5-Core`, `K=10, 40D`, Pre-Tax):**
  * **`2021`:** **`+42.6%`** | **`2022` (Global Bear):** **`+17.4%`** | **`2023`:** **`+56.2%`** | **`2024`:** **`+35.7%`** | **`2025` (Momentum Drought):** **`+9.7%`** | **`2026 YTD` (Market Selloff):** **`-1.4%`** *(while NIFTY 50 TRI fell `-10.6%`!)*
* **Proof It Isn't Overfit ("Overfitting Score: `1.5 / 10`" — Institutional Grade):**
  * **Zero Parameter Cliff:** Swept `9/9` `(Top-K × Rebalance)` combinations when adding the Delivery Shield, and pruning `z_macro_tailwind_fx` won `7/9` PT / `7/9` MaxDD (`3/3` at `40D`).
  * **Beats 50/50 Random Portfolios (`p = 0.0196`):** Identical risk shields on randomly picked stocks earned `+14.52%` (`SR 0.78`); our stock selector adds `+12%–15%/yr` of pure selection skill.
  * **17 Real Indian Crashes ("Fallen Angels" — `YESBANK -95%`, `ZEEL -88%`, `ATGL -86%`, `IDEA -84%`, etc.):** `V7.5-Core` beat unshielded `V7` (`+25.85%` vs `+24.46%`, `SR 1.54` vs `1.37`).
  * **Blind 126 Unseen Indian Smallcaps (`0%` overlap):** `V7.5-Core` beat `V7` completely blindly (`+18.26%` vs `+16.89%`, `SR 0.98` vs `0.86`).
  * **Long Multi-Cycle Proof (`2007–2026` 19Y & `2004–2026` 22Y):** `V7`'s core prior + crash shields survived `2008` GFC, `2009` snapback, `2011` bear, `2018` NBFC crisis, and `2020` COVID with `~-17%` MaxDD vs. NIFTY's `-59.9%`.
  * **Extreme Cost Torture Test (`50 bps/side` = `100 bps` round-trip, ~3× real NSE costs):** Still earned **`+22.1%+` post-tax**.
  * **Deflated Sharpe Ratio (`DSR`):** Clears **`99.77%`** confidence even after penalizing for `N = 80` cumulative experiments!

### 5.2 How `V7.5-Core` Compares Against Real-World Quant Firms & Indian PMS Peers

| Peer / Benchmark | What They Do Well | How `V7.5-Core` Compares Honestly |
| :--- | :--- | :--- |
| **Robeco Quantitative Equities (David Blitz)** | Pioneer of Residual Momentum, Low-Vol, Net Payout Value, and deep multi-decade global PIT fundamentals across thousands of stocks. | **Closest intellectual cousin!** We share CAPM Residual Momentum, Low-Vol/BAB, and zero-overfit discipline. Robeco's edge over us: decades of paid PIT fundamental data (Value/Quality) across 3,000+ stocks. Our edge for Joe: zero `1%–2%` management fees, tiny personal capital (zero market impact), and official NSE Delivery Shield. |
| **AQR Capital Management** | Multi-factor (Value + Momentum + Low-Beta + Quality) + Volatility Targeting + Daniel–Moskowitz momentum crash research. | We adopted their exact BAB (`-Vasicek β`), vol-targeting, and crash-damper research. AQR's advantage: true multi-asset + Value/Quality diversification during momentum droughts (our deferred Phase-2 roadmap). |
| **Newfound Research (Corey Hoffstein)** | Invented/popularized **Rebalance Timing Luck** defense (overlapping staggered tranches). | Our **4-Tranche Staggering (`40D` cycle, 10D apart)** directly implements Hoffstein's institutional standard. |
| **Nifty 200 Alpha Low-Vol 30 Index** | Simple rules-based NSE index combining Alpha + Low Volatility. | Great passive benchmark! `V7.5-Core` improves on it with residual momentum (less beta), CGO overhang, HRP sizing, 6-Gauge + `-10%` crash defense, `70/30` cash/gold reserve, and the `<35%` NSE Delivery Shield (cutting MaxDD roughly in half). |
| **Indian Momentum PMS / Smallcases** *(Capitalmind Adaptive Momentum, Wright Research, WeekendInvesting, Qode Advisors)* | Strong Indian momentum focus; some use cash/gold regimes or low-vol blends. | Most retail/PMS momentum products use raw momentum without residual beta stripping, lack HRP correlation clustering, lack staggered 4-tranche timing defense, lack the official 20D NSE Delivery Shield + T2T guard, and charge `1.5%–2.5%` annual fees + `10%–20%` profit-sharing (which compounds away `₹20L–₹40L` per `₹1 Cr` over 10 years). |

### 5.3 Realistic Live Money Expectations (Budget on THIS, Not the Backtest Peak!)

Backtests ALWAYS look slightly rosier than real life (2021–2024 was one of the strongest Indian bull runs in history, plus INR gold had an unusual `+1.85%/yr` tailwind). Apply a healthy **30–40% real-world safety haircut** when planning your financial life:

| Scenario (5–10 Year Horizon) | Expected Net Post-Tax CAGR *(after all taxes, STT & slippage)* | What `₹10 Lakh` Becomes in 5 Yrs / 10 Yrs | What `₹1 Crore` Becomes in 5 Yrs / 10 Yrs |
| :--- | :---: | :---: | :---: |
| **Conservative / Choppy Decade** *(lots of 2018 & 2025-style sideways years)* | **`+12% to +15%`** | `₹17.6L – ₹20.1L` / `₹31.1L – ₹40.5L` | `₹1.76 Cr – ₹2.01 Cr` / `₹3.11 Cr – ₹4.05 Cr` |
| **Realistic Base Case** *(normal Indian cycle — **PLAN YOUR LIFE ON THIS**)* | **`+16% to +19%`** | `₹21.0L – ₹23.9L` / `₹44.1L – ₹56.9L` | `₹2.10 Cr – ₹2.39 Cr` / `₹4.41 Cr – ₹5.69 Cr` |
| **Strong Bull Cycle** *(similar to 2021–2024)* | **`+22% to +26%`** | `₹27.0L – ₹31.8L` / `₹73.0L – ₹100.9L` | `₹2.70 Cr – ₹3.18 Cr` / `₹7.30 Cr – ₹10.09 Cr` |
| *NIFTY 50 TRI Reference* | *`~+9% to +11%`* | *`₹15.4L – ₹16.9L` / `₹23.7L – ₹28.4L`* | *`₹1.54 Cr – ₹1.69 Cr` / `₹2.37 Cr – ₹2.84 Cr`* |

* **Emotional Pain You MUST Expect in Advance:**
  * Expect **`-10% to -18%`** drops on your screen every 2–4 years (`-₹1L to -₹1.8L` on `₹10L`; `-₹10L to -₹18L` on `₹1 Cr`).
  * Expect **6 to 12 month stretches where your account goes nowhere** (flat or down `-3%` to `-6%`), while your cousin brags about a random smallcap stock doubling. **That is completely normal — NEVER override the system during boring months.**

---

## 6. Will `V7.5-Core` Stand the Test of Time? (Honest Durability Rating: **`8.0 / 10`**)

No honest quant tells you a strategy is "10/10 invincible." Here is the unvarnished truth:

### Why It Scores a Strong `8.0 / 10` (Built to Last 10+ Years)
1. **Zero "Magic Indicator" Fragility:** Its top 6 drivers (`BAB`, `CAPM Residual Mom 12-1M & 6-1M`, `Intermediate Mom 12-7M`, `Capital Gains Overhang`, `Low-Vol`) are rooted in **permanent human psychology** (loss aversion, slow institutional decision-making, lottery-seeking) and **math** (volatility drag, HRP diversification) — proven across 30–90 years of global markets.
2. **Simple, Heavy-Handed ML Restraint:** 50% of the score uses **zero ML parameters**, and the 50% ML half uses heavily penalized linear `Ridge(alpha=500)` + ultra-shallow `depth=2` trees. It cannot memorize 2023's market quirks.
3. **Liquid Large/Midcap Fortress:** By sticking to India's 130 most liquid NSE stocks (mostly F&O stocks with dynamic price bands), you never get trapped in illiquid smallcap lower-circuit lockouts or market-impact decay as your account grows toward `₹1–5 Crore`.
4. **3 Independent Safety Nets:** Even if stock selection goes completely cold for a year, the 6-Gauge Risk Gate + 18% Vol Cap + `-10%` Drawdown Brake + 70/30 Cash/Gold reserve protect your capital.

### The 4 Black-Swan Scenarios Where `V7.5-Core` CAN Struggle (Know Your Enemy!)
1. **Scenario 1 — Violent V-Shaped "Junk Rally" Snapback** *(like April–June 2009 or May 2020)*:
   * *What happens:* After a sharp crash, the Risk Gate & `-10%` Drawdown Brake have cut equity exposure to `25%–50%`. Suddenly the RBI/Fed prints emergency money, and beaten-down, high-debt, high-beta junk stocks rocket `+40%` in 3 weeks.
   * *Impact:* `V7.5-Core` will **lag the rebound badly** for 1–3 months while waiting for the `-6%` brake release and staggered sleeves to re-enter. *(This is the unavoidable insurance premium you pay to avoid `-50%` crashes.)*
2. **Scenario 2 — Multi-Year Sideways Chop / Pure Value-Only Cycle** *(like 2018 or a prolonged 2025–2026)*:
   * *What happens:* Leadership rotates violently every 4–6 weeks so 6–12 month winners reverse right after we buy them, while dirt-cheap "deep value" stocks (which we don't yet have a PIT sleeve for) lead.
   * *Impact:* `V7.5-Core` treads water (`0% to +8%/yr`) for 12–18 months. *(This is why we are silently logging Day-1 PIT fundamental snapshots every day for a future Value/Quality sleeve!)*
3. **Scenario 3 — SEBI Market-Structure Drift (`T+0` Settlement / Margin Changes):**
   * *What happens:* SEBI's Peak Margin (2021) and `T+1` settlement (2023) already pushed median NSE delivery from **`34.4%` (2020) → `51.5%` (2026)**, shrinking the `<35%` shield hit rate from `51.4%` down to **`5.3%` (`7/131` stocks in Nov 2026)**. If `T+0` pushes almost *every* stock above `35%`, the static `<35%` bouncer goes to sleep.
   * *Built-in Defense:* We already built **SEBI Delivery Drift Telemetry** (`drift_status: "warning"` if `below_35_pct < 5.0%`) and pre-tested the exact fallback (`Hybrid: <35% OR Bottom-20%`).
4. **Scenario 4 — Extreme Geopolitical / Capital-Control / Correlation-1 Shock:**
   * *What happens:* Overnight war, sovereign default, exchange closure, or a 1970s-style rate shock where Indian stocks AND gold fall together while the market gaps down `-15%` before any limit order can execute.
   * *Impact:* No quantitative model on earth stops an overnight gap. This is why you **never invest emergency living money, short-term goal money (< 3 years), or borrowed money** in the stock market!

---

## 7. What We Tried, What Worked, and The Complete Research Graveyard (Do NOT Re-Test!)

Over months of disciplined research, we tested **dozens** of famous Wall Street, Indian PMS, and textbook ideas strictly Out-of-Sample (`OOS`) with real Indian taxes and costs. **Most ideas that sound brilliant at a dinner party FAIL in real NSE data.**

### 7.1 What Actually WORKED (Earned Its Place in `V7.5-Core`)
* **CAPM Residual Momentum (`12-1M` & `6-1M`) + Intermediate Momentum (`12-7M`) + Capital Gains Overhang (`CGO 126D`) + Betting Against Beta (`-Vasicek β`):** Crushed raw momentum across clean OOS (`+2.8%/yr` PT), 17 real Indian crashes (`+6.3%/yr` PT), 8.8Y (`9/9` cells), and 19Y (`2007–2026`).
* **Vasicek Residual-Sharpe ML Target + Excluding Date-Constant Macro Cols from Linear Ridge Only:** Doubled pure 2-factor alpha (`+4.57% → +8.24%`) while *lowering* market beta (`0.59 → 0.49`).
* **López de Prado `HRP` (`126D Ledoit–Wolf`):** Beat equal-weight and heuristic inverse-vol across `9/9` parameter cells and transformed `2022` bear-market return (`+4.1% → +14.8%`).
* **Strictly Causal `T-1` 20-Day Official NSE Delivery Shield (`35% ≤ deliv_20d < 98%`, `≥ ₹2 Cr/day`, Entry + Retention Exit):** Added **`~+2.5%/yr` post-tax**, boosted Sharpe `1.50 → 1.77`, cut MaxDD `-12.18% → -10.30%`, and swept `9/9` plateau cells + 17 Fallen Angels + 126 blind smallcaps.
* **Indian `320–364d` LTCG Tax-Cliff Guard:** Improved every single metric (`+0.53%/yr` PT, lower MaxDD, lower STCG tax).
* **Pruning RBI-Managed `z_macro_tailwind_fx` ONLY (`20 → 19` features):** Lifted Pre-Tax `+28.91% → +29.97%`, Honest PT `+24.45% → +25.67%`, `SR 1.77 → 1.84`, `MaxDD -10.56% → -10.23%`, `4F Alpha +10.22% → +10.96% (t=+3.14)`, winning `7/9` plateau cells while making the model *simpler*.

### 7.2 The Complete Research Graveyard (Strictly Tested OOS & KILLED — Never Waste Time Re-Testing!)

| # | Idea We Tested | Why It Sounded So Smart | Exact OOS Proof of Why We Killed It |
| :-: | :--- | :--- | :--- |
| **1** | **Pure Delivery % Stock Picker** | "Just buy the highest delivery % stocks!" | Earned only **`+9.84%` CAGR** (bought sleepy dividend stocks). Delivery % works ONLY as a `<35%` bouncer (`p = 0.0000`), NOT a ranker. |
| **2** | **Upper Delivery Cap (`> 72%`)** | "Avoid slow stocks with too much delivery." | Hurt CAGR & Sharpe — institutional breakouts often show `75%–85%` delivery. (Only `≥ 98%` T2T surveillance traps are blocked.) |
| **3** | **Adaptive Bottom-20%/25% Delivery Cutoff** | "Automatically adapt to SEBI's rising median delivery." | Trailed Static `<35%` on CAGR (`+28.66–29.03%` vs `+29.42%`), Post-Tax, MaxDD (`-12.1% to -12.5%` vs `-10.3%`), and `2026 YTD` — threw away healthy `42%`-delivery winners. *(Kept ONLY as future fallback if `below_35_pct < 5%`.)* |
| **4** | **Earnings Surprises / PEAD** *(Quarterly EPS + `CAR[-1,+2]`)* | "Buy stocks that beat earnings estimates (Wright/Robeco style)." | Tested TWICE: (a) 2,974 NSE reports (`+2d` lag): IC **`-0.0065`**, hurt `5/6` years; (b) 1,375 events `CAR+SUE`: IC **`-0.0041 (t=-0.23)`**, only `43.3%` coverage, worsened MaxDD. Indian earnings leak before results day! |
| **5** | **Robeco 3-Factor Residual Momentum (`MKT+SMB+HML`)** | "Strip Size & Value beta too, not just Market beta." | Standalone IC **`-0.0176 (t=-0.79)`**; `15%` blend cut CAGR **`-1.90%/yr`** & Sharpe `1.77 → 1.62`. On `N=130` NSE stocks, 3-factor rolling regression is too noisy; 1-factor CAPM residual momentum is far cleaner. |
| **6** | **Indian March Tax-Loss Harvesting (`TLH`)** | "Sell losers before March 31 to save STCG tax, then buy back." | Because our 40D rebalance already cuts losers mechanically, only `16` tiny lots (`₹14k` loss) existed across 5.2 years. NSE round-trip STT (`₹459`) exceeded tax saved (`₹114`) → **net loss `-₹345`**. |
| **7** | **Forced 9–12 Month LTCG Hold** | "Refuse to sell any stock aged 270–364d to avoid STCG." | Holding fading losers for 3 extra months destroyed `-0.45%/yr` alpha → **less after-tax money** (`₹2.76 Cr` vs `₹2.79 Cr`). Only the tight **`320–364d` profitable Top-60 cliff guard** works! |
| **8** | **Official NSE FII Derivatives OI Veto (`FII < 20%`)** | "Go defensive when foreign investors are 4:1 short." | Cost **`-15.6%` in `2023`** when domestic Indian SIP money powered a massive rally while FIIs stayed short. Keep FII OI visual-only! |
| **9** | **Expanding Universe (`130 → 231/264` Stocks)** | "More mid/smallcaps = more winners!" | `130→231` diluted OOS IC (`+0.0446 → +0.0322`) and cut CAGR/Sharpe; `264`-stock expansion with `Screener.in` fundamentals carried **`5.5–6/10` survivorship/restatement bias** (kept quarantined on `feat/v75-frontier`). |
| **10** | **Smart `NIFTYBEES` Idle Cash Parking** | "Park Risk-Gate cash in NIFTY when NIFTY > 200-DMA." | Hurt BOTH 5Y (`SR 1.96 → 1.69`, `MaxDD -10.6% → -14.5%`) and 22Y (`MaxDD -26.8% → -31.9%`). `70/30` Cash/Gold won decisively. |
| **11** | **Single-Stock Trailing Stops (`-12% / -15%`)** | "Cut individual losing stocks quickly." | Whipsawed out of volatile winners right before rebounds; hurt CAGR & Sharpe. Portfolio-level `-10%` Drawdown Brake is vastly superior. |
| **12** | **Dropping `HGB` / 4Y Decay / Blind Monotonic HGB** | "Simplify ML even more or weight recent years heavier." | Dropping HGB cut **`-1.96%/yr`** (now `-1.13%` PT in `V7.5-Core`); 4Y decay cut `-1.27%/yr` (forgot 2018–2020 bear data!); blind monotonic HGB cut `-0.64%/yr`. |
| **13** | **Dropping ALL 5 ML Macro Features** | "If FX hurt, drop all macro features from ML." | Failed 9-cell plateau (`SR 4/9`, `MaxDD 4/9`); `2025/2026` dropped to `+5.1% / -3.9%`. Prune ONLY `z_macro_tailwind_fx`; keep `z_macro_tailwind_sp500` & `bench_dist_200`. |
| **14** | **Global Dollar (`DXY`) / US-10Y (`^TNX`) Squeeze** | "Hélène Rey Global Financial Cycle kills EM stocks." | Zero predictive power on India `2017–2026` (`p > 0.85`). India's `₹25,000+ Cr/month` domestic mutual-fund SIP engine decoupled NIFTY from classic EM dollar squeezes. |
| **15** | **Double-Orthogonalized Amihud Illiquidity** | "Illiquid stocks earn an academic liquidity premium." | Looked good (`+0.0339` IC) on surviving winners, then **collapsed to `-0.0050 IC (t=-0.40)` on the 17 Fallen Angels** — classic survivorship-bias trap that buys dying companies! |
| **16** | **Symmetric Mid-Cycle Re-Entry** | "Jump back into 100% stocks immediately if VIX drops." | Got whipsawed in bear rallies, **worsened MaxDD `-2.52%`** (`-9.99% → -12.51%`), and inflated turnover `+50%`. One-way de-risking + 10D staggered sleeves is cleaner. |
| **17** | **Smooth Sigmoid Regime Gate (`V8`)** | "Replace step-function 6-Gauge with a smooth curve." | Cut Sharpe `1.50 → 1.34`, worsened MaxDD `-12.18% → -13.45%`, and slashed 4F Alpha `7.05% → 4.78%`. |
| **18** | **`V8-TopRank` (`fwd_rank²`) & `V8-FullAFML`** | "Convex top-decile target & AFML uniqueness weights." | Looked flashier on surviving stocks, **failed on the 17 Real NSE Fallen Angels** (`-0.29%` to `-1.68%` behind `V7`, `PBO 55.7%`). Rejected to prevent overfitting. |
| **19** | **Ledoit–Wolf `QIS` Nonlinear Shrinkage, `KMZ` RFF-Ridge, `JKMP` Signal Smoothing, `Gârleanu–Pedersen` Aim `0.65`, `LedoitWolf` inside 60D Vol Cap** | Cutting-edge 2020–2026 quant papers. | Every single one tied or trailed `V7.5-Core` on `N=130` NSE stocks (`p=10, T=126` too small for QIS; Aim `0.65` lost `-1.7%/yr`; `sklearn.LedoitWolf` inside 60D scalar vol cap underestimated long-only equity variance). Confirms `V7.5-Core` is at the OHLCV ceiling! |
| **20** | **Extra Macro/Index Triggers** *(Midcap/BankNIFTY in Risk Gate, EWMA fast vol cap, Gold TSMOM 45/15, Brent mean-reversion, 21D Frenzy)* | More indicators = smarter system? | Redundant or pure OOS noise (`t = -0.56`); added false alarms during healthy rallies. |

---

## 8. What We Left for Later (Parked on the Shelf — With Exact Unlock Conditions)

We deliberately kept these promising ideas **OUT of live production** until the right real-world conditions are met:

1. **True Point-In-Time (`PIT`) Indian Value + Quality Second Sleeve** *(Files already built: `backend/pit_fundamentals.py`, `backend/forensic_quality_sleeve.py`, `backend/multi_sleeve_allocator.py`)*
   * **Why we want it someday:** In `2025` (a momentum drought), a zero-ML Value/Payout sleeve had `-0.203` correlation with `V7` and earned **`+20.5%`** while momentum slowed — cutting combined MaxDD below `-9.5%`.
   * **Why it's parked right now:** Free Indian websites (`Screener.in` / Yahoo) retroactively restate old financial statements and omit delisted companies — creating fake backtests (`5.5–6/10` overfitting risk).
   * **How we are solving it for FREE:** Every time you click **Sync & Retrain**, `pit_fundamentals_history` silently saves an immutable, timestamped snapshot of all 130 stocks' fundamentals (already logging since Day 1!).
   * **Unlock Condition:** Wait until we accumulate **3–5 years of true non-restated live snapshots** (or a verified BSE/NSE XBRL `+2d` filing-lag archive), then validate a `75% V7.5-Core / 25% PIT Value+Quality` blend using Robert Carver's `multi_sleeve_allocator.py`.
2. **Global Dual-Engine Portfolio: `India V7.5-Core` + `US (V7 + Track 2 SEC EDGAR)`** *(Tested in `equilens-v75-us`)*
   * **Why it's our biggest future upgrade:** India `V7` and US `V7 + SEC EDGAR` have **near-zero daily correlation (`+0.071` in INR)**! Combining them in INR Risk Parity jumped Excess Sharpe to **`1.91–1.93`**, slashed MaxDD in half to **`-5.27%`**, and hit **`Calmar 4.42`** (`+23.3%` CAGR with only `8%` volatility!).
   * **Unlock Condition:** When Joe is comfortable opening an international US brokerage account (IBKR / Vested / INDmoney) under RBI LRS rules and handling Indian ITR `Schedule FA` + LRS TCS reporting.
3. **SEBI Delivery Drift Fallback (`Hybrid: <35% OR Bottom-20%`)**
   * **Unlock Condition:** Keep Static `<35%` untouched as long as `below_35_pct ≥ 5.0%`. ONLY consider switching to the pre-tested `Hybrid (<35% OR Bottom-20%)` (`SR 1.81`, `4F Alpha +10.74%`) if SEBI `T+0` settlement or further margin changes push `below_35_pct` persistently **below `5.0%`** for multiple consecutive months.
4. **Opt-In `₹1 Cr+` Convex Portfolio Optimizer (`backend/convex_optimizer.py`)**
   * **What it is:** Boyd et al. (2017) SLSQP solver jointly optimizing alpha, 126D covariance, statutory fees, Almgren–Chriss market impact, and FIFO tax friction (`cut turnover 212.7% → 203.7%` in `V8` research).
   * **Unlock Condition:** Only needed if live capital scales past `₹1–2 Crore` and market-impact/turnover friction becomes noticeable; must beat `HRP` cleanly across all 5 Promotion Gates first.

---

## 9. Indian Taxes, Costs & Your Exact Live Execution Routine (Zero-Mistake Checklist)

### 9.1 Indian Taxes & Where to Park Idle Reserve Cash (Post-Budget July 2024 Rules!)

When you buy and sell NSE delivery shares in India, two things eat into your headline return: **exchange/statutory charges** and **income tax**. `V7.5-Core` models EVERY rupee of both honestly:

* **Statutory NSE Delivery Charges (Automatic at Zerodha/Groww/Dhan):**
  * **STT (Securities Transaction Tax):** `0.10%` on BUY + `0.10%` on SELL.
  * **Stamp Duty:** `0.015%` on BUY only.
  * **NSE Exchange + SEBI + 18% GST:** `~0.0036%` per side.
  * **CDSL/NSDL DP Charge:** `₹15.93` flat per stock on SELL day.
  * *Total statutory + realistic limit-order spread:* **`~0.15%` (`15 bps`) per side** (~`₹150` per `₹1 Lakh` traded).
* **Indian Equity Capital Gains Tax (Finance Act 2024, incl. 4% Cess):**
  * **Sold in `< 365` days (STCG):** **`20.8%`** of net profit.
  * **Sold in `≥ 365` days (LTCG):** **`13.0%`** of net profit — AND your first **`₹1,25,000` of LTCG profit every Financial Year (Apr 1 – Mar 31) is 100% TAX-FREE!**
  * **Losses help you:** Short-term losses (`STCL`) offset both STCG and LTCG; long-term losses (`LTCL`) offset LTCG; unused losses carry forward up to 8 FYs if you file ITR on time.
* **CRITICAL Post-Budget 2024 Trap on Your `70 / 30` Defensive Reserve (`LIQUIDCASE` vs. Arbitrage Fund & `GOLDBEES`):**
  * **`70%` Cash Slot — `LIQUIDCASE` vs. Direct-Plan Arbitrage Mutual Fund:**
    * Under Finance Act 2024 (effective July 23, 2024), debt ETFs like **`LIQUIDCASE`** are taxed at your **income tax slab rate (`31.2%` if you are in the 30% slab!)** no matter how long you hold them.
    * A **Direct-Plan Arbitrage Mutual Fund** (e.g., Kotak / Parag Parikh / Tata / Edelweiss Arbitrage Direct Growth) holds `>65%` hedged equity, so Indian tax law taxes it as **equity: `20.8%` STCG (`< 12m`) or `13.0%` LTCG (`≥ 12m`, shares your `₹1.25L` exemption!)** while delivering virtually the same `~6.5%–7%` low-risk yield!
    * **Joe's Money-Saving Rule:** If you are in the `30%` tax slab, use a **Direct Arbitrage Fund** for the `70%` cash reserve (saves `~₹1.2L–₹2.2L` per `₹1 Cr` over 5 years; trade-off is `T+2/T+3` redemption instead of instant ETF liquidity). If you are in a lower tax slab (`≤ 20%`) or prioritize instant brokerage liquidity, **`LIQUIDCASE`** is totally fine (and far cleaner than `LIQUIDBEES`, which creates daily dividend ITR headaches!).
  * **`30%` Gold Slot (`GOLDBEES`):**
    * Held `< 12 months`: taxed at your **slab rate (`31.2%`)**.
    * Held `≥ 12 months`: taxed at **`13.0%` LTCG**.
    * **Rule:** Avoid pointless tiny trims/flips in `GOLDBEES` so your gold lots age past 12 months whenever possible.

### 9.2 Your Exact Step-by-Step Routine (Takes 15 Minutes Every 2 Weeks)

1. **Step 1 — Sync Window (`6:30 PM – 11:00 PM IST` on NSE Trading Days):**
   * Open `http://127.0.0.1:8000/` and click **Sync & Retrain** strictly between **`6:30 PM and 11:00 PM IST`**.
   * *Why?* NSE publishes official daily delivery CSVs after `~6:00 PM IST`, and Yahoo Finance sometimes rolls back bars around `~4:00 AM IST`. Syncing in the evening guarantees 100% settled, clean data.
   * Verify the top-bar **Data Freshness** pill says **`HEALTHY`**.
2. **Step 2 — Check If a Sleeve Rebalance Is Due (Every 10 NSE Trading Days):**
   * Check your **4-Sleeve Schedule** in the Paper Ledger tab. Only trade on your scheduled sleeve rebalance date (~every 2 weeks).
   * *Live Paper Account Schedule (already running untouched since `2026-11-20` at `₹10,00,000`, currently `₹10,10,032.98`):*
     * **Sleeve 1:** Active (`2026-11-20`) → Next due **`2027-01-19`**
     * **Sleeve 2:** Next due **`2026-12-07`**
     * **Sleeve 3:** Next due **`2026-12-21`**
     * **Sleeve 4:** Next due **`2027-01-05`**
   * Ignore tiny 1–2 share noise trims between scheduled dates.
3. **Step 3 — Place Broker Orders Inside NSE's Sweet-Spot Windows (NEVER at 9:15 AM!):**
   * Place **Limit Orders** (near the current bid/ask — NEVER blind Market Orders!) strictly during:
     * **Morning Window:** **`10:00 AM – 10:30 AM IST`** *(after chaotic 9:15–9:45 AM opening-bell spreads calm down)*, OR
     * **Afternoon Window:** **`2:45 PM – 3:15 PM IST`** *(tightest institutional spreads; avoid 3:25–3:30 PM closing-print spikes)*.
   * **Order Sequence:** Execute **SELLs first** (frees up cash/margin), then **BUYs**, then park leftover reserve cash `70%` Cash Vehicle / `30%` `GOLDBEES`.
   * **Liquidity Cap (`≤ 2%` of 20D ADV):** Never place a single order larger than `2%` of a stock's 20-day average daily rupee volume (the Trade Ticket automatically flags any `adv_cap_warning`). If flagged at large capital, split the order across 2 sessions.
4. **Step 4 — Log & Back Up:**
   * Click **Commit Paper Rebalance** (when paper trading) and keep an offline backup copy of `data/equilens_india.db` (e.g., encrypted Google Drive / local backup folder — **NEVER commit `data/equilens_india.db` to GitHub!**).
5. **Step 5 — The Foolproof Real-Money Capital Ramp Plan:**
   * **Months 1–3 (Now — Started Nov 2026):** Run the `₹10L` live forward paper portfolio with 100% discipline. Prove zero bugs, zero missed rebalances, and emotional calm.
   * **Months 4–6:** Deploy **`₹15L – ₹25L`** of real money (`Top-10`). Verify live Zerodha/broker STT, spreads, and your real emotional reaction to red days.
   * **Months 7+:** Scale gradually to **`₹1 Crore`** in 3–4 monthly tranches (`Top-10` caps at `≤ ₹12L/stock`; if you want even smoother swings at `₹1 Cr+`, the built-in `Top-20` setting caps at `≤ ₹5L/stock`).

---

## 10. "What Do I Do When...?" — The Foolproof Crisis & Circumstance Handbook

Consult this table **whenever ANYTHING happens** — market crash, boredom, computer glitch, SEBI rule change, or life emergency.

| # | Circumstance / Scenario | What Is Actually Happening | Exact Action to Take (DO / DO NOT) |
| :-: | :--- | :--- | :--- |
| **1** | **Portfolio drops `-5%` from its peak** | Totally routine stock market noise (happens 2–3× every year). | **DO NOTHING.** Do not sell, do not change settings, do not check prices every hour. Wait for your scheduled 10-day rebalance. |
| **2** | **Portfolio drops `-10%` from its peak** | Normal 1-in-2-year correction; the **Drawdown Brake (`-10%`)** automatically trips ON. | **FOLLOW THE TRADE TICKET.** The engine will automatically cut equity exposure in half and park the freed cash in `70/30` Cash/`GOLDBEES`. **DO NOT** override the brake to "buy the dip" — and **DO NOT** panic-sell the remaining half! Wait for automatic release (recovers past `-6%` AND closes above 20D SMA `× 1.005`). |
| **3** | **Portfolio drops `-15%` to `-18%`** | Severe macro storm (like 2008 / 2020 / major war). Your shields are already working hard (NIFTY is likely down `-25%` to `-50%`). | **STAY CALM & FOLLOW THE SYSTEM.** Check that Risk Gate + Drawdown Brake are active. Do NOT turn off the system at the bottom — locking in losses at the trough is the #1 way investors ruin wealth. |
| **4** | **6–12 months of flat/boring returns while friends brag about +50% smallcaps** | Normal momentum drought / sideways chop (just like `2018` or `2025`). | **STAY THE COURSE.** Friends never brag when their tip drops `-60%`. Remember: `V7.5-Core` made `+9.7%` in choppy `2025` and `-1.4%` while NIFTY fell `-10.6%` in `2026`. Never tinker during a flat year! |
| **5** | **Shocking news overnight (war, election surprise, RBI shock, market opens `-6%`)** | Opening-bell panic with massive bid-ask spreads. | **DO NOT TRADE AT 9:15 AM!** Wait until market close (`6:30 PM IST`), run **Sync & Retrain**, let the 6-Gauge Risk Gate & Drawdown Brake evaluate the *settled* close, and execute any required de-risking with limit orders at **`10:00–10:30 AM`** or **`2:45–3:15 PM`** the next day. |
| **6** | **Risk Gate drops equity to `55%` or `25%`** | 2 or 3+ market storm gauges are flashing red simultaneously. | **OBEY IT FULLY.** Sell the trimmed equity portions as shown on the Trade Ticket and park the proceeds `70%` Arbitrage/`LIQUIDCASE` + `30%` `GOLDBEES`. Do NOT leave it uninvested in `0%` broker cash for months. |
| **7** | **A stock you own gets flagged `"EXIT — Low 20D NSE Delivery (<35%)"` or `"Balance-Sheet Risk"`** | Institutional buyers left and speculative intraday churn took over — OR the company's balance sheet developed `≥ 2` blowup red flags. | **SELL IT AT REBALANCE** exactly as the ticket instructs (even if it's still Rank #15!). This exit rule alone adds `~+2.5%/yr` and prevents holding blowups. |
| **8** | **A stock you own hits a lower circuit / gets put in T2T / suspended** | Extreme company-specific shock (rare in our 130 liquid F&O universe, but possible). | **DO NOT** place desperate pre-market orders. Place a limit sell at the daily lower-circuit floor at `9:15 AM` if the ticket says SELL, and wait for liquidity to return. Never average down on a collapsing stock. |
| **9** | **SEBI Delivery Drift Telemetry shows `warning` (`below_35_pct < 5.0%`)** | SEBI rule changes (like `T+0` settlement) pushed almost every stock above `35%` delivery, weakening the static `<35%` bouncer. | **DO NOT panic or change code on Day 1.** Watch it for **3 consecutive months**. If `below_35_pct` stays `< 5.0%` persistently, ask a future AI iteration to run the 5-Gate comparison of Static `<35%` vs. our pre-tested **`Hybrid (<35% OR Bottom-20%)`** fallback. |
| **10** | **Dashboard shows `Data: DEGRADED` or `STALE` when you try to rebalance** | Either you synced outside `6:30–11:00 PM IST`, Yahoo Finance glitched, or NSE had a holiday. | **DO NOT force `allow_stale=True`!** Wait 30 minutes and click **Sync & Retrain** again between `6:30–11:00 PM IST`. If Yahoo is temporarily down, simply delay the rebalance by 1 trading day — 1 day never matters over a 40-day cycle. |
| **11** | **You are on vacation / sick / miss a rebalance day** | Life happened! | **ZERO STRESS.** Just run **Sync & Retrain** and execute the Trade Ticket on the very next trading day you are back. Missing by 1–5 days has almost zero effect on a 4-tranche 40D system. |
| **12** | **You want to ADD fresh money (`+₹5L` or `+₹50L`)** | You received a bonus/savings and want to scale up. | **Small addition (`< 20%` of portfolio):** Park in Arbitrage/`LIQUIDCASE` until the next scheduled 10-day rebalance and let the ticket deploy it cleanly. **Large addition (`> 20%`):** Split across the next **4 scheduled 10-day rebalances** (1/4 per sleeve) so you never expose a lump sum to one lucky/unlucky day! |
| **13** | **You urgently need to WITHDRAW cash for a real-life need** | Medical, home, or family need. | **Pull in this exact order:** (1) Idle `LIQUIDCASE`/Arbitrage cash first → (2) `GOLDBEES` reserve second → (3) Only if still needed, trim all 10 stock positions **proportionally** (keeping weights intact) — preferring lots `≥ 365` days old or small-gain lots to minimize STCG tax. |
| **14** | **Your laptop dies, VM resets, or `http://127.0.0.1:8000` won't load** | Software/hardware hiccup — your real stocks sit safely in your CDSL/NSDL demat locker, completely unaffected! | Restart the server (`uvicorn backend.main:app --host 127.0.0.1 --port 8000`). On a fresh machine, `git clone` the repo, restore your offline backup of `data/equilens_india.db` (or re-enter your broker's current holdings in the Trade Ticket), and run **Sync & Retrain**. |
| **15** | **March 31 (Indian Financial Year End) & ITR Filing** | Tax season! Someone on YouTube tells you to "tax-loss harvest." | **IGNORE March TLH** (we proved it *loses* money on NSE after STT!). Simply download your broker's **Tax P&L Report** (Zerodha Console → Reports → Tax P&L), hand it to your CA, ensure all STCL/LTCL losses are carried forward, and claim your `₹1.25 Lakh` annual LTCG exemption. |
| **16** | **A friend, YouTuber, or new AI suggests "adding RSI/MACD/FII/Crypto/Deep Learning"** | Classic tinkerer's itch — the #1 destroyer of good quant systems. | **SAY NO BY DEFAULT.** Point them to **Section 7.2 (Graveyard)** and **Section 16 (5-Gate Promotion Standard)**. Nothing touches live code unless it passes all 5 Gates strictly OOS, beats 17 Fallen Angels, and wins a 9-cell plateau! |
| **17** | **WHEN SHOULD YOU ACTUALLY STOP RUNNING THIS SYSTEM? (Objective Kill-Switch)** | How to tell real structural decay from normal bad luck. | **Do NOT quit after 1 bad year.** Only pause and launch a full forensic review if **ANY** of these 3 objective alarms trip: **(a)** Cumulative live drawdown breaches **`-25%`** (more than 2× our backtested `-10.2%` MaxDD), **(b)** Live strategy trails `NIFTYBEES` TRI by **`> -10%` annualized over 3 full consecutive years (36 months)** with no obvious macro explanation, OR **(c)** Live OOS 40D Rank IC stays **negative (`< 0.0`) across 12+ consecutive non-overlapping 40D cohorts (~2 years)**. |

---

# PART II — ENGINEERING & QUANTITATIVE HANDOFF BIBLE (FOR FUTURE AI ITERATIONS)

> **Instructions to Future AI Assistants:** Read Part II completely **before** touching a single line of code, running any backtest, or proposing any research upgrade. All live numbers below are empirically verified directly from the codebase and SQLite store on commit `5d68654` (`main` synced with `feat/v7-book-research`).

---

## 11. Production System Identity, Repo State & Hard Guardrails

### 11.1 Live Production State Snapshot
* **Repository Path:** `/home/joe/.gemini/antigravity/scratch/equilens-india-quant`
* **Remote:** `https://github.com/Joestilllives/equilens-india-quant`
* **Live Branch:** `main` (synced with `feat/v7-book-research`; historical `V6` playbook at `7614ab0`; quarantined research branches: `feat/v75-frontier`, `feat/v8-frontier-research`, `v6-institutional-hardening`)
* **Strategy Identifiers:**
  * Live Backtest / Production Default: `strategy_id = "v75_core_delivery_shield"` (note: `config.py` constants retain `STRATEGY_ID_ML = "v7_book_hybrid_hrp"`, `STRATEGY_ID_EXECUTION = "v7_book_live_execution"`, `STRATEGY_ID_SHORTLIST = "unvalidated_research_composite"` — keep shortlist strictly separated from ML execution!)
* **Live Local Server:** `http://127.0.0.1:8000/` (FastAPI + static frontend; `uvicorn backend.main:app --host 127.0.0.1 --port 8000`)
* **Live SQLite Database (`data/equilens_india.db` — NEVER commit to git!):**
  * `tickers`: `133` rows (`130` NSE equities + `^NSEI` + `NIFTYBEES.NS` + `GOLDBEES.NS`)
  * `ohlcv`: `301,069` daily bars (`10y` history)
  * `nse_delivery`: **`220,229` rows across `1,700` consecutive NSE trading days (`2020-01-01` → `2026-11-26`)** seeded from `nsearchives.nseindia.com` (`sec_bhavdata_full_DDMMYYYY.csv`)
  * `ml_predictions`: `130` equity rows (`0` ETF leaks) | `ml_oos_history`: `161,586` rows | `ml_runs`: latest Run `#8` (`262,188` training samples, `130` symbols, `40D` horizon)
  * `pit_fundamentals_history`: `130` immutable Day-1 snapshots (auto-appended on every `sync_universe`)
  * **Live Forward Paper Ledger (`paper_*` — STRICTLY UNTOUCHED):**
    * Started `2026-11-20` with `₹10,00,000` (`balanced`, `top_k=10`); NAV `₹10,10,032.98` (`+1.00%` vs `NIFTYBEES.NS` TRI `+0.38%`), `cash_inr = ₹57,917.70`, `10` open positions (`ADANIPORTS.NS`, `AUROPHARMA.NS`, `BHEL.NS`, `BOSCHLTD.NS`, `DIVISLAB.NS`, `FEDERALBNK.NS`, `MARICO.NS`, `MCX.NS`, `MOTHERSON.NS`, `TECHM.NS`), `10` FIFO lots (`paper_lots`), `4` staggered sleeves (`paper_sleeves`: Sleeve 1 active `2026-11-20 → 2027-01-19`, Sleeve 2 next `2026-12-07`, Sleeve 3 next `2026-12-21`, Sleeve 4 next `2027-01-05`).

### 11.2 Absolute Engineering Do-Not-Break Rules
1. **NEVER modify, reset, or migrate-destroy `paper_account`, `paper_positions`, `paper_trades`, `paper_lots`, `paper_sleeves`, or `paper_nav_history`** during any code or DB maintenance. Always back up `data/equilens_india.db` to an outside scratch path before any schema or ingestion work.
2. **NEVER commit `data/equilens_india.db`** (or cache DBs like `data/multicycle_2004_2026_cache.db`) to git.
3. **NEVER run unconstrained deep ML or wide hyperparameter grids on `N ≈ 130` stocks.** Small-panel financial ML overfits fast; preserve the `50%` zero-parameter Prior + strongly regularized `Ridge(alpha=500)` + shallow `HGB(max_depth=2)` architecture.
4. **NEVER re-test Graveyard items (Section 7.2)** unless a genuinely new, strictly Point-In-Time data source exists (e.g., 5+ years of accumulated `pit_fundamentals_history` or official BSE/NSE XBRL archive).
5. **NEVER let visual UI composites (`shortlister.py`) bleed into execution (`execution_engine.py`).** Shortlist is explicitly tagged `unvalidated_research_composite`.

---

## 12. Exact Mathematical & Implementation Specification (`V7.5-Core`)

### 12.1 Universe & Non-Equity Hygiene (`backend/config.py`)
* `BENCHMARK_SYMBOL = "^NSEI"`
* `TRI_BENCHMARK_SYMBOL = "NIFTYBEES.NS"` (dividend-adjusted TRI proxy; fallback to `^NSEI` only if `< 120` bars)
* `GOLDBEES_SYMBOL = "GOLDBEES.NS"` (real Indian Gold ETF daily returns for the `30%` defensive reserve; causal `GC=F * INR=X` proxy fallback)
* `NON_EQUITY_SYMBOLS = frozenset({"^NSEI", "NIFTYBEES.NS", "GOLDBEES.NS"})` — **MUST** be excluded from ML training, ML inference, cross-sectional z-scoring, NSE breadth, 52W Hi-Lo sentiment, Pollet–Wilson correlation, execution ranking, and shortlist defaults.
* `HISTORY_PERIOD = "10y"`, `ML_HORIZON_DAYS = 40`, `TB_HORIZON_DAYS = 40`, `TB_UP_ATR_MULT = 2.0`, `TB_DOWN_ATR_MULT = 2.0`, `DEFAULT_COST_BPS = 15.0`.

### 12.2 The 19 Active Features (`FEATURE_COLS` in `backend/ml_predictor.py`)
Daily cross-sectional z-scoring (`ddof=0`, clipped to `[-3.0, +3.0]`, missing filled `0.0`) is applied across clean equities only:

* **Canonical Prior + ML Features (`6`):**
  1. `z_resid_12m1m`: `sum(eps_1d.shift(21), 231d) / idio_vol_252`, where `eps_1d = ret_1d - beta_vasicek.shift(1) * bench_ret_1d` (strictly causal lagged beta)
  2. `z_resid_6m1m`: `sum(eps_1d.shift(21), 105d) / idio_vol_126`
  3. `z_mom_12_7`: `(close.shift(126) / close.shift(252) - 1) / vol_safe` (`vol_safe = clip(vol_252d, min=0.12)`)
  4. `z_cgo_126`: `clip((close - vwap_126) / close, -1.5, 0.9)` (`vwap_126 = sum(close*vol, 126) / sum(vol, 126)`)
  5. `z_neg_beta`: `-beta_vasicek`, where `beta_vasicek = clip(0.60 * beta_raw_252d + 0.40 * 1.0, 0.10, 2.50)`
  6. `z_vol_252d`: `std(log_ret, 252) * sqrt(252)` (enters prior with negative sign)
* **ML-Only Cross-Sectional Features (`10`):**
  7. `z_mom_6m1m_voladj`: `(close.shift(21) / close.shift(126) - 1) / vol_safe`
  8. `z_mom_12m1m_voladj`: `(close.shift(21) / close.shift(252) - 1) / vol_safe`
  9. `z_fip_smooth_6m`: `mean(log_ret > 0, 126) - mean(log_ret < 0, 126)`
  10. `z_down_vol_126d`: `std(clip(log_ret, upper=0), 126) * sqrt(252)`
  11. `z_sec_rel_63d`: `ret_63d - sector_median(ret_63d)`
  12. `z_ret_21d`: trailing 21D return
  13. `z_rsi14`: 14-day RSI
  14. `z_macro_tailwind_sp500`: `clip(cov(ret_1d, sp500_ret_1d, 126) / var(sp500_ret_1d, 126), -1.5, 2.0) * sp500_63d_chg` *(strictly `macro_date < nse_date` via `merge_asof(allow_exact_matches=False)`!)*
  15. `z_max_ret_21d`: `max(ret_1d, 21)`
  16. `z_overnight_sent_63d`: `sum(clip(open / close.shift(1) - 1, -0.12, 0.12), 63)`
* **Date-Constant Macro Features (`3` — `_DATE_MACRO_COLS`, fed ONLY to `HistGradientBoostingRegressor`, EXCLUDED from `Ridge`, `LogisticRegression`, and binary `HistGradientBoostingClassifier` via `CS_FEATURE_INDICES`):**
  17. `vix_level`: India VIX (`^INDIAVIX`, closes 3:30 PM IST with NSE — same-day causal)
  18. `bench_dist_200`: `^NSEI / SMA200(^NSEI) - 1`
  19. `nse_breadth_50d`: fraction of clean NSE equities with `close > SMA50`

### 12.3 Canonical Prior, Daniel–Moskowitz Damper & Walk-Forward ML Ensemble (`backend/ml_predictor.py`)
* **Base Canonical Prior (`prior_variant="book"`):**
  ```python
  base_prior = (
      0.25 * z_resid_12m1m
      + 0.25 * z_mom_12_7
      + 0.15 * z_resid_6m1m
      + 0.15 * z_cgo_126
      + 0.10 * z_neg_beta
      - 0.10 * z_vol_252d
  )
  ```
* **Daniel–Moskowitz (2016) Bear-Rebound Damper:**
  * Active when `bench_ret_252d < 0.0` AND `vix_level >= 19.0`.
  * Blends `0.50 * base_prior + 0.50 * (0.40 * z_neg_beta - 0.35 * z_vol_252d + 0.25 * z_resid_12m1m)`.
* **ML Target (`40D` horizon):**
  * `target_resid_sharpe = clip((fwd_ret - beta_vasicek * bench_fwd_ret) / clip(vol_safe, 0.12), -3.0, 3.0)`
  * Daily cross-sectional percentile rank: `fwd_rank = groupby("date")["target_resid_sharpe"].rank(pct=True)`
* **Walk-Forward Protocol:**
  * `5` expanding chronological folds (`first_test_start = int(n_dates * 0.42)`), strict **`embargo = max(horizon, tb_horizon) = 40` trading days** between `train_cutoff` and `test_start_dt`.
* **Ensemble Models (`_fit_r4_ensemble`):**
  * **Ranker (`pred_rank = 0.60 * Ridge + 0.40 * HGB`):**
    * `Ridge(alpha=500.0)` inside `StandardScaler` pipeline on `CS_FEATURE_INDICES` (`16` features)
    * `HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=65, min_samples_leaf=100, l2_regularization=10.0, random_state=42)` on all `19` features
  * **Payoff Classifiers (`prob_beat` & `prob_tb_win` = `0.65 * Logistic + 0.35 * HGBClassifier`):**
    * `LogisticRegression(C=0.05, max_iter=300)` + `HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=55, min_samples_leaf=120, l2_regularization=15.0, random_state=42)` strictly on `CS_FEATURE_INDICES` (`16` features — prevents VIX time-series inversion!)
* **Final Hybrid Score:**
  * `r4_score = 0.50 * nse_factor_prior + 0.50 * cs_zscore(pred_rank)`
  * `pred_alpha_pct = clip(r4_score * 2.5, -12.0, 12.0)`

### 12.4 Portfolio Construction, Shields, Risk Defense & Accounting
* **Selection & Hysteresis (`execution_engine.py` & `portfolio_backtester.py`):**
  * Default `top_k = 10`, `rank_buffer = min(top_k * 3, 40) = 30`, `max_per_sector = 2`, `rebalance_days = 40`, `n_tranches = 4` (offsets `0, 10, 20, 30`).
  * **20D NSE Delivery & Balance-Sheet Shield (Retention + Entry Parity):** Blocks/exits any stock if ANY of:
    1. Strictly `T-1` `deliv_20d_pct < 35.0`
    2. `deliv_20d_pct >= 98.0` (T2T / ASM surveillance trap)
    3. `0 < deliv_turnover_20d_cr < 2.0` (`₹2 Cr/day` deliverable turnover floor)
    4. `shield_status == "Risk"` (`≥ 2` flags: non-financial `D/E > 180%`, `ROE < 6%`, non-financial `FCF Yield < -4%`, `Revenue Growth < -10%`).
  * **Indian `320–364d` LTCG Tax-Cliff Guard (`ltcg_cliff_guard=True`):** Retains profitable existing lots aged `320 <= days < 365` if rank `<= max(rank_buffer, 60)` AND passing all tradability/delivery/T2T/turnover/balance-sheet shields.
* **Weighting (`backend/hrp_allocator.py`):**
  * López de Prado (2016) `HRP` (`single` linkage, quasi-diagonalization, recursive bisection inverse-variance) on **126D Ledoit–Wolf shrunk covariance** (`LedoitWolf(assume_centered=False)`).
  * Iterative hard concentration caps (`apply_concentration_caps`): `max_stock_w = 0.05` if `top_k >= 20` else `min(0.15, 1.20 / top_k)` (`= 0.12` at `K=10`); `max_sector_w = 0.25 if top_k < 10 else 0.20`.
* **3-Layer Crash Defense:**
  1. **6-Gauge Risk Gate:** `+1` pt each for `vix >= 19.0`, `vix >= 23.0`, `^NSEI < SMA200`, `breadth_50d < 42.0%`, `hilo_sentiment < -10.0%`, `gold_nifty_63d > +14.0%`. Target equity: `0 pts → 100%`, `1 pt → 85%`, `2 pts → 55%`, `3+ pts → 25%`. One-way intra-cycle de-risking in backtester.
  2. **18% Ex-Ante Vol Target:** `vol_cap = clip(0.18 / max(0.08, ex_ante_vol_60d), 0.25, 1.0)` using **60D sample covariance** (do NOT replace with `sklearn.LedoitWolf`!).
  3. **`-10% / -6%` Peak-NAV Drawdown Brake:** Triggers at `dd <= -0.10` (cuts equity exposure `× 0.50`); releases only when `dd > -0.06` AND `nav > SMA20(nav) * 1.005`.
* **Defensive Reserve (`70%` Cash / `30%` `GOLDBEES`) & Statutory/Tax Engine (`backend/indian_costs.py`):**
  * Cash benchmark yield: `INDIA_CASH_ANNUAL_YIELD = 0.062`.
  * Statutory NSE delivery charges: `0.10%` STT both sides, `0.015%` buy stamp duty, `0.00297%` NSE fee, `0.0001%` SEBI fee, `18%` GST, `₹15.93` sell DP charge, `3.0 bps` half-spread, `0.25 * daily_vol * sqrt(order / ADV_20d)` impact.
  * FIFO tax (`FifoTaxLedger`): `20.8%` STCG (`< 365d`), `13.0%` LTCG (`≥ 365d`, `₹1,25,000`/FY exemption), STCL/LTCL set-off & carryforward.
  * Post-Budget 2024 Reserve Tax: `31.2%` slab rate on `LIQUIDCASE` & `<365d GOLDBEES` gains vs. `20.8%` STCG (`13.0%` LTCG) on Direct Arbitrage Fund — both reported (`after_tax_cagr_liquidcase_pct`, `after_tax_cagr_arbitrage_pct`).
* **SEBI Delivery Drift Telemetry (`backend/money_signals.py`):**
  * Emits `delivery_regime_telemetry` on `/api/money-signals` and `/api/execution/ticket`; triggers `drift_status = "warning"` when `universe_count >= 30` and `below_35_pct < 5.0%`.

---

## 13. Complete Codebase, Database & API Architecture Map

### 13.1 Backend Modules (`backend/`)
| Module | Role | Status |
| :--- | :--- | :--- |
| `config.py` | Universe (`DEFAULT_UNIVERSE`), `NON_EQUITY_SYMBOLS`, CORS/auth/freshness env vars, `40D` horizon, `15 bps` default | **CORE LIVE** |
| `database.py` | Thread-safe WAL SQLite store (26 tables), OHLCV, macro, NSE delivery, ML, PIT fundamentals, paper ledger | **CORE LIVE** |
| `data_fetcher.py` | Yahoo Finance OHLCV (`10y`), `NIFTYBEES.NS`, `GOLDBEES.NS`, macro series, fundamentals + Day-1 PIT snapshot logger | **CORE LIVE** |
| `data_health.py` | `build_clean_price_matrix(max_ffill_days=2)`, `tradable_df`, `evaluate_data_freshness` (60s TTL fail-safe) | **CORE LIVE** |
| `nse_calendar.py` | NSE equity holiday calendar (`2020–2027`), `add_nse_trading_days` for staggered sleeves & probability cones | **CORE LIVE** |
| `indicators.py` | Technical indicator frame (`SMA50/200`, `ATR14`, `RSI14`, `52W Hi/Lo`, `ret_63d/126d`) & chart setups | **CORE LIVE** |
| `ml_predictor.py` | 19-feature causal panel, 6-factor Canonical Prior, DM Crash Damper, 5-fold expanding Ridge+HGB, 6-Gauge Macro Regime, Pollet–Wilson `avg_corr_63` | **CORE LIVE** |
| `hrp_allocator.py` | López de Prado `HRP` (`126D Ledoit–Wolf`), `apply_concentration_caps`, budget-normalized L1 no-trade band | **CORE LIVE** |
| `money_signals.py` | Official NSE delivery (`sec_bhavdata_full_*.csv`), FII/DII OI, bulk/block deals, Balance-Sheet Shield, SEBI Delivery Drift Telemetry | **CORE LIVE** |
| `indian_costs.py` | Statutory NSE delivery fees, square-root ADV impact, `FifoTaxLedger`, Post-Budget 2024 reserve tax constants | **CORE LIVE** |
| `portfolio_backtester.py` | 4-tranche OOS simulator, `build_backtest_market_context()`, strictly `T-1` 20D delivery lookup, LTCG cliff guard, 2F + IIMA 4F alpha, reserve tax accounting | **CORE LIVE** |
| `execution_engine.py` | Live rebalance ticket generator, 4-sleeve schedule, peak-NAV `-10%/-6%` brake, `2%` ADV cap guard, paper ledger commit/M2M | **CORE LIVE** |
| `stress_tester.py` | Multi-friction stress suite (reusing `market_ctx`), regime replay, tail risk & factor attribution | **CORE LIVE** |
| `main.py` | FastAPI server, `X-EquiLens-Token` / Bearer auth on all `POST` routes, CORS, single-flight caching, static frontend host | **CORE LIVE** |
| `shortlister.py`, `backtester.py`, `journal.py`, `alerts.py`, `catalyst_nlp.py`, `sec_edgar.py` | UI research shortlist (`unvalidated_research_composite`), single-stock next-open chart backtester, prediction journal, alerts, news NLP | **UI / AUXILIARY** |
| `pit_fundamentals.py` | Day-1 `pit_fundamentals_history` logger + NSE Bhavcopy & BSE/NSE XBRL `+2d` parsers | **PHASE-2 PIPELINE (LOGGING)** |
| `india_structural_signals.py` | Causal IIMA 4-Factor (`MKT/SMB/HML/WML`) alpha engine (live in reporting) + SEBI shareholding `+2d` parser | **REPORTING + FUTURE** |
| `forensic_quality_sleeve.py`, `multi_sleeve_allocator.py`, `convex_optimizer.py`, `india_liquidity_regime.py` | Parked `V8` toolkit: Indian Forensic/QMJ+Value, Carver FDM combiner, Boyd convex optimizer, RBI DBIE liquidity telemetry | **PARKED OPT-IN TOOLKIT** |

### 13.2 Key API Routes (`backend/main.py`)
* **Read (`GET`, no auth):** `/api/health`, `/api/shortlist`, `/api/stock/{symbol}`, `/api/model/metrics`, `/api/backtest/{symbol}`, `/api/portfolio/stress`, `/api/execution/ticket`, `/api/execution/paper/status`, `/api/money-signals`, `/api/journal`, `/api/alerts`.
* **Mutating (`POST`, requires `X-EquiLens-Token` or `Authorization: Bearer <SESSION_API_TOKEN>`, auto-injected into `<meta name="equilens-token">` on `/`):** `/api/sync`, `/api/tickers/add`, `/api/tickers/{symbol}/star`, `/api/execution/paper/commit`, `/api/execution/paper/reset` (also requires `{"confirm": true}`), `/api/money-signals/sync`, `/api/journal/log`, `/api/alerts/scan`, `/api/alerts/{alert_id}/ack`.

---

## 14. The 7 Non-Negotiable Anti-Leakage & Safety Invariants

Every future code change MUST preserve these 7 invariants (all verified by automated `pytest` tests):
1. **Strict Cross-Timezone Macro Causality:** US/Global series (`^GSPC`, `GC=F`, `BZ=F`, `INR=X`) MUST merge with `allow_exact_matches=False` (`macro_date < nse_date`) so 2:30 AM IST US closes never leak into 3:30 PM IST NSE bars. Only India-close `^INDIAVIX` merges same-day.
2. **Zero `.bfill()` Anywhere on Returns or Macro/Gold Series:** Use `ffill` or causal `merge_asof(..., direction="backward")` only.
3. **Strict `T-1` NSE Delivery Causality:** Both backtester (`bisect_left(dts, dt) - 1`) and live execution use strictly prior settled NSE delivery prints; never same-day delivery.
4. **40-Day Walk-Forward Embargo:** `train_end_idx = test_start_idx - max(horizon, tb_horizon)` (`40` trading days).
5. **Strict `NON_EQUITY_SYMBOLS` Filtering:** `^NSEI`, `NIFTYBEES.NS`, and `GOLDBEES.NS` never enter ranking, ML panels, z-scores, or breadth.
6. **Bounded Missing-Bar Hygiene (`max_ffill_days=2`):** Delisted/suspended/pre-IPO tickers become `tradable=False` after 2 missing bars instead of silently flat-lining forever.
7. **Single-Source Indian Cost & FIFO Tax Parity:** Backtester and live execution share `indian_costs.py` (`compute_indian_delivery_cost`, `FifoTaxLedger`) and `hrp_allocator.py` (`compute_hrp_weights`, `apply_concentration_caps`).

---

## 15. The 7-Point Red-Team Blind-Spot Audit Record

An adversarial audit identified 7 blind spots in `V7.5-Core`: **4 were fixed in code/data; 3 are structural trade-offs accepted with eyes open:**

1. **[FIXED] T2T / ASM `100%` Delivery Surveillance Trap:** Added `deliv_20d_pct < 98.0` exclusion + `deliv_turnover_20d_cr >= 2.0` (`₹2 Cr/day` deliverable turnover floor). *(Confirmed `0.00%` T2T days in current liquid 130-stock universe; future-proofs any down-cap expansion.)*
2. **[FIXED] Missing Indian LTCG Tax-Cliff Awareness:** Added `320–364d` profitable Top-60 holding guard (`+0.53%/yr` Honest PT, lower STCG, lower MaxDD).
3. **[FIXED] Post-Budget 2024 Reserve Tax Blind Spot:** Updated `indian_costs.py` & `portfolio_backtester.py` to track exact daily `liquid_gain_inr` and `gold_gain_inr`, taxing `LIQUIDCASE` & `<12m GOLDBEES` at `31.2%` slab rate and reporting Direct Arbitrage Fund (`20.8%` STCG / `13.0%` LTCG) side-by-side.
4. **[FIXED] Unmonitored SEBI Delivery Regime Drift:** Wired `delivery_regime_telemetry` (`below_35_pct < 5.0%` warning; Nov 2026 live reading `5.3%` vs. `51.4%` in 2020) onto `/api/money-signals` and `/api/execution/ticket`, and pre-tested the `Hybrid (<35% OR Bottom-20%)` fallback.
5. **[ACCEPTED TRADE-OFF] Drawdown Brake Lag on V-Shaped Bottoms:** Accepted as essential multi-year bear-market insurance (`2008` `-17%` vs NIFTY `-60%`).
6. **[ACCEPTED TRADE-OFF] 4-Tranche Staggering Tax-Lot Fragmentation:** Trades `~1.05%/yr` PT and `4×` FIFO lots vs. 1-tranche in exchange for `-1.58%` shallower MaxDD and zero start-date timing luck (partially offset by the LTCG cliff guard).
7. **[ACCEPTED TRADE-OFF] Gold Bull-Market Tailwind (`2021–2026`):** INR gold added `~+1.2–1.5%/yr` during `2021–2026`. Do not extrapolate this gold tailwind forever; budget live expectations at `16%–19%` net post-tax CAGR.

---

## 16. Standard Operating Commands & The 5-Gate Promotion Standard

### 16.1 Exact CLI Commands (Local or Fresh VM)
```bash
# 1. Navigate to repository & activate virtualenv
cd /home/joe/.gemini/antigravity/scratch/equilens-india-quant
source .venv/bin/activate

# 2. Run the full automated regression & integrity test suite
pytest -v

# 3. Start the live EquiLens India V7.5-Core server
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 4. Run full overfitting / DSR / permutation / plateau sign-off audit (read-only)
python scripts/audit_overfitting_v6.py --weighting hrp --perms 50

# 5. Run 17 Real NSE Fallen Angels survivorship stress test
python scripts/test_survivorship_fallen_angels.py

# 6. Run multi-cycle (8.8Y + 19Y) validation harness
python scripts/validate_v7_multicycle.py
```

### 16.2 The 5-Gate Promotion Standard (Mandatory Before ANY Future Model Upgrade)
No future AI iteration may promote a strategy change onto `main` unless it passes **ALL FIVE GATES** on an isolated git branch/worktree (leaving live `paper_*` tables untouched):

1. **Gate 1 — Hardened Net Core OOS (`2021–2026`, `NIFTYBEES` TRI, Statutory Costs, FIFO Equity + Post-Budget 2024 Reserve Taxes):** Must beat `V7.5-Core` (`+29.97%` CAGR, `+25.67%` Honest PT `LIQUIDCASE`, `SR(6.2%) 1.84`, `MaxDD -10.23%`, `4F Alpha +10.96%`) on Honest Post-Tax CAGR, Excess Sharpe, AND 4-Factor Alpha without blowing up MaxDD or turnover.
2. **Gate 2 — Multi-Cycle & Survivorship Stress:** Must hold up on **17 Real NSE Fallen Angels** AND **blind unseen Indian stocks** AND multi-cycle (`2007–2026` / `2017–2026`) tests.
3. **Gate 3 — Multiple-Testing Rigor:** Must clear corrected Bailey–López de Prado **DSR (`RF = 6.2%`, full cumulative `N` trials)** ≥ `95%` and keep CPCV **PBO** low (`< 30%`).
4. **Gate 4 — Parameter Plateau & Cost Stress:** Must win **at least `6/9`** cells across the `(K ∈ {8, 10, 12} × Rebalance ∈ {30D, 40D, 60D})` grid (including `3/3` or `2/3` at `40D`) AND survive `50 bps/side` (`100 bps` RT) cost stress.
5. **Gate 5 — Zero-Leakage, Live Parity & Full Suite:** `100%` of `pytest` tests passing, zero ETF leaks, strict backtest↔live parity, zero modification to `paper_*` tables, and `PLAYBOOK.md` updated cleanly.

---
*End of EquiLens India `V7.5-Core` Master Playbook. Stay disciplined, trust the process during flat months, never skip the evening sync window, and let compounding do the heavy lifting.*
