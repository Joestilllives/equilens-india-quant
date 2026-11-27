# EquiLens India Quant — The Complete `V7.5-Core` Master Playbook

> **Active Production Engine:** `EquiLens India V7.5-Core` (`strategy_id: v75_core_delivery_shield`)  
> **Repository & Branches:** `github.com/Joestilllives/equilens-india-quant` — live on `main` (synced with `feat/v7-book-research`; historical `V6` playbook preserved in git history at commit `7614ab0`)  
> **Live Status:** **STRICTLY FROZEN FOR FORWARD PAPER TRACKING** — Do NOT tweak parameters, add indicators, or touch the live `paper_*` ledger without passing all 5 Promotion Gates in Part II.  
> **Disclaimer:** *Quantitative research & personal forward paper-trading guide. Not SEBI-registered investment advice.*

---

## How to Use This Guidebook

This document is written for **two very different readers** — and you should always know which half you are in:

1. **Part I (Sections 1–10) — For Joe (Kitchen-Table Plain English):** Written assuming **you have never bought a single stock in your life**. Every financial word is translated into everyday language with real Indian Rupee (`₹10 Lakh` and `₹1 Crore`) examples. Consult Part I whenever you want to understand *how* the engine works, *why* it owns or sells a company, *what* to expect emotionally, or *what to do* in ANY real-life or market circumstance.
2. **Part II (Sections 11–16) — For the Next AI Iteration (Engineering & Quant Bible):** Exact file paths, formulas, hyperparameters, SQLite schemas, backtest-vs-live parity nuances, API routes/auth, anti-leakage invariants, safe CLI commands, and mandatory 5-Gate governance rules so a future AI can step in months or years later without breaking a single rule or re-testing dead ends.

---

# PART I — THE KITCHEN-TABLE GUIDEBOOK (ZERO STOCK-MARKET BACKGROUND REQUIRED)

---

## 1. Start Here: How Stock Investing Works in India & The Zero-Jargon Dictionary

### 1.1 The 2-Minute Crash Course: How Buying Shares Actually Works in India
If you have never bought a stock before, here is the entire physical plumbing in 5 sentences:

1. **Three Linked Accounts:** You keep money in your **Bank Account** (e.g., HDFC/ICICI/SBI), place buy/sell orders through a **Broker App** (e.g., Zerodha Kite, Groww, Dhan), and the actual shares are stored inside your government-regulated **Demat Locker** (**CDSL** or **NSDL** — think of it like a digital safe-deposit locker registered to your PAN).
2. **Your Shares Are Safe Even If Your Laptop or Broker Dies:** Your broker is only the middleman that sends orders to the exchange. The shares themselves sit safely in CDSL/NSDL under your own name and PAN. Even if your laptop breaks or the EquiLens app stops working, your shares remain 100% intact.
3. **Market Order vs. Limit Order (CRITICAL!):**
   * A **Market Order** says *"Buy right now at whatever price the seller demands!"* — dangerous if prices jump for 2 seconds.
   * A **Limit Order** says *"Buy ONLY at ₹917 or better"* (or *"Sell ONLY at ₹917 or better"*). **We ALWAYS use Limit Orders** near the current screen price so we never get overcharged.
4. **Bid–Ask Spread & Slippage:** At any second, buyers might offer `₹916.80` (bid) while sellers ask `₹917.20` (ask). That tiny `₹0.40` gap is the **spread**. If you trade calmly with Limit Orders during quiet hours, you lose only pennies (`~0.03%`). If you panic-trade at the `9:15 AM` opening bell, spreads blow out and you lose far more (**slippage**).
5. **`T+1` Settlement:** When you buy or sell a share on NSE on Monday (`T`), the shares/cash officially settle in your demat/bank account on Tuesday (`T+1`, the next business day).

### 1.2 The 20-Term Zero-Jargon Dictionary
Every market term used anywhere in Part I is translated right here:

| Market Word | Kitchen-Table Meaning |
| :--- | :--- |
| **Stock / Share** | A tiny slice of ownership in a real Indian company (like HDFC Bank, Reliance, or Marico). When the business grows profits, your slice becomes more valuable. |
| **NSE & NIFTY 50** | **NSE** (National Stock Exchange) is India's main stock marketplace, open Mon–Fri `9:15 AM – 3:30 PM IST` (closed on NSE holidays). **NIFTY 50** is the average scorecard of India's 50 biggest companies — the yardstick we measure ourselves against. |
| **NIFTYBEES (TRI)** | **TRI** (*Total Return Index*) means NIFTY 50 **plus** the cash dividends companies pay shareholders. Measuring against NIFTY without dividends would be cheating, so we compare against `NIFTYBEES.NS`, which includes dividends. |
| **ETF** | *Exchange-Traded Fund* — a ready-made basket you buy like a single stock. We use **`GOLDBEES`** (holds real physical gold in India) and **`LIQUIDCASE`** (holds ultra-safe overnight government/bank loans paying ~`6.2%/yr`). |
| **CAGR** | *Compound Annual Growth Rate* — your average yearly wealth-building speed. At `15%` CAGR, `₹10 Lakh` doubles roughly every 5 years (`₹10L → ₹20L → ₹40L`). At `25%` CAGR, it triples in 5 years (`₹10L → ₹30.5L`). |
| **Volatility** | How bumpy the ride is. A fixed deposit has `0%` bumpiness (smooth, slow). Stocks wiggle up and down every day (`12%–25%` yearly bumpiness). |
| **Beta (`β`)** | How jumpy a stock is compared to the whole Indian market. `Beta = 1.0` moves in lockstep with NIFTY. `Beta = 1.6` zooms up `1.6×` on good days and crashes `1.6×` on bad days. `Beta = 0.6` is a calm cruiser. |
| **Sharpe Ratio (`SR 6.2%`)** | **Bang-for-your-bumpiness** *above* what a safe Indian liquid fund pays (`6.2%/yr`). Below `0.5` is poor; `0.8–1.0` is solid; `1.5+` is world-class; **`V7.5-Core` achieved `1.84`**. |
| **Max Drawdown (`MaxDD`)** | The worst peak-to-bottom drop you had to stomach before recovering. Falling from `₹10,00,000` to `₹8,97,700` is a `-10.23%` drawdown. NIFTY 50 fell `-59.9%` in 2008 and `-16.1%` (TRI) in 2021–2026. |
| **Alpha (`4-Factor Alpha`)** | Pure skill *after* stripping out luck. If NIFTY rises `+20%` and you make `+20%`, your alpha is `0%`. **`+10.96%` 4-Factor Alpha** means our engine made nearly `11%/yr` of real stock-picking edge that cannot be explained away by riding the market, small companies, cheap stocks, or basic momentum. |
| **Moving Average (`50-DMA` / `200-DMA`)** | Simply the average closing price over the last 50 days (~2.5 months) or 200 days (~10 months). Trading *above* the 200-DMA means the big trend is healthy; *below* means storm clouds. |
| **Basis Points (`bps`)** | Tiny percentage slices used by bankers: **`100 bps = 1.00%`**, so **`15 bps = 0.15%`** (`₹150` on a `₹1 Lakh` trade). |
| **Rebalance & Turnover** | Every **10 trading days** (~2 weeks), we check one quarter (`25%`) of the portfolio, selling stocks that faded and buying fresh leaders. **Turnover** is how much of the portfolio gets replaced per year. |
| **Delivery % (NSE)** | Day-traders buy at 10 AM and sell at 2 PM without ever taking shares home (like flipping lottery slips). **Delivery %** is the share of trades where buyers paid full cash and actually took shares into their demat locker. High delivery (`≥ 35%`) = serious investors; low delivery (`< 35%`) = casino day-trading churn. |
| **F&O Stocks, Circuits & T2T/ASM** | **F&O stocks** are India's largest, most heavily traded companies; they have flexible price limits so you almost never get locked out. Tiny stocks can hit a **lower circuit** (price freezes at `-5%` with zero buyers!). **T2T / ASM** is NSE's "naughty list" where day-trading is banned on shady stocks, artificially making their delivery read `100%`. |
| **Out-of-Sample (`OOS`)** | Testing a strategy strictly on **future years it was NEVER allowed to see while learning** — like grading a student on an unseen final exam instead of letting them memorize the answer key. |
| **Point-In-Time (`PIT`)** | Recording financial numbers *exactly as they were publicly known on that historical date* — never peeking at revised/restated reports published years later. |
| **Rank IC & Statistical Significance (`t-stat` / `p-value`)** | **Rank IC** measures whether our `#1–#130` rankings actually lined up with which stocks won over the next 2 months (`+0.03` to `+0.04` is strong in quant finance; below `0.00` means rankings stopped working). **`t-stat > 2.0`** (or **`p < 0.05`**) means the result is very unlikely to be a coin-flip fluke. |
| **STCG / LTCG & FIFO** | Indian capital gains tax (Finance Act 2024, incl. `4%` cess). Sell shares in **`< 365` days** = **STCG (`20.8%` of profit)**. Hold **`≥ 365` days** = **LTCG (`13.0%` of profit**, with your first **`₹1,25,000`/year tax-free!**). **FIFO** (*First-In, First-Out*) means Indian tax law always treats your oldest shares as sold first. |
| **Direct Arbitrage Fund** | A low-risk Indian mutual fund (`Direct Plan` = zero distributor commission) that earns `~6.5%–7%/yr` like a liquid fund, *but* legally counts as equity (`>65%` hedged shares) so you pay `20.8%` STCG / `13.0%` LTCG instead of `31.2%` slab tax! |

---

## 2. The Big Picture: What `V7.5-Core` Actually Does (The 60-Second Story)

Imagine picking a winning **10-player cricket squad** from India's **130 biggest, most actively traded companies** — while keeping a storm roof ready to slide shut the moment dark clouds roll in.

Here is the exact 5-step assembly line `EquiLens India V7.5-Core` runs automatically:

1. **Step 1 — Score All 130 Companies (Half Timeless Rules, Half Cautious AI):**
   Every evening, the engine ranks all 130 NSE companies from `#1` (strongest) to `#130` (weakest):
   - **50% of the score** comes from **6 timeless, textbook financial rules** discovered by leading finance professors over the last 30 years (zero machine-learning guesswork).
   - **50% of the score** comes from **two heavily restrained Machine Learning models** (a straight-line scorecard + ultra-shallow 2-step decision trees) looking at **19 clean signals** to estimate which stocks will quietly outperform over the next **40 trading days (~2 months)**.
2. **Step 2 — Run the Bouncers at the Door (NSE Delivery Shield + Balance-Sheet Shield):**
   Even if a company scores `#1`, it gets **blocked on new buys AND kicked out at rebalance if we already own it** whenever:
   - Its 20-day official NSE **Delivery % drops below `35%`** (serious buyers vanished; day-traders took over the casino), OR
   - Its 20-day Delivery % is **`≥ 98%`** (NSE's T2T/ASM surveillance "naughty list" trap), OR
   - Less than **`₹2 Crore/day`** of real delivery money changes hands (too thinly traded), OR
   - *(Live execution safety check)* Its financial health flashes **`≥ 2` balance-sheet red flags** (crushing debt, burning cash, weak profitability, or shrinking sales).
3. **Step 3 — Pick the Squad & Save Indian Taxes (Hysteresis + Tax-Cliff Guard):**
   - We buy the **Top 10** companies passing the bouncers (never more than **2 stocks from the same industry**, such as Banking or IT).
   - Once we own a winner, we **don't panic-sell just because it slips to `#11`**. We keep holding it as long as it stays inside the **Top 30** — cutting unnecessary brokerage and STCG tax bills.
   - **Smart Indian Tax Guard:** If a profitable stock slips outside Top 30 (say `#35`, still inside Top 60 and passing all bouncers) AND we have already held it for **`320 to 364 days`**, we hold it one extra cycle so it crosses Day 365 and qualifies for **`13.0%` LTCG tax instead of `20.8%` STCG tax**.
4. **Step 4 — Size Each Bet Smartly (`HRP` + Hard Caps + 4 Staggered Sleeves):**
   - Instead of throwing `10%` blindly at every stock, **Hierarchical Risk Parity (`HRP`)** groups stocks that move like twins (say, two PSU lenders) into branches of a family tree and gives calmer, independent companies slightly bigger allocations — strictly capped at **`≤ 12%` in any single stock** and **`≤ 20%` in any single industry**.
   - Instead of risking 100% of your timing luck on one random Tuesday every 2 months, your money is split into **4 equal sleeves (`25%` each)**, staggered **10 trading days apart** across a **40-day cycle**.
5. **Step 5 — The 3-Layer Storm Shelter (Crash Defense + 70/30 Reserve):**
   - **Layer A (6-Gauge Market Weather Radar):** Watches India's Fear Index (VIX), NIFTY's long-term trend, how many stocks are healthy, highs vs. lows, and whether investors are fleeing into Gold. As storm gauges turn red, it automatically steps stock exposure down **`100% → 85% → 55% → 25%`**.
   - **Layer B (`18%` Speed Limiter):** If our 10 stocks suddenly become too wild, it trims stock sizing so expected portfolio bumpiness stays capped at `18%/yr`.
   - **Layer C (`-10%` Emergency Airbag):** If *your own account* ever falls `-10%` from its all-time high, it cuts remaining stock exposure in half until your account recovers past `-6%` and trends back up.
   - **Where does the sidelined cash go?** Never sitting idle at `0%`! It parks **`70%` in safe Indian cash (`LIQUIDCASE` or a Direct Arbitrage Fund)** and **`30%` in Indian Gold (`GOLDBEES`)** — which often *rises* when Indian stocks fall!

---

## 3. Every Single One of the 19 Factors in `V7.5-Core` (What It Means, Why It Works, How Much It Matters & Will It Last?)

We ran a brutal **33-run Leave-One-Out (LOO) experiment** (`2021–2026`, exact Indian statutory NSE fees, FIFO equity taxes, and Post-Budget 2024 reserve taxes). For every single factor below, we literally **ripped it out of the system, retrained the whole AI from scratch, and measured what broke**.

> **How to read the "LOO Impact When Removed" numbers:**  
> A **negative** number (like `-4.59%` Post-Tax CAGR) means **removing that factor HURT performance badly — proving the factor is a huge positive contributor!**

### Group A — The 6 "Core Engine" MVPs (Inside BOTH the 50% Timeless Prior AND the 50% AI)

Every single one of these 6 factors comes from a landmark peer-reviewed finance study, uses zero curve-fitting in the prior, and ranked among the **top 6 MVPs** of the entire engine.

#### 1. Betting Against Beta (`z_neg_beta`) — **#1 MVP OF THE ENTIRE SYSTEM**
* **Prior Weight:** `+10%` (jumps to `+40%` defensive weight during bear-rebound storms)
* **Kitchen-Table Meaning:** Picture two delivery scooters. Scooter A zooms wildly whenever traffic speeds up and slams hard whenever traffic brakes (`Beta = 1.8`). Scooter B cruises smoothly through any traffic (`Beta = 0.6`). We measure how jumpy each stock is versus NIFTY over the past year (gently nudged toward `1.0` so one freak month can't distort it) and **reward the calmer, lower-beta stocks**.
* **Why It Works:** *Frazzini & Pedersen (2014).* Regular investors and mutual funds can't easily borrow money to amplify returns, so they crowd into flashy, roller-coaster high-beta stocks hoping to get rich fast — overpaying for them. Steady, "boring" low-beta compounders get left underpriced and quietly deliver far better risk-adjusted wealth.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR crashes **`-4.18%/yr`**, Honest Post-Tax crashes **`-4.59%/yr`**, Sharpe collapses **`-0.40`** (`1.77 → 1.36`). Removing this single signal hurts more than removing any other factor in the codebase.
* **Will It Stand the Test of Time?** **YES — 9.5 / 10 Durability.** Rooted in permanent institutional leverage limits and human thrill-seeking; proven across 90+ years of global data and multiple Indian cycles.

#### 2. Market-Adjusted Winning Streak — 12-to-1 Month (`z_resid_12m1m`) — **#2 MVP**
* **Prior Weight:** `+25%` (kept at `+25%` even during bear-rebound storms)
* **Kitchen-Table Meaning:** Suppose Company X rose `+30%` over the past year (skipping the noisy most-recent month). Did it rise because the business is genuinely crushing competitors — or just because the whole Indian market rose `+25%` and Company X is a hyper-jumpy stock? This factor **subtracts out the "free market ride"** (called *CAPM Residual Momentum*) and rewards only the stock's *own* company-specific progress divided by its own bumpiness.
* **Why It Works:** *Blitz, Huij & Martens (2011).* Ordinary momentum accidentally loads up on jumpy high-beta stocks during bull markets and gets wrecked when markets snap back. Stripping out market beta isolates genuine company/earnings leadership that big funds take months to fully price in.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR drops **`-1.08%/yr`**, Honest Post-Tax drops **`-0.99%/yr`**, Sharpe drops **`-0.07`**.
* **Will It Stand the Test of Time?** **YES — 9 / 10 Durability.** One of the cleanest, least crash-prone momentum signals in institutional finance (a flagship pillar of Robeco's global quant funds).

#### 3. Market-Adjusted Winning Streak — 6-to-1 Month (`z_resid_6m1m`) — **#3 MVP**
* **Prior Weight:** `+15%`
* **Kitchen-Table Meaning:** The exact same "strip out the free market ride" test as `#2`, measured over the last **6 months** (skipping the last 1 month). It catches fresh emerging winners twice as fast.
* **Why It Works:** *Blitz et al. (2011).* Indian sector waves (Defence, PSU Banks, Pharma, Power, Auto) often build over 3–6 month legs. Pairing 6–1M with 12–1M spots institutional accumulation earlier while staying insulated from market-beta noise.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR drops **`-0.87%/yr`**, Honest Post-Tax drops **`-0.98%/yr`**, Sharpe drops **`-0.07`**, and **MaxDD worsens by `-0.74%`** (`-10.56% → -11.30%`).
* **Will It Stand the Test of Time?** **YES — 8.5 / 10 Durability.** Structural institutional under-reaction to medium-term earnings turns.

#### 4. Early-Stage Winning Streak — 12-to-7 Month (`z_mom_12_7`) — **#4 MVP**
* **Prior Weight:** `+25%`
* **Kitchen-Table Meaning:** Instead of staring at last week's headlines, this measures how strongly the stock climbed **between 12 months ago and 6 months ago** (adjusted for bumpiness).
* **Why It Works:** *Novy-Marx (2012).* Surprisingly, research proved that durable winners aren't driven by last month's retail hype — companies that started a genuine turnaround 7–12 months ago keep drifting upward because large funds take *multiple quarters* to finish research, committee approvals, and buying.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR drops **`-1.14%/yr`**, Honest Post-Tax drops **`-0.91%/yr`**, Sharpe drops **`-0.09`**. Ranked `#1` standalone signal across the 8.8-year `2017–2026` test.
* **Will It Stand the Test of Time?** **YES — 9 / 10 Durability.** Slow institutional capital deployment and multi-quarter business cycles are permanent features of stock markets.

#### 5. Investor Profit Cushion — Capital Gains Overhang (`z_cgo_126`) — **#5 MVP**
* **Prior Weight:** `+15%`
* **Kitchen-Table Meaning:** Estimates the **average purchase price paid by everyone who bought this stock over the past 6 months** (weighted by daily shares traded) and checks whether today's price is *above* or *below* their average cost:
  * **Above average cost (`CGO > 0`):** Most owners are happily in profit — they aren't desperate to dump at break-even, so good news lifts the stock smoothly.
  * **Below average cost (`CGO < 0`):** Thousands of trapped buyers are praying *"please just get back to my buy price so I can sell and escape!"* — creating an invisible ceiling that kills rallies.
* **Why It Works:** *Grinblatt & Han (2005); Kahneman & Tversky's Loss Aversion.* Humans deeply hate locking in a loss. Trapped bag-holders sell every bounce, while stocks where owners sit comfortably in profit face much lighter selling ceilings. Replaced `V6`'s crude 52-week-high gauge and added **`+1.69%/yr`** over `2017–2026`.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax CAGR drops **`-0.49%/yr`**, Honest Post-Tax drops **`-0.63%/yr`**, and **MaxDD worsens by `-0.92%`** (`-10.56% → -11.48%`).
* **Will It Stand the Test of Time?** **YES — 9 / 10 Durability.** Hard-wired human loss aversion does not evolve away.

#### 6. Calmness Over Chaos — 1-Year Volatility (`z_vol_252d`) — **#6 MVP (Inverted / Low-Vol)**
* **Prior Weight:** `-10%` (minus sign = *calmer* stocks win; jumps to `-35%` defensive weight in bear-rebound storms)
* **Kitchen-Table Meaning:** Simply measures how wildly a stock's daily price bounced around over the past year. Calm, steady businesses get rewarded; chaotic roller-coasters get penalized.
* **Why It Works:** *Baker, Bradley & Wurgler (2011).* Compounding math punishes wild swings (`-50%` requires `+100%` just to get back to even!), and retail traders systematically overpay for exciting "thrill" stocks.
* **How Much Difference It Makes (LOO Impact when removed):** Pre-Tax barely budges (`-0.02%/yr`), **YET Honest Post-Tax drops `-0.63%/yr`** and Sharpe drops `-0.02`! Why? Because removing calmness pushes the portfolio into choppier stocks that churn faster and trigger far more `20.8%` STCG tax!
* **Will It Stand the Test of Time?** **YES — 9 / 10 Durability.** Pure compounding math + permanent human attraction to exciting stocks.

---

### Group B — The 13 Supporting Factors (Inside the 50% Restrained Machine Learning Sleeve)

These 13 signals give the restrained AI (straight-line scorecard + shallow 2-step trees) extra nuance on climb smoothness, lottery spikes, downside drops, industry leadership, overnight institutional buying, and market weather.

| # | Factor Name (Code) | Kitchen-Table Meaning | Why It Works | LOO Impact When Removed | Will It Last? |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **7** | **US Market Tailwind** (`z_macro_tailwind_sp500`) | Measures how closely an Indian stock moves with America's S&P 500 (e.g., IT exporters, global pharma/auto) multiplied by the S&P 500's 3-month trend *(strictly yesterday's US close!)*. | Global risk appetite and US tech spending genuinely move Indian exporters and foreign fund (FII) flows. | **`-0.67%` CAGR · `-0.62%` PT · `-0.04` SR · MaxDD worsens `-1.32%`** (`-11.88%`) — **ESSENTIAL** | **7.5 / 10.** US remains ~60% of global stock wealth; Indian IT & FII flows stay linked. |
| **8** | **Smooth Climb — "Frog in the Pan"** (`z_fip_smooth_6m`) | Over 6 months, did the stock rise via *many quiet green days* (like slowly warming water) or via *one flashy headline spike* followed by chop? (`% green days − % red days`). | *Da, Gurun & Warachka (2014).* People ignore slow, boring daily gains (so the trend continues) but overreact to dramatic 1-day news spikes (which fizzle). | **`-0.48%` CAGR · `-0.55%` PT · `-0.03` SR** — **ESSENTIAL** | **9 / 10.** Pure human attention psychology; works across US, Europe, and India. |
| **9** | **Lottery-Ticket Spike Avoidance** (`z_max_ret_21d`) | The stock's single biggest 1-day jump over the past month. AI uses this with a *negative* sign to **steer clear** of sudden casino spikes. | *Bali, Cakici & Whitelaw (2011).* Retail traders chase stocks that just surged `+12%` in a day hoping for a lottery win — temporarily overpricing them so they lag next month. | **`-0.38%` CAGR · `-0.54%` PT · `-0.03` SR** — **ESSENTIAL** | **9 / 10.** Booming retail trading apps in India make lottery-chasing *stronger*, not weaker. |
| **10** | **Bad-Day Bumpiness — 6M Downside Vol** (`z_down_vol_126d`) | Like overall volatility, but measures bumpiness **only on red (losing) days** — never punishing a stock for having big *green* days! | Separates healthy rising companies from fragile ones prone to sudden air-pocket drops. | **`-0.45%` CAGR · `-0.52%` PT · `-0.03` SR · MaxDD `-0.27%`** — **ESSENTIAL** | **8.5 / 10.** Downside jump risk permanently signals fragile businesses or crowded bets. |
| **11** | **Best-in-Industry 3M Leader** (`z_sec_rel_63d`) | How much a stock beat (or trailed) the **typical company in its own industry** over 3 months (comparing HDFC Bank to other Banks, not to Pharma). | Separates true company execution from lucky industry tailwinds, helping find winners even in quiet sectors. | **`-0.30%` CAGR · `-0.35%` PT · `-0.02` SR** — **HELPFUL** | **8.5 / 10.** Market-share gainers vs. laggards within an industry is timeless. |
| **12** | **NIFTY Health vs. 10-Month Average** (`bench_dist_200`) *(Decision tree only)* | How far above or below its 200-day moving average NIFTY 50 sits today (e.g., `+8%` healthy bull vs. `-6%` storm). | Lets the shallow decision tree subtly dial how aggressively it trusts momentum vs. calmness depending on India's market weather. | **`-0.25%` CAGR · `-0.31%` PT · `-0.02` SR · MaxDD `-0.11%`** — **HELPFUL** | **8 / 10.** Market weather awareness is foundational; excluded from linear scorecard so it cannot distort stock rankings. |
| **13** | **Classic 12-to-1M Trend** (`z_mom_12m1m_voladj`) | Traditional 12-to-1 month total gain divided by 1-year bumpiness (before subtracting NIFTY beta). | Complements market-adjusted momentum when genuine broad-market leaders rise alongside a healthy bull market. | **`-0.37%` CAGR · `-0.23%` PT · `-0.03` SR** — **HELPFUL** | **8 / 10.** 200-year empirical phenomenon; kept heavily damped inside AI so beta stays controlled. |
| **14** | **2-Week Speedometer (`RSI 14`)** (`z_rsi14`) | A 14-day speedometer (`0–100`) comparing a stock's recent up-days vs. down-days against the other 129 stocks. | Helps the AI distinguish overheated short-term exhaustion from healthy temporary breathers inside a strong uptrend. | **`-0.36%` CAGR · `-0.22%` PT · `0.00` SR · MaxDD `-0.14%`** — **HELPFUL** | **7.5 / 10.** Short-term breathers inside longer trends persist due to normal trading flows. |
| **15** | **Classic 6-to-1M Trend** (`z_mom_6m1m_voladj`) | Traditional 6-to-1 month gain divided by bumpiness. | Reinforces medium-term trend persistence alongside `z_resid_6m1m`. | **`-0.29%` CAGR · `-0.19%` PT · `-0.02` SR** — **HELPFUL** | **8 / 10.** Pairs cleanly with residual momentum under strong mathematical damping. |
| **16** | **1-Month Cool-Off** (`z_ret_21d`) | How much the stock jumped or dipped over the *most recent month* (21 trading days). | *Jegadeesh (1990).* Stocks that sprinted too fast over the last 3 weeks often pause; slight dips in strong 1-year winners offer cleaner entries. | **`-0.05%` CAGR · `-0.14%` PT · `-0.01` SR** — **MILD POSITIVE** | **7.5 / 10.** Classic short-term liquidity mean-reversion effect. |
| **17** | **Market Breadth (`% Stocks > 50-DMA`)** (`nse_breadth_50d`) *(Decision tree only)* | What share of our 130 Indian companies are trading above their 50-day average (`75%` = broad healthy rally; `25%` = widespread selloff). | Gives the shallow decision tree context on broad vs. narrow markets. *(Primary crash defense happens in the 6-Gauge Risk Gate, not here!)* | **`0.00%` CAGR · `0.00%` PT · `0.00` SR** — **NEUTRAL IN AI** *(ESSENTIAL inside 6-Gauge Risk Gate!)* | **8 / 10.** Harmless inside the heavily constrained tree; kept for market-weather symmetry. |
| **18** | **India Fear Index (`India VIX`)** (`vix_level`) *(Decision tree only)* | NSE's official "Fear Meter" from NIFTY options (`~12–15` calm; `19+` nervous; `23+` panic). | Lets the tree tone down aggressive bets during fear spikes. *(Primary crash defense is in the 6-Gauge Gate & Bear-Rebound Damper.)* | **`-0.03%` CAGR · `+0.07%` PT · `-0.01` SR · MaxDD `-0.13%`** — **NEUTRAL / SLIGHT RISK BUFFER** | **8 / 10.** Options-implied fear reliably marks stressful market regimes. |
| **19** | **Overnight Big-Money Gaps** (`z_overnight_sent_63d`) | Sum of a stock's **overnight jumps** (today's `9:15 AM` open minus yesterday's `3:30 PM` close) over 3 months. | *Lou, Polk & Skouras (2019).* Institutions digest filings and global cues overnight and place opening orders, while noisy retail chatter dominates midday trading. | **`-0.02%` CAGR · `+0.10%` PT · `-0.03` SR · MaxDD worsens `-0.48%`** (`-11.04%`) — **HELPFUL RISK BUFFER** | **7 / 10.** Structural institutional-vs-retail timing split; worth revisiting only if NSE ever launches evening stock hours. |

> **What happened to Feature #20 (`z_macro_tailwind_fx` — Rupee/Dollar stock beta)?**  
> In our 33-run Leave-One-Out audit, **`z_macro_tailwind_fx` was the ONLY factor that actively HURT the system** — removing it **raised** Pre-Tax CAGR by `+1.06%/yr`, **raised** Honest Post-Tax by `+1.23%/yr`, **boosted** Sharpe by `+0.07` (`1.77 → 1.84`), and **shrunk** MaxDD (`-10.56% → -10.23%`)! Why? Because **the Reserve Bank of India (RBI) heavily manages the USD/INR exchange rate**, keeping it artificially flat for long stretches so rolling FX betas are mostly noise. We double-checked removing it across a 9-setting stability grid (`7/9` Post-Tax wins, `3/3` at `40D`, won `2024`, `2025`, and `2026 YTD`) and **permanently deleted it** (`20 → 19` features).

---

## 4. Every Single One of the 13 System Layers (Engine, Steering, Brakes & Airbags)

Picking good stocks is only half the battle. **How you size positions and protect your money during crashes** matters just as much. Here are all **13 architectural layers** of `V7.5-Core`, with their complete, unvarnished Leave-One-Out proof from `task-1981.log`:

| # | System Layer | Kitchen-Table Explanation | What Happens If You Remove It (Complete LOO Proof) | Will It Last? |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **6-Gauge India Risk Gate** *(VIX ≥19, VIX ≥23, NIFTY <200D, Breadth <42%, Hi-Lo <-10%, Gold/NIFTY >+14% → 100/85/55/25% equity)* | The **storm radar**. 1 red gauge → `85%` stocks; 2 red gauges → `55%` stocks; 3+ red gauges → `25%` stocks (`75%` safe cash + gold). | **MaxDD EXPLODES from `-10.56%` to `-25.22%` (`-14.66%` worse!)** and Sharpe crashes `-0.32` (`1.77 → 1.45`). You give up some bull-market raw return (`+4.24%` Pre-Tax / `+5.92%` PT) in exchange for surviving real crashes! | **9 / 10.** 6 independent gauges avoid single-indicator false alarms. |
| **2** | **20-Day Official NSE Delivery Shield** *(`35% ≤ deliv_20d < 98%`, `≥ ₹2 Cr/day`, Entry + Exit parity)* | The **casino bouncer**. Uses official NSE exchange files (`T-1` prior day) to refuse buying — and to *sell at rebalance* — any stock whose real demat delivery collapses below `35%`, hits `≥98%` (T2T surveillance trap), or trades `<₹2 Cr/day` delivery. | **Pre-Tax drops `-2.06%/yr`, Post-Tax drops `-1.78%/yr`, Sharpe crashes `-0.24`, MaxDD worsens `-1.91%`** (`-12.47%`). Our #1 India-specific structural edge! | **7 / 10 (Monitor!).** Works brilliantly today, *but* SEBI rules pushed median delivery up (`34.4% → 51.5%`). Watch the `5.0%` drift alarm! |
| **3** | **50% Walk-Forward AI Sleeve** *(5 chronological folds, 40D blindfold/embargo)* | The **pattern-recognition co-pilot**. Learns subtle combinations across all 19 signals strictly from past years (with a 40-day blindfold so it can never peek ahead). | Removing AI (`100%` Prior) cuts **Pre-Tax `-1.12%`, Post-Tax `-1.56%/yr`, Sharpe `-0.10`, MaxDD `-0.45%`**. | **8 / 10.** Heavy mathematical restraints prevent memorizing historical noise. |
| **4** | **50% Timeless Textbook Prior** *(Zero AI parameters — 6 textbook factors)* | The **steady anchor**. Guarantees the system never goes off the rails or chases a weird historical fluke if market conditions suddenly change. | Removing the Prior (`100%` AI) cuts **Pre-Tax `-1.47%`, Post-Tax `-1.42%/yr`, Sharpe `-0.15`, MaxDD `-0.17%`**. The 50/50 marriage beats *both* halves alone! | **9.5 / 10.** Zero fitted parameters; pure textbook finance. |
| **5** | **30% `GOLDBEES` Defensive Reserve** *(Idle cash split `70%` Cash / `30%` Gold)* | Whenever the storm radar pulls cash out of stocks, `30%` buys Indian Gold (`GOLDBEES`). Gold tends to climb during crises AND when the Rupee weakens. | Removing Gold cuts **Pre-Tax `-1.47%/yr`, Post-Tax `-1.17%/yr`, and worsens MaxDD `-1.58%`** (`-12.14%`), while nudging **Sharpe up `+0.06`** (`1.77 → 1.83`) because gold wiggles daily whereas cash is flat. *(Worth the slight wiggle for crash insurance; don't expect gold to add +1.5%/yr every decade.)* | **8.5 / 10.** Centuries-old INR/crisis hedge. |
| **6** | **Shallow Decision Trees (`HGB`)** *(`40%` of AI ranker; max 2 questions deep)* | Allows simple "if-then" nuance (e.g., *"trust momentum strongly when VIX is calm, be gentler when VIX is high"*) without memorizing individual stocks. | Removing Trees (linear-only AI) cuts **Pre-Tax `-0.78%`, Post-Tax `-1.13%/yr`, Sharpe `-0.09`, MaxDD `-0.63%`**. | **8 / 10.** Ultra-shallow 2-step trees cannot memorize noise. |
| **7** | **`-10% / -6%` Own-Account Drawdown Brake** | Your **personal emergency airbag**. If *your* portfolio falls `-10%` from its all-time high, it halves equity exposure until it heals past `-6%` and rises above its 20-day average. | Removing it cuts **Pre-Tax `-0.68%`, Post-Tax `-0.65%/yr`, Sharpe `-0.07`, and worsens MaxDD `-1.37%`** (`-11.93%`) — and across 2008/2020 it cut peak-to-trough pain by `10.3%`! | **9 / 10.** Model-blind safety net — works even if every factor fails at once. |
| **8** | **Indian `320–364d` LTCG Tax-Cliff Guard** | The **smart CA on your shoulder**. If a profitable stock slips outside Top-30 (still inside Top-60 and passing all bouncers) while aged `320–364` days, hold it one extra cycle to pay `13.0%` LTCG instead of `20.8%` STCG. | Removing it hurts **every single metric simultaneously**: **Pre-Tax `-0.32%`, Post-Tax `-0.53%/yr`, Sharpe `-0.01`, MaxDD `-0.08%`** (and raises STCG tax by `₹13.7k` per `₹10L`). | **8.5 / 10.** Valid as long as India maintains a big STCG vs. LTCG gap at 365 days. |
| **9** | **Family-Tree Sizing (`HRP`)** *(6-month correlation clustering + `≤12%` stock / `≤20%` sector caps)* | Groups stocks that behave like twins into branches so you never accidentally double-bet on the same hidden risk. | Replacing HRP with Equal Weight cuts **Pre-Tax `-0.42%`, Post-Tax `-0.50%/yr`, Sharpe `-0.08`, MaxDD `-0.52%`** (and HRP lifted 2022 bear-market return from `+4.1%` to `+14.8%`!). | **9.5 / 10.** Pure diversification math with zero fragile return forecasts. |
| **10** | **4-Tranche Staggering** *(4 equal 25% sleeves, 10 trading days apart, 40D cycle)* | Instead of betting 100% of your timing luck on one day every 2 months, refresh 25% every 2 weeks. Eliminates "unlucky start-date" lottery. | Dropping to 1 tranche raised return in this 5Y window (`+1.62%` Pre-Tax, `+1.05%` PT, `+0.02` SR) **at the cost of `-1.58%` worse MaxDD (`-12.14%`) and huge start-date luck dispersion** (over 22Y, 4-tranche added `+0.23` Sharpe). | **9.5 / 10.** Institutional standard against rebalance timing luck. |
| **11** | **Damped Straight-Line Scorecard (`Ridge`)** *(`60%` of AI ranker)* | A heavily restrained linear scorecard keeping AI rankings smooth, stable, and immune to jumpy tree quirks. | Removing Ridge (trees-only AI) cuts **Pre-Tax `-0.37%`, Post-Tax `-0.23%/yr`, Sharpe `-0.04`, MaxDD `-0.59%`**. | **9 / 10.** Classic small-universe stability anchor. |
| **12** | **`18%` Portfolio Speed Limiter** *(60-day volatility cap)* | **Cruise control**. If your 10 chosen stocks become unusually bumpy (`>18%/yr`), gently trims stock exposure until calm returns. | Removing it adds a tiny `+0.19%` Pre-Tax / `+0.12%` PT while **worsening Sharpe (`-0.02`) and MaxDD (`-0.78%`, to `-11.34%`)**. | **9 / 10.** Volatility clusters reliably in every market on earth. |
| **13** | **Bear-Rebound Crash Damper** *(Daniel–Moskowitz: `NIFTY 1Y < 0 & VIX ≥ 19`)* | **Free catastrophe insurance.** Only switches on after a true 1-year bear market while fear stays high (like Mar 2009 or Apr 2020), shifting half the prior toward low-beta/low-vol defense so violent junk-stock rebounds don't whip-saw momentum. | Almost silent in `2021–2026` (`-0.03%` Pre-Tax, `0.00%` PT, `0.00` MaxDD — because `2021–2026` had no prolonged `-20%` bear!). Proves its worth in 2009/2020-style bear bottoms. | **9 / 10.** Costs `0%` in normal years and guards momentum's #1 historical weakness. |

**Bottom line of the 33-run audit:** **12 of 13 layers** actively earned their keep during `2021–2026`, and Layer #13 is free catastrophe insurance for the next 2009/2020-style bear-market bottom. There is **zero dead weight** left in `V7.5-Core`.

---

## 5. The Scorecard: How Good Is `V7.5-Core`, How Does It Compare to the Pros, and What Should You Expect?

### 5.1 Official Verified Scorecard (`2021-07-01 → 2026-09-25`, `Top-10`, `40D`, Exact Statutory NSE Costs + FIFO Equity & Post-Budget 2024 Reserve Taxes)

| Engine / Benchmark | Pre-Tax CAGR | Honest Post-Tax (`LIQUIDCASE 31.2%`) | Honest Post-Tax (`Arbitrage 20.8%`) | Excess Sharpe (`6.2%` RF) | Max Drawdown | IIMA 4-Factor Pure Alpha (`t-stat`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NIFTY 50 TRI (`NIFTYBEES.NS`)** | `+9.06%` | `+8.25%` | `+8.25%` | `0.26` | `-16.11%` | `0.00%` (Benchmark) |
| **Old Frozen `V6` Base** *(Unshielded)* | `+23.40%` | `~+19.6%` | `~+19.8%` | `1.21` | `-9.90%` | `+3.01%` (`t=+0.90`) |
| **Frozen `V7` Base** *(Unshielded)* | `+26.54%` | `+22.22%` | `+22.39%` | `1.50` | `-12.18%` | `+7.05%` (`t=+1.98`) |
| **`V7.5-Core` (`20` feats, Pre-FX Prune)** | `+28.91%` | `+24.45%` | `+24.61%` | `1.77` | `-10.56%` | `+10.22%` (`t=+2.95`) |
| **LIVE `V7.5-Core` (`19` feats — `main` DEFAULT)** | **`+29.97%`** | **`+25.67%`** | **`+25.84%`** | **`1.84`** | **`-10.23%`** | **`+10.96%` (`t=+3.14`)** |

#### Year-by-Year Reality Check (Pre-Tax, Inside OOS Window `2021-07-01 → 2026-09-25`)
| Calendar Year | Market Mood | Live 19-Feat `V7.5-Core` | `NIFTYBEES.NS` (NIFTY 50 TRI) | Beat NIFTY? |
| :---: | :--- | :---: | :---: | :---: |
| **2021** *(Jul–Dec)* | Post-COVID boom | **`+42.6%`** | `+11.2%` | **YES** (`+31.4%`) |
| **2022** | Global bear market / rate hikes | **`+17.4%`** | `+5.5%` | **YES** (`+11.9%`) |
| **2023** | Broad Indian rally | **`+56.2%`** | `+21.0%` | **YES** (`+35.2%`) |
| **2024** | Election year / strong bull | **`+35.7%`** | `+10.4%` | **YES** (`+25.3%`) |
| **2025** | Choppy momentum drought | `+9.7%` | **`+11.7%`** | No (`-2.0%`) |
| **2026** *(Jan–Sep 25)* | Broad Indian market selloff | **`-1.4%`** | `-10.6%` | **YES** (`+9.2%`) |

*Notice the honesty in `2025`!* Even a world-class engine will occasionally trail NIFTY by `2%` during a choppy year — and then shine during a `-10.6%` selloff (`2026: -1.4%`) by protecting your wealth!

#### Proof It Isn't Curve-Fit ("Overfitting Score: `1.5 / 10`" — Institutional Grade)
* **No Fragile Magic Numbers:** Adding the Delivery Shield swept **`9/9`** combinations of `(8, 10, 12 stocks × 30, 40, 60 day cycles)`; removing `z_macro_tailwind_fx` won **`7/9`** Post-Tax & **`7/9`** MaxDD (`3/3` at `40D`).
* **Beats All 50 Random-Pick Portfolios (`p = 0.0196`, i.e., `1/51`):** Running the exact same storm shields on randomly picked NSE stocks earned `+14.52%` (`SR 0.78`); our stock selector adds `+12%–15%/yr` of pure selection skill.
* **17 Real Indian Market Wipeouts ("Fallen Angels" — `YESBANK -95%`, `ZEEL -88%`, `ATGL -86%`, `IDEA -84%`, etc.):** `V7.5-Core` beat unshielded `V7` (`+25.85%` vs `+24.46%`, `SR 1.54` vs `1.37`).
* **126 Completely Unseen Indian Smallcaps (`0%` overlap):** `V7.5-Core` beat `V7` completely blindly (`+18.26%` vs `+16.89%`, `SR 0.98` vs `0.86`).
* **Long Multi-Cycle Survival (`2007–2026` 19Y & `2004–2026` 22Y):** Core prior + storm shields survived the `2008` crash, `2009` snapback, `2011` bear, `2018` NBFC crisis, and `2020` COVID crash with `~-17%` MaxDD vs. NIFTY's `-59.9%`.
* **Extreme Fee Torture Test (`0.50%` per side = `1.00%` round-trip, ~3× real NSE costs):** Still earned **`+22.1%+` post-tax**.
* **Multiple-Testing Penalty (`Deflated Sharpe Ratio`):** Even after penalizing our score for **`80` cumulative experiments** we tried over time, the odds that our edge is real exceed **`99.77%`**.

### 5.2 How `V7.5-Core` Compares Against Real-World Quant Firms & Indian PMS Products

| Firm / Product | What They Do Well | Honest Comparison to `V7.5-Core` |
| :--- | :--- | :--- |
| **Robeco Quantitative Equities (David Blitz)** | Global pioneer of Residual Momentum, Low-Vol, and deep multi-decade Point-In-Time fundamentals across 3,000+ stocks. | **Closest intellectual cousin!** We share Residual Momentum, Low-Vol/BAB, and strict anti-overfit discipline. Robeco's edge: decades of paid institutional PIT fundamental data. Joe's edge: zero `1%–2%` annual management fee, zero large-fund market impact, and the official NSE Delivery Shield. |
| **AQR Capital Management** | Multi-factor (Value + Momentum + Low-Beta + Quality) + Volatility Targeting + Bear-Rebound crash research. | We adopted their exact BAB (`-Vasicek β`), vol-cap, and bear-rebound damper research. AQR's edge: adding Value + Quality sleeves for flat momentum years (our parked Phase-2 plan). |
| **Newfound Research (Corey Hoffstein)** | Pioneered **Rebalance Timing Luck** protection (staggered overlapping sleeves). | Our **4-Tranche Staggering (`40D` cycle, 10D apart)** implements Hoffstein's exact institutional design. |
| **Nifty 200 Alpha Low-Vol 30 Index** | Simple rules-based NSE index blending Alpha + Low Volatility. | Solid passive benchmark! `V7.5-Core` adds beta-stripped residual momentum, CGO investor cushion, family-tree HRP sizing, 6-Gauge + `-10%` crash defense, `70/30` cash/gold reserve, and the `<35%` NSE Delivery Shield — roughly halving peak-to-bottom pain. |
| **Indian Momentum PMS / Smallcases** *(Capitalmind, Wright Research, WeekendInvesting, Qode Advisors)* | Strong Indian momentum focus; some blend gold/cash regimes or low-vol filters. | Most retail/PMS products use raw momentum without beta-stripping, lack HRP correlation clustering, lack 4-tranche staggering, lack the official 20D NSE Delivery + T2T shield, and charge **`1.5%–2.5%` annual fees + `10%–20%` profit-sharing** (which quietly siphons `₹20L–₹40L` per `₹1 Cr` over 10 years). |

### 5.3 Realistic Live Money Expectations (Plan Your Life on THIS, Not the Backtest!)

Backtests ALWAYS look slightly shinier than real life (`2021–2024` was one of India's best bull runs ever, and INR gold added a `+1.47%/yr` Pre-Tax / `+1.17%/yr` Post-Tax tailwind). Apply a healthy **30–40% real-world safety haircut** when planning your financial goals:

| 5–10 Year Real-World Scenario | Expected Net Post-Tax Speed *(after all fees & taxes)* | What `₹10 Lakh` Becomes in 5 Yrs / 10 Yrs | What `₹1 Crore` Becomes in 5 Yrs / 10 Yrs |
| :--- | :---: | :---: | :---: |
| **Conservative / Choppy Decade** *(lots of 2018 & 2025-style flat years)* | **`+12% to +15%`** | `₹17.6L – ₹20.1L` / `₹31.1L – ₹40.5L` | `₹1.76 Cr – ₹2.01 Cr` / `₹3.11 Cr – ₹4.05 Cr` |
| **Realistic Base Case** *(normal Indian cycle — **PLAN ON THIS**)* | **`+16% to +19%`** | `₹21.0L – ₹23.9L` / `₹44.1L – ₹56.9L` | `₹2.10 Cr – ₹2.39 Cr` / `₹4.41 Cr – ₹5.69 Cr` |
| **Strong Bull Cycle** *(similar to 2021–2024)* | **`+22% to +26%`** | `₹27.0L – ₹31.8L` / `₹73.0L – ₹100.9L` | `₹2.70 Cr – ₹3.18 Cr` / `₹7.30 Cr – ₹10.09 Cr` |
| *NIFTY 50 TRI Reference* | *`~+9% to +11%`* | *`₹15.4L – ₹16.9L` / `₹23.7L – ₹28.4L`* | *`₹1.54 Cr – ₹1.69 Cr` / `₹2.37 Cr – ₹2.84 Cr`* |

* **Emotional Discomfort You MUST Expect in Advance:**
  * Expect **`-10% to -18%`** drops on screen every 2–4 years (`-₹1L to -₹1.8L` on `₹10L`; `-₹10L to -₹18L` on `₹1 Cr`).
  * Expect **6 to 12 month stretches where your account sits flat or down `-3%` to `-6%`** while friends brag about a hot smallcap tip. **That is totally normal — NEVER abandon or tinker with the system during boring months.**

---

## 6. Will `V7.5-Core` Stand the Test of Time? (Honest Durability Rating: **`8.0 / 10`**)

No honest quant promises "10/10 invincible." Here is the straight truth:

### Why It Scores a Strong `8.0 / 10` (Built for 10+ Years)
1. **Zero "Magic Indicator" Fragility:** Its top 6 drivers (`BAB`, `Residual Mom 12-1M & 6-1M`, `12-7M Mom`, `Capital Gains Overhang`, `Low-Vol`) come from **permanent human psychology** (loss aversion, slow institutional buying, lottery chasing) and **compounding math** — proven across 30–90 years of global markets.
2. **Heavily Restrained AI:** Half the score uses zero AI parameters, and the AI half uses heavily damped linear math + ultra-shallow 2-step trees that physically cannot memorize past quirks.
3. **Liquid Large/Midcap Fortress:** Sticking to India's 130 most liquid NSE companies (mostly F&O stocks with flexible price bands) keeps you out of illiquid smallcap lower-circuit traps and lets your account scale smoothly toward `₹1–5 Crore`.
4. **3 Independent Safety Nets:** Even if stock picking goes cold for a year, the 6-Gauge Risk Gate + 18% Speed Limiter + `-10%` Emergency Airbag + 70/30 Cash/Gold reserve protect your wealth.

### The 4 Black-Swan Scenarios Where `V7.5-Core` CAN Struggle (Know Your Enemy!)
1. **Scenario 1 — Violent V-Shaped "Junk Snapback"** *(like April–June 2009 or May 2020)*:
   * *What happens:* After a scary crash, our Risk Gate & `-10%` Airbag have stepped stocks down to `25%–50%`. Suddenly central banks flood the system with emergency cash, and beaten-down, high-debt junk stocks rocket `+40%` in 3 weeks.
   * *Impact:* `V7.5-Core` will **lag the bounce badly** for 1–3 months while waiting for the `-6%` brake release and staggered sleeves to re-enter. *(That lag is the unavoidable insurance premium you pay to avoid `-50%` disasters.)*
2. **Scenario 2 — Multi-Year Sideways Chop / Pure Deep-Value Cycle** *(like 2018 or an extended 2025)*:
   * *What happens:* Winning sectors flip every 4–6 weeks so 6–12 month winners reverse right after we buy them, while dirt-cheap "deep value" stocks (which we don't yet have a clean PIT sleeve for) lead.
   * *Impact:* `V7.5-Core` treads water (`0% to +8%/yr`) for 12–18 months. *(This is why we silently archive clean daily fundamental snapshots every time you sync — building a future Value/Quality second sleeve for free!)*
3. **Scenario 3 — SEBI Market-Structure Drift (`T+0` Settlement / Margin Rules):**
   * *What happens:* SEBI's Peak Margin (2021) and `T+1` settlement (2023) already pushed median NSE delivery from **`34.4%` (2020) → `51.5%` (2026)**, shrinking the `<35%` hit rate from `51.4%` down to **`5.3%` (`7/131` stocks as of Nov 2026)**. If `T+0` pushes almost *every* stock above `35%`, the `<35%` bouncer stops catching anyone.
   * *Built-in Defense:* **SEBI Delivery Drift Telemetry** automatically flags `"warning"` if `below_35_pct < 5.0%`, and we already pre-tested the exact fallback (`Hybrid: <35% OR Bottom-20%`).
4. **Scenario 4 — Extreme Geopolitical / Correlation-1 Overnight Shock:**
   * *What happens:* Sudden war, exchange halt, or a 1970s-style interest-rate shock where Indian stocks AND gold drop together overnight (`-15%` at the opening bell) before any limit order can fill.
   * *Impact:* No quantitative model on earth prevents an overnight gap. This is why you **NEVER invest emergency living money, cash needed within 3 years, or borrowed money** in the stock market!

---

## 7. What We Tried, What Worked, and The Complete Research Graveyard (Do NOT Re-Test!)

Over months of disciplined testing, we ran **dozens** of famous Wall Street, Indian PMS, and textbook ideas strictly Out-of-Sample with real Indian taxes and statutory fees. **Most ideas that sound brilliant over dinner FAIL on real NSE data.**

### 7.1 What Actually WORKED (Earned Its Place in `V7.5-Core`)
* **CAPM Residual Momentum (`12-1M` & `6-1M`) + `12-7M` Momentum + Capital Gains Overhang (`CGO`) + Betting Against Beta (`-Vasicek β`):** Crushed raw momentum on clean OOS (`+2.8%/yr` PT), 17 real Indian wipeouts (`+6.3%/yr` PT), 8.8Y (`9/9` settings), and 19Y (`2007–2026`).
* **Risk-Adjusted Residual Target + Keeping Market-Wide Macro Out of the Linear Scorecard:** Doubled pure 2-factor alpha (`+4.57% → +8.24%`) while *reducing* market bumpiness (`beta 0.59 → 0.49`).
* **Family-Tree Weighting (`HRP`):** Beat equal-weight and correlation-penalized sizing across `9/9` settings and lifted `2022` bear-market return from `+4.1%` to `+14.8%`.
* **Strictly Prior-Day (`T-1`) 20-Day Official NSE Delivery Shield (`35% ≤ deliv_20d < 98%`, `≥ ₹2 Cr/day`, Entry + Exit):** Added **`~+2.5%/yr` post-tax**, boosted Sharpe `1.50 → 1.77`, cut MaxDD `-12.18% → -10.30%`, and swept `9/9` settings + 17 Fallen Angels + 126 blind smallcaps.
* **Indian `320–364d` LTCG Tax-Cliff Guard:** Improved every single metric (`+0.53%/yr` PT, lower STCG, shallower MaxDD).
* **Pruning RBI-Managed `z_macro_tailwind_fx` ONLY (`20 → 19` features):** Lifted Pre-Tax `+28.91% → +29.97%`, Honest PT `+24.45% → +25.67%`, `SR 1.77 → 1.84`, `MaxDD -10.56% → -10.23%`, `4F Alpha +10.22% → +10.96% (t=+3.14)`, winning `7/9` settings while making the engine *simpler*.

### 7.2 The Complete Research Graveyard (Strictly Tested OOS & KILLED — Never Waste Time Re-Testing!)

| # | Idea We Tested | Why It Sounded Smart | Exact Out-of-Sample Proof of Why We Killed It |
| :-: | :--- | :--- | :--- |
| **1** | **Pure Delivery % Stock Picker** | "Just buy the stocks with the highest delivery %!" | Earned only **`+9.84%` CAGR** (picked sleepy slow-movers). Delivery % works ONLY as a `<35%` bouncer (`p < 0.0001`), NOT a stock ranker. |
| **2** | **Upper Delivery Cap (`> 72%`)** | "Avoid slow stocks with too much delivery." | Hurt return & Sharpe — strong institutional breakouts frequently print `75%–85%` delivery. (Only `≥ 98%` T2T surveillance traps are blocked.) |
| **3** | **Adaptive Bottom-20%/25% Delivery Cutoff** | "Auto-adjust to SEBI's rising median delivery." | Trailed Static `<35%` on CAGR (`+28.66–29.03%` vs `+29.42%`), Post-Tax, MaxDD (`-12.1% to -12.5%` vs `-10.3%`), and `2026 YTD` — threw away healthy `42%`-delivery winners. *(Saved ONLY as a future fallback if `below_35_pct < 5%`.)* |
| **4** | **Earnings Surprises (`PEAD`)** *(Quarterly EPS + post-earnings price jump)* | "Buy stocks that beat profit estimates (Wright/Robeco style)." | Tested TWICE: (a) 2,974 NSE reports (`+2d` lag): Rank IC **`-0.0065`**, hurt `5/6` years; (b) 1,375 events: standalone IC **`-0.0041 (t=-0.23)`**, only `43.3%` coverage; `10%`, jump-only, and EPS-only blends ALL hurt; a noisy `20%` blend ticked slightly up on CAGR (`+24.89%` vs `+24.44%`) but **worsened MaxDD (`-10.92%`) and slashed `2025` (`+4.6%` vs `+7.7%`)**. Indian earnings leak before results day! |
| **5** | **Robeco 3-Factor Residual Momentum (`MKT+SMB+HML`)** | "Strip Size & Value swings too, not just Market beta." | Standalone IC **`-0.0176 (t=-0.79)`**; a `15%` blend cut CAGR **`-1.90%/yr`** & Sharpe `1.77 → 1.62`. On `130` NSE stocks, 3-factor rolling regression is too noisy; 1-factor CAPM residual momentum is far cleaner. |
| **6** | **Indian March Tax-Loss Harvesting (`TLH`)** | "Sell losing stocks before March 31 to trim STCG tax, then re-buy." | Because our 40-day rebalance already cuts losers automatically, only `16` tiny losing lots (`₹14k` total) existed across 5.2 years. NSE round-trip STT (`₹459`) cost MORE than the tax deferred (`₹114`) → **net loss `-₹345`**. |
| **7** | **Forced 9–12 Month LTCG Hold** | "Never sell any stock aged 270–364d to dodge STCG." | Holding fading losers for 3 extra months destroyed `-0.45%/yr` return → **less money after tax** (`₹2.76 Cr` vs `₹2.79 Cr`). Only our tight **`320–364d` profitable Top-60 cliff guard** works! |
| **8** | **Official NSE Foreign Investor (`FII`) Futures Veto** | "Hide in cash whenever foreign funds are 4:1 short." | Cost **`-15.6%` in `2023`** when domestic Indian mutual-fund SIP money powered a huge rally while foreigners stayed short. Keep FII positioning visual-only! |
| **9** | **Expanding Universe (`130 → 231/264` Stocks)** | "More mid/smallcaps = more big winners!" | `130→231` diluted OOS ranking skill (`IC +0.0446 → +0.0322`) and hurt CAGR/Sharpe; `264`-stock expansion with `Screener.in` data carried **`5.5–6/10` hindsight/restatement bias** (kept quarantined on `feat/v75-frontier`). |
| **10** | **Parking Idle Cash in `NIFTYBEES`** | "Keep Risk-Gate cash in NIFTY when NIFTY > 200-DMA." | Hurt BOTH 5Y (`SR 1.96 → 1.69`, `MaxDD -10.6% → -14.5%`) and 22Y (`MaxDD -26.8% → -31.9%`). `70/30` Cash/Gold won cleanly. |
| **11** | **Single-Stock Trailing Stops (`-12% / -15%`)** | "Cut any single stock that drops 12%." | Shook us out of healthy volatile winners right before rebounds; hurt CAGR & Sharpe. Portfolio-level `-10%` Drawdown Brake is far better. |
| **12** | **Dropping Trees / 4Y Decay / Blind Monotonic Trees** | "Simplify AI even further or favor recent years." | Dropping trees cut `-1.13%/yr` PT in `V7.5-Core`; 4Y decay cut `-1.27%/yr` (forgot 2018–2020 crashes!); blind monotonic trees cut `-0.64%/yr`. |
| **13** | **Dropping ALL 5 AI Macro Signals** | "If USD/INR hurt, rip out every macro signal from AI." | Failed stability grid (`SR 4/9`, `MaxDD 4/9`); `2025/2026` fell to `+5.1% / -3.9%`. Prune ONLY `z_macro_tailwind_fx`; keep `z_macro_tailwind_sp500` & `bench_dist_200`. |
| **14** | **US Dollar (`DXY`) / US-10Y Bond Yield Squeeze** | "Strong US dollar crushes emerging markets." | Zero predictive power on India `2017–2026` (`p > 0.85`). India's `₹25,000+ Cr/month` domestic SIP inflows decoupled NSE from classic EM dollar squeezes. |
| **15** | **Illiquidity Premium (`Amihud`)** | "Hard-to-trade stocks pay a textbook liquidity bonus." | Looked great (`+0.0339` IC) on surviving stocks, then **collapsed to `-0.0050 IC (t=-0.40)` on the 17 Fallen Angels** — a classic survivorship trap that buys dying companies! |
| **16** | **Instant Mid-Cycle Re-Entry** | "Jump right back into 100% stocks the day VIX falls." | Got whipsawed in fake bear rallies, **worsened MaxDD `-2.52%`** (`-9.99% → -12.51%`), and jumped turnover `+50%`. One-way de-risking + 10D staggered sleeves is cleaner. |
| **17** | **Smooth Sigmoid Risk Curve (`V8`)** | "Replace step-function 6-Gauge with a smooth dial." | Cut Sharpe `1.50 → 1.34`, worsened MaxDD `-12.18% → -13.45%`, and slashed 4F Alpha `7.05% → 4.78%`. |
| **18** | **Convex Top-Decile AI Target & Uniqueness Weights (`V8`)** | "Weight top-10 winners quadratically & adjust overlap." | Looked flashier on survivors, **failed on the 17 Real NSE Fallen Angels** (`-0.29%` to `-1.68%` behind `V7`, high overfitting probability `55.7%`). Rejected! |
| **19** | **Exotic 2020–2026 Math Swaps** *(Nonlinear `QIS` covariance, Random Fourier Ridge, Signal Smoothing, Partial Aim `0.65`, Shrunk 60D Vol Cap)* | Newer academic papers must beat older math, right? | Every single one tied or trailed `V7.5-Core` on `130` NSE stocks (Partial Aim lost `-1.7%/yr`; shrunk 60D vol cap underestimated stock risk). Confirms `V7.5-Core` is already at the price-data ceiling! |
| **20** | **Extra Macro/Index Alarms** *(Midcap/BankNIFTY in Risk Gate, fast EWMA vol cap, Gold trend timing, Brent mean-reversion, 21D Frenzy)* | More gauges = smarter radar? | Redundant or pure OOS noise (`t = -0.56`); created false alarms during healthy rallies. |
| **21** | **NSE Industry Momentum & Bellwether Lead-Lag (`H1A`)** | "Buy hot sectors or laggards behind sector leaders (Moskowitz–Grinblatt / Hou)." | Standalone 40D IC: IndMom **`+0.0025 (t=+0.08)`**, Lead-Lag **`+0.0158 (t=+0.65)`**. Every blend lost badly: **`-2.77%` to `-3.18%` PT**, `SR -0.20` to `-0.22`, `MaxDD -2.33%`. Already subsumed by stock residual mom + `max 2/sector`. |
| **22** | **Factor Momentum Prior Timing (`H1B`)** | "Dynamically tilt the 6 canonical prior weights toward factors that worked over the last 6–12 months (Ehsani–Linnainmaa)." | Chased lagging factor mean-reversion: `126D` lost **`-2.73%` PT (`SR -0.19`)**, `252D` lost **`-3.13%` PT (`SR -0.23`)**. Static textbook prior weights win decisively. |
| **23** | **Relaxing Sector Cap (`max 3–4/sector`, `H1C`)** | "Let hot sectors hold 3 or 4 stocks instead of capping at 2." | `max=3` lost **`-1.02%` PT (`SR -0.08`, `MaxDD -1.33%`, `2025: +3.5%` vs `+9.7%`)**; `max=4` lost **`-0.92%` PT (`MaxDD -1.03%`)**. `max 2/sector` is strictly optimal. |
| **24** | **NSE Single-Stock F&O Microstructure (`H2A–C`: Futures Basis, OI Build-up, Options PCR-OI, F&O/Cash Spec Ratio)** | "Harvest smart-money derivatives positioning from official NSE F&O Bhavcopies." | Backfilled **`482` official NSE F&O Bhavcopy dates (`55,474` rows, `130` stocks, `2021–2026`, Legacy + UDiFF, strictly `T-1`)**. All **`9 / 9`** blends & hard bouncers LOST (**`-0.73%` to `-2.90%` PT**, `SR -0.05` to `-0.23`). Derivatives positioning decays over `1–10D` (too fast for a `40D` LTCG-aware horizon), backwardation (`<-5%`) false-positives on PSU/IT ex-dividend dates, and Cash Delivery `<35%` already beats every F&O speculation filter! |
| **25** | **India Variance Risk Premium (`VRP = IndiaVIX − RV_21d`, `H3A`)** | "Add a panic-reversal booster whenever realized vol exceeds implied VIX (`VRP < -2`)." | Completely redundant (**`-0.01%` PT, `0.00%` MaxDD**): our existing 6-Gauge Risk Gate already captures all `83` panic days cleanly. |
| **26** | **3-Leg Sovereign Gilt (`LTGILTBEES.NS`) + Gold + Arbitrage Reserve (`H3B`)** | "Add long-term Indian government bonds (`1,850` bars) to cushion equity crashes." | Cutting Gold (`70C/15G/15Gilt`) lost **`-2.16%` PT (`MaxDD -1.14%`)**; keeping `30%` Gold + `20%–30%` Gilt was dead flat (**`-0.05%` to `+0.04%` PT**, `SR -0.01`) while adding Debt-ETF `31.2%` slab-tax clutter. Simple `70/30` Arbitrage/Gold wins. |
| **27** | **Strict PIT NSE Promoter Shareholding & SEBI PIT Insider Trading (`H4A–B`)** | "Buy stocks where promoters/insiders buy and block stocks where they sell." | Harvested **`2,819` strict `broadcastDate` quarterly shareholding filings (`128` stocks)** + **`31,347` SEBI PIT insider filings (`109` stocks)**. Promoter QoQ/YoY had zero 40D IC (`-0.0010` / `+0.0051`); `90D` Promoter/KMP open-market insider net flow had genuine sparse IC (`+0.0313`, `t=+1.85`) but only `13.9%` coverage. **`6 / 7`** portfolio variants LOST (**`-0.65%` to `-2.17%` PT**) and `QoQ < -2%` was flat (`-0.01%`): in NIFTY 100 bluechips, promoter sales are overwhelmingly benign **SEBI 75% MPS OFS compliance or PE exits**, not fraud! |
| **28** | **Blind Hard Promoter-Pledge Filter (`H4C`)** | "Ban any stock whose promoter pledges shares." | NSE `/api/corporate-pledgedata` is **latest-snapshot only** (no historical time series → impossible to backtest PIT without look-ahead), AND blindly bans healthy PE-backed bluechips (`MPHASIS.NS = 100%` Blackstone LBO pledge!). Use strictly as a **read-only advisory badge** if ever displayed — never a hard quant bouncer. |
| **29** | **Raffinot (2018) `HERC` Allocator (`H5A`)** | "Upgrade López de Prado `HRP` to Ward-clustered Hierarchical Equal Risk Contribution." | Lost across every metric: **`-0.59%` PT (`SR -0.06`, `MaxDD -0.89%`, `4F -0.54%`, `2025: +5.9%` vs `+9.7%`)**. On `K=10` sector-capped holdings, discrete cluster cuts overweight noisy singletons vs. smooth `HRP` bisection. |
| **30** | **Always-On or Retail OTM NIFTY Put Tail-Hedge (`H5B`)** | "Buy `10%`-OTM 60D NIFTY puts for COVID-style crash protection." | Always-On puts burned **`-1.77%/yr` (`9.09%` over 5.2Y, `0` ITM expiries)**. Conditional low-VIX (`VIX < 14`) puts cost only `-0.22%/yr` mathematically, but a single NSE NIFTY lot (`~₹18.4L` notional) over-hedges a `₹10L` account `1.8×` and forces F&O ITR-3 tax filing. Unsuitable `< ₹50L`; `6-Gauge + GOLDBEES` already caps MaxDD at `-10.23%`. |

> **Why `0 / 32` Post-`V7.5` Single-Country Equity Ideas Beat Live `V7.5-Core` (Four Permanent Laws):**
> 1. **Horizon Mismatch:** `1–10D` F&O microstructure signals decay before our `40D` LTCG-aware holding period.
> 2. **High-Sharpe Dilution:** Once a 19-feature core sits at `SR = 1.84` / `4F +10.96%`, blending any weak (`|IC| < 0.03`) or sparse (`13.9%` coverage) signal mathematically dilutes Sharpe.
> 3. **India-Specific Structural Traps:** NSE bluechip backwardation is mostly ex-dividend PSU/IT pricing; promoter cuts are mostly SEBI 75% MPS OFS/PE exits; `100%` pledge can be PE LBO debt (`MPHASIS`). Cash Delivery `<35%` dominates every alternative microstructure filter.
> 4. **Static Parsimony Wins at `K = 10`:** Static priors, `max 2/sector`, `HRP`, and `70/30` Arbitrage/Gold beat every dynamic/clustered/3-leg substitute. The **only** remaining Pareto jump is cross-country allocation (`India V7.5-Core + US V7` in Section 8 #2).

---

## 8. What We Left for Later (Parked on the Shelf — With Exact Unlock Conditions)

We deliberately kept 4 promising upgrades **OUT of live production** until the right real-world conditions arrive:

1. **True Point-In-Time (`PIT`) Indian Value + Quality Second Sleeve** *(Code already built: `pit_fundamentals.py`, `forensic_quality_sleeve.py`, `multi_sleeve_allocator.py`)*
   * **Why we want it someday:** In choppy `2025`, a zero-AI Indian Value/Payout sleeve had `-0.203` correlation with `V7` and returned **`+20.5%`** while momentum cooled — pulling combined MaxDD below `-9.5%`.
   * **Why it is parked right now:** Free Indian websites (`Screener.in` / Yahoo) silently rewrite past financial statements and drop dead companies — creating artificially rosy backtests (`5.5–6/10` overfitting risk).
   * **How we are fixing it for FREE:** Every evening you click **Sync & Retrain**, `pit_fundamentals_history` quietly locks away an unchangeable, date-stamped snapshot of all 130 companies' real numbers (already recording since Day 1!).
   * **Unlock Condition:** Once we collect **3–5 years of genuine live snapshots** (or a verified BSE/NSE XBRL filing-date archive), validate a `75% V7.5-Core / 25% PIT Value+Quality` blend across all 5 Promotion Gates.
2. **Global Dual-Engine Portfolio: `India V7.5-Core (19f)` + `US V7 (SEC EDGAR)`** *(Verified across `1,250` aligned INR trading days `2021-07 → 2026-09`)*
   * **Why it is the ONLY proven frontier leap beyond `V7.5-Core`:** Finalized 19-feature **India `V7.5-Core`** and **US `V7` (in INR)** have a **`+0.039` daily INR correlation** (virtually zero)! With **zero code edits and zero new parameters**, allocating capital across both untouched engines transforms risk-adjusted performance in INR (`RF = 6.2%`):
     * `100% India V7.5-Core` *(aligned 1,250d)*: `+29.08%` Pre-Tax CAGR · `11.43%` Vol · `SR 1.77` · `MaxDD -10.48%` · `Calmar 2.77`
     * **`70% India / 30% US (INR)`:** **`+26.72%`** Pre-Tax CAGR · **`8.81%`** Vol · **`SR 2.05`** · **`MaxDD -7.80%`** · **`Calmar 3.43`**
     * **`60% India / 40% US (INR)`** *(Peak Sharpe)*: **`+25.88%`** Pre-Tax CAGR · **`8.35%`** Vol · **`SR 2.08`** · **`MaxDD -6.98%`** · **`Calmar 3.71`**
     * **`50% India / 50% US (INR)`** *(Lowest Drawdown / Peak Calmar)*: **`+25.02%`** Pre-Tax CAGR · **`8.18%`** Vol · **`SR 2.04`** · **`MaxDD -6.15%`** · **`Calmar 4.07`**
   * **Unlock Condition:** Whenever Joe feels ready (`₹25L+` scale recommended) to open an international US brokerage account (IBKR / Vested / INDmoney) under RBI's LRS rules (`20%` TCS above `₹10L`/FY, claimable against ITR; mandatory `Schedule FA` + Form 67 DTAA; mind the USD 60,000 US estate-tax threshold on direct US equities). Rebalance between India and US only semi-annually/annually when weights drift `> ±10%`.
3. **SEBI Delivery Drift Fallback (`Hybrid: <35% OR Bottom-20%`)**
   * **Unlock Condition:** Leave Static `<35%` untouched while `below_35_pct ≥ 5.0%`. ONLY consider switching to our pre-tested **`Hybrid (<35% OR Bottom-20%)`** fallback (`SR 1.81`, `4F Alpha +10.74%`) if SEBI `T+0` settlement pushes `below_35_pct` persistently **below `5.0%`** for **3+ consecutive months**.
4. **Opt-In `₹1 Crore+` Convex Portfolio Optimizer (`backend/convex_optimizer.py`)**
   * **What it is:** Mathematical solver jointly balancing return, risk, NSE fees, market impact, and FIFO tax lots (trimmed turnover `212.7% → 203.7%` in `V8` research).
   * **Unlock Condition:** Only worth testing if live capital grows past `₹1–2 Crore` and order impact becomes noticeable; must cleanly beat `HRP` across all 5 Promotion Gates first.

---

## 9. Indian Taxes, Costs, Dashboard Guide & Your Exact Execution Routine

### 9.1 Indian Costs, Taxes & Where to Park Idle Cash (Post-Budget July 2024 Rules!)

When you trade NSE delivery shares in India, two real-world drains apply — and `V7.5-Core` accounts for every rupee of both:

* **Automatic NSE Statutory Delivery Charges (Charged by Zerodha / Groww / Dhan):**
  * **STT (Securities Transaction Tax):** `0.10%` on BUY + `0.10%` on SELL.
  * **Stamp Duty:** `0.015%` on BUY only.
  * **NSE Exchange + SEBI + 18% GST:** `~0.0036%` per side.
  * **CDSL/NSDL DP Charge:** `₹15.93` flat per stock on a SELL day.
  * *Total statutory + realistic limit-order spread:* **`~0.15%` (`15 bps`) per side** (`~₹150` per `₹1 Lakh` traded).
* **Indian Equity Capital Gains Tax (Finance Act 2024, incl. `4%` Cess):**
  * **Sold in `< 365` days (STCG):** **`20.8%`** of net profit.
  * **Sold in `≥ 365` days (LTCG):** **`13.0%`** of net profit — AND your first **`₹1,25,000` of LTCG profit every Financial Year (Apr 1 – Mar 31) is 100% TAX-FREE!**
  * **Losses reduce future taxes:** Short-term losses (`STCL`) offset both STCG and LTCG; long-term losses (`LTCL`) offset LTCG; unused losses carry forward up to **8 financial years** if you file ITR before the due date!
  * **IMPORTANT — Indian Advance Tax Rule:** Once you trade real money, if your estimated capital-gains tax bill for the financial year exceeds **`₹10,000`**, Indian tax law requires paying **Advance Tax** in quarterly installments (**June 15, Sept 15, Dec 15, Mar 15**) — otherwise the IT department charges `1%/month` interest (Sections 234B/234C). Check your broker's unrealized/realized Tax P&L report once a quarter and ask your CA if an Advance Tax challan is due!
* **CRITICAL Post-Budget 2024 Rule on Your `70 / 30` Defensive Reserve (`LIQUIDCASE` vs. Arbitrage Fund & `GOLDBEES`):**
  * **`70%` Safe Cash Slot:**
    * Under Finance Act 2024 (effective July 23, 2024), debt ETFs like **`LIQUIDCASE`** are taxed at your **income tax slab rate (`31.2%` if you are in the 30% slab!)** regardless of holding period.
    * A **Direct-Plan Arbitrage Mutual Fund** (e.g., Kotak / Parag Parikh / Tata / Edelweiss Arbitrage Direct Growth) holds `>65%` hedged equity, so Indian law taxes it like equity (**`20.8%` STCG `< 12m` or `13.0%` LTCG `≥ 12m`**, sharing your `₹1.25L` exemption) while earning virtually the same `~6.5%–7%` low-risk yield!
    * **Joe's Rule:** If you are in the `30%` tax slab, park the `70%` cash reserve in a **Direct Arbitrage Fund** (saves `~₹1.2L–₹2.2L` per `₹1 Cr` over 5 years; trade-off is `T+2/T+3` redemption instead of instant ETF liquidity). If you are in a lower slab (`≤ 20%`) or want instant brokerage cash, **`LIQUIDCASE`** is totally fine (and far cleaner than `LIQUIDBEES`, which creates daily dividend ITR clutter!).
  * **`30%` Gold Slot (`GOLDBEES`):**
    * Held `< 12 months`: taxed at your **slab rate (`31.2%`)**. Held `≥ 12 months`: taxed at **`13.0%` LTCG**.
    * **Rule:** Avoid pointless tiny flips in `GOLDBEES` so your gold lots cross the 12-month mark whenever possible.

### 9.2 How to Read the EquiLens Dashboard (`http://127.0.0.1:8000/`) in 30 Seconds
* **Settings to NEVER Change:** Keep **Risk Profile = `balanced`** (never switch to `growth`, which turns off the `18%` Speed Limiter!) and **`Top-K = 10`** (only consider `Top-20` if your portfolio exceeds `₹1 Cr` and you prefer `≤ 5%` max per stock). Keep **Smart Money Shield = ON**.
* **Top-Bar `Data Freshness` Pill:** Must read **`HEALTHY`** before trading. If it says `DEGRADED` or `STALE`, do NOT trade — wait and re-sync in the evening window.
* **Risk Gate Banner:** Shows current target equity (`100%`, `85%`, `55%`, or `25%`) and whether the `-10%` Drawdown Brake is `ACTIVE`.
* **Trade Ticket Actions:**
  * **`SELL`** = Exit completely (stock slipped past Top-30 or failed the Delivery/Balance-Sheet Shield). Place SELLs first!
  * **`BUY`** = Fresh Top-10 entry.
  * **`ADD` / `TRIM`** = Top up or dial back an existing holding (the engine automatically turns tiny `< 8%` share tweaks into **`HOLD`** so you never waste brokerage/DP fees!).
  * **`HOLD`** = Do nothing; keep holding.
* **Ignore the "Shortlist" Tab for Execution:** The Shortlist tab (`Visual Composite — Unvalidated`) is only for casual browsing/research — **ALWAYS execute strictly from the Execution / Paper Ledger Trade Ticket!**

### 9.3 Your Exact Step-by-Step Routine (15 Minutes Every 2 Weeks)

1. **Step 1 — Evening Sync (`6:30 PM – 11:00 PM IST` on NSE Trading Days):**
   * Open `http://127.0.0.1:8000/` and click **Sync & Retrain** strictly between **`6:30 PM and 11:00 PM IST`**.
   * *Why?* NSE publishes official daily delivery files after `~6:00 PM IST`, and Yahoo Finance sometimes glitches around `~4:00 AM IST`. Evening syncing guarantees clean, settled data.
   * Confirm the **Data Freshness** pill reads **`HEALTHY`**.
2. **Step 2 — Check Your 4-Sleeve Calendar (Trade Every 10 NSE Trading Days):**
   * Only rebalance on a scheduled sleeve date (~every 2 weeks).
   * *Live Forward Paper Account Schedule (running untouched since `2026-11-20` at `₹10,00,000`, marked `₹10,10,032.98` as of `2026-11-26` vs. NIFTY TRI `+0.38%`):*
     * **Sleeve 1:** Active (`2026-11-20`) → Next due **`2027-01-19`**
     * **Sleeve 2:** Next due **`2026-12-07`**
     * **Sleeve 3:** Next due **`2026-12-21`**
     * **Sleeve 4:** Next due **`2027-01-05`**
3. **Step 3 — Place Broker Limit Orders Inside NSE's Calm Windows (NEVER at 9:15 AM!):**
   * Place **Limit Orders** right near the screen bid/ask (NEVER blind Market Orders!) strictly during:
     * **Morning Window:** **`10:00 AM – 10:30 AM IST`** *(after chaotic 9:15–9:45 AM opening spreads settle)*, OR
     * **Afternoon Window:** **`2:45 PM – 3:15 PM IST`** *(tightest institutional spreads; avoid 3:25–3:30 PM closing spikes)*.
   * **Order Sequence:** Execute **`SELL` / `TRIM` orders first** (frees up cash), then **`BUY` / `ADD` orders**, then park leftover reserve cash **`70%` Arbitrage/`LIQUIDCASE` + `30%` `GOLDBEES`**.
   * **Liquidity Check (`≤ 2%` of daily volume):** If the Trade Ticket ever flashes an `adv_cap_warning` at large capital (`₹1 Cr+`), split that stock's order across 2 days.
4. **Step 4 — Commit & Back Up:**
   * Click **Commit Paper Rebalance** (for the paper tracker) and copy `data/equilens_india.db` periodically to a private backup folder / encrypted cloud drive (**NEVER commit `data/equilens_india.db` to GitHub!**).
5. **Step 5 — The Foolproof Real-Money Capital Ramp Plan:**
   * **Months 1–3 (Now — Started Nov 2026):** Run the `₹10L` live forward paper portfolio with 100% discipline. Verify zero bugs, zero missed rebalances, and calm nerves.
   * **Months 4–6:** Deploy **`₹15L – ₹25L`** of real money (`Top-10`, `balanced`). Confirm real Zerodha/broker charges, limit fills, and your emotional comfort on red days.
   * **Months 7+:** Scale gradually toward **`₹1 Crore`** across 3–4 monthly tranches (`Top-10` caps at `≤ ₹12L/stock`; `Top-20` caps at `≤ ₹5L/stock` if you want smaller individual bets).

---

## 10. "What Do I Do When...?" — The 21-Scenario Foolproof Crisis & Circumstance Handbook

Consult this table **whenever ANYTHING happens** — market crash, boring year, dividend/stock split, broker outage, SEBI rule change, tax deadline, or life emergency.

| # | Circumstance / Scenario | What Is Actually Happening | Exact Action to Take (DO / DO NOT) |
| :-: | :--- | :--- | :--- |
| **1** | **Portfolio drops `-5%` from its peak** | Routine market wiggle (happens 2–3× every year). | **DO NOTHING.** Don't sell, don't touch settings, don't stare at prices hourly. Wait for your scheduled 10-day rebalance. |
| **2** | **Portfolio drops `-10%` from its peak** | Normal 1-in-2-year correction; the **Drawdown Brake (`-10%`)** flips ON automatically. | **FOLLOW THE TRADE TICKET.** The engine halves stock exposure and parks cash `70/30` in Arbitrage/`LIQUIDCASE` + `GOLDBEES`. **DO NOT** override it to "buy the dip," and **DO NOT** panic-sell the remaining half! Wait for automatic release (recovers past `-6%` and trends back above its 20-day average). |
| **3** | **Portfolio drops `-15%` to `-18%`** | Major macro storm (like 2008 / 2020 / war). Your shields are doing heavy lifting (NIFTY is likely down `-25%` to `-50%`). | **STAY CALM & OBEY THE ENGINE.** Verify Risk Gate + Drawdown Brake are active. **NEVER** switch off the system at the bottom — selling at the trough locks in permanent damage. |
| **4** | **6–12 months of flat returns while friends brag about +50% smallcaps** | Normal momentum drought / sideways chop (like `2018` or `2025`). | **STAY THE COURSE.** Friends never brag when their tip crashes `-60%`. Remember `V7.5-Core` made `+9.7%` in choppy `2025` and `-1.4%` while NIFTY fell `-10.6%` in `2026`. Never tinker during a boring stretch! |
| **5** | **Scary overnight news (war, election shock, RBI surprise — market opens `-6%`)** | Opening-bell panic with huge bid–ask spreads. | **DO NOT TRADE AT 9:15 AM!** Wait until after `6:30 PM IST`, run **Sync & Retrain**, let the 6-Gauge Radar & Airbag evaluate settled closing prices, and execute any trims with Limit Orders at **`10:00–10:30 AM`** or **`2:45–3:15 PM`** the next day. |
| **6** | **Risk Gate drops equity to `55%` or `25%`** | 2 or 3+ market storm gauges turned red together. | **OBEY FULLY.** Execute the SELL/TRIM orders on the Trade Ticket and park proceeds `70%` Arbitrage/`LIQUIDCASE` + `30%` `GOLDBEES`. Don't leave cash sitting at `0%` in your broker wallet for months. |
| **7** | **A stock you own says `"EXIT — Low 20D NSE Delivery (<35%)"` or `"Balance-Sheet Risk"`** | Serious buyers left and intraday speculators took over — OR the company's finances tripped `≥ 2` red flags. | **SELL IT AT REBALANCE** even if it is still Rank #15! This exit rule alone adds `~+2.5%/yr` and keeps blowups out of your portfolio. |
| **8** | **A stock you own hits a lower circuit / T2T / trading halt** | Rare company shock (uncommon in our 130 F&O-heavy liquid stocks, but possible). | **DO NOT** average down on a falling knife. If the ticket says SELL, place a Limit Sell at the daily lower-circuit floor at `9:15 AM` and wait patiently for liquidity to return. |
| **9** | **A stock pays a Cash Dividend** | The company sent free cash directly to your linked bank account! | **ENJOY IT & LET IT COMPOUND.** Add accumulated dividends back into your reserve cash (`LIQUIDCASE`/Arbitrage) and deploy them at the next scheduled rebalance. *(Remember dividends are taxed at your slab rate in your ITR.)* |
| **10** | **A stock announces a Stock Split (e.g., 1:5) or Bonus Shares (e.g., 1:1)** | Your 10 shares at `₹1,000` turn into 50 shares at `₹200` — your total rupee value is unchanged! | **DO NOT PANIC IF THE CHART LOOKS WEIRD FOR 1 DAY.** Yahoo Finance auto-adjusts prices for splits/bonuses (occasionally taking 24–48 hrs). Wait until `Data Freshness = HEALTHY` and share counts settle in your demat before rebalancing; update your manual share count if using a custom ticket. |
| **11** | **A company announces a Demerger / Spin-off (e.g., ITC Hotels / JioFin) or Delisting/Buyback** | You receive shares of a new spun-off company in your demat locker, or a delisting tender offer. | **Spun-off stock not in our 130 universe:** Sell the spun-off shares calmly with a Limit Order during `10:00–10:30 AM` or `2:45–3:15 PM` once listed and fold the cash into the next rebalance. **Delisting:** Tender/sell before delisting so you never hold unlisted paper. |
| **12** | **SEBI Delivery Drift Telemetry shows `warning` (`below_35_pct < 5.0%`)** | SEBI `T+0` or margin rules pushed almost every stock above `35%` delivery, weakening the `<35%` bouncer. | **DO NOT change code on Day 1.** Watch it for **3 consecutive months**. If it stays persistently `< 5.0%`, have a future AI run the 5-Gate test comparing Static `<35%` vs. our pre-tested **`Hybrid (<35% OR Bottom-20%)`** fallback. |
| **13** | **Dashboard shows `Data: DEGRADED` or `STALE`** | Synced outside `6:30–11:00 PM IST`, Yahoo Finance hiccuped, or NSE had a holiday. | **NEVER bypass with `allow_stale=True`!** Wait 30 minutes and click **Sync & Retrain** again inside `6:30–11:00 PM IST`. If Yahoo is temporarily down, simply shift the rebalance by 1 trading day — 1 day never matters over a 40-day cycle. |
| **14** | **Broker app (Zerodha/Groww) or NSE exchange has a technical outage during your window** | Occasional exchange/broker glitch. | **DO NOT panic or spam duplicate orders!** Confirm no pending/partial orders remain open, then simply execute the remaining Limit Orders in the afternoon window (`2:45–3:15 PM`) or the next trading day. |
| **15** | **You are traveling / sick / miss a rebalance day** | Life happens! | **ZERO STRESS.** Run **Sync & Retrain** and execute the ticket on the next trading day you are free. Being 1–5 days late barely affects a 4-sleeve 40-day system. |
| **16** | **You want to ADD fresh money (`+₹5L` or `+₹50L`)** | Bonus, savings, or scaling up toward `₹1 Cr`. | **Small add (`< 20%` of account):** Park in Arbitrage/`LIQUIDCASE` until the next scheduled 10-day rebalance. **Large add (`> 20%`):** Split evenly across the next **4 scheduled 10-day rebalances** (`25%` per sleeve) so you never bet a big lump sum on one lucky/unlucky day! |
| **17** | **You urgently need to WITHDRAW money for life** | Home, medical, or family priority. | **Pull in this strict order:** (1) Idle Arbitrage/`LIQUIDCASE` cash first → (2) `GOLDBEES` reserve second → (3) Only if still needed, trim all 10 stocks **proportionally** (keeping balance intact), preferring lots `≥ 365` days old or smallest-gain lots to minimize STCG tax. |
| **18** | **Your laptop breaks or `http://127.0.0.1:8000` won't open** | Local software hiccup — your real stocks sit untouched in CDSL/NSDL! | Restart `uvicorn backend.main:app --host 127.0.0.1 --port 8000`. On a brand-new machine, `git clone` the repo, restore your offline backup of `data/equilens_india.db` (or type your current broker holdings into the Trade Ticket), and click **Sync & Retrain**. |
| **19** | **Advance Tax Quarters (`Jun 15 / Sep 15 / Dec 15 / Mar 15`) & March 31 ITR** | Indian tax calendar! Someone on YouTube pushes "March Tax-Loss Harvesting." | **IGNORE March TLH** (we proved it *loses* money on NSE after STT!). Check your broker's realized Tax P&L once a quarter so you pay Advance Tax if total tax owed `> ₹10,000`. After March 31, download your broker's **Tax P&L** + IT portal **AIS/TIS**, hand both to your CA, carry forward any STCL/LTCL losses, and claim your `₹1.25 Lakh` tax-free LTCG exemption! |
| **20** | **A friend, YouTuber, or AI suggests "adding RSI/MACD/FII/Crypto/Deep Learning"** | Tinkerer's itch — the #1 silent killer of good quant portfolios. | **SAY NO BY DEFAULT.** Point them to **Section 7.2 (Graveyard)** and **Section 16.2 (5-Gate Promotion Standard)**. Nothing touches live code unless it passes all 5 Gates strictly OOS, beats 17 Fallen Angels, and wins a 9-cell plateau! |
| **21** | **WHEN SHOULD YOU ACTUALLY STOP RUNNING THIS SYSTEM? (Objective Kill-Switch)** | Telling real structural decay apart from normal bad luck. | **NEVER quit after 1 flat year.** Pause and trigger a full forensic audit ONLY if **ANY** of these 3 plain-English alarms fire: **(a)** Your live account falls **`-25%`** from its peak (more than double our backtested `-10.2%` MaxDD), **(b)** Over **3 full consecutive years (36 months)** your account lags `NIFTYBEES` by more than **`-10%/yr`** with no obvious macro shock, OR **(c)** Ask an AI to check the **Model Metrics / Journal** screen: if the engine's live 2-month stock rankings fail to beat random guessing (`Rank IC < 0.0`, meaning its Top-10 picks consistently trail the rest of the 130 stocks) for **~2 straight years (12 consecutive 40-day cycles)**. |

---

# PART II — ENGINEERING & QUANTITATIVE HANDOFF BIBLE (FOR FUTURE AI ITERATIONS)

> **Instructions to Future AI Assistants:** Read Part II completely **before** touching a single line of code, running any backtest, or proposing any research upgrade. All live figures, formulas, routes, and schemas below were verified directly against `backend/` and read-only `data/equilens_india.db` at commit `5d68654` (`main` synced with `feat/v7-book-research`).

---

## 11. Production System Identity, Repo State & Hard Guardrails

### 11.1 Live Production State Snapshot *(as of Nov 27, 2026)*
* **Repository Path:** `/home/joe/.gemini/antigravity/scratch/equilens-india-quant`
* **Remote:** `https://github.com/Joestilllives/equilens-india-quant`
* **Live Branch:** `main` (synced with `feat/v7-book-research`; historical `V6` playbook at `7614ab0`; quarantined research branches: `feat/v75-frontier`, `feat/v8-frontier-research`, `v6-institutional-hardening`)
* **Strategy Identifiers:**
  * Live Backtest Production Default (`portfolio_backtester.py`): `strategy_id = "v75_core_delivery_shield"` (when `enforce_smart_money_shield=True`, the default).
  * `config.py` constants retain `STRATEGY_ID_ML = "v7_book_hybrid_hrp"`, `STRATEGY_ID_EXECUTION = "v7_book_live_execution"`, `STRATEGY_ID_SHORTLIST = "unvalidated_research_composite"` — keep shortlist strictly separated from ML execution! Note: `ml_runs.strategy_id` in SQLite still defaults to legacy `'v6_ml_core'` / `benchmark_symbol='^NSEI'` unless explicitly overwritten.
* **Live Local Server:** `http://127.0.0.1:8000/` (`uvicorn backend.main:app --host 127.0.0.1 --port 8000`)
* **Universe Parity Note (IMPORTANT FOR REBUILDS!):**
  * `config.DEFAULT_UNIVERSE` contains **`123` symbols** (`122` NSE equities + `^NSEI`).
  * Live `data/equilens_india.db` contains **`133` tickers** (`130` NSE equities + `^NSEI`, `NIFTYBEES.NS`, `GOLDBEES.NS`) — including **8 additional NSE equities** added beyond `DEFAULT_UNIVERSE`: `ADANIGREEN.NS`, `BAJAJHLDNG.NS`, `BIOCON.NS`, `ICICIPRULI.NS`, `JINDALSTEL.NS`, `JSWENERGY.NS`, `LTTS.NS`, `TATAELXSI.NS`. Preserve these 8 tickers if ever rebuilding from scratch!
* **Live SQLite Database (`data/equilens_india.db` — 26 user tables + `sqlite_sequence`, NEVER commit to git!):**
  * `tickers`: `133` rows | `ohlcv`: `301,069` daily bars (`10y` history, latest `^NSEI` bar `2026-11-26`)
  * `nse_delivery`: **`220,229` rows across `1,700` consecutive NSE trading days (`2020-01-01` → `2026-11-26`)** seeded from `nsearchives.nseindia.com` (`sec_bhavdata_full_DDMMYYYY.csv`). Live 20D telemetry (`2026-10-28 → 2026-11-26`): `n=131`, `below_35=7 (5.3%)`, `median=51.5%`, `p25=46.2%`, `t2t=0`, `low_turn=0`.
  * `ml_predictions`: `130` equity rows (`0` ETF leaks) | `ml_oos_history` & `ml_oos_history_v2`: `161,586` rows (`2021-07-01 → 2026-09-25`) | `ml_runs`: latest Run `#8` (`2026-11-27T16:00:54Z`, `sample_count=262,188`, `symbol_count=130`, `horizon_days=40`, `ic_spearman=0.033`, `hit_rate=73.8%`, `top5_spread=+3.57%`, `tb_auc=0.523`).
  * `pit_fundamentals_history`: `130` immutable Day-1 snapshots (auto-appended on every `sync_universe`).
  * **Live Forward Paper Ledger (`paper_*` — STRICTLY UNTOUCHED):**
    * Started `2026-11-20` with `₹10,00,000` (`balanced`, `top_k=10`); marked as of `2026-11-26` at **`₹10,10,032.98` NAV** (`+1.00%` vs `NIFTYBEES.NS` TRI `+0.38%`), equity `₹9,52,057.98`, cash reserve `₹57,917.70` (`liquidcase_inr=₹57,917.70`, `goldbees_inr=₹0.00` pre-first-rebalance + `₹57.30` accrued interest), `10` open positions (`ADANIPORTS.NS`, `AUROPHARMA.NS`, `BHEL.NS`, `BOSCHLTD.NS`, `DIVISLAB.NS`, `FEDERALBNK.NS`, `MARICO.NS`, `MCX.NS`, `MOTHERSON.NS`, `TECHM.NS`), `10` FIFO lots (`paper_lots`), `4` staggered sleeves (`paper_sleeves`: Sleeve 1 active `2026-11-20 → 2027-01-19`, Sleeve 2 next `2026-12-07`, Sleeve 3 next `2026-12-21`, Sleeve 4 next `2027-01-05`).

### 11.2 Absolute Engineering Do-Not-Break Rules
1. **NEVER modify, reset, or migrate-destroy `paper_account`, `paper_positions`, `paper_trades`, `paper_lots`, `paper_sleeves`, or `paper_nav_history`.** Always copy `data/equilens_india.db` to an outside backup path before any schema/ingestion work, and ALWAYS run `pytest` / audit scripts against a temporary DB copy via `EQUILENS_DB_PATH` or `--db`.
2. **NEVER commit `data/equilens_india.db`** (or cache files like `data/multicycle_2004_2026_cache.db`, `data/fallen_angels_cache.json`) to git.
3. **NEVER run unconstrained deep ML or wide hyperparameter grids on `N ≈ 130` stocks.** Preserve the `50%` zero-parameter Prior + strongly regularized `Ridge(alpha=500)` + shallow `HGB(max_depth=2)` architecture.
4. **NEVER re-test Graveyard items (Section 7.2)** unless a genuinely new, strictly Point-In-Time data source exists (e.g., 3–5+ years of accumulated `pit_fundamentals_history` or official BSE/NSE XBRL archive).
5. **NEVER let visual UI composites (`shortlister.py`) bleed into execution (`execution_engine.py`).** Shortlist stays explicitly tagged `unvalidated_research_composite`.

---

## 12. Exact Mathematical & Implementation Specification (`V7.5-Core`)

### 12.1 Universe & Non-Equity Hygiene (`backend/config.py`)
* `BENCHMARK_SYMBOL = "^NSEI"`
* `TRI_BENCHMARK_SYMBOL = "NIFTYBEES.NS"` (dividend-adjusted TRI proxy; fallback to `^NSEI` only if `< 120` bars)
* `GOLDBEES_SYMBOL = "GOLDBEES.NS"` (real Indian Gold ETF daily returns for the `30%` defensive reserve; causal `GC=F * INR=X` proxy fallback)
* `NON_EQUITY_SYMBOLS = frozenset({"^NSEI", "NIFTYBEES.NS", "GOLDBEES.NS"})` — **MUST** be excluded from ML training, ML inference, cross-sectional z-scoring, NSE breadth, 52W Hi-Lo sentiment, Pollet–Wilson correlation, execution ranking, and shortlist defaults.
* `HISTORY_PERIOD = "10y"`, `ML_HORIZON_DAYS = 40`, `TB_HORIZON_DAYS = 40`, `TB_UP_ATR_MULT = 2.0`, `TB_DOWN_ATR_MULT = 2.0`, `DEFAULT_COST_BPS = 15.0`.

### 12.2 The 19 Active Features (`FEATURE_COLS` in `backend/ml_predictor.py`)
Daily cross-sectional z-scoring (`ddof=0`, clipped to `[-3.0, +3.0]`, missing filled `0.0`) is applied across clean equities only (`vol_safe = clip(vol_252d.fillna(vol_20d).fillna(0.25), min=0.12)`):

* **Canonical Prior + ML Features (`6`):**
  1. `z_resid_12m1m`: `sum(eps_1d.shift(21), 231d) / clip(idio_vol_252, min=0.10)`, where `eps_1d = ret_1d - beta_vasicek.shift(1) * bench_ret_1d` (strictly causal lagged beta)
  2. `z_resid_6m1m`: `sum(eps_1d.shift(21), 105d) / clip(idio_vol_126, min=0.10)`
  3. `z_mom_12_7`: `(close.shift(126) / close.shift(252) - 1) / vol_safe`
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
* **Final Hybrid Score & Legacy DB Column Naming Gotcha:**
  * `r4_score = 0.50 * nse_factor_prior + 0.50 * cs_zscore(pred_rank)`
  * **OOS folds (`ml_oos_history`):** `ml_oos_history.pred_alpha` stores **raw unscaled `r4_score`** (rounded to 5 decimals); display `pred_alpha_pct = clip(r4_score * 2.5, -12.0, 12.0)`.
  * **Live inference (`ml_predictions`):** `alpha_pct = clip(r4_live * 2.5, -14.0, 14.0)` — stored in legacy SQLite column name `pred_alpha_20d` (represents `40D` alpha!), while `vol_20d_ann` stores `252D` annualized vol and `ml_oos_history.actual_fwd_20d` stores `40D` forward return. Do NOT confuse these legacy column names with a 20D horizon!

### 12.4 Portfolio Construction, Shields, Risk Defense, Accounting & Backtest↔Live Parity
* **Selection & Hysteresis (`execution_engine.py` & `portfolio_backtester.py`):**
  * Default `top_k = 10`, `rank_buffer = min(top_k * 3, 40) = 30`, `max_per_sector = 2` (in `execution_engine.py`, `effective_max_per_sec` automatically widens to `round(max_sector_w / max_stock_w)` when `top_k >= 12`, e.g., `4` stocks/sector at `top_k=20`), `rebalance_days = 40`, `n_tranches = 4` (offsets `0, 10, 20, 30`).
  * **20D NSE Delivery Shield (Backtest + Live Entry & Retention Parity):** Blocks/exits any stock if ANY of:
    1. Strictly `T-1` `deliv_20d_pct < 35.0`
    2. `deliv_20d_pct >= 98.0` (T2T / ASM surveillance trap)
    3. `0 < deliv_turnover_20d_cr < 2.0` (`₹2 Cr/day` deliverable turnover floor)
  * **Balance-Sheet Blowup Shield (LIVE EXECUTION ONLY — intentionally NOT in historical backtester!):** In `execution_engine.py`, blocks/exits `shield_status == "Risk"` (`≥ 2` flags: non-financial `D/E > 180%`, `ROE < 6%`, non-financial `FCF Yield < -4%`, `Revenue Growth < -10%`). **Deliberately excluded from `portfolio_backtester.py`** because Yahoo Finance fundamentals are non-PIT current snapshots (applying them to 2021–2025 would introduce lookahead bias!).
  * **Indian `320–364d` LTCG Tax-Cliff Guard (`ltcg_cliff_guard=True` default):** Retains profitable existing holdings aged `320 <= days < 365` if rank `<= max(rank_buffer, 60)` AND passing all tradability/delivery/T2T/turnover (and live balance-sheet) shields.
* **Weighting (`backend/hrp_allocator.py`):**
  * López de Prado (2016) `HRP` (`single` linkage, quasi-diagonalization, recursive bisection inverse-variance) on **126D Ledoit–Wolf shrunk covariance** (`LedoitWolf(assume_centered=False)`).
  * Iterative hard concentration caps (`apply_concentration_caps`): `max_stock_w = 0.05` if `top_k >= 20` else `min(0.15, 1.20 / top_k)` (`= 0.12` at `K=10`); `max_sector_w = 0.25 if top_k < 10 else 0.20`.
* **3-Layer Crash Defense:**
  1. **6-Gauge Risk Gate:** `+1` pt each for `vix >= 19.0`, `vix >= 23.0`, `^NSEI < SMA200`, `breadth_50d < 42.0%`, `hilo_sentiment < -10.0%`, `gold_nifty_63d > +14.0%`. Target equity: `0 pts → 100%`, `1 pt → 85%`, `2 pts → 55%`, `3+ pts → 25%`. One-way intra-cycle de-risking in backtester (`cur_exp > target_exposure + 0.15`).
  2. **18% Ex-Ante Vol Target:** `vol_cap = clip(0.18 / max(0.08, ex_ante_vol_60d), 0.25, 1.0)` using **60D sample covariance** (do NOT replace with `sklearn.LedoitWolf`!).
  3. **`-10% / -6%` Peak-NAV Drawdown Brake:** Triggers at `dd <= -0.10` (cuts equity exposure `× 0.50`); releases only when `dd > -0.06` AND `nav > SMA20(nav) * 1.005` (backtester charges `0.35 * cost_rate` transition slippage on toggle).
* **Defensive Reserve (`70%` Cash / `30%` `GOLDBEES`) & Statutory/Tax Engine (`backend/indian_costs.py`):**
  * Cash yield: `INDIA_CASH_ANNUAL_YIELD = 0.062`.
  * Statutory NSE delivery charges: `0.10%` STT both sides, `0.015%` buy stamp duty, `0.00297%` NSE fee, `0.0001%` SEBI fee, `18%` GST, `₹15.93` sell DP charge, `3.0 bps` half-spread, `min(250 bps, 0.25 * clip(daily_vol, 0.005, 0.10) * sqrt(order / ADV_20d) * 10000)` impact.
  * FIFO equity tax (`FifoTaxLedger`): `20.8%` STCG (`< 365d`), `13.0%` LTCG (`≥ 365d`, `₹1,25,000`/FY exemption), STCL/LTCL set-off & carryforward.
  * **Post-Budget 2024 Reserve Tax Modeling (`portfolio_backtester.py` lines 1011–1012):** Conservatively taxes `100%` of positive `liquid_gain_inr` at `31.2%` (`reserve_tax_liquidcase`) or `20.8%` (`reserve_tax_arbitrage`), and `100%` of positive `gold_gain_inr` at `31.2%` (conservative `<365d` assumption — real-life `≥12m` arbitrage/gold lots enjoy `13.0%` LTCG). Both `after_tax_cagr_liquidcase_pct` and `after_tax_cagr_arbitrage_pct` are emitted.
* **Known Intentional Backtest vs. Live Execution Differences (Do Not "Fix" Blindly!):**
  | Dimension | `portfolio_backtester.py` | `execution_engine.py` (Live) | Why Intentional |
  | :--- | :--- | :--- | :--- |
  | Balance-Sheet Shield (`Risk`) | Off | **On** | Yahoo fundamentals are non-PIT; running them historically would leak future financials. |
  | Sell DP Charge (`₹15.93`) | `include_dp_charge=False` | `include_dp_charge=True` | Fractional tranche simulation vs. integer live share sells. |
  | Tiny-Trim Deadband | `no_trade_band=0.0` | `< 8%` share delta → `HOLD` | Avoids pointless 1–2 share live broker/DP churn. |
  | `ltcg_cliff_guard` Age Check | Oldest `FifoTaxLedger` lot | `paper_positions.entry_date` | Hardcoded default `True` (not exposed as toggle in `main.py`). |
  | 4-Tranche Execution | 4 independent 25% sub-ledgers | Whole-account rebalance advancing due sleeve `+40` NSE days | Keeps live retail demat tracking simple with single clean trade ticket. |
* **SEBI Delivery Drift Telemetry (`backend/money_signals.py`):**
  * Emits `delivery_regime_telemetry` on `/api/money-signals` and `/api/execution/ticket`; triggers `drift_status = "warning"` when `universe_count >= 30` and `below_35_pct < 5.0%`.

---

## 13. Complete Codebase, Database & API Architecture Map

### 13.1 Backend Modules (`backend/` — 26 Modules + `__init__.py`)
| Module | Role | Status |
| :--- | :--- | :--- |
| `config.py` | `DEFAULT_UNIVERSE` (`123` syms), `NON_EQUITY_SYMBOLS`, `EQUILENS_DB_PATH` override, CORS/auth/freshness env vars, `40D` horizon, `15 bps` default | **CORE LIVE** |
| `database.py` | Thread-safe WAL SQLite store (26 user tables), OHLCV, macro, NSE delivery, ML, PIT fundamentals, paper ledger | **CORE LIVE** |
| `data_fetcher.py` | Yahoo Finance OHLCV (`10y`), `NIFTYBEES.NS`, `GOLDBEES.NS`, macro series, fundamentals + Day-1 PIT snapshot logger | **CORE LIVE** |
| `data_health.py` | `build_clean_price_matrix(max_ffill_days=2)`, `tradable_df`, `evaluate_data_freshness` (60s TTL fail-safe) | **CORE LIVE** |
| `nse_calendar.py` | NSE equity holiday calendar (`2020–2027`), `add_nse_trading_days` for staggered sleeves | **CORE LIVE** |
| `indicators.py` | Technical indicator frame (`SMA50/200`, `ATR14`, `RSI14`, `52W Hi/Lo`, `ret_63d/126d`) & chart setups | **CORE LIVE** |
| `ml_predictor.py` | 19-feature causal panel, 6-factor Canonical Prior, DM Crash Damper, 5-fold expanding Ridge+HGB, 6-Gauge Macro Regime, Pollet–Wilson `avg_corr_63` | **CORE LIVE** |
| `hrp_allocator.py` | López de Prado `HRP` (`126D Ledoit–Wolf`), `apply_concentration_caps`, budget-normalized L1 no-trade band | **CORE LIVE** |
| `money_signals.py` | Official NSE delivery (`sec_bhavdata_full_*.csv`), FII/DII OI, bulk/block deals, Balance-Sheet Shield, SEBI Delivery Drift Telemetry | **CORE LIVE** |
| `indian_costs.py` | Statutory NSE delivery fees, square-root ADV impact, `FifoTaxLedger`, Post-Budget 2024 reserve tax constants | **CORE LIVE** |
| `portfolio_backtester.py` | 4-tranche OOS simulator, `build_backtest_market_context()`, strictly `T-1` 20D delivery lookup, LTCG cliff guard, 2F + IIMA 4F alpha, reserve tax accounting | **CORE LIVE** |
| `execution_engine.py` | Live rebalance ticket generator, 4-sleeve schedule, peak-NAV `-10%/-6%` brake, `2%` ADV cap guard, `<8%` noise deadband, paper ledger commit/M2M | **CORE LIVE** |
| `stress_tester.py` | Multi-friction stress suite (reusing `market_ctx`), regime replay, tail risk & factor attribution | **CORE LIVE** |
| `main.py` | FastAPI server, `X-EquiLens-Token` / Bearer auth on all 9 state-mutating `POST` routes, CORS, single-flight caching, static frontend host | **CORE LIVE** |
| `shortlister.py`, `backtester.py`, `journal.py`, `alerts.py`, `catalyst_nlp.py`, `sec_edgar.py` | UI research shortlist (`unvalidated_research_composite`), single-stock next-open chart backtester, prediction journal, alerts, news NLP | **UI / AUXILIARY** |
| `pit_fundamentals.py` | Day-1 `pit_fundamentals_history` logger + NSE Bhavcopy & BSE/NSE XBRL `+2d` parsers | **PHASE-2 PIPELINE (LOGGING)** |
| `india_structural_signals.py` | Causal IIMA 4-Factor (`MKT/SMB/HML/WML`) alpha engine (live in reporting) + SEBI shareholding `+2d` parser | **REPORTING + FUTURE** |
| `forensic_quality_sleeve.py`, `multi_sleeve_allocator.py`, `convex_optimizer.py`, `india_liquidity_regime.py` | Parked `V8` toolkit: Indian Forensic/QMJ+Value, Carver FDM combiner, Boyd convex optimizer, RBI DBIE liquidity telemetry | **PARKED OPT-IN TOOLKIT** |

### 13.2 Complete API Route & Authentication Map (`backend/main.py`)
* **Read-Only (`GET`, no auth):** `/`, `/api/health`, `/api/shortlist`, `/api/stock/{symbol}`, `/api/model/metrics`, `/api/backtest/{symbol}`, `/api/portfolio/stress`, `/api/execution/ticket`, `/api/execution/paper/status`, `/api/money-signals`, `/api/journal`, `/api/alerts`.
* **Read-Only Custom Calculator (`POST`, intentionally NO auth — does not mutate DB):** `POST /api/execution/ticket` (accepts custom `current_holdings` JSON to preview a trade ticket).
* **State-Mutating (`POST`, requires `X-EquiLens-Token` or `Authorization: Bearer <SESSION_API_TOKEN>`, auto-injected into `<meta name="equilens-token">` on `/`):** `/api/sync`, `/api/tickers/add`, `/api/tickers/{symbol}/star`, `/api/execution/paper/commit`, `/api/execution/paper/reset` (also requires `{"confirm": true}`), `/api/money-signals/sync`, `/api/journal/log`, `/api/alerts/scan`, `/api/alerts/{alert_id}/ack`.

---

## 14. The 7 Non-Negotiable Anti-Leakage & Safety Invariants

Every future code change MUST preserve these 7 invariants (verified by `pytest`):
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

1. **[FIXED] T2T / ASM `100%` Delivery Surveillance Trap:** Added `deliv_20d_pct < 98.0` exclusion + `deliv_turnover_20d_cr >= 2.0` (`₹2 Cr/day` deliverable turnover floor). *(Confirmed `0.00%` T2T days in current liquid 130-stock universe; future-proofs down-cap expansion.)*
2. **[FIXED] Missing Indian LTCG Tax-Cliff Awareness:** Added `320–364d` profitable Top-60 holding guard (`+0.53%/yr` Honest PT, lower STCG, lower MaxDD).
3. **[FIXED] Post-Budget 2024 Reserve Tax Blind Spot:** Updated `indian_costs.py` & `portfolio_backtester.py` to track exact daily `liquid_gain_inr` and `gold_gain_inr`, conservatively taxing `LIQUIDCASE` & `<12m GOLDBEES` at `31.2%` slab rate and reporting Direct Arbitrage Fund (`20.8%` STCG) side-by-side.
4. **[FIXED] Unmonitored SEBI Delivery Regime Drift:** Wired `delivery_regime_telemetry` (`below_35_pct < 5.0%` warning; Nov 2026 live reading `5.3%` / `7/131` vs. `51.4%` in 2020) onto `/api/money-signals` and `/api/execution/ticket`, and pre-tested the `Hybrid (<35% OR Bottom-20%)` fallback.
5. **[ACCEPTED TRADE-OFF] Drawdown Brake Lag on V-Shaped Bottoms:** Accepted as essential multi-year bear-market insurance (`2008` `-17%` vs NIFTY `-60%`).
6. **[ACCEPTED TRADE-OFF] 4-Tranche Staggering Tax-Lot Fragmentation:** Trades `~1.05%/yr` PT and `4×` FIFO lots vs. 1-tranche in exchange for `-1.58%` shallower MaxDD and zero start-date timing luck (partially offset by the LTCG cliff guard).
7. **[ACCEPTED TRADE-OFF] Gold Bull-Market Tailwind (`2021–2026`):** INR gold added `+1.47%/yr` Pre-Tax / `+1.17%/yr` Honest Post-Tax during `2021–2026` (while slightly lifting daily vol, `SR -0.06`). Do not extrapolate this gold tailwind forever; budget live expectations at `16%–19%` net post-tax CAGR.

---

## 16. Standard Operating Commands & The 5-Gate Promotion Standard

### 16.1 Exact CLI Commands (Strictly Protecting Live `data/equilens_india.db`!)
```bash
# 1. Navigate to repository & activate virtualenv
cd /home/joe/.gemini/antigravity/scratch/equilens-india-quant
source .venv/bin/activate

# 2. Run the full automated regression & integrity test suite SAFELY on a temp copy
#    (NEVER point pytest directly at live data/equilens_india.db!)
cp data/equilens_india.db /tmp/equilens_test_copy.db
EQUILENS_DB_PATH=/tmp/equilens_test_copy.db pytest -v
rm -f /tmp/equilens_test_copy.db

# 3. Start the live EquiLens India V7.5-Core server
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 4. Run full overfitting / DSR / permutation / plateau sign-off audit on a temp copy
cp data/equilens_india.db /tmp/equilens_audit_copy.db
python scripts/audit_overfitting_v6.py --db /tmp/equilens_audit_copy.db --weighting hrp --perms 50
rm -f /tmp/equilens_audit_copy.db

# 5. Run 17 Real NSE Fallen Angels survivorship stress test (clones DB internally)
python scripts/test_survivorship_fallen_angels.py

# 6. Run multi-cycle (8.8Y + 19Y) validation harness (clones DB internally)
python scripts/validate_v7_multicycle.py --skip-20y-download
```

### 16.2 The 5-Gate Promotion Standard (Mandatory Before ANY Future Model Upgrade)
No future AI iteration may promote a strategy change onto `main` unless it passes **ALL FIVE GATES** on an isolated git branch/worktree (leaving live `paper_*` tables untouched):

1. **Gate 1 — Hardened Net Core OOS (`2021–2026`, `NIFTYBEES` TRI, Statutory Costs, FIFO Equity + Post-Budget 2024 Reserve Taxes):** Must beat `V7.5-Core` (`+29.97%` CAGR, `+25.67%` Honest PT `LIQUIDCASE`, `SR(6.2%) 1.84`, `MaxDD -10.23%`, `4F Alpha +10.96%`) on Honest Post-Tax CAGR, Excess Sharpe, AND 4-Factor Alpha without blowing up MaxDD or turnover.
2. **Gate 2 — Multi-Cycle & Survivorship Stress:** Must hold up on **17 Real NSE Fallen Angels** AND **blind unseen Indian stocks** AND multi-cycle (`2007–2026` / `2017–2026`) tests.
3. **Gate 3 — Multiple-Testing Rigor:** Must clear corrected Bailey–López de Prado **DSR (`RF = 6.2%`, full cumulative `N ≥ 100` documented trials including the 35+ Post-`V7.5` `H1–H6` frontier experiments)** ≥ `95%` and keep CPCV **PBO** low (`< 30%`). *(Note for future AI PIT data work: official NSE JSON endpoints `/api/corporate-announcements`, `/api/corporate-share-holdings-master`, `/api/corporates-pit`, and `/api/corporate-sast-reg29` return HTTP 200 with multi-year timestamped PIT history without cookies, whereas `/api/corporate-pledgedata` is latest-snapshot only and MUST NOT be backtested.)*
4. **Gate 4 — Parameter Plateau & Cost Stress:** Must win **at least `6/9`** cells across the `(K ∈ {8, 10, 12} × Rebalance ∈ {30D, 40D, 60D})` grid (including `3/3` or `2/3` at `40D`) AND survive `50 bps/side` (`100 bps` RT) cost stress.
5. **Gate 5 — Zero-Leakage, Live Parity & Full Suite:** `100%` of `pytest` tests passing (run on an isolated `EQUILENS_DB_PATH` copy), zero ETF leaks, strict backtest↔live parity, zero modification to `paper_*` tables, and `PLAYBOOK.md` updated cleanly.

---
*End of EquiLens India `V7.5-Core` Master Playbook. Stay disciplined, trust the process during flat months, never skip the evening sync window, and let compounding do the heavy lifting.*
