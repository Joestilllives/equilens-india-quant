"""
EquiLens India V6 — Brutal Institutional Overfitting & Survivorship Audit:
Answers "Are we sure we're not overfitting?" with 6 hard mathematical proofs (zero spin):
  1. Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014) adjusted for N = 1, 50, 100, 500 trials
  2. Universe Decomposition: NIFTY 50 vs. 131-Stock Equal-Weight Universe (measures survivorship/midcap bias!)
  3. Year-by-Year Consistency (2021 H2, 2022, 2023, 2024, 2025, 2026) — did 1 lucky year carry everything?
  4. Permutation Null Test (100 random stock-selection portfolios with IDENTICAL universe, 6-Gauge Gate & 18% Vol Target)
     -> Exact p-value of pure V6 stock-selection skill!
  5. Synthetic Indian Survivorship Blowup Stress (injecting 1, 2, 3 catastrophic -70% stock blowups/yr)
  6. Neighboring Parameter Grid (Top-6/8/10/12 x Rebalance 30/40/60D) — smooth plateau vs sharp cliff?
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from backend.config import BENCHMARK_SYMBOL, GOLD_MACRO_SYMBOL, VOL_INDEX_SYMBOL
from backend.database import Database
from backend.portfolio_backtester import _simulate_single_tranche, run_oos_portfolio_backtest

warnings.filterwarnings("ignore")


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int = 50, annual_rf: float = 0.0) -> dict[str, float]:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR) accounting for skew, kurtosis & N trials."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    ex = r - (annual_rf / 252.0)
    sr_daily = float(np.mean(ex) / max(1e-12, np.std(ex, ddof=1)))
    sr_ann = sr_daily * np.sqrt(252.0)
    g3 = float(skew(ex))
    g4 = float(kurtosis(ex, fisher=False))  # Pearson kurtosis (normal = 3.0)

    if n_trials <= 1:
        sr_star_daily = 0.0
    else:
        euler = 0.5772156649
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_star_daily = float(((1.0 - euler) * z1 + euler * z2) / np.sqrt(max(1.0, n - 1.0)))

    denom = np.sqrt(max(1e-12, (1.0 - g3 * sr_daily + ((g4 - 1.0) / 4.0) * (sr_daily ** 2)) / max(1.0, n - 1.0)))
    z_score = (sr_daily - sr_star_daily) / denom
    dsr_prob = float(norm.cdf(z_score))
    return {
        "n_trials": n_trials,
        "sr_ann": round(sr_ann, 2),
        "sr_threshold_ann": round(sr_star_daily * np.sqrt(252.0), 2),
        "skew": round(g3, 3),
        "kurtosis": round(g4, 2),
        "dsr_prob_pct": round(dsr_prob * 100.0, 2),
    }


def compute_stats(daily_rets: np.ndarray) -> dict[str, float]:
    yrs = max(0.25, len(daily_rets) / 252.0)
    eq = float(np.prod(1.0 + daily_rets))
    cagr = (eq ** (1.0 / yrs) - 1.0) * 100.0
    vol = float(np.std(daily_rets, ddof=1) * np.sqrt(252.0))
    shrp = float((np.mean(daily_rets) * 252.0) / max(1e-9, vol))
    cum = np.cumprod(1.0 + daily_rets)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum(1e-9, np.maximum.accumulate(cum))) * 100.0
    return {"cagr": round(cagr, 2), "vol": round(vol * 100.0, 2), "sharpe": round(shrp, 2), "max_dd": round(float(np.min(dd)), 2)}


def main() -> None:
    db = Database()
    bt_v6 = run_oos_portfolio_backtest(db)
    p_rets = np.array(bt_v6["_daily_returns"], dtype=float)
    b_rets = np.array(bt_v6["_spy_daily_returns"], dtype=float)
    dates = bt_v6["_dates"]

    print("========================================================================================")
    print("   EQUILENS INDIA V6 — BRUTAL INSTITUTIONAL OVERFITTING & SURVIVORSHIP AUDIT")
    print("========================================================================================\n")

    # TEST 1: DEFLATED SHARPE RATIO (DSR)
    print("=== 1. DEFLATED SHARPE RATIO (DSR — Bailey & Lopez de Prado, Multiple-Testing Adjusted) ===")
    print(f"    (Sample: T = {len(p_rets)} OOS daily returns | Observed V6 Sharpe = {bt_v6['portfolio_metrics']['sharpe']:.2f})\n")
    for rf_label, rf_val in [("0% Excess (Standard)", 0.0), ("6.2% India LiquidBees RF", 0.062)]:
        print(f"  Risk-Free Benchmark: {rf_label}")
        for nt in (1, 50, 100, 500):
            d = deflated_sharpe_ratio(p_rets, n_trials=nt, annual_rf=rf_val)
            print(
                f"    Trials N={nt:3d} | Realized SR={d['sr_ann']:.2f} vs Hurdle SR*={d['sr_threshold_ann']:.2f} "
                f"| Skew={d['skew']:+.2f} Kurt={d['kurtosis']:.1f} -> DSR Confidence: {d['dsr_prob_pct']:6.2f}%"
            )

    # TEST 2: SURVIVORSHIP / MIDCAP UNIVERSE DECOMPOSITION
    print("\n=== 2. UNIVERSE & SURVIVORSHIP DECOMPOSITION (How Much Is Just Being in Today's 131 Stocks?) ===")
    ohlcv = db.get_all_ohlcv()
    prices = pd.DataFrame({s: pd.Series({b["date"]: float(b["close"]) for b in bars}) for s, bars in ohlcv.items()}).sort_index().ffill()
    stock_cols = [c for c in prices.columns if c != BENCHMARK_SYMBOL]
    eqw_daily = prices[stock_cols].pct_change().mean(axis=1).reindex(dates).fillna(0.0).to_numpy(dtype=float)

    st_nifty = compute_stats(b_rets)
    st_eqw = compute_stats(eqw_daily)
    st_v6 = compute_stats(p_rets)
    print(f"  1) NIFTY 50 Index (Benchmark)                    | CAGR {st_nifty['cagr']:+6.2f}% | Vol {st_nifty['vol']:5.2f}% | Sharpe {st_nifty['sharpe']:4.2f} | MaxDD {st_nifty['max_dd']:6.2f}%")
    print(f"  2) Equal-Weight ALL 131 NSE Stocks (No Strategy) | CAGR {st_eqw['cagr']:+6.2f}% | Vol {st_eqw['vol']:5.2f}% | Sharpe {st_eqw['sharpe']:4.2f} | MaxDD {st_eqw['max_dd']:6.2f}%")
    print(f"  3) V6 Strategy (Live Production, 15 bps STT)     | CAGR {st_v6['cagr']:+6.2f}% | Vol {st_v6['vol']:5.2f}% | Sharpe {st_v6['sharpe']:4.2f} | MaxDD {st_v6['max_dd']:6.2f}%")
    print(f"  --> Universe / Midcap + Survivorship Tilt (2 - 1): {st_eqw['cagr'] - st_nifty['cagr']:+.2f}%/yr")
    print(f"  --> TRUE Pure V6 Alpha Above Own 131-Stock Univ (3 - 2): {st_v6['cagr'] - st_eqw['cagr']:+.2f}%/yr | Sharpe Gain: {st_v6['sharpe'] - st_eqw['sharpe']:+.2f} | MaxDD Reduction: {st_v6['max_dd'] - st_eqw['max_dd']:+.2f}%")

    # TEST 3: YEAR-BY-YEAR CONSISTENCY
    print("\n=== 3. YEAR-BY-YEAR CONSISTENCY (V6 vs. 131-Stock Equal-Weight vs. NIFTY 50) ===\n")
    df_y = pd.DataFrame({"date": dates, "v6": p_rets, "eqw": eqw_daily, "nifty": b_rets})
    df_y["year"] = df_y["date"].str[:4]
    print(f"  {'Year':6s} | {'Days':>4s} | {'V6 Ret':>8s} | {'131-EqW':>8s} | {'NIFTY 50':>8s} | {'V6 - EqW':>9s} | {'V6 - NIFTY':>10s} | {'V6 MaxDD':>8s}")
    print("  " + "-" * 82)
    beat_eqw_yrs, beat_nifty_yrs = 0, 0
    for yr, grp in df_y.groupby("year"):
        r_v6 = (float(np.prod(1.0 + grp["v6"])) - 1.0) * 100.0
        r_eq = (float(np.prod(1.0 + grp["eqw"])) - 1.0) * 100.0
        r_ni = (float(np.prod(1.0 + grp["nifty"])) - 1.0) * 100.0
        c = np.cumprod(1.0 + grp["v6"].to_numpy())
        dd = float(np.min((c - np.maximum.accumulate(c)) / np.maximum.accumulate(c)) * 100.0)
        if r_v6 > r_eq:
            beat_eqw_yrs += 1
        if r_v6 > r_ni:
            beat_nifty_yrs += 1
        print(f"  {yr:6s} | {len(grp):4d} | {r_v6:+7.2f}% | {r_eq:+7.2f}% | {r_ni:+7.2f}% | {r_v6 - r_eq:+8.2f}% | {r_v6 - r_ni:+9.2f}% | {dd:7.2f}%")
    n_yrs = df_y["year"].nunique()
    print(f"\n  Consistency: V6 beat NIFTY 50 in {beat_nifty_yrs}/{n_yrs} calendar years | beat its OWN 131-stock universe in {beat_eqw_yrs}/{n_yrs} years.")

    # Precompute matrices once in memory for fast Test 4 & Test 6
    macro = db.get_macro_series()
    vix_map = macro.get(VOL_INDEX_SYMBOL, {})
    gold_map = macro.get(GOLD_MACRO_SYMBOL, {})
    sector_map = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    log_rets_df = np.log(prices / prices.shift(1))
    vol_df = log_rets_df.rolling(252, min_periods=63).std() * np.sqrt(252)
    sma50_df = prices[stock_cols].rolling(50, min_periods=30).mean()
    breadth_series = (prices[stock_cols] > sma50_df).mean(axis=1)
    h52 = prices[stock_cols].rolling(252, min_periods=60).max().replace(0, np.nan)
    l52 = prices[stock_cols].rolling(252, min_periods=60).min().replace(0, np.nan)
    hilo_series = ((prices[stock_cols] / h52 >= 0.90).mean(axis=1) - (prices[stock_cols] / l52 <= 1.12).mean(axis=1)).fillna(0.0)
    bench_series = prices[BENCHMARK_SYMBOL]
    bench_sma200 = bench_series.rolling(200, min_periods=50).mean()
    gold_s = pd.Series(gold_map, dtype=float).reindex(prices.index).ffill().bfill()
    gold_nifty_chg = (gold_s / bench_series.replace(0, np.nan)).pct_change(63).fillna(0.0)

    oos_raw = pd.DataFrame(db.get_oos_history())
    oos_dates = sorted(set(oos_raw["date"].unique()).intersection(set(prices.index)))
    sim_dates = [d for d in prices.index if oos_dates[0] <= d <= oos_dates[-1]]
    idx_loc = {d: i for i, d in enumerate(prices.index)}
    oos_by_date_real = {dt: grp for dt, grp in oos_raw.groupby("date")}

    # TEST 4: PERMUTATION NULL TEST (100 Random Stock-Selection Portfolios)
    print("\n=== 4. PERMUTATION NULL TEST (100 Random Stock-Selection Portfolios Using IDENTICAL Risk Gate & Universe) ===")
    rng = np.random.default_rng(777)
    rand_cagrs, rand_sharpes, rand_dds = [], [], []
    rebal_dts_0 = set(oos_dates[0::40])
    rebal_dts_0.add(oos_dates[0])
    for _ in range(100):
        shuf_by_date = {}
        for dt in rebal_dts_0:
            g = oos_by_date_real[dt][["symbol", "pred_alpha"]].copy()
            g["pred_alpha"] = rng.normal(0.0, 1.0, size=len(g))
            shuf_by_date[dt] = g
        r_daily, _, _ = _simulate_single_tranche(
            sim_dates=sim_dates, oos_dates=oos_dates, oos_by_date=shuf_by_date,
            prices_df=prices, log_rets_df=log_rets_df, vol_df=vol_df,
            bench_series=bench_series, bench_sma200=bench_sma200,
            breadth_series=breadth_series, hilo_series=hilo_series,
            gold_nifty_chg=gold_nifty_chg, vix_map=vix_map, sector_map=sector_map, idx_loc=idx_loc,
            top_k=8, rebalance_days=40, rebal_offset=0, cost_bps=15.0,
            regime_gate=True, max_per_sector=2, corr_penalized_weight=True, vol_target=0.18,
        )
        s = compute_stats(r_daily[1:])
        rand_cagrs.append(s["cagr"]); rand_sharpes.append(s["sharpe"]); rand_dds.append(s["max_dd"])

    rc, rs, rdd = np.array(rand_cagrs), np.array(rand_sharpes), np.array(rand_dds)
    print(f"  Random Stock Selection + SAME V6 Risk Gate/VolTgt: Mean CAGR {np.mean(rc):+.2f}% (P95 {np.percentile(rc, 95):+.2f}%) | Mean Sharpe {np.mean(rs):.2f} (P95 {np.percentile(rs, 95):.2f}) | Mean DD {np.mean(rdd):.2f}%")
    print(f"  Actual V6 Stock Selection:                         CAGR      {st_v6['cagr']:+.2f}% (p-value = {float(np.mean(rc >= st_v6['cagr'])):.4f}) | Sharpe      {st_v6['sharpe']:.2f} (p-value = {float(np.mean(rs >= st_v6['sharpe'])):.4f}) | MaxDD   {st_v6['max_dd']:.2f}%")

    # TEST 5: SURVIVORSHIP-BIAS BLOWUP SHOCK TEST
    print("\n=== 5. SURVIVORSHIP-BIAS BLOWUP SHOCK TEST (Injecting Random -70% Single-Stock Blowups into 8-Stock Portfolio) ===")
    rng2 = np.random.default_rng(999)
    for blows_per_yr in (0, 1, 2, 3):
        sim_cagrs, sim_sharpes, sim_dds = [], [], []
        n_shocks = int(round(blows_per_yr * (len(p_rets) / 252.0)))
        for _ in range(500):
            r_sim = p_rets.copy()
            if n_shocks > 0:
                shock_days = rng2.choice(len(r_sim), size=n_shocks, replace=False)
                r_sim[shock_days] -= 0.0875
            s = compute_stats(r_sim)
            sim_cagrs.append(s["cagr"]); sim_sharpes.append(s["sharpe"]); sim_dds.append(s["max_dd"])
        print(
            f"  {blows_per_yr} Blowup(s)/yr (-70% stock crash) | Med CAGR: {np.median(sim_cagrs):+6.2f}% "
            f"| Med Sharpe: {np.median(sim_sharpes):.2f} | Med MaxDD: {np.median(sim_dds):6.2f}% "
            f"| Excess vs NIFTY: {np.median(sim_cagrs) - st_nifty['cagr']:+6.2f}%/yr"
        )

    # TEST 6: NEIGHBORING PARAMETER PLATEAU CHECK
    print("\n=== 6. NEIGHBORING PARAMETER PLATEAU (4-Tranche Top-K x Rebalance Horizon | 15 bps STT) ===\n")
    print(f"  {'Params':18s} | {'CAGR':>7s} | {'Sharpe':>6s} | {'MaxDD':>7s} | {'Calmar':>6s}")
    print("  " + "-" * 56)
    for k in (6, 8, 10, 12):
        for reb in (30, 40, 60):
            offsets = sorted({(reb * t) // 4 for t in range(4)})
            tr_arrs = []
            for off in offsets:
                rd, _, _ = _simulate_single_tranche(
                    sim_dates=sim_dates, oos_dates=oos_dates, oos_by_date=oos_by_date_real,
                    prices_df=prices, log_rets_df=log_rets_df, vol_df=vol_df,
                    bench_series=bench_series, bench_sma200=bench_sma200,
                    breadth_series=breadth_series, hilo_series=hilo_series,
                    gold_nifty_chg=gold_nifty_chg, vix_map=vix_map, sector_map=sector_map, idx_loc=idx_loc,
                    top_k=k, rebalance_days=reb, rebal_offset=off, cost_bps=15.0,
                    regime_gate=True, max_per_sector=2, corr_penalized_weight=True, vol_target=0.18,
                )
                tr_arrs.append(rd)
            bl = np.mean(np.vstack(tr_arrs), axis=0)[1:]
            m = compute_stats(bl)
            calm = m["cagr"] / max(1.0, abs(m["max_dd"]))
            tag = " <-- LIVE V6" if (k == 8 and reb == 40) else ""
            print(f"  K={k:2d}, Reb={reb:2d}D    | {m['cagr']:+6.2f}% | {m['sharpe']:6.2f} | {m['max_dd']:6.2f}% | {calm:6.2f}{tag}")


if __name__ == "__main__":
    main()
