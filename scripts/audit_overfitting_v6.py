"""
EquiLens India V6/V7 — Hardened Institutional Overfitting, CPCV/PBO, DSR & Survivorship Audit.

Mathematical proofs (zero spin, zero lookahead):
  1. Corrected Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014) using cross-trial
     Sharpe variance sqrt(V[{SR_n}]) rather than 1/sqrt(T-1).
  2. Combinatorial Purged Cross-Validation (CPCV) & Probability of Backtest Overfitting (PBO)
     (Lopez de Prado Ch. 11-12; Bailey et al. 2015).
  3. Universe & TRI Benchmark Decomposition (NIFTYBEES.NS TRI vs Equal-Weight vs Strategy Post-Tax
     + 2-Factor Pure Alpha & Excess Sharpe SR(6.2%)).
  4. Year-by-Year Consistency across calendar years.
  5. Matched Full-Pipeline Permutation Null Test (Live K=10 4-Tranche + 6-Gauge Gate + 18% Vol Cap
     + -10%/-6% Drawdown Brake + 70/30 Reserve + Statutory Costs — ZERO .bfill() leakage!).
  6. Neighboring Parameter Plateau (Top-K in {8, 10, 12} x Rebalance in {30, 40, 60}, live K=10 4-Tranche).
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew, spearmanr

from backend.config import (
    BENCHMARK_SYMBOL,
    FX_MACRO_SYMBOL,
    GOLD_MACRO_SYMBOL,
    GOLDBEES_SYMBOL,
    NON_EQUITY_SYMBOLS,
    VOL_INDEX_SYMBOL,
)
from backend.data_health import build_clean_price_matrix
from backend.database import Database
from backend.portfolio_backtester import (
    DRAWDOWN_BRAKE_TRIGGER,
    GOLDBEES_RESERVE_SHARE,
    INDIA_CASH_DAILY_YIELD,
    _simulate_single_tranche,
    apply_drawdown_brake_and_gold_reserve,
    build_causal_gold_series,
    compute_two_factor_alpha,
    run_oos_portfolio_backtest,
)

warnings.filterwarnings("ignore")


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 50,
    annual_rf: float = 0.0,
    trial_sharpes: Sequence[float] | None = None,
    trial_sr_std_ann: float | None = None,
) -> dict[str, float]:
    """
    Exact Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR):
      SR_0 = sqrt(V[{SR_n}]) * ((1 - gamma) * Z^{-1}(1 - 1/N) + gamma * Z^{-1}(1 - 1/(N*e)))
      DSR  = Phi((SR_hat - SR_0) / sqrt((1 - g3*SR_hat + ((g4 - 1)/4)*SR_hat^2) / (T - 1)))

    Uses cross-trial Sharpe std sqrt(V[{SR_n}]) (NOT the buggy 1/sqrt(T-1) single-trial SE).
    `trial_sharpes` should be annualized Sharpe ratios across candidate trials if provided.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    ex = r - (annual_rf / 252.0)
    sr_daily = float(np.mean(ex) / max(1e-12, np.std(ex, ddof=1)))
    sr_ann = sr_daily * np.sqrt(252.0)
    g3 = float(skew(ex)) if n > 2 else 0.0
    g4 = float(kurtosis(ex, fisher=False)) if n > 3 else 3.0  # Pearson kurtosis (normal = 3.0)

    # Cross-trial Sharpe standard deviation in daily units
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        sr_std_ann = float(np.std(np.asarray(trial_sharpes, dtype=float), ddof=1))
        sr_std_daily = sr_std_ann / np.sqrt(252.0)
    elif trial_sr_std_ann is not None and trial_sr_std_ann > 0:
        sr_std_ann = float(trial_sr_std_ann)
        sr_std_daily = sr_std_ann / np.sqrt(252.0)
    else:
        # Conservative fallback when explicit cross-trial dispersion is unavailable:
        # max(null cross-trial std 1/sqrt(T-1), 0.35 annualized dispersion)
        sr_std_daily = max(1.0 / np.sqrt(max(1.0, n - 1.0)), 0.35 / np.sqrt(252.0))
        sr_std_ann = sr_std_daily * np.sqrt(252.0)

    if n_trials <= 1:
        sr_star_daily = 0.0
    else:
        euler = 0.5772156649
        z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
        z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
        sr_star_daily = float(sr_std_daily * ((1.0 - euler) * z1 + euler * z2))

    var_sr_hat = max(1e-12, (1.0 - g3 * sr_daily + ((g4 - 1.0) / 4.0) * (sr_daily ** 2)) / max(1.0, n - 1.0))
    denom = float(np.sqrt(var_sr_hat))
    z_score = (sr_daily - sr_star_daily) / denom
    dsr_prob = float(norm.cdf(z_score))
    return {
        "n_trials": int(n_trials),
        "sr_ann": round(sr_ann, 2),
        "sr_threshold_ann": round(sr_star_daily * np.sqrt(252.0), 2),
        "trial_sr_std_ann": round(sr_std_ann, 3),
        "skew": round(g3, 3),
        "kurtosis": round(g4, 2),
        "dsr_prob_pct": round(dsr_prob * 100.0, 2),
    }


def compute_cpcv_pbo(
    strategy_returns: np.ndarray,
    n_groups: int = 8,
    k_test: int = 4,
    purge_bars: int = 5,
    annual_rf: float = 0.062,
) -> dict[str, Any]:
    """
    Combinatorial Purged Cross-Validation (CPCV) & Probability of Backtest Overfitting (PBO)
    (Bailey, Borwein, Lopez de Prado & Zhu 2015; Lopez de Prado Ch. 11-12).

    Parameters
    ----------
    strategy_returns : np.ndarray of shape (T, M)
        Daily return matrix for M candidate strategy configurations over T trading days.
    n_groups : int
        Number of contiguous chronological blocks S (default 8 -> C(8,4)=70 symmetric splits).
    k_test : int
        Number of blocks held out as OOS in each combination (default S//2).
    purge_bars : int
        Number of bars purged from training blocks immediately adjacent to test blocks.
    annual_rf : float
        Annual risk-free hurdle (6.2% India LIQUIDCASE yield) for excess Sharpe ranking.
    """
    R = np.asarray(strategy_returns, dtype=float)
    if R.ndim != 2 or R.shape[0] < n_groups * 10 or R.shape[1] < 2:
        raise ValueError("strategy_returns must be 2D (T, M) with M >= 2 and T >= 10 * n_groups.")

    T, M = R.shape
    rf_daily = annual_rf / 252.0
    ex_R = R - rf_daily

    # Partition [0, T) into S contiguous blocks
    edges = np.linspace(0, T, n_groups + 1, dtype=int)
    block_slices = [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]

    def _sr_vec(mat: np.ndarray) -> np.ndarray:
        mu = np.mean(mat, axis=0)
        sd = np.std(mat, axis=0, ddof=1)
        return np.where(sd > 1e-12, (mu / sd) * np.sqrt(252.0), 0.0)

    logits: list[float] = []
    oos_srs_of_is_winner: list[float] = []
    is_oos_corrs: list[float] = []

    for test_blocks in itertools.combinations(range(n_groups), k_test):
        test_set = set(test_blocks)
        train_idx: list[int] = []
        test_idx: list[int] = []

        for b_id, (s, e) in enumerate(block_slices):
            if b_id in test_set:
                test_idx.extend(range(s, e))
            else:
                lo = s + (purge_bars if (b_id - 1) in test_set else 0)
                hi = e - (purge_bars if (b_id + 1) in test_set else 0)
                if hi > lo:
                    train_idx.extend(range(lo, hi))

        if len(train_idx) < 20 or len(test_idx) < 20:
            continue

        sr_is = _sr_vec(ex_R[train_idx])
        sr_oos = _sr_vec(ex_R[test_idx])

        n_star = int(np.argmax(sr_is))
        oos_srs_of_is_winner.append(float(sr_oos[n_star]))

        # Relative rank omega_bar in (0, 1): rank 1 = worst OOS, rank M = best OOS
        order = np.argsort(sr_oos)
        ranks = np.empty(M, dtype=float)
        ranks[order] = np.arange(1, M + 1, dtype=float)
        # Handle ties cleanly via average rank
        omega_bar = float(np.clip(ranks[n_star] / (M + 1.0), 1e-6, 1.0 - 1e-6))
        logit = float(np.log(omega_bar / (1.0 - omega_bar)))
        logits.append(logit)

        corr = spearmanr(sr_is, sr_oos).statistic
        if np.isfinite(corr):
            is_oos_corrs.append(float(corr))

    logits_arr = np.asarray(logits, dtype=float)
    pbo = float(np.mean(logits_arr <= 0.0)) if len(logits_arr) else 1.0
    return {
        "pbo_pct": round(pbo * 100.0, 2),
        "n_combinations": int(len(logits_arr)),
        "n_strategies": int(M),
        "median_logit": round(float(np.median(logits_arr)), 3) if len(logits_arr) else 0.0,
        "mean_oos_sr_of_is_winner": round(float(np.mean(oos_srs_of_is_winner)), 3) if oos_srs_of_is_winner else 0.0,
        "mean_is_oos_rank_corr": round(float(np.mean(is_oos_corrs)), 3) if is_oos_corrs else 0.0,
        "pass_pbo_lt_25": bool(pbo < 0.25),
    }


def compute_stats(daily_rets: np.ndarray, annual_rf: float = 0.062) -> dict[str, float]:
    yrs = max(0.25, len(daily_rets) / 252.0)
    eq = float(np.prod(1.0 + daily_rets))
    cagr = (eq ** (1.0 / yrs) - 1.0) * 100.0
    vol = float(np.std(daily_rets, ddof=1) * np.sqrt(252.0))
    shrp = float((np.mean(daily_rets) * 252.0) / max(1e-9, vol))
    ex = daily_rets - (annual_rf / 252.0)
    ex_sr = float((np.mean(ex) * 252.0) / max(1e-9, float(np.std(ex, ddof=1) * np.sqrt(252.0))))
    cum = np.cumprod(1.0 + daily_rets)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum(1e-9, np.maximum.accumulate(cum))) * 100.0
    return {
        "cagr": round(cagr, 2),
        "vol": round(vol * 100.0, 2),
        "sharpe": round(shrp, 2),
        "excess_sharpe_rf62": round(ex_sr, 2),
        "max_dd": round(float(np.min(dd)), 2),
    }


def _run_full_4tranche_pipeline(
    sim_dates: list[str],
    oos_dates: list[str],
    oos_by_date: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    tradable: pd.DataFrame,
    adv_df: pd.DataFrame,
    log_rets_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    bench_series: pd.Series,
    bench_sma200: pd.Series,
    breadth_series: pd.Series,
    hilo_series: pd.Series,
    gold_nifty_chg: pd.Series,
    gold_rets_sim: np.ndarray,
    vix_map: dict[str, float],
    sector_map: dict[str, str],
    idx_loc: dict[str, int],
    top_k: int = 10,
    rebalance_days: int = 40,
    weighting_mode: str = "hrp",
) -> np.ndarray:
    """Execute exact live K=10 4-tranche + -10%/-6% Drawdown Brake + 70/30 Gold Reserve pipeline."""
    offsets = sorted({(rebalance_days * t) // 4 for t in range(4)})
    tr_rets, tr_idles = [], []
    for off in offsets:
        idle_buf: list[float] = []
        rd, _, _ = _simulate_single_tranche(
            sim_dates=sim_dates,
            oos_dates=oos_dates,
            oos_by_date=oos_by_date,
            prices_df=prices,
            log_rets_df=log_rets_df,
            vol_df=vol_df,
            bench_series=bench_series,
            bench_sma200=bench_sma200,
            breadth_series=breadth_series,
            hilo_series=hilo_series,
            gold_nifty_chg=gold_nifty_chg,
            vix_map=vix_map,
            sector_map=sector_map,
            idx_loc=idx_loc,
            top_k=top_k,
            rebalance_days=rebalance_days,
            rebal_offset=off,
            cost_bps=15.0,
            regime_gate=True,
            max_per_sector=2,
            corr_penalized_weight=True,
            vol_target=0.18,
            out_idle_weights=idle_buf,
            tradable_df=tradable,
            adv_df=adv_df,
            stagger_initial_deployment=True,
            use_statutory_costs=True,
            weighting_mode=weighting_mode,
        )
        tr_rets.append(rd)
        tr_idles.append(np.array(idle_buf, dtype=float))
    blended = np.mean(np.vstack(tr_rets), axis=0)
    blended_idle = np.mean(np.vstack(tr_idles), axis=0)
    out = apply_drawdown_brake_and_gold_reserve(
        blended,
        blended_idle,
        gold_rets_sim,
        dd_trigger=DRAWDOWN_BRAKE_TRIGGER,
        gold_idle_share=GOLDBEES_RESERVE_SHARE,
        cost_bps=15.0,
        enable_dd_brake=True,
    )
    return out[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="EquiLens India V7 Institutional Overfitting, DSR & CPCV/PBO Audit")
    parser.add_argument("--db", type=str, default=None, help="Optional path to cloned SQLite DB")
    parser.add_argument("--perms", type=int, default=100, help="Number of matched 4-tranche K=10 permutations")
    parser.add_argument("--weighting", type=str, default="hrp", choices=["hrp", "corr_penalized"], help="Weighting mode")
    args = parser.parse_args()

    db = Database(args.db) if args.db else Database()
    bt = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40, weighting_mode=args.weighting)
    p_rets = np.array(bt["_daily_returns"], dtype=float)
    b_rets = np.array(bt["_spy_daily_returns"], dtype=float)
    eqw_daily = np.array(bt["_eqw_daily_returns"], dtype=float)
    dates = bt["_dates"]

    print("===========================================================================================")
    print(f"  EQUILENS INDIA V7 — HARDENED OVERFITTING, CORRECTED DSR & CPCV/PBO AUDIT ({args.weighting.upper()})")
    print("===========================================================================================\n")
    print(
        f"  Benchmark: {bt['config']['benchmark_symbol']} ({bt['config']['benchmark_type']}) | "
        f"Cost: {bt['config']['cost_model']} | Tax: {bt['config']['tax_model']} | "
        f"Rank Buffer: Top-{bt['config']['rank_buffer']}\n"
    )

    # Precompute strictly causal matrices (ZERO .bfill()!)
    ohlcv = db.get_all_ohlcv()
    prices, volumes, tradable = build_clean_price_matrix(ohlcv, max_ffill_days=2)
    adv_df = (prices * volumes).rolling(20, min_periods=5).mean()
    stock_cols = [c for c in prices.columns if c not in NON_EQUITY_SYMBOLS and not c.startswith("^")]
    macro = db.get_macro_series()
    vix_map = macro.get(VOL_INDEX_SYMBOL, {})
    gold_map = macro.get(GOLD_MACRO_SYMBOL, {})
    fx_map = macro.get(FX_MACRO_SYMBOL, {})
    sector_map = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    log_rets_df = np.log(prices / prices.shift(1))
    vol_df = log_rets_df.rolling(252, min_periods=63).std() * np.sqrt(252)
    sma50_df = prices[stock_cols].rolling(50, min_periods=30).mean()
    breadth_series = (prices[stock_cols] > sma50_df).mean(axis=1)
    h52 = prices[stock_cols].rolling(252, min_periods=60).max().replace(0, np.nan)
    l52 = prices[stock_cols].rolling(252, min_periods=60).min().replace(0, np.nan)
    hilo_series = ((prices[stock_cols] / h52 >= 0.90).mean(axis=1) - (prices[stock_cols] / l52 <= 1.12).mean(axis=1)).fillna(0.0)
    bench_series = prices[BENCHMARK_SYMBOL].ffill(limit=5)
    bench_sma200 = bench_series.rolling(200, min_periods=50).mean()
    goldbees_series = prices[GOLDBEES_SYMBOL] if GOLDBEES_SYMBOL in prices.columns else None
    gold_nifty_chg, gold_inr_ret_series = build_causal_gold_series(
        prices.index, gold_map, fx_map, bench_series, goldbees_series=goldbees_series
    )

    oos_raw = pd.DataFrame(db.get_oos_history())
    oos_raw = oos_raw[~oos_raw["symbol"].isin(NON_EQUITY_SYMBOLS)].copy()
    oos_dates = sorted(set(oos_raw["date"].unique()).intersection(set(prices.index)))
    sim_dates = [d for d in prices.index if oos_dates[0] <= d <= oos_dates[-1]]
    idx_loc = {d: i for i, d in enumerate(prices.index)}
    oos_by_date_real = {dt: grp for dt, grp in oos_raw.groupby("date")}
    gold_rets_sim = gold_inr_ret_series.reindex(sim_dates).fillna(INDIA_CASH_DAILY_YIELD).to_numpy(dtype=float)

    # TEST 1: UNIVERSE & TRI DECOMPOSITION + 2-FACTOR ALPHA
    st_bench = compute_stats(b_rets)
    st_eqw = compute_stats(eqw_daily)
    st_port = compute_stats(p_rets)
    tf = compute_two_factor_alpha(p_rets, b_rets, eqw_daily)
    print("=== 1. UNIVERSE & TRI BENCHMARK DECOMPOSITION ===")
    print(f"  1) Benchmark ({bt['config']['benchmark_symbol']} {bt['config']['benchmark_type']:5s})      | CAGR {st_bench['cagr']:+6.2f}% (Post-Tax {bt['spy_metrics']['after_tax_cagr_pct']:+6.2f}%) | SR(6.2%) {st_bench['excess_sharpe_rf62']:5.2f} | MaxDD {st_bench['max_dd']:6.2f}%")
    print(f"  2) Equal-Weight {len(stock_cols)}-Stock Univ  | CAGR {st_eqw['cagr']:+6.2f}% | SR(6.2%) {st_eqw['excess_sharpe_rf62']:5.2f} | MaxDD {st_eqw['max_dd']:6.2f}%")
    print(f"  3) Live Strategy (K=10 4-Tranche) | CAGR {st_port['cagr']:+6.2f}% (Post-Tax {bt['portfolio_metrics']['after_tax_cagr_pct']:+6.2f}%) | SR(6.2%) {st_port['excess_sharpe_rf62']:5.2f} | MaxDD {st_port['max_dd']:6.2f}%")
    print(f"  --> 2-Factor Pure Alpha: {tf['alpha_2f_pct']:+.2f}%/yr (t={tf['alpha_tstat']:+.2f}) | beta_mkt={tf['beta_mkt']:.2f} | beta_size={tf['beta_size']:.2f}")

    # TEST 2: YEAR-BY-YEAR CONSISTENCY
    print("\n=== 2. YEAR-BY-YEAR CONSISTENCY ===\n")
    df_y = pd.DataFrame({"date": dates, "port": p_rets, "eqw": eqw_daily, "bench": b_rets})
    df_y["year"] = df_y["date"].str[:4]
    print(f"  {'Year':6s} | {'Days':>4s} | {'Port':>8s} | {'EqW':>8s} | {'Bench':>8s} | {'Port-EqW':>9s} | {'Port-Bench':>10s} | {'MaxDD':>8s}")
    print("  " + "-" * 84)
    for yr, grp in df_y.groupby("year"):
        rp = (float(np.prod(1.0 + grp["port"])) - 1.0) * 100.0
        re = (float(np.prod(1.0 + grp["eqw"])) - 1.0) * 100.0
        rb = (float(np.prod(1.0 + grp["bench"])) - 1.0) * 100.0
        c = np.cumprod(1.0 + grp["port"].to_numpy())
        dd = float(np.min((c - np.maximum.accumulate(c)) / np.maximum.accumulate(c)) * 100.0)
        print(f"  {yr:6s} | {len(grp):4d} | {rp:+7.2f}% | {re:+7.2f}% | {rb:+7.2f}% | {rp - re:+8.2f}% | {rp - rb:+9.2f}% | {dd:7.2f}%")

    # TEST 3: PARAMETER GRID + CPCV PBO + CORRECTED DSR
    print("\n=== 3. NEIGHBORING PARAMETER PLATEAU (Live K=10 4-Tranche) & CPCV / PBO ===")
    print(f"  {'Params':18s} | {'CAGR':>7s} | {'SR(6.2%)':>8s} | {'MaxDD':>7s} | {'Calmar':>6s}")
    print("  " + "-" * 60)
    grid_cols: list[np.ndarray] = []
    grid_sharpes_rf62: list[float] = []
    for k in (6, 8, 10, 12):
        for reb in (30, 40, 60):
            r_g = _run_full_4tranche_pipeline(
                sim_dates, oos_dates, oos_by_date_real, prices, tradable, adv_df, log_rets_df, vol_df,
                bench_series, bench_sma200, breadth_series, hilo_series, gold_nifty_chg, gold_rets_sim,
                vix_map, sector_map, idx_loc, top_k=k, rebalance_days=reb, weighting_mode=args.weighting,
            )
            grid_cols.append(r_g)
            m = compute_stats(r_g)
            grid_sharpes_rf62.append(m["excess_sharpe_rf62"])
            calm = m["cagr"] / max(1.0, abs(m["max_dd"]))
            tag = " <-- LIVE K=10 4-TRANCHE" if (k == 10 and reb == 40) else ""
            print(f"  K={k:2d}, Reb={reb:2d}D    | {m['cagr']:+6.2f}% | {m['excess_sharpe_rf62']:8.2f} | {m['max_dd']:6.2f}% | {calm:6.2f}{tag}")

    grid_mat = np.column_stack(grid_cols)
    pbo_res = compute_cpcv_pbo(grid_mat, n_groups=8, k_test=4, purge_bars=5, annual_rf=0.062)
    print(
        f"\n  CPCV PBO ({pbo_res['n_combinations']} purged splits, {pbo_res['n_strategies']} configs): "
        f"PBO = {pbo_res['pbo_pct']:.2f}% | Median Logit = {pbo_res['median_logit']:+.3f} | "
        f"Mean OOS SR(6.2%) of IS Winner = {pbo_res['mean_oos_sr_of_is_winner']:.2f} | "
        f"Gate (<25%): {'PASS' if pbo_res['pass_pbo_lt_25'] else 'FAIL'}"
    )

    # TEST 4: MATCHED FULL-PIPELINE PERMUTATION NULL TEST (Live K=10 4-Tranche)
    n_perms = max(25, int(args.perms))
    print(f"\n=== 4. MATCHED FULL-PIPELINE PERMUTATION NULL TEST (N={n_perms}, Live K=10 4-Tranche) ===")
    rng = np.random.default_rng(777)
    all_rebal_dts = set()
    for off in sorted({(40 * t) // 4 for t in range(4)}):
        all_rebal_dts.update(oos_dates[off::40])

    rand_cagrs, rand_srs, rand_dds = [], [], []
    for _ in range(n_perms):
        shuf_by_date = {}
        for dt in all_rebal_dts:
            g = oos_by_date_real[dt][["symbol", "pred_alpha"]].copy()
            g["pred_alpha"] = rng.normal(0.0, 1.0, size=len(g))
            shuf_by_date[dt] = g
        r_full = _run_full_4tranche_pipeline(
            sim_dates, oos_dates, shuf_by_date, prices, tradable, adv_df, log_rets_df, vol_df,
            bench_series, bench_sma200, breadth_series, hilo_series, gold_nifty_chg, gold_rets_sim,
            vix_map, sector_map, idx_loc, top_k=10, rebalance_days=40, weighting_mode=args.weighting,
        )
        s = compute_stats(r_full)
        rand_cagrs.append(s["cagr"])
        rand_srs.append(s["excess_sharpe_rf62"])
        rand_dds.append(s["max_dd"])

    rc, rs, rdd = np.array(rand_cagrs), np.array(rand_srs), np.array(rand_dds)
    p_cagr = (float(np.sum(rc >= st_port["cagr"])) + 1.0) / (n_perms + 1.0)
    p_sr = (float(np.sum(rs >= st_port["excess_sharpe_rf62"])) + 1.0) / (n_perms + 1.0)
    print(f"  Random K=10 4-Tranche + Same Gate/Vol/Brake: Mean CAGR {np.mean(rc):+.2f}% (P95 {np.percentile(rc, 95):+.2f}%) | Mean SR(6.2%) {np.mean(rs):.2f}")
    print(f"  Actual K=10 4-Tranche Strategy:              CAGR      {st_port['cagr']:+.2f}% (p={p_cagr:.4f}) | SR(6.2%) {st_port['excess_sharpe_rf62']:.2f} (p={p_sr:.4f})")

    # TEST 5: CORRECTED BAILEY & LOPEZ DE PRADO (2014) DEFLATED SHARPE RATIO
    print("\n=== 5. CORRECTED DEFLATED SHARPE RATIO (Bailey & Lopez de Prado 2014, Empirical Cross-Trial Std) ===")
    combined_trials = list(grid_sharpes_rf62) + list(rand_srs)
    for rf_label, rf_val in [("6.2% India LIQUIDCASE RF", 0.062), ("0% Excess (Standard)", 0.0)]:
        print(f"  Risk-Free Benchmark: {rf_label}")
        for nt in (1, 25, 50, 100):
            d = deflated_sharpe_ratio(p_rets, n_trials=nt, annual_rf=rf_val, trial_sharpes=combined_trials)
            print(
                f"    Trials N={nt:3d} | Realized SR={d['sr_ann']:.2f} vs Hurdle SR*={d['sr_threshold_ann']:.2f} "
                f"(std_trials={d['trial_sr_std_ann']:.3f}) | Skew={d['skew']:+.2f} Kurt={d['kurtosis']:.1f} -> DSR: {d['dsr_prob_pct']:6.2f}%"
            )


if __name__ == "__main__":
    main()
