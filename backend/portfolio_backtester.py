"""
Portfolio-level strictly out-of-sample Shortlist Backtester (Promoted V6 Institutional India Architecture):
  - 4-Tranche Staggered Overlapping 40-Day Sleeves (25% capital per sleeve, staggered 10 trading days apart
    to eliminate rebalance-date timing luck with zero extra turnover per rupee)
  - Top-8 NSE Portfolio, 2xK Rank-Buffer Hysteresis, Max 2 per NSE Sector
  - Correlation-Penalized Risk Weighting: w_i proportional to 1 / (vol_i * (1 + avg_pos_corr_i))
  - Ex-Ante 18% Annualized Portfolio Volatility Target (trailing 60D covariance)
    + Daily 6-Gauge India Sentiment & Intermarket Macro Risk Gate
    (India VIX, NIFTY 200-DMA, NSE Breadth, NSE 52W Hi-Lo Sentiment, Gold/NIFTY Fear Gauge)
  - Indian LiquidBees Cash Yield (~6.2% p.a.) & 15 bps Indian STT + Brokerage + Slippage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import BENCHMARK_SYMBOL, DEFAULT_COST_BPS, GOLD_MACRO_SYMBOL, VOL_INDEX_SYMBOL
from .database import Database

INDIA_CASH_DAILY_YIELD = (1.062 ** (1.0 / 252.0)) - 1.0


def _simulate_single_tranche(
    sim_dates: list[str],
    oos_dates: list[str],
    oos_by_date: dict[str, pd.DataFrame],
    prices_df: pd.DataFrame,
    log_rets_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    bench_series: pd.Series,
    bench_sma200: pd.Series,
    breadth_series: pd.Series,
    hilo_series: pd.Series,
    gold_nifty_chg: pd.Series,
    vix_map: dict[str, float],
    sector_map: dict[str, str],
    idx_loc: dict[str, int],
    top_k: int,
    rebalance_days: int,
    rebal_offset: int,
    cost_bps: float,
    regime_gate: bool,
    max_per_sector: int,
    corr_penalized_weight: bool,
    vol_target: float | None,
) -> tuple[np.ndarray, float, int]:
    """Run one strictly causal overlapping sleeve starting at `rebal_offset`."""
    rebalance_set = set(oos_dates[rebal_offset::rebalance_days])
    # Always invest on day 0 as well so tranche doesn't sit idle before its first staggered slot
    if oos_dates:
        rebalance_set.add(oos_dates[0])

    rank_buffer = top_k * 2
    fee_rate = cost_bps / 10000.0
    current_weights: dict[str, float] = {}
    total_turnover = 0.0
    rebalance_count = 0
    daily_rets = np.zeros(len(sim_dates), dtype=float)

    for i, dt in enumerate(sim_dates):
        if i > 0:
            prev_dt = sim_dates[i - 1]
            cash_w = max(0.0, 1.0 - sum(current_weights.values()))
            day_ret = cash_w * INDIA_CASH_DAILY_YIELD
            for sym, w in current_weights.items():
                p0 = float(prices_df.loc[prev_dt, sym])
                p1 = float(prices_df.loc[dt, sym])
                if p0 > 0 and np.isfinite(p0) and np.isfinite(p1):
                    day_ret += w * ((p1 / p0) - 1.0)
        else:
            day_ret = 0.0

        # Daily V5/V6 6-Gauge India Sentiment & Intermarket Macro Risk Gate
        target_exposure = 1.0
        if regime_gate:
            vix_val = vix_map.get(dt, 15.0)
            br_val = float(breadth_series.loc[dt]) if pd.notna(breadth_series.loc[dt]) else 0.6
            hilo_val = float(hilo_series.loc[dt]) if pd.notna(hilo_series.loc[dt]) else 0.1
            gn_val = float(gold_nifty_chg.loc[dt]) if pd.notna(gold_nifty_chg.loc[dt]) else 0.0
            b_px = float(bench_series.loc[dt])
            s200 = float(bench_sma200.loc[dt]) if pd.notna(bench_sma200.loc[dt]) else b_px

            stress_pts = (
                int(vix_val >= 19.0)
                + int(vix_val >= 23.0)
                + int(b_px < s200)
                + int(br_val < 0.42)
                + int(hilo_val < -0.10)
                + int(gn_val > 0.14)
            )
            if stress_pts >= 3:
                target_exposure = 0.25
            elif stress_pts == 2:
                target_exposure = 0.55
            elif stress_pts == 1:
                target_exposure = 0.85

        pos = idx_loc[dt]

        # Ex-Ante Portfolio Volatility Targeting on active holdings (strictly causal trailing 60D cov)
        if regime_gate and vol_target is not None and current_weights and pos >= 65:
            syms = list(current_weights.keys())
            w_vec = np.array([current_weights[s] for s in syms], dtype=float)
            w_sum_cur = float(np.sum(w_vec))
            if w_sum_cur > 0.1:
                w_unit = w_vec / w_sum_cur
                sub_r = log_rets_df[syms].iloc[pos - 60 : pos].fillna(0.0).to_numpy(dtype=float)
                cov_60 = np.cov(sub_r, rowvar=False) * 252.0
                if cov_60.ndim == 0:
                    est_vol = float(np.sqrt(max(1e-6, float(cov_60))))
                else:
                    est_vol = float(np.sqrt(max(1e-6, float(w_unit @ cov_60 @ w_unit))))
                vol_cap = float(np.clip(vol_target / max(0.08, est_vol), 0.25, 1.0))
                target_exposure = min(target_exposure, vol_cap)

        cost_drag = 0.0
        if dt in rebalance_set and dt in oos_by_date:
            day_slice = oos_by_date[dt].sort_values("pred_alpha", ascending=False).reset_index(drop=True)
            day_slice["rank"] = np.arange(1, len(day_slice) + 1)
            rank_map = dict(zip(day_slice["symbol"], day_slice["rank"]))

            chosen: list[str] = []
            sec_counts: dict[str, int] = {}

            def _can_add(sym: str) -> bool:
                sec = sector_map.get(sym, "Other")
                return sec_counts.get(sec, 0) < max_per_sector

            def _add(sym: str) -> None:
                chosen.append(sym)
                sec = sector_map.get(sym, "Other")
                sec_counts[sec] = sec_counts.get(sec, 0) + 1

            # Rank-Buffer hysteresis: hold existing winners while inside Top 2xK
            for sym in list(current_weights.keys()):
                if rank_map.get(sym, 999) <= rank_buffer and sym in prices_df.columns and _can_add(sym):
                    _add(sym)

            for sym in day_slice["symbol"].tolist():
                if len(chosen) >= top_k:
                    break
                if sym in chosen or sym == BENCHMARK_SYMBOL or sym not in prices_df.columns or not _can_add(sym):
                    continue
                _add(sym)

            # Graceful fallback for tiny synthetic universes with very few sectors
            if len(chosen) < top_k:
                for sym in day_slice["symbol"].tolist():
                    if len(chosen) >= top_k:
                        break
                    if sym in chosen or sym == BENCHMARK_SYMBOL or sym not in prices_df.columns:
                        continue
                    chosen.append(sym)

            if chosen:
                vols = np.array(
                    [
                        max(0.12, float(vol_df.loc[dt, sym]) if (sym in vol_df.columns and pd.notna(vol_df.loc[dt, sym])) else 0.25)
                        for sym in chosen
                    ],
                    dtype=float,
                )
                if corr_penalized_weight and pos >= 130 and len(chosen) >= 3:
                    sub_r = log_rets_df[chosen].iloc[pos - 126 : pos].fillna(0.0)
                    cmat = sub_r.corr().fillna(0.0).to_numpy(dtype=float, copy=True)
                    np.fill_diagonal(cmat, np.nan)
                    avg_c = np.nanmean(np.clip(cmat, 0.0, 1.0), axis=1)
                    avg_c = np.where(np.isfinite(avg_c), avg_c, 0.2)
                    raw_vec = 1.0 / (vols * (1.0 + avg_c))
                else:
                    raw_vec = 1.0 / vols

                raw_vec = raw_vec / max(1e-9, float(np.sum(raw_vec)))

                rebal_exp = target_exposure
                if regime_gate and vol_target is not None and pos >= 65 and len(chosen) >= 2:
                    sub60 = log_rets_df[chosen].iloc[pos - 60 : pos].fillna(0.0).to_numpy(dtype=float)
                    cov60 = np.cov(sub60, rowvar=False) * 252.0
                    est_v = float(np.sqrt(max(1e-6, float(raw_vec @ cov60 @ raw_vec))))
                    rebal_exp = min(target_exposure, float(np.clip(vol_target / max(0.08, est_v), 0.25, 1.0)))

                new_weights = {sym: float(w * rebal_exp) for sym, w in zip(chosen, raw_vec)}
            else:
                new_weights = {}

            all_keys = set(current_weights.keys()).union(new_weights.keys())
            turnover = sum(abs(new_weights.get(k, 0.0) - current_weights.get(k, 0.0)) for k in all_keys) * 0.5
            cost_drag = turnover * fee_rate * 2.0
            total_turnover += turnover
            rebalance_count += 1
            current_weights = new_weights
        elif regime_gate and current_weights:
            cur_exp = sum(current_weights.values())
            if cur_exp > target_exposure + 0.15:
                scale = target_exposure / max(1e-9, cur_exp)
                turnover = abs(cur_exp - target_exposure) * 0.5
                cost_drag = turnover * fee_rate * 2.0
                total_turnover += turnover
                current_weights = {sym: w * scale for sym, w in current_weights.items()}

        daily_rets[i] = (1.0 + day_ret) * max(0.0, 1.0 - cost_drag) - 1.0

    return daily_rets, total_turnover, rebalance_count


def run_oos_portfolio_backtest(
    db: Database,
    top_k: int = 10,
    rebalance_days: int = 40,
    cost_bps: float = DEFAULT_COST_BPS,
    regime_gate: bool = True,
    initial_capital: float = 1000000.0,
    max_per_sector: int = 2,
    inv_vol_weight: bool = True,
    n_tranches: int = 4,
    vol_target: float | None = 0.18,
) -> dict[str, Any]:
    """Simulate V6 strictly out-of-sample 4-Tranche NSE portfolio vs. NIFTY 50 benchmark."""
    oos = db.get_oos_history()
    if not oos:
        raise ValueError("No walk-forward OOS history available. Train the ML model first.")

    ohlcv_map = db.get_all_ohlcv()
    sector_map = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    macro = db.get_macro_series()
    vix_map = macro.get(VOL_INDEX_SYMBOL, {})
    gold_map = macro.get(GOLD_MACRO_SYMBOL, {})
    bench_bars = ohlcv_map.get(BENCHMARK_SYMBOL, [])
    if len(bench_bars) < 60:
        raise ValueError("Benchmark NIFTY 50 price history missing.")

    prices_df = pd.DataFrame(
        {sym: pd.Series({b["date"]: float(b["close"]) for b in bars}) for sym, bars in ohlcv_map.items()}
    ).sort_index().ffill()

    # Precompute 1Y rolling annualized volatility, NSE Breadth (>50D SMA), and NSE 52W Hi-Lo Sentiment
    log_rets_df = np.log(prices_df / prices_df.shift(1))
    vol_df = log_rets_df.rolling(252, min_periods=63).std() * np.sqrt(252)
    stock_cols = [c for c in prices_df.columns if c != BENCHMARK_SYMBOL]
    sma50_df = prices_df[stock_cols].rolling(50, min_periods=30).mean()
    breadth_series = (prices_df[stock_cols] > sma50_df).mean(axis=1)

    high_52w_df = prices_df[stock_cols].rolling(252, min_periods=60).max().replace(0, np.nan)
    low_52w_df = prices_df[stock_cols].rolling(252, min_periods=60).min().replace(0, np.nan)
    pct_near_hi = (prices_df[stock_cols] / high_52w_df >= 0.90).mean(axis=1)
    pct_near_lo = (prices_df[stock_cols] / low_52w_df <= 1.12).mean(axis=1)
    hilo_series = (pct_near_hi - pct_near_lo).fillna(0.0)

    bench_series = prices_df[BENCHMARK_SYMBOL]
    bench_sma200 = bench_series.rolling(200, min_periods=50).mean()

    if gold_map:
        gold_s = pd.Series(gold_map, dtype=float).reindex(prices_df.index).ffill().bfill()
        gold_nifty_chg = (gold_s / bench_series.replace(0, np.nan)).pct_change(63).fillna(0.0)
    else:
        gold_nifty_chg = pd.Series(0.0, index=prices_df.index)

    oos_df = pd.DataFrame(oos)
    oos_dates = sorted(set(oos_df["date"].unique()).intersection(set(prices_df.index)))
    if len(oos_dates) < 30:
        raise ValueError("Insufficient overlapping OOS trading dates for portfolio simulation.")

    start_date, end_date = oos_dates[0], oos_dates[-1]
    sim_dates = [d for d in prices_df.index if start_date <= d <= end_date]
    oos_by_date = {dt: grp for dt, grp in oos_df.groupby("date")}
    idx_loc = {d: i for i, d in enumerate(prices_df.index)}

    effective_tranches = max(1, min(n_tranches, rebalance_days))
    offsets = sorted({(rebalance_days * t) // effective_tranches for t in range(effective_tranches)})

    tranche_rets: list[np.ndarray] = []
    turn_list: list[float] = []
    rebal_counts: list[int] = []
    for off in offsets:
        t_rets, t_turn, t_cnt = _simulate_single_tranche(
            sim_dates=sim_dates,
            oos_dates=oos_dates,
            oos_by_date=oos_by_date,
            prices_df=prices_df,
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
            cost_bps=cost_bps,
            regime_gate=regime_gate,
            max_per_sector=max_per_sector,
            corr_penalized_weight=inv_vol_weight,
            vol_target=vol_target,
        )
        tranche_rets.append(t_rets)
        turn_list.append(t_turn)
        rebal_counts.append(t_cnt)

    # Equal-weighted daily return across staggered overlapping tranches
    blended_daily = np.mean(np.vstack(tranche_rets), axis=0)

    equity = float(initial_capital)
    bench_equity = float(initial_capital)
    equity_curve: list[dict[str, Any]] = []
    port_daily_rets: list[float] = []
    bench_daily_rets: list[float] = []
    dates_record: list[str] = []

    for i, dt in enumerate(sim_dates):
        if i > 0:
            prev_dt = sim_dates[i - 1]
            day_ret = float(blended_daily[i])
            b_ret = (float(bench_series.loc[dt]) / float(bench_series.loc[prev_dt])) - 1.0
            equity *= 1.0 + day_ret
            bench_equity *= 1.0 + b_ret
            port_daily_rets.append(day_ret)
            bench_daily_rets.append(b_ret)
            dates_record.append(dt)
        else:
            equity *= 1.0 + float(blended_daily[0])

        equity_curve.append(
            {
                "date": dt,
                "portfolio": round(equity, 2),
                "spy": round(bench_equity, 2),
                "vix": round(float(vix_map.get(dt, 15.0)), 2),
            }
        )

    p_rets = np.array(port_daily_rets, dtype=float)
    s_rets = np.array(bench_daily_rets, dtype=float)
    years = max(0.25, len(p_rets) / 252.0)

    def _stats(rets: np.ndarray, final_eq: float, tax_rate: float = 0.20) -> dict[str, float]:
        tot_ret = ((final_eq / initial_capital) - 1.0) * 100.0
        cagr = ((final_eq / initial_capital) ** (1.0 / years) - 1.0) * 100.0
        # Annualized Indian STCG (20% active) or LTCG (12.5% buy-and-hold) post-tax estimate
        after_tax_cagr = cagr * (1.0 - tax_rate) if cagr > 0 else cagr
        vol = float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 2 else 0.0
        sharpe = float((np.mean(rets) * 252.0) / max(1e-9, vol))
        down = rets[rets < 0]
        down_vol = float(np.std(down, ddof=1) * np.sqrt(252)) if len(down) > 2 else 1e-9
        sortino = float((np.mean(rets) * 252.0) / max(1e-9, down_vol))
        cum = np.cumprod(1.0 + rets)
        peaks = np.maximum.accumulate(cum)
        dd = ((cum - peaks) / np.maximum(1e-9, peaks)) * 100.0
        max_dd = float(np.min(dd)) if len(dd) else 0.0
        calmar = cagr / abs(max_dd) if abs(max_dd) > 0.1 else 0.0
        return {
            "total_return_pct": round(tot_ret, 2),
            "cagr_pct": round(cagr, 2),
            "after_tax_cagr_pct": round(after_tax_cagr, 2),
            "ann_vol_pct": round(vol * 100.0, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar": round(calmar, 2),
        }

    p_stats = _stats(p_rets, equity, tax_rate=0.20)
    s_stats = _stats(s_rets, bench_equity, tax_rate=0.125)
    avg_rebal = int(round(float(np.mean(rebal_counts)))) if rebal_counts else 1
    avg_turn = float(np.mean(turn_list)) if turn_list else 0.0

    return {
        "config": {
            "top_k": top_k,
            "rebalance_days": rebalance_days,
            "rank_buffer": top_k * 2,
            "max_per_sector": max_per_sector,
            "inv_vol_weight": inv_vol_weight,
            "weighting_mode": "corr_penalized" if inv_vol_weight else "equal",
            "n_tranches": len(offsets),
            "vol_target_pct": round(vol_target * 100.0, 1) if vol_target else None,
            "cost_bps": cost_bps,
            "regime_gate": regime_gate,
            "initial_capital": initial_capital,
            "period": f"{start_date} to {end_date}",
            "rebalance_count": avg_rebal,
            "avg_turnover_pct": round((avg_turn / max(1, avg_rebal)) * 100.0, 1),
        },
        "portfolio_metrics": p_stats,
        "spy_metrics": s_stats,
        "excess_cagr_pct": round(p_stats["cagr_pct"] - s_stats["cagr_pct"], 2),
        "equity_curve": equity_curve,
        "_daily_returns": p_rets.tolist(),
        "_spy_daily_returns": s_rets.tolist(),
        "_dates": dates_record,
    }
