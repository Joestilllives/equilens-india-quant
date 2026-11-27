"""
Portfolio-level strictly out-of-sample Shortlist Backtester (EquiLens India V7 Architecture):
  - 4-Tranche Staggered Overlapping 40-Day Sleeves (25% capital per sleeve, staggered 10 trading days apart)
  - Top-10 NSE Portfolio, Top-30 (min(3*K, 40)) Rank-Buffer Hysteresis, Max 2 per NSE Sector,
    Hard <=12% Stock / <=20% Sector Caps
  - López de Prado (2016) Hierarchical Risk Parity (HRP on 126D Ledoit-Wolf Cov) alongside
    V6 Correlation-Penalized Inverse-Vol for clean side-by-side ablations
  - Optional mild Grinold-Kahn alpha tilt & budget-normalized Gârleanu-Pedersen L1 no-trade band
  - V6 Risk Controls preserved: Ex-Ante 18% Annualized Portfolio Vol Target (60D sample cov),
    One-Way Intra-Cycle De-Risking, 6-Gauge India Sentiment & Macro Gate, -10%/-6% Drawdown Brake,
    and 70/30 LIQUIDCASE/GOLDBEES Reserve (using actual GOLDBEES.NS returns)
  - Hardened Institutional Accounting: NIFTYBEES.NS TRI benchmark, exact Indian NSE statutory
    delivery costs + ADV impact, FIFO STCG/LTCG tax ledger, Excess Sharpe above 6.2% RF (SR 6.2%),
    and 2-Factor Pure Alpha (Market Beta + Equal-Weight Size Tilt).
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_SYMBOL,
    DEFAULT_COST_BPS,
    FX_MACRO_SYMBOL,
    GOLD_MACRO_SYMBOL,
    GOLDBEES_SYMBOL,
    NON_EQUITY_SYMBOLS,
    STRATEGY_ID_ML,
    TRI_BENCHMARK_SYMBOL,
    VOL_INDEX_SYMBOL,
)
from .convex_optimizer import solve_convex_nse_portfolio
from .data_health import build_clean_price_matrix
from .database import Database
from .hrp_allocator import (
    apply_budget_normalized_no_trade_band,
    apply_concentration_caps,
    compute_hrp_weights,
)
from .india_structural_signals import build_causal_nse_four_factors, compute_iima_four_factor_alpha
from .indian_costs import FifoTaxLedger, compute_indian_delivery_cost

INDIA_CASH_ANNUAL_YIELD = 0.062
INDIA_CASH_DAILY_YIELD = (1.0 + INDIA_CASH_ANNUAL_YIELD) ** (1.0 / 252.0) - 1.0
DRAWDOWN_BRAKE_TRIGGER = -0.10
DRAWDOWN_BRAKE_RELEASE = -0.06
GOLDBEES_RESERVE_SHARE = 0.30


def build_causal_gold_series(
    index: pd.Index,
    gold_map: dict[str, float],
    fx_map: dict[str, float],
    bench_series: pd.Series,
    goldbees_series: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Strictly causal T-1 Gold/NIFTY 63D ratio & INR Gold daily returns (ZERO .bfill())."""
    if gold_map:
        nse_dt_df = pd.DataFrame({"date": index, "_dt": pd.to_datetime(index)}).sort_values("_dt")
        g_df = pd.DataFrame({"_dt": pd.to_datetime(list(gold_map.keys())), "gold": list(gold_map.values())}).sort_values("_dt")
        g_merged = pd.merge_asof(nse_dt_df, g_df, on="_dt", direction="backward", allow_exact_matches=False)
        gold_s = pd.Series(g_merged["gold"].to_numpy(dtype=float), index=g_merged["date"]).ffill()
        gold_nifty_chg = (gold_s / bench_series.replace(0, np.nan)).pct_change(63).fillna(0.0)
        if goldbees_series is not None and goldbees_series.notna().sum() >= 60:
            gb_ret = goldbees_series.pct_change().clip(-0.10, 0.10)
            if fx_map:
                f_df = pd.DataFrame({"_dt": pd.to_datetime(list(fx_map.keys())), "fx": list(fx_map.values())}).sort_values("_dt")
                f_merged = pd.merge_asof(nse_dt_df, f_df, on="_dt", direction="backward", allow_exact_matches=False)
                fx_s = pd.Series(f_merged["fx"].to_numpy(dtype=float), index=f_merged["date"]).ffill()
                proxy_ret = (gold_s * fx_s).pct_change().clip(-0.10, 0.10)
            else:
                proxy_ret = gold_s.pct_change().clip(-0.10, 0.10)
            gold_inr_ret_series = gb_ret.fillna(proxy_ret).fillna(INDIA_CASH_DAILY_YIELD)
        elif fx_map:
            f_df = pd.DataFrame({"_dt": pd.to_datetime(list(fx_map.keys())), "fx": list(fx_map.values())}).sort_values("_dt")
            f_merged = pd.merge_asof(nse_dt_df, f_df, on="_dt", direction="backward", allow_exact_matches=False)
            fx_s = pd.Series(f_merged["fx"].to_numpy(dtype=float), index=f_merged["date"]).ffill()
            gold_inr_ret_series = (gold_s * fx_s).pct_change().clip(-0.10, 0.10).fillna(INDIA_CASH_DAILY_YIELD)
        else:
            gold_inr_ret_series = gold_s.pct_change().clip(-0.10, 0.10).fillna(INDIA_CASH_DAILY_YIELD)
        return gold_nifty_chg, gold_inr_ret_series
    if goldbees_series is not None and goldbees_series.notna().sum() >= 60:
        gb_ret = goldbees_series.pct_change().clip(-0.10, 0.10).fillna(INDIA_CASH_DAILY_YIELD)
        return pd.Series(0.0, index=index), gb_ret
    return pd.Series(0.0, index=index), pd.Series(INDIA_CASH_DAILY_YIELD, index=index)


def compute_two_factor_alpha(
    port_rets: np.ndarray,
    mkt_rets: np.ndarray,
    eqw_rets: np.ndarray,
    rf_daily: float = INDIA_CASH_DAILY_YIELD,
) -> dict[str, float]:
    """
    2-Factor Pure Alpha regression controlling for Market Beta (MKT) and Equal-Weight Size Tilt (SMB_EW = EqW - MKT):
        (r_p - r_f) = alpha_daily + beta_mkt * (r_mkt - r_f) + beta_size * (r_eqw - r_mkt) + eps
    Returns annualized 2-Factor Pure Alpha (%), beta_mkt, beta_size, and t-stat of alpha.
    """
    p = np.asarray(port_rets, dtype=float)
    m = np.asarray(mkt_rets, dtype=float)
    e = np.asarray(eqw_rets, dtype=float)
    n = min(len(p), len(m), len(e))
    if n < 30:
        return {"alpha_2f_pct": 0.0, "beta_mkt": 0.0, "beta_size": 0.0, "alpha_tstat": 0.0, "r2": 0.0}
    y = p[:n] - rf_daily
    x_mkt = m[:n] - rf_daily
    x_size = e[:n] - m[:n]
    X = np.column_stack([np.ones(n), x_mkt, x_size])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(1, n - 3)
    s2 = float(np.sum(resid ** 2) / dof)
    try:
        cov_b = s2 * np.linalg.pinv(X.T @ X)
        se_a = float(np.sqrt(max(1e-18, cov_b[0, 0])))
    except Exception:
        se_a = 1e-9
    t_alpha = float(coef[0] / max(1e-12, se_a))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / max(1e-12, ss_tot)
    return {
        "alpha_2f_pct": round(float(coef[0] * 252.0 * 100.0), 2),
        "beta_mkt": round(float(coef[1]), 3),
        "beta_size": round(float(coef[2]), 3),
        "alpha_tstat": round(t_alpha, 2),
        "r2": round(float(np.clip(r2, 0.0, 1.0)), 3),
    }


def apply_drawdown_brake_and_gold_reserve(
    base_rets: np.ndarray,
    idle_weights: np.ndarray,
    gold_inr_rets: np.ndarray,
    dd_trigger: float = DRAWDOWN_BRAKE_TRIGGER,
    gold_idle_share: float = GOLDBEES_RESERVE_SHARE,
    cost_bps: float = DEFAULT_COST_BPS,
    enable_dd_brake: bool = True,
) -> np.ndarray:
    """Strictly causal -10% own-portfolio drawdown brake (-6% release) + 70% LIQUIDCASE / 30% GOLDBEES reserve."""
    n = len(base_rets)
    out = np.zeros(n, dtype=float)
    eq_hist = np.ones(n + 1, dtype=float)
    peak = 1.0
    brake_on = False
    cost_rate = max(0.0, cost_bps) / 10000.0

    for i in range(n):
        idle_w = float(idle_weights[i]) if i < len(idle_weights) else 0.0
        r_base = float(base_rets[i])
        r_cash = INDIA_CASH_DAILY_YIELD
        r_gold = float(gold_inr_rets[i]) if (i < len(gold_inr_rets) and np.isfinite(gold_inr_rets[i])) else r_cash

        r_day = r_base + (idle_w * gold_idle_share) * (r_gold - r_cash)
        if enable_dd_brake and brake_on:
            eq_w = max(0.0, 1.0 - idle_w)
            r_eq_part = r_base - idle_w * r_cash
            def_ret = (1.0 - gold_idle_share) * r_cash + gold_idle_share * r_gold
            r_day = 0.5 * r_eq_part + (idle_w + 0.5 * eq_w) * def_ret

        out[i] = r_day
        eq_now = eq_hist[i] * (1.0 + r_day)
        eq_hist[i + 1] = eq_now
        peak = max(peak, eq_now)
        dd_now = (eq_now / peak) - 1.0

        if enable_dd_brake and i >= 20:
            sma20 = float(np.mean(eq_hist[i - 19 : i + 2]))
            new_brake = brake_on
            if not brake_on and dd_now <= dd_trigger:
                new_brake = True
            elif brake_on and eq_now > sma20 * 1.005 and dd_now > dd_trigger * 0.6:
                new_brake = False
            if new_brake != brake_on:
                out[i] -= 0.35 * cost_rate
                eq_hist[i + 1] = eq_hist[i] * (1.0 + out[i])
            brake_on = new_brake
    return out


def _build_causal_delivery_20d_lookup(
    deliv_hist: dict[str, dict[str, float]] | None,
    window: int = 20,
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Precompute sorted dates & trailing `window`-day rolling mean NSE delivery % per symbol."""
    if not deliv_hist:
        return {}
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    w = max(1, int(window))
    for sym, sym_map in deliv_hist.items():
        if not sym_map:
            continue
        dts = sorted(sym_map.keys())
        vals = np.array([float(sym_map[d]) for d in dts], dtype=float)
        csum = np.concatenate(([0.0], np.cumsum(vals)))
        idx = np.arange(len(vals))
        starts = np.maximum(0, idx - w + 1)
        rolling = (csum[idx + 1] - csum[starts]) / (idx - starts + 1).astype(float)
        out[sym] = (dts, rolling)
    return out


def _prior_delivery_pct(
    deliv_hist: dict[str, Any] | None,
    sym: str,
    dt: str,
    window: int = 20,
) -> float | None:
    """Return strictly prior-day (T-1) trailing `window`-day mean NSE delivery % before `dt`, or None."""
    if not deliv_hist or sym not in deliv_hist:
        return None
    entry = deliv_hist[sym]
    if isinstance(entry, tuple) and len(entry) == 2:
        dts, rolling = entry
        pos = bisect_left(dts, dt) - 1
        if pos < 0:
            return None
        return float(rolling[pos])
    prior_dates = sorted(d for d in entry.keys() if d < dt)
    if not prior_dates:
        return None
    tail = prior_dates[-max(1, int(window)) :]
    return float(np.mean([float(entry[d]) for d in tail]))


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
    out_idle_weights: list[float] | None = None,
    tradable_df: pd.DataFrame | None = None,
    adv_df: pd.DataFrame | None = None,
    tranche_capital: float = 250_000.0,
    stagger_initial_deployment: bool = True,
    use_statutory_costs: bool = True,
    deliv_history: dict[str, dict[str, float]] | None = None,
    enforce_smart_money_shield: bool = False,
    weighting_mode: str = "hrp",
    hrp_alpha_tilt: float = 0.0,
    no_trade_band: float = 0.0,
    out_accounting: dict[str, Any] | None = None,
    out_holdings_log: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Run one strictly causal overlapping sleeve starting at `rebal_offset` with Indian cost & FIFO tax ledger."""
    rebalance_set = set(oos_dates[rebal_offset::rebalance_days])
    if not stagger_initial_deployment and oos_dates:
        rebalance_set.add(oos_dates[0])

    rank_buffer = min(top_k * 3, 40)
    extra_slip_bps = max(0.0, float(cost_bps) - DEFAULT_COST_BPS)
    zero_cost_mode = cost_bps <= 0.0
    eff_weighting = weighting_mode if corr_penalized_weight else "equal"

    current_weights: dict[str, float] = {}
    held_shares: dict[str, float] = {}
    tranche_eq = float(tranche_capital)
    tax_ledger = FifoTaxLedger()
    total_statutory_inr = 0.0
    total_slippage_inr = 0.0
    max_participation_pct = 0.0
    total_turnover = 0.0
    rebalance_count = 0
    daily_rets = np.zeros(len(sim_dates), dtype=float)

    # Pre-extract contiguous numpy views for fast inner-loop indexing (bit-identical to .loc)
    px_arr = prices_df.to_numpy(dtype=float)
    px_col = {c: j for j, c in enumerate(prices_df.columns)}
    lr_arr = np.nan_to_num(log_rets_df.to_numpy(dtype=float), nan=0.0)
    lr_col = {c: j for j, c in enumerate(log_rets_df.columns)}
    br_arr = breadth_series.to_numpy(dtype=float)
    hilo_arr = hilo_series.to_numpy(dtype=float)
    gn_arr = gold_nifty_chg.to_numpy(dtype=float)
    b_arr = bench_series.to_numpy(dtype=float)
    s200_arr = bench_sma200.to_numpy(dtype=float)

    def _is_tradable(sym: str, dt: str) -> bool:
        if sym in NON_EQUITY_SYMBOLS or sym not in px_col:
            return False
        px = px_arr[idx_loc[dt], px_col[sym]]
        if not np.isfinite(px) or px <= 0:
            return False
        if tradable_df is not None and sym in tradable_df.columns:
            return bool(tradable_df.loc[dt, sym])
        return True

    def _order_cost(side: str, sym: str, dt: str, gross_inr: float) -> tuple[float, float, float]:
        nonlocal max_participation_pct
        if zero_cost_mode or gross_inr <= 0.0:
            return 0.0, 0.0, 0.0
        if not use_statutory_costs:
            c = gross_inr * (float(cost_bps) / 10000.0)
            return c, c, 0.0
        adv_val = None
        if adv_df is not None and sym in adv_df.columns:
            v = adv_df.loc[dt, sym]
            if pd.notna(v) and float(v) > 0:
                adv_val = float(v)
        dvol = 0.02
        if sym in vol_df.columns and pd.notna(vol_df.loc[dt, sym]):
            dvol = float(vol_df.loc[dt, sym]) / np.sqrt(252.0)
        res = compute_indian_delivery_cost(
            side=side,
            gross_value_inr=gross_inr,
            adv_20d_inr=adv_val,
            daily_vol=dvol,
            half_spread_bps=3.0,
            extra_slippage_bps=extra_slip_bps,
            include_dp_charge=False,
        )
        max_participation_pct = max(max_participation_pct, res["participation_pct"])
        return res["total_cost_inr"], res["statutory_inr"], res["slippage_inr"]

    for i, dt in enumerate(sim_dates):
        pos = idx_loc[dt]
        if i > 0:
            prev_pos = idx_loc[sim_dates[i - 1]]
            cash_w = max(0.0, 1.0 - sum(current_weights.values()))
            if out_idle_weights is not None:
                out_idle_weights.append(cash_w)
            day_ret = cash_w * INDIA_CASH_DAILY_YIELD
            for sym, w in current_weights.items():
                cj = px_col.get(sym)
                if cj is not None:
                    p0 = px_arr[prev_pos, cj]
                    p1 = px_arr[pos, cj]
                    if p0 > 0.0 and np.isfinite(p0) and np.isfinite(p1):
                        day_ret += w * ((p1 / p0) - 1.0)
            tranche_eq = max(1.0, tranche_eq * (1.0 + day_ret))
        else:
            if out_idle_weights is not None:
                out_idle_weights.append(1.0)
            day_ret = 0.0

        # Daily 6-Gauge India Sentiment & Intermarket Macro Risk Gate (V6 preserved)
        target_exposure = 1.0
        if regime_gate:
            vix_val = vix_map.get(dt, 15.0)
            br_raw = br_arr[pos]
            hilo_raw = hilo_arr[pos]
            gn_raw = gn_arr[pos]
            b_raw = b_arr[pos]
            s200_raw = s200_arr[pos]
            br_val = float(br_raw) if np.isfinite(br_raw) else 0.6
            hilo_val = float(hilo_raw) if np.isfinite(hilo_raw) else 0.1
            gn_val = float(gn_raw) if np.isfinite(gn_raw) else 0.0
            b_px = float(b_raw) if np.isfinite(b_raw) else 1.0
            s200 = float(s200_raw) if np.isfinite(s200_raw) else b_px

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

        # Ex-Ante Portfolio Volatility Targeting (strictly causal trailing 60D SAMPLE cov — NOT LedoitWolf!)
        if regime_gate and vol_target is not None and current_weights and pos >= 65:
            syms = [s for s in current_weights.keys() if s in lr_col]
            w_vec = np.array([current_weights[s] for s in syms], dtype=float)
            w_sum_cur = float(np.sum(w_vec))
            if syms and w_sum_cur > 0.1:
                w_unit = w_vec / w_sum_cur
                c_idxs = [lr_col[s] for s in syms]
                sub_r = lr_arr[pos - 60 : pos, c_idxs]
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
            alpha_map = dict(zip(day_slice["symbol"], day_slice["pred_alpha"]))

            chosen: list[str] = []
            sec_counts: dict[str, int] = {}

            def _can_add(sym: str) -> bool:
                sec = sector_map.get(sym, "Other")
                return sec_counts.get(sec, 0) < max_per_sector

            def _add(sym: str) -> None:
                chosen.append(sym)
                sec = sector_map.get(sym, "Other")
                sec_counts[sec] = sec_counts.get(sec, 0) + 1

            def _passes_delivery_shield(sym: str, cur_dt: str) -> bool:
                if not enforce_smart_money_shield:
                    return True
                d_prev = _prior_delivery_pct(deliv_history, sym, cur_dt)
                return d_prev is None or d_prev >= 35.0

            # Rank-Buffer hysteresis: hold existing winners while inside Top-30 buffer, tradable, and passing 20D delivery shield
            for sym in list(current_weights.keys()):
                if (
                    rank_map.get(sym, 999) <= rank_buffer
                    and _is_tradable(sym, dt)
                    and _passes_delivery_shield(sym, dt)
                    and _can_add(sym)
                ):
                    _add(sym)

            for sym in day_slice["symbol"].tolist():
                if len(chosen) >= top_k:
                    break
                if sym in chosen or not _is_tradable(sym, dt) or not _can_add(sym):
                    continue
                if not _passes_delivery_shield(sym, dt):
                    continue
                _add(sym)

            if len(chosen) < top_k:
                for sym in day_slice["symbol"].tolist():
                    if len(chosen) >= top_k:
                        break
                    if sym in chosen or not _is_tradable(sym, dt) or not _passes_delivery_shield(sym, dt):
                        continue
                    chosen.append(sym)

            if chosen:
                vols_dict = {
                    sym: max(0.12, float(vol_df.loc[dt, sym]) if (sym in vol_df.columns and pd.notna(vol_df.loc[dt, sym])) else 0.25)
                    for sym in chosen
                }
                if eff_weighting == "convex":
                    ret_win = log_rets_df[chosen].iloc[max(0, pos - 126) : pos] if pos >= 30 else None
                    max_stk_cap = 0.05 if top_k >= 20 else round(min(0.15, 1.20 / max(1, top_k)), 4)
                    max_sec_cap = 0.25 if top_k < 10 else 0.20
                    adv_now = {
                        sym: float(adv_df.loc[dt, sym])
                        for sym in chosen
                        if adv_df is not None and sym in adv_df.columns and pd.notna(adv_df.loc[dt, sym])
                    }
                    px_now = {sym: float(px_arr[pos, px_col[sym]]) for sym in chosen if sym in px_col}
                    unit_map = solve_convex_nse_portfolio(
                        symbols=chosen,
                        alpha_scores=alpha_map,
                        returns_window=ret_win,
                        fallback_vols=vols_dict,
                        sector_map=sector_map,
                        current_weights=current_weights,
                        target_exposure=1.0,
                        max_stock_w=max_stk_cap,
                        max_sector_w=max_sec_cap,
                        capital_inr=tranche_eq,
                        adv_map=adv_now,
                        fifo_lots=tax_ledger.export_lots(),
                        prices=px_now,
                        as_of_date=dt,
                    )
                    raw_vec = np.array([unit_map.get(s, 0.0) for s in chosen], dtype=float)
                    r_sum = float(np.sum(raw_vec))
                    raw_vec = raw_vec / r_sum if r_sum > 1e-9 else np.full(len(chosen), 1.0 / len(chosen))
                elif eff_weighting == "hrp":
                    ret_win = log_rets_df[chosen].iloc[max(0, pos - 126) : pos] if pos >= 30 else None
                    unit_map = compute_hrp_weights(
                        symbols=chosen,
                        returns_window=ret_win,
                        fallback_vols=vols_dict,
                        alpha_scores=alpha_map,
                        alpha_tilt=hrp_alpha_tilt,
                        target_exposure=1.0,
                    )
                    raw_vec = np.array([unit_map[s] for s in chosen], dtype=float)
                elif eff_weighting == "corr_penalized":
                    vols = np.array([vols_dict[s] for s in chosen], dtype=float)
                    if pos >= 130 and len(chosen) >= 3:
                        sub_r = log_rets_df[chosen].iloc[pos - 126 : pos].fillna(0.0)
                        cmat = sub_r.corr().fillna(0.0).to_numpy(dtype=float, copy=True)
                        np.fill_diagonal(cmat, np.nan)
                        avg_c = np.nanmean(np.clip(cmat, 0.0, 1.0), axis=1)
                        avg_c = np.where(np.isfinite(avg_c), avg_c, 0.2)
                        raw_vec = 1.0 / (vols * (1.0 + avg_c))
                    else:
                        raw_vec = 1.0 / vols
                    raw_vec = raw_vec / max(1e-9, float(np.sum(raw_vec)))
                else:
                    raw_vec = np.full(len(chosen), 1.0 / len(chosen), dtype=float)

                # Keep V6 60D sample covariance scalar vol target intact
                rebal_exp = target_exposure
                if regime_gate and vol_target is not None and pos >= 65 and len(chosen) >= 2:
                    sub60 = log_rets_df[chosen].iloc[pos - 60 : pos].fillna(0.0).to_numpy(dtype=float)
                    cov60 = np.cov(sub60, rowvar=False) * 252.0
                    est_v = float(np.sqrt(max(1e-6, float(raw_vec @ cov60 @ raw_vec))))
                    rebal_exp = min(target_exposure, float(np.clip(vol_target / max(0.08, est_v), 0.25, 1.0)))

                new_weights = {sym: float(w * rebal_exp) for sym, w in zip(chosen, raw_vec)}
                if eff_weighting in ("hrp", "corr_penalized", "convex") and len(chosen) >= 8:
                    max_stk_cap = 0.05 if top_k >= 20 else round(min(0.15, 1.20 / max(1, top_k)), 4)
                    max_sec_cap = 0.25 if top_k < 10 else 0.20
                    new_weights = apply_concentration_caps(new_weights, sector_map, max_stk_cap, max_sec_cap, rebal_exp)

                if no_trade_band > 0.0 and current_weights:
                    tgt_sum = float(sum(new_weights.values()))
                    new_weights = apply_budget_normalized_no_trade_band(
                        new_weights,
                        current_weights=current_weights,
                        band=no_trade_band,
                        target_total=tgt_sum,
                    )
                    if eff_weighting in ("hrp", "corr_penalized", "convex") and len(chosen) >= 8:
                        max_stk_cap = 0.05 if top_k >= 20 else round(min(0.15, 1.20 / max(1, top_k)), 4)
                        max_sec_cap = 0.25 if top_k < 10 else 0.20
                        new_weights = apply_concentration_caps(new_weights, sector_map, max_stk_cap, max_sec_cap, tgt_sum)
            else:
                new_weights = {}

            if out_holdings_log is not None:
                out_holdings_log.append({"date": dt, "weights": dict(new_weights)})

            # Side-explicit Indian delivery cost & FIFO tax accounting
            all_keys = set(current_weights.keys()).union(new_weights.keys())
            tot_cost_inr = 0.0
            one_way_turn = 0.0
            for sym in all_keys:
                w_old = current_weights.get(sym, 0.0)
                w_new = new_weights.get(sym, 0.0)
                dw = w_new - w_old
                if abs(dw) < 1e-6:
                    continue
                one_way_turn += abs(dw) * 0.5
                px_val = prices_df.loc[dt, sym] if sym in prices_df.columns else np.nan
                px = float(px_val) if pd.notna(px_val) and float(px_val) > 0 else 0.0
                gross_inr = abs(dw) * tranche_eq
                if dw < 0:
                    c_tot, c_stat, c_slip = _order_cost("SELL", sym, dt, gross_inr)
                    tot_cost_inr += c_tot
                    total_statutory_inr += c_stat
                    total_slippage_inr += c_slip
                    if px > 0 and held_shares.get(sym, 0.0) > 0:
                        sh_sell = held_shares[sym] if w_new <= 1e-9 else min(held_shares[sym], gross_inr / px)
                        tax_ledger.sell(sym, dt, sh_sell, px, sell_cost_inr=c_tot)
                        held_shares[sym] = max(0.0, held_shares[sym] - sh_sell)
                        if held_shares[sym] <= 1e-9:
                            held_shares.pop(sym, None)
                else:
                    c_tot, c_stat, c_slip = _order_cost("BUY", sym, dt, gross_inr)
                    tot_cost_inr += c_tot
                    total_statutory_inr += c_stat
                    total_slippage_inr += c_slip
                    if px > 0:
                        sh_buy = gross_inr / px
                        tax_ledger.buy(sym, dt, sh_buy, px, buy_cost_inr=c_tot)
                        held_shares[sym] = held_shares.get(sym, 0.0) + sh_buy

            cost_drag = min(0.25, tot_cost_inr / max(1.0, tranche_eq))
            tranche_eq = max(1.0, tranche_eq * (1.0 - cost_drag))
            total_turnover += one_way_turn
            rebalance_count += 1
            current_weights = new_weights
        elif regime_gate and current_weights:
            # One-way intra-cycle de-risking ONLY (empirically proven superior to symmetric re-entry)
            cur_exp = sum(current_weights.values())
            if cur_exp > target_exposure + 0.15:
                scale = target_exposure / max(1e-9, cur_exp)
                tot_cost_inr = 0.0
                for sym, w_old in list(current_weights.items()):
                    w_new = w_old * scale
                    gross_inr = (w_old - w_new) * tranche_eq
                    c_tot, c_stat, c_slip = _order_cost("SELL", sym, dt, gross_inr)
                    tot_cost_inr += c_tot
                    total_statutory_inr += c_stat
                    total_slippage_inr += c_slip
                    px_val = prices_df.loc[dt, sym] if sym in prices_df.columns else np.nan
                    if pd.notna(px_val) and float(px_val) > 0 and held_shares.get(sym, 0.0) > 0:
                        sh_sell = min(held_shares[sym], gross_inr / float(px_val))
                        tax_ledger.sell(sym, dt, sh_sell, float(px_val), sell_cost_inr=c_tot)
                        held_shares[sym] = max(0.0, held_shares[sym] - sh_sell)
                cost_drag = min(0.25, tot_cost_inr / max(1.0, tranche_eq))
                tranche_eq = max(1.0, tranche_eq * (1.0 - cost_drag))
                total_turnover += abs(cur_exp - target_exposure) * 0.5
                current_weights = {sym: w * scale for sym, w in current_weights.items()}

        daily_rets[i] = (1.0 + day_ret) * max(0.0, 1.0 - cost_drag) - 1.0

    if out_accounting is not None and sim_dates:
        end_dt = sim_dates[-1]
        last_px_map = {
            s: float(prices_df.loc[end_dt, s])
            for s in held_shares.keys()
            if s in prices_df.columns and pd.notna(prices_df.loc[end_dt, s]) and float(prices_df.loc[end_dt, s]) > 0
        }
        adv_end = {
            s: float(adv_df.loc[end_dt, s])
            for s in held_shares.keys()
            if adv_df is not None and s in adv_df.columns and pd.notna(adv_df.loc[end_dt, s]) and float(adv_df.loc[end_dt, s]) > 0
        }
        liq = tax_ledger.estimate_liquidation(
            end_dt,
            last_px_map,
            adv_map=adv_end,
            half_spread_bps=0.0 if zero_cost_mode else 3.0,
            include_dp_charge=False,
        )
        out_accounting.update(
            {
                "realized_stcg_tax_inr": round(tax_ledger.total_stcg_tax_inr, 2),
                "realized_ltcg_tax_inr": round(tax_ledger.total_ltcg_tax_inr, 2),
                "unrealized_liq_cost_inr": 0.0 if zero_cost_mode else liq["liquidation_cost_inr"],
                "unrealized_stcg_tax_inr": liq["unrealized_stcg_tax_inr"],
                "unrealized_ltcg_tax_inr": liq["unrealized_ltcg_tax_inr"],
                "total_statutory_inr": round(total_statutory_inr, 2),
                "total_slippage_inr": round(total_slippage_inr, 2),
                "max_participation_pct": round(max_participation_pct, 4),
            }
        )

    return daily_rets, total_turnover, rebalance_count


def build_backtest_market_context(db: Database) -> dict[str, Any]:
    """Precompute read-only market matrices once for reuse across multi-cell backtest grids."""
    ohlcv_map = db.get_all_ohlcv()
    sector_map = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    macro = db.get_macro_series()
    vix_map = macro.get(VOL_INDEX_SYMBOL, {})
    gold_map = macro.get(GOLD_MACRO_SYMBOL, {})
    fx_map = macro.get(FX_MACRO_SYMBOL, {})
    bench_bars = ohlcv_map.get(BENCHMARK_SYMBOL, [])
    if len(bench_bars) < 60:
        raise ValueError("Benchmark NIFTY 50 price history missing.")

    prices_df, volumes_df, tradable_df = build_clean_price_matrix(ohlcv_map, max_ffill_days=2)
    adv_df = (prices_df * volumes_df).rolling(20, min_periods=5).mean()

    log_rets_df = np.log(prices_df / prices_df.shift(1))
    vol_df = log_rets_df.rolling(252, min_periods=63).std() * np.sqrt(252)
    stock_cols = [c for c in prices_df.columns if c not in NON_EQUITY_SYMBOLS and not c.startswith("^")]
    sma50_df = prices_df[stock_cols].rolling(50, min_periods=30).mean()
    breadth_series = (prices_df[stock_cols] > sma50_df).mean(axis=1)

    high_52w_df = prices_df[stock_cols].rolling(252, min_periods=60).max().replace(0, np.nan)
    low_52w_df = prices_df[stock_cols].rolling(252, min_periods=60).min().replace(0, np.nan)
    pct_near_hi = (prices_df[stock_cols] / high_52w_df >= 0.90).mean(axis=1)
    pct_near_lo = (prices_df[stock_cols] / low_52w_df <= 1.12).mean(axis=1)
    hilo_series = (pct_near_hi - pct_near_lo).fillna(0.0)

    bench_price_series = prices_df[BENCHMARK_SYMBOL].ffill(limit=5)
    bench_sma200 = bench_price_series.rolling(200, min_periods=50).mean()

    use_tri = TRI_BENCHMARK_SYMBOL in prices_df.columns and int(prices_df[TRI_BENCHMARK_SYMBOL].notna().sum()) >= 120
    eval_bench_sym = TRI_BENCHMARK_SYMBOL if use_tri else BENCHMARK_SYMBOL
    eval_bench_type = "TRI" if use_tri else "PRICE"
    eval_bench_series = prices_df[eval_bench_sym].ffill(limit=5)

    goldbees_series = prices_df[GOLDBEES_SYMBOL] if GOLDBEES_SYMBOL in prices_df.columns else None
    gold_nifty_chg, gold_inr_ret_series = build_causal_gold_series(
        prices_df.index, gold_map, fx_map, bench_price_series, goldbees_series=goldbees_series
    )
    four_factors_df = build_causal_nse_four_factors(
        prices_df, eval_bench_series.pct_change().fillna(0.0), annual_rf=0.062
    )
    raw_deliv_hist = db.get_nse_delivery_history() if hasattr(db, "get_nse_delivery_history") else {}
    deliv_20d_lookup = _build_causal_delivery_20d_lookup(raw_deliv_hist, window=20)
    idx_loc = {d: i for i, d in enumerate(prices_df.index)}

    return {
        "sector_map": sector_map,
        "vix_map": vix_map,
        "prices_df": prices_df,
        "tradable_df": tradable_df,
        "adv_df": adv_df,
        "log_rets_df": log_rets_df,
        "vol_df": vol_df,
        "stock_cols": stock_cols,
        "breadth_series": breadth_series,
        "hilo_series": hilo_series,
        "bench_price_series": bench_price_series,
        "bench_sma200": bench_sma200,
        "eval_bench_sym": eval_bench_sym,
        "eval_bench_type": eval_bench_type,
        "eval_bench_series": eval_bench_series,
        "gold_nifty_chg": gold_nifty_chg,
        "gold_inr_ret_series": gold_inr_ret_series,
        "four_factors_df": four_factors_df,
        "deliv_20d_lookup": deliv_20d_lookup,
        "idx_loc": idx_loc,
    }


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
    weighting_mode: str = "hrp",
    hrp_alpha_tilt: float = 0.0,
    no_trade_band: float = 0.0,
    stagger_initial_deployment: bool = True,
    use_statutory_costs: bool = True,
    enable_dd_brake: bool = True,
    gold_reserve_share: float = GOLDBEES_RESERVE_SHARE,
    enforce_smart_money_shield: bool = True,
    oos_override: list[dict[str, Any]] | pd.DataFrame | None = None,
    market_ctx: dict[str, Any] | None = None,
    oos_by_date_override: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Simulate strictly out-of-sample 4-Tranche NSE portfolio vs. NIFTY 50 TRI (NIFTYBEES.NS)."""
    ctx = market_ctx if market_ctx is not None else build_backtest_market_context(db)
    sector_map = ctx["sector_map"]
    vix_map = ctx["vix_map"]
    prices_df = ctx["prices_df"]
    tradable_df = ctx["tradable_df"]
    adv_df = ctx["adv_df"]
    log_rets_df = ctx["log_rets_df"]
    vol_df = ctx["vol_df"]
    stock_cols = ctx["stock_cols"]
    breadth_series = ctx["breadth_series"]
    hilo_series = ctx["hilo_series"]
    bench_price_series = ctx["bench_price_series"]
    bench_sma200 = ctx["bench_sma200"]
    eval_bench_sym = ctx["eval_bench_sym"]
    eval_bench_type = ctx["eval_bench_type"]
    eval_bench_series = ctx["eval_bench_series"]
    gold_nifty_chg = ctx["gold_nifty_chg"]
    gold_inr_ret_series = ctx["gold_inr_ret_series"]
    idx_loc = ctx["idx_loc"]

    if oos_by_date_override is not None:
        oos_by_date = oos_by_date_override
        oos_dates = sorted(set(oos_by_date.keys()).intersection(set(prices_df.index)))
    else:
        if oos_override is not None:
            oos_df = oos_override.copy() if isinstance(oos_override, pd.DataFrame) else pd.DataFrame(oos_override)
        else:
            oos = db.get_oos_history()
            if not oos:
                raise ValueError("No walk-forward OOS history available. Train the ML model first.")
            oos_df = pd.DataFrame(oos)

        oos_df = oos_df[~oos_df["symbol"].isin(NON_EQUITY_SYMBOLS)].copy()
        if oos_df.empty:
            raise ValueError("No equity OOS rows available after excluding non-equity symbols.")

        oos_dates = sorted(set(oos_df["date"].unique()).intersection(set(prices_df.index)))
        oos_by_date = {dt: grp for dt, grp in oos_df.groupby("date")}

    if len(oos_dates) < 30:
        raise ValueError("Insufficient overlapping OOS trading dates for portfolio simulation.")

    start_date, end_date = oos_dates[0], oos_dates[-1]
    sim_dates = [d for d in prices_df.index if start_date <= d <= end_date]

    effective_tranches = max(1, min(n_tranches, rebalance_days))
    offsets = sorted({(rebalance_days * t) // effective_tranches for t in range(effective_tranches)})
    per_tranche_cap = float(initial_capital) / max(1, len(offsets))

    if enforce_smart_money_shield:
        deliv_history = ctx.get("deliv_20d_lookup")
        if deliv_history is None and hasattr(db, "get_nse_delivery_history"):
            deliv_history = _build_causal_delivery_20d_lookup(db.get_nse_delivery_history(), window=20)
    else:
        deliv_history = None
    eff_weighting = weighting_mode if inv_vol_weight else "equal"

    tranche_rets: list[np.ndarray] = []
    tranche_idles: list[np.ndarray] = []
    turn_list: list[float] = []
    rebal_counts: list[int] = []
    acct_list: list[dict[str, Any]] = []
    holdings_log: list[dict[str, Any]] = []

    for off in offsets:
        idle_buf: list[float] = []
        acct_buf: dict[str, Any] = {}
        t_rets, t_turn, t_cnt = _simulate_single_tranche(
            sim_dates=sim_dates,
            oos_dates=oos_dates,
            oos_by_date=oos_by_date,
            prices_df=prices_df,
            log_rets_df=log_rets_df,
            vol_df=vol_df,
            bench_series=bench_price_series,
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
            out_idle_weights=idle_buf,
            tradable_df=tradable_df,
            adv_df=adv_df,
            tranche_capital=per_tranche_cap,
            stagger_initial_deployment=stagger_initial_deployment,
            use_statutory_costs=use_statutory_costs,
            deliv_history=deliv_history,
            enforce_smart_money_shield=enforce_smart_money_shield,
            weighting_mode=eff_weighting,
            hrp_alpha_tilt=hrp_alpha_tilt,
            no_trade_band=no_trade_band,
            out_accounting=acct_buf,
            out_holdings_log=holdings_log if off == offsets[0] else None,
        )
        tranche_rets.append(t_rets)
        tranche_idles.append(np.array(idle_buf, dtype=float))
        turn_list.append(t_turn)
        rebal_counts.append(t_cnt)
        acct_list.append(acct_buf)

    # Equal-weighted daily return across staggered overlapping tranches + -10% Drawdown Brake & 70/30 Reserve
    blended_daily = np.mean(np.vstack(tranche_rets), axis=0)
    if regime_gate:
        blended_idle = np.mean(np.vstack(tranche_idles), axis=0)
        gold_rets_sim = gold_inr_ret_series.reindex(sim_dates).fillna(INDIA_CASH_DAILY_YIELD).to_numpy(dtype=float)
        blended_daily = apply_drawdown_brake_and_gold_reserve(
            blended_daily,
            blended_idle,
            gold_rets_sim,
            dd_trigger=DRAWDOWN_BRAKE_TRIGGER,
            gold_idle_share=gold_reserve_share,
            cost_bps=cost_bps,
            enable_dd_brake=enable_dd_brake,
        )

    equity = float(initial_capital)
    b_entry_cost = (
        0.0
        if cost_bps <= 0.0
        else compute_indian_delivery_cost("BUY", initial_capital, half_spread_bps=3.0, include_dp_charge=False)["total_cost_inr"]
        if use_statutory_costs
        else initial_capital * (cost_bps / 10000.0)
    )
    bench_equity = max(1.0, float(initial_capital) - b_entry_cost)
    equity_curve: list[dict[str, Any]] = []
    port_daily_rets: list[float] = []
    bench_daily_rets: list[float] = []
    dates_record: list[str] = []

    for i, dt in enumerate(sim_dates):
        if i > 0:
            prev_dt = sim_dates[i - 1]
            day_ret = float(blended_daily[i])
            b0 = eval_bench_series.loc[prev_dt]
            b1 = eval_bench_series.loc[dt]
            if not (pd.notna(b0) and pd.notna(b1) and float(b0) > 0 and float(b1) > 0):
                b0 = bench_price_series.loc[prev_dt]
                b1 = bench_price_series.loc[dt]
            b_ret = (float(b1) / float(b0)) - 1.0 if (pd.notna(b0) and pd.notna(b1) and float(b0) > 0) else 0.0
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

    # Aggregate FIFO tax & consistent terminal liquidation costs across all tranches
    tot_real_stcg = sum(a.get("realized_stcg_tax_inr", 0.0) for a in acct_list)
    tot_real_ltcg = sum(a.get("realized_ltcg_tax_inr", 0.0) for a in acct_list)
    tot_liq_cost = sum(a.get("unrealized_liq_cost_inr", 0.0) for a in acct_list)
    tot_unr_stcg = sum(a.get("unrealized_stcg_tax_inr", 0.0) for a in acct_list)
    tot_unr_ltcg = sum(a.get("unrealized_ltcg_tax_inr", 0.0) for a in acct_list)
    tot_stat_cost = sum(a.get("total_statutory_inr", 0.0) for a in acct_list)
    tot_slip_cost = sum(a.get("total_slippage_inr", 0.0) for a in acct_list)
    max_part = max((a.get("max_participation_pct", 0.0) for a in acct_list), default=0.0)

    final_port_post_tax = max(1.0, equity - tot_liq_cost - tot_real_stcg - tot_real_ltcg - tot_unr_stcg - tot_unr_ltcg)

    b_liq_cost = (
        0.0
        if cost_bps <= 0.0
        else compute_indian_delivery_cost("SELL", bench_equity, half_spread_bps=3.0, include_dp_charge=False)["total_cost_inr"]
        if use_statutory_costs
        else bench_equity * (cost_bps / 10000.0)
    )
    b_ledger = FifoTaxLedger()
    b_ledger.buy(eval_bench_sym, start_date, 1.0, initial_capital - b_entry_cost, buy_cost_inr=b_entry_cost)
    b_tx = b_ledger.sell(eval_bench_sym, end_date, 1.0, bench_equity, sell_cost_inr=b_liq_cost)
    final_bench_post_tax = max(1.0, bench_equity - b_liq_cost - b_tx["total_tax_inr"])

    p_rets = np.array(port_daily_rets, dtype=float)
    s_rets = np.array(bench_daily_rets, dtype=float)
    eqw_daily = prices_df[stock_cols].pct_change().mean(axis=1).reindex(dates_record).fillna(0.0).to_numpy(dtype=float)
    years = max(0.25, len(p_rets) / 252.0)

    def _stats(rets: np.ndarray, final_eq: float, final_post_tax_eq: float) -> dict[str, float]:
        tot_ret = ((final_eq / initial_capital) - 1.0) * 100.0
        cagr = ((final_eq / initial_capital) ** (1.0 / years) - 1.0) * 100.0
        after_tax_cagr = ((final_post_tax_eq / initial_capital) ** (1.0 / years) - 1.0) * 100.0
        vol = float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 2 else 0.0
        sharpe = float((np.mean(rets) * 252.0) / max(1e-9, vol))
        ex_rets = rets - INDIA_CASH_DAILY_YIELD
        excess_sharpe_rf62 = float((np.mean(ex_rets) * 252.0) / max(1e-9, float(np.std(ex_rets, ddof=1) * np.sqrt(252)))) if len(rets) > 2 else 0.0
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
            "excess_sharpe_rf62": round(excess_sharpe_rf62, 2),
            "sortino": round(sortino, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar": round(calmar, 2),
        }

    p_stats = _stats(p_rets, equity, final_port_post_tax)
    s_stats = _stats(s_rets, bench_equity, final_bench_post_tax)
    two_factor = compute_two_factor_alpha(p_rets, s_rets, eqw_daily, rf_daily=INDIA_CASH_DAILY_YIELD)
    four_factor = compute_iima_four_factor_alpha(dates_record, p_rets, ctx["four_factors_df"])
    p_stats.update(
        {
            "alpha_2f_pct": two_factor["alpha_2f_pct"],
            "beta_mkt": two_factor["beta_mkt"],
            "beta_size": two_factor["beta_size"],
            "alpha_2f_tstat": two_factor["alpha_tstat"],
            "alpha_4f_pct": four_factor["alpha_4f_pct"],
            "alpha_4f_tstat": four_factor["alpha_tstat"],
            "beta_hml": four_factor["beta_hml"],
            "beta_wml": four_factor["beta_wml"],
        }
    )

    avg_rebal = int(round(float(np.mean(rebal_counts)))) if rebal_counts else 1
    avg_turn = float(np.mean(turn_list)) if turn_list else 0.0
    actual_rank_buffer = min(top_k * 3, 40)

    return {
        "strategy_id": "v75_core_delivery_shield" if enforce_smart_money_shield else STRATEGY_ID_ML,
        "config": {
            "top_k": top_k,
            "rebalance_days": rebalance_days,
            "rank_buffer": actual_rank_buffer,
            "max_per_sector": max_per_sector,
            "inv_vol_weight": inv_vol_weight,
            "weighting_mode": eff_weighting,
            "hrp_alpha_tilt": hrp_alpha_tilt,
            "no_trade_band": no_trade_band,
            "n_tranches": len(offsets),
            "stagger_initial_deployment": stagger_initial_deployment,
            "vol_target_pct": round(vol_target * 100.0, 1) if vol_target else None,
            "cost_bps": cost_bps,
            "cost_model": "indian_statutory_delivery_adv" if (use_statutory_costs and cost_bps > 0) else "flat_bps",
            "tax_model": "fifo_indian_stcg_20.8_ltcg_13.0_1.25L_exemption",
            "regime_gate": regime_gate,
            "enable_dd_brake": enable_dd_brake,
            "gold_reserve_share_pct": round(gold_reserve_share * 100.0, 1),
            "enforce_smart_money_shield": enforce_smart_money_shield,
            "delivery_shield_window_days": 20,
            "delivery_shield_threshold_pct": 35.0,
            "benchmark_symbol": eval_bench_sym,
            "benchmark_type": eval_bench_type,
            "initial_capital": initial_capital,
            "period": f"{start_date} to {end_date}",
            "rebalance_count": avg_rebal,
            "avg_turnover_pct": round((avg_turn / max(1, avg_rebal)) * 100.0, 1),
            "ann_turnover_pct": round((avg_turn / years) * 100.0, 1),
        },
        "tax_and_cost_summary": {
            "statutory_fees_inr": round(tot_stat_cost, 2),
            "slippage_impact_inr": round(tot_slip_cost, 2),
            "terminal_liquidation_cost_inr": round(tot_liq_cost, 2),
            "realized_stcg_tax_inr": round(tot_real_stcg, 2),
            "realized_ltcg_tax_inr": round(tot_real_ltcg, 2),
            "unrealized_liq_tax_inr": round(tot_unr_stcg + tot_unr_ltcg, 2),
            "final_pre_tax_equity_inr": round(equity, 2),
            "final_post_tax_liquidation_inr": round(final_port_post_tax, 2),
            "benchmark_post_tax_liquidation_inr": round(final_bench_post_tax, 2),
            "max_adv_participation_pct": round(max_part, 4),
        },
        "two_factor_attribution": two_factor,
        "portfolio_metrics": p_stats,
        "spy_metrics": s_stats,
        "excess_cagr_pct": round(p_stats["cagr_pct"] - s_stats["cagr_pct"], 2),
        "excess_after_tax_cagr_pct": round(p_stats["after_tax_cagr_pct"] - s_stats["after_tax_cagr_pct"], 2),
        "equity_curve": equity_curve,
        "_daily_returns": p_rets.tolist(),
        "_spy_daily_returns": s_rets.tolist(),
        "_eqw_daily_returns": eqw_daily.tolist(),
        "_dates": dates_record,
        "_holdings_log": holdings_log,
    }


def run_capacity_analysis(
    db: Database,
    capitals: tuple[float, ...] = (1_000_000.0, 2_500_000.0, 10_000_000.0, 50_000_000.0, 250_000_000.0),
    top_k: int = 10,
    rebalance_days: int = 40,
    weighting_mode: str = "hrp",
) -> list[dict[str, Any]]:
    """Measure ADV participation, market impact, and post-tax CAGR scalability across capital tiers."""
    rows: list[dict[str, Any]] = []
    for cap in capitals:
        res = run_oos_portfolio_backtest(
            db,
            top_k=top_k,
            rebalance_days=rebalance_days,
            initial_capital=cap,
            use_statutory_costs=True,
            weighting_mode=weighting_mode,
        )
        pm = res["portfolio_metrics"]
        tc = res["tax_and_cost_summary"]
        rows.append(
            {
                "capital_inr": cap,
                "capital_label": f"₹{cap / 1e7:.2f} Cr" if cap >= 1e7 else f"₹{cap / 1e5:.1f} L",
                "cagr_pct": pm["cagr_pct"],
                "after_tax_cagr_pct": pm["after_tax_cagr_pct"],
                "sharpe": pm["sharpe"],
                "excess_sharpe_rf62": pm["excess_sharpe_rf62"],
                "max_drawdown_pct": pm["max_drawdown_pct"],
                "max_adv_participation_pct": tc["max_adv_participation_pct"],
                "total_slippage_pct_of_cap": round((tc["slippage_impact_inr"] / max(1.0, cap)) * 100.0, 2),
            }
        )
    return rows


def run_risk_control_ablation(
    db: Database,
    top_k: int = 10,
    rebalance_days: int = 40,
    weighting_mode: str = "hrp",
) -> list[dict[str, Any]]:
    """Independently validate each risk layer on identical TRI, statutory cost, and FIFO tax rules."""
    configs = [
        ("1. Base Top-10 Equal-Weight (No Risk Controls)", dict(inv_vol_weight=False, weighting_mode="equal", regime_gate=False, vol_target=None, enable_dd_brake=False, gold_reserve_share=0.0)),
        (f"2. + {weighting_mode.upper()} Weighting", dict(inv_vol_weight=True, weighting_mode=weighting_mode, regime_gate=False, vol_target=None, enable_dd_brake=False, gold_reserve_share=0.0)),
        ("3. + 6-Gauge India Macro Risk Gate", dict(inv_vol_weight=True, weighting_mode=weighting_mode, regime_gate=True, vol_target=None, enable_dd_brake=False, gold_reserve_share=0.0)),
        ("4. + 18% Ex-Ante Volatility Cap", dict(inv_vol_weight=True, weighting_mode=weighting_mode, regime_gate=True, vol_target=0.18, enable_dd_brake=False, gold_reserve_share=0.0)),
        ("5. + 70/30 LIQUIDCASE/GOLDBEES Reserve", dict(inv_vol_weight=True, weighting_mode=weighting_mode, regime_gate=True, vol_target=0.18, enable_dd_brake=False, gold_reserve_share=GOLDBEES_RESERVE_SHARE)),
        ("6. Full V7 (+ -10% Peak Drawdown Brake)", dict(inv_vol_weight=True, weighting_mode=weighting_mode, regime_gate=True, vol_target=0.18, enable_dd_brake=True, gold_reserve_share=GOLDBEES_RESERVE_SHARE)),
    ]
    out: list[dict[str, Any]] = []
    for label, kw in configs:
        res = run_oos_portfolio_backtest(db, top_k=top_k, rebalance_days=rebalance_days, **kw)
        pm = res["portfolio_metrics"]
        out.append(
            {
                "layer": label,
                "cagr_pct": pm["cagr_pct"],
                "after_tax_cagr_pct": pm["after_tax_cagr_pct"],
                "ann_vol_pct": pm["ann_vol_pct"],
                "sharpe": pm["sharpe"],
                "excess_sharpe_rf62": pm["excess_sharpe_rf62"],
                "alpha_2f_pct": pm["alpha_2f_pct"],
                "max_drawdown_pct": pm["max_drawdown_pct"],
                "calmar": pm["calmar"],
            }
        )
    return out
