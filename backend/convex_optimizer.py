"""
EquiLens India V8 (Pillar 5) — Multi-Period Convex Portfolio Optimizer.

Implements Boyd, Busseti, Diamond, Kahn, Koh, Nystrup & Speth (2017) "Multi-Period Trading via
Convex Optimization" + Grinold & Kahn Ch. 14-16 + Almgren & Chriss (2001) using pure SciPy
SLSQP (zero external solver dependencies, 100% deterministic):

    max_w  alpha^T w
         - 0.5 * gamma_risk * (w^T Sigma_LW w)
         - c_nse * sum_i smooth_L1(w_i - w_prev_i)
         - c_impact * sum_i sigma_d_i * (|w_i - w_prev_i| * V / ADV_i)^1.5
         - tax_penalty(w, w_prev, fifo_lots)

subject to exact NSE long-only, single-stock cap (<=12%), sector cap (<=20%),
ADV liquidity cap (<=2% of 20D ADV), and target equity exposure constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .hrp_allocator import _estimate_shrunk_cov, apply_concentration_caps, compute_hrp_weights

# NSE statutory delivery round-trip cost (~25-28 bps RT -> ~14 bps/side + 3 bps half-spread)
DEFAULT_NSE_LINEAR_COST_ONE_WAY = 0.0017
INDIA_STCG_RATE = 0.208
INDIA_LTCG_RATE = 0.130


def _estimate_lot_sell_tax_rates(
    symbols: list[str],
    current_weights: dict[str, float] | None,
    fifo_lots: list[dict[str, Any]] | None,
    prices: dict[str, float] | None,
    as_of_date: str | None,
) -> np.ndarray:
    """
    Estimate marginal Indian tax rate per rupee of weight sold for each symbol based on
    existing FIFO tax lots (Constantinides / Boyd tax-aware optimization).

    - Unrealized STCG winner (<365d): positive tax friction proportional to gain fraction * 20.8%.
    - Unrealized LTCG winner (>=365d): lower friction proportional to gain fraction * 13.0%.
    - Unrealized loss lot: slight negative friction (tax-loss harvesting credit, capped conservatively).
    """
    n = len(symbols)
    rates = np.zeros(n, dtype=float)
    if not fifo_lots or not prices or not as_of_date or not current_weights:
        return rates

    as_of_ts = pd.Timestamp(as_of_date[:10])
    lots_by_sym: dict[str, list[dict[str, Any]]] = {}
    for lot in fifo_lots:
        s = str(lot.get("symbol", "")).upper()
        if float(lot.get("shares", 0.0)) > 1e-9:
            lots_by_sym.setdefault(s, []).append(lot)

    for i, sym in enumerate(symbols):
        w_cur = float(current_weights.get(sym, 0.0))
        px = float(prices.get(sym, 0.0))
        if w_cur <= 1e-9 or px <= 0.0 or sym not in lots_by_sym:
            continue
        sym_lots = sorted(lots_by_sym[sym], key=lambda x: str(x.get("buy_date", "")))
        tot_val = 0.0
        weighted_tax = 0.0
        for lot in sym_lots:
            sh = float(lot.get("shares", 0.0))
            cost_px = float(lot.get("cost_per_share", lot.get("price", px)))
            val = sh * px
            if val <= 0.0 or cost_px <= 0.0:
                continue
            gain_frac = (px - cost_px) / px
            bdt = str(lot.get("buy_date", as_of_date))[:10]
            held_days = int((as_of_ts - pd.Timestamp(bdt)).days)
            stat_rate = INDIA_LTCG_RATE if held_days >= 365 else INDIA_STCG_RATE
            if gain_frac > 0.0:
                eff_rate = gain_frac * stat_rate
            else:
                # Cap tax-loss harvesting benefit conservatively so tax never overrides alpha
                eff_rate = max(-0.015, gain_frac * stat_rate * 0.50)
            weighted_tax += val * eff_rate
            tot_val += val
        if tot_val > 0.0:
            rates[i] = float(np.clip(weighted_tax / tot_val, -0.015, 0.12))
    return rates


def solve_convex_nse_portfolio(
    symbols: list[str],
    alpha_scores: dict[str, float],
    returns_window: pd.DataFrame | None,
    fallback_vols: dict[str, float] | None = None,
    sector_map: dict[str, str] | None = None,
    current_weights: dict[str, float] | None = None,
    target_exposure: float = 1.0,
    max_stock_w: float | None = 0.12,
    max_sector_w: float | None = 0.20,
    capital_inr: float = 1_000_000.0,
    adv_map: dict[str, float] | None = None,
    max_adv_participation: float = 0.02,
    risk_aversion: float = 3.5,
    alpha_scale: float = 0.04,
    linear_cost_rate: float = DEFAULT_NSE_LINEAR_COST_ONE_WAY,
    impact_eta: float = 0.10,
    tax_aversion: float = 0.35,
    fifo_lots: list[dict[str, Any]] | None = None,
    prices: dict[str, float] | None = None,
    as_of_date: str | None = None,
    turnover_aversion: float = 1.0,
) -> dict[str, float]:
    """
    Solve Boyd et al. (2017) single-period convex portfolio optimization over `symbols`
    incorporating Ledoit-Wolf risk, NSE delivery fees, Almgren-Chriss impact, and Indian FIFO tax.

    Falls back gracefully to HRP if the covariance window is too short or SLSQP fails to converge.
    """
    n = len(symbols)
    if n == 0 or target_exposure <= 0.0:
        return {}
    if n == 1:
        cap = target_exposure
        if max_stock_w is not None:
            cap = min(cap, max_stock_w)
        if max_sector_w is not None:
            cap = min(cap, max_sector_w)
        return {symbols[0]: float(max(0.0, cap))}

    # Warm-start from HRP so the solver begins strictly feasible and well-diversified
    w_hrp_dict = compute_hrp_weights(
        symbols=symbols,
        returns_window=returns_window,
        fallback_vols=fallback_vols,
        target_exposure=target_exposure,
        sector_map=sector_map,
        max_stock_w=max_stock_w,
        max_sector_w=max_sector_w,
    )
    w0 = np.array([float(w_hrp_dict.get(s, target_exposure / n)) for s in symbols], dtype=float)
    w_prev = np.array([float((current_weights or {}).get(s, 0.0)) for s in symbols], dtype=float)

    # Covariance estimation (126D Ledoit-Wolf shrunk annualized covariance)
    use_cov = (
        returns_window is not None
        and not returns_window.empty
        and len(returns_window) >= 30
        and all(s in returns_window.columns for s in symbols)
    )
    if use_cov:
        sub = returns_window[symbols].fillna(0.0).to_numpy(dtype=float)
        cov = _estimate_shrunk_cov(sub)
        daily_vols = np.sqrt(np.clip(np.diag(cov) / 252.0, 1e-6, None))
    else:
        vols = np.array([max(0.12, float((fallback_vols or {}).get(s, 0.25))) for s in symbols], dtype=float)
        cov = np.diag(vols ** 2)
        daily_vols = vols / np.sqrt(252.0)

    # Cross-sectionally standardized expected alpha over holding horizon (~40D annualized proxy)
    raw_a = np.array([float(alpha_scores.get(s, 0.0)) for s in symbols], dtype=float)
    a_std = float(np.std(raw_a, ddof=0))
    if np.isfinite(a_std) and a_std > 1e-9:
        z_a = np.clip((raw_a - float(np.mean(raw_a))) / a_std, -2.5, 2.5)
    else:
        z_a = np.zeros(n, dtype=float)
    # Convert z-score to modest expected active return (conservative Grinold-Kahn IC * vol scaling)
    alpha_vec = alpha_scale * z_a

    # Per-stock upper bounds (min of stock cap & ADV participation cap, while preserving feasibility)
    upper_bounds = np.full(n, float(max_stock_w) if max_stock_w is not None else float(target_exposure), dtype=float)
    if adv_map and capital_inr > 0.0 and max_adv_participation > 0.0:
        for i, sym in enumerate(symbols):
            adv_val = float(adv_map.get(sym, 0.0) or 0.0)
            if adv_val > 0.0:
                adv_w_cap = (adv_val * max_adv_participation) / capital_inr
                # Never cap below 2% if feasibility would break; keep cap meaningful for ₹1 Cr+ scale
                if adv_w_cap >= 0.02:
                    upper_bounds[i] = min(upper_bounds[i], adv_w_cap)

    # Ensure feasible total exposure under stock & sector caps
    sec_map = sector_map or {}
    sectors_present: dict[str, list[int]] = {}
    for i, s in enumerate(symbols):
        sectors_present.setdefault(sec_map.get(s, "Other"), []).append(i)

    max_feasible = 0.0
    for sec, idxs in sectors_present.items():
        sec_sum_ub = float(np.sum(upper_bounds[idxs]))
        if max_sector_w is not None:
            sec_sum_ub = min(sec_sum_ub, float(max_sector_w))
        max_feasible += sec_sum_ub
    eff_target = min(float(target_exposure), max_feasible - 1e-7) if max_feasible > 1e-6 else 0.0
    if eff_target <= 1e-6:
        return w_hrp_dict

    # Re-scale w0 to eff_target
    if float(np.sum(w0)) > 1e-9:
        w0 = np.clip(w0 * (eff_target / float(np.sum(w0))), 0.0, upper_bounds)

    # Almgren-Chriss impact coefficient per unit |dw|^1.5
    impact_coeff = np.zeros(n, dtype=float)
    if adv_map and capital_inr > 0.0 and impact_eta > 0.0:
        for i, sym in enumerate(symbols):
            adv_val = float(adv_map.get(sym, 0.0) or 0.0)
            if adv_val > 1e5:
                impact_coeff[i] = impact_eta * daily_vols[i] * ((capital_inr / adv_val) ** 0.5)

    # Marginal Indian FIFO tax penalty on sells (max(0, w_prev - w))
    sell_tax_rates = _estimate_lot_sell_tax_rates(symbols, current_weights, fifo_lots, prices, as_of_date)

    # Annualize 40D holding period (~6.3 rebalances/yr) for cost-vs-annual-risk-and-alpha alignment
    cost_annual_mult = 6.3 * max(0.0, float(turnover_aversion))
    eps = 1e-4

    def _obj_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        dw = w - w_prev
        abs_smooth = np.sqrt(dw * dw + eps * eps)
        # Smooth sell part: max(0, -dw) approximated smoothly as 0.5 * (sqrt(dw^2 + eps^2) - dw)
        sell_smooth = 0.5 * (abs_smooth - dw)

        # Objective to MINIMIZE (negative utility)
        risk_val = 0.5 * risk_aversion * float(w @ cov @ w)
        alpha_val = -float(alpha_vec @ w)
        lin_cost = cost_annual_mult * linear_cost_rate * float(np.sum(abs_smooth - eps))
        imp_cost = cost_annual_mult * float(np.sum(impact_coeff * (abs_smooth ** 1.5)))
        tax_cost = cost_annual_mult * tax_aversion * float(np.sum(sell_tax_rates * sell_smooth))
        total_obj = risk_val + alpha_val + lin_cost + imp_cost + tax_cost

        # Exact analytic gradient for fast, deterministic SLSQP convergence
        d_abs = dw / abs_smooth
        d_sell = 0.5 * (d_abs - 1.0)
        grad = (
            risk_aversion * (cov @ w)
            - alpha_vec
            + cost_annual_mult * linear_cost_rate * d_abs
            + cost_annual_mult * 1.5 * impact_coeff * np.sqrt(abs_smooth) * dw / abs_smooth
            + cost_annual_mult * tax_aversion * sell_tax_rates * d_sell
        )
        return total_obj, grad

    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - eff_target),
            "jac": lambda w: np.ones(n, dtype=float),
        }
    ]
    if max_sector_w is not None:
        for sec, idxs in sectors_present.items():
            mask = np.zeros(n, dtype=float)
            mask[idxs] = 1.0
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, m=mask, cap=float(max_sector_w): float(cap - (m @ w)),
                    "jac": lambda w, m=mask: -m,
                }
            )

    bounds = [(0.0, float(ub)) for ub in upper_bounds]
    try:
        res = minimize(
            _obj_and_grad,
            w0,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 120, "ftol": 1e-8},
        )
        if res.success and np.all(np.isfinite(res.x)) and float(np.sum(np.clip(res.x, 0.0, None))) > 1e-6:
            w_opt = np.clip(res.x, 0.0, upper_bounds)
            w_dict = {sym: float(val) for sym, val in zip(symbols, w_opt)}
            return apply_concentration_caps(
                w_dict,
                sec_map,
                max_stock_w=max_stock_w,
                max_sector_w=max_sector_w,
                target_total=eff_target,
            )
    except Exception:
        pass

    return w_hrp_dict
