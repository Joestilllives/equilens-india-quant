"""Quantitative & Model Integrity Stress Testing Suite."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DEFAULT_COST_BPS, VOL_INDEX_SYMBOL
from .database import Database
from .portfolio_backtester import build_backtest_market_context, run_oos_portfolio_backtest
from .shortlister import build_shortlist


def run_quantitative_stress_tests(
    db: Database,
    top_k: int = 10,
    rebalance_days: int = 40,
    regime_gate: bool = True,
    risk_profile: str = "balanced",
) -> dict[str, Any]:
    """
    Execute 4 quantitative stress tests on the NSE out-of-sample portfolio & ranking engine:
      1. STT + Brokerage + Slippage Friction Stress (0 / 15 / 30 / 50 bps)
      2. Historical India VIX & NIFTY 50 Drawdown Replay
      3. 1,000-Path Block-Bootstrap Tail Risk & Worst-Case Drawdown Stress
      4. Weight Perturbation & Ranking Stability Stress
    """
    vol_target = None if risk_profile.lower() == "growth" else 0.18
    market_ctx = build_backtest_market_context(db)
    base = run_oos_portfolio_backtest(
        db,
        top_k=top_k,
        rebalance_days=rebalance_days,
        cost_bps=DEFAULT_COST_BPS,
        regime_gate=regime_gate,
        vol_target=vol_target,
        market_ctx=market_ctx,
    )
    p_rets = np.array(base["_daily_returns"], dtype=float)
    s_rets = np.array(base["_spy_daily_returns"], dtype=float)
    dates = base["_dates"]

    # 1. Indian STT + Brokerage + Slippage Stress
    slippage_rows = []
    for bps in (0.0, 15.0, 30.0, 50.0):
        res = run_oos_portfolio_backtest(
            db,
            top_k=top_k,
            rebalance_days=rebalance_days,
            cost_bps=bps,
            regime_gate=regime_gate,
            vol_target=vol_target,
            market_ctx=market_ctx,
        )
        pm = res["portfolio_metrics"]
        slippage_rows.append(
            {
                "cost_bps": bps,
                "cagr_pct": pm["cagr_pct"],
                "sharpe": pm["sharpe"],
                "max_drawdown_pct": pm["max_drawdown_pct"],
                "excess_cagr_pct": res["excess_cagr_pct"],
                "survives_friction": pm["cagr_pct"] > 0 and pm["sharpe"] > 0.3,
            }
        )

    # 2. Historical India VIX & NIFTY 50 Drawdown Replay
    macro = db.get_macro_series()
    vix_map = macro.get(VOL_INDEX_SYMBOL, {})
    vix_arr = np.array([vix_map.get(d, 15.0) for d in dates], dtype=float)

    spy_cum = np.cumprod(1.0 + s_rets)
    spy_peaks = np.maximum.accumulate(spy_cum)
    spy_dd = ((spy_cum - spy_peaks) / np.maximum(1e-9, spy_peaks)) * 100.0

    def _slice_perf(mask: np.ndarray, label: str) -> dict[str, Any]:
        days = int(np.sum(mask))
        if days < 5:
            return {"regime": label, "trading_days": days, "portfolio_ret_pct": 0.0, "spy_ret_pct": 0.0, "spread_pct": 0.0}
        pr = (float(np.prod(1.0 + p_rets[mask])) - 1.0) * 100.0
        sr = (float(np.prod(1.0 + s_rets[mask])) - 1.0) * 100.0
        return {
            "regime": label,
            "trading_days": days,
            "portfolio_ret_pct": round(pr, 2),
            "spy_ret_pct": round(sr, 2),
            "spread_pct": round(pr - sr, 2),
        }

    regime_replay = [
        _slice_perf(vix_arr < 14.0, "Calm Bull (India VIX < 14)"),
        _slice_perf((vix_arr >= 14.0) & (vix_arr < 19.0), "Normal Volatility (VIX 14–19)"),
        _slice_perf(vix_arr >= 19.0, "High India VIX Stress (VIX >= 19)"),
        _slice_perf(spy_dd <= -5.0, "NIFTY 50 Correction Days (DD <= -5%)"),
    ]

    # 3. 1,000-Path Block-Bootstrap Tail Risk Stress (5-day blocks preserve volatility clustering)
    rng = np.random.default_rng(42)
    n_paths = 1000
    block = 5
    horizon_days = min(252, len(p_rets))
    boot_mdds = np.empty(n_paths, dtype=float)
    boot_1y_rets = np.empty(n_paths, dtype=float)

    max_start = max(1, len(p_rets) - block)
    n_blocks = int(np.ceil(horizon_days / block))
    for p in range(n_paths):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([p_rets[st : st + block] for st in starts])[:horizon_days]
        cum = np.cumprod(1.0 + sample)
        peaks = np.maximum.accumulate(cum)
        dds = ((cum - peaks) / np.maximum(1e-9, peaks)) * 100.0
        boot_mdds[p] = float(np.min(dds))
        boot_1y_rets[p] = (float(cum[-1]) - 1.0) * 100.0

    var_95 = float(np.percentile(p_rets, 5) * 100.0)
    tail = p_rets[p_rets <= np.percentile(p_rets, 5)]
    cvar_95 = float(np.mean(tail) * 100.0) if len(tail) else var_95

    bootstrap_stress = {
        "simulations": n_paths,
        "horizon_trading_days": horizon_days,
        "median_1y_return_pct": round(float(np.median(boot_1y_rets)), 2),
        "p05_1y_return_pct": round(float(np.percentile(boot_1y_rets, 5)), 2),
        "prob_positive_1y_pct": round(float(np.mean(boot_1y_rets > 0)) * 100.0, 1),
        "median_max_dd_pct": round(float(np.median(boot_mdds)), 2),
        "worst_95_max_dd_pct": round(float(np.percentile(boot_mdds, 5)), 2),
        "worst_99_max_dd_pct": round(float(np.percentile(boot_mdds, 1)), 2),
        "daily_var_95_pct": round(var_95, 2),
        "daily_cvar_95_pct": round(cvar_95, 2),
    }

    # 4. Ranking Stability Stress (Perturb factor weights +/- 30% & check Top-10 overlap)
    base_list = build_shortlist(db)
    base_top10 = {x["symbol"] for x in base_list[:10]}
    overlaps = []
    perturbations = [
        {"ml": 0.45, "momentum": 0.20, "valuation": 0.15, "quality": 0.10, "sentiment": 0.10},
        {"ml": 0.20, "momentum": 0.35, "valuation": 0.15, "quality": 0.15, "sentiment": 0.15},
        {"ml": 0.25, "momentum": 0.20, "valuation": 0.25, "quality": 0.20, "sentiment": 0.10},
        {"ml": 0.25, "momentum": 0.25, "valuation": 0.10, "quality": 0.15, "sentiment": 0.25},
    ]
    for pw in perturbations:
        alt = build_shortlist(db, weights=pw)
        alt_top10 = {x["symbol"] for x in alt[:10]}
        jaccard = len(base_top10.intersection(alt_top10)) / max(1, len(base_top10.union(alt_top10)))
        overlaps.append(jaccard)

    avg_jaccard = round(float(np.mean(overlaps)) * 100.0, 1)

    # Overall Stress Grade
    checks_passed = 0
    checks_total = 4
    if slippage_rows[1]["cagr_pct"] > 0:
        checks_passed += 1
    if bootstrap_stress["prob_positive_1y_pct"] >= 55.0:
        checks_passed += 1
    if bootstrap_stress["worst_95_max_dd_pct"] > -35.0:
        checks_passed += 1
    if avg_jaccard >= 45.0:
        checks_passed += 1

    return {
        "overall": {
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "verdict": "Resilient" if checks_passed >= 3 else "Moderate Risk" if checks_passed == 2 else "Fragile",
            "ranking_stability_jaccard_pct": avg_jaccard,
        },
        "base_portfolio": {
            "config": base["config"],
            "portfolio_metrics": base["portfolio_metrics"],
            "spy_metrics": base["spy_metrics"],
            "excess_cagr_pct": base["excess_cagr_pct"],
            "equity_curve": base["equity_curve"],
        },
        "slippage_stress": slippage_rows,
        "regime_replay": regime_replay,
        "bootstrap_stress": bootstrap_stress,
    }
