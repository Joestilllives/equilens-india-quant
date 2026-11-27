"""Historical technical strategy backtester comparing active rules vs Buy & Hold.

Strictly causal execution:
  - Signals are evaluated on bar `i - 1` close and executed at bar `i` OPEN
    (eliminates same-close execution bias).
  - Applies consistent Indian statutory NSE delivery costs on every buy/sell leg
    and consistent terminal liquidation costs for both active strategy and Buy & Hold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .indian_costs import compute_indian_delivery_cost
from .indicators import compute_indicator_frame


def run_backtest(
    bars: list[dict[str, Any]],
    strategy: str = "sma_crossover",
    fast_period: int = 10,
    slow_period: int = 35,
    initial_capital: float = 10000.0,
    commission_bps: float = 15.0,
    use_statutory_indian_costs: bool = True,
) -> dict[str, Any]:
    """Run a strictly causal next-open bar-by-bar backtest vs Buy & Hold."""
    df = compute_indicator_frame(bars)
    if len(df) < max(slow_period + 12, 62):
        raise ValueError("Not enough historical bars to run backtest.")

    close = df["close"].astype(float)
    open_p = df["open"].astype(float).replace(0, np.nan).fillna(close)
    fast_arr = close.rolling(fast_period).mean().to_numpy(dtype=float)
    slow_arr = close.rolling(slow_period).mean().to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)
    open_arr = open_p.to_numpy(dtype=float)
    dates_arr = df["date"].astype(str).to_numpy()
    rsi_arr = df["rsi14"].to_numpy(dtype=float) if "rsi14" in df.columns else np.full(len(df), np.nan)
    sma50_arr = df["sma50"].to_numpy(dtype=float) if "sma50" in df.columns else np.full(len(df), np.nan)
    macd_arr = df["macd_hist"].to_numpy(dtype=float) if "macd_hist" in df.columns else np.full(len(df), np.nan)

    start_idx = max(slow_period + 1, 51)
    extra_slip_bps = max(0.0, float(commission_bps) - 15.0)

    def _leg_cost(side: str, gross: float) -> float:
        if gross <= 0.0:
            return 0.0
        if use_statutory_indian_costs:
            return compute_indian_delivery_cost(
                side=side,
                gross_value_inr=gross,
                half_spread_bps=3.0,
                extra_slippage_bps=extra_slip_bps,
                include_dp_charge=False,
            )["total_cost_inr"]
        return gross * (float(commission_bps) / 10000.0)

    cash = float(initial_capital)
    shares = 0.0
    entry_date = None
    entry_px = 0.0
    trades: list[dict[str, Any]] = []

    # Buy & Hold enters at bar `start_idx` OPEN with identical buy-side transaction cost
    bh_entry_open = float(open_arr[start_idx])
    bh_buy_cost = _leg_cost("BUY", initial_capital)
    bh_shares = max(0.0, initial_capital - bh_buy_cost) / max(1e-9, bh_entry_open)

    equity_curve: list[dict[str, Any]] = []
    peak = initial_capital
    max_dd = 0.0
    daily_rets: list[float] = []
    prev_eq = initial_capital

    for i in range(start_idx, len(df)):
        # Signal computed strictly from PRIOR bar `i - 1` (known before bar `i` open)
        sig_close = float(close_arr[i - 1])
        fill_px = float(open_arr[i])
        mark_px = float(close_arr[i])
        dt = str(dates_arr[i])

        buy_sig = False
        sell_sig = False

        if strategy == "sma_crossover":
            f_now, s_now = fast_arr[i - 1], slow_arr[i - 1]
            f_prev, s_prev = fast_arr[i - 2], slow_arr[i - 2]
            if np.isfinite(f_now) and np.isfinite(s_now) and np.isfinite(f_prev) and np.isfinite(s_prev):
                buy_sig = f_now > s_now and f_prev <= s_prev
                sell_sig = f_now < s_now and f_prev >= s_prev
        elif strategy == "rsi_trend":
            rsi = rsi_arr[i - 1]
            sma50 = sma50_arr[i - 1]
            if np.isfinite(rsi) and np.isfinite(sma50):
                buy_sig = sig_close > sma50 and rsi < 42.0
                sell_sig = rsi > 72.0 or sig_close < sma50 * 0.965
        elif strategy == "macd_momentum":
            mh = macd_arr[i - 1]
            pmh = macd_arr[i - 2]
            sma50 = sma50_arr[i - 1]
            if np.isfinite(mh) and np.isfinite(pmh) and np.isfinite(sma50):
                buy_sig = mh > 0.0 and pmh <= 0.0 and sig_close > sma50
                sell_sig = mh < 0.0 and pmh >= 0.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if buy_sig and shares == 0.0 and fill_px > 0:
            cost = _leg_cost("BUY", cash)
            net_cash = max(0.0, cash - cost)
            shares = net_cash / fill_px
            cash = 0.0
            entry_date = dt
            entry_px = fill_px
        elif sell_sig and shares > 0.0 and fill_px > 0:
            gross = shares * fill_px
            cost = _leg_cost("SELL", gross)
            cash = max(0.0, gross - cost)
            ret_pct = ((fill_px / max(1e-9, entry_px)) - 1.0) * 100.0
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "entry_price": round(entry_px, 2),
                    "exit_price": round(fill_px, 2),
                    "return_pct": round(ret_pct, 2),
                }
            )
            shares = 0.0
            entry_date = None

        eq = cash + shares * mark_px
        bh_eq = bh_shares * mark_px
        equity_curve.append(
            {
                "date": dt,
                "strategy": round(eq, 2),
                "buy_hold": round(bh_eq, 2),
            }
        )

        if eq > peak:
            peak = eq
        dd = ((eq - peak) / max(1e-9, peak)) * 100.0
        if dd < max_dd:
            max_dd = dd

        daily_rets.append((eq - prev_eq) / max(1e-9, prev_eq))
        prev_eq = eq

    # Consistent terminal liquidation costs for both Strategy and Buy & Hold
    last_px = float(df["close"].iloc[-1])
    if shares > 0.0 and entry_date is not None:
        gross_open = shares * last_px
        liq_cost = _leg_cost("SELL", gross_open)
        final_strat = cash + max(0.0, gross_open - liq_cost)
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": "Open (Mark-to-Market)",
                "entry_price": round(entry_px, 2),
                "exit_price": round(last_px, 2),
                "return_pct": round(((last_px / max(1e-9, entry_px)) - 1.0) * 100.0, 2),
            }
        )
    else:
        final_strat = equity_curve[-1]["strategy"]

    bh_gross_final = bh_shares * last_px
    bh_liq_cost = _leg_cost("SELL", bh_gross_final)
    final_bh = max(0.0, bh_gross_final - bh_liq_cost)
    equity_curve[-1]["strategy"] = round(final_strat, 2)
    equity_curve[-1]["buy_hold"] = round(final_bh, 2)

    strat_ret = ((final_strat / initial_capital) - 1.0) * 100.0
    bh_ret = ((final_bh / initial_capital) - 1.0) * 100.0

    rets_arr = np.array(daily_rets, dtype=float)
    sharpe = float((rets_arr.mean() / rets_arr.std()) * np.sqrt(252)) if rets_arr.std() > 1e-9 else 0.0
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0

    return {
        "strategy": strategy,
        "execution_convention": "signal_close_t_minus_1_fill_open_t",
        "cost_model": "indian_statutory_delivery" if use_statutory_indian_costs else "flat_bps",
        "fast_period": fast_period,
        "slow_period": slow_period,
        "initial_capital": initial_capital,
        "commission_bps": commission_bps,
        "metrics": {
            "strategy_return_pct": round(strat_ret, 2),
            "buy_hold_return_pct": round(bh_ret, 2),
            "alpha_pct": round(strat_ret - bh_ret, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 1),
            "trade_count": len(trades),
            "final_equity": round(final_strat, 2),
        },
        "equity_curve": equity_curve,
        "trades": list(reversed(trades)),
    }
