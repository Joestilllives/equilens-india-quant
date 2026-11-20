"""Historical strategy backtester comparing active technical rules vs Buy & Hold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .indicators import compute_indicator_frame


def run_backtest(
    bars: list[dict[str, Any]],
    strategy: str = "sma_crossover",
    fast_period: int = 10,
    slow_period: int = 35,
    initial_capital: float = 10000.0,
    commission_bps: float = 5.0,
) -> dict[str, Any]:
    """Run a bar-by-bar backtest with transaction costs and compare against Buy & Hold."""
    df = compute_indicator_frame(bars)
    if len(df) < max(slow_period + 10, 60):
        raise ValueError("Not enough historical bars to run backtest.")

    close = df["close"].astype(float)
    fast_sma = close.rolling(fast_period).mean()
    slow_sma = close.rolling(slow_period).mean()

    start_idx = max(slow_period, 50)
    fee_mult = commission_bps / 10000.0

    cash = float(initial_capital)
    shares = 0.0
    entry_date = None
    entry_px = 0.0
    trades: list[dict[str, Any]] = []

    bh_shares = initial_capital / float(df["close"].iloc[start_idx])
    equity_curve: list[dict[str, Any]] = []

    peak = initial_capital
    max_dd = 0.0
    daily_rets: list[float] = []
    prev_eq = initial_capital

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        px = float(row["close"])
        dt = str(row["date"])

        buy_sig = False
        sell_sig = False

        if strategy == "sma_crossover":
            f_now, s_now = fast_sma.iloc[i], slow_sma.iloc[i]
            f_prev, s_prev = fast_sma.iloc[i - 1], slow_sma.iloc[i - 1]
            if pd.notna(f_now) and pd.notna(s_now) and pd.notna(f_prev) and pd.notna(s_prev):
                buy_sig = f_now > s_now and f_prev <= s_prev
                sell_sig = f_now < s_now and f_prev >= s_prev
        elif strategy == "rsi_trend":
            rsi = row.get("rsi14")
            sma50 = row.get("sma50")
            if pd.notna(rsi) and pd.notna(sma50):
                buy_sig = px > sma50 and rsi < 42
                sell_sig = rsi > 72 or px < sma50 * 0.965
        elif strategy == "macd_momentum":
            mh = row.get("macd_hist")
            pmh = prev.get("macd_hist")
            sma50 = row.get("sma50")
            if pd.notna(mh) and pd.notna(pmh) and pd.notna(sma50):
                buy_sig = mh > 0 and pmh <= 0 and px > sma50
                sell_sig = mh < 0 and pmh >= 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if buy_sig and shares == 0.0:
            net_cash = cash * (1.0 - fee_mult)
            shares = net_cash / px
            cash = 0.0
            entry_date = dt
            entry_px = px
        elif sell_sig and shares > 0.0:
            gross = shares * px
            cash = gross * (1.0 - fee_mult)
            ret_pct = ((px / entry_px) - 1.0) * 100.0
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": dt,
                    "entry_price": round(entry_px, 2),
                    "exit_price": round(px, 2),
                    "return_pct": round(ret_pct, 2),
                }
            )
            shares = 0.0
            entry_date = None

        eq = cash + shares * px
        bh_eq = bh_shares * px
        equity_curve.append(
            {
                "date": dt,
                "strategy": round(eq, 2),
                "buy_hold": round(bh_eq, 2),
            }
        )

        if eq > peak:
            peak = eq
        dd = ((eq - peak) / peak) * 100.0
        if dd < max_dd:
            max_dd = dd

        daily_rets.append((eq - prev_eq) / max(1e-9, prev_eq))
        prev_eq = eq

    if shares > 0.0 and entry_date is not None:
        last_px = float(df["close"].iloc[-1])
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": "Open",
                "entry_price": round(entry_px, 2),
                "exit_price": round(last_px, 2),
                "return_pct": round(((last_px / entry_px) - 1.0) * 100.0, 2),
            }
        )

    final_strat = equity_curve[-1]["strategy"]
    final_bh = equity_curve[-1]["buy_hold"]
    strat_ret = ((final_strat / initial_capital) - 1.0) * 100.0
    bh_ret = ((final_bh / initial_capital) - 1.0) * 100.0

    rets_arr = np.array(daily_rets, dtype=float)
    sharpe = float((rets_arr.mean() / rets_arr.std()) * np.sqrt(252)) if rets_arr.std() > 1e-9 else 0.0
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0

    return {
        "strategy": strategy,
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
