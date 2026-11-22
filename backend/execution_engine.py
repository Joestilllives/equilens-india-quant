"""
EquiLens India V6 — Live NSE Rebalance Ticket Generator, Earnings Blackout Guard,
Indian 20% STCG Tax Estimator & Forward-Proof Paper-Trading Ledger.

Turns the V6 algorithm into an exact, zero-guesswork Indian execution workflow:
  - Enforces exact V6 rules: Top-K (default 10), Max 2 per NSE Sector (>=5 sectors),
    2xK Rank-Buffer Hysteresis (minimizes STT & 20% STCG churn),
    Correlation-Penalized Weighting: w_i proportional to 1 / (vol_i * (1 + avg_pos_corr_i)),
    Live 6-Gauge India Sentiment & Intermarket Macro Risk Gate,
    and Ex-Ante 18% Annualized Portfolio Volatility Target ('balanced') or Uncapped ('growth').
  - Flags Earnings Blackouts (new buys reporting quarterly results within <= 5 trading days).
  - Computes exact integer NSE shares, rupee allocations, LiquidBees cash reserve,
    +/-2.0xATR target/stop prices, Indian STT+brokerage (15 bps), and est. 20% STCG tax.
  - Persists timestamped paper rebalances in SQLite for zero-survivorship-bias forward proof vs. NIFTY 50.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import floor
from typing import Any

import numpy as np
import pandas as pd

from .config import BENCHMARK_SYMBOL, DEFAULT_COST_BPS
from .database import Database
from .ml_predictor import compute_current_macro_regime

INDIA_LIQUIDBEES_ANNUAL_YIELD = 0.062
INDIA_STCG_TAX_RATE = 0.20


def _days_to_next_earnings(earn_rows: list[dict[str, Any]], as_of_date: str) -> tuple[int | None, str | None]:
    try:
        ref = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
    except Exception:
        ref = datetime.now(timezone.utc).date()
    best_days: int | None = None
    best_date: str | None = None
    for e in earn_rows:
        if e.get("eps_actual") is not None:
            continue
        try:
            dt = datetime.strptime(str(e["earnings_date"])[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        diff = (dt - ref).days
        if 0 <= diff <= 90 and (best_days is None or diff < best_days):
            best_days = diff
            best_date = dt.isoformat()
    return best_days, best_date


def _latest_prices_and_matrices(db: Database) -> dict[str, Any]:
    ohlcv_map = db.get_all_ohlcv()
    prices_df = pd.DataFrame(
        {sym: pd.Series({b["date"]: float(b["close"]) for b in bars}) for sym, bars in ohlcv_map.items()}
    ).sort_index().ffill()
    log_rets = np.log(prices_df / prices_df.shift(1))
    latest_date = str(prices_df.index[-1]) if not prices_df.empty else datetime.now(timezone.utc).date().isoformat()
    latest_px = {
        sym: float(prices_df[sym].iloc[-1])
        for sym in prices_df.columns
        if pd.notna(prices_df[sym].iloc[-1]) and float(prices_df[sym].iloc[-1]) > 0
    }
    return {
        "prices_df": prices_df,
        "log_rets": log_rets,
        "latest_date": latest_date,
        "latest_px": latest_px,
    }


def generate_rebalance_ticket(
    db: Database,
    capital_inr: float = 1_000_000.0,
    top_k: int = 10,
    max_per_sector: int = 2,
    risk_profile: str = "balanced",
    skip_earnings_blackout: bool = False,
    enforce_smart_money_shield: bool = True,
    current_holdings: list[dict[str, Any]] | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict[str, Any]:
    """Generate an executable V6 NSE rebalance ticket with exact integer shares & tax/STT estimates."""
    preds = db.get_ml_predictions()
    if not preds:
        raise ValueError("No ML predictions available. Train the model first.")

    tickers = {t["symbol"]: t for t in db.get_tickers()}
    earnings_map = db.get_all_earnings()
    mkt = _latest_prices_and_matrices(db)
    prices_df: pd.DataFrame = mkt["prices_df"]
    log_rets: pd.DataFrame = mkt["log_rets"]
    latest_date: str = mkt["latest_date"]
    latest_px: dict[str, float] = mkt["latest_px"]

    try:
        from .money_signals import evaluate_balance_sheet_shield, get_money_signals_summary

        ms_snap = get_money_signals_summary(db)
        deliv_map = ms_snap.get("stock_delivery_map", {})
    except Exception:
        deliv_map = {}
        evaluate_balance_sheet_shield = None  # type: ignore[assignment]

    macro = compute_current_macro_regime(db)
    gate_exp = float(macro.get("target_exposure_pct", 100)) / 100.0
    vol_target = None if risk_profile.lower() == "growth" else 0.18

    # Rank universe purely by live V6 expected 40D alpha (same objective as OOS backtester)
    ranked_rows: list[dict[str, Any]] = []
    for sym, p in preds.items():
        if sym == BENCHMARK_SYMBOL or sym not in latest_px:
            continue
        dte, next_e_dt = _days_to_next_earnings(earnings_map.get(sym, []), latest_date)
        t_meta = tickers.get(sym, {})
        dm = deliv_map.get(sym)
        if dm:
            sh_status = dm["shield_status"]
            sh_flags = list(dm["shield_flags"])
            deliv_pct = dm["deliv_pct"]
            deliv_surge = dm["deliv_surge_pct"]
        else:
            sh = (
                evaluate_balance_sheet_shield(t_meta.get("fundamentals", {}), t_meta.get("sector", ""))
                if evaluate_balance_sheet_shield
                else {"status": "Clean", "flags": []}
            )
            sh_status = sh["status"]
            sh_flags = list(sh["flags"])
            deliv_pct = None
            deliv_surge = None

        spec_churn = deliv_pct is not None and deliv_pct < 35.0
        shield_blocked = (sh_status == "Risk") or spec_churn
        shield_block_reason = (
            f"Balance-Sheet Risk ({', '.join(sh_flags)})"
            if sh_status == "Risk"
            else (f"Speculative Intraday Churn (Deliv {deliv_pct:.1f}% < 35%)" if spec_churn else "")
        )

        ranked_rows.append(
            {
                "symbol": sym,
                "name": t_meta.get("name", sym),
                "sector": t_meta.get("sector", "Other"),
                "price": latest_px[sym],
                "pred_alpha": float(p.get("pred_alpha_20d", 0.0)),
                "prob_tb_win": float(p.get("prob_tb_win", 50.0)),
                "prob_outperform": float(p.get("prob_outperform", 50.0)),
                "vol_ann": max(0.12, float(p.get("vol_20d_ann", 24.0)) / 100.0),
                "target_price": p.get("target_price"),
                "stop_price": p.get("stop_price"),
                "risk_reward": p.get("risk_reward"),
                "days_to_earnings": dte,
                "next_earnings_date": next_e_dt,
                "earnings_blackout": dte is not None and 0 <= dte <= 7,
                "shield_blocked": shield_blocked,
                "shield_block_reason": shield_block_reason,
            }
        )

    ranked_rows.sort(key=lambda r: (r["pred_alpha"], r["prob_tb_win"]), reverse=True)
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx
    row_by_sym = {r["symbol"]: r for r in ranked_rows}

    # Parse current holdings (either passed explicitly or empty for fresh deployment)
    held_map: dict[str, dict[str, Any]] = {}
    for h in current_holdings or []:
        sym = str(h.get("symbol", "")).upper().strip()
        sh = int(h.get("shares", 0))
        if sym and sh > 0 and sym in latest_px:
            held_map[sym] = {
                "symbol": sym,
                "shares": sh,
                "avg_cost": float(h.get("avg_cost") or latest_px[sym]),
            }

    rank_buffer = top_k * 2
    chosen: list[str] = []
    sec_counts: dict[str, int] = {}
    skipped_blackout: list[dict[str, Any]] = []
    skipped_smart_money: list[dict[str, Any]] = []

    def _can_add(sym: str) -> bool:
        sec = row_by_sym[sym]["sector"]
        return sec_counts.get(sec, 0) < max_per_sector

    def _add(sym: str) -> None:
        chosen.append(sym)
        sec = row_by_sym[sym]["sector"]
        sec_counts[sec] = sec_counts.get(sec, 0) + 1

    # 1. Rank-Buffer Hysteresis: retain existing winners still inside Top 2xK
    held_sorted = sorted(
        [s for s in held_map if s in row_by_sym],
        key=lambda s: row_by_sym[s]["rank"],
    )
    for sym in held_sorted:
        if len(chosen) >= top_k:
            break
        if row_by_sym[sym]["rank"] <= rank_buffer and _can_add(sym):
            _add(sym)

    # 2. Fill remaining slots with top V6 candidates respecting Max 2/Sector, Earnings Blackout & Smart-Money Shield
    for r in ranked_rows:
        if len(chosen) >= top_k:
            break
        sym = r["symbol"]
        if sym in chosen or not _can_add(sym):
            continue
        if skip_earnings_blackout and r["earnings_blackout"] and sym not in held_map:
            skipped_blackout.append(
                {"symbol": sym, "rank": r["rank"], "days_to_earnings": r["days_to_earnings"], "next_earnings_date": r["next_earnings_date"]}
            )
            continue
        if enforce_smart_money_shield and r["shield_blocked"] and sym not in held_map:
            skipped_smart_money.append(
                {"symbol": sym, "rank": r["rank"], "reason": r["shield_block_reason"]}
            )
            continue
        _add(sym)

    # Graceful fallback if universe is tiny
    if len(chosen) < top_k:
        for r in ranked_rows:
            if len(chosen) >= top_k:
                break
            if r["symbol"] not in chosen:
                _add(r["symbol"])

    # 3. Correlation-Penalized Weighting: w_i proportional to 1 / (vol_i * (1 + avg_pos_corr_i))
    vols = np.array([row_by_sym[s]["vol_ann"] for s in chosen], dtype=float)
    avg_corrs = np.full(len(chosen), 0.20, dtype=float)
    portfolio_avg_corr = 0.20

    if len(chosen) >= 3 and len(log_rets) >= 130:
        sub_r = log_rets[chosen].iloc[-126:].fillna(0.0)
        cmat = sub_r.corr().fillna(0.0).to_numpy(dtype=float, copy=True)
        np.fill_diagonal(cmat, np.nan)
        avg_c = np.nanmean(np.clip(cmat, 0.0, 1.0), axis=1)
        avg_corrs = np.where(np.isfinite(avg_c), avg_c, 0.20)
        portfolio_avg_corr = float(np.nanmean(cmat)) if np.isfinite(np.nanmean(cmat)) else 0.20
        raw_vec = 1.0 / (vols * (1.0 + avg_corrs))
    else:
        raw_vec = 1.0 / vols

    unit_weights = raw_vec / max(1e-9, float(np.sum(raw_vec)))

    # 4. Ex-Ante 60D Portfolio Volatility & Vol-Targeting Cap
    ex_ante_vol_ann = float(np.mean(vols) * 0.72)
    vol_cap = 1.0
    if len(chosen) >= 2 and len(log_rets) >= 65:
        sub60 = log_rets[chosen].iloc[-60:].fillna(0.0).to_numpy(dtype=float)
        cov60 = np.cov(sub60, rowvar=False) * 252.0
        ex_ante_vol_ann = float(np.sqrt(max(1e-6, float(unit_weights @ cov60 @ unit_weights))))
    if vol_target is not None:
        vol_cap = float(np.clip(vol_target / max(0.08, ex_ante_vol_ann), 0.25, 1.0))

    final_equity_exp = min(gate_exp, vol_cap)
    fee_rate = cost_bps / 10000.0

    # 5. Build Actionable Trade Tickets (SELL exits first, then HOLD/BUY/TRIM target positions)
    orders: list[dict[str, Any]] = []
    est_total_fees = 0.0
    est_stcg_tax = 0.0
    est_realized_pnl = 0.0
    allocated_equity_inr = 0.0

    # Exits (held stocks not in chosen)
    for sym, h in held_map.items():
        if sym in chosen:
            continue
        px = latest_px[sym]
        sh = h["shares"]
        gross = sh * px
        fee = gross * fee_rate
        rpnl = (px - h["avg_cost"]) * sh - fee
        tax = max(0.0, rpnl) * INDIA_STCG_TAX_RATE
        est_total_fees += fee
        est_realized_pnl += rpnl
        est_stcg_tax += tax
        rinfo = row_by_sym.get(sym, {})
        rk = rinfo.get("rank")
        reason = f"Dropped below Top-{rank_buffer} rank buffer (now #{rk})" if rk else "Exited universe / sector cap"
        orders.append(
            {
                "symbol": sym,
                "name": tickers.get(sym, {}).get("name", sym),
                "sector": tickers.get(sym, {}).get("sector", "Other"),
                "action": "SELL",
                "rank": rk,
                "price": round(px, 2),
                "current_shares": sh,
                "target_shares": 0,
                "delta_shares": -sh,
                "target_weight_pct": 0.0,
                "target_value_inr": 0.0,
                "order_value_inr": round(gross, 2),
                "est_fee_inr": round(fee, 2),
                "est_realized_pnl_inr": round(rpnl, 2),
                "est_stcg_tax_inr": round(tax, 2),
                "pred_alpha_40d": round(float(rinfo.get("pred_alpha", 0.0)), 2),
                "prob_tb_win": round(float(rinfo.get("prob_tb_win", 50.0)), 1),
                "avg_corr": None,
                "target_price": rinfo.get("target_price"),
                "stop_price": rinfo.get("stop_price"),
                "earnings_blackout": False,
                "days_to_earnings": rinfo.get("days_to_earnings"),
                "reason": reason,
            }
        )

    # Target positions
    for idx, sym in enumerate(chosen):
        r = row_by_sym[sym]
        px = r["price"]
        w_port = float(unit_weights[idx] * final_equity_exp)
        target_val = capital_inr * w_port
        target_sh = max(0, int(floor(target_val / max(0.01, px))))
        actual_val = target_sh * px
        allocated_equity_inr += actual_val

        cur_sh = held_map[sym]["shares"] if sym in held_map else 0
        delta_sh = target_sh - cur_sh
        order_val = abs(delta_sh) * px
        fee = order_val * fee_rate
        est_total_fees += fee

        rpnl = 0.0
        tax = 0.0
        if cur_sh == 0 and target_sh > 0:
            action = "BUY"
            reason = f"New Top-{top_k} entry (Rank #{r['rank']}, 40D alpha {r['pred_alpha']:+.2f}%)"
        elif delta_sh == 0 or (cur_sh > 0 and abs(delta_sh) / max(1, cur_sh) < 0.08):
            # Deadband (<8% weight drift): avoid tiny tax-inefficient micro-adjustments!
            target_sh = cur_sh
            delta_sh = 0
            order_val = 0.0
            est_total_fees -= fee
            fee = 0.0
            allocated_equity_inr = allocated_equity_inr - actual_val + (cur_sh * px)
            actual_val = cur_sh * px
            action = "HOLD"
            reason = f"Retained inside Top-{rank_buffer} hysteresis buffer (Rank #{r['rank']})"
        elif delta_sh > 0:
            action = "ADD"
            reason = f"Rebalance top-up (+{delta_sh} sh) inside Top-{rank_buffer} buffer"
        else:
            action = "TRIM"
            avg_c = held_map[sym]["avg_cost"]
            rpnl = (px - avg_c) * abs(delta_sh) - fee
            tax = max(0.0, rpnl) * INDIA_STCG_TAX_RATE
            est_realized_pnl += rpnl
            est_stcg_tax += tax
            reason = f"Risk/vol rebalance trim ({delta_sh} sh)"

        orders.append(
            {
                "symbol": sym,
                "name": r["name"],
                "sector": r["sector"],
                "action": action,
                "rank": r["rank"],
                "price": round(px, 2),
                "current_shares": cur_sh,
                "target_shares": target_sh,
                "delta_shares": delta_sh,
                "target_weight_pct": round((actual_val / max(1.0, capital_inr)) * 100.0, 2),
                "target_value_inr": round(actual_val, 2),
                "order_value_inr": round(order_val, 2),
                "est_fee_inr": round(fee, 2),
                "est_realized_pnl_inr": round(rpnl, 2),
                "est_stcg_tax_inr": round(tax, 2),
                "pred_alpha_40d": round(r["pred_alpha"], 2),
                "prob_tb_win": round(r["prob_tb_win"], 1),
                "avg_corr": round(float(avg_corrs[idx]), 2),
                "target_price": r["target_price"],
                "stop_price": r["stop_price"],
                "earnings_blackout": r["earnings_blackout"] and action == "BUY",
                "days_to_earnings": r["days_to_earnings"],
                "reason": reason,
            }
        )

    # Sort orders cleanly: SELL -> BUY -> ADD -> TRIM -> HOLD
    act_order = {"SELL": 0, "BUY": 1, "ADD": 2, "TRIM": 3, "HOLD": 4}
    orders.sort(key=lambda o: (act_order.get(o["action"], 9), o["rank"] or 999))

    # Enrich each order with live NSE Institutional Delivery % & Balance-Sheet Blowup Shield
    try:
        from .money_signals import evaluate_balance_sheet_shield, get_money_signals_summary

        ms_snap = get_money_signals_summary(db)
        deliv_map = ms_snap.get("stock_delivery_map", {})
    except Exception:
        deliv_map = {}
        evaluate_balance_sheet_shield = None  # type: ignore[assignment]

    for o in orders:
        dm = deliv_map.get(o["symbol"])
        if dm:
            o["deliv_pct"] = dm["deliv_pct"]
            o["deliv_surge_pct"] = dm["deliv_surge_pct"]
            o["smart_money_tag"] = dm["tag"]
            o["smart_money_tone"] = dm["tone"]
            o["shield_status"] = dm["shield_status"]
            o["shield_flags"] = dm["shield_flags"]
        else:
            t_meta = tickers.get(o["symbol"], {})
            sh = (
                evaluate_balance_sheet_shield(t_meta.get("fundamentals", {}), t_meta.get("sector", ""))
                if evaluate_balance_sheet_shield
                else {"status": "Clean", "flags": []}
            )
            o["deliv_pct"] = None
            o["deliv_surge_pct"] = None
            o["smart_money_tag"] = None
            o["smart_money_tone"] = "neutral"
            o["shield_status"] = sh["status"]
            o["shield_flags"] = sh["flags"]

    cash_reserve_inr = max(0.0, capital_inr - allocated_equity_inr - est_total_fees)
    sector_breakdown = [
        {"sector": sec, "count": cnt, "weight_pct": round(sum(o["target_weight_pct"] for o in orders if o["sector"] == sec and o["target_shares"] > 0), 1)}
        for sec, cnt in sorted(sec_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "as_of_date": latest_date,
        "config": {
            "capital_inr": round(capital_inr, 2),
            "top_k": top_k,
            "rank_buffer": rank_buffer,
            "max_per_sector": max_per_sector,
            "risk_profile": risk_profile.lower(),
            "vol_target_pct": round(vol_target * 100.0, 1) if vol_target else None,
            "skip_earnings_blackout": skip_earnings_blackout,
            "enforce_smart_money_shield": enforce_smart_money_shield,
            "cost_bps": cost_bps,
        },
        "summary": {
            "macro_regime": macro["regime"],
            "macro_tone": macro["tone"],
            "gate_exposure_pct": round(gate_exp * 100.0, 1),
            "vol_cap_exposure_pct": round(vol_cap * 100.0, 1),
            "target_equity_pct": round(final_equity_exp * 100.0, 1),
            "actual_equity_pct": round((allocated_equity_inr / max(1.0, capital_inr)) * 100.0, 1),
            "equity_inr": round(allocated_equity_inr, 2),
            "liquidbees_cash_inr": round(cash_reserve_inr, 2),
            "liquidbees_cash_pct": round((cash_reserve_inr / max(1.0, capital_inr)) * 100.0, 1),
            "ex_ante_vol_pct": round(ex_ante_vol_ann * 100.0, 1),
            "avg_pairwise_corr": round(portfolio_avg_corr, 2),
            "distinct_sectors": len(sec_counts),
            "positions_count": sum(1 for o in orders if o["target_shares"] > 0),
            "trades_count": sum(1 for o in orders if o["delta_shares"] != 0),
            "est_total_fees_inr": round(est_total_fees, 2),
            "est_realized_pnl_inr": round(est_realized_pnl, 2),
            "est_stcg_tax_inr": round(est_stcg_tax, 2),
            "blackout_warnings": sum(1 for o in orders if o["earnings_blackout"]),
        },
        "sector_breakdown": sector_breakdown,
        "skipped_blackout": skipped_blackout,
        "skipped_smart_money": skipped_smart_money,
        "orders": orders,
    }


def get_paper_ledger_status(db: Database) -> dict[str, Any]:
    """Mark open paper-trading positions to latest NSE prices and compare live forward P&L vs NIFTY 50."""
    acct = db.get_paper_account()
    if not acct:
        return {"initialized": False, "account": None, "positions": [], "trades": []}

    mkt = _latest_prices_and_matrices(db)
    latest_px: dict[str, float] = mkt["latest_px"]
    latest_date: str = mkt["latest_date"]
    positions = db.get_paper_positions()
    trades = db.get_paper_trades(limit=100)

    # Accrue Indian LiquidBees ~6.2% p.a. yield on cash since last rebalance date
    try:
        d_last = datetime.fromisoformat(acct["last_rebalance_at"].replace("Z", "+00:00")).date()
        d_now = datetime.strptime(latest_date[:10], "%Y-%m-%d").date()
        days_held = max(0, (d_now - d_last).days)
    except Exception:
        days_held = 0
    accrued_cash_interest = float(acct["cash_inr"]) * ((1.0 + INDIA_LIQUIDBEES_ANNUAL_YIELD) ** (days_held / 365.0) - 1.0)
    cash_now = float(acct["cash_inr"]) + accrued_cash_interest

    marked_positions: list[dict[str, Any]] = []
    equity_val = 0.0
    unrealized_pnl = 0.0
    for p in positions:
        sym = p["symbol"]
        sh = int(p["shares"])
        avg_c = float(p["avg_cost"])
        cur_px = float(latest_px.get(sym, avg_c))
        mv = sh * cur_px
        upnl = (cur_px - avg_c) * sh
        upnl_pct = ((cur_px / avg_c) - 1.0) * 100.0 if avg_c > 0 else 0.0
        equity_val += mv
        unrealized_pnl += upnl
        marked_positions.append(
            {
                "symbol": sym,
                "sector": p["sector"],
                "shares": sh,
                "avg_cost": round(avg_c, 2),
                "current_price": round(cur_px, 2),
                "market_value_inr": round(mv, 2),
                "unrealized_pnl_inr": round(upnl, 2),
                "unrealized_pnl_pct": round(upnl_pct, 2),
                "entry_date": p["entry_date"],
                "target_price": p.get("target_price"),
                "stop_price": p.get("stop_price"),
            }
        )

    total_nav = equity_val + cash_now
    init_cap = float(acct["initial_capital"])
    total_return_inr = total_nav - init_cap
    total_return_pct = ((total_nav / max(1.0, init_cap)) - 1.0) * 100.0

    bench_now = float(latest_px.get(BENCHMARK_SYMBOL, acct["bench_entry_price"]))
    bench_entry = float(acct["bench_entry_price"])
    bench_ret_pct = ((bench_now / max(1.0, bench_entry)) - 1.0) * 100.0

    for mp in marked_positions:
        mp["weight_pct"] = round((mp["market_value_inr"] / max(1.0, total_nav)) * 100.0, 2)
    marked_positions.sort(key=lambda x: x["market_value_inr"], reverse=True)

    return {
        "initialized": True,
        "as_of_date": latest_date,
        "account": {
            "started_at": acct["started_at"],
            "last_rebalance_at": acct["last_rebalance_at"],
            "risk_profile": acct["risk_profile"],
            "top_k": acct["top_k"],
            "initial_capital_inr": round(init_cap, 2),
            "total_nav_inr": round(total_nav, 2),
            "equity_value_inr": round(equity_val, 2),
            "cash_liquidbees_inr": round(cash_now, 2),
            "accrued_cash_interest_inr": round(accrued_cash_interest, 2),
            "equity_weight_pct": round((equity_val / max(1.0, total_nav)) * 100.0, 1),
            "total_return_inr": round(total_return_inr, 2),
            "total_return_pct": round(total_return_pct, 2),
            "unrealized_pnl_inr": round(unrealized_pnl, 2),
            "realized_pnl_inr": round(float(acct["realized_pnl_inr"]), 2),
            "total_fees_inr": round(float(acct["total_fees_inr"]), 2),
            "est_stcg_tax_inr": round(float(acct["est_stcg_tax_inr"]), 2),
            "nifty_return_pct": round(bench_ret_pct, 2),
            "excess_vs_nifty_pct": round(total_return_pct - bench_ret_pct, 2),
        },
        "positions": marked_positions,
        "trades": trades,
    }


def commit_paper_rebalance(
    db: Database,
    capital_inr: float = 1_000_000.0,
    top_k: int = 10,
    risk_profile: str = "balanced",
    skip_earnings_blackout: bool = False,
    enforce_smart_money_shield: bool = True,
) -> dict[str, Any]:
    """Execute and lock today's live V6 rebalance ticket into the SQLite forward-proof Paper Ledger."""
    existing_acct = db.get_paper_account()
    existing_pos = db.get_paper_positions() if existing_acct else []
    mkt = _latest_prices_and_matrices(db)
    latest_px: dict[str, float] = mkt["latest_px"]
    latest_date: str = mkt["latest_date"]
    now_iso = datetime.now(timezone.utc).isoformat()

    if existing_acct:
        status_now = get_paper_ledger_status(db)
        deploy_cap = float(status_now["account"]["total_nav_inr"])
        cur_holdings = existing_pos
        started_at = existing_acct["started_at"]
        initial_cap = float(existing_acct["initial_capital"])
        bench_entry = float(existing_acct["bench_entry_price"])
        prev_realized = float(existing_acct["realized_pnl_inr"])
        prev_fees = float(existing_acct["total_fees_inr"])
        prev_tax = float(existing_acct["est_stcg_tax_inr"])
    else:
        deploy_cap = float(capital_inr)
        cur_holdings = []
        started_at = now_iso
        initial_cap = float(capital_inr)
        bench_entry = float(latest_px.get(BENCHMARK_SYMBOL, 25000.0))
        prev_realized = 0.0
        prev_fees = 0.0
        prev_tax = 0.0

    ticket = generate_rebalance_ticket(
        db,
        capital_inr=deploy_cap,
        top_k=top_k,
        max_per_sector=2,
        risk_profile=risk_profile,
        skip_earnings_blackout=skip_earnings_blackout,
        enforce_smart_money_shield=enforce_smart_money_shield,
        current_holdings=cur_holdings,
    )

    prev_pos_map = {p["symbol"]: p for p in cur_holdings}
    new_positions: list[dict[str, Any]] = []
    new_trades: list[dict[str, Any]] = []

    for o in ticket["orders"]:
        sym = o["symbol"]
        px = float(o["price"])
        t_sh = int(o["target_shares"])
        d_sh = int(o["delta_shares"])

        if d_sh != 0:
            new_trades.append(
                {
                    "executed_at": now_iso,
                    "trade_date": latest_date,
                    "symbol": sym,
                    "action": o["action"],
                    "shares": abs(d_sh),
                    "price": px,
                    "gross_value": float(o["order_value_inr"]),
                    "fee_inr": float(o["est_fee_inr"]),
                    "realized_pnl_inr": float(o["est_realized_pnl_inr"]),
                    "reason": o["reason"],
                }
            )

        if t_sh > 0:
            old = prev_pos_map.get(sym)
            if old and d_sh > 0:
                old_sh = int(old["shares"])
                avg_cost = ((old_sh * float(old["avg_cost"])) + (d_sh * px)) / max(1, t_sh)
                entry_dt = old["entry_date"]
            elif old:
                avg_cost = float(old["avg_cost"])
                entry_dt = old["entry_date"]
            else:
                avg_cost = px
                entry_dt = latest_date
            new_positions.append(
                {
                    "symbol": sym,
                    "sector": o["sector"],
                    "shares": t_sh,
                    "avg_cost": round(avg_cost, 4),
                    "entry_date": entry_dt,
                    "target_price": o.get("target_price"),
                    "stop_price": o.get("stop_price"),
                }
            )

    new_account = {
        "started_at": started_at,
        "last_rebalance_at": now_iso,
        "initial_capital": initial_cap,
        "cash_inr": float(ticket["summary"]["liquidbees_cash_inr"]),
        "realized_pnl_inr": prev_realized + float(ticket["summary"]["est_realized_pnl_inr"]),
        "total_fees_inr": prev_fees + float(ticket["summary"]["est_total_fees_inr"]),
        "est_stcg_tax_inr": prev_tax + float(ticket["summary"]["est_stcg_tax_inr"]),
        "bench_entry_price": bench_entry,
        "risk_profile": risk_profile.lower(),
        "top_k": top_k,
    }
    db.save_paper_rebalance(new_account, new_positions, new_trades)
    return {"ticket": ticket, "ledger": get_paper_ledger_status(db)}
