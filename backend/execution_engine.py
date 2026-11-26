"""
EquiLens India V6 — Live NSE Rebalance Ticket Generator, Data Freshness Guard,
Peak-NAV Drawdown Brake, 70/30 LIQUIDCASE/GOLDBEES Reserve, FIFO Indian Tax
& Forward-Proof Paper-Trading Ledger.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import floor
from typing import Any

import numpy as np
import pandas as pd

from .config import BENCHMARK_SYMBOL, DEFAULT_COST_BPS, FX_MACRO_SYMBOL, GOLD_MACRO_SYMBOL, GOLDBEES_SYMBOL, NON_EQUITY_SYMBOLS
from .data_health import build_clean_price_matrix, evaluate_data_freshness
from .database import Database
from .indian_costs import FifoTaxLedger, compute_indian_delivery_cost
from .ml_predictor import compute_current_macro_regime

INDIA_LIQUIDCASE_ANNUAL_YIELD = 0.062
INDIA_STCG_TAX_RATE = 0.208
DRAWDOWN_BRAKE_RELEASE = -0.06


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
    prices_df, volumes_df, tradable_df = build_clean_price_matrix(ohlcv_map, max_ffill_days=2)
    log_rets = np.log(prices_df / prices_df.shift(1)) if not prices_df.empty else pd.DataFrame()
    adv_df = (prices_df * volumes_df).rolling(20, min_periods=5).mean() if not prices_df.empty else pd.DataFrame()
    latest_date = str(prices_df.index[-1]) if not prices_df.empty else datetime.now(timezone.utc).date().isoformat()

    latest_px: dict[str, float] = {}
    latest_adv: dict[str, float] = {}
    latest_tradable: dict[str, bool] = {}
    if not prices_df.empty:
        for sym in prices_df.columns:
            s_valid = prices_df[sym].dropna()
            if not s_valid.empty and float(s_valid.iloc[-1]) > 0:
                latest_px[sym] = float(s_valid.iloc[-1])
            if sym in adv_df.columns:
                a_valid = adv_df[sym].dropna()
                if not a_valid.empty and float(a_valid.iloc[-1]) > 0:
                    latest_adv[sym] = float(a_valid.iloc[-1])
            latest_tradable[sym] = bool(tradable_df[sym].iloc[-1]) if sym in tradable_df.columns else False

    return {
        "prices_df": prices_df,
        "log_rets": log_rets,
        "latest_date": latest_date,
        "latest_px": latest_px,
        "latest_adv": latest_adv,
        "latest_tradable": latest_tradable,
    }


def _gold_return_since(db: Database, from_date: str, to_date: str) -> float:
    """Compute INR gold return between two dates using GOLDBEES.NS or causal GC=F * INR=X proxy."""
    if from_date >= to_date:
        return 0.0
    gb_bars = db.get_ohlcv(GOLDBEES_SYMBOL)
    if len(gb_bars) >= 5:
        df_g = pd.DataFrame(gb_bars)[["date", "close"]].sort_values("date")
        p0_s = df_g[df_g["date"] <= from_date]["close"]
        p1_s = df_g[df_g["date"] <= to_date]["close"]
        if not p0_s.empty and not p1_s.empty and float(p0_s.iloc[-1]) > 0:
            return float((float(p1_s.iloc[-1]) / float(p0_s.iloc[-1])) - 1.0)

    macro = db.get_macro_series()
    g_map = macro.get(GOLD_MACRO_SYMBOL, {})
    fx_map = macro.get(FX_MACRO_SYMBOL, {})
    if g_map:
        g_dates0 = [d for d in g_map if d <= from_date]
        g_dates1 = [d for d in g_map if d <= to_date]
        if g_dates0 and g_dates1:
            g0 = float(g_map[max(g_dates0)])
            g1 = float(g_map[max(g_dates1)])
            fx0 = float(fx_map[max(d for d in fx_map if d <= from_date)]) if any(d <= from_date for d in fx_map) else 85.0
            fx1 = float(fx_map[max(d for d in fx_map if d <= to_date)]) if any(d <= to_date for d in fx_map) else fx0
            if g0 > 0 and fx0 > 0:
                return float(((g1 * fx1) / (g0 * fx0)) - 1.0)
    return 0.0


def _ensure_fifo_ledger(db: Database, acct: dict[str, Any] | None, positions: list[dict[str, Any]]) -> FifoTaxLedger:
    lot_rows = db.get_paper_lots() if hasattr(db, "get_paper_lots") else []
    stcl = float(acct.get("stcl_carry_inr", 0.0) or 0.0) if acct else 0.0
    ltcl = float(acct.get("ltcl_carry_inr", 0.0) or 0.0) if acct else 0.0
    if lot_rows:
        return FifoTaxLedger.from_rows(lot_rows, stcl_carry_inr=stcl, ltcl_carry_inr=ltcl)
    ledger = FifoTaxLedger(stcl_carry_inr=stcl, ltcl_carry_inr=ltcl)
    for p in positions:
        ledger.buy(
            symbol=p["symbol"],
            buy_date=str(p["entry_date"])[:10],
            shares=float(p["shares"]),
            price=float(p["avg_cost"]),
            buy_cost_inr=0.0,
        )
    return ledger


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
    use_statutory_costs: bool = True,
) -> dict[str, Any]:
    """Generate an executable V6 NSE rebalance ticket with exact integer shares & FIFO Indian tax/costs."""
    preds = db.get_ml_predictions()
    if not preds:
        raise ValueError("No ML predictions available. Train the model first.")

    freshness = evaluate_data_freshness(db)
    tickers = {t["symbol"]: t for t in db.get_tickers()}
    earnings_map = db.get_all_earnings()
    mkt = _latest_prices_and_matrices(db)
    log_rets: pd.DataFrame = mkt["log_rets"]
    latest_date: str = mkt["latest_date"]
    latest_px: dict[str, float] = mkt["latest_px"]
    latest_adv: dict[str, float] = mkt["latest_adv"]
    latest_tradable: dict[str, bool] = mkt["latest_tradable"]

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

    aux_excluded = set(NON_EQUITY_SYMBOLS)
    ranked_rows: list[dict[str, Any]] = []
    for sym, p in preds.items():
        if sym in aux_excluded or sym not in latest_px:
            continue
        dte, next_e_dt = _days_to_next_earnings(earnings_map.get(sym, []), latest_date)
        t_meta = tickers.get(sym, {})
        dm = deliv_map.get(sym)
        if dm:
            deliv_pct = dm.get("deliv_pct")
            shield_status = dm.get("shield_status", "Clean")
            shield_flags = dm.get("shield_flags", [])
        else:
            sh = (
                evaluate_balance_sheet_shield(t_meta.get("fundamentals", {}), t_meta.get("sector", ""))
                if evaluate_balance_sheet_shield
                else {"status": "Clean", "flags": []}
            )
            deliv_pct = None
            shield_status = sh["status"]
            shield_flags = sh["flags"]

        ranked_rows.append(
            {
                "symbol": sym,
                "name": t_meta.get("name", sym),
                "sector": t_meta.get("sector", "Other"),
                "price": latest_px[sym],
                "tradable": latest_tradable.get(sym, True),
                "pred_alpha": float(p.get("pred_alpha_20d", 0.0)),
                "prob_tb_win": float(p.get("prob_tb_win", 50.0)),
                "vol_ann": max(0.12, float(p.get("vol_20d_ann", 24.0)) / 100.0),
                "target_price": p.get("target_price"),
                "stop_price": p.get("stop_price"),
                "days_to_earnings": dte,
                "next_earnings_date": next_e_dt,
                "earnings_blackout": dte is not None and 0 <= dte <= 5,
                "deliv_pct": deliv_pct,
                "shield_status": shield_status,
                "shield_flags": shield_flags,
                "smart_money_blocked": (deliv_pct is not None and float(deliv_pct) < 35.0) or shield_status == "Risk",
            }
        )

    ranked_rows.sort(key=lambda r: (r["pred_alpha"], r["prob_tb_win"]), reverse=True)
    for idx, r in enumerate(ranked_rows, start=1):
        r["rank"] = idx
    row_by_sym = {r["symbol"]: r for r in ranked_rows}

    held_map: dict[str, dict[str, Any]] = {}
    if current_holdings:
        for h in current_holdings:
            sym = str(h["symbol"]).upper()
            sh = int(h.get("shares", 0))
            if sh > 0 and sym in latest_px:
                held_map[sym] = {
                    "symbol": sym,
                    "shares": sh,
                    "avg_cost": float(h.get("avg_cost") or latest_px[sym]),
                    "entry_date": str(h.get("entry_date") or latest_date)[:10],
                }

    rank_buffer = min(top_k * 3, 40)
    max_stock_w = 0.05 if top_k >= 20 else round(min(0.15, 1.20 / max(1, top_k)), 4)
    max_sector_w = 0.25 if top_k < 10 else 0.20
    effective_max_per_sec = max(max_per_sector, int(round(max_sector_w / max(0.01, max_stock_w)))) if top_k >= 12 else max_per_sector

    chosen: list[str] = []
    sec_counts: dict[str, int] = {}
    skipped_blackout: list[dict[str, Any]] = []
    skipped_smart_money: list[dict[str, Any]] = []
    skipped_stale: list[str] = []

    def _can_add(sym: str) -> bool:
        sec = row_by_sym[sym]["sector"]
        return sec_counts.get(sec, 0) < effective_max_per_sec

    def _add(sym: str) -> None:
        chosen.append(sym)
        sec = row_by_sym[sym]["sector"]
        sec_counts[sec] = sec_counts.get(sec, 0) + 1

    for sym in sorted(held_map.keys(), key=lambda s: row_by_sym.get(s, {}).get("rank", 999)):
        if sym in row_by_sym and row_by_sym[sym]["rank"] <= rank_buffer and _can_add(sym):
            _add(sym)

    for r in ranked_rows:
        if len(chosen) >= top_k:
            break
        sym = r["symbol"]
        if sym in chosen or not _can_add(sym):
            continue
        if not r["tradable"]:
            skipped_stale.append(sym)
            continue
        if skip_earnings_blackout and r["earnings_blackout"]:
            skipped_blackout.append(
                {
                    "symbol": sym,
                    "rank": r["rank"],
                    "days_to_earnings": r["days_to_earnings"],
                    "next_earnings_date": r["next_earnings_date"],
                }
            )
            continue
        if enforce_smart_money_shield and r["smart_money_blocked"]:
            reasons = []
            if r["deliv_pct"] is not None and float(r["deliv_pct"]) < 35.0:
                reasons.append(f"Low NSE Delivery ({r['deliv_pct']:.1f}% < 35%)")
            if r["shield_status"] == "Risk":
                reasons.append("Balance-Sheet Risk (" + ", ".join(r["shield_flags"]) + ")")
            skipped_smart_money.append(
                {
                    "symbol": sym,
                    "rank": r["rank"],
                    "deliv_pct": r["deliv_pct"],
                    "shield_status": r["shield_status"],
                    "reason": " + ".join(reasons),
                }
            )
            continue
        _add(sym)

    if len(chosen) < top_k:
        for r in ranked_rows:
            if len(chosen) >= top_k:
                break
            sym = r["symbol"]
            if sym in chosen or not r["tradable"]:
                continue
            if skip_earnings_blackout and r["earnings_blackout"]:
                continue
            if enforce_smart_money_shield and r["smart_money_blocked"]:
                continue
            _add(sym)

    if not chosen:
        raise ValueError("Unable to construct a valid NSE portfolio from current predictions.")

    vols = np.array([row_by_sym[s]["vol_ann"] for s in chosen], dtype=float)
    avg_corrs = np.full(len(chosen), 0.20, dtype=float)
    portfolio_avg_corr = 0.20
    if len(chosen) >= 3 and len(log_rets) >= 65 and all(s in log_rets.columns for s in chosen):
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

    ex_ante_vol_ann = float(np.mean(vols) * 0.72)
    vol_cap = 1.0
    if len(chosen) >= 2 and len(log_rets) >= 65 and all(s in log_rets.columns for s in chosen):
        sub60 = log_rets[chosen].iloc[-60:].fillna(0.0).to_numpy(dtype=float)
        cov60 = np.cov(sub60, rowvar=False) * 252.0
        ex_ante_vol_ann = float(np.sqrt(max(1e-6, float(unit_weights @ cov60 @ unit_weights))))
    if vol_target is not None:
        vol_cap = float(np.clip(vol_target / max(0.08, ex_ante_vol_ann), 0.25, 1.0))

    from .portfolio_backtester import DRAWDOWN_BRAKE_TRIGGER, GOLDBEES_RESERVE_SHARE, apply_concentration_caps

    # Fixed Safety #1: True Peak-to-Trough -10% Drawdown Brake with hysteresis release
    final_equity_exp = min(gate_exp, vol_cap)
    dd_brake_active = False
    peak_nav_ref = capital_inr
    cur_dd_pct = 0.0
    acct_chk = db.get_paper_account()
    if current_holdings and acct_chk:
        prev_peak = float(acct_chk.get("peak_nav_inr") or acct_chk.get("initial_capital") or capital_inr)
        peak_nav_ref = max(prev_peak, capital_inr)
        cur_dd = (capital_inr / max(1.0, peak_nav_ref)) - 1.0
        cur_dd_pct = cur_dd * 100.0
        prev_brake = bool(acct_chk.get("dd_brake_active", 0))
        if not prev_brake and cur_dd <= DRAWDOWN_BRAKE_TRIGGER:
            dd_brake_active = True
        elif prev_brake:
            nav_hist = db.get_paper_nav_history(limit=25) if hasattr(db, "get_paper_nav_history") else []
            sma_ok = True
            if len(nav_hist) >= 20:
                sma20_nav = float(np.mean([h["nav_inr"] for h in nav_hist[-20:]]))
                sma_ok = capital_inr > sma20_nav * 1.005
            dd_brake_active = not (cur_dd > DRAWDOWN_BRAKE_RELEASE and sma_ok)
        if dd_brake_active:
            final_equity_exp *= 0.50

    sector_lookup = {s: row_by_sym[s]["sector"] for s in chosen}
    raw_port_weights = {s: float(unit_weights[i] * final_equity_exp) for i, s in enumerate(chosen)}
    capped_port_weights = apply_concentration_caps(raw_port_weights, sector_lookup, max_stock_w, max_sector_w, final_equity_exp)

    fifo_sim = _ensure_fifo_ledger(db, acct_chk, list(held_map.values())).clone()

    def _calc_order_cost(side: str, sym: str, gross: float) -> dict[str, float]:
        if not use_statutory_costs:
            c = round(gross * (cost_bps / 10000.0), 2)
            return {"total_cost_inr": c, "statutory_inr": c, "slippage_inr": 0.0}
        dvol = row_by_sym.get(sym, {}).get("vol_ann", 0.24) / np.sqrt(252.0)
        return compute_indian_delivery_cost(
            side=side,
            gross_value_inr=gross,
            adv_20d_inr=latest_adv.get(sym),
            daily_vol=dvol,
            half_spread_bps=3.0,
            include_dp_charge=True,
        )

    orders: list[dict[str, Any]] = []
    est_total_fees = 0.0
    est_stcg_tax = 0.0
    est_ltcg_tax = 0.0
    est_realized_pnl = 0.0
    allocated_equity_inr = 0.0

    for sym, h in held_map.items():
        if sym in chosen:
            continue
        px = latest_px[sym]
        sh = h["shares"]
        gross = sh * px
        cd = _calc_order_cost("SELL", sym, gross)
        fee = cd["total_cost_inr"]
        tx = fifo_sim.sell(sym, latest_date, float(sh), px, sell_cost_inr=fee)
        rpnl = tx["realized_pnl_inr"]
        stcg_t = tx["stcg_tax_inr"]
        ltcg_t = tx["ltcg_tax_inr"]
        est_total_fees += fee
        est_realized_pnl += rpnl
        est_stcg_tax += stcg_t
        est_ltcg_tax += ltcg_t
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
                "est_stcg_tax_inr": round(stcg_t, 2),
                "est_ltcg_tax_inr": round(ltcg_t, 2),
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

    for idx, sym in enumerate(chosen):
        r = row_by_sym[sym]
        px = r["price"]
        w_port = float(capped_port_weights.get(sym, unit_weights[idx] * final_equity_exp))
        target_val = capital_inr * w_port
        target_sh = max(0, int(floor(target_val / max(0.01, px))))
        actual_val = target_sh * px
        allocated_equity_inr += actual_val

        cur_sh = held_map[sym]["shares"] if sym in held_map else 0
        delta_sh = target_sh - cur_sh
        order_val = abs(delta_sh) * px

        rpnl = 0.0
        stcg_t = 0.0
        ltcg_t = 0.0
        fee = 0.0

        if cur_sh == 0 and target_sh > 0:
            action = "BUY"
            fee = _calc_order_cost("BUY", sym, order_val)["total_cost_inr"]
            reason = f"New Top-{top_k} entry (Rank #{r['rank']}, 40D alpha {r['pred_alpha']:+.2f}%)"
        elif delta_sh == 0 or (cur_sh > 0 and abs(delta_sh) / max(1, cur_sh) < 0.08):
            target_sh = cur_sh
            delta_sh = 0
            order_val = 0.0
            allocated_equity_inr = allocated_equity_inr - actual_val + (cur_sh * px)
            actual_val = cur_sh * px
            action = "HOLD"
            reason = f"Retained inside Top-{rank_buffer} hysteresis buffer (Rank #{r['rank']})"
        elif delta_sh > 0:
            action = "ADD"
            fee = _calc_order_cost("BUY", sym, order_val)["total_cost_inr"]
            reason = f"Rebalance top-up (+{delta_sh} sh) inside Top-{rank_buffer} buffer"
        else:
            action = "TRIM"
            fee = _calc_order_cost("SELL", sym, order_val)["total_cost_inr"]
            tx = fifo_sim.sell(sym, latest_date, float(abs(delta_sh)), px, sell_cost_inr=fee)
            rpnl = tx["realized_pnl_inr"]
            stcg_t = tx["stcg_tax_inr"]
            ltcg_t = tx["ltcg_tax_inr"]
            est_realized_pnl += rpnl
            est_stcg_tax += stcg_t
            est_ltcg_tax += ltcg_t
            reason = f"Risk/vol rebalance trim ({delta_sh} sh)"

        est_total_fees += fee

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
                "est_stcg_tax_inr": round(stcg_t, 2),
                "est_ltcg_tax_inr": round(ltcg_t, 2),
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

    act_order = {"SELL": 0, "BUY": 1, "ADD": 2, "TRIM": 3, "HOLD": 4}
    orders.sort(key=lambda o: (act_order.get(o["action"], 9), o["rank"] or 999))

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
    goldbees_alloc_inr = cash_reserve_inr * GOLDBEES_RESERVE_SHARE
    liquidcase_alloc_inr = cash_reserve_inr - goldbees_alloc_inr
    sector_breakdown = [
        {"sector": sec, "count": cnt, "weight_pct": round(sum(o["target_weight_pct"] for o in orders if o["sector"] == sec and o["target_shares"] > 0), 1)}
        for sec, cnt in sorted(sec_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "as_of_date": latest_date,
        "data_freshness": freshness,
        "config": {
            "capital_inr": round(capital_inr, 2),
            "top_k": top_k,
            "rank_buffer": rank_buffer,
            "max_per_sector": effective_max_per_sec,
            "max_stock_weight_pct": round(max_stock_w * 100.0, 1),
            "max_sector_weight_pct": round(max_sector_w * 100.0, 1),
            "dd_brake_trigger_pct": round(DRAWDOWN_BRAKE_TRIGGER * 100.0, 1),
            "dd_brake_release_pct": round(DRAWDOWN_BRAKE_RELEASE * 100.0, 1),
            "goldbees_reserve_pct": round(GOLDBEES_RESERVE_SHARE * 100.0, 1),
            "risk_profile": risk_profile.lower(),
            "vol_target_pct": round(vol_target * 100.0, 1) if vol_target else None,
            "skip_earnings_blackout": skip_earnings_blackout,
            "enforce_smart_money_shield": enforce_smart_money_shield,
            "cost_bps": cost_bps,
        },
        "summary": {
            "macro_regime": macro["regime"],
            "macro_tone": macro["tone"],
            "dd_brake_active": dd_brake_active,
            "peak_nav_ref_inr": round(peak_nav_ref, 2),
            "current_dd_pct": round(cur_dd_pct, 2),
            "gate_exposure_pct": round(gate_exp * 100.0, 1),
            "vol_cap_exposure_pct": round(vol_cap * 100.0, 1),
            "target_equity_pct": round(final_equity_exp * 100.0, 1),
            "actual_equity_pct": round((allocated_equity_inr / max(1.0, capital_inr)) * 100.0, 1),
            "equity_inr": round(allocated_equity_inr, 2),
            "liquidcase_cash_inr": round(cash_reserve_inr, 2),
            "liquidcase_cash_pct": round((cash_reserve_inr / max(1.0, capital_inr)) * 100.0, 1),
            "liquidcase_alloc_inr": round(liquidcase_alloc_inr, 2),
            "goldbees_alloc_inr": round(goldbees_alloc_inr, 2),
            "ex_ante_vol_pct": round(ex_ante_vol_ann * 100.0, 1),
            "avg_pairwise_corr": round(portfolio_avg_corr, 2),
            "distinct_sectors": len(sec_counts),
            "positions_count": sum(1 for o in orders if o["target_shares"] > 0),
            "trades_count": sum(1 for o in orders if o["delta_shares"] != 0),
            "est_total_fees_inr": round(est_total_fees, 2),
            "est_realized_pnl_inr": round(est_realized_pnl, 2),
            "est_stcg_tax_inr": round(est_stcg_tax, 2),
            "est_ltcg_tax_inr": round(est_ltcg_tax, 2),
            "blackout_warnings": sum(1 for o in orders if o["earnings_blackout"]),
        },
        "sector_breakdown": sector_breakdown,
        "skipped_blackout": skipped_blackout,
        "skipped_smart_money": skipped_smart_money,
        "skipped_stale": skipped_stale,
        "orders": orders,
    }


def get_paper_ledger_status(db: Database) -> dict[str, Any]:
    """Mark open paper positions & 70/30 LIQUIDCASE/GOLDBEES reserve forward vs NIFTY 50."""
    acct = db.get_paper_account()
    if not acct:
        return {"initialized": False, "account": None, "positions": [], "trades": []}

    mkt = _latest_prices_and_matrices(db)
    latest_px: dict[str, float] = mkt["latest_px"]
    latest_adv: dict[str, float] = mkt["latest_adv"]
    latest_date: str = mkt["latest_date"]
    positions = db.get_paper_positions()
    trades = db.get_paper_trades(limit=100)

    try:
        d_last = datetime.fromisoformat(acct["last_rebalance_at"].replace("Z", "+00:00")).date()
        d_now = datetime.strptime(latest_date[:10], "%Y-%m-%d").date()
        days_held = max(0, (d_now - d_last).days)
        last_dt_iso = d_last.isoformat()
    except Exception:
        days_held = 0
        last_dt_iso = latest_date[:10]

    base_liq = float(acct["liquidcase_inr"]) if acct.get("liquidcase_inr") is not None else float(acct["cash_inr"])
    base_gold = float(acct.get("goldbees_inr") or 0.0)
    accrued_cash_interest = base_liq * ((1.0 + INDIA_LIQUIDCASE_ANNUAL_YIELD) ** (days_held / 365.0) - 1.0)
    gold_ret = _gold_return_since(db, last_dt_iso, latest_date[:10]) if base_gold > 0 else 0.0
    gold_pnl = base_gold * gold_ret
    liquidcase_now = base_liq + accrued_cash_interest
    goldbees_now = base_gold + gold_pnl
    cash_now = liquidcase_now + goldbees_now

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
    peak_nav = max(float(acct.get("peak_nav_inr") or init_cap), total_nav)
    cur_dd_pct = ((total_nav / max(1.0, peak_nav)) - 1.0) * 100.0
    total_return_inr = total_nav - init_cap
    total_return_pct = ((total_nav / max(1.0, init_cap)) - 1.0) * 100.0

    bench_now = float(latest_px.get(BENCHMARK_SYMBOL, acct["bench_entry_price"]))
    bench_entry = float(acct["bench_entry_price"])
    bench_ret_pct = ((bench_now / max(1.0, bench_entry)) - 1.0) * 100.0

    fifo_ledger = _ensure_fifo_ledger(db, acct, positions)
    liq_est = fifo_ledger.estimate_liquidation(latest_date[:10], latest_px, adv_map=latest_adv)

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
            "peak_nav_inr": round(peak_nav, 2),
            "drawdown_pct": round(cur_dd_pct, 2),
            "dd_brake_active": bool(acct.get("dd_brake_active", 0)),
            "equity_value_inr": round(equity_val, 2),
            "cash_liquidcase_inr": round(cash_now, 2),
            "liquidcase_only_inr": round(liquidcase_now, 2),
            "goldbees_reserve_inr": round(goldbees_now, 2),
            "accrued_cash_interest_inr": round(accrued_cash_interest, 2),
            "goldbees_pnl_inr": round(gold_pnl, 2),
            "equity_weight_pct": round((equity_val / max(1.0, total_nav)) * 100.0, 1),
            "total_return_inr": round(total_return_inr, 2),
            "total_return_pct": round(total_return_pct, 2),
            "unrealized_pnl_inr": round(unrealized_pnl, 2),
            "realized_pnl_inr": round(float(acct["realized_pnl_inr"]), 2),
            "total_fees_inr": round(float(acct["total_fees_inr"]), 2),
            "est_stcg_tax_inr": round(float(acct["est_stcg_tax_inr"]), 2),
            "est_ltcg_tax_inr": round(float(acct.get("est_ltcg_tax_inr") or 0.0), 2),
            "liquidation_net_nav_inr": round(cash_now + liq_est["net_after_liquidation_inr"], 2),
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
    allow_stale: bool = False,
) -> dict[str, Any]:
    """Execute and lock today's live V6 rebalance ticket with freshness guard & FIFO tax lots."""
    freshness = evaluate_data_freshness(db)
    if freshness["block_execution"] and not allow_stale:
        raise ValueError(
            "Blocked paper rebalance due to stale/incomplete market data: "
            + "; ".join(freshness["warnings"])
            + " (Run Sync & Retrain or pass allow_stale=true)."
        )

    existing_acct = db.get_paper_account()
    existing_pos = db.get_paper_positions() if existing_acct else []
    mkt = _latest_prices_and_matrices(db)
    latest_px: dict[str, float] = mkt["latest_px"]
    latest_date: str = mkt["latest_date"]
    now_iso = datetime.now(timezone.utc).isoformat()

    if existing_acct:
        status_now = get_paper_ledger_status(db)
        deploy_cap = float(status_now["account"]["total_nav_inr"])
        peak_nav = float(status_now["account"]["peak_nav_inr"])
        cur_holdings = existing_pos
        started_at = existing_acct["started_at"]
        initial_cap = float(existing_acct["initial_capital"])
        bench_entry = float(existing_acct["bench_entry_price"])
        prev_realized = float(existing_acct["realized_pnl_inr"])
        prev_fees = float(existing_acct["total_fees_inr"])
        prev_stcg = float(existing_acct["est_stcg_tax_inr"])
        prev_ltcg = float(existing_acct.get("est_ltcg_tax_inr") or 0.0)
    else:
        deploy_cap = float(capital_inr)
        peak_nav = float(capital_inr)
        cur_holdings = []
        started_at = now_iso
        initial_cap = float(capital_inr)
        bench_entry = float(latest_px.get(BENCHMARK_SYMBOL, 25000.0))
        prev_realized = 0.0
        prev_fees = 0.0
        prev_stcg = 0.0
        prev_ltcg = 0.0

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

    fifo_ledger = _ensure_fifo_ledger(db, existing_acct, cur_holdings)
    prev_pos_map = {p["symbol"]: p for p in cur_holdings}
    new_positions: list[dict[str, Any]] = []
    new_trades: list[dict[str, Any]] = []

    for o in ticket["orders"]:
        sym = o["symbol"]
        px = float(o["price"])
        t_sh = int(o["target_shares"])
        d_sh = int(o["delta_shares"])
        fee = float(o["est_fee_inr"])

        if d_sh < 0:
            fifo_ledger.sell(sym, latest_date[:10], float(abs(d_sh)), px, sell_cost_inr=fee)
        elif d_sh > 0:
            fifo_ledger.buy(sym, latest_date[:10], float(d_sh), px, buy_cost_inr=fee)

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
                    "fee_inr": fee,
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

    new_nav = float(ticket["summary"]["equity_inr"]) + float(ticket["summary"]["liquidcase_cash_inr"])
    new_peak = max(peak_nav, new_nav)
    new_account = {
        "started_at": started_at,
        "last_rebalance_at": now_iso,
        "initial_capital": initial_cap,
        "cash_inr": float(ticket["summary"]["liquidcase_cash_inr"]),
        "liquidcase_inr": float(ticket["summary"]["liquidcase_alloc_inr"]),
        "goldbees_inr": float(ticket["summary"]["goldbees_alloc_inr"]),
        "peak_nav_inr": new_peak,
        "dd_brake_active": bool(ticket["summary"]["dd_brake_active"]),
        "realized_pnl_inr": prev_realized + float(ticket["summary"]["est_realized_pnl_inr"]),
        "total_fees_inr": prev_fees + float(ticket["summary"]["est_total_fees_inr"]),
        "est_stcg_tax_inr": prev_stcg + float(ticket["summary"]["est_stcg_tax_inr"]),
        "est_ltcg_tax_inr": prev_ltcg + float(ticket["summary"]["est_ltcg_tax_inr"]),
        "stcl_carry_inr": fifo_ledger.stcl_carry_inr,
        "ltcl_carry_inr": fifo_ledger.ltcl_carry_inr,
        "bench_entry_price": bench_entry,
        "risk_profile": risk_profile.lower(),
        "top_k": top_k,
    }
    lots_out = [dict(l, created_at=now_iso) for l in fifo_ledger.export_lots()]
    db.save_paper_rebalance(new_account, new_positions, new_trades, lots=lots_out)
    if hasattr(db, "record_paper_nav"):
        db.record_paper_nav(
            {
                "date": latest_date[:10],
                "nav_inr": new_nav,
                "equity_inr": float(ticket["summary"]["equity_inr"]),
                "liquidcase_inr": float(ticket["summary"]["liquidcase_alloc_inr"]),
                "goldbees_inr": float(ticket["summary"]["goldbees_alloc_inr"]),
                "peak_nav_inr": new_peak,
                "dd_pct": ((new_nav / max(1.0, new_peak)) - 1.0) * 100.0,
                "dd_brake_active": bool(ticket["summary"]["dd_brake_active"]),
            }
        )
    return {"ticket": ticket, "ledger": get_paper_ledger_status(db)}
