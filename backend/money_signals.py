"""
EquiLens India — Official NSE Institutional "Money Signals" Engine:
Pulls genuine, public institutional money-flow footprints directly from the
National Stock Exchange of India (nsearchives.nseindia.com) & official NSE indices:

1. NSE Daily Delivery Volume % & Institutional Ticket Size (sec_bhavdata_full_DDMMYYYY.csv)
   - Separates true institutional demat accumulation from retail intraday churn
   - Tracks average rupee value per trade (Turnover / No. of Trades)
2. NSE Daily FII / DII / Client Derivatives Positioning (fao_participant_oi_DDMMYYYY.csv)
   - Tracks FII Index Futures Long % & Net Positioning (India's #1 institutional sentiment gauge)
3. NSE Daily Bulk & Block Deals (bulk.csv, block.csv)
   - Surfaces large fund / institutional exchange prints
4. India Credit & Midcap Liquidity Ratios (^NSEBANK / ^NSEI and ^NSEMDCP50 / ^NSEI)
5. Balance-Sheet Blowup Shield (flags negative OCF, extreme debt/equity, or weak ROE)
"""
from __future__ import annotations

import csv
import io
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .config import BENCHMARK_SYMBOL
from .database import Database

logger = logging.getLogger("equilens.money_signals")

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
}

INDIA_LIQUIDITY_SYMBOLS = ["^NSEBANK", "^NSEMDCP50"]


def _fetch_text(url: str, timeout: float = 8.0) -> str | None:
    try:
        req = urllib.request.Request(url, headers=NSE_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        s = str(val).strip().replace(",", "")
        if not s or s == "-":
            return default
        return float(s)
    except Exception:
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    return int(round(_safe_float(val, float(default))))


def _recent_nse_dates(db: Database, n_days: int = 15) -> list[str]:
    """Use actual NIFTY 50 trading dates from OHLCV so Indian market holidays are automatically skipped."""
    bars = db.get_ohlcv(BENCHMARK_SYMBOL)
    if bars:
        return [str(b["date"])[:10] for b in bars[-n_days:]]
    out = []
    d = date.today()
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return list(reversed(out))


def _fetch_delivery_for_date(iso_date: str, tracked_raw: set[str]) -> list[dict[str, Any]]:
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    ddmmyyyy = dt.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
    txt = _fetch_text(url)
    if not txt:
        return []
    rows_out: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(txt))
    for raw in reader:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
        if row.get("SERIES") != "EQ":
            continue
        sym_raw = row.get("SYMBOL", "").upper()
        if tracked_raw and sym_raw not in tracked_raw:
            continue
        turnover_lacs = _safe_float(row.get("TURNOVER_LACS"))
        no_trades = _safe_int(row.get("NO_OF_TRADES"))
        avg_trade_inr = (turnover_lacs * 100_000.0) / max(1, no_trades)
        rows_out.append(
            {
                "symbol": f"{sym_raw}.NS",
                "date": iso_date,
                "close_price": _safe_float(row.get("CLOSE_PRICE")),
                "traded_qty": _safe_int(row.get("TTL_TRD_QNTY")),
                "deliv_qty": _safe_int(row.get("DELIV_QTY")),
                "deliv_pct": _safe_float(row.get("DELIV_PER")),
                "turnover_lacs": turnover_lacs,
                "no_of_trades": no_trades,
                "avg_trade_val_inr": round(avg_trade_inr, 2),
            }
        )
    return rows_out


def _fetch_fii_oi_for_date(iso_date: str) -> dict[str, Any] | None:
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    ddmmyyyy = dt.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{ddmmyyyy}.csv"
    txt = _fetch_text(url)
    if not txt:
        return None
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    # Skip title banner line ("Participant wise Open Interest...")
    header_idx = 0
    for i, ln in enumerate(lines[:4]):
        if "Client Type" in ln:
            header_idx = i
            break
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    by_type: dict[str, dict[str, Any]] = {}
    for raw in reader:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
        ct = row.get("Client Type", "").upper()
        if ct in ("FII", "DII", "CLIENT", "PRO"):
            by_type[ct] = row
    fii = by_type.get("FII")
    if not fii:
        return None
    fii_l = _safe_int(fii.get("Future Index Long"))
    fii_s = _safe_int(fii.get("Future Index Short"))
    fii_long_pct = (fii_l / max(1, fii_l + fii_s)) * 100.0
    dii = by_type.get("DII", {})
    cli = by_type.get("CLIENT", {})
    return {
        "date": iso_date,
        "fii_idx_fut_long": fii_l,
        "fii_idx_fut_short": fii_s,
        "fii_idx_long_pct": round(fii_long_pct, 1),
        "fii_net_idx_fut": fii_l - fii_s,
        "dii_net_idx_fut": _safe_int(dii.get("Future Index Long")) - _safe_int(dii.get("Future Index Short")),
        "client_net_idx_fut": _safe_int(cli.get("Future Index Long")) - _safe_int(cli.get("Future Index Short")),
        "fii_stk_fut_net": _safe_int(fii.get("Future Stock Long")) - _safe_int(fii.get("Future Stock Short")),
    }


def _fetch_bulk_block_deals() -> list[dict[str, Any]]:
    deals: list[dict[str, Any]] = []
    for deal_type, url in [
        ("BULK", "https://nsearchives.nseindia.com/content/equities/bulk.csv"),
        ("BLOCK", "https://nsearchives.nseindia.com/content/equities/block.csv"),
    ]:
        txt = _fetch_text(url)
        if not txt:
            continue
        reader = csv.DictReader(io.StringIO(txt))
        for idx, raw in enumerate(reader):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
            sym_raw = row.get("Symbol", "").upper()
            if not sym_raw:
                continue
            qty = _safe_int(row.get("Quantity Traded"))
            px = _safe_float(row.get("Trade Price / Wght. Avg. Price"))
            val_cr = (qty * px) / 1e7
            dt_raw = row.get("Date", "")
            try:
                iso_dt = datetime.strptime(dt_raw, "%d-%b-%Y").strftime("%Y-%m-%d")
            except Exception:
                iso_dt = dt_raw
            client = row.get("Client Name", "Unknown")[:80]
            action = row.get("Buy/Sell", "BUY").upper()
            deals.append(
                {
                    "id": f"{deal_type}_{iso_dt}_{sym_raw}_{idx}",
                    "date": iso_dt,
                    "deal_type": deal_type,
                    "symbol": f"{sym_raw}.NS",
                    "client_name": client,
                    "action": action,
                    "quantity": qty,
                    "price": round(px, 2),
                    "value_cr": round(val_cr, 2),
                }
            )
    return deals


def sync_indian_liquidity_indices(db: Database) -> dict[str, int]:
    """Sync 10Y official NSE Credit (^NSEBANK) & Midcap Liquidity (^NSEMDCP50) indices into macro_series."""
    synced: dict[str, int] = {}
    for sym in INDIA_LIQUIDITY_SYMBOLS:
        try:
            df = yf.download(sym, period="10y", progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            closes = df["Close"].squeeze()
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            rows = [(idx.strftime("%Y-%m-%d"), float(val)) for idx, val in closes.dropna().items()]
            if rows:
                db.upsert_macro_series(sym, rows)
                synced[sym] = len(rows)
        except Exception as exc:
            logger.warning("Failed to sync %s: %s", sym, exc)
    return synced


def sync_nse_money_signals(db: Database, lookback_days: int = 15) -> dict[str, Any]:
    """Fetch official NSE Delivery %, FII/DII F&O Participant OI, Bulk/Block Deals & India Liquidity Indices."""
    tickers = [t["symbol"] for t in db.get_tickers() if t["symbol"].endswith(".NS")]
    tracked_raw = {s.replace(".NS", "").upper() for s in tickers}
    dates = _recent_nse_dates(db, n_days=lookback_days)

    deliv_rows: list[dict[str, Any]] = []
    fii_rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=6) as pool:
        d_futs = [pool.submit(_fetch_delivery_for_date, dt, tracked_raw) for dt in dates]
        f_futs = [pool.submit(_fetch_fii_oi_for_date, dt) for dt in dates]
        for f in d_futs:
            deliv_rows.extend(f.result())
        for f in f_futs:
            res = f.result()
            if res:
                fii_rows.append(res)

    deals = _fetch_bulk_block_deals()
    liq_synced = sync_indian_liquidity_indices(db)

    if deliv_rows:
        db.upsert_nse_delivery(deliv_rows)
    if fii_rows:
        db.upsert_nse_fii_oi(fii_rows)
    if deals:
        db.replace_nse_bulk_deals(deals)

    return {
        "dates_checked": len(dates),
        "delivery_rows": len(deliv_rows),
        "fii_oi_days": len(fii_rows),
        "bulk_block_deals": len(deals),
        "liquidity_indices": liq_synced,
    }


def evaluate_balance_sheet_shield(fundamentals: dict[str, Any], sector: str = "") -> dict[str, Any]:
    """Flag balance-sheet blowup risks (high debt, negative cash flow, weak ROE) cleanly."""
    flags: list[str] = []
    is_fin = "financial" in sector.lower() or "bank" in sector.lower()
    de = fundamentals.get("debt_to_equity")
    roe = fundamentals.get("roe_pct")
    fcf_y = fundamentals.get("fcf_yield_pct")
    rev_g = fundamentals.get("revenue_growth_pct")

    if not is_fin and de is not None and float(de) > 180.0:
        flags.append(f"High Debt/Equity ({float(de):.0f}%)")
    if roe is not None and float(roe) < 6.0:
        flags.append(f"Weak ROE ({float(roe):.1f}%)")
    if not is_fin and fcf_y is not None and float(fcf_y) < -4.0:
        flags.append(f"Negative FCF Yield ({float(fcf_y):.1f}%)")
    if rev_g is not None and float(rev_g) < -10.0:
        flags.append(f"Revenue Contraction ({float(rev_g):.1f}%)")

    status = "Watch" if len(flags) == 1 else ("Risk" if len(flags) >= 2 else "Clean")
    return {"status": status, "flags": flags}


def get_money_signals_summary(db: Database) -> dict[str, Any]:
    """Compute live Smart Money dashboard snapshot: FII/DII F&O, India Liquidity Ratios, Delivery Leaders & Bulk Deals."""
    fii_hist = db.get_nse_fii_oi(limit=20)
    deliv_all = db.get_nse_delivery_recent(days=20)
    deals = db.get_nse_bulk_deals(limit=60)
    tickers = {t["symbol"]: t for t in db.get_tickers()}

    # 1. FII / DII Derivatives Positioning
    fii_latest = fii_hist[0] if fii_hist else None
    fii_5d_chg = None
    if len(fii_hist) >= 6:
        fii_5d_chg = round(float(fii_hist[0]["fii_idx_long_pct"]) - float(fii_hist[5]["fii_idx_long_pct"]), 1)

    fii_tone = "neutral"
    fii_label = "Unavailable"
    if fii_latest:
        lp = float(fii_latest["fii_idx_long_pct"])
        if lp >= 55.0:
            fii_tone, fii_label = "bullish", f"FII Risk-On ({lp:.1f}% Long)"
        elif lp <= 35.0:
            fii_tone, fii_label = "bearish", f"FII Defensive / Short ({lp:.1f}% Long)"
        else:
            fii_tone, fii_label = "warning", f"FII Neutral ({lp:.1f}% Long)"

    # 2. India Midcap Liquidity & Credit Cycle Ratios (100% India 3:30 PM IST — Zero US lookahead!)
    macro = db.get_macro_series()
    bench_bars = db.get_ohlcv(BENCHMARK_SYMBOL)
    midcap_nifty_63d = None
    bank_nifty_63d = None
    if len(bench_bars) >= 65:
        bdf = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date")
        for sym, key in [("^NSEMDCP50", "mid"), ("^NSEBANK", "bnk")]:
            m_dict = macro.get(sym, {})
            if m_dict:
                s = pd.Series(m_dict, name=key, dtype=float)
                tmp = bdf.merge(s.rename_axis("date").reset_index(), on="date", how="left")
                tmp[key] = tmp[key].ffill()
                ratio = tmp[key] / tmp["close"].replace(0, np.nan)
                chg = ratio.pct_change(63).dropna()
                if not chg.empty:
                    if key == "mid":
                        midcap_nifty_63d = round(float(chg.iloc[-1] * 100.0), 2)
                    else:
                        bank_nifty_63d = round(float(chg.iloc[-1] * 100.0), 2)

    # 3. Stock-Level NSE Delivery % & Institutional Ticket Size Accumulation
    by_sym: dict[str, list[dict[str, Any]]] = {}
    for row in deliv_all:
        by_sym.setdefault(row["symbol"], []).append(row)

    stock_delivery: dict[str, dict[str, Any]] = {}
    for sym, rows in by_sym.items():
        rows_sorted = sorted(rows, key=lambda r: r["date"])
        latest = rows_sorted[-1]
        prev_rows = rows_sorted[:-1] if len(rows_sorted) > 1 else rows_sorted
        avg_deliv = float(np.mean([r["deliv_pct"] for r in prev_rows]))
        avg_ticket = float(np.mean([r["avg_trade_val_inr"] for r in prev_rows]))
        avg_turn = float(np.mean([r["turnover_lacs"] for r in prev_rows]))

        deliv_now = float(latest["deliv_pct"])
        deliv_surge = deliv_now - avg_deliv
        ticket_ratio = float(latest["avg_trade_val_inr"]) / max(1.0, avg_ticket)
        turn_ratio = float(latest["turnover_lacs"]) / max(1.0, avg_turn)

        # Smart Money Accumulation Score (0-100): rewards high absolute delivery + rising delivery + larger ticket size
        sm_score = float(
            np.clip(
                45.0
                + (deliv_now - 45.0) * 0.75
                + deliv_surge * 0.85
                + (ticket_ratio - 1.0) * 20.0
                + (turn_ratio - 1.0) * 8.0,
                5.0,
                98.0,
            )
        )

        if deliv_now >= 55.0 and deliv_surge >= 5.0:
            tag = "Strong Inst. Delivery"
            tone = "bullish"
        elif deliv_now >= 52.0:
            tag = "Steady Delivery"
            tone = "bullish"
        elif deliv_now < 35.0:
            tag = "Speculative / Intraday"
            tone = "warning"
        else:
            tag = "Balanced Flow"
            tone = "neutral"

        t_meta = tickers.get(sym, {})
        shield = evaluate_balance_sheet_shield(t_meta.get("fundamentals", {}), t_meta.get("sector", ""))
        stock_delivery[sym] = {
            "symbol": sym,
            "sector": t_meta.get("sector", "Other"),
            "as_of_date": latest["date"],
            "deliv_pct": round(deliv_now, 1),
            "avg_deliv_pct": round(avg_deliv, 1),
            "deliv_surge_pct": round(deliv_surge, 1),
            "turnover_cr": round(float(latest["turnover_lacs"]) / 100.0, 2),
            "avg_trade_val_inr": round(float(latest["avg_trade_val_inr"]), 0),
            "ticket_ratio": round(ticket_ratio, 2),
            "smart_money_score": round(sm_score, 1),
            "tag": tag,
            "tone": tone,
            "shield_status": shield["status"],
            "shield_flags": shield["flags"],
        }

    top_accum = sorted(stock_delivery.values(), key=lambda x: x["smart_money_score"], reverse=True)[:12]
    universe_deals = [d for d in deals if d["symbol"] in tickers]
    all_top_deals = sorted(deals, key=lambda d: d["value_cr"], reverse=True)[:15]

    return {
        "fii_positioning": {
            "latest": fii_latest,
            "fii_long_pct_5d_chg": fii_5d_chg,
            "label": fii_label,
            "tone": fii_tone,
            "history": list(reversed(fii_hist[:15])),
        },
        "liquidity_ratios": {
            "midcap_vs_nifty_63d_pct": midcap_nifty_63d,
            "bank_vs_nifty_63d_pct": bank_nifty_63d,
        },
        "stock_delivery_map": stock_delivery,
        "top_accumulation": top_accum,
        "universe_bulk_deals": universe_deals[:15],
        "top_nse_deals": all_top_deals,
    }
