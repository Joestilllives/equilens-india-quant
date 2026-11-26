"""
EquiLens India V7 (Phase 2) — 100% Free Public Point-In-Time (PIT) Data Pipelines.

Implements three strictly zero-lookahead, 100% free data pipelines:
  1. Day-1 Immutable PIT Fundamental Snapshot Logger (`pit_fundamentals_history` in SQLite):
     Persists immutable (symbol, as_of_date, trailing_pe, ev_to_ebitda, fcf_yield,
     gross_profit_to_assets, accruals, roe, debt_to_equity) snapshots via INSERT OR IGNORE.
  2. NSE Bhavcopy Delivery Archive Backfill (`nsearchives.nseindia.com`):
     Parses and ingests historical daily `% Deliverable Quantity` and delivery value from
     public NSE `sec_bhavdata_full_DDMMYYYY.csv` / `MTO_DDMMYYYY.DAT` archives into `nse_delivery`.
  3. BSE/NSE Quarterly XBRL Filing Parser (+2 NSE Trading-Day Availability Lag) &
     Orthogonal PIT Value + Gross Profitability + Low Accruals Sleeve
     (Asness, Moskowitz & Pedersen 2013; Novy-Marx 2013; Sloan 1996).
"""

from __future__ import annotations

import csv
import io
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import NON_EQUITY_SYMBOLS
from .database import Database
from .nse_calendar import add_nse_trading_days, is_nse_trading_day

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        v = float(str(val).strip().replace(",", ""))
        return v if np.isfinite(v) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Day-1 Immutable PIT Fundamental Snapshot Logger
# ---------------------------------------------------------------------------
def snapshot_current_fundamentals(
    db: Database,
    as_of_date: str | None = None,
    source: str = "yfinance_daily_snapshot",
) -> dict[str, Any]:
    """
    Write immutable Day-1 PIT fundamental snapshots from `tickers` table into `pit_fundamentals_history`.
    Uses INSERT OR IGNORE on (symbol, as_of_date) so historical dates are never overwritten.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    snap_date = str(as_of_date or datetime.now(timezone.utc).date().isoformat())[:10]
    tickers = db.get_tickers()

    rows: list[dict[str, Any]] = []
    for t in tickers:
        sym = str(t["symbol"]).upper()
        if sym in NON_EQUITY_SYMBOLS or sym.startswith("^"):
            continue
        f = t.get("fundamentals") or {}
        if not isinstance(f, dict) or not f:
            continue

        trailing_pe = _safe_float(f.get("trailing_pe"))
        ev_to_ebitda = _safe_float(f.get("ev_to_ebitda"))
        fcf_yield_pct = _safe_float(f.get("fcf_yield_pct"))
        fcf_yield = (fcf_yield_pct / 100.0) if fcf_yield_pct is not None else _safe_float(f.get("fcf_yield"))
        roe_pct = _safe_float(f.get("roe_pct"))
        roe = (roe_pct / 100.0) if roe_pct is not None else _safe_float(f.get("roe"))
        debt_to_equity = _safe_float(f.get("debt_to_equity"))
        market_cap_cr = _safe_float(f.get("market_cap_cr"))

        # Gross profitability proxy if direct GP/Assets not provided in Yahoo summary
        gp_to_assets = _safe_float(f.get("gross_profit_to_assets"))
        if gp_to_assets is None:
            gm_pct = _safe_float(f.get("gross_margin_pct"))
            p_to_s = _safe_float(f.get("price_to_sales"))
            if gm_pct is not None and p_to_s is not None and p_to_s > 0.05:
                # Approximate Gross Profit / Market Value proxy when total assets unavailable
                gp_to_assets = round((gm_pct / 100.0) / max(0.25, p_to_s), 5)

        accruals = _safe_float(f.get("accruals"))
        if accruals is None and fcf_yield is not None and trailing_pe is not None and trailing_pe > 0.5:
            # Cash-conversion accrual proxy: E/P - FCF/P (lower = higher cash quality, Sloan 1996)
            accruals = round((1.0 / trailing_pe) - fcf_yield, 5)

        rows.append(
            {
                "symbol": sym,
                "as_of_date": snap_date,
                "filing_date": None,
                "trailing_pe": trailing_pe,
                "ev_to_ebitda": ev_to_ebitda,
                "fcf_yield": fcf_yield,
                "gross_profit_to_assets": gp_to_assets,
                "accruals": accruals,
                "roe": roe,
                "debt_to_equity": debt_to_equity,
                "market_cap_cr": market_cap_cr,
                "source": source,
                "created_at": now_iso,
            }
        )

    inserted = db.insert_pit_fundamentals_snapshots(rows)
    return {
        "as_of_date": snap_date,
        "candidates": len(rows),
        "inserted_new": inserted,
        "total_rows_in_store": db.get_pit_fundamentals_count(),
    }


# ---------------------------------------------------------------------------
# 2. NSE Bhavcopy Delivery Archive Parser & Backfill (100% Free Public NSE)
# ---------------------------------------------------------------------------
def parse_nse_bhavcopy_csv(
    csv_text: str,
    trade_date: str,
    allowed_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Parse official NSE `sec_bhavdata_full_DDMMYYYY.csv` delivery archive into `nse_delivery` rows.
    Only keeps EQ / BE series equity rows.
    """
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    for raw in reader:
        rec = {str(k).strip().upper(): str(v).strip() for k, v in raw.items() if k is not None}
        series = rec.get("SERIES", "")
        if series not in ("EQ", "BE"):
            continue
        base_sym = rec.get("SYMBOL", "").upper()
        if not base_sym:
            continue
        sym_ns = f"{base_sym}.NS"
        if allowed_symbols is not None and sym_ns not in allowed_symbols:
            continue
        close_px = _safe_float(rec.get("CLOSE_PRICE")) or 0.0
        ttl_qty = int(_safe_float(rec.get("TTL_TRD_QNTY")) or 0)
        deliv_qty = int(_safe_float(rec.get("DELIV_QTY")) or 0)
        deliv_pct = _safe_float(rec.get("DELIV_PER"))
        if deliv_pct is None and ttl_qty > 0:
            deliv_pct = (deliv_qty / ttl_qty) * 100.0
        if deliv_pct is None:
            continue
        turnover_lacs = _safe_float(rec.get("TURNOVER_LACS")) or 0.0
        no_of_trades = int(_safe_float(rec.get("NO_OF_TRADES")) or 0)
        avg_trade_val = (turnover_lacs * 100_000.0 / max(1, no_of_trades)) if no_of_trades > 0 else 0.0
        rows.append(
            {
                "symbol": sym_ns,
                "date": trade_date[:10],
                "close_price": round(close_px, 4),
                "traded_qty": ttl_qty,
                "deliv_qty": deliv_qty,
                "deliv_pct": round(float(np.clip(deliv_pct, 0.0, 100.0)), 2),
                "turnover_lacs": round(turnover_lacs, 4),
                "no_of_trades": no_of_trades,
                "avg_trade_val_inr": round(avg_trade_val, 2),
            }
        )
    return rows


def parse_nse_mto_dat(
    dat_text: str,
    trade_date: str,
    allowed_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Parse legacy NSE `MTO_DDMMYYYY.DAT` (Record Type 20) daily delivery report."""
    rows: list[dict[str, Any]] = []
    for line in dat_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7 or parts[0] != "20":
            continue
        if parts[3] not in ("EQ", "BE"):
            continue
        sym_ns = f"{parts[2].upper()}.NS"
        if allowed_symbols is not None and sym_ns not in allowed_symbols:
            continue
        traded_qty = int(_safe_float(parts[4]) or 0)
        deliv_qty = int(_safe_float(parts[5]) or 0)
        deliv_pct = _safe_float(parts[6])
        if deliv_pct is None:
            continue
        rows.append(
            {
                "symbol": sym_ns,
                "date": trade_date[:10],
                "close_price": 0.0,
                "traded_qty": traded_qty,
                "deliv_qty": deliv_qty,
                "deliv_pct": round(float(np.clip(deliv_pct, 0.0, 100.0)), 2),
                "turnover_lacs": 0.0,
                "no_of_trades": 0,
                "avg_trade_val_inr": 0.0,
            }
        )
    return rows


def backfill_nse_bhavcopy_delivery(
    db: Database,
    start_date: str,
    end_date: str | None = None,
    sleep_sec: float = 0.25,
    max_days: int | None = None,
) -> dict[str, Any]:
    """
    Backfill historical NSE daily delivery % from public `nsearchives.nseindia.com` CSV/DAT archives.
    Skips dates already stored in `nse_delivery` for idempotent resumable runs.
    """
    d_start = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
    d_end = datetime.strptime(end_date[:10], "%Y-%m-%d").date() if end_date else date.today()
    allowed_syms = {
        str(t["symbol"]).upper()
        for t in db.get_tickers()
        if str(t["symbol"]).upper() not in NON_EQUITY_SYMBOLS and str(t["symbol"]).upper().endswith(".NS")
    }
    existing_hist = db.get_nse_delivery_history()
    existing_dates = {dt for sym_map in existing_hist.values() for dt in sym_map.keys()}

    cur = d_start
    trading_days: list[date] = []
    while cur <= d_end:
        if is_nse_trading_day(cur) and cur.isoformat() not in existing_dates:
            trading_days.append(cur)
        cur += timedelta(days=1)
    if max_days is not None:
        trading_days = trading_days[-max_days:]

    fetched_days = 0
    total_rows = 0
    errors: list[str] = []

    for d in trading_days:
        ddmmyyyy = d.strftime("%d%m%Y")
        iso = d.isoformat()
        urls = [
            ("csv", f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"),
            ("dat", f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{ddmmyyyy}.DAT"),
        ]
        day_rows: list[dict[str, Any]] = []
        last_err = None
        for kind, url in urls:
            try:
                req = urllib.request.Request(url, headers=_NSE_HEADERS)
                with urllib.request.urlopen(req, timeout=8) as resp:
                    if resp.status == 200:
                        text = resp.read().decode("utf-8", errors="ignore")
                        day_rows = (
                            parse_nse_bhavcopy_csv(text, iso, allowed_syms)
                            if kind == "csv"
                            else parse_nse_mto_dat(text, iso, allowed_syms)
                        )
                        if day_rows:
                            break
            except Exception as exc:
                last_err = f"{iso}: {type(exc).__name__}: {exc}"
        if day_rows:
            db.upsert_nse_delivery(day_rows)
            fetched_days += 1
            total_rows += len(day_rows)
        elif last_err and len(errors) < 10:
            errors.append(last_err)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return {
        "requested_days": len(trading_days),
        "fetched_days": fetched_days,
        "inserted_rows": total_rows,
        "errors_sample": errors,
    }


# ---------------------------------------------------------------------------
# 3. BSE/NSE Quarterly XBRL Parser (+2 Trading-Day Lag) & PIT Value-Quality Sleeve
# ---------------------------------------------------------------------------
_XBRL_TAG_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("revenuefromoperations", "totalrevenuefromoperations", "revenue"),
    "cogs": ("costofmaterialsconsumed", "costofgoodssold", "purchasesofstockintrade"),
    "gross_profit": ("grossprofit",),
    "net_income": ("profitlossforperiod", "profitlossfortheperiod", "netprofitlossfortheperiod"),
    "total_assets": ("assets", "totalassets"),
    "cfo": ("cashflowsfromusedinoperatingactivities", "netcashflowfromoperatingactivities"),
    "capex": ("purchaseofpropertyplantandequipment", "paymentsforpropertyplantandequipment"),
    "total_equity": ("equity", "totalequity", "equityattributabletoownersofparent"),
    "total_debt": ("borrowings", "noncurrentborrowings", "totalborrowings"),
}


def parse_xbrl_quarterly_filing(
    xml_bytes: bytes | str,
    symbol: str,
    filing_date: str,
    market_cap_cr: float | None = None,
    enterprise_value_cr: float | None = None,
    annualize_flow_mult: float = 4.0,
) -> dict[str, Any]:
    """
    Parse a public BSE/NSE Ind-AS XBRL quarterly/annual filing XML into a strictly causal
    PIT fundamental record stamped with `as_of_date = add_nse_trading_days(filing_date, 2)`.
    """
    raw_bytes = xml_bytes.encode("utf-8") if isinstance(xml_bytes, str) else xml_bytes
    root = ET.fromstring(raw_bytes)

    extracted: dict[str, float] = {}
    for elem in root.iter():
        tag_local = elem.tag.split("}")[-1].lower()
        val = _safe_float(elem.text)
        if val is None:
            continue
        for field, aliases in _XBRL_TAG_ALIASES.items():
            if field not in extracted and tag_local in aliases:
                extracted[field] = val

    rev = extracted.get("revenue")
    cogs = extracted.get("cogs", 0.0)
    gp = extracted.get("gross_profit")
    if gp is None and rev is not None:
        gp = rev - cogs
    net_inc = extracted.get("net_income")
    tot_assets = extracted.get("total_assets")
    cfo = extracted.get("cfo")
    capex = abs(extracted.get("capex", 0.0))
    tot_eq = extracted.get("total_equity")
    tot_debt = extracted.get("total_debt")

    # Convert to annualized ratios; note Ind-AS XBRL values are typically in Lakhs or Crores (ratios cancel units)
    gp_to_assets = None
    if gp is not None and tot_assets is not None and tot_assets > 0:
        gp_to_assets = round((gp * annualize_flow_mult) / tot_assets, 5)

    accruals = None
    if net_inc is not None and cfo is not None and tot_assets is not None and tot_assets > 0:
        # Sloan (1996): Accruals = (Net Income - Operating Cash Flow) / Total Assets (lower is better)
        accruals = round(((net_inc - cfo) * annualize_flow_mult) / tot_assets, 5)

    roe = None
    if net_inc is not None and tot_eq is not None and tot_eq > 0:
        roe = round((net_inc * annualize_flow_mult) / tot_eq, 5)

    debt_to_equity = None
    if tot_debt is not None and tot_eq is not None and tot_eq > 0:
        debt_to_equity = round((tot_debt / tot_eq) * 100.0, 2)

    fcf_yield = None
    trailing_pe = None
    if cfo is not None:
        fcf_ann = (cfo - capex) * annualize_flow_mult
        denom_ev = enterprise_value_cr or market_cap_cr
        if denom_ev is not None and denom_ev > 0:
            fcf_yield = round(fcf_ann / denom_ev, 5)
    if net_inc is not None and net_inc > 0 and market_cap_cr is not None and market_cap_cr > 0:
        trailing_pe = round(market_cap_cr / (net_inc * annualize_flow_mult), 3)

    # Strict +2 NSE trading-day availability lag (zero filing-day lookahead!)
    effective_as_of = add_nse_trading_days(str(filing_date)[:10], 2)
    return {
        "symbol": str(symbol).upper(),
        "as_of_date": effective_as_of,
        "filing_date": str(filing_date)[:10],
        "trailing_pe": trailing_pe,
        "ev_to_ebitda": None,
        "fcf_yield": fcf_yield,
        "gross_profit_to_assets": gp_to_assets,
        "accruals": accruals,
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "market_cap_cr": market_cap_cr,
        "source": "bse_nse_xbrl_lag2d",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _cs_z(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 3:
        return pd.Series(0.0, index=series.index)
    std = float(valid.std(ddof=0))
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return ((series - float(valid.mean())) / std).clip(-3.0, 3.0).fillna(0.0)


def compute_pit_value_quality_sleeve(
    db: Database,
    as_of_date: str,
) -> pd.DataFrame:
    """
    Construct strictly causal Point-In-Time Value + Gross Profitability + Low Accruals sleeve
    (Asness, Moskowitz & Pedersen 2013; Novy-Marx 2013; Sloan 1996) using ONLY snapshots with
    `as_of_date <= query_date`.

    Composite score:
      - 45% Value: 0.5 * z(E/P) + 0.5 * z(FCF/EV)  (or available subset)
      - 35% Gross Profitability: z(GP/Assets)
      - 20% Earnings Quality: -z(Accruals)
    """
    snaps = db.get_pit_fundamentals_as_of(as_of_date)
    if not snaps:
        return pd.DataFrame(columns=["symbol", "as_of_date", "pit_value_score", "pit_gp_score", "pit_accrual_score", "pit_vq_composite"])

    rows = []
    for sym, r in snaps.items():
        if sym in NON_EQUITY_SYMBOLS:
            continue
        pe = _safe_float(r.get("trailing_pe"))
        ep = (1.0 / pe) if (pe is not None and pe > 1.0) else np.nan
        fcf_y = _safe_float(r.get("fcf_yield"))
        ev_ebitda = _safe_float(r.get("ev_to_ebitda"))
        ebitda_ev = (1.0 / ev_ebitda) if (ev_ebitda is not None and ev_ebitda > 0.5) else np.nan
        gp_a = _safe_float(r.get("gross_profit_to_assets"))
        acc = _safe_float(r.get("accruals"))
        rows.append(
            {
                "symbol": sym,
                "as_of_date": r["as_of_date"],
                "ep": ep,
                "fcf_yield": fcf_y if fcf_y is not None else np.nan,
                "ebitda_ev": ebitda_ev,
                "gp_to_assets": gp_a if gp_a is not None else np.nan,
                "accruals": acc if acc is not None else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "pit_value_score", "pit_gp_score", "pit_accrual_score", "pit_vq_composite"])

    z_ep = _cs_z(df["ep"])
    z_fcf = _cs_z(df["fcf_yield"])
    z_eb = _cs_z(df["ebitda_ev"])
    df["pit_value_score"] = (0.40 * z_ep + 0.40 * z_fcf + 0.20 * z_eb).round(4)
    df["pit_gp_score"] = _cs_z(df["gp_to_assets"]).round(4)
    df["pit_accrual_score"] = (-_cs_z(df["accruals"])).round(4)
    df["pit_vq_composite"] = (
        0.45 * df["pit_value_score"]
        + 0.35 * df["pit_gp_score"]
        + 0.20 * df["pit_accrual_score"]
    ).round(4)
    return df.sort_values("pit_vq_composite", ascending=False).reset_index(drop=True)
