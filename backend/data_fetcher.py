"""Live Indian (NSE) market data ingestion: 5Y OHLCV, India VIX & USD/INR, fundamentals, earnings, insiders & news."""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .catalyst_nlp import analyze_financial_news
from .config import (
    DEFAULT_STARRED,
    DEFAULT_UNIVERSE,
    GOLDBEES_SYMBOL,
    HISTORY_PERIOD,
    MACRO_SYMBOLS,
    TRI_BENCHMARK_SYMBOL,
)
from .database import Database

logger = logging.getLogger(__name__)


def score_financial_text(title: str, summary: str = "") -> tuple[float, str]:
    """Backward-compatible wrapper returning (sentiment_score, sentiment_label)."""
    score, label, _ = analyze_financial_news(title, summary)
    return score, label


def normalize_indian_symbol(raw_symbol: str) -> str:
    """Normalize user ticker input: auto-append .NS for plain NSE tickers like RELIANCE or ZOMATO."""
    sym = raw_symbol.strip().upper()
    if not re.match(r"^[A-Z0-9.\-^&=]{1,16}$", sym):
        raise ValueError(f"Invalid ticker symbol: {raw_symbol}")
    if sym.startswith("^") or "." in sym or "=" in sym:
        return sym
    return f"{sym}.NS"


def _safe_float(val: Any) -> float | None:
    try:
        if val is None or pd.isna(val):
            return None
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _extract_fundamentals(info: dict[str, Any]) -> dict[str, Any]:
    mcap = _safe_float(info.get("marketCap"))
    fcf = _safe_float(info.get("freeCashflow"))
    fcf_yield = (fcf / mcap * 100.0) if (fcf is not None and mcap and mcap > 0) else None
    rev_growth = _safe_float(info.get("revenueGrowth"))
    gross_m = _safe_float(info.get("grossMargins"))
    op_m = _safe_float(info.get("operatingMargins"))
    net_m = _safe_float(info.get("profitMargins"))
    roe = _safe_float(info.get("returnOnEquity"))
    div_yield = _safe_float(info.get("dividendYield"))
    if div_yield is not None and div_yield < 0.25:
        div_yield = div_yield * 100.0

    return {
        "currency": info.get("currency") or "INR",
        "market_cap_cr": round(mcap / 1e7, 1) if mcap else None,  # INR Crores (1 Cr = 10^7)
        "market_cap_b": round(mcap / 1e9, 2) if mcap else None,
        "trailing_pe": _safe_float(info.get("trailingPE")),
        "forward_pe": _safe_float(info.get("forwardPE")),
        "peg_ratio": _safe_float(info.get("pegRatio") or info.get("trailingPegRatio")),
        "price_to_sales": _safe_float(info.get("priceToSalesTrailing12Months")),
        "ev_to_ebitda": _safe_float(info.get("enterpriseToEbitda")),
        "eps_ttm": _safe_float(info.get("trailingEps")),
        "eps_forward": _safe_float(info.get("forwardEps")),
        "revenue_growth_pct": round(rev_growth * 100.0, 2) if rev_growth is not None else None,
        "gross_margin_pct": round(gross_m * 100.0, 2) if gross_m is not None else None,
        "operating_margin_pct": round(op_m * 100.0, 2) if op_m is not None else None,
        "net_margin_pct": round(net_m * 100.0, 2) if net_m is not None else None,
        "roe_pct": round(roe * 100.0, 2) if roe is not None else None,
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "fcf_yield_pct": round(fcf_yield, 2) if fcf_yield is not None else None,
        "dividend_yield_pct": round(div_yield, 2) if div_yield is not None else None,
        "beta": _safe_float(info.get("beta")),
        "analyst_target": _safe_float(info.get("targetMeanPrice")),
        "analyst_high": _safe_float(info.get("targetHighPrice")),
        "analyst_low": _safe_float(info.get("targetLowPrice")),
        "analyst_count": info.get("numberOfAnalystOpinions"),
        "recommendation": (info.get("recommendationKey") or "").replace("_", " ").title() or None,
        "promoter_held_pct": round(float(info["heldPercentInsiders"]) * 100.0, 1)
        if _safe_float(info.get("heldPercentInsiders")) is not None
        else None,
        "institution_held_pct": round(float(info["heldPercentInstitutions"]) * 100.0, 1)
        if _safe_float(info.get("heldPercentInstitutions")) is not None
        else None,
    }


def _parse_news_items(symbol: str, raw_news: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed = []
    for idx, item in enumerate(raw_news[:12]):
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = (content.get("title") or "").strip()
        if not title:
            continue
        summary = (content.get("summary") or content.get("description") or "").strip()
        provider = content.get("provider")
        publisher = (
            provider.get("displayName", "")
            if isinstance(provider, dict)
            else str(content.get("publisher") or "")
        )
        canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        url = canonical.get("url", "") if isinstance(canonical, dict) else str(content.get("link") or "")
        pub_date = str(content.get("pubDate") or content.get("displayTime") or "")
        if not pub_date and content.get("providerPublishTime"):
            pub_date = datetime.fromtimestamp(int(content["providerPublishTime"]), tz=timezone.utc).isoformat()

        score, label, tags = analyze_financial_news(title, summary)
        nid = str(item.get("id") or hashlib.sha1(f"{symbol}:{title}:{idx}".encode()).hexdigest()[:16])
        parsed.append(
            {
                "id": f"{symbol}_{nid}",
                "title": title,
                "summary": summary[:320],
                "publisher": publisher,
                "url": url,
                "published_at": pub_date,
                "sentiment_score": score,
                "sentiment_label": label,
                "catalyst_tags": tags,
            }
        )
    return parsed


def _parse_earnings(t: yf.Ticker) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return items
        for ts, row in ed.head(12).iterrows():
            dt = pd.Timestamp(ts).strftime("%Y-%m-%d")
            est = _safe_float(row.get("EPS Estimate"))
            act = _safe_float(row.get("Reported EPS"))
            surp = _safe_float(row.get("Surprise(%)"))
            if surp is None and est is not None and act is not None and abs(est) > 1e-6:
                surp = round(((act - est) / abs(est)) * 100.0, 2)
            items.append(
                {
                    "earnings_date": dt,
                    "eps_estimate": round(est, 4) if est is not None else None,
                    "eps_actual": round(act, 4) if act is not None else None,
                    "surprise_pct": round(surp, 2) if surp is not None else None,
                }
            )
    except Exception as exc:
        logger.debug("Earnings parse skipped: %s", exc)
    return items


def _parse_insiders(symbol: str, t: yf.Ticker) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        df = t.insider_transactions
        if df is None or df.empty:
            return items
        for idx, row in df.head(25).iterrows():
            raw_dt = row.get("Start Date")
            if pd.isna(raw_dt):
                continue
            dt = pd.Timestamp(raw_dt).strftime("%Y-%m-%d")
            insider = str(row.get("Insider") or "Unknown").strip()
            position = str(row.get("Position") or "").strip()
            tx = str(row.get("Transaction") or "").strip()
            text = str(row.get("Text") or "").strip()
            combined = f"{tx} {text}".lower()
            shares = _safe_float(row.get("Shares")) or 0.0
            value = _safe_float(row.get("Value")) or 0.0

            is_purchase = ("purchase" in combined or "buy" in combined or "acquisition" in combined) and "sale" not in combined and shares > 0
            if not tx:
                if is_purchase:
                    tx = "Open-Market Buy"
                elif "sale" in combined or "sold" in combined or "disposal" in combined:
                    tx = "Sale"
                elif "gift" in combined:
                    tx = "Gift"
                elif "award" in combined or "grant" in combined or "allotment" in combined:
                    tx = "ESOP / Allotment"
                else:
                    tx = text[:40] if text else "Promoter / Insider Filing"

            tid = hashlib.sha1(f"{symbol}:{dt}:{insider}:{shares}:{idx}".encode()).hexdigest()[:16]
            items.append(
                {
                    "id": f"{symbol}_ins_{tid}",
                    "trade_date": dt,
                    "insider": insider,
                    "position": position,
                    "transaction_type": tx[:50],
                    "shares": shares,
                    "value": value,
                    "is_purchase": is_purchase,
                    "source": "NSE / SEBI Insider Disclosure",
                }
            )
    except Exception as exc:
        logger.debug("Insider parse skipped for %s: %s", symbol, exc)
    return items


def _parse_options_snapshot(
    symbol: str,
    t: yf.Ticker,
    last_close: float | None,
    hv_20d_pct: float | None,
) -> dict[str, Any] | None:
    # Yahoo Finance does not publish NSE F&O option chains for .NS/.BO symbols
    if symbol.endswith(".NS") or symbol.endswith(".BO") or symbol.startswith("^"):
        return None
    try:
        exps = t.options
        if not exps:
            return None
        today = datetime.now(timezone.utc).date()
        chosen = exps[0]
        for exp in exps[:8]:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            if 10 <= dte <= 45:
                chosen = exp
                break
        chain = t.option_chain(chosen)
        calls, puts = chain.calls, chain.puts
        if calls.empty and puts.empty:
            return None

        c_vol = int(calls["volume"].fillna(0).sum()) if "volume" in calls else 0
        p_vol = int(puts["volume"].fillna(0).sum()) if "volume" in puts else 0
        c_oi = int(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls else 0
        p_oi = int(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts else 0

        pc_vol = round(p_vol / max(1, c_vol), 3) if (c_vol + p_vol) > 0 else None
        pc_oi = round(p_oi / max(1, c_oi), 3) if (c_oi + p_oi) > 0 else None

        atm_iv = None
        if last_close and last_close > 0 and not calls.empty and "strike" in calls and "impliedVolatility" in calls:
            near = calls.iloc[(calls["strike"] - last_close).abs().argsort()[:3]]
            ivs = near["impliedVolatility"].dropna()
            ivs = ivs[(ivs > 0.02) & (ivs < 3.5)]
            if len(ivs):
                atm_iv = round(float(ivs.median()) * 100.0, 1)

        iv_hv = round(atm_iv / max(5.0, hv_20d_pct), 2) if (atm_iv and hv_20d_pct and hv_20d_pct > 0) else None
        sent_parts = []
        if pc_vol is not None:
            sent_parts.append(float(np.clip((0.85 - pc_vol) * 1.4, -1.0, 1.0)))
        if pc_oi is not None:
            sent_parts.append(float(np.clip((0.90 - pc_oi) * 1.2, -1.0, 1.0)))
        opt_sent = round(float(np.mean(sent_parts)), 3) if sent_parts else 0.0

        return {
            "as_of_date": today.isoformat(),
            "expiry": chosen,
            "call_volume": c_vol,
            "put_volume": p_vol,
            "call_oi": c_oi,
            "put_oi": p_oi,
            "pc_vol_ratio": pc_vol,
            "pc_oi_ratio": pc_oi,
            "atm_iv_pct": atm_iv,
            "iv_hv_ratio": iv_hv,
            "options_sentiment": opt_sent,
        }
    except Exception:
        return None


def _df_to_ohlcv_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if df is None or df.empty:
        return rows
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    for ts, row in df.iterrows():
        date_str = pd.Timestamp(ts).strftime("%Y-%m-%d")
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        v = int(row["Volume"]) if pd.notna(row.get("Volume")) else 0
        if c <= 0 or h <= 0 or l <= 0:
            continue
        rows.append(
            {
                "date": date_str,
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(l, 4),
                "close": round(c, 4),
                "volume": max(0, v),
            }
        )
    return rows


def _fetch_ticker_enrichment(
    symbol: str,
    last_close: float | None = None,
    hv_20d_pct: float | None = None,
) -> dict[str, Any]:
    t = yf.Ticker(symbol)
    info: dict[str, Any] = {}
    news_items: list[dict[str, Any]] = []
    try:
        info = t.info or {}
    except Exception as exc:
        logger.debug("Info fetch failed for %s: %s", symbol, exc)
    try:
        news_items = _parse_news_items(symbol, t.news or [])
    except Exception as exc:
        logger.debug("News fetch failed for %s: %s", symbol, exc)

    is_index = symbol.startswith("^")
    earnings = [] if is_index else _parse_earnings(t)
    insiders = [] if is_index else _parse_insiders(symbol, t)
    options_snap = _parse_options_snapshot(symbol, t, last_close, hv_20d_pct)

    default_name = "NIFTY 50 Index" if symbol == "^NSEI" else symbol.replace(".NS", "")
    name = info.get("shortName") or info.get("longName") or default_name
    sector = info.get("sector") or ("Benchmark Index" if is_index else "Unknown")
    industry = info.get("industry") or (info.get("category") or "")
    exchange = info.get("exchange") or ("NSE" if symbol.endswith(".NS") or symbol == "^NSEI" else "")
    summary = (info.get("longBusinessSummary") or "")[:500]
    fundamentals = _extract_fundamentals(info)

    return {
        "symbol": symbol,
        "name": name,
        "sector": sector,
        "industry": industry,
        "exchange": exchange,
        "summary": summary,
        "fundamentals": fundamentals,
        "news": news_items,
        "earnings": earnings,
        "insiders": insiders,
        "options": options_snap,
    }


def _hv_from_rows(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 22:
        return None
    closes = np.array([r["close"] for r in rows[-22:]], dtype=float)
    lr = np.diff(np.log(closes))
    return round(float(np.std(lr, ddof=1) * np.sqrt(252) * 100.0), 1)


def sync_macro_series(db: Database, period: str = HISTORY_PERIOD) -> dict[str, int]:
    """Fetch 5-year Indian macro indicators (^INDIAVIX, INR=X) into SQLite."""
    counts: dict[str, int] = {}
    try:
        raw = yf.download(
            tickers=MACRO_SYMBOLS,
            period=period,
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if raw is not None and not raw.empty:
            for sym in MACRO_SYMBOLS:
                sub = raw[sym] if isinstance(raw.columns, pd.MultiIndex) and sym in raw.columns.get_level_values(0) else None
                if sub is None or sub.empty or "Close" not in sub:
                    continue
                pairs = [
                    (pd.Timestamp(ts).strftime("%Y-%m-%d"), round(float(val), 4))
                    for ts, val in sub["Close"].dropna().items()
                ]
                counts[sym] = db.upsert_macro_series(sym, pairs)
    except Exception as exc:
        logger.warning("India macro sync failed: %s", exc)
    return counts


def sync_single_ticker(db: Database, symbol: str, period: str = HISTORY_PERIOD) -> dict[str, Any]:
    """Fetch live NSE (auto-.NS) or specified ticker data and persist to SQLite."""
    candidate = normalize_indian_symbol(symbol)
    raw_clean = symbol.strip().upper()

    t = yf.Ticker(candidate)
    hist = t.history(period=period, interval="1d", auto_adjust=True)
    rows = _df_to_ohlcv_rows(hist)
    sym = candidate

    # Fallback to raw symbol (e.g., if user entered a BSE .BO or global ticker)
    if len(rows) < 25 and candidate != raw_clean:
        t2 = yf.Ticker(raw_clean)
        hist2 = t2.history(period=period, interval="1d", auto_adjust=True)
        rows2 = _df_to_ohlcv_rows(hist2)
        if len(rows2) >= 25:
            rows = rows2
            sym = raw_clean

    if len(rows) < 25:
        raise ValueError(f"Could not find sufficient NSE/market history for '{symbol}' on Yahoo Finance.")

    last_close = rows[-1]["close"]
    hv = _hv_from_rows(rows)
    meta = _fetch_ticker_enrichment(sym, last_close=last_close, hv_20d_pct=hv)
    now_iso = datetime.now(timezone.utc).isoformat()

    db.upsert_ticker(
        symbol=sym,
        name=meta["name"],
        sector=meta["sector"],
        industry=meta["industry"],
        exchange=meta["exchange"],
        summary=meta["summary"],
        fundamentals=meta["fundamentals"],
        data_source="NSE / Yahoo Finance (live)",
        updated_at=now_iso,
        default_starred=(sym in DEFAULT_STARRED),
    )
    db.upsert_ohlcv(sym, rows)
    db.replace_news(sym, meta["news"])
    db.replace_earnings(sym, meta["earnings"])
    db.replace_insiders(sym, meta["insiders"])
    db.upsert_options_snapshot(sym, meta["options"])

    return {
        "symbol": sym,
        "name": meta["name"],
        "bars": len(rows),
        "news_count": len(meta["news"]),
        "earnings_count": len(meta["earnings"]),
        "insiders_count": len(meta["insiders"]),
        "has_options": meta["options"] is not None,
        "data_source": "NSE / Yahoo Finance (live)",
        "updated_at": now_iso,
    }


def sync_universe(
    db: Database,
    symbols: list[str] | None = None,
    period: str = HISTORY_PERIOD,
) -> dict[str, Any]:
    """Batch-sync 10Y NSE OHLCV (+ NIFTYBEES TRI & GOLDBEES reserve ETFs), India VIX & macro, fundamentals, earnings, insiders & news."""
    target_symbols = [s.upper() for s in (symbols or DEFAULT_UNIVERSE)]
    for aux_etf in (TRI_BENCHMARK_SYMBOL, GOLDBEES_SYMBOL):
        if aux_etf and aux_etf.upper() not in target_symbols:
            target_symbols.append(aux_etf.upper())
    now_iso = datetime.now(timezone.utc).isoformat()

    macro_counts = sync_macro_series(db, period=period)

    ohlcv_map: dict[str, list[dict[str, Any]]] = {}
    try:
        raw = yf.download(
            tickers=target_symbols,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                top_levels = set(raw.columns.get_level_values(0))
                for sym in target_symbols:
                    if sym in top_levels:
                        ohlcv_map[sym] = _df_to_ohlcv_rows(raw[sym])
            elif len(target_symbols) == 1:
                ohlcv_map[target_symbols[0]] = _df_to_ohlcv_rows(raw)
    except Exception as exc:
        logger.warning("Batch yf.download failed (%s); falling back per-symbol", exc)

    meta_map: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {}
        for sym in target_symbols:
            rows = ohlcv_map.get(sym, [])
            lc = rows[-1]["close"] if rows else None
            hv = _hv_from_rows(rows)
            futures[pool.submit(_fetch_ticker_enrichment, sym, lc, hv)] = sym
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                meta_map[sym] = fut.result()
            except Exception as exc:
                logger.debug("Enrichment error for %s: %s", sym, exc)

    synced = 0
    failed: list[str] = []
    for sym in target_symbols:
        rows = ohlcv_map.get(sym, [])
        if len(rows) < 30:
            try:
                hist = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=True)
                rows = _df_to_ohlcv_rows(hist)
            except Exception:
                rows = []
        if len(rows) < 25:
            failed.append(sym)
            continue

        meta = meta_map.get(
            sym,
            {
                "name": sym.replace(".NS", ""),
                "sector": "Unknown",
                "industry": "",
                "exchange": "NSE",
                "summary": "",
                "fundamentals": {},
                "news": [],
                "earnings": [],
                "insiders": [],
                "options": None,
            },
        )
        db.upsert_ticker(
            symbol=sym,
            name=meta["name"],
            sector=meta["sector"],
            industry=meta["industry"],
            exchange=meta["exchange"],
            summary=meta["summary"],
            fundamentals=meta["fundamentals"],
            data_source="NSE / Yahoo Finance (live)",
            updated_at=now_iso,
            default_starred=(sym in DEFAULT_STARRED),
        )
        db.upsert_ohlcv(sym, rows)
        db.replace_news(sym, meta["news"])
        db.replace_earnings(sym, meta["earnings"])
        db.replace_insiders(sym, meta["insiders"])
        db.upsert_options_snapshot(sym, meta["options"])
        synced += 1

    # Phase 2 Day-1 Immutable PIT Fundamental Snapshot Logger (zero lookahead)
    pit_summary: dict[str, Any] = {}
    try:
        from .pit_fundamentals import snapshot_current_fundamentals

        pit_summary = snapshot_current_fundamentals(db, as_of_date=now_iso[:10])
    except Exception as exc:
        logger.debug("PIT fundamental snapshot skipped: %s", exc)

    return {
        "synced_count": synced,
        "failed": failed,
        "macro_synced": macro_counts,
        "pit_fundamentals": pit_summary,
        "updated_at": now_iso,
        "data_source": "NSE / Yahoo Finance (live)",
    }
