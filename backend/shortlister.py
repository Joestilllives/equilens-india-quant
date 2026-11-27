"""Multi-factor bullish opportunity shortlister integrating ML Triple-Barrier, PEAD, Insiders, Options & Catalysts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import BENCHMARK_SYMBOL, BENCHMARK_TRI_SYMBOL, GOLDBEES_SYMBOL, STRATEGY_ID_SHORTLIST
from .database import Database
from .indicators import compute_indicator_frame, detect_chart_setups


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, val)))


def _summarize_earnings(earnings: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute EPS beat streak, average 4Q surprise %, and days until next earnings."""
    today = datetime.now(timezone.utc).date()
    upcoming_days: int | None = None
    next_date: str | None = None
    past_surprises: list[float] = []

    for e in earnings:
        try:
            dt = datetime.strptime(e["earnings_date"][:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if e.get("eps_actual") is None and dt >= today:
            days = (dt - today).days
            if upcoming_days is None or days < upcoming_days:
                upcoming_days = days
                next_date = dt.isoformat()
        elif e.get("surprise_pct") is not None:
            past_surprises.append(float(e["surprise_pct"]))

    beat_streak = 0
    for s in past_surprises:
        if s > 0:
            beat_streak += 1
        else:
            break
    avg_surp = round(float(np.mean(past_surprises[:4])), 2) if past_surprises else None
    return {
        "next_earnings_date": next_date,
        "days_to_earnings": upcoming_days,
        "beat_streak": beat_streak,
        "avg_4q_surprise_pct": avg_surp,
        "last_surprise_pct": round(past_surprises[0], 2) if past_surprises else None,
    }


def _summarize_insiders(insiders: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect recent 90-day open-market insider purchases and executive cluster buys."""
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=90)).isoformat()
    buys = [x for x in insiders if x.get("is_purchase") and x.get("trade_date", "") >= cutoff]
    sales = [
        x
        for x in insiders
        if (not x.get("is_purchase"))
        and "sale" in (x.get("transaction_type") or "").lower()
        and x.get("trade_date", "") >= cutoff
    ]
    unique_buyers = {b["insider"] for b in buys}
    buy_val = sum(float(b.get("value") or 0.0) for b in buys)
    sale_val = sum(float(s.get("value") or 0.0) for s in sales)
    return {
        "buys_90d": len(buys),
        "unique_buyers_90d": len(unique_buyers),
        "buy_value_90d": round(buy_val, 0),
        "sales_90d": len(sales),
        "sale_value_90d": round(sale_val, 0),
        "cluster_buy": len(unique_buyers) >= 2 or (len(buys) >= 1 and buy_val >= 500_000),
    }


def _score_ml_alpha(pred: dict[str, Any] | None) -> float:
    if not pred:
        return 50.0
    alpha = float(pred.get("pred_alpha_40d", pred.get("pred_alpha_20d", 0.0)))
    prob_beat = float(pred.get("prob_outperform", 50.0))
    prob_tb = float(pred.get("prob_tb_win", 50.0))
    s_alpha = _clamp(50.0 + alpha * 6.5, 5.0, 98.0)
    s_beat = _clamp((prob_beat - 25.0) * 1.9, 5.0, 98.0)
    s_tb = _clamp((prob_tb - 25.0) * 1.9, 5.0, 98.0)
    return round(0.60 * s_alpha + 0.20 * s_beat + 0.20 * s_tb, 1)


def _score_momentum(last: pd.Series) -> float:
    score = 50.0
    close = float(last["close"])
    sma20, sma50, sma200 = last.get("sma20"), last.get("sma50"), last.get("sma200")
    rs63 = last.get("rs_spy_63d")
    rs21 = last.get("rs_spy_21d")
    macd_h = last.get("macd_hist")
    vol_r = last.get("vol_ratio_5_20")

    if pd.notna(sma50):
        score += 12.0 if close > sma50 else -12.0
    if pd.notna(sma200):
        score += 10.0 if close > sma200 else -10.0
    if pd.notna(sma20) and pd.notna(sma50):
        score += 6.0 if sma20 > sma50 else -6.0
    if pd.notna(rs63):
        score += float(np.clip(rs63 * 110.0, -18.0, 20.0))
    if pd.notna(rs21):
        score += float(np.clip(rs21 * 80.0, -8.0, 10.0))
    if pd.notna(macd_h):
        score += 5.0 if macd_h > 0 else -4.0
    if pd.notna(vol_r) and vol_r >= 1.15 and pd.notna(last.get("ret_5d")) and last["ret_5d"] > 0:
        score += 6.0
    return round(_clamp(score), 1)


def _is_financial_sector(sector: str) -> bool:
    s = (sector or "").lower()
    return "financial" in s or "bank" in s


def _build_sector_percentile_tables(
    tickers: list[dict[str, Any]],
    last_prices: dict[str, float],
    earn_summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute within-NSE-sector percentile ranks for Valuation and Quality pillars.

    Accounts for structural industry differences (e.g., skips FCF yield and gross margin
    for Financial Services / Banks; normalizes P/E, ROE, margins within each NSE sector,
    falling back to universe-wide percentiles only for sectors with < 4 peers).
    """
    aux_excluded = {BENCHMARK_SYMBOL, BENCHMARK_TRI_SYMBOL, GOLDBEES_SYMBOL}
    rows: list[dict[str, Any]] = []
    for t in tickers:
        sym = t["symbol"]
        if sym in aux_excluded or sym not in last_prices:
            continue
        fund = t.get("fundamentals") or {}
        sec = t.get("sector") or "Unknown"
        px = last_prices[sym]
        is_fin = _is_financial_sector(sec)
        target = fund.get("analyst_target")
        upside = (((float(target) / px) - 1.0) * 100.0) if (target and float(target) > 0 and px > 0) else np.nan
        fwd_pe = float(fund["forward_pe"]) if (fund.get("forward_pe") and float(fund["forward_pe"]) > 0) else np.nan
        peg = float(fund["peg_ratio"]) if (fund.get("peg_ratio") and float(fund["peg_ratio"]) > 0) else np.nan
        fcf_y = np.nan if is_fin else (float(fund["fcf_yield_pct"]) if fund.get("fcf_yield_pct") is not None else np.nan)
        rev_g = float(fund["revenue_growth_pct"]) if fund.get("revenue_growth_pct") is not None else np.nan
        gm = np.nan if is_fin else (float(fund["gross_margin_pct"]) if fund.get("gross_margin_pct") is not None else np.nan)
        om = float(fund["operating_margin_pct"]) if fund.get("operating_margin_pct") is not None else np.nan
        roe = float(fund["roe_pct"]) if fund.get("roe_pct") is not None else np.nan
        surp = earn_summaries.get(sym, {}).get("avg_4q_surprise_pct")
        rows.append(
            {
                "symbol": sym,
                "sector": sec,
                "neg_fwd_pe": -fwd_pe if np.isfinite(fwd_pe) else np.nan,
                "neg_peg": -peg if np.isfinite(peg) else np.nan,
                "fcf_y": fcf_y,
                "upside": upside,
                "rev_g": rev_g,
                "gm": gm,
                "om": om,
                "roe": roe,
                "surp": float(surp) if surp is not None else np.nan,
            }
        )

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    sec_counts = df["sector"].value_counts().to_dict()

    def _sector_pct(col: str) -> pd.Series:
        univ_pct = df[col].rank(pct=True) * 100.0
        sec_pct = df.groupby("sector")[col].rank(pct=True) * 100.0
        # Use within-sector rank when sector has >= 4 valid peers, else universe rank
        sec_valid = df.groupby("sector")[col].transform(lambda s: s.notna().sum())
        use_sec = (df["sector"].map(sec_counts) >= 4) & (sec_valid >= 4)
        return np.where(use_sec, sec_pct, univ_pct)

    val_cols = ["neg_fwd_pe", "neg_peg", "fcf_y", "upside"]
    qual_cols = ["rev_g", "gm", "om", "roe", "surp"]
    for c in val_cols + qual_cols:
        df[f"p_{c}"] = _sector_pct(c)

    out: dict[str, dict[str, float]] = {}
    for _, r in df.iterrows():
        v_vals = [float(r[f"p_{c}"]) for c in val_cols if pd.notna(r[f"p_{c}"])]
        q_vals = [float(r[f"p_{c}"]) for c in qual_cols if pd.notna(r[f"p_{c}"])]
        out[str(r["symbol"])] = {
            "valuation": round(_clamp(float(np.mean(v_vals)), 10.0, 96.0), 1) if v_vals else 50.0,
            "quality": round(_clamp(float(np.mean(q_vals)), 10.0, 96.0), 1) if q_vals else 50.0,
        }
    return out


def _score_valuation(fund: dict[str, Any], last_price: float, sector_scores: dict[str, float] | None = None) -> float:
    if sector_scores is not None:
        return sector_scores["valuation"]
    parts: list[float] = []
    fwd_pe = fund.get("forward_pe")
    if fwd_pe is not None and fwd_pe > 0:
        parts.append(_clamp(108.0 - fwd_pe * 1.8, 10.0, 95.0))
    peg = fund.get("peg_ratio")
    if peg is not None and peg > 0:
        parts.append(_clamp(105.0 - peg * 26.0, 10.0, 95.0))
    fcf_y = fund.get("fcf_yield_pct")
    if fcf_y is not None:
        parts.append(_clamp(35.0 + fcf_y * 10.0, 10.0, 95.0))
    target = fund.get("analyst_target")
    if target is not None and target > 0 and last_price > 0:
        upside_pct = ((target / last_price) - 1.0) * 100.0
        parts.append(_clamp(50.0 + upside_pct * 1.9, 10.0, 98.0))
    return round(float(np.mean(parts)), 1) if parts else 50.0


def _score_quality(fund: dict[str, Any], earn_sum: dict[str, Any], sector_scores: dict[str, float] | None = None) -> float:
    if sector_scores is not None:
        return sector_scores["quality"]
    parts: list[float] = []
    rev_g = fund.get("revenue_growth_pct")
    if rev_g is not None:
        parts.append(_clamp(42.0 + rev_g * 2.1, 10.0, 98.0))
    gm = fund.get("gross_margin_pct")
    if gm is not None:
        parts.append(_clamp(18.0 + gm * 1.0, 15.0, 96.0))
    om = fund.get("operating_margin_pct")
    if om is not None:
        parts.append(_clamp(25.0 + om * 1.8, 10.0, 98.0))
    roe = fund.get("roe_pct")
    if roe is not None:
        parts.append(_clamp(25.0 + roe * 1.4, 10.0, 96.0))
    avg_surp = earn_sum.get("avg_4q_surprise_pct")
    if avg_surp is not None:
        parts.append(_clamp(52.0 + avg_surp * 3.5 + earn_sum.get("beat_streak", 0) * 4.0, 15.0, 98.0))
    return round(float(np.mean(parts)), 1) if parts else 50.0


def _score_catalysts_flow(
    last: pd.Series,
    avg_sentiment: float,
    options_snap: dict[str, Any] | None,
    insider_sum: dict[str, Any],
) -> float:
    rsi = last.get("rsi14")
    if pd.notna(rsi):
        r = float(rsi)
        if 42 <= r <= 62:
            s_rsi = 78.0
        elif 32 <= r < 42:
            s_rsi = 72.0
        elif r < 32:
            s_rsi = 64.0
        elif 62 < r <= 72:
            s_rsi = 58.0
        else:
            s_rsi = 26.0
    else:
        s_rsi = 50.0

    s_news = _clamp(52.0 + avg_sentiment * 44.0, 10.0, 95.0)
    insider_boost = 0.0
    if insider_sum.get("cluster_buy"):
        insider_boost = 14.0
    elif insider_sum.get("buys_90d", 0) >= 1:
        insider_boost = 7.0

    if options_snap:
        opt_sent = float(options_snap.get("options_sentiment", 0.0))
        s_opt = _clamp(52.0 + opt_sent * 38.0, 12.0, 94.0)
        combined = 0.35 * s_rsi + 0.35 * s_news + 0.30 * s_opt + insider_boost
    else:
        # For NSE tickers (no Yahoo US options chain), weight entry timing + catalyst news + insider buying cleanly
        combined = 0.45 * s_rsi + 0.55 * s_news + insider_boost
    return round(_clamp(combined), 1)


def _explain_stock(
    last: pd.Series,
    fund: dict[str, Any],
    pred: dict[str, Any] | None,
    avg_sentiment: float,
    setups: list[dict[str, str]],
    earn_sum: dict[str, Any],
    insider_sum: dict[str, Any],
    options_snap: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    drivers: list[str] = []
    risks: list[str] = []

    if pred:
        tb_p = float(pred.get("prob_tb_win", 50.0))
        alpha = float(pred.get("pred_alpha_40d", pred.get("pred_alpha_20d", 0.0)))
        if alpha >= 1.5 or tb_p >= 58.0:
            drivers.append(f"V6 Quant ML signals {alpha:+.1f}% 40D alpha vs NIFTY TRI ({tb_p:.0f}% calibrated odds of +2.0×ATR before -2.0×ATR)")
        elif alpha <= -1.2 or tb_p <= 44.0:
            risks.append(f"ML model signals weak 40D payoff odds ({alpha:+.1f}% 40D alpha vs NIFTY TRI, {tb_p:.0f}% calibrated TB win)")

    if insider_sum.get("cluster_buy"):
        drivers.append(
            f"Yahoo insider proxy: {insider_sum['buys_90d']} purchase(s) (₹{insider_sum['buy_value_90d']:,.0f}) in 90D (unverified vs SEBI SAST)"
        )

    if earn_sum.get("beat_streak", 0) >= 3 and (earn_sum.get("avg_4q_surprise_pct") or 0) > 2.0:
        drivers.append(f"PEAD strength: {earn_sum['beat_streak']} consecutive EPS beats (avg +{earn_sum['avg_4q_surprise_pct']:.1f}%)")

    if options_snap and options_snap.get("pc_vol_ratio") is not None:
        pc = float(options_snap["pc_vol_ratio"])
        if pc <= 0.65:
            drivers.append(f"Bullish options flow: Put/Call volume ratio at {pc:.2f} (call-skewed)")
        elif pc >= 1.25:
            risks.append(f"Defensive options skew: Put/Call volume ratio elevated at {pc:.2f}")

    for st in setups:
        if st["tone"] == "bullish":
            drivers.append(f"{st['label']}: {st['detail']}")
        elif st["tone"] == "warning":
            risks.append(f"{st['label']}: {st['detail']}")

    close = float(last["close"])
    target = fund.get("analyst_target")
    if target and close > 0:
        up = ((target / close) - 1.0) * 100.0
        if up >= 12.0:
            drivers.append(f"Analyst mean target (₹{target:,.2f}) implies +{up:.1f}% upside")
        elif up <= -6.0:
            risks.append(f"Trading {abs(up):.1f}% above mean analyst target (₹{target:,.2f})")

    dte = earn_sum.get("days_to_earnings")
    if dte is not None and 0 <= dte <= 10:
        risks.append(f"Binary quarterly results in {dte} day(s) ({earn_sum['next_earnings_date']})")

    if avg_sentiment >= 0.18:
        drivers.append(f"Constructive recency-weighted news flow ({avg_sentiment:+.2f})")
    elif avg_sentiment <= -0.18:
        risks.append(f"Negative recent news catalysts ({avg_sentiment:+.2f})")

    vol20 = last.get("vol_20d")
    if pd.notna(vol20) and float(vol20) >= 0.45:
        risks.append(f"Elevated 20D annualized volatility ({float(vol20) * 100:.0f}%)")

    sma50 = last.get("sma50")
    if pd.notna(sma50) and close < float(sma50):
        risks.append(f"Trading below 50-day SMA (₹{float(sma50):,.2f})")

    if not drivers:
        drivers.append("Balanced technical, fundamental, and catalyst profile")
    if not risks:
        risks.append("Standard NIFTY beta & macro regime risk")

    return drivers[:5], risks[:4]


import threading
import time

_CACHE_LOCK = threading.Lock()
_BASE_SNAPSHOT_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_CACHE_TTL_SECONDS = 300.0


def invalidate_shortlist_cache() -> None:
    """Clear cached base stock snapshots after DB sync, ticker add, or star toggle."""
    with _CACHE_LOCK:
        _BASE_SNAPSHOT_CACHE.clear()


def _compute_base_snapshots(db: Database) -> list[dict[str, Any]]:
    """Compute and cache per-ticker technicals, pillars, setups, and drivers across the universe."""
    cache_key = str(db.db_path.resolve())
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _BASE_SNAPSHOT_CACHE.get(cache_key)
        if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

    tickers = db.get_tickers()
    ohlcv_map = db.get_all_ohlcv()
    ml_preds = db.get_ml_predictions()
    news_sent = db.get_all_news_sentiment()
    earnings_map = db.get_all_earnings()
    insiders_map = db.get_all_insiders()
    options_map = db.get_all_options_snapshots()
    bench_bars = ohlcv_map.get(BENCHMARK_SYMBOL)

    last_prices = {sym: float(bars[-1]["close"]) for sym, bars in ohlcv_map.items() if len(bars) >= 25}
    earn_summaries = {t["symbol"]: _summarize_earnings(earnings_map.get(t["symbol"], [])) for t in tickers}
    sector_pct_map = _build_sector_percentile_tables(tickers, last_prices, earn_summaries)

    snapshots: list[dict[str, Any]] = []
    for t in tickers:
        sym = t["symbol"]
        bars = ohlcv_map.get(sym, [])
        if len(bars) < 25:
            continue

        df = compute_indicator_frame(bars, bench_bars)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        chg_1d = close - float(prev["close"])
        chg_1d_pct = (chg_1d / max(1e-9, float(prev["close"]))) * 100.0
        chg_1m_pct = float(last["ret_21d"] * 100.0) if pd.notna(last.get("ret_21d")) else 0.0
        chg_3m_pct = float(last["ret_63d"] * 100.0) if pd.notna(last.get("ret_63d")) else 0.0
        chg_1y_pct = float(last["ret_252d"] * 100.0) if pd.notna(last.get("ret_252d")) else 0.0
        rs_63d = float(last["rs_spy_63d"] * 100.0) if pd.notna(last.get("rs_spy_63d")) else 0.0

        setups = detect_chart_setups(df)
        earn_sum = earn_summaries[sym]
        insider_sum = _summarize_insiders(insiders_map.get(sym, []))
        opt_snap = options_map.get(sym)

        if insider_sum["cluster_buy"]:
            setups.append(
                {
                    "id": "insider_buy",
                    "label": "Insider Proxy Buy",
                    "tone": "bullish",
                    "detail": f"{insider_sum['buys_90d']} Yahoo-reported purchase(s) totaling ₹{insider_sum['buy_value_90d']:,.0f} (unverified vs SEBI SAST).",
                }
            )
        if opt_snap and opt_snap.get("pc_vol_ratio") is not None and opt_snap["pc_vol_ratio"] <= 0.65:
            setups.append(
                {
                    "id": "bullish_options",
                    "label": "Call Skew Flow",
                    "tone": "bullish",
                    "detail": f"Low Put/Call volume ratio ({opt_snap['pc_vol_ratio']:.2f}) on {opt_snap['expiry']} expiry.",
                }
            )
        if earn_sum["days_to_earnings"] is not None and 0 <= earn_sum["days_to_earnings"] <= 10:
            setups.append(
                {
                    "id": "earnings_soon",
                    "label": f"Earnings {earn_sum['days_to_earnings']}D",
                    "tone": "warning",
                    "detail": f"Upcoming earnings report on {earn_sum['next_earnings_date']}.",
                }
            )

        pred = ml_preds.get(sym)
        sent_info = news_sent.get(sym, {"avg_sentiment": 0.0, "article_count": 0, "catalyst_tags": []})
        avg_sent = sent_info["avg_sentiment"]
        fund = t["fundamentals"] or {}

        p_ml = _score_ml_alpha(pred)
        p_mom = _score_momentum(last)
        p_val = _score_valuation(fund, close, sector_scores=sector_pct_map.get(sym))
        p_qual = _score_quality(fund, earn_sum, sector_scores=sector_pct_map.get(sym))
        p_sent = _score_catalysts_flow(last, avg_sent, opt_snap, insider_sum)

        drivers, risks = _explain_stock(last, fund, pred, avg_sent, setups, earn_sum, insider_sum, opt_snap)
        spark = [round(float(x), 2) for x in df["close"].tail(24).tolist()]
        target_px = fund.get("analyst_target")
        analyst_upside = round(((target_px / close) - 1.0) * 100.0, 1) if (target_px and close > 0) else None

        snapshots.append(
            {
                "symbol": sym,
                "name": t["name"],
                "sector": t["sector"],
                "exchange": t["exchange"],
                "starred": t["starred"],
                "data_source": t["data_source"],
                "updated_at": t["updated_at"],
                "price": round(close, 2),
                "chg_1d": round(chg_1d, 2),
                "chg_1d_pct": round(chg_1d_pct, 2),
                "chg_1m_pct": round(chg_1m_pct, 2),
                "chg_3m_pct": round(chg_3m_pct, 2),
                "chg_1y_pct": round(chg_1y_pct, 2),
                "rs_spy_63d_pct": round(rs_63d, 2),
                "rsi14": round(float(last["rsi14"]), 1) if pd.notna(last.get("rsi14")) else None,
                "vol_20d_pct": round(float(last["vol_20d"] * 100.0), 1) if pd.notna(last.get("vol_20d")) else None,
                "market_cap_cr": fund.get("market_cap_cr"),
                "market_cap_b": fund.get("market_cap_b"),
                "promoter_held_pct": fund.get("promoter_held_pct"),
                "forward_pe": round(float(fund["forward_pe"]), 1) if fund.get("forward_pe") else None,
                "analyst_upside_pct": analyst_upside,
                "news_sentiment": round(avg_sent, 2),
                "news_count": sent_info["article_count"],
                "catalyst_tags": sent_info.get("catalyst_tags", []),
                "earnings_summary": earn_sum,
                "insider_summary": insider_sum,
                "options_summary": {
                    "pc_vol_ratio": opt_snap.get("pc_vol_ratio"),
                    "atm_iv_pct": opt_snap.get("atm_iv_pct"),
                    "iv_hv_ratio": opt_snap.get("iv_hv_ratio"),
                    "options_sentiment": opt_snap.get("options_sentiment"),
                }
                if opt_snap
                else None,
                "pillars": {
                    "ml": p_ml,
                    "momentum": p_mom,
                    "valuation": p_val,
                    "quality": p_qual,
                    "sentiment": p_sent,
                },
                "ml": {
                    "pred_alpha_40d": pred.get("pred_alpha_40d", pred["pred_alpha_20d"]) if pred else 0.0,
                    "pred_alpha_20d": pred["pred_alpha_20d"] if pred else 0.0,
                    "prob_outperform": pred["prob_outperform"] if pred else 50.0,
                    "prob_tb_win": pred.get("prob_tb_win", 50.0) if pred else 50.0,
                    "expected_return_40d": pred.get("expected_return_40d", pred["expected_return_20d"]) if pred else 0.0,
                    "expected_return_20d": pred["expected_return_20d"] if pred else 0.0,
                    "target_price": pred.get("target_price") if pred else None,
                    "stop_price": pred.get("stop_price") if pred else None,
                    "risk_reward": pred.get("risk_reward") if pred else None,
                },
                "strategy_id": STRATEGY_ID_SHORTLIST,
                "validation_status": "unvalidated_research_composite",
                "setups": setups,
                "bullish_drivers": drivers,
                "risk_flags": risks,
                "sparkline": spark,
            }
        )

    with _CACHE_LOCK:
        _BASE_SNAPSHOT_CACHE[cache_key] = (now, snapshots)
    return snapshots


def build_shortlist(
    db: Database,
    weights: dict[str, float] | None = None,
    sector: str = "ALL",
    setup_filter: str = "ALL",
    include_benchmark: bool = False,
) -> list[dict[str, Any]]:
    """Compute ranked opportunity shortlist across the full universe using cached base snapshots."""
    w = {"ml": 0.30, "momentum": 0.25, "valuation": 0.15, "quality": 0.15, "sentiment": 0.15}
    if weights:
        w.update({k: max(0.0, float(v)) for k, v in weights.items() if k in w})
    w_sum = sum(w.values())
    if w_sum <= 0:
        w = {"ml": 0.30, "momentum": 0.25, "valuation": 0.15, "quality": 0.15, "sentiment": 0.15}
        w_sum = 1.0
    w = {k: v / w_sum for k, v in w.items()}

    aux_excluded = {BENCHMARK_SYMBOL, BENCHMARK_TRI_SYMBOL, GOLDBEES_SYMBOL}
    base_snapshots = _compute_base_snapshots(db)
    results: list[dict[str, Any]] = []
    for snap in base_snapshots:
        if not include_benchmark and snap["symbol"] in aux_excluded:
            continue
        if sector != "ALL" and snap["sector"] != sector:
            continue
        if setup_filter != "ALL" and not any(st["id"] == setup_filter for st in snap["setups"]):
            continue

        p = snap["pillars"]
        composite = round(
            w["ml"] * p["ml"]
            + w["momentum"] * p["momentum"]
            + w["valuation"] * p["valuation"]
            + w["quality"] * p["quality"]
            + w["sentiment"] * p["sentiment"],
            1,
        )
        if composite >= 70:
            conviction = "Strong Candidate"
        elif composite >= 59:
            conviction = "Accumulate / Watch"
        elif composite >= 46:
            conviction = "Neutral"
        else:
            conviction = "Caution"

        item = dict(snap)
        item["composite_score"] = composite
        item["conviction"] = conviction
        results.append(item)

    results.sort(key=lambda r: r["composite_score"], reverse=True)
    for idx, item in enumerate(results, start=1):
        item["rank"] = idx
    return results
