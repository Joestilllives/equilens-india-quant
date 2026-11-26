"""Minimal Data Integrity & Live Freshness Guard for Safe Paper/Live Execution.

Prevents silent forward-fill distortion across suspended/pre-IPO/delisted tickers
and detects stale market data before committing rebalance orders.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_SYMBOL,
    DATA_FRESHNESS_MIN_COVERAGE,
    FX_MACRO_SYMBOL,
    GOLD_MACRO_SYMBOL,
    MAX_STALE_TRADING_DAYS,
    VOL_INDEX_SYMBOL,
)
from .database import Database
from .nse_calendar import is_nse_trading_day, nse_trading_days_between, previous_nse_trading_day

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))
_FRESHNESS_LOCK = threading.Lock()
_FRESHNESS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_FRESHNESS_TTL_SEC = 60.0


def invalidate_freshness_cache() -> None:
    with _FRESHNESS_LOCK:
        _FRESHNESS_CACHE.clear()


def build_clean_price_matrix(
    ohlcv_map: dict[str, list[dict[str, Any]]],
    max_ffill_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aligned close, volume, and tradable-bar masks with bounded forward-fill."""
    raw_closes: dict[str, pd.Series] = {}
    raw_vols: dict[str, pd.Series] = {}
    for sym, bars in ohlcv_map.items():
        if not bars:
            continue
        c_dict: dict[str, float] = {}
        v_dict: dict[str, float] = {}
        for b in bars:
            dt = str(b["date"])[:10]
            px = float(b["close"])
            if np.isfinite(px) and px > 0:
                c_dict[dt] = px
                v_dict[dt] = float(b.get("volume", 0) or 0)
        if c_dict:
            raw_closes[sym] = pd.Series(c_dict, dtype=float)
            raw_vols[sym] = pd.Series(v_dict, dtype=float)

    if not raw_closes:
        empty = pd.DataFrame()
        return empty, empty, empty

    raw_prices_df = pd.DataFrame(raw_closes).sort_index()
    volumes_df = pd.DataFrame(raw_vols).reindex(raw_prices_df.index)
    prices_df = raw_prices_df.ffill(limit=max_ffill_days) if max_ffill_days > 0 else raw_prices_df.copy()

    observed = raw_prices_df.notna()
    tradable_cols: dict[str, pd.Series] = {}
    for sym in prices_df.columns:
        is_idx = sym.startswith("^")
        recent_obs = observed[sym].rolling(max_ffill_days + 1, min_periods=1).max().astype(bool) if max_ffill_days > 0 else observed[sym]
        has_px = prices_df[sym].notna() & (prices_df[sym] > 0)
        if is_idx:
            tradable_cols[sym] = has_px & recent_obs
        else:
            vol_ok = volumes_df[sym].ffill(limit=max_ffill_days).fillna(0.0) > 0
            tradable_cols[sym] = has_px & recent_obs & vol_ok

    return prices_df, volumes_df, pd.DataFrame(tradable_cols, index=prices_df.index)


def expected_latest_nse_close_date(now_ist: datetime | None = None) -> date:
    """Return the latest NSE trading day whose EOD close should be available."""
    now = now_ist or datetime.now(IST_OFFSET)
    d = now.date()
    if is_nse_trading_day(d) and now.hour >= 18:
        return d
    return previous_nse_trading_day(d, inclusive=False)


def evaluate_data_freshness(
    db: Database,
    now_ist: datetime | None = None,
    max_stale_days: int = MAX_STALE_TRADING_DAYS,
    min_coverage: float = DATA_FRESHNESS_MIN_COVERAGE,
) -> dict[str, Any]:
    """Evaluate benchmark, universe, macro, and delivery feed freshness via O(symbols) SQL."""
    cache_key = f"{db.db_path.resolve()}:{max_stale_days}:{min_coverage}"
    if now_ist is None:
        now_m = time.monotonic()
        with _FRESHNESS_LOCK:
            cached = _FRESHNESS_CACHE.get(cache_key)
            if cached and (now_m - cached[0]) < _FRESHNESS_TTL_SEC:
                return cached[1]

    ohlcv_latest = db.get_ohlcv_latest_dates()
    macro_latest = db.get_macro_latest_dates()
    expected_dt = expected_latest_nse_close_date(now_ist)

    warnings: list[str] = []
    bench_latest = ohlcv_latest.get(BENCHMARK_SYMBOL)
    bench_lag_days: int | None = None
    if bench_latest:
        bench_lag_days = max(0, nse_trading_days_between(bench_latest, expected_dt))
        if bench_lag_days > max_stale_days:
            warnings.append(
                f"Benchmark {BENCHMARK_SYMBOL} is {bench_lag_days} NSE trading day(s) behind expected close {expected_dt.isoformat()} (latest: {bench_latest})."
            )
    else:
        warnings.append(f"Benchmark {BENCHMARK_SYMBOL} OHLCV history is missing.")

    stock_syms = [s for s in ohlcv_latest if s != BENCHMARK_SYMBOL and not s.startswith("^")]
    stale_symbols: list[str] = []
    fresh_count = 0
    if bench_latest and stock_syms:
        for sym in stock_syms:
            sym_last = ohlcv_latest.get(sym)
            if sym_last and nse_trading_days_between(sym_last, bench_latest) <= 1:
                fresh_count += 1
            else:
                stale_symbols.append(sym)
        coverage_ratio = fresh_count / max(1, len(stock_syms))
    else:
        coverage_ratio = 0.0

    if stock_syms and coverage_ratio < min_coverage:
        warnings.append(
            f"Universe coverage is {coverage_ratio * 100:.1f}% ({fresh_count}/{len(stock_syms)} fresh stocks), below {min_coverage * 100:.0f}% threshold."
        )

    for m_sym in (VOL_INDEX_SYMBOL, FX_MACRO_SYMBOL, GOLD_MACRO_SYMBOL):
        m_last = macro_latest.get(m_sym)
        if not m_last:
            warnings.append(f"Macro series {m_sym} is missing.")
        elif bench_latest and nse_trading_days_between(m_last, bench_latest) > max_stale_days + 2:
            warnings.append(f"Macro series {m_sym} is stale (latest: {m_last} vs {bench_latest}).")

    critical_stale = (
        bench_latest is None
        or (bench_lag_days is not None and bench_lag_days > max_stale_days)
        or (len(stock_syms) > 0 and coverage_ratio < min_coverage)
    )
    status = "stale" if critical_stale else ("degraded" if warnings else "healthy")
    res = {
        "status": status,
        "block_execution": critical_stale,
        "expected_close_date": expected_dt.isoformat(),
        "benchmark_latest_date": bench_latest,
        "benchmark_lag_trading_days": bench_lag_days,
        "universe_count": len(stock_syms),
        "fresh_symbol_count": fresh_count,
        "coverage_pct": round(coverage_ratio * 100.0, 1),
        "stale_symbols": sorted(stale_symbols)[:25],
        "stale_symbols_total": len(stale_symbols),
        "warnings": warnings,
    }
    if now_ist is None:
        with _FRESHNESS_LOCK:
            _FRESHNESS_CACHE[cache_key] = (time.monotonic(), res)
    return res
