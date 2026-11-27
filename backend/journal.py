"""Automated Prediction Journal & Live Calibration Engine (Strategy-Separated Accountability Loop).

Hardened accountability:
  - Eliminates synthetic historical composite score proxies (`62 + 12*alpha`).
  - Separates `v6_ml_oos` (strictly non-overlapping 40D OOS cohorts graded against NIFTY TRI)
    from `live_shortlist` (real live forward snapshots logged day-by-day).
  - Grades entries at +5D, +20D, +40D, and +60D against actual OHLCV & TRI benchmark.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_SYMBOL,
    BENCHMARK_TRI_SYMBOL,
    ML_HORIZON_DAYS,
    STRATEGY_ID_ML,
    STRATEGY_ID_SHORTLIST,
    TB_DOWN_ATR_MULT,
    TB_UP_ATR_MULT,
)
from .database import Database
from .ml_predictor import compute_current_macro_regime
from .shortlister import build_shortlist

_JOURNAL_LOCK = threading.Lock()
_JOURNAL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def invalidate_journal_cache() -> None:
    with _JOURNAL_LOCK:
        _JOURNAL_CACHE.clear()


def log_and_grade_journal(db: Database, top_n: int = 5) -> dict[str, Any]:
    """Log live shortlist picks and grade non-overlapping 40D OOS cohorts without synthetic proxies."""
    now_iso = datetime.now(timezone.utc).isoformat()
    shortlist = build_shortlist(db)
    regime = compute_current_macro_regime(db)["regime"]
    ohlcv_map = db.get_all_ohlcv()

    use_tri = len(ohlcv_map.get(BENCHMARK_TRI_SYMBOL, [])) >= 120
    bench_sym = BENCHMARK_TRI_SYMBOL if use_tri else BENCHMARK_SYMBOL
    bench_bars = ohlcv_map.get(bench_sym) or ohlcv_map.get(BENCHMARK_SYMBOL, [])

    price_series: dict[str, tuple[list[str], np.ndarray]] = {}
    for sym, bars in ohlcv_map.items():
        dates = [b["date"] for b in bars]
        closes = np.array([b["close"] for b in bars], dtype=float)
        price_series[sym] = (dates, closes)

    def _forward_ret(sym: str, start_date: str, offset: int) -> float | None:
        item = price_series.get(sym)
        if not item:
            return None
        dates, closes = item
        try:
            idx = dates.index(start_date)
        except ValueError:
            return None
        if idx + offset >= len(closes):
            return None
        return round(((float(closes[idx + offset]) / float(closes[idx])) - 1.0) * 100.0, 2)

    entries: list[dict[str, Any]] = []

    # 1. Today's LIVE top-N visual composite shortlist snapshot
    if shortlist and bench_bars:
        as_of = bench_bars[-1]["date"]
        for pick in shortlist[:top_n]:
            alpha_40 = pick["ml"].get("pred_alpha_40d", pick["ml"].get("pred_alpha_20d", 0.0))
            entries.append(
                {
                    "id": f"live_{as_of}_{pick['symbol']}",
                    "logged_date": as_of,
                    "symbol": pick["symbol"],
                    "entry_price": pick["price"],
                    "composite_score": pick["composite_score"],
                    "ml_tb_prob": pick["ml"]["prob_tb_win"],
                    "ml_alpha_20d": alpha_40,
                    "target_price": pick["ml"]["target_price"],
                    "stop_price": pick["ml"]["stop_price"],
                    "conviction": pick["conviction"],
                    "regime": regime,
                    "is_live": True,
                    "strategy_id": STRATEGY_ID_SHORTLIST,
                    "ret_5d": None,
                    "ret_20d": None,
                    "ret_40d": None,
                    "ret_60d": None,
                    "spy_ret_20d": None,
                    "bench_ret_40d": None,
                    "outcome": "Open (Live)",
                    "graded_at": None,
                }
            )

    # 2. Historical V6-ML OOS cohorts (STRICTLY NON-OVERLAPPING 40D steps; NO synthetic composite score!)
    oos = db.get_oos_history()
    if oos:
        oos_df = pd.DataFrame(oos)
        all_dates = sorted(oos_df["date"].unique())
        for dt in all_dates[::ML_HORIZON_DAYS]:
            day_df = oos_df[oos_df["date"] == dt].copy()
            if day_df.empty:
                continue
            rank_col = "pred_score" if "pred_score" in day_df.columns else "pred_alpha"
            day_df = day_df.sort_values(rank_col, ascending=False).head(top_n)
            for _, row in day_df.iterrows():
                sym = str(row["symbol"])
                ps = price_series.get(sym)
                if not ps or dt not in ps[0]:
                    continue
                idx = ps[0].index(dt)
                entry_px = float(ps[1][idx])
                bars = ohlcv_map[sym]
                win = bars[max(0, idx - 14) : idx + 1]
                atr = float(np.mean([b["high"] - b["low"] for b in win])) if win else entry_px * 0.02
                target_px = round(entry_px + TB_UP_ATR_MULT * atr, 2)
                stop_px = round(max(0.5, entry_px - TB_DOWN_ATR_MULT * atr), 2)

                r5 = _forward_ret(sym, dt, 5)
                r20 = _forward_ret(sym, dt, 20)
                r40 = _forward_ret(sym, dt, ML_HORIZON_DAYS)
                r60 = _forward_ret(sym, dt, 60)
                b20 = _forward_ret(bench_sym, dt, 20)
                b40 = _forward_ret(bench_sym, dt, ML_HORIZON_DAYS)

                tb_out = row.get("tb_outcome")
                if tb_out == 1:
                    outcome = "Hit Target / Win"
                elif tb_out == 0:
                    outcome = "Stopped / Miss"
                elif r40 is not None and b40 is not None:
                    outcome = "Beat NIFTY TRI" if r40 > b40 else "Lagged NIFTY TRI"
                else:
                    outcome = "Open"

                # Record actual V6-ML score (not a fabricated composite proxy!)
                raw_score = round(float(row[rank_col]), 3)
                alpha_40 = round(float(row.get("pred_alpha_40d", raw_score)), 2)
                entries.append(
                    {
                        "id": f"oos40_{dt}_{sym}",
                        "logged_date": dt,
                        "symbol": sym,
                        "entry_price": round(entry_px, 2),
                        "composite_score": raw_score,
                        "ml_tb_prob": round(float(row["prob_tb_win"]), 1),
                        "ml_alpha_20d": alpha_40,
                        "target_price": target_px,
                        "stop_price": stop_px,
                        "conviction": "V6-ML Top-5 (OOS)",
                        "regime": "OOS Walk-Forward (40D)",
                        "is_live": False,
                        "strategy_id": STRATEGY_ID_ML,
                        "ret_5d": r5,
                        "ret_20d": r20,
                        "ret_40d": r40,
                        "ret_60d": r60,
                        "spy_ret_20d": b20,
                        "bench_ret_40d": b40,
                        "outcome": outcome,
                        "graded_at": now_iso if r40 is not None else None,
                    }
                )

    db.upsert_journal_entries(entries)
    invalidate_journal_cache()
    return get_journal_report(db)


def get_journal_report(db: Database) -> dict[str, Any]:
    """Return strategy-separated journal entries plus calibrated 40D probability bucket audit."""
    cache_key = str(db.db_path.resolve())
    now = time.monotonic()
    with _JOURNAL_LOCK:
        cached = _JOURNAL_CACHE.get(cache_key)
        if cached and (now - cached[0]) < 300.0:
            return cached[1]

    all_entries = db.get_journal_entries(limit=300)
    graded_40 = [e for e in all_entries if e.get("ret_40d") is not None]
    graded = graded_40 if graded_40 else [e for e in all_entries if e.get("ret_20d") is not None]

    oos = db.get_oos_history()
    buckets_spec = [
        ("< 45% TB Prob", 0.0, 45.0),
        ("45% – 52% TB Prob", 45.0, 52.0),
        ("52% – 58% TB Prob", 52.0, 58.0),
        ("> 58% TB Prob", 58.0, 100.1),
    ]
    calibration = []
    if oos:
        odf = pd.DataFrame(oos)
        fwd_col = "actual_fwd_40d" if "actual_fwd_40d" in odf.columns else "actual_fwd_20d"
        bn_col = "bench_fwd_40d" if "bench_fwd_40d" in odf.columns else "bench_fwd_20d"
        for lbl, lo, hi in buckets_spec:
            sub = odf[(odf["prob_tb_win"] >= lo) & (odf["prob_tb_win"] < hi)]
            if sub.empty:
                continue
            avg_f = round(float(sub[fwd_col].mean()), 2)
            avg_ex = round(float((sub[fwd_col] - sub[bn_col]).mean()), 2)
            calibration.append(
                {
                    "bucket": lbl,
                    "count": int(len(sub)),
                    "actual_tb_win_pct": round(float(sub["tb_outcome"].mean()) * 100.0, 1),
                    "avg_40d_ret_pct": avg_f,
                    "avg_20d_ret_pct": avg_f,  # deprecated alias
                    "avg_excess_vs_spy_pct": avg_ex,
                }
            )

    ret_key = "ret_40d" if graded_40 else "ret_20d"
    bench_key = "bench_ret_40d" if graded_40 else "spy_ret_20d"
    summary = {
        "horizon_days": ML_HORIZON_DAYS if graded_40 else 20,
        "total_logged": len(all_entries),
        "live_shortlist_count": sum(1 for e in all_entries if e.get("is_live")),
        "v6_ml_oos_count": sum(1 for e in all_entries if not e.get("is_live")),
        "graded_count": len(graded),
        "avg_40d_return_pct": round(float(np.mean([e[ret_key] for e in graded])), 2) if graded else 0.0,
        "avg_20d_return_pct": round(float(np.mean([e[ret_key] for e in graded])), 2) if graded else 0.0,
        "avg_bench_40d_pct": round(float(np.mean([e[bench_key] for e in graded if e.get(bench_key) is not None])), 2)
        if any(e.get(bench_key) is not None for e in graded)
        else 0.0,
        "avg_spy_20d_pct": round(float(np.mean([e[bench_key] for e in graded if e.get(bench_key) is not None])), 2)
        if any(e.get(bench_key) is not None for e in graded)
        else 0.0,
        "win_rate_positive_pct": round(float(np.mean([1 if e[ret_key] > 0 else 0 for e in graded])) * 100.0, 1) if graded else 0.0,
        "beat_spy_rate_pct": round(
            float(np.mean([1 if e[ret_key] > (e.get(bench_key) or 0) else 0 for e in graded])) * 100.0, 1
        )
        if graded
        else 0.0,
    }
    res = {"summary": summary, "calibration": calibration, "entries": all_entries[:120]}
    with _JOURNAL_LOCK:
        _JOURNAL_CACHE[cache_key] = (now, res)
    return res
