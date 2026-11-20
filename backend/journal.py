"""Automated Prediction Journal & Live Calibration Engine (Phase 2 Accountability Loop)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .config import BENCHMARK_SYMBOL, TB_DOWN_ATR_MULT, TB_UP_ATR_MULT
from .database import Database
from .ml_predictor import compute_current_macro_regime
from .shortlister import build_shortlist


def log_and_grade_journal(db: Database, top_n: int = 5) -> dict[str, Any]:
    """
    1. Logs today's live Top-N shortlist picks into prediction_journal (idempotent per date+symbol).
    2. Seeds historical strictly out-of-sample walk-forward Top-N picks (sampled monthly) so calibration works immediately.
    3. Grades all entries at +5D, +20D, +60D against actual OHLCV prices & SPY.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    shortlist = build_shortlist(db)
    regime = compute_current_macro_regime(db)["regime"]
    ohlcv_map = db.get_all_ohlcv()
    spy_bars = ohlcv_map.get(BENCHMARK_SYMBOL, [])

    # Build fast price lookup per symbol
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

    # 1. Today's LIVE top-N snapshot
    if shortlist and spy_bars:
        as_of = spy_bars[-1]["date"]
        for pick in shortlist[:top_n]:
            entries.append(
                {
                    "id": f"live_{as_of}_{pick['symbol']}",
                    "logged_date": as_of,
                    "symbol": pick["symbol"],
                    "entry_price": pick["price"],
                    "composite_score": pick["composite_score"],
                    "ml_tb_prob": pick["ml"]["prob_tb_win"],
                    "ml_alpha_20d": pick["ml"]["pred_alpha_20d"],
                    "target_price": pick["ml"]["target_price"],
                    "stop_price": pick["ml"]["stop_price"],
                    "conviction": pick["conviction"],
                    "regime": regime,
                    "is_live": True,
                    "ret_5d": None,
                    "ret_20d": None,
                    "ret_60d": None,
                    "spy_ret_20d": None,
                    "outcome": "Open (Live)",
                    "graded_at": None,
                }
            )

    # 2. Historical out-of-sample walk-forward picks (monthly rebalance snapshots)
    oos = db.get_oos_history()
    if oos:
        oos_df = pd.DataFrame(oos)
        all_dates = sorted(oos_df["date"].unique())
        # Sample every 20 trading days (non-overlapping monthly cohorts)
        for dt in all_dates[::20]:
            day_df = oos_df[oos_df["date"] == dt].copy()
            if day_df.empty:
                continue
            day_df["score_proxy"] = np.clip(
                62.0 + day_df["pred_alpha"] * 12.0,
                25.0,
                95.0,
            )
            day_df = day_df.sort_values("pred_alpha", ascending=False).head(top_n)
            for _, row in day_df.iterrows():
                sym = str(row["symbol"])
                ps = price_series.get(sym)
                if not ps or dt not in ps[0]:
                    continue
                idx = ps[0].index(dt)
                entry_px = float(ps[1][idx])
                # Rough ATR estimate from trailing 14-day range for target/stop
                bars = ohlcv_map[sym]
                win = bars[max(0, idx - 14) : idx + 1]
                atr = float(np.mean([b["high"] - b["low"] for b in win])) if win else entry_px * 0.02
                target_px = round(entry_px + TB_UP_ATR_MULT * atr, 2)
                stop_px = round(max(0.5, entry_px - TB_DOWN_ATR_MULT * atr), 2)

                r5 = _forward_ret(sym, dt, 5)
                r20 = _forward_ret(sym, dt, 20)
                r60 = _forward_ret(sym, dt, 60)
                spy20 = _forward_ret(BENCHMARK_SYMBOL, dt, 20)

                tb_out = row.get("tb_outcome")
                if tb_out == 1:
                    outcome = "Hit Target / Win"
                elif tb_out == 0:
                    outcome = "Stopped / Miss"
                elif r20 is not None and spy20 is not None:
                    outcome = "Beat NIFTY" if r20 > spy20 else "Lagged NIFTY"
                else:
                    outcome = "Open"

                sc = round(float(row["score_proxy"]), 1)
                conv = "Strong Candidate" if sc >= 70 else "Accumulate / Watch" if sc >= 58 else "Neutral"
                entries.append(
                    {
                        "id": f"oos_{dt}_{sym}",
                        "logged_date": dt,
                        "symbol": sym,
                        "entry_price": round(entry_px, 2),
                        "composite_score": sc,
                        "ml_tb_prob": round(float(row["prob_tb_win"]), 1),
                        "ml_alpha_20d": round(float(row["pred_alpha"]), 2),
                        "target_price": target_px,
                        "stop_price": stop_px,
                        "conviction": conv,
                        "regime": "OOS Walk-Forward",
                        "is_live": False,
                        "ret_5d": r5,
                        "ret_20d": r20,
                        "ret_60d": r60,
                        "spy_ret_20d": spy20,
                        "outcome": outcome,
                        "graded_at": now_iso if r20 is not None else None,
                    }
                )

    db.upsert_journal_entries(entries)
    invalidate_journal_cache()
    return get_journal_report(db)


import threading
import time

_JOURNAL_LOCK = threading.Lock()
_JOURNAL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def invalidate_journal_cache() -> None:
    with _JOURNAL_LOCK:
        _JOURNAL_CACHE.clear()


def get_journal_report(db: Database) -> dict[str, Any]:
    """Return journal entries plus score-bucket calibration statistics (cached)."""
    cache_key = str(db.db_path.resolve())
    now = time.monotonic()
    with _JOURNAL_LOCK:
        cached = _JOURNAL_CACHE.get(cache_key)
        if cached and (now - cached[0]) < 300.0:
            return cached[1]

    all_entries = db.get_journal_entries(limit=300)
    graded = [e for e in all_entries if e.get("ret_20d") is not None]

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
        for lbl, lo, hi in buckets_spec:
            sub = odf[(odf["prob_tb_win"] >= lo) & (odf["prob_tb_win"] < hi)]
            if sub.empty:
                continue
            calibration.append(
                {
                    "bucket": lbl,
                    "count": int(len(sub)),
                    "actual_tb_win_pct": round(float(sub["tb_outcome"].mean()) * 100.0, 1),
                    "avg_20d_ret_pct": round(float(sub["actual_fwd_20d"].mean()), 2),
                    "avg_excess_vs_spy_pct": round(float((sub["actual_fwd_20d"] - sub["bench_fwd_20d"]).mean()), 2),
                }
            )

    summary = {
        "total_logged": len(all_entries),
        "graded_count": len(graded),
        "avg_20d_return_pct": round(float(np.mean([e["ret_20d"] for e in graded])), 2) if graded else 0.0,
        "avg_spy_20d_pct": round(float(np.mean([e["spy_ret_20d"] for e in graded if e.get("spy_ret_20d") is not None])), 2) if graded else 0.0,
        "win_rate_positive_pct": round(float(np.mean([1 if e["ret_20d"] > 0 else 0 for e in graded])) * 100.0, 1) if graded else 0.0,
        "beat_spy_rate_pct": round(
            float(np.mean([1 if e["ret_20d"] > (e.get("spy_ret_20d") or 0) else 0 for e in graded])) * 100.0, 1
        )
        if graded
        else 0.0,
    }
    res = {"summary": summary, "calibration": calibration, "entries": all_entries[:120]}
    with _JOURNAL_LOCK:
        _JOURNAL_CACHE[cache_key] = (now, res)
    return res
