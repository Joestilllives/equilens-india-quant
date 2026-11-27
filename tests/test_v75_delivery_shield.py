"""Regression tests for EquiLens India V7.5-Core (20-Day Trailing NSE Delivery <35% Shield)."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import DB_PATH
from backend.database import Database
from backend.money_signals import get_money_signals_summary
from backend.portfolio_backtester import (
    _build_causal_delivery_20d_lookup,
    _prior_delivery_pct,
    build_backtest_market_context,
    run_oos_portfolio_backtest,
)


def test_causal_delivery_20d_lookup_strict_t_minus_1_and_smoothing() -> None:
    """20D shield must ignore same-day/future values and smooth out single-day noisy blips."""
    raw_hist = {
        "SMOOTH.NS": {f"2024-01-{d:02d}": (20.0 if d == 20 else 50.0) for d in range(1, 22)},
        "CHURN.NS": {f"2024-01-{d:02d}": 24.0 for d in range(1, 22)},
    }
    lookup = _build_causal_delivery_20d_lookup(raw_hist, window=20)

    # Before first date -> None (no lookahead)
    assert _prior_delivery_pct(lookup, "SMOOTH.NS", "2024-01-01") is None
    assert _prior_delivery_pct(raw_hist, "SMOOTH.NS", "2024-01-01") is None

    # On 2024-01-20, same-day 20.0% MUST NOT leak into T-1 lookup (all 19 prior days are 50.0%)
    val_on_20 = _prior_delivery_pct(lookup, "SMOOTH.NS", "2024-01-20")
    val_on_20_raw = _prior_delivery_pct(raw_hist, "SMOOTH.NS", "2024-01-20")
    assert val_on_20 is not None and abs(val_on_20 - 50.0) < 1e-9
    assert val_on_20_raw is not None and abs(val_on_20_raw - 50.0) < 1e-9

    # On 2024-01-21, the single-day 20% dip on Jan 20 is smoothed over 20 days -> (19*50 + 20)/20 = 48.5% (>= 35%!)
    val_on_21 = _prior_delivery_pct(lookup, "SMOOTH.NS", "2024-01-21")
    assert val_on_21 is not None and abs(val_on_21 - 48.5) < 1e-9
    assert val_on_21 >= 35.0

    # Sustained low-delivery stock stays strictly below 35%
    churn_val = _prior_delivery_pct(lookup, "CHURN.NS", "2024-01-21")
    assert churn_val is not None and abs(churn_val - 24.0) < 1e-9
    assert churn_val < 35.0


def test_money_signals_20d_smoothing_vs_single_day_blip(tmp_path: Path) -> None:
    """Money signals must compute 20D trailing delivery % and avoid flagging a 1-day dip on a high-delivery stock."""
    db = Database(tmp_path / "deliv_unit.db")
    db.upsert_ticker("STABLE.NS", "Stable Institutional Co", "IT", "Software", 1e11, "", {}, "test", "2026-10-20")
    db.upsert_ticker("SPECUL.NS", "Speculative Churn Co", "Metals", "Trading", 5e10, "", {}, "test", "2026-10-20")

    rows = []
    for day in range(1, 21):
        dt = f"2026-10-{day:02d}"
        rows.append(
            {
                "symbol": "STABLE.NS",
                "date": dt,
                "close_price": 1000.0,
                "traded_qty": 100000,
                "deliv_qty": 25000 if day == 20 else 55000,
                "deliv_pct": 25.0 if day == 20 else 55.0,
                "turnover_lacs": 1000.0,
                "no_of_trades": 5000,
                "avg_trade_val_inr": 20000.0,
            }
        )
        rows.append(
            {
                "symbol": "SPECUL.NS",
                "date": dt,
                "close_price": 500.0,
                "traded_qty": 200000,
                "deliv_qty": 80000 if day == 20 else 44000,
                "deliv_pct": 40.0 if day == 20 else 22.0,
                "turnover_lacs": 1000.0,
                "no_of_trades": 8000,
                "avg_trade_val_inr": 12500.0,
            }
        )
    db.upsert_nse_delivery(rows)

    snap = get_money_signals_summary(db)
    s_map = snap["stock_delivery_map"]

    # STABLE.NS dipped to 25% on day 20, but its 20D mean is 53.5% -> NOT blocked/low-delivery
    assert s_map["STABLE.NS"]["deliv_pct"] == 25.0
    assert s_map["STABLE.NS"]["deliv_20d_pct"] == 53.5
    assert s_map["STABLE.NS"]["tone"] != "warning"

    # SPECUL.NS popped to 40% on day 20, but its 20D mean is 22.9% -> flagged Low 20D Delivery!
    assert s_map["SPECUL.NS"]["deliv_pct"] == 40.0
    assert s_map["SPECUL.NS"]["deliv_20d_pct"] == 22.9
    assert s_map["SPECUL.NS"]["tone"] == "warning"
    assert "20D" in s_map["SPECUL.NS"]["tag"]

    # Verify T2T (>=98%) and low deliverable turnover (<Rs 2 Cr/day) guards + SEBI telemetry
    db.upsert_ticker("T2TTRAP.NS", "Surveillance T2T Co", "Infra", "Construction", 1e10, "", {}, "test", "2026-10-20")
    db.upsert_ticker("ILLIQ.NS", "Tiny Deliverable Value Co", "Textiles", "Apparel", 5e9, "", {}, "test", "2026-10-20")
    extra_rows = []
    for day in range(1, 21):
        dt = f"2026-10-{day:02d}"
        extra_rows.append(
            {
                "symbol": "T2TTRAP.NS",
                "date": dt,
                "close_price": 100.0,
                "traded_qty": 500000,
                "deliv_qty": 500000,
                "deliv_pct": 100.0,
                "turnover_lacs": 500.0,
                "no_of_trades": 2000,
                "avg_trade_val_inr": 25000.0,
            }
        )
        extra_rows.append(
            {
                "symbol": "ILLIQ.NS",
                "date": dt,
                "close_price": 200.0,
                "traded_qty": 5000,
                "deliv_qty": 3000,
                "deliv_pct": 60.0,
                "turnover_lacs": 100.0,  # 100 lacs * 60% = Rs 0.60 Cr deliverable (< Rs 2 Cr floor!)
                "no_of_trades": 300,
                "avg_trade_val_inr": 33333.0,
            }
        )
    db.upsert_nse_delivery(extra_rows)
    snap2 = get_money_signals_summary(db)
    s2 = snap2["stock_delivery_map"]
    assert s2["T2TTRAP.NS"]["t2t_suspected"] is True
    assert s2["T2TTRAP.NS"]["tone"] == "warning"
    assert "T2T" in s2["T2TTRAP.NS"]["tag"]
    assert s2["ILLIQ.NS"]["low_deliv_turnover"] is True
    assert s2["ILLIQ.NS"]["deliv_turnover_20d_cr"] < 2.0
    assert s2["ILLIQ.NS"]["tone"] == "warning"

    tel = snap2["delivery_regime_telemetry"]
    assert tel["universe_count"] == 4
    assert tel["t2t_flagged_count"] == 1
    assert tel["low_deliv_turnover_count"] == 1


def test_v75_core_live_db_parity_speed_and_outperformance() -> None:
    """Verify 100% daily 6Y archive, Budget 2024 reserve tax, LTCG cliff guard, and fast live APIs."""
    db = Database(DB_PATH)
    hist = db.get_nse_delivery_history()
    assert len(hist) >= 120
    all_dates = {d for m in hist.values() for d in m.keys()}
    assert len(all_dates) >= 1600, f"Expected full daily 6Y NSE delivery archive (>=1600 dates), got {len(all_dates)}"

    # Verify V7.5-Core default shield + LTCG cliff guard beats V7 Base (unshielded) cleanly
    ctx = build_backtest_market_context(db)
    res_v75 = run_oos_portfolio_backtest(db, enforce_smart_money_shield=True, market_ctx=ctx)
    res_v75_arb = run_oos_portfolio_backtest(db, enforce_smart_money_shield=True, reserve_cash_vehicle="arbitrage", market_ctx=ctx)
    res_v7 = run_oos_portfolio_backtest(db, enforce_smart_money_shield=False, ltcg_cliff_guard=False, market_ctx=ctx)
    assert res_v75["strategy_id"] == "v75_core_delivery_shield"
    assert res_v75["config"]["enforce_smart_money_shield"] is True
    assert res_v75["config"]["delivery_shield_window_days"] == 20
    assert res_v75["config"]["delivery_t2t_cap_pct"] == 98.0
    assert res_v75["config"]["min_deliv_turnover_cr"] == 2.0
    assert res_v75["config"]["ltcg_cliff_guard"] is True
    assert res_v75["portfolio_metrics"]["cagr_pct"] >= res_v7["portfolio_metrics"]["cagr_pct"] + 2.0
    assert res_v75["portfolio_metrics"]["sharpe"] > res_v7["portfolio_metrics"]["sharpe"]
    assert res_v75["portfolio_metrics"]["alpha_4f_pct"] >= res_v7["portfolio_metrics"]["alpha_4f_pct"] + 2.0

    # Budget 2024 reserve tax honesty: Arbitrage Fund (20.8% STCG) strictly beats LIQUIDCASE (31.2% slab)
    tcs = res_v75["tax_and_cost_summary"]
    assert tcs["reserve_tax_liquidcase_inr"] > tcs["reserve_tax_arbitrage_inr"] > 0.0
    assert res_v75_arb["portfolio_metrics"]["after_tax_cagr_pct"] > res_v75["portfolio_metrics"]["after_tax_cagr_pct"]

    # Verify /api/money-signals and /api/execution/ticket stay fast even with 220k delivery rows
    from backend.main import app

    client = TestClient(app)
    t0 = time.monotonic()
    r_ms = client.get("/api/money-signals")
    t_ms = time.monotonic() - t0
    assert r_ms.status_code == 200
    assert t_ms < 2.0, f"/api/money-signals took {t_ms:.2f}s"
    ms_json = r_ms.json()
    assert "delivery_regime_telemetry" in ms_json
    assert ms_json["delivery_regime_telemetry"]["universe_count"] >= 100
    sample = next(iter(ms_json["stock_delivery_map"].values()))
    assert "deliv_20d_pct" in sample and "deliv_turnover_20d_cr" in sample

    t1 = time.monotonic()
    r_tk = client.get("/api/execution/ticket?capital_inr=1000000&top_k=10&risk_profile=balanced&enforce_smart_money_shield=true")
    t_tk = time.monotonic() - t1
    assert r_tk.status_code == 200
    assert t_tk < 20.0, f"/api/execution/ticket took {t_tk:.2f}s"
    tk = r_tk.json()
    assert tk["config"]["delivery_shield_window_days"] == 20
    assert tk["config"]["delivery_shield_threshold_pct"] == 35.0
    assert tk["config"]["delivery_t2t_cap_pct"] == 98.0
    assert tk["config"]["min_deliv_turnover_cr"] == 2.0
    assert tk["config"]["ltcg_cliff_guard"] is True
    assert "delivery_regime_telemetry" in tk
    assert all(o.get("deliv_20d_pct") is None or (35.0 <= o["deliv_20d_pct"] < 98.0) for o in tk["orders"])
