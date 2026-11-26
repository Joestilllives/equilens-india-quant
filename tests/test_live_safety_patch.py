"""Targeted regression tests for the 4 minimal live-money safety fixes on main."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.config import DB_PATH
from backend.data_health import build_clean_price_matrix, evaluate_data_freshness
from backend.database import Database
from backend.execution_engine import commit_paper_rebalance, generate_rebalance_ticket
from backend.indian_costs import (
    INDIA_LTCG_EFFECTIVE_RATE,
    INDIA_LTCG_FY_EXEMPTION_INR,
    INDIA_STCG_EFFECTIVE_RATE,
    FifoTaxLedger,
    compute_indian_delivery_cost,
)
from backend.main import SESSION_API_TOKEN, app


def test_peak_to_trough_drawdown_brake_triggers_above_initial_capital(tmp_path) -> None:
    """If NAV grows 10L -> 14L and falls -15% to 11.9L, peak-to-trough brake MUST trigger."""
    tmp_db = tmp_path / "dd_brake.db"
    shutil.copyfile(DB_PATH, tmp_db)
    db = Database(tmp_db)
    db.reset_paper_account()

    commit_paper_rebalance(db, capital_inr=1_000_000.0, top_k=10, risk_profile="balanced", allow_stale=True)
    # Simulate peak NAV of ₹14L while current NAV is ₹11.9L (still +19% above ₹10L initial, but -15% off peak!)
    with db.connect() as conn:
        conn.execute("UPDATE paper_account SET peak_nav_inr = 1400000.0, dd_brake_active = 0 WHERE id = 1")

    tk = generate_rebalance_ticket(
        db,
        capital_inr=1_190_000.0,
        top_k=10,
        risk_profile="balanced",
        current_holdings=db.get_paper_positions(),
    )
    assert tk["summary"]["dd_brake_active"] is True
    assert tk["summary"]["current_dd_pct"] == -15.0
    assert tk["summary"]["target_equity_pct"] <= 50.1


def test_stale_and_suspended_data_guards(tmp_path) -> None:
    # 1. Bounded ffill prevents buying suspended/delisted stocks
    dates = [f"2025-01-{d:02d}" for d in range(1, 10)]
    ohlcv = {
        "ALIVE.NS": [{"date": d, "open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0 + i, "volume": 10000} for i, d in enumerate(dates)],
        "SUSP.NS": [{"date": d, "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0, "volume": 5000} for d in dates[:3]],
    }
    px, _, tradable = build_clean_price_matrix(ohlcv, max_ffill_days=2)
    assert pd.isna(px.loc["2025-01-06", "SUSP.NS"])
    assert bool(tradable.loc["2025-01-06", "SUSP.NS"]) is False

    # 2. Freshness check blocks stale data
    tmp_db = tmp_path / "stale.db"
    shutil.copyfile(DB_PATH, tmp_db)
    db = Database(tmp_db)
    stale = evaluate_data_freshness(db, now_ist=datetime(2028, 6, 1, 19, 0))
    assert stale["status"] == "stale"
    assert stale["block_execution"] is True


def test_fifo_indian_tax_and_statutory_delivery_costs() -> None:
    buy_c = compute_indian_delivery_cost("BUY", 100_000.0, adv_20d_inr=500_000_000.0)
    sell_c = compute_indian_delivery_cost("SELL", 100_000.0, adv_20d_inr=500_000_000.0)
    assert buy_c["stt_inr"] == 100.0 and sell_c["stt_inr"] == 100.0
    assert buy_c["stamp_duty_inr"] == 15.0 and sell_c["stamp_duty_inr"] == 0.0
    assert buy_c["dp_charge_inr"] == 0.0 and sell_c["dp_charge_inr"] == 15.93

    ledger = FifoTaxLedger()
    ledger.buy("INFY.NS", "2024-05-01", shares=100, price=1000.0)
    ledger.buy("INFY.NS", "2025-10-01", shares=100, price=1200.0)
    tx = ledger.sell("INFY.NS", "2026-06-01", shares=150, price=2500.0)
    assert abs(tx["ltcg_tax_inr"] - round((150_000.0 - INDIA_LTCG_FY_EXEMPTION_INR) * INDIA_LTCG_EFFECTIVE_RATE, 2)) <= 0.02
    assert abs(tx["stcg_tax_inr"] - round(65_000.0 * INDIA_STCG_EFFECTIVE_RATE, 2)) <= 0.02


def test_api_mutation_auth_cors_reset_confirm_and_safe_urls() -> None:
    client = TestClient(app)

    # Unauthenticated POST blocked
    assert client.post("/api/execution/paper/reset", json={"confirm": True}).status_code == 401
    assert client.post("/api/execution/paper/commit", json={"capital_inr": 1000000}).status_code == 401

    # Authenticated reset without confirm=True blocked
    r_no_conf = client.post(
        "/api/execution/paper/reset",
        json={"confirm": False},
        headers={"X-EquiLens-Token": SESSION_API_TOKEN},
    )
    assert r_no_conf.status_code == 400

    # Strict CORS rejects arbitrary external origins
    r_cors = client.options("/api/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert r_cors.headers.get("access-control-allow-origin") != "*"
    assert r_cors.headers.get("access-control-allow-origin") != "https://evil.example"

    # Index injects session token with no-store
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert r_home.headers.get("cache-control") == "no-store"
    assert f'<meta name="equilens-token" content="{SESSION_API_TOKEN}" />' in r_home.text

    # Frontend validates external URL schemes
    app_js = (Path(__file__).resolve().parent.parent / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "function safeExternalUrl(" in app_js
    assert not re.search(r'href="\$\{escHtml\(n\.url\)\}"', app_js)
