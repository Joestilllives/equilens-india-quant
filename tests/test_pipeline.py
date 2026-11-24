"""Automated tests for EquiLens India (NSE / NIFTY 50) quantitative pipeline and FastAPI server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.backtester import run_backtest
from backend.data_fetcher import normalize_indian_symbol, score_financial_text
from backend.indicators import compute_indicator_frame, detect_chart_setups
from backend.main import app, db as live_db
from backend.ml_predictor import build_probability_cone
from backend.shortlister import build_shortlist


def test_indian_symbol_normalization() -> None:
    assert normalize_indian_symbol("reliance") == "RELIANCE.NS"
    assert normalize_indian_symbol("TCS.NS") == "TCS.NS"
    assert normalize_indian_symbol("^NSEI") == "^NSEI"
    assert normalize_indian_symbol("HDFCBANK.BO") == "HDFCBANK.BO"


def test_financial_sentiment_scorer() -> None:
    pos_score, pos_label = score_financial_text(
        "TCS beats estimates and raises guidance on strong deal wins",
        "Brokerages upgrade target price after record quarterly PAT growth.",
    )
    neg_score, neg_label = score_financial_text(
        "Company misses estimates and cuts guidance amid SEBI notice",
        "Shares plunge after weak demand and RBI restriction warning.",
    )
    assert pos_score > 0.2 and pos_label == "Bullish"
    assert neg_score < -0.2 and neg_label == "Bearish"


def test_probability_cone_monotonic() -> None:
    cone = build_probability_cone("2026-11-19", 1400.0, 0.03, 0.22)
    assert len(cone) >= 6
    assert cone[0]["p50"] == 1400.0
    end = cone[-1]
    assert end["p10"] < end["p25"] < end["p50"] < end["p75"] < end["p90"]


def test_indicators_and_backtest_on_live_nse_db() -> None:
    bars = live_db.get_ohlcv("RELIANCE.NS")
    nifty = live_db.get_ohlcv("^NSEI")
    assert len(bars) >= 200
    assert len(nifty) >= 200
    df = compute_indicator_frame(bars, nifty)
    for col in ("sma20", "sma50", "rsi14", "macd_hist", "rs_spy_63d"):
        assert col in df.columns
        assert df[col].dropna().shape[0] > 50

    setups = detect_chart_setups(df)
    assert isinstance(setups, list)

    bt = run_backtest(bars, strategy="sma_crossover", fast_period=10, slow_period=35, initial_capital=100000.0, commission_bps=15.0)
    assert "metrics" in bt and len(bt["equity_curve"]) > 100
    assert bt["metrics"]["final_equity"] > 0


def test_shortlist_and_custom_weights() -> None:
    default_list = build_shortlist(live_db)
    assert len(default_list) >= 45
    scores = [item["composite_score"] for item in default_list]
    assert scores == sorted(scores, reverse=True)

    ml_heavy = build_shortlist(live_db, weights={"ml": 1.0, "momentum": 0.0, "valuation": 0.0, "quality": 0.0, "sentiment": 0.0})
    assert len(ml_heavy) == len(default_list)
    assert ml_heavy[0]["pillars"]["ml"] >= ml_heavy[-1]["pillars"]["ml"]


def test_fastapi_india_endpoints() -> None:
    client = TestClient(app)

    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "EquiLens India" in r_home.text

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    hj = r_health.json()
    assert hj["market"] == "IN"
    assert hj["currency"] == "₹"
    assert hj["benchmark"] == "^NSEI"
    assert hj["tickers_count"] >= 45

    r_short = client.get("/api/shortlist")
    assert r_short.status_code == 200
    payload = r_short.json()
    assert len(payload["items"]) >= 45
    assert payload["macro_regime"]["vix"] > 0
    assert payload["macro_regime"]["usdinr"] > 50

    r_stock = client.get("/api/stock/RELIANCE.NS")
    assert r_stock.status_code == 200
    stock = r_stock.json()
    assert stock["ticker"]["symbol"] == "RELIANCE.NS"
    assert len(stock["series"]) >= 200
    assert len(stock["ml_prediction"]["cone"]) >= 6

    r_metrics = client.get("/api/model/metrics")
    assert r_metrics.status_code == 200
    m = r_metrics.json()
    assert len(m["run"]["folds"]) >= 2

    r_bt = client.get("/api/backtest/TCS.NS?strategy=rsi_trend")
    assert r_bt.status_code == 200
    assert "equity_curve" in r_bt.json()

    r_tk = client.get("/api/execution/ticket?capital_inr=1000000&top_k=10&risk_profile=balanced&enforce_smart_money_shield=true")
    assert r_tk.status_code == 200
    tk = r_tk.json()
    assert tk["summary"]["positions_count"] == 10
    assert tk["summary"]["distinct_sectors"] >= 5
    assert tk["summary"]["liquidcase_cash_inr"] >= 0
    assert isinstance(tk["skipped_smart_money"], list)
    assert tk["config"]["enforce_smart_money_shield"] is True

    r_ps = client.get("/api/execution/paper/status")
    assert r_ps.status_code == 200
    assert "initialized" in r_ps.json()

    r_ms = client.get("/api/money-signals")
    assert r_ms.status_code == 200
    ms = r_ms.json()
    assert "fii_positioning" in ms
    assert "liquidity_ratios" in ms
    assert len(ms["stock_delivery_map"]) >= 50


def test_v6_execution_ticket_and_paper_ledger_lifecycle(tmp_path) -> None:
    import shutil
    from backend.config import DB_PATH
    from backend.database import Database
    from backend.execution_engine import commit_paper_rebalance, generate_rebalance_ticket, get_paper_ledger_status

    tmp_db_path = tmp_path / "test_paper.db"
    shutil.copyfile(DB_PATH, tmp_db_path)
    db = Database(tmp_db_path)

    # 1. Verify Top-10 Balanced ticket enforces Max 2 per sector, >= 5 sectors, and Smart-Money Shield
    t_bal = generate_rebalance_ticket(
        db,
        capital_inr=1_000_000.0,
        top_k=10,
        max_per_sector=2,
        risk_profile="balanced",
        enforce_smart_money_shield=True,
    )
    assert t_bal["summary"]["positions_count"] == 10
    assert t_bal["summary"]["distinct_sectors"] >= 5
    assert isinstance(t_bal["skipped_smart_money"], list)
    for sb in t_bal["sector_breakdown"]:
        assert sb["count"] <= 2
        assert sb["weight_pct"] <= 20.1
    assert all(isinstance(o["target_shares"], int) and o["target_shares"] >= 0 for o in t_bal["orders"])
    assert all(o["target_weight_pct"] <= 12.1 for o in t_bal["orders"])
    assert all(o.get("shield_status") != "Risk" for o in t_bal["orders"])
    assert all(o.get("deliv_pct") is None or o["deliv_pct"] >= 35.0 for o in t_bal["orders"])
    assert t_bal["config"]["dd_brake_trigger_pct"] == -10.0
    assert t_bal["config"]["goldbees_reserve_pct"] == 30.0
    assert abs((t_bal["summary"]["liquidcase_alloc_inr"] + t_bal["summary"]["goldbees_alloc_inr"]) - t_bal["summary"]["liquidcase_cash_inr"]) <= 0.05

    # Verify Top-20 ticket strictly enforces <=5% per stock and <=20% per sector
    t_20 = generate_rebalance_ticket(
        db,
        capital_inr=1_000_000.0,
        top_k=20,
        risk_profile="balanced",
        enforce_smart_money_shield=True,
    )
    assert t_20["summary"]["positions_count"] == 20
    assert all(o["target_weight_pct"] <= 5.1 for o in t_20["orders"])
    assert all(sb["weight_pct"] <= 20.1 for sb in t_20["sector_breakdown"])

    # Verify disabling the shield works cleanly too
    t_unshielded = generate_rebalance_ticket(
        db,
        capital_inr=1_000_000.0,
        top_k=10,
        max_per_sector=2,
        risk_profile="balanced",
        enforce_smart_money_shield=False,
    )
    assert t_unshielded["skipped_smart_money"] == []

    # 2. Growth profile should deploy >= Balanced equity exposure
    t_gro = generate_rebalance_ticket(db, capital_inr=1_000_000.0, top_k=10, max_per_sector=2, risk_profile="growth")
    assert t_gro["summary"]["target_equity_pct"] >= t_bal["summary"]["target_equity_pct"]

    # 3. Paper-Trading Ledger Commit -> Status -> Rebalance Hysteresis (zero churn on immediate re-run!) -> Reset
    db.reset_paper_account()
    assert get_paper_ledger_status(db)["initialized"] is False

    res1 = commit_paper_rebalance(db, capital_inr=1_000_000.0, top_k=10, risk_profile="balanced")
    assert res1["ledger"]["initialized"] is True
    assert len(res1["ledger"]["positions"]) == 10
    assert len(res1["ledger"]["trades"]) == 10
    assert res1["ledger"]["account"]["total_nav_inr"] > 990_000.0

    # Immediate re-check against paper holdings should produce HOLDs (0 SELLs, 0 BUYs due to deadband + 2xK buffer)
    t_hold = generate_rebalance_ticket(
        db,
        capital_inr=res1["ledger"]["account"]["total_nav_inr"],
        top_k=10,
        risk_profile="balanced",
        current_holdings=db.get_paper_positions(),
    )
    assert all(o["action"] == "HOLD" for o in t_hold["orders"])

    db.reset_paper_account()
    assert get_paper_ledger_status(db)["initialized"] is False

