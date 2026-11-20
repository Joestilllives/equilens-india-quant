"""Comprehensive Stress Test Suite for EquiLens India (NSE / NIFTY 50): Leakage, Chaos, Quant Stress & Load."""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi.testclient import TestClient

from backend.backtester import run_backtest
from backend.catalyst_nlp import analyze_financial_news
from backend.indicators import compute_indicator_frame
from backend.main import app, db as live_db
from backend.ml_predictor import compute_triple_barrier_labels
from backend.stress_tester import run_quantitative_stress_tests


def test_zero_lookahead_leakage_canary() -> None:
    """Mutating future bars after day T must have ZERO effect on any feature computed at day T."""
    bars = live_db.get_ohlcv("RELIANCE.NS")[:300]
    nifty = live_db.get_ohlcv("^NSEI")[:300]
    cutoff = 200

    df_orig = compute_indicator_frame(bars, nifty)

    corrupted = [dict(b) for b in bars]
    for i in range(cutoff + 1, len(corrupted)):
        corrupted[i]["open"] *= 3.5
        corrupted[i]["high"] *= 4.0
        corrupted[i]["low"] *= 0.2
        corrupted[i]["close"] *= 3.8
        corrupted[i]["volume"] *= 50

    df_mut = compute_indicator_frame(corrupted, nifty)

    feature_cols = [
        "sma20", "sma50", "sma200", "ema12", "rsi14", "macd", "macd_hist",
        "bb_upper", "bb_lower", "atr14", "vol_20d", "vol_ratio_5_20", "vwap20",
        "ret_5d", "ret_21d", "ret_63d", "rs_spy_21d", "rs_spy_63d", "pos_52w",
    ]
    for col in feature_cols:
        v1 = df_orig.loc[cutoff, col]
        v2 = df_mut.loc[cutoff, col]
        if np.isnan(v1) and np.isnan(v2):
            continue
        assert np.isclose(v1, v2, rtol=1e-9, atol=1e-9), f"Look-ahead leakage detected in {col}!"


def test_triple_barrier_exact_mechanics() -> None:
    n = 40
    closes = np.full(n, 1000.0)
    highs = np.full(n, 1005.0)
    lows = np.full(n, 995.0)
    atrs = np.full(n, 20.0)  # +2.5*ATR = 1050.0, -1.5*ATR = 970.0

    highs[3] = 1060.0
    lows[7] = 960.0

    labels = compute_triple_barrier_labels(closes, highs, lows, atrs, horizon=10, up_mult=2.5, down_mult=1.5)
    assert labels[0] == 1.0
    assert labels[5] == 0.0
    assert np.all(np.isnan(labels[-10:]))


def test_chaos_flash_crash_and_zero_volume() -> None:
    bars = [dict(b) for b in live_db.get_ohlcv("RELIANCE.NS")[-160:]]
    for i in range(80, 85):
        bars[i]["volume"] = 0
    bars[100]["open"] *= 0.25
    bars[100]["low"] *= 0.15
    bars[100]["high"] *= 0.30
    bars[100]["close"] *= 0.18

    df = compute_indicator_frame(bars)
    assert not np.isinf(df["rsi14"].dropna()).any()
    assert not np.isinf(df["atr_pct"].dropna()).any()

    bt = run_backtest(bars, strategy="sma_crossover", fast_period=10, slow_period=30, initial_capital=100000.0)
    assert np.isfinite(bt["metrics"]["final_equity"])
    assert bt["metrics"]["final_equity"] > 0


def test_indian_catalyst_nlp_tagging() -> None:
    score, label, tags = analyze_financial_news(
        "L&T bags Rs 4,500 crore order win; net profit jumps 22% beating estimates",
        "Brokerages upgrade target price on strong order book.",
    )
    assert score > 0.25 and label == "Bullish"
    assert any(t in tags for t in ("Order / Deal Win", "Earnings Beat", "Brokerage Upgrade"))


def test_quantitative_stress_suite_integrity() -> None:
    res = run_quantitative_stress_tests(live_db, top_k=6, rebalance_days=20, regime_gate=True)

    slips = res["slippage_stress"]
    assert len(slips) == 4
    assert slips[0]["cagr_pct"] >= slips[-1]["cagr_pct"]

    bs = res["bootstrap_stress"]
    assert bs["simulations"] == 1000
    assert bs["worst_99_max_dd_pct"] <= bs["worst_95_max_dd_pct"] <= bs["median_max_dd_pct"] <= 0.0
    assert bs["daily_cvar_95_pct"] <= bs["daily_var_95_pct"]

    assert 0.0 <= res["overall"]["ranking_stability_jaccard_pct"] <= 100.0
    assert len(res["regime_replay"]) == 4


def test_api_edge_cases_and_security() -> None:
    client = TestClient(app)

    r_zero = client.get("/api/shortlist?w_ml=0&w_mom=0&w_val=0&w_qual=0&w_sent=0")
    assert r_zero.status_code == 200
    assert len(r_zero.json()["items"]) >= 45

    for bad in ["'; DROP TABLE tickers;--", "!!!BAD!!!", "TOO_LONG_TICKER_NAME_999"]:
        r_bad = client.post("/api/tickers/add", json={"symbol": bad})
        assert r_bad.status_code == 400

    assert client.get("/api/stock/NOTAREALNSE99.NS").status_code == 404


def test_concurrent_api_load_stress() -> None:
    endpoints = [
        "/api/health",
        "/api/shortlist",
        "/api/stock/RELIANCE.NS",
        "/api/stock/HDFCBANK.NS",
        "/api/model/metrics",
        "/api/journal",
        "/api/alerts",
        "/api/backtest/TCS.NS?strategy=rsi_trend",
    ]
    tasks = endpoints * 6  # 48 concurrent requests

    def _hit(ep: str) -> tuple[int, float]:
        client = TestClient(app)
        t0 = time.perf_counter()
        resp = client.get(ep)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return resp.status_code, dt_ms

    warm = TestClient(app)
    warm.get("/api/shortlist")
    warm.get("/api/journal")

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_hit, tasks))

    codes = [r[0] for r in results]
    latencies = sorted(r[1] for r in results)
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]

    assert all(c == 200 for c in codes), f"Non-200 status codes under load: {set(codes)}"
    assert p95 < 2500.0, f"p95 latency too high under load: {p95:.1f}ms"
    print(f"\n[INDIA LOAD STRESS] 48 concurrent requests -> 100% HTTP 200 | p50={p50:.1f}ms | p95={p95:.1f}ms | max={latencies[-1]:.1f}ms")
