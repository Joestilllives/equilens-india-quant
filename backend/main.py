"""FastAPI REST server and static frontend host for EquiLens Quant v2."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .alerts import run_alert_scan
from .backtester import run_backtest
from .config import BENCHMARK_LABEL, BENCHMARK_SYMBOL, CURRENCY_SYMBOL, DEFAULT_COST_BPS, FRONTEND_DIR, MARKET_REGION
from .data_fetcher import sync_single_ticker, sync_universe
from .database import Database
from .execution_engine import commit_paper_rebalance, generate_rebalance_ticket, get_paper_ledger_status
from .indicators import compute_indicator_frame, detect_chart_setups, serialize_indicator_series
from .journal import get_journal_report, log_and_grade_journal
from .ml_predictor import compute_current_macro_regime, train_and_predict_universe
from .sec_edgar import fetch_sec_form4_summary
from .shortlister import build_shortlist, invalidate_shortlist_cache
from .stress_tester import run_quantitative_stress_tests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("equilens")

app = FastAPI(
    title="EquiLens India Quant v6 API",
    version="2.0.0-IN",
    description="Live NSE data ingestion, V6 Sentiment & Macro ML, 4-Tranche Correlation-Penalized Portfolio Engine, Rebalance Trade-Ticket Generator, and Paper-Trading Ledger.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database()


class AddTickerRequest(BaseModel):
    symbol: str


class StarRequest(BaseModel):
    starred: bool


class HoldingInput(BaseModel):
    symbol: str
    shares: int
    avg_cost: float | None = None


class TicketRequest(BaseModel):
    capital_inr: float = 1_000_000.0
    top_k: int = 10
    risk_profile: str = "balanced"
    skip_earnings_blackout: bool = False
    use_paper_holdings: bool = True
    current_holdings: list[HoldingInput] | None = None


class CommitPaperRequest(BaseModel):
    capital_inr: float = 1_000_000.0
    top_k: int = 10
    risk_profile: str = "balanced"
    skip_earnings_blackout: bool = False


@app.get("/api/health")
def health() -> dict[str, Any]:
    tickers = db.get_tickers()
    ml_run = db.get_latest_ml_run()
    alerts = db.get_alerts(limit=100)
    return {
        "status": "ok",
        "version": "2.0.0-IN",
        "market": MARKET_REGION,
        "currency": CURRENCY_SYMBOL,
        "tickers_count": len(tickers),
        "ml_trained_at": ml_run["trained_at"] if ml_run else None,
        "active_alerts": sum(1 for a in alerts if not a["acknowledged"]),
        "benchmark": BENCHMARK_SYMBOL,
        "benchmark_label": BENCHMARK_LABEL,
    }


@app.get("/api/shortlist")
def get_shortlist(
    w_ml: float = Query(0.30, ge=0.0, le=1.0),
    w_mom: float = Query(0.25, ge=0.0, le=1.0),
    w_val: float = Query(0.15, ge=0.0, le=1.0),
    w_qual: float = Query(0.15, ge=0.0, le=1.0),
    w_sent: float = Query(0.15, ge=0.0, le=1.0),
    sector: str = Query("ALL"),
    setup: str = Query("ALL"),
    include_benchmark: bool = Query(False),
) -> dict[str, Any]:
    weights = {"ml": w_ml, "momentum": w_mom, "valuation": w_val, "quality": w_qual, "sentiment": w_sent}
    items = build_shortlist(
        db,
        weights=weights,
        sector=sector,
        setup_filter=setup,
        include_benchmark=include_benchmark,
    )
    all_tickers = db.get_tickers()
    sectors = sorted({t["sector"] for t in all_tickers if t["symbol"] != BENCHMARK_SYMBOL})
    ml_run = db.get_latest_ml_run()
    macro_regime = compute_current_macro_regime(db)
    alerts = db.get_alerts(limit=100)

    bench_summary = None
    bench_bars = db.get_ohlcv(BENCHMARK_SYMBOL)
    if len(bench_bars) >= 2:
        last_b, prev_b = bench_bars[-1], bench_bars[-2]
        chg = last_b["close"] - prev_b["close"]
        bench_summary = {
            "symbol": BENCHMARK_SYMBOL,
            "label": BENCHMARK_LABEL,
            "price": round(last_b["close"], 2),
            "chg_1d_pct": round((chg / max(1e-9, prev_b["close"])) * 100.0, 2),
            "as_of": last_b["date"],
        }

    return {
        "items": items,
        "sectors": sectors,
        "benchmark": bench_summary,
        "macro_regime": macro_regime,
        "unread_alerts": sum(1 for a in alerts if not a["acknowledged"]),
        "ml_summary": {
            "trained_at": ml_run["trained_at"],
            "horizon_days": ml_run["horizon_days"],
            "hit_rate": ml_run["hit_rate"],
            "ic_spearman": ml_run["ic_spearman"],
            "top5_spread": ml_run["top5_spread"],
            "tb_auc": ml_run["tb_auc"],
            "tb_top5_precision": ml_run["tb_top5_precision"],
        }
        if ml_run
        else None,
    }


import threading
import time

_DETAIL_LOCK = threading.Lock()
_DETAIL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _clear_all_caches() -> None:
    invalidate_shortlist_cache()
    with _DETAIL_LOCK:
        _DETAIL_CACHE.clear()


@app.get("/api/stock/{symbol}")
def get_stock_detail(symbol: str, include_sec: bool = Query(False)) -> dict[str, Any]:
    sym = symbol.upper()
    now = time.monotonic()
    if not include_sec:
        with _DETAIL_LOCK:
            cached = _DETAIL_CACHE.get(sym)
            if cached and (now - cached[0]) < 300.0:
                return cached[1]

    ticker = db.get_ticker(sym)
    if not ticker:
        raise HTTPException(status_code=404, detail=f"Ticker '{sym}' not found in local store.")

    bars = db.get_ohlcv(sym)
    if len(bars) < 5:
        raise HTTPException(status_code=404, detail=f"No price history stored for '{sym}'.")

    bench_bars = db.get_ohlcv(BENCHMARK_SYMBOL)
    df = compute_indicator_frame(bars, bench_bars)
    series = serialize_indicator_series(df.tail(800))
    setups = detect_chart_setups(df)
    news = db.get_news(sym, limit=12)
    earnings = db.get_earnings(sym)
    insiders = db.get_insiders(sym, limit=15)
    options = db.get_options_snapshot(sym)
    ml_pred = db.get_ml_predictions().get(sym)
    sec_form4 = fetch_sec_form4_summary(sym, limit=5) if include_sec else []

    scored_list = build_shortlist(db, include_benchmark=True)
    score_card = next((x for x in scored_list if x["symbol"] == sym), None)

    last = df.iloc[-1]
    prev = df.iloc[-2]
    stats = {
        "open": round(float(last["open"]), 2),
        "high": round(float(last["high"]), 2),
        "low": round(float(last["low"]), 2),
        "close": round(float(last["close"]), 2),
        "prev_close": round(float(prev["close"]), 2),
        "volume": int(last["volume"]),
        "high_52w": round(float(last["high_52w"]), 2) if pd.notna(last.get("high_52w")) else None,
        "low_52w": round(float(last["low_52w"]), 2) if pd.notna(last.get("low_52w")) else None,
        "rsi14": round(float(last["rsi14"]), 1) if pd.notna(last.get("rsi14")) else None,
        "atr14": round(float(last["atr14"]), 2) if pd.notna(last.get("atr14")) else None,
        "atr_pct": round(float(last["atr_pct"] * 100.0), 2) if pd.notna(last.get("atr_pct")) else None,
        "vol_20d_pct": round(float(last["vol_20d"] * 100.0), 1) if pd.notna(last.get("vol_20d")) else None,
        "rs_spy_63d_pct": round(float(last["rs_spy_63d"] * 100.0), 2) if pd.notna(last.get("rs_spy_63d")) else None,
    }

    payload = {
        "ticker": ticker,
        "stats": stats,
        "score_card": score_card,
        "setups": setups,
        "ml_prediction": ml_pred,
        "earnings": earnings,
        "insiders": insiders,
        "sec_form4": sec_form4,
        "options": options,
        "news": news,
        "series": series,
    }
    if not include_sec:
        with _DETAIL_LOCK:
            _DETAIL_CACHE[sym] = (now, payload)
    return payload


@app.post("/api/tickers/add")
def add_ticker(req: AddTickerRequest) -> dict[str, Any]:
    sym = req.symbol.strip().upper()
    try:
        synced = sync_single_ticker(db, sym)
        ml_res = train_and_predict_universe(db)
        _clear_all_caches()
        log_and_grade_journal(db)
        run_alert_scan(db)
        return {"synced": synced, "ml_trained_at": ml_res["run"]["trained_at"]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error adding ticker %s", sym)
        raise HTTPException(status_code=500, detail=f"Failed to fetch ticker '{sym}': {exc}") from exc


@app.post("/api/tickers/{symbol}/star")
def toggle_star(symbol: str, req: StarRequest) -> dict[str, Any]:
    ok = db.set_starred(symbol, req.starred)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Ticker '{symbol}' not found.")
    _clear_all_caches()
    return {"symbol": symbol.upper(), "starred": req.starred}


@app.post("/api/sync")
def sync_all_data() -> dict[str, Any]:
    existing = [t["symbol"] for t in db.get_tickers()]
    symbols = sorted(set(existing)) if len(existing) >= 30 else None
    sync_res = sync_universe(db, symbols=symbols)
    ml_res = train_and_predict_universe(db)
    _clear_all_caches()
    journal_res = log_and_grade_journal(db)
    alerts = run_alert_scan(db)
    return {
        "sync": sync_res,
        "ml": ml_res["run"],
        "journal_logged": journal_res["summary"]["total_logged"],
        "alerts_generated": len(alerts),
    }


@app.get("/api/model/metrics")
def get_model_metrics() -> dict[str, Any]:
    run = db.get_latest_ml_run()
    if not run:
        raise HTTPException(status_code=404, detail="ML model has not been trained yet.")
    preds = sorted(
        db.get_ml_predictions().values(),
        key=lambda p: (p.get("prob_tb_win", 50.0) * 0.6 + p["pred_alpha_20d"] * 4.0),
        reverse=True,
    )
    return {"run": run, "macro_regime": compute_current_macro_regime(db), "predictions": preds}


@app.get("/api/backtest/{symbol}")
def backtest_symbol(
    symbol: str,
    strategy: str = Query("sma_crossover"),
    fast: int = Query(10, ge=3, le=60),
    slow: int = Query(35, ge=10, le=200),
    capital: float = Query(100000.0, ge=5000.0, le=100000000.0),
) -> dict[str, Any]:
    bars = db.get_ohlcv(symbol.upper())
    if not bars:
        raise HTTPException(status_code=404, detail=f"No price data for '{symbol}'.")
    if strategy == "sma_crossover" and fast >= slow:
        slow = fast + 5
    try:
        return run_backtest(
            bars=bars,
            strategy=strategy,
            fast_period=fast,
            slow_period=slow,
            initial_capital=capital,
            commission_bps=DEFAULT_COST_BPS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/portfolio/stress")
def portfolio_stress_test(
    top_k: int = Query(10, ge=2, le=20),
    rebalance_days: int = Query(40, ge=10, le=80),
    regime_gate: bool = Query(True),
    risk_profile: str = Query("balanced"),
) -> dict[str, Any]:
    try:
        return run_quantitative_stress_tests(
            db,
            top_k=top_k,
            rebalance_days=rebalance_days,
            regime_gate=regime_gate,
            risk_profile=risk_profile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/execution/ticket")
def get_execution_ticket(
    capital_inr: float = Query(1_000_000.0, ge=25_000.0, le=500_000_000.0),
    top_k: int = Query(10, ge=4, le=20),
    risk_profile: str = Query("balanced"),
    skip_earnings_blackout: bool = Query(False),
    use_paper_holdings: bool = Query(True),
) -> dict[str, Any]:
    try:
        holdings = None
        effective_cap = capital_inr
        if use_paper_holdings and db.get_paper_account():
            st = get_paper_ledger_status(db)
            holdings = db.get_paper_positions()
            effective_cap = float(st["account"]["total_nav_inr"])
        return generate_rebalance_ticket(
            db,
            capital_inr=effective_cap,
            top_k=top_k,
            max_per_sector=2,
            risk_profile=risk_profile,
            skip_earnings_blackout=skip_earnings_blackout,
            current_holdings=holdings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/execution/ticket")
def post_execution_ticket(req: TicketRequest) -> dict[str, Any]:
    try:
        if req.current_holdings is not None:
            holdings = [h.model_dump() for h in req.current_holdings]
            effective_cap = req.capital_inr
        elif req.use_paper_holdings and db.get_paper_account():
            st = get_paper_ledger_status(db)
            holdings = db.get_paper_positions()
            effective_cap = float(st["account"]["total_nav_inr"])
        else:
            holdings = None
            effective_cap = req.capital_inr
        return generate_rebalance_ticket(
            db,
            capital_inr=effective_cap,
            top_k=max(4, min(20, req.top_k)),
            max_per_sector=2,
            risk_profile=req.risk_profile,
            skip_earnings_blackout=req.skip_earnings_blackout,
            current_holdings=holdings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/execution/paper/status")
def paper_ledger_status() -> dict[str, Any]:
    return get_paper_ledger_status(db)


@app.post("/api/execution/paper/commit")
def paper_ledger_commit(req: CommitPaperRequest) -> dict[str, Any]:
    try:
        return commit_paper_rebalance(
            db,
            capital_inr=max(25_000.0, req.capital_inr),
            top_k=max(4, min(20, req.top_k)),
            risk_profile=req.risk_profile,
            skip_earnings_blackout=req.skip_earnings_blackout,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/execution/paper/reset")
def paper_ledger_reset() -> dict[str, Any]:
    db.reset_paper_account()
    return {"reset": True, "ledger": get_paper_ledger_status(db)}


@app.get("/api/journal")
def get_journal() -> dict[str, Any]:
    return get_journal_report(db)


@app.post("/api/journal/log")
def trigger_journal_log() -> dict[str, Any]:
    return log_and_grade_journal(db)


@app.get("/api/alerts")
def list_alerts() -> dict[str, Any]:
    items = db.get_alerts(limit=80)
    return {
        "alerts": items,
        "unread_count": sum(1 for a in items if not a["acknowledged"]),
    }


@app.post("/api/alerts/scan")
def scan_alerts() -> dict[str, Any]:
    items = run_alert_scan(db)
    return {"alerts": items, "count": len(items)}


@app.post("/api/alerts/{alert_id}/ack")
def ack_alert(alert_id: str) -> dict[str, Any]:
    ok = db.acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return {"id": alert_id, "acknowledged": True}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_path))
