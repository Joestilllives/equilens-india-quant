"""Automated Catalyst & Signal Alert Scanner for Indian NSE / NIFTY 50 equities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .config import BENCHMARK_SYMBOL
from .database import Database
from .ml_predictor import compute_current_macro_regime
from .shortlister import build_shortlist


def run_alert_scan(db: Database) -> list[dict[str, Any]]:
    """Scan the NSE universe and India macro regime to generate prioritized actionable alerts."""
    now_iso = datetime.now(timezone.utc).isoformat()
    regime = compute_current_macro_regime(db)
    shortlist = build_shortlist(db)
    alerts: list[dict[str, Any]] = []

    if regime["tone"] == "bearish":
        alerts.append(
            {
                "id": f"macro_riskoff_{now_iso[:10]}",
                "created_at": now_iso,
                "symbol": BENCHMARK_SYMBOL,
                "alert_type": "India Macro Regime",
                "severity": "high",
                "title": f"Risk-Off NSE Regime (India VIX {regime['vix']} · Breadth {regime['breadth_50d_pct']}%)",
                "detail": "Elevated India VIX or deteriorating NSE breadth detected. Tighten stop-losses and favor smaller position sizes.",
                "acknowledged": False,
            }
        )
    elif regime["tone"] == "warning":
        alerts.append(
            {
                "id": f"macro_caution_{now_iso[:10]}",
                "created_at": now_iso,
                "symbol": BENCHMARK_SYMBOL,
                "alert_type": "India Macro Regime",
                "severity": "medium",
                "title": f"Transitional NSE Regime (India VIX {regime['vix']} · Breadth {regime['breadth_50d_pct']}%)",
                "detail": "Mixed NIFTY breadth; prioritize high relative-strength leaders over speculative laggards.",
                "acknowledged": False,
            }
        )

    for it in shortlist:
        sym = it["symbol"]
        disp = sym.replace(".NS", "")
        if it["composite_score"] >= 69.0 and it["ml"]["prob_tb_win"] >= 53.0:
            alerts.append(
                {
                    "id": f"hc_{sym}_{now_iso[:10]}",
                    "created_at": now_iso,
                    "symbol": sym,
                    "alert_type": "High-Conviction Pick",
                    "severity": "bullish",
                    "title": f"{disp} ranked #{it['rank']} (Score {it['composite_score']} · TB Win {it['ml']['prob_tb_win']}%)",
                    "detail": f"Target ₹{it['ml']['target_price']} / Stop ₹{it['ml']['stop_price']} ({it['ml']['risk_reward']}R). {it['bullish_drivers'][0]}",
                    "acknowledged": False,
                }
            )

        ins = it.get("insider_summary") or {}
        if ins.get("cluster_buy"):
            alerts.append(
                {
                    "id": f"ins_{sym}_{now_iso[:10]}",
                    "created_at": now_iso,
                    "symbol": sym,
                    "alert_type": "Promoter / Insider Buy",
                    "severity": "bullish",
                    "title": f"{disp}: Open-Market Insider Buying (₹{ins['buy_value_90d']:,.0f})",
                    "detail": f"{ins['buys_90d']} purchase(s) across {ins['unique_buyers_90d']} insider(s) in the last 90 days.",
                    "acknowledged": False,
                }
            )

        es = it.get("earnings_summary") or {}
        dte = es.get("days_to_earnings")
        if dte is not None and 0 <= dte <= 10:
            alerts.append(
                {
                    "id": f"earn_{sym}_{now_iso[:10]}",
                    "created_at": now_iso,
                    "symbol": sym,
                    "alert_type": "Results Risk",
                    "severity": "medium",
                    "title": f"{disp} reports quarterly results in {dte} day(s) ({es['next_earnings_date']})",
                    "detail": f"Beat streak: {es['beat_streak']}Q (avg 4Q surprise: {es['avg_4q_surprise_pct']}%). Manage binary gap exposure.",
                    "acknowledged": False,
                }
            )

        for st in it.get("setups", []):
            if st["id"] == "breakout_20d":
                alerts.append(
                    {
                        "id": f"brk_{sym}_{now_iso[:10]}",
                        "created_at": now_iso,
                        "symbol": sym,
                        "alert_type": "NSE Breakout",
                        "severity": "bullish",
                        "title": f"{disp}: 20-Day High Breakout on Elevated Volume",
                        "detail": st["detail"],
                        "acknowledged": False,
                    }
                )

    order = {"high": 0, "bullish": 1, "medium": 2}
    alerts.sort(key=lambda a: (order.get(a["severity"], 3), a["symbol"]))
    db.replace_alerts(alerts[:60])
    return alerts[:60]
