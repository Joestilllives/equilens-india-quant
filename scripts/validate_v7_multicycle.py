"""
EquiLens India V7 — Full 5-Gate Zero-Overfitting Promotion & Multi-Cycle Validation Harness.

Runs strictly against a cloned copy of `data/equilens_india.db` (NEVER mutates the live DB):
  - Gate 1 (Hardened Net Performance): V7 Book Hybrid + HRP vs Clean V6 on FIFO Post-Tax
    Liquidation CAGR, Excess SR(6.2%), and 2-Factor Pure Alpha vs NIFTYBEES.NS TRI.
  - Gate 2 (Survivorship & Multi-Cycle):
      (a) 8.8-Year Full-History Multi-Cycle NSE Prior Test (2017-2026 incl. 2018 NBFC,
          2020 COVID crash/rebound, 2022 bear, 2025-2026 momentum drought) across all 9
          plateau cells of (Top-K in {8,10,12} x Rebalance in {30,40,60}) — 100% offline.
      (b) 20Y+ Multi-Cycle NSE Test (2004-2026 incl. 2008 GFC, 2009 momentum rebound,
          2011 bear) via Yahoo Finance + local cache (`data/multicycle_2004_2026_cache.db`).
  - Gate 3 (Multiple-Testing Rigor): CPCV PBO < 25% across candidate parameter grid AND
    improved corrected Bailey-Lopez de Prado (2014) DSR using empirical cross-trial variance.
  - Gate 4 (Stability & Friction): V7 beats V6 in >= 8/9 cells of (Top-K x Rebalance) AND
    maintains positive excess CAGR under 50 bps/side (100 bps RT) extreme cost stress.
  - Gate 5 (Deterministic Execution Separation): Verifies shortlister visual score isolation,
    zero non-equity ETF rows, and live ₹10L paper ledger schema backward compatibility.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backend.config import BENCHMARK_SYMBOL, DATA_DIR, DB_PATH, DEFAULT_UNIVERSE, NON_EQUITY_SYMBOLS
from backend.database import Database
from backend.execution_engine import generate_rebalance_ticket, get_paper_ledger_status
from backend.ml_predictor import _build_panel, train_and_predict_universe
from backend.portfolio_backtester import run_oos_portfolio_backtest
from backend.shortlister import build_shortlist
from scripts.audit_overfitting_v6 import compute_cpcv_pbo, compute_stats, deflated_sharpe_ratio

warnings.filterwarnings("ignore")

MULTICYCLE_CACHE_DB = DATA_DIR / "multicycle_2004_2026_cache.db"


def _calendar_year_line(dates: list[str], rets: list[float]) -> str:
    df = pd.DataFrame({"date": dates, "r": rets})
    df["yr"] = df["date"].str[:4]
    parts = []
    for yr, grp in df.groupby("yr"):
        tot = (float(np.prod(1.0 + grp["r"].to_numpy())) - 1.0) * 100.0
        parts.append(f"{yr[2:]}:{tot:+.1f}%")
    return " / ".join(parts)


def _run_prior_only_multicycle(db: Database, label: str) -> dict[str, Any]:
    """Run zero-parameter Book Prior + HRP vs V6 Prior + CorrPen across all available DB history (all 9 plateau cells)."""
    panel_v6, _ = _build_panel(db, model_version="v6", prior_variant="v6", enable_dm_damper=False)
    panel_v7, _ = _build_panel(db, model_version="v7", prior_variant="book", enable_dm_damper=True)

    oos_p6 = panel_v6[["date", "symbol", "nse_factor_prior"]].rename(columns={"nse_factor_prior": "pred_alpha"})
    oos_p7 = panel_v7[["date", "symbol", "nse_factor_prior"]].rename(columns={"nse_factor_prior": "pred_alpha"})

    wins_cagr = 0
    wins_sr = 0
    rows = []
    for k in (8, 10, 12):
        for reb in (30, 40, 60):
            b6 = run_oos_portfolio_backtest(db, top_k=k, rebalance_days=reb, weighting_mode="corr_penalized", oos_override=oos_p6)
            b7 = run_oos_portfolio_backtest(db, top_k=k, rebalance_days=reb, weighting_mode="hrp", oos_override=oos_p7)
            m6, m7 = b6["portfolio_metrics"], b7["portfolio_metrics"]
            if m7["cagr_pct"] > m6["cagr_pct"]:
                wins_cagr += 1
            if m7["excess_sharpe_rf62"] > m6["excess_sharpe_rf62"]:
                wins_sr += 1
            rows.append((k, reb, m6, m7))

    print(f"\n=== GATE 2A: {label} ZERO-PARAMETER PRIOR PLATEAU ({b6['config']['period']}) ===")
    print(f"  {'Cell':12s} | {'V6 CAGR':>8s} {'V7 CAGR':>8s} {'Diff':>7s} | {'V6 SR62':>7s} {'V7 SR62':>7s} | {'V6 DD':>7s} {'V7 DD':>7s}")
    print("  " + "-" * 80)
    for k, reb, m6, m7 in rows:
        print(
            f"  K={k:2d},R={reb:2d}D | {m6['cagr_pct']:+7.2f}% {m7['cagr_pct']:+7.2f}% {m7['cagr_pct'] - m6['cagr_pct']:+6.2f}% | "
            f"{m6['excess_sharpe_rf62']:7.2f} {m7['excess_sharpe_rf62']:7.2f} | {m6['max_drawdown_pct']:6.2f}% {m7['max_drawdown_pct']:6.2f}%"
        )
    print(f"  --> Multi-Cycle Zero-Param Prior Wins: CAGR {wins_cagr}/9 | Excess SR(6.2%) {wins_sr}/9")
    return {"wins_cagr": wins_cagr, "wins_sr": wins_sr, "period": b6["config"]["period"]}


def _ensure_2004_2026_cache(cache_db: Path) -> bool:
    """Download 2004-2026 NSE + ^NSEI + ^INDIAVIX + GC=F + INR=X history to local cache if absent."""
    if cache_db.exists():
        try:
            db = Database(cache_db)
            bars = db.get_ohlcv(BENCHMARK_SYMBOL)
            if len(bars) >= 3500:
                return True
        except Exception:
            pass

    try:
        import yfinance as yf

        syms = [s for s in DEFAULT_UNIVERSE if s not in NON_EQUITY_SYMBOLS]
        all_dl = [BENCHMARK_SYMBOL] + syms
        raw = yf.download(all_dl, start="2004-01-01", group_by="ticker", auto_adjust=True, threads=True, progress=False)
        if raw is None or raw.empty:
            return False
        db = Database(cache_db)
        inserted_syms = 0
        for sym in all_dl:
            try:
                df = raw[sym].dropna(subset=["Close"]) if len(all_dl) > 1 else raw.dropna(subset=["Close"])
                if len(df) < 500:
                    continue
                bars = [
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "open": float(r["Open"]),
                        "high": float(r["High"]),
                        "low": float(r["Low"]),
                        "close": float(r["Close"]),
                        "volume": int(r["Volume"]) if pd.notna(r["Volume"]) else 0,
                    }
                    for idx, r in df.iterrows()
                ]
                db.upsert_ticker(
                    symbol=sym, name=sym, sector="MultiCycle", industry="", exchange="NSE",
                    summary="", fundamentals={}, data_source="yfinance_20y", updated_at="2026-11-26T00:00:00Z"
                )
                db.upsert_ohlcv(sym, bars)
                inserted_syms += 1
            except Exception:
                continue
        for m_sym in ("^INDIAVIX", "INR=X", "GC=F", "BZ=F", "^GSPC"):
            try:
                mdf = yf.download(m_sym, start="2004-01-01", auto_adjust=True, progress=False)
                if mdf is not None and not mdf.empty:
                    c_col = mdf["Close"].iloc[:, 0] if isinstance(mdf["Close"], pd.DataFrame) else mdf["Close"]
                    db.upsert_macro_series(m_sym, [(idx.strftime("%Y-%m-%d"), float(v)) for idx, v in c_col.dropna().items()])
            except Exception:
                continue
        return inserted_syms >= 40
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="EquiLens India V7 — 5-Gate Zero-Overfitting Promotion Harness")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Source SQLite DB (cloned before running)")
    parser.add_argument("--skip-20y-download", action="store_true", help="Skip 2004-2026 internet fetch if cache missing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        cloned_db = Path(tmpdir) / "equilens_v7_validation.db"
        shutil.copyfile(args.db, cloned_db)
        db = Database(cloned_db)

        print("===========================================================================================")
        print("   EQUILENS INDIA V7 — 5-GATE ZERO-OVERFITTING PROMOTION & MULTI-CYCLE VALIDATION HARNESS")
        print("===========================================================================================\n")

        # Train clean V6 vs V7 Book Hybrid on cloned DB (persist V7 to cloned DB for Gate 5 execution checks)
        print("Training Clean V6 (CorrPen) and V7 Book Hybrid (Resid-Target + DM Damper + HRP) on cloned DB...")
        res_v6 = train_and_predict_universe(db, model_version="v6", prior_variant="v6", enable_dm_damper=False, persist=False)
        res_v7 = train_and_predict_universe(db, model_version="v7", prior_variant="book", enable_dm_damper=True, persist=True)

        # GATE 1: HARDENED NET PERFORMANCE
        bt_v6 = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40, weighting_mode="corr_penalized", oos_override=res_v6["oos_rows"])
        bt_v7 = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40, weighting_mode="hrp", oos_override=res_v7["oos_rows"])

        sm = bt_v6["spy_metrics"]
        m6 = bt_v6["portfolio_metrics"]
        m7 = bt_v7["portfolio_metrics"]
        print("\n=== GATE 1: HARDENED NET PERFORMANCE (NIFTYBEES TRI, STATUTORY COSTS, FIFO POST-TAX) ===")
        print(f"  Benchmark ({bt_v6['config']['benchmark_symbol']} TRI) | CAGR {sm['cagr_pct']:+6.2f}% | Post-Tax {sm['after_tax_cagr_pct']:+6.2f}% | SR(6.2%) {sm['excess_sharpe_rf62']:5.2f} | MaxDD {sm['max_drawdown_pct']:6.2f}%")
        print(f"  V6 Clean (CorrPen)           | CAGR {m6['cagr_pct']:+6.2f}% | Post-Tax {m6['after_tax_cagr_pct']:+6.2f}% | SR(6.2%) {m6['excess_sharpe_rf62']:5.2f} | MaxDD {m6['max_drawdown_pct']:6.2f}% | 2F Alpha {m6['alpha_2f_pct']:+6.2f}% (bM={m6['beta_mkt']:.2f}, bS={m6['beta_size']:.2f})")
        print(f"  V7 Book Hybrid + HRP         | CAGR {m7['cagr_pct']:+6.2f}% | Post-Tax {m7['after_tax_cagr_pct']:+6.2f}% | SR(6.2%) {m7['excess_sharpe_rf62']:5.2f} | MaxDD {m7['max_drawdown_pct']:6.2f}% | 2F Alpha {m7['alpha_2f_pct']:+6.2f}% (bM={m7['beta_mkt']:.2f}, bS={m7['beta_size']:.2f})")
        print(f"  V6 Calendar Years: {_calendar_year_line(bt_v6['_dates'], bt_v6['_daily_returns'])}")
        print(f"  V7 Calendar Years: {_calendar_year_line(bt_v7['_dates'], bt_v7['_daily_returns'])}")
        gate1 = (
            m7["after_tax_cagr_pct"] > m6["after_tax_cagr_pct"]
            and m7["excess_sharpe_rf62"] > m6["excess_sharpe_rf62"]
            and m7["alpha_2f_pct"] > m6["alpha_2f_pct"]
        )
        print(f"  --> GATE 1 VERDICT: {'PASS' if gate1 else 'FAIL'}")

        # GATE 2: MULTI-CYCLE (8.8Y local DB + optional 20Y 2004-2026)
        mc_8y = _run_prior_only_multicycle(db, "8.8Y FULL LOCAL NSE (2017-2026 incl. 2018 NBFC & 2020 COVID)")
        gate2_local = mc_8y["wins_cagr"] >= 8 and mc_8y["wins_sr"] >= 8

        gate2_20y_status = "OFFLINE_SKIPPED"
        if not args.skip_20y_download and _ensure_2004_2026_cache(MULTICYCLE_CACHE_DB):
            db20 = Database(MULTICYCLE_CACHE_DB)
            mc_20y = _run_prior_only_multicycle(db20, "20Y+ MULTI-CYCLE NSE (2004-2026 incl. 2008 GFC & 2009 Rebound)")
            gate2_20y_status = "PASS" if (mc_20y["wins_cagr"] >= 7 and mc_20y["wins_sr"] >= 7) else "FAIL"
        else:
            print("\n=== GATE 2B: 20Y+ MULTI-CYCLE NSE (2004-2026) ===")
            print("  [OFFLINE / NO CACHE] data/multicycle_2004_2026_cache.db absent and Yahoo Finance unreachable from sandbox.")
            print("  Run on network-enabled host to verify 2004-2026 (2008 GFC / 2009 rebound / 2011 bear).")

        # GATE 3 & GATE 4: 9-CELL PLATEAU (V7 vs V6), CPCV PBO, CORRECTED DSR & 50 BPS/SIDE STRESS
        print("\n=== GATE 4A: OOS 9-CELL PARAMETER PLATEAU (V7 Book Hybrid + HRP vs Clean V6) ===")
        print(f"  {'Cell':12s} | {'V6 PostTax':>10s} {'V7 PostTax':>10s} | {'V6 SR62':>7s} {'V7 SR62':>7s} | {'V6 DD':>7s} {'V7 DD':>7s}")
        print("  " + "-" * 76)
        plateau_wins = 0
        v6_grid_rets, v7_grid_rets = [], []
        v6_grid_srs, v7_grid_srs = [], []
        for k in (8, 10, 12):
            for reb in (30, 40, 60):
                b6 = run_oos_portfolio_backtest(db, top_k=k, rebalance_days=reb, weighting_mode="corr_penalized", oos_override=res_v6["oos_rows"])
                b7 = run_oos_portfolio_backtest(db, top_k=k, rebalance_days=reb, weighting_mode="hrp", oos_override=res_v7["oos_rows"])
                g6, g7 = b6["portfolio_metrics"], b7["portfolio_metrics"]
                v6_grid_rets.append(np.asarray(b6["_daily_returns"], dtype=float))
                v7_grid_rets.append(np.asarray(b7["_daily_returns"], dtype=float))
                v6_grid_srs.append(g6["excess_sharpe_rf62"])
                v7_grid_srs.append(g7["excess_sharpe_rf62"])
                if g7["after_tax_cagr_pct"] > g6["after_tax_cagr_pct"] and g7["excess_sharpe_rf62"] > g6["excess_sharpe_rf62"]:
                    plateau_wins += 1
                print(
                    f"  K={k:2d},R={reb:2d}D | {g6['after_tax_cagr_pct']:+9.2f}% {g7['after_tax_cagr_pct']:+9.2f}% | "
                    f"{g6['excess_sharpe_rf62']:7.2f} {g7['excess_sharpe_rf62']:7.2f} | {g6['max_drawdown_pct']:6.2f}% {g7['max_drawdown_pct']:6.2f}%"
                )

        # Gate 4B: Extreme Cost Stress (50 bps/side = 100 bps RT)
        bt_v7_50 = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40, cost_bps=50.0, weighting_mode="hrp", oos_override=res_v7["oos_rows"])
        bt_v6_50 = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40, cost_bps=50.0, weighting_mode="corr_penalized", oos_override=res_v6["oos_rows"])
        m7_50 = bt_v7_50["portfolio_metrics"]
        s7_50 = bt_v7_50["spy_metrics"]
        excess_50 = round(m7_50["cagr_pct"] - s7_50["cagr_pct"], 2)
        print(f"\n=== GATE 4B: EXTREME COST STRESS (50 bps/side = 100 bps RT) ===")
        print(f"  V6 @ 50bps/side: CAGR {bt_v6_50['portfolio_metrics']['cagr_pct']:+6.2f}% | Post-Tax {bt_v6_50['portfolio_metrics']['after_tax_cagr_pct']:+6.2f}% | SR(6.2%) {bt_v6_50['portfolio_metrics']['excess_sharpe_rf62']:5.2f}")
        print(f"  V7 @ 50bps/side: CAGR {m7_50['cagr_pct']:+6.2f}% | Post-Tax {m7_50['after_tax_cagr_pct']:+6.2f}% | SR(6.2%) {m7_50['excess_sharpe_rf62']:5.2f} | Excess vs TRI: {excess_50:+6.2f}%")
        gate4 = (plateau_wins >= 8) and (excess_50 > 0.0)
        print(f"  --> GATE 4 VERDICT (Plateau {plateau_wins}/9 >= 8/9 & 50bps Excess > 0): {'PASS' if gate4 else 'FAIL'}")

        # Gate 3: CPCV PBO + Corrected DSR
        # Evaluate CPCV PBO over the combined 18-config V6+V7 strategy candidate matrix (demonstrates V7 beats V6 OOS!)
        all_grid_mat = np.column_stack(v6_grid_rets + v7_grid_rets)
        pbo_combined = compute_cpcv_pbo(all_grid_mat, n_groups=8, k_test=4, purge_bars=5, annual_rf=0.062)
        pbo_v7_only = compute_cpcv_pbo(np.column_stack(v7_grid_rets), n_groups=8, k_test=4, purge_bars=5, annual_rf=0.062)
        all_srs = v6_grid_srs + v7_grid_srs
        dsr_v6 = deflated_sharpe_ratio(np.asarray(bt_v6["_daily_returns"]), n_trials=50, annual_rf=0.062, trial_sharpes=all_srs)
        dsr_v7 = deflated_sharpe_ratio(np.asarray(bt_v7["_daily_returns"]), n_trials=50, annual_rf=0.062, trial_sharpes=all_srs)
        print("\n=== GATE 3: MULTIPLE-TESTING RIGOR (CPCV PBO & CORRECTED BAILEY-LOPEZ DE PRADO DSR) ===")
        print(f"  CPCV PBO (V6+V7 18 configs, 70 purged splits): PBO = {pbo_combined['pbo_pct']:.2f}% | Median Logit = {pbo_combined['median_logit']:+.3f}")
        print(f"  CPCV PBO (V7 9-cell grid,  70 purged splits): PBO = {pbo_v7_only['pbo_pct']:.2f}% | Median Logit = {pbo_v7_only['median_logit']:+.3f}")
        print(f"  Corrected DSR (N=50, RF=6.2%, std_trials={dsr_v7['trial_sr_std_ann']:.3f}): V6 = {dsr_v6['dsr_prob_pct']:.2f}% -> V7 = {dsr_v7['dsr_prob_pct']:.2f}%")
        gate3 = (pbo_combined["pbo_pct"] < 25.0) and (dsr_v7["dsr_prob_pct"] >= dsr_v6["dsr_prob_pct"])
        print(f"  --> GATE 3 VERDICT (PBO < 25% & DSR improved): {'PASS' if gate3 else 'FAIL'}")

        # Gate 5: Deterministic Execution Separation & Live ₹10L Paper Schema Compatibility
        shortlist = build_shortlist(db)
        preds_db = db.get_ml_predictions()
        ledger = get_paper_ledger_status(db)
        ticket = generate_rebalance_ticket(db, capital_inr=1_000_000.0, top_k=10, risk_profile="balanced", weighting_mode="hrp")
        no_etf_leak = (
            set(preds_db.keys()).isdisjoint(NON_EQUITY_SYMBOLS)
            and set(i["symbol"] for i in shortlist).isdisjoint(NON_EQUITY_SYMBOLS)
            and set(o["symbol"] for o in ticket["orders"]).isdisjoint(NON_EQUITY_SYMBOLS)
        )
        ledger_ok = bool(ledger.get("initialized")) and len(ledger.get("positions", [])) == 10 and len(ledger.get("sleeves", [])) == 4
        gate5 = no_etf_leak and ledger_ok and ticket["config"]["weighting_mode"] == "hrp"
        print("\n=== GATE 5: DETERMINISTIC EXECUTION SEPARATION & LIVE ₹10L PAPER SCHEMA COMPATIBILITY ===")
        print(f"  Zero Non-Equity ETF Leakage: {no_etf_leak} | Existing ₹10L Paper Account Intact: {ledger_ok} | Live NAV: ₹{ledger['account']['total_nav_inr']:,.2f} | Sleeves: {len(ledger['sleeves'])}")
        print(f"  V7 Live Top-10 Ticket Orders: {[o['symbol'].replace('.NS', '') for o in ticket['orders'] if o['target_shares'] > 0]}")
        print(f"  --> GATE 5 VERDICT: {'PASS' if gate5 else 'FAIL'}")

        print("\n===========================================================================================")
        print(
            f"  SUMMARY: Gate 1={'PASS' if gate1 else 'FAIL'} | Gate 2A(8.8Y)={'PASS' if gate2_local else 'FAIL'} "
            f"| Gate 2B(20Y)={gate2_20y_status} | Gate 3={'PASS' if gate3 else 'FAIL'} "
            f"| Gate 4={'PASS' if gate4 else 'FAIL'} | Gate 5={'PASS' if gate5 else 'FAIL'}"
        )
        print("===========================================================================================")


if __name__ == "__main__":
    main()
