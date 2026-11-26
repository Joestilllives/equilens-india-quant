"""
EquiLens India V6 vs V7 — Real NSE Fallen-Angel Survivorship Stress Test:
Pulls 18 infamous former NIFTY 50 / Next 50 / 200 darlings that suffered brutal -50% to -95% crashes
(YESBANK, IDEA, ZEEL, BANDHANBNK, RBLBANK, PAYTM, ATGL, GLAND, IEX, BATAINDIA, WHIRLPOOL,
RELAXO, TATACHEM, STARHEALTH, NYKAA, BIOCON, UPL, PEL) into a clean copy of the DB (with local
JSON cache at `data/fallen_angels_cache.json` so subsequent runs work 100% offline), retrains
strictly causal V6 vs V7 (zero US-timezone lookahead, zero non-equity ETF leakage), and compares:
  - NIFTYBEES.NS TRI Benchmark
  - De-biased Equal-Weight Universe
  - V6 Clean (CorrPen) vs V7 Book Hybrid + HRP (FIFO Post-Tax, SR 6.2%, 2-Factor Pure Alpha)
  - Double-orthogonalized Amihud illiquidity diagnostic on the Fallen-Angel universe.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yfinance as yf

from backend.config import DATA_DIR, DB_PATH, NON_EQUITY_SYMBOLS
from backend.database import Database
from backend.ml_predictor import train_and_predict_universe
from backend.portfolio_backtester import run_oos_portfolio_backtest

FALLEN_ANGELS = [
    ("YESBANK.NS", "Financial Services"),
    ("IDEA.NS", "Communication Services"),
    ("ZEEL.NS", "Communication Services"),
    ("BANDHANBNK.NS", "Financial Services"),
    ("RBLBANK.NS", "Financial Services"),
    ("PAYTM.NS", "Financial Services"),
    ("ATGL.NS", "Utilities"),
    ("GLAND.NS", "Healthcare"),
    ("IEX.NS", "Financial Services"),
    ("BATAINDIA.NS", "Consumer Cyclical"),
    ("WHIRLPOOL.NS", "Consumer Cyclical"),
    ("RELAXO.NS", "Consumer Cyclical"),
    ("TATACHEM.NS", "Basic Materials"),
    ("STARHEALTH.NS", "Financial Services"),
    ("NYKAA.NS", "Consumer Cyclical"),
    ("BIOCON.NS", "Healthcare"),
    ("UPL.NS", "Basic Materials"),
    ("PEL.NS", "Financial Services"),
]

CACHE_FILE = DATA_DIR / "fallen_angels_cache.json"


def _load_or_fetch_fallen_angels(existing: set[str], cache_path: Path) -> tuple[dict[str, dict], bool]:
    """Load cached Fallen Angels bars if present; otherwise fetch via Yahoo Finance and cache."""
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and len(cached) >= 10:
                return cached, True
        except Exception:
            pass

    to_fetch = [(sym, sec) for sym, sec in FALLEN_ANGELS if sym not in existing]
    if not to_fetch:
        return {}, True

    fetched: dict[str, dict] = {}
    try:
        raw = yf.download(
            [s for s, _ in to_fetch],
            period="10y",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return {}, False
        for sym, sec in to_fetch:
            try:
                df = raw[sym].dropna(subset=["Close"]) if len(to_fetch) > 1 else raw.dropna(subset=["Close"])
                if len(df) < 260:
                    continue
                bars = [
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                    }
                    for idx, row in df.iterrows()
                ]
                fetched[sym] = {"sector": sec, "bars": bars}
            except Exception:
                continue
        if len(fetched) >= 10:
            cache_path.write_text(json.dumps(fetched), encoding="utf-8")
            return fetched, True
        return fetched, False
    except Exception as exc:
        print(f"  [OFFLINE / NETWORK BLOCKED] Yahoo Finance Fallen Angels fetch failed: {exc}")
        return {}, False


def _evaluate_amihud_double_orthogonalized(db: Database) -> dict[str, float]:
    """
    Diagnostic test: Double-orthogonalized Amihud illiquidity (orthogonalized against log dollar volume
    and 252D volatility) on the de-biased Fallen-Angel universe. Must earn positive IC on Fallen Angels
    before ever considering Amihud for production!
    """
    ohlcv = db.get_all_ohlcv()
    frames = []
    for sym, bars in ohlcv.items():
        if sym in NON_EQUITY_SYMBOLS or len(bars) < 300:
            continue
        df = pd.DataFrame(bars)[["date", "close", "volume"]].sort_values("date").reset_index(drop=True)
        c = df["close"].astype(float)
        v = df["volume"].astype(float).clip(lower=0.0)
        ret = c.pct_change().abs()
        dv = (c * v).replace(0, np.nan)
        amihud = (ret / dv).rolling(63, min_periods=30).mean()
        log_dv = np.log1p(dv.rolling(63, min_periods=30).mean())
        vol252 = np.log(c / c.shift(1)).rolling(252, min_periods=63).std() * np.sqrt(252.0)
        fwd40 = c.shift(-40) / c - 1.0
        frames.append(pd.DataFrame({
            "date": df["date"], "symbol": sym,
            "amihud": np.log1p(amihud * 1e9), "log_dv": log_dv, "vol252": vol252, "fwd40": fwd40
        }))
    if not frames:
        return {"mean_cs_ic": 0.0, "t_stat": 0.0, "n_dates": 0}
    panel = pd.concat(frames, ignore_index=True).dropna()
    ics = []
    for dt in sorted(panel["date"].unique())[::20]:
        sub = panel[panel["date"] == dt]
        if len(sub) < 30:
            continue
        X = np.column_stack([np.ones(len(sub)), sub["log_dv"].to_numpy(), sub["vol252"].to_numpy()])
        y = sub["amihud"].to_numpy()
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        orth_amihud = y - X @ coef
        ic = spearmanr(orth_amihud, sub["fwd40"].to_numpy()).statistic
        if np.isfinite(ic):
            ics.append(float(ic))
    if len(ics) < 5:
        return {"mean_cs_ic": 0.0, "t_stat": 0.0, "n_dates": len(ics)}
    arr = np.asarray(ics, dtype=float)
    t_stat = float(np.mean(arr) / (np.std(arr, ddof=1) / np.sqrt(len(arr)))) if np.std(arr, ddof=1) > 1e-12 else 0.0
    return {"mean_cs_ic": round(float(np.mean(arr)), 4), "t_stat": round(t_stat, 2), "n_dates": len(ics)}


def main() -> None:
    parser = argparse.ArgumentParser(description="EquiLens India V6 vs V7 Fallen-Angel Survivorship Stress Test")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Source DB path (copied to temp dir)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db_path = Path(tmpdir) / "debiased_india.db"
        shutil.copyfile(args.db, tmp_db_path)
        db = Database(tmp_db_path)

        existing = {t["symbol"] for t in db.get_tickers()}
        print("Loading / fetching 10Y NSE history for 18 real Indian Fallen Angels...")
        fa_data, ok = _load_or_fetch_fallen_angels(existing, CACHE_FILE)
        if not ok or len(fa_data) < 5:
            print("  RESULT: SKIPPED / NETWORK BLOCKED — Cannot download 18 Fallen Angels without internet access or cached data/fallen_angels_cache.json.")
            print("  Re-run on a network-enabled machine to populate data/fallen_angels_cache.json and verify Gate 2 Fallen Angels.")
            return

        added = []
        for sym, payload in fa_data.items():
            bars = payload["bars"]
            if len(bars) < 260:
                continue
            db.upsert_ticker(
                symbol=sym,
                name=sym.replace(".NS", ""),
                sector=payload["sector"],
                industry="Fallen Angel Stress",
                exchange="NSI",
                summary="",
                fundamentals={},
                data_source="yfinance",
                updated_at="2026-11-26T00:00:00Z",
            )
            db.upsert_ohlcv(sym, bars)
            closes = [b["close"] for b in bars]
            dd_from_peak = ((closes[-1] / max(closes)) - 1.0) * 100.0
            added.append((sym, len(bars), dd_from_peak))

        print(f"Added {len(added)} real NSE Fallen Angels:")
        for sym, nb, ddp in added:
            print(f"  + {sym:14s} ({nb:4d} bars | {ddp:+5.1f}% below 10Y peak)")

        print("\nRetraining strictly causal V6 vs V7 on De-Biased Fallen-Angel Universe...")
        res_v6 = train_and_predict_universe(db, model_version="v6", prior_variant="v6", enable_dm_damper=False, persist=False)
        res_v7 = train_and_predict_universe(db, model_version="v7", prior_variant="book", enable_dm_damper=True, persist=False)

        bt_v6 = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40, weighting_mode="corr_penalized", oos_override=res_v6["oos_rows"])
        bt_v7 = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40, weighting_mode="hrp", oos_override=res_v7["oos_rows"])

        sm = bt_v6["spy_metrics"]
        m6 = bt_v6["portfolio_metrics"]
        m7 = bt_v7["portfolio_metrics"]
        print("\n=== DE-BIASED FALLEN-ANGEL OOS RESULTS (V6 vs V7, NIFTYBEES TRI & FIFO POST-TAX) ===")
        print(f"  Benchmark ({bt_v6['config']['benchmark_symbol']} TRI) | CAGR {sm['cagr_pct']:+6.2f}% (Post-Tax {sm['after_tax_cagr_pct']:+5.2f}%) | SR(6.2%) {sm['excess_sharpe_rf62']:5.2f} | MaxDD {sm['max_drawdown_pct']:6.2f}%")
        print(f"  V6 Clean (CorrPen)           | CAGR {m6['cagr_pct']:+6.2f}% (Post-Tax {m6['after_tax_cagr_pct']:+5.2f}%) | SR(6.2%) {m6['excess_sharpe_rf62']:5.2f} | MaxDD {m6['max_drawdown_pct']:6.2f}% | 2F Alpha {m6['alpha_2f_pct']:+5.2f}%")
        print(f"  V7 Book Hybrid + HRP         | CAGR {m7['cagr_pct']:+6.2f}% (Post-Tax {m7['after_tax_cagr_pct']:+5.2f}%) | SR(6.2%) {m7['excess_sharpe_rf62']:5.2f} | MaxDD {m7['max_drawdown_pct']:6.2f}% | 2F Alpha {m7['alpha_2f_pct']:+5.2f}%")
        passed = (m7["after_tax_cagr_pct"] > m6["after_tax_cagr_pct"]) and (m7["excess_sharpe_rf62"] > m6["excess_sharpe_rf62"])
        print(f"  --> Gate 2 (Fallen Angels V7 > V6): {'PASS' if passed else 'FAIL'}")

        amihud_diag = _evaluate_amihud_double_orthogonalized(db)
        print(f"\n  [Diagnostic] Double-Orthogonalized Amihud Illiquidity on Fallen Angels: Mean CS IC = {amihud_diag['mean_cs_ic']:+.4f} (t={amihud_diag['t_stat']:+.2f}, n={amihud_diag['n_dates']})")


if __name__ == "__main__":
    main()
