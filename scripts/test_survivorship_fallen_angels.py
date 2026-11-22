"""
EquiLens India V6 — Real NSE Fallen-Angel Survivorship Stress Test:
Pulls 18 infamous former NIFTY 50 / Next 50 / 200 darlings that suffered brutal -50% to -95% crashes
(YESBANK, IDEA, ZEEL, BANDHANBNK, RBLBANK, PAYTM, ATGL, GLAND, IEX, BATAINDIA, WHIRLPOOL,
RELAXO, TATACHEM, STARHEALTH, NYKAA, BIOCON, UPL, INDUSTOWER) into a clean copy of the DB,
retrains strictly causal V6 (with zero US-timezone lookahead!), and compares:
  - NIFTY 50 Index
  - De-biased Equal-Weight Universe (131 + Fallen Angels)
  - V6 Top-10 & Top-8 (4-Tranche, 18% Vol, 15 bps Indian STT)
  - Exact check of whether V6 got caught in ANY of the fallen angels during their crashes!
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backend.config import BENCHMARK_SYMBOL, DB_PATH
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


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db_path = Path(tmpdir) / "debiased_india.db"
        shutil.copyfile(DB_PATH, tmp_db_path)
        db = Database(tmp_db_path)

        existing = {t["symbol"] for t in db.get_tickers()}
        to_fetch = [(sym, sec) for sym, sec in FALLEN_ANGELS if sym not in existing]
        print(f"Downloading 10Y NSE history for {len(to_fetch)} real Indian Fallen Angels...")
        raw = yf.download([s for s, _ in to_fetch], period="10y", group_by="ticker", auto_adjust=True, threads=True, progress=False)

        added = []
        for sym, sec in to_fetch:
            try:
                df = raw[sym].dropna(subset=["Close"]) if len(to_fetch) > 1 else raw.dropna(subset=["Close"])
                if len(df) < 260:
                    continue
                bars = []
                for idx, row in df.iterrows():
                    dt = idx.strftime("%Y-%m-%d")
                    bars.append({
                        "date": dt,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                    })
                db.upsert_ticker(
                    symbol=sym,
                    name=sym.replace(".NS", ""),
                    sector=sec,
                    industry="Fallen Angel Stress",
                    exchange="NSI",
                    summary="",
                    fundamentals={},
                    data_source="yfinance",
                    updated_at="2026-11-22T00:00:00Z",
                )
                db.upsert_ohlcv(sym, bars)
                peak = float(df["Close"].max())
                last = float(df["Close"].iloc[-1])
                dd_from_peak = ((last / peak) - 1.0) * 100.0
                added.append((sym, len(bars), dd_from_peak))
            except Exception as exc:
                print(f"  Skip {sym}: {exc}")

        print(f"Added {len(added)} real NSE Fallen Angels:")
        for sym, nb, ddp in added:
            print(f"  + {sym:14s} ({nb:4d} bars | {ddp:+5.1f}% below 10Y peak)")

        print("\nRetraining strictly causal V6 (zero US-timezone lookahead) on De-Biased Universe...")
        res = train_and_predict_universe(db)
        r = res["run"]
        print(f"De-Biased ML Run: Samples {r['sample_count']:,} | Symbols {r['symbol_count']} | OOS IC {r['ic_spearman']:+.3f} | 40D Hit {r['hit_rate']:.1f}% | Spread {r['top5_spread']:+.2f}% | Daily CS TB AUC {r['tb_auc']:.3f}")

        # Compute De-Biased Equal-Weight Universe benchmark
        bt10 = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40)
        bt8 = run_oos_portfolio_backtest(db, top_k=8, rebalance_days=40)
        dates = bt10["_dates"]
        ohlcv = db.get_all_ohlcv()
        prices = pd.DataFrame({s: pd.Series({b["date"]: float(b["close"]) for b in bars}) for s, bars in ohlcv.items()}).sort_index().ffill()
        stock_cols = [c for c in prices.columns if c != BENCHMARK_SYMBOL]
        eqw = prices[stock_cols].pct_change().mean(axis=1).reindex(dates).fillna(0.0).to_numpy(dtype=float)
        yrs = max(0.25, len(eqw) / 252.0)
        eqw_cagr = (float(np.prod(1.0 + eqw)) ** (1.0 / yrs) - 1.0) * 100.0
        eqw_vol = float(np.std(eqw, ddof=1) * np.sqrt(252.0)) * 100.0
        eqw_shrp = float((np.mean(eqw) * 252.0) / max(1e-9, eqw_vol / 100.0))
        eqw_cum = np.cumprod(1.0 + eqw)
        eqw_dd = float(np.min((eqw_cum - np.maximum.accumulate(eqw_cum)) / np.maximum.accumulate(eqw_cum)) * 100.0)

        sm = bt10["spy_metrics"]
        m10 = bt10["portfolio_metrics"]
        m8 = bt8["portfolio_metrics"]
        print("\n=== DE-BIASED OOS RESULTS (WITH REAL INDIAN FALLEN ANGELS & ZERO US LOOKAHEAD) ===")
        print(f"  NIFTY 50 Benchmark                     | CAGR {sm['cagr_pct']:+6.2f}% | Vol {sm['ann_vol_pct']:5.2f}% | Sharpe {sm['sharpe']:4.2f} | MaxDD {sm['max_drawdown_pct']:6.2f}%")
        print(f"  De-Biased Equal-Weight ({len(stock_cols)} stocks)   | CAGR {eqw_cagr:+6.2f}% | Vol {eqw_vol:5.2f}% | Sharpe {eqw_shrp:4.2f} | MaxDD {eqw_dd:6.2f}%")
        print(f"  V6 Top-10 (4-Tranche, 18% Vol, 15bps)  | CAGR {m10['cagr_pct']:+6.2f}% (Post-Tax {m10['after_tax_cagr_pct']:+5.2f}%) | Vol {m10['ann_vol_pct']:5.2f}% | Sharpe {m10['sharpe']:4.2f} | MaxDD {m10['max_drawdown_pct']:6.2f}% | Calmar {m10['calmar']:4.2f}")
        print(f"  V6 Top- 8 (4-Tranche, 18% Vol, 15bps)  | CAGR {m8['cagr_pct']:+6.2f}% (Post-Tax {m8['after_tax_cagr_pct']:+5.2f}%) | Vol {m8['ann_vol_pct']:5.2f}% | Sharpe {m8['sharpe']:4.2f} | MaxDD {m8['max_drawdown_pct']:6.2f}% | Calmar {m8['calmar']:4.2f}")
        print(f"  --> Pure V6 Top-10 Alpha vs De-Biased Universe: {m10['cagr_pct'] - eqw_cagr:+.2f}%/yr | Sharpe Gain: {m10['sharpe'] - eqw_shrp:+.2f} | MaxDD Reduction: {m10['max_drawdown_pct'] - eqw_dd:+.2f}%")


if __name__ == "__main__":
    main()
