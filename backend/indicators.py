"""Vectorized technical indicators and actionable chart setup detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_indicator_frame(
    bars: list[dict[str, Any]],
    benchmark_bars: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compute technical indicators and causal features on a DataFrame indexed by date."""
    df = pd.DataFrame(bars).copy()
    if df.empty:
        return df

    df["date"] = df["date"].astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # Moving averages
    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()
    df["ema12"] = close.ewm(span=12, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()

    # MACD (12, 26, 9)
    df["macd"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_norm"] = df["macd_hist"] / close.replace(0, np.nan)

    # RSI (14) via Wilder smoothing
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi14"] = df["rsi14"].where(avg_loss != 0, 100.0)

    # Bollinger Bands (20, 2)
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["sma20"] + 2.0 * bb_std
    df["bb_lower"] = df["sma20"] - 2.0 * bb_std
    bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct_b"] = (close - df["bb_lower"]) / bb_width

    # ATR (14) & Realized Volatility
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["atr_pct"] = df["atr14"] / close.replace(0, np.nan)

    log_ret = np.log(close / close.shift(1))
    df["vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["vol_60d"] = log_ret.rolling(60).std() * np.sqrt(252)

    # Multi-horizon returns
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_21d"] = close.pct_change(21)
    df["ret_63d"] = close.pct_change(63)
    df["ret_126d"] = close.pct_change(126)
    df["ret_252d"] = close.pct_change(252)

    # Distance from moving averages & 52W range
    df["dist_sma20"] = (close / df["sma20"]) - 1.0
    df["dist_sma50"] = (close / df["sma50"]) - 1.0
    df["dist_sma200"] = (close / df["sma200"]) - 1.0
    df["sma20_50_ratio"] = (df["sma20"] / df["sma50"]) - 1.0

    high_252 = high.rolling(252, min_periods=20).max()
    low_252 = low.rolling(252, min_periods=20).min()
    df["high_52w"] = high_252
    df["low_52w"] = low_252
    df["pos_52w"] = (close - low_252) / (high_252 - low_252).replace(0, np.nan)

    # Volume surge & rolling VWAP (20D)
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    df["vol_ratio_5_20"] = vol_ma5 / vol_ma20.replace(0, np.nan)
    tp = (high + low + close) / 3.0
    df["vwap20"] = (tp * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    # Relative Strength vs Benchmark (SPY)
    if benchmark_bars:
        bdf = pd.DataFrame(benchmark_bars)[["date", "close"]].rename(columns={"close": "bench_close"})
        bdf["date"] = bdf["date"].astype(str)
        merged = df[["date"]].merge(bdf, on="date", how="left")
        bclose = merged["bench_close"].ffill()
        df["bench_ret_21d"] = bclose.pct_change(21)
        df["bench_ret_63d"] = bclose.pct_change(63)
        df["rs_spy_21d"] = df["ret_21d"] - df["bench_ret_21d"]
        df["rs_spy_63d"] = df["ret_63d"] - df["bench_ret_63d"]
    else:
        df["rs_spy_21d"] = 0.0
        df["rs_spy_63d"] = 0.0

    return df


def detect_chart_setups(df: pd.DataFrame) -> list[dict[str, str]]:
    """Detect concrete technical setups on the latest bar."""
    if df.empty or len(df) < 55:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2]
    setups: list[dict[str, str]] = []

    close = float(last["close"])
    sma20 = last.get("sma20")
    sma50 = last.get("sma50")
    sma200 = last.get("sma200")
    rsi = last.get("rsi14")
    vol_ratio = last.get("vol_ratio_5_20")
    rs63 = last.get("rs_spy_63d")

    # 1. 20D High Breakout with Volume
    prev_20d_high = float(df["high"].iloc[-21:-1].max())
    if close >= prev_20d_high * 0.995 and pd.notna(vol_ratio) and vol_ratio >= 1.15:
        setups.append(
            {
                "id": "breakout_20d",
                "label": "20D Breakout + Vol",
                "tone": "bullish",
                "detail": f"Price breaking above 20-day high (₹{prev_20d_high:,.2f}) with {vol_ratio:.2f}x volume.",
            }
        )

    # 2. Golden Cross (SMA 50 above SMA 200, or fresh crossover)
    if pd.notna(sma50) and pd.notna(sma200):
        sma50_10ago = df["sma50"].iloc[-10]
        sma200_10ago = df["sma200"].iloc[-10]
        if sma50 > sma200 and pd.notna(sma50_10ago) and pd.notna(sma200_10ago) and sma50_10ago <= sma200_10ago:
            setups.append(
                {
                    "id": "golden_cross",
                    "label": "Fresh Golden Cross",
                    "tone": "bullish",
                    "detail": "50-day SMA recently crossed above 200-day SMA.",
                }
            )
        elif close > sma50 > sma200:
            setups.append(
                {
                    "id": "stage2_uptrend",
                    "label": "Stage-2 Uptrend",
                    "tone": "bullish",
                    "detail": "Price > SMA 50 > SMA 200 trend alignment.",
                }
            )

    # 3. Healthy Uptrend Pullback
    if (
        pd.notna(sma50)
        and pd.notna(sma20)
        and close > sma50
        and close <= sma20 * 1.01
        and pd.notna(rsi)
        and 36 <= rsi <= 52
    ):
        setups.append(
            {
                "id": "uptrend_pullback",
                "label": "Uptrend Pullback",
                "tone": "bullish",
                "detail": f"Constructive dip near SMA 20 above SMA 50 with reset RSI ({rsi:.1f}).",
            }
        )

    # 4. Oversold Mean-Reversion Watch
    if pd.notna(rsi) and rsi <= 34:
        setups.append(
            {
                "id": "oversold_watch",
                "label": "Oversold RSI",
                "tone": "neutral",
                "detail": f"RSI(14) at {rsi:.1f} signals stretched selling / potential bounce zone.",
            }
        )

    # 5. MACD Bullish Turn
    if pd.notna(last.get("macd_hist")) and pd.notna(prev.get("macd_hist")):
        if last["macd_hist"] > 0 and prev["macd_hist"] <= 0:
            setups.append(
                {
                    "id": "macd_cross",
                    "label": "MACD Bull Cross",
                    "tone": "bullish",
                    "detail": "MACD histogram just flipped positive.",
                }
            )

    # 6. Relative Strength Leader
    if pd.notna(rs63) and rs63 >= 0.08:
        setups.append(
            {
                "id": "rs_leader",
                "label": "RS Leader vs NIFTY",
                "tone": "bullish",
                "detail": f"Outperforming NIFTY 50 by +{rs63 * 100:.1f}% over the last quarter.",
            }
        )

    # 7. Overextended Risk Flag
    if pd.notna(rsi) and rsi >= 72:
        setups.append(
            {
                "id": "overbought",
                "label": "Overbought RSI",
                "tone": "warning",
                "detail": f"RSI(14) at {rsi:.1f} — elevated near-term pullback risk.",
            }
        )

    return setups


def serialize_indicator_series(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize OHLCV + key indicators to JSON-safe list of dicts."""
    cols = [
        "date", "open", "high", "low", "close", "volume",
        "sma20", "sma50", "sma200", "ema12", "bb_upper", "bb_lower",
        "vwap20", "rsi14", "macd", "macd_signal", "macd_hist",
    ]
    out = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for c in cols:
            if c not in row:
                rec[c] = None
                continue
            v = row[c]
            if c == "date":
                rec[c] = str(v)
            elif c == "volume":
                rec[c] = int(v) if pd.notna(v) else 0
            else:
                rec[c] = round(float(v), 4) if pd.notna(v) else None
        out.append(rec)
    return out
