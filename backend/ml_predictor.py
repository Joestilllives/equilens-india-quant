"""
EquiLens India V7 Walk-Forward Quantitative ML Engine (Canonical Finance & Economics Architecture):
  - Strictly Causal Canonical NSE Signals (Zero Lookahead):
      * Blitz, Huij & Martens (2011) CAPM Residual Momentum (12-1M & 6-1M)
      * Novy-Marx (2012) Intermediate-Horizon Momentum (12-7M)
      * Grinblatt & Han (2005) Capital Gains Overhang (CGO 126D via turnover-weighted VWAP)
      * Frazzini & Pedersen (2014) Betting Against Beta (-Vasicek shrunk 252D beta)
      * Daniel & Moskowitz (2016) Bear-Rebound Momentum Crash Damper (NIFTY 252D < 0 & VIX >= 19)
      * Pollet & Wilson (2010) Cross-Stock Average Pairwise Correlation (63D telemetry)
      * Proven V6 survivors: FIP Smoothness, 1Y & Downside Vol, Sector-Neutral 3M Alpha,
        MAX Retail-Lottery Avoidance, Overnight Institutional Sentiment, FX/S&P 500 Macro Tailwinds
  - 50% Canonical Multi-Factor Prior + 50% Walk-Forward 40D Vasicek Residual-Sharpe ML
    (Ridge on strictly cross-sectional features + shallow HistGradientBoosting with macro interactions),
    evaluated across 5 expanding chronological folds with strict 40-day embargo.
  - Strict Non-Equity Hygiene: ^NSEI, NIFTYBEES.NS, and GOLDBEES.NS are excluded everywhere.
  - Triple-Barrier Payoff Classifier + ATR Target/Stop + 60D Probability Cones.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BENCHMARK_SYMBOL,
    BRENT_MACRO_SYMBOL,
    FX_MACRO_SYMBOL,
    GOLD_MACRO_SYMBOL,
    ML_HORIZON_DAYS,
    NON_EQUITY_SYMBOLS,
    SP500_MACRO_SYMBOL,
    TB_DOWN_ATR_MULT,
    TB_HORIZON_DAYS,
    TB_UP_ATR_MULT,
    VOL_INDEX_SYMBOL,
)
from .database import Database
from .indicators import compute_indicator_frame

# Legacy V6 19-feature list retained for clean side-by-side V6 vs V7 ablations
V6_FEATURE_COLS: list[str] = [
    "z_mom_6m1m_voladj",
    "z_mom_12m1m_voladj",
    "z_fip_smooth_6m",
    "z_prox_52w_high",
    "z_vol_252d",
    "z_down_vol_126d",
    "z_sec_rel_63d",
    "z_sec_rel_126d",
    "z_ret_21d",
    "z_rsi14",
    "z_vol_ratio_5_20",
    "vix_level",
    "bench_dist_200",
    "nse_breadth_50d",
    "z_macro_tailwind_brent",
    "z_macro_tailwind_fx",
    "z_macro_tailwind_sp500",
    "z_max_ret_21d",
    "z_overnight_sent_63d",
]

# V7.5-Core Canonical Book Feature Set (pruned collinear/noise: z_prox_52w_high, z_sec_rel_126d,
# z_vol_ratio_5_20, z_macro_tailwind_brent, and RBI-managed z_macro_tailwind_fx;
# retains CAPM Residual Mom, 12-7M, CGO, -Vasicek Beta & proven V6 survivors)
FEATURE_COLS: list[str] = [
    "z_resid_12m1m",
    "z_resid_6m1m",
    "z_mom_12_7",
    "z_cgo_126",
    "z_neg_beta",
    "z_mom_6m1m_voladj",
    "z_mom_12m1m_voladj",
    "z_fip_smooth_6m",
    "z_vol_252d",
    "z_down_vol_126d",
    "z_sec_rel_63d",
    "z_ret_21d",
    "z_rsi14",
    "vix_level",
    "bench_dist_200",
    "nse_breadth_50d",
    "z_macro_tailwind_sp500",
    "z_max_ret_21d",
    "z_overnight_sent_63d",
]

FEATURE_LABELS: dict[str, str] = {
    "z_resid_12m1m": "CAPM Residual Momentum (12-1M)",
    "z_resid_6m1m": "CAPM Residual Momentum (6-1M)",
    "z_mom_12_7": "Intermediate Momentum (12-7M)",
    "z_cgo_126": "Capital Gains Overhang (126D)",
    "z_neg_beta": "Betting Against Beta (-Vasicek β)",
    "z_mom_6m1m_voladj": "NSE 6-1M Vol-Adj Momentum",
    "z_mom_12m1m_voladj": "NSE 12-1M Vol-Adj Momentum",
    "z_fip_smooth_6m": "Frog-in-the-Pan Smoothness",
    "z_prox_52w_high": "52W High Proximity",
    "z_vol_252d": "1Y Annualized Volatility",
    "z_down_vol_126d": "6M Downside Volatility",
    "z_sec_rel_63d": "Sector-Neutral 3M Alpha",
    "z_sec_rel_126d": "Sector-Neutral 6M Alpha",
    "z_ret_21d": "1M Short-Term Reversal",
    "z_rsi14": "Cross-Sec RSI (14)",
    "z_vol_ratio_5_20": "Volume Surge (5D/20D)",
    "vix_level": "India VIX Regime",
    "bench_dist_200": "NIFTY 50 vs 200-DMA",
    "nse_breadth_50d": "NSE Breadth (>50D)",
    "z_macro_tailwind_brent": "Brent Crude Stock Beta Tailwind",
    "z_macro_tailwind_fx": "USD/INR Stock Beta Tailwind",
    "z_macro_tailwind_sp500": "Global S&P 500 Beta Tailwind",
    "z_max_ret_21d": "Retail Lottery Spike (MAX 21D)",
    "z_overnight_sent_63d": "Overnight Institutional Sentiment",
}

_DATE_MACRO_COLS = {"vix_level", "bench_dist_200", "nse_breadth_50d"}
CS_FEATURE_INDICES: list[int] = [i for i, c in enumerate(FEATURE_COLS) if c not in _DATE_MACRO_COLS]

_MACRO_LOCK = threading.Lock()
_MACRO_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def invalidate_macro_cache() -> None:
    with _MACRO_LOCK:
        _MACRO_CACHE.clear()


def compute_triple_barrier_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    horizon: int = TB_HORIZON_DAYS,
    up_mult: float = TB_UP_ATR_MULT,
    down_mult: float = TB_DOWN_ATR_MULT,
) -> np.ndarray:
    """1 if price hits +up_mult*ATR before -down_mult*ATR within `horizon` bars; NaN on last `horizon` bars."""
    n = len(closes)
    labels = np.full(n, np.nan, dtype=float)
    for i in range(0, n - horizon):
        atr = atrs[i]
        c0 = closes[i]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(c0) or c0 <= 0:
            continue
        up_b = c0 + up_mult * atr
        dn_b = c0 - down_mult * atr
        hit = np.nan
        for k in range(1, horizon + 1):
            if highs[i + k] >= up_b and lows[i + k] <= dn_b:
                hit = 1.0 if closes[i + k] >= c0 else 0.0
                break
            if highs[i + k] >= up_b:
                hit = 1.0
                break
            if lows[i + k] <= dn_b:
                hit = 0.0
                break
        if np.isnan(hit):
            hit = 1.0 if closes[i + horizon] > c0 else 0.0
        labels[i] = hit
    return labels


def _next_trading_dates(last_date_str: str, steps: list[int]) -> list[str]:
    cur = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    max_step = max(steps)
    future: list[str] = []
    while len(future) < max_step:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            future.append(cur.isoformat())
    return [future[s - 1] for s in steps]


def build_probability_cone(
    last_date: str,
    last_price: float,
    expected_horizon_return: float,
    vol_ann: float,
    horizon: int = ML_HORIZON_DAYS,
) -> list[dict[str, Any]]:
    """Compute log-normal 10/25/50/75/90 percentile price cone for next 60 trading days."""
    steps = [0, 5, 10, 15, 20, 30, 45, 60]
    future_dates = [last_date] + _next_trading_dates(last_date, steps[1:])
    mu_ann = float(np.clip(expected_horizon_return * (252.0 / max(10, horizon)), -0.35, 0.45))
    sigma = float(np.clip(vol_ann, 0.10, 0.95))

    z10, z25, z75, z90 = norm.ppf(0.10), norm.ppf(0.25), norm.ppf(0.75), norm.ppf(0.90)
    cone = []
    for step, dt in zip(steps, future_dates):
        t = step / 252.0
        if step == 0:
            cone.append(
                {
                    "day": 0, "date": dt,
                    "p10": round(last_price, 2), "p25": round(last_price, 2),
                    "p50": round(last_price, 2), "p75": round(last_price, 2),
                    "p90": round(last_price, 2),
                }
            )
            continue
        drift = (mu_ann - 0.5 * sigma * sigma) * t
        sd_t = sigma * np.sqrt(t)
        cone.append(
            {
                "day": step,
                "date": dt,
                "p10": round(float(last_price * np.exp(drift + z10 * sd_t)), 2),
                "p25": round(float(last_price * np.exp(drift + z25 * sd_t)), 2),
                "p50": round(float(last_price * np.exp(drift)), 2),
                "p75": round(float(last_price * np.exp(drift + z75 * sd_t)), 2),
                "p90": round(float(last_price * np.exp(drift + z90 * sd_t)), 2),
            }
        )
    return cone


def compute_pollet_wilson_avg_corr(
    ohlcv_map: dict[str, list[dict[str, Any]]],
    window: int = 63,
    as_of_date: str | None = None,
) -> float:
    """
    Strictly causal Pollet & Wilson (2010) cross-stock average pairwise correlation over trailing `window` days.
    Used strictly as a forward volatility/risk telemetry gauge (NOT a panic-to-cash trigger).
    """
    series_dict: dict[str, pd.Series] = {}
    for sym, bars in ohlcv_map.items():
        if sym in NON_EQUITY_SYMBOLS or len(bars) < max(30, window + 5):
            continue
        s = pd.Series({str(b["date"])[:10]: float(b["close"]) for b in bars}).sort_index()
        if as_of_date is not None:
            s = s[s.index <= as_of_date]
        if len(s) >= max(30, window + 2):
            series_dict[sym] = s.tail(window + 2)
    if len(series_dict) < 5:
        return 0.25
    px = pd.DataFrame(series_dict).sort_index()
    rets = px.pct_change().tail(window)
    # Retain columns with sufficient non-null observations and non-zero variance
    valid_cols = [c for c in rets.columns if rets[c].notna().sum() >= max(20, window // 2) and float(rets[c].std() or 0.0) > 1e-8]
    if len(valid_cols) < 5:
        return 0.25
    cmat = rets[valid_cols].corr().to_numpy(dtype=float, copy=True)
    np.fill_diagonal(cmat, np.nan)
    val = float(np.nanmean(cmat))
    return round(val, 4) if np.isfinite(val) else 0.25


def compute_current_macro_regime(db: Database) -> dict[str, Any]:
    """Compute live 6-Gauge Indian Sentiment & Intermarket Macro regime snapshot + V7 telemetry."""
    cache_key = str(db.db_path.resolve())
    now = time.monotonic()
    with _MACRO_LOCK:
        cached = _MACRO_CACHE.get(cache_key)
        if cached and (now - cached[0]) < 300.0:
            return cached[1]

    macro = db.get_macro_series()
    ohlcv_map = db.get_all_ohlcv()
    bench_bars = ohlcv_map.get(BENCHMARK_SYMBOL, [])

    vix_map = macro.get(VOL_INDEX_SYMBOL, {})
    fx_map = macro.get(FX_MACRO_SYMBOL, {})
    brent_map = macro.get(BRENT_MACRO_SYMBOL, {})
    gold_map = macro.get(GOLD_MACRO_SYMBOL, {})

    vix_dates = sorted(vix_map.keys())
    fx_dates = sorted(fx_map.keys())
    brent_dates = sorted(brent_map.keys())
    vix_now = vix_map[vix_dates[-1]] if vix_dates else 15.0
    usdinr_now = fx_map[fx_dates[-1]] if fx_dates else 85.0
    brent_now = brent_map[brent_dates[-1]] if brent_dates else None

    bench_dist_200 = 0.0
    bench_above_200 = True
    bench_ret_252d_pct = 0.0
    gold_nifty_63d_pct = 0.0
    if len(bench_bars) >= 200:
        closes = np.array([b["close"] for b in bench_bars], dtype=float)
        sma200 = float(np.mean(closes[-200:]))
        bench_dist_200 = ((closes[-1] / sma200) - 1.0) * 100.0
        bench_above_200 = bool(closes[-1] >= sma200)
        if len(closes) >= 253 and closes[-253] > 0:
            bench_ret_252d_pct = ((closes[-1] / closes[-253]) - 1.0) * 100.0

    if len(bench_bars) >= 65 and gold_map:
        bdf_tmp = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date").reset_index(drop=True)
        bdf_tmp["_dt"] = pd.to_datetime(bdf_tmp["date"])
        g_df = pd.DataFrame({"_dt": pd.to_datetime(list(gold_map.keys())), "gold": list(gold_map.values())}).sort_values("_dt")
        # Strictly causal US/global timezone guard: Gold close must be STRICTLY BEFORE NSE date T
        bdf_tmp = pd.merge_asof(bdf_tmp, g_df, on="_dt", direction="backward", allow_exact_matches=False)
        ratio = bdf_tmp["gold"] / bdf_tmp["close"].replace(0, np.nan)
        chg = ratio.pct_change(63).dropna()
        if not chg.empty:
            gold_nifty_63d_pct = float(chg.iloc[-1] * 100.0)

    above_50 = 0
    near_hi = 0
    near_lo = 0
    counted = 0
    for sym, bars in ohlcv_map.items():
        if sym in NON_EQUITY_SYMBOLS or len(bars) < 50:
            continue
        c = np.array([b["close"] for b in bars[-252:]], dtype=float)
        counted += 1
        if c[-1] > float(np.mean(c[-50:])):
            above_50 += 1
        hi52 = float(np.max(c))
        lo52 = float(np.min(c))
        if hi52 > 0 and (c[-1] / hi52) >= 0.90:
            near_hi += 1
        if lo52 > 0 and (c[-1] / lo52) <= 1.12:
            near_lo += 1

    breadth_pct = round((above_50 / max(1, counted)) * 100.0, 1)
    hilo_sent_pct = round(((near_hi - near_lo) / max(1, counted)) * 100.0, 1)

    # Pollet & Wilson (2010) 63D Cross-Stock Average Correlation telemetry (NOT a panic-to-cash trigger)
    avg_corr_63 = compute_pollet_wilson_avg_corr(ohlcv_map, window=63)
    dm_crash_damper_active = bool(bench_ret_252d_pct < 0.0 and vix_now >= 19.0)

    # Exact V6/V7 6-Gauge Stress Score (preserved unchanged)
    stress_pts = 0
    if vix_now >= 19.0:
        stress_pts += 1
    if vix_now >= 23.0:
        stress_pts += 1
    if not bench_above_200:
        stress_pts += 1
    if breadth_pct < 42.0:
        stress_pts += 1
    if hilo_sent_pct < -10.0:
        stress_pts += 1
    if gold_nifty_63d_pct > 14.0:
        stress_pts += 1

    if stress_pts >= 3:
        regime = "Risk-Off / High Fear (25% Equity)"
        tone = "bearish"
        target_exp_pct = 25
    elif stress_pts == 2:
        regime = "Cautious / Hedged (55% Equity)"
        tone = "warning"
        target_exp_pct = 55
    elif stress_pts == 1:
        regime = "Constructive / Watchful (85% Equity)"
        tone = "bullish"
        target_exp_pct = 85
    else:
        regime = "Risk-On / Full Bullish (100% Equity)"
        tone = "bullish"
        target_exp_pct = 100

    res = {
        "regime": regime,
        "tone": tone,
        "vix": round(vix_now, 2),
        "usdinr": round(usdinr_now, 2),
        "brent_usd": round(brent_now, 2) if brent_now is not None else None,
        "gold_nifty_63d_pct": round(gold_nifty_63d_pct, 1),
        "hilo_sentiment_pct": hilo_sent_pct,
        "bench_dist_sma200_pct": round(float(bench_dist_200), 2),
        "bench_ret_252d_pct": round(float(bench_ret_252d_pct), 2),
        "breadth_50d_pct": breadth_pct,
        "avg_corr_63": avg_corr_63,
        "avg_corr_elevated": bool(avg_corr_63 >= 0.313),
        "dm_crash_damper_active": dm_crash_damper_active,
        "stress_score": stress_pts,
        "target_exposure_pct": target_exp_pct,
        "universe_size": counted,
    }
    with _MACRO_LOCK:
        _MACRO_CACHE[cache_key] = (now, res)
    return res


def _cs_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return ((series - series.mean()) / std).clip(-3.0, 3.0)


def _merge_causal_macro(
    bdf: pd.DataFrame,
    series_dict: dict[str, float],
    col: str,
    default_v: float,
    strict_prior_day: bool,
) -> pd.DataFrame:
    """
    Merge macro series causally onto NSE dates (NEVER backfills future dates!).
    For US/global markets closing AFTER NSE 3:30 PM IST (S&P 500, Brent, Gold, USD/INR),
    `strict_prior_day=True` enforces macro_date < nse_date via merge_asof(allow_exact_matches=False),
    eliminating all cross-timezone lookahead bias cleanly across US & Indian holidays.
    """
    if not series_dict:
        bdf[col] = default_v
        return bdf
    m_df = pd.DataFrame({"_dt": pd.to_datetime(list(series_dict.keys())), col: list(series_dict.values())}).sort_values("_dt")
    merged = pd.merge_asof(
        bdf,
        m_df,
        on="_dt",
        direction="backward",
        allow_exact_matches=not strict_prior_day,
    )
    merged[col] = merged[col].fillna(default_v)
    return merged


def _build_panel(
    db: Database,
    horizon: int = ML_HORIZON_DAYS,
    tb_horizon: int = TB_HORIZON_DAYS,
    model_version: str = "v7",
    prior_variant: str = "book",
    enable_dm_damper: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly causal NSE canonical factor & ML panel (supports v7 and v6 for ablations)."""
    ohlcv_map = db.get_all_ohlcv()
    tickers_meta = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    macro = db.get_macro_series()

    bench_bars = ohlcv_map.get(BENCHMARK_SYMBOL) or max(ohlcv_map.values(), key=len)
    bdf = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date").reset_index(drop=True)
    bdf["_dt"] = pd.to_datetime(bdf["date"])
    bdf["bench_ret_1d"] = bdf["close"].pct_change().fillna(0.0).clip(-0.15, 0.15)
    bdf["bench_ret_252d"] = (bdf["close"] / bdf["close"].shift(252) - 1.0).fillna(0.0)
    bdf["bench_fwd_ret"] = bdf["close"].shift(-horizon) / bdf["close"] - 1.0
    bdf["bench_sma200"] = bdf["close"].rolling(200, min_periods=50).mean()
    bdf["bench_dist_200"] = (bdf["close"] / bdf["bench_sma200"]) - 1.0

    # India VIX closes at 3:30 PM IST with NSE (same-day causal); US/Global series strictly lagged < nse_date!
    for m_sym, col, default_v, strict_lag in [
        (VOL_INDEX_SYMBOL, "vix_level", 15.0, False),
        (FX_MACRO_SYMBOL, "usdinr", 85.0, True),
        (BRENT_MACRO_SYMBOL, "brent", 75.0, True),
        (SP500_MACRO_SYMBOL, "sp500", 5000.0, True),
    ]:
        bdf = _merge_causal_macro(bdf, macro.get(m_sym, {}), col, default_v, strict_prior_day=strict_lag)

    bdf["brent_ret_1d"] = bdf["brent"].pct_change().fillna(0.0).clip(-0.15, 0.15)
    bdf["usdinr_ret_1d"] = bdf["usdinr"].pct_change().fillna(0.0).clip(-0.05, 0.05)
    bdf["sp500_ret_1d"] = bdf["sp500"].pct_change().fillna(0.0).clip(-0.10, 0.10)
    bdf["brent_63d_chg"] = bdf["brent"].pct_change(63).fillna(0.0).clip(-0.5, 0.5)
    bdf["usdinr_63d_chg"] = bdf["usdinr"].pct_change(63).fillna(0.0).clip(-0.15, 0.15)
    bdf["sp500_63d_chg"] = bdf["sp500"].pct_change(63).fillna(0.0).clip(-0.3, 0.3)

    macro_lookup = bdf[
        [
            "date",
            "bench_ret_1d",
            "bench_ret_252d",
            "bench_fwd_ret",
            "vix_level",
            "bench_dist_200",
            "brent_ret_1d",
            "usdinr_ret_1d",
            "sp500_ret_1d",
            "brent_63d_chg",
            "usdinr_63d_chg",
            "sp500_63d_chg",
        ]
    ]

    all_frames: list[pd.DataFrame] = []
    for sym, bars in ohlcv_map.items():
        if sym in NON_EQUITY_SYMBOLS or len(bars) < 80:
            continue
        df = compute_indicator_frame(bars, bench_bars)
        df = df.merge(macro_lookup, on="date", how="left")

        close = df["close"].astype(float)
        open_p = df["open"].astype(float).replace(0, np.nan).fillna(close)
        vol_sh = df["volume"].astype(float).clip(lower=0.0)
        ret_1d = close.pct_change().fillna(0.0).clip(-0.25, 0.25)
        log_ret = np.log(close / close.shift(1))
        bench_r = df["bench_ret_1d"].fillna(0.0)

        df["symbol"] = sym
        df["sector"] = tickers_meta.get(sym, "Unknown")
        df["vol_252d"] = log_ret.rolling(252, min_periods=63).std() * np.sqrt(252)
        vol_safe = df["vol_252d"].fillna(df["vol_20d"]).fillna(0.25).clip(lower=0.12)
        df["vol_safe"] = vol_safe

        # 1. Frazzini & Pedersen (2014) Vasicek-Shrunk Rolling Beta (strictly causal 252D)
        bench_var_252 = bench_r.rolling(252, min_periods=63).var().replace(0, np.nan)
        beta_raw = (ret_1d.rolling(252, min_periods=63).cov(bench_r) / bench_var_252).clip(-0.5, 3.0).fillna(1.0)
        df["beta_vasicek"] = (0.60 * beta_raw + 0.40 * 1.0).clip(0.10, 2.50)
        df["neg_beta"] = -df["beta_vasicek"]

        # 2. Blitz, Huij & Martens (2011) CAPM Residual Momentum (12-1M & 6-1M, strictly causal)
        # Uses lagged causal beta_{t-1} so day-t residual depends only on prices <= t.
        eps_1d = ret_1d - df["beta_vasicek"].shift(1).fillna(1.0) * bench_r
        idio_vol_252 = (eps_1d.rolling(252, min_periods=63).std() * np.sqrt(252.0)).fillna(vol_safe).clip(lower=0.10)
        idio_vol_126 = (eps_1d.rolling(126, min_periods=40).std() * np.sqrt(252.0)).fillna(vol_safe).clip(lower=0.10)
        resid_12m1m_sum = eps_1d.shift(21).rolling(231, min_periods=120).sum()
        resid_6m1m_sum = eps_1d.shift(21).rolling(105, min_periods=60).sum()
        df["resid_12m1m"] = resid_12m1m_sum.fillna(eps_1d.rolling(126, min_periods=40).sum()).fillna(0.0) / idio_vol_252
        df["resid_6m1m"] = resid_6m1m_sum.fillna(eps_1d.rolling(63, min_periods=20).sum()).fillna(0.0) / idio_vol_126

        # 3. Novy-Marx (2012) Intermediate-Horizon Momentum (12-7M, vol-adjusted)
        ret_12_7 = close.shift(126) / close.shift(252).replace(0, np.nan) - 1.0
        df["mom_12_7"] = ret_12_7.fillna(df["ret_126d"]).fillna(0.0) / vol_safe

        # 4. Grinblatt & Han (2005) Capital Gains Overhang (126D turnover-weighted VWAP)
        pv_126 = (close * vol_sh).rolling(126, min_periods=40).sum()
        v_126 = vol_sh.rolling(126, min_periods=40).sum().replace(0, np.nan)
        vwap_126 = (pv_126 / v_126).fillna(close.rolling(126, min_periods=40).mean())
        df["cgo_126"] = ((close - vwap_126) / close.replace(0, np.nan)).clip(-1.5, 0.9).fillna(0.0)

        # NSE Normalized Skip-1M Momentum (6M-1M and 12M-1M divided by 1Y volatility)
        ret_12m_1m = close.shift(21) / close.shift(252) - 1.0
        ret_6m_1m = close.shift(21) / close.shift(126) - 1.0
        df["mom_12m1m_voladj"] = ret_12m_1m.fillna(df["ret_126d"]) / vol_safe
        df["mom_6m1m_voladj"] = ret_6m_1m.fillna(df["ret_63d"]) / vol_safe

        # Frog-in-the-Pan (FIP) Smoothness
        pos_d = (log_ret > 0).astype(float).rolling(126, min_periods=40).mean()
        neg_d = (log_ret < 0).astype(float).rolling(126, min_periods=40).mean()
        df["fip_smooth_6m"] = pos_d - neg_d

        # 52W High Proximity & Downside Volatility
        df["prox_52w_high"] = close / df["high_52w"].replace(0, np.nan)
        df["down_vol_126d"] = log_ret.clip(upper=0.0).rolling(126, min_periods=40).std() * np.sqrt(252)
        df["above_sma50"] = (close > df["sma50"]).astype(float)

        # Behavioral Sentiment 1: MAX Retail-Lottery Hype Avoidance (Bali et al. JFE 2011)
        df["max_ret_21d"] = ret_1d.rolling(21, min_periods=10).max().fillna(0.0)

        # Behavioral Sentiment 2: Overnight Institutional Gap Sentiment (Lou, Polk & Skouras JFE 2019)
        prev_c = close.shift(1).replace(0, np.nan)
        overnight_r = ((open_p / prev_c) - 1.0).fillna(0.0).clip(-0.12, 0.12)
        df["overnight_sent_63d"] = overnight_r.rolling(63, min_periods=20).sum().fillna(0.0)

        # Stock-Specific Rolling Macro Betas x 63D Global Macro Trend (strictly causal)
        brent_r = df["brent_ret_1d"].fillna(0.0)
        fx_r = df["usdinr_ret_1d"].fillna(0.0)
        sp_r = df["sp500_ret_1d"].fillna(0.0)
        brent_var = brent_r.rolling(126, min_periods=40).var().replace(0, np.nan)
        fx_var = fx_r.rolling(126, min_periods=40).var().replace(0, np.nan)
        sp_var = sp_r.rolling(126, min_periods=40).var().replace(0, np.nan)
        beta_brent = (ret_1d.rolling(126, min_periods=40).cov(brent_r) / brent_var).clip(-1.5, 1.5).fillna(0.0)
        beta_fx = (ret_1d.rolling(126, min_periods=40).cov(fx_r) / fx_var).clip(-3.0, 3.0).fillna(0.0)
        beta_sp = (ret_1d.rolling(126, min_periods=40).cov(sp_r) / sp_var).clip(-1.5, 2.0).fillna(0.0)
        df["macro_tailwind_brent"] = (beta_brent * df["brent_63d_chg"].fillna(0.0)).fillna(0.0)
        df["macro_tailwind_fx"] = (beta_fx * df["usdinr_63d_chg"].fillna(0.0)).fillna(0.0)
        df["macro_tailwind_sp500"] = (beta_sp * df["sp500_63d_chg"].fillna(0.0)).fillna(0.0)

        # Strictly forward-looking targets
        df["fwd_ret"] = close.shift(-horizon) / close - 1.0
        df["tb_label"] = compute_triple_barrier_labels(
            close.to_numpy(dtype=float),
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["atr14"].to_numpy(dtype=float),
            horizon=tb_horizon,
        )
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame(), pd.DataFrame()

    full = pd.concat(all_frames, ignore_index=True)

    stock_mask = ~full["symbol"].isin(NON_EQUITY_SYMBOLS)
    breadth = full[stock_mask].groupby("date")["above_sma50"].mean().rename("nse_breadth_50d")
    full = full.merge(breadth.reset_index(), on="date", how="left")
    full["nse_breadth_50d"] = full["nse_breadth_50d"].fillna(0.5)

    # Sector-relative 3M & 6M momentum
    full["sec_rel_63d"] = (full["ret_63d"] - full.groupby(["date", "sector"])["ret_63d"].transform("median")).fillna(0.0)
    full["sec_rel_126d"] = (full["ret_126d"] - full.groupby(["date", "sector"])["ret_126d"].transform("median")).fillna(0.0)

    # Daily cross-sectional z-scores across the clean NSE equity universe (vectorized, exact ddof=0)
    z_cols = [
        "resid_12m1m", "resid_6m1m", "mom_12_7", "cgo_126", "neg_beta",
        "mom_6m1m_voladj", "mom_12m1m_voladj", "fip_smooth_6m", "prox_52w_high",
        "vol_252d", "down_vol_126d", "sec_rel_63d", "sec_rel_126d",
        "ret_21d", "rsi14", "vol_ratio_5_20",
        "macro_tailwind_brent", "macro_tailwind_fx", "macro_tailwind_sp500",
        "max_ret_21d", "overnight_sent_63d",
    ]
    g_date = full.groupby("date", sort=False)
    for col in z_cols:
        mu = g_date[col].transform("mean")
        sd = g_date[col].transform("std", ddof=0)
        full[f"z_{col}"] = ((full[col] - mu) / sd.where(sd >= 1e-9)).clip(-3.0, 3.0).fillna(0.0)

    # Multi-Factor Prior selection
    if model_version == "v6" or prior_variant == "v6":
        nse_mom = 0.5 * full["z_mom_6m1m_voladj"] + 0.5 * full["z_mom_12m1m_voladj"]
        nse_lowvol = -0.6 * full["z_vol_252d"] - 0.4 * full["z_down_vol_126d"]
        base_prior = (
            0.42 * nse_mom
            + 0.25 * nse_lowvol
            + 0.18 * full["z_fip_smooth_6m"]
            + 0.15 * full["z_prox_52w_high"]
        )
    elif prior_variant == "book_fip":
        # 2017-2021 hardened variant retaining 10% FIP smoothness & 25% Low-Risk (Vol + BAB)
        base_prior = (
            0.22 * full["z_resid_12m1m"]
            + 0.20 * full["z_mom_12_7"]
            + 0.13 * full["z_resid_6m1m"]
            + 0.10 * full["z_cgo_126"]
            + 0.10 * full["z_fip_smooth_6m"]
            + 0.125 * full["z_neg_beta"]
            - 0.125 * full["z_vol_252d"]
        )
    else:
        # Canonical V7 Book Prior (zero ML params)
        base_prior = (
            0.25 * full["z_resid_12m1m"]
            + 0.25 * full["z_mom_12_7"]
            + 0.15 * full["z_resid_6m1m"]
            + 0.15 * full["z_cgo_126"]
            + 0.10 * full["z_neg_beta"]
            - 0.10 * full["z_vol_252d"]
        )

    # Daniel & Moskowitz (2016) Bear-Rebound Momentum Crash Damper
    if model_version != "v6" and enable_dm_damper:
        dm_zone = (full["bench_ret_252d"] < 0.0) & (full["vix_level"] >= 19.0)
        dm_defensive = (
            0.40 * full["z_neg_beta"]
            - 0.35 * full["z_vol_252d"]
            + 0.25 * full["z_resid_12m1m"]
        )
        full["nse_factor_prior"] = np.where(dm_zone, 0.50 * base_prior + 0.50 * dm_defensive, base_prior)
    else:
        full["nse_factor_prior"] = base_prior

    full["target_alpha"] = (full["fwd_ret"] - full["bench_fwd_ret"]).clip(-0.45, 0.45)
    # V7 Vasicek Beta-Adjusted Residual Sharpe target (strips market-beta bull chasing)
    resid_fwd = full["fwd_ret"] - full["beta_vasicek"] * full["bench_fwd_ret"]
    full["target_resid_sharpe"] = (resid_fwd / full["vol_safe"].clip(lower=0.12)).clip(-3.0, 3.0)
    full["target_beat"] = (full["target_alpha"] > 0.0).astype(int)

    active_feats = V6_FEATURE_COLS if model_version == "v6" else FEATURE_COLS

    # Latest row per equity symbol for live inference (strictly excluding benchmark & reserve ETFs)
    latest_df = full[~full["symbol"].isin(NON_EQUITY_SYMBOLS)].sort_values("date").groupby("symbol", as_index=False).last()
    for col in active_feats:
        latest_df[col] = latest_df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Labeled training panel (strictly excluding benchmark & reserve ETFs)
    panel = full[~full["symbol"].isin(NON_EQUITY_SYMBOLS)].dropna(
        subset=["target_alpha", "target_resid_sharpe", "fwd_ret", "bench_fwd_ret", "tb_label", "sma200", "vol_252d", "fip_smooth_6m"]
    ).copy()
    if model_version == "v6":
        panel["fwd_rank"] = panel.groupby("date")["target_alpha"].rank(pct=True)
    else:
        panel["fwd_rank"] = panel.groupby("date")["target_resid_sharpe"].rank(pct=True)
    for col in active_feats:
        panel[col] = panel[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    panel["tb_label"] = panel["tb_label"].astype(int)

    return panel, latest_df


def _mean_daily_cs_auc(df: pd.DataFrame, label_col: str, prob_col: str) -> float:
    """Compute institutional mean daily cross-sectional ROC-AUC (immune to regime base-rate shifts)."""
    aucs: list[float] = []
    for _, grp in df.groupby("date"):
        if grp[label_col].nunique() == 2:
            try:
                aucs.append(float(roc_auc_score(grp[label_col], grp[prob_col])))
            except ValueError:
                pass
    if aucs:
        return float(np.mean(aucs))
    try:
        return float(roc_auc_score(df[label_col], df[prob_col]))
    except ValueError:
        return 0.5


def compute_afml_uniqueness_weights(
    dates: pd.Series | np.ndarray,
    symbols: pd.Series | np.ndarray,
    horizon: int = ML_HORIZON_DAYS,
) -> np.ndarray:
    """
    López de Prado (2018, AFML Ch. 4) Average Label Uniqueness Sample Weights utility
    (opt-in research helper; default V7 training keeps standard equal weights).
    """
    n = len(dates)
    if n == 0:
        return np.array([], dtype=float)
    dt_arr = np.asarray(dates)
    sym_arr = np.asarray(symbols)
    uniq_dates, inv_dt = np.unique(dt_arr, return_inverse=True)
    n_dt = len(uniq_dates)
    if n_dt <= 1:
        return np.ones(n, dtype=float)
    counts_per_day = np.bincount(inv_dt, minlength=n_dt).astype(float)
    csum = np.concatenate([[0.0], np.cumsum(counts_per_day)])
    starts = np.maximum(0, np.arange(n_dt) - horizon + 1)
    conc_per_day = np.maximum(1.0, csum[np.arange(n_dt) + 1] - csum[starts])
    inv_conc = 1.0 / conc_per_day
    ic_sum = np.concatenate([[0.0], np.cumsum(inv_conc)])
    ends = np.minimum(n_dt, np.arange(n_dt) + horizon)
    span = np.maximum(1.0, (ends - np.arange(n_dt)).astype(float))
    avg_u_by_day = (ic_sum[ends] - ic_sum[np.arange(n_dt)]) / span
    _, inv_sym, sym_cnts = np.unique(sym_arr, return_inverse=True, return_counts=True)
    sym_w = np.clip(np.median(sym_cnts) / np.maximum(1.0, sym_cnts[inv_sym]), 0.5, 1.5)
    w = avg_u_by_day[inv_dt] * sym_w
    w_mean = float(np.mean(w))
    return (w / w_mean) if (np.isfinite(w_mean) and w_mean > 1e-12) else np.ones(n, dtype=float)


def _fit_r4_ensemble(
    X: np.ndarray,
    y_rank: np.ndarray,
    y_beat: np.ndarray,
    y_tb: np.ndarray,
    cs_indices: list[int] | None = None,
    ridge_cs_only: bool = True,
) -> dict[str, Any]:
    idx_cs = CS_FEATURE_INDICES if cs_indices is None else cs_indices
    X_cs = X[:, idx_cs]
    X_ridge = X_cs if ridge_cs_only else X

    ridge_rank = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=500.0))])
    gbr_rank = HistGradientBoostingRegressor(
        max_depth=2, learning_rate=0.03, max_iter=65, min_samples_leaf=100, l2_regularization=10.0, random_state=42
    )
    # Binary payoff heads use strictly cross-sectional stock features (prevents VIX time-series inversion!)
    logit_beat = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.05, max_iter=300))])
    gbc_beat = HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.03, max_iter=55, min_samples_leaf=120, l2_regularization=15.0, random_state=42
    )
    logit_tb = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.05, max_iter=300))])
    gbc_tb = HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.03, max_iter=55, min_samples_leaf=120, l2_regularization=15.0, random_state=42
    )

    ridge_rank.fit(X_ridge, y_rank)
    gbr_rank.fit(X, y_rank)
    logit_beat.fit(X_cs, y_beat)
    gbc_beat.fit(X_cs, y_beat)
    logit_tb.fit(X_cs, y_tb)
    gbc_tb.fit(X_cs, y_tb)

    return {
        "ridge_rank": ridge_rank,
        "gbr_rank": gbr_rank,
        "logit_beat": logit_beat,
        "gbc_beat": gbc_beat,
        "logit_tb": logit_tb,
        "gbc_tb": gbc_tb,
        "cs_indices": idx_cs,
        "ridge_cs_only": ridge_cs_only,
    }


def _predict_r4_ensemble(models: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_cs = models.get("cs_indices", CS_FEATURE_INDICES)
    X_cs = X[:, idx_cs]
    X_ridge = X_cs if models.get("ridge_cs_only", True) else X
    pred_rank = 0.60 * models["ridge_rank"].predict(X_ridge) + 0.40 * models["gbr_rank"].predict(X)
    prob_beat = 0.65 * models["logit_beat"].predict_proba(X_cs)[:, 1] + 0.35 * models["gbc_beat"].predict_proba(X_cs)[:, 1]
    prob_tb = 0.65 * models["logit_tb"].predict_proba(X_cs)[:, 1] + 0.35 * models["gbc_tb"].predict_proba(X_cs)[:, 1]
    return pred_rank, prob_beat, prob_tb


def train_and_predict_universe(
    db: Database,
    horizon: int = ML_HORIZON_DAYS,
    tb_horizon: int = TB_HORIZON_DAYS,
    model_version: str = "v7",
    prior_variant: str = "book",
    enable_dm_damper: bool = True,
    persist: bool = True,
) -> dict[str, Any]:
    """Run strict 5-fold expanding walk-forward validation (V7 Canonical Book Hybrid) and save predictions."""
    panel, latest_df = _build_panel(
        db,
        horizon=horizon,
        tb_horizon=tb_horizon,
        model_version=model_version,
        prior_variant=prior_variant,
        enable_dm_damper=enable_dm_damper,
    )
    if panel.empty or len(panel) < 250:
        raise ValueError("Insufficient historical observations to train ML ranking model.")

    active_feats = V6_FEATURE_COLS if model_version == "v6" else FEATURE_COLS
    cs_indices = [i for i, c in enumerate(active_feats) if c not in _DATE_MACRO_COLS]
    ridge_cs_only = model_version != "v6"

    unique_dates = sorted(panel["date"].unique())
    n_dates = len(unique_dates)
    embargo = max(horizon, tb_horizon)
    if n_dates < 120:
        raise ValueError("Need at least 120 trading dates for walk-forward validation.")

    n_folds = 5
    first_test_start = int(n_dates * 0.42)
    fold_len = max(20, (n_dates - first_test_start) // n_folds)

    folds_summary: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []
    top8_returns: list[float] = []
    bench_returns: list[float] = []
    top8_hits: list[int] = []
    top8_tb_wins: list[int] = []

    for f in range(n_folds):
        test_start_idx = first_test_start + f * fold_len
        test_end_idx = n_dates if f == n_folds - 1 else min(n_dates, test_start_idx + fold_len)
        train_end_idx = test_start_idx - embargo
        if train_end_idx < 60 or test_start_idx >= n_dates:
            continue

        train_cutoff = unique_dates[train_end_idx - 1]
        test_start_dt = unique_dates[test_start_idx]
        test_end_dt = unique_dates[test_end_idx - 1]

        train_df = panel[panel["date"] <= train_cutoff]
        test_df = panel[(panel["date"] >= test_start_dt) & (panel["date"] <= test_end_dt)].copy()
        if len(train_df) < 200 or len(test_df) < 30:
            continue

        X_tr = train_df[active_feats].to_numpy(dtype=float)
        models = _fit_r4_ensemble(
            X_tr,
            train_df["fwd_rank"].to_numpy(dtype=float),
            train_df["target_beat"].to_numpy(dtype=int),
            train_df["tb_label"].to_numpy(dtype=int),
            cs_indices=cs_indices,
            ridge_cs_only=ridge_cs_only,
        )

        X_te = test_df[active_feats].to_numpy(dtype=float)
        p_rank, p_beat, p_tb = _predict_r4_ensemble(models, X_te)
        test_df["ml_rank"] = p_rank
        test_df["prob_beat"] = p_beat
        test_df["prob_tb_win"] = p_tb

        # Daily standardized Hybrid Score: 50% Canonical Factor Prior + 50% 40D Residual-Target ML
        test_df["z_ml_rank"] = test_df.groupby("date")["ml_rank"].transform(_cs_zscore).fillna(0.0)
        test_df["r4_score"] = 0.50 * test_df["nse_factor_prior"] + 0.50 * test_df["z_ml_rank"]
        test_df["pred_alpha_pct"] = (test_df["r4_score"] * 2.5).clip(-12.0, 12.0)

        oos_frames.append(
            test_df[
                ["date", "symbol", "r4_score", "pred_alpha_pct", "prob_tb_win", "fwd_ret", "bench_fwd_ret", "tb_label", "vix_level"]
            ]
        )

        test_dates = sorted(test_df["date"].unique())
        f_top, f_bench, f_hits, f_tb = [], [], [], []
        for dt in test_dates[::20]:
            top_k = test_df[test_df["date"] == dt].sort_values("r4_score", ascending=False).head(10)
            if top_k.empty:
                continue
            r_top = float(top_k["fwd_ret"].mean())
            r_b = float(top_k["bench_fwd_ret"].mean())
            f_top.append(r_top)
            f_bench.append(r_b)
            f_hits.append(1 if r_top > r_b else 0)
            f_tb.extend(top_k["tb_label"].tolist())

        top8_returns.extend(f_top)
        bench_returns.extend(f_bench)
        top8_hits.extend(f_hits)
        top8_tb_wins.extend(f_tb)

        ic_fold = float(spearmanr(test_df["r4_score"], test_df["target_alpha"]).statistic)
        if np.isnan(ic_fold):
            ic_fold = 0.0
        auc_fold = _mean_daily_cs_auc(test_df, "tb_label", "prob_tb_win")

        folds_summary.append(
            {
                "fold": f + 1,
                "train_until": train_cutoff,
                "test_period": f"{test_start_dt} to {test_end_dt}",
                "samples": int(len(test_df)),
                "ic_spearman": round(ic_fold, 3),
                "tb_auc": round(auc_fold, 3),
                "top5_return_pct": round(float(np.mean(f_top)) * 100.0, 2) if f_top else 0.0,
                "bench_return_pct": round(float(np.mean(f_bench)) * 100.0, 2) if f_bench else 0.0,
                "hit_rate_pct": round(float(np.mean(f_hits)) * 100.0, 1) if f_hits else 50.0,
            }
        )

    oos_all = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    ic_overall = 0.0
    tb_auc_overall = 0.5
    regime_stats: dict[str, Any] = {}

    if not oos_all.empty:
        ic_raw = spearmanr(oos_all["r4_score"], oos_all["fwd_ret"] - oos_all["bench_fwd_ret"]).statistic
        ic_overall = 0.0 if np.isnan(ic_raw) else float(ic_raw)
        tb_auc_overall = _mean_daily_cs_auc(oos_all, "tb_label", "prob_tb_win")

        for label, mask in [("calm_vix_lt20", oos_all["vix_level"] < 19.0), ("elevated_vix_ge20", oos_all["vix_level"] >= 19.0)]:
            sub = oos_all[mask]
            if len(sub) < 20:
                continue
            rt_l, rb_l = [], []
            for dt in sorted(sub["date"].unique())[::20]:
                tk = sub[sub["date"] == dt].sort_values("r4_score", ascending=False).head(10)
                if not tk.empty:
                    rt_l.append(float(tk["fwd_ret"].mean()))
                    rb_l.append(float(tk["bench_fwd_ret"].mean()))
            if rt_l:
                regime_stats[label] = {
                    "periods": len(rt_l),
                    "top5_20d_pct": round(float(np.mean(rt_l)) * 100.0, 2),
                    "spy_20d_pct": round(float(np.mean(rb_l)) * 100.0, 2),
                    "spread_pct": round((float(np.mean(rt_l)) - float(np.mean(rb_l))) * 100.0, 2),
                }

    avg_top = float(np.mean(top8_returns)) * 100.0 if top8_returns else 0.0
    avg_bench = float(np.mean(bench_returns)) * 100.0 if bench_returns else 0.0
    hit_rate = float(np.mean(top8_hits)) * 100.0 if top8_hits else 50.0
    tb_top_prec = float(np.mean(top8_tb_wins)) * 100.0 if top8_tb_wins else 50.0

    # Fit final production ensemble on ALL labeled NSE history
    X_all = panel[active_feats].to_numpy(dtype=float)
    final_models = _fit_r4_ensemble(
        X_all,
        panel["fwd_rank"].to_numpy(dtype=float),
        panel["target_beat"].to_numpy(dtype=int),
        panel["tb_label"].to_numpy(dtype=int),
        cs_indices=cs_indices,
        ridge_cs_only=ridge_cs_only,
    )

    c_rank_raw = final_models["ridge_rank"].named_steps["model"].coef_
    c_rank_full = np.zeros(len(active_feats), dtype=float)
    if ridge_cs_only:
        for pos_cs, feat_idx in enumerate(cs_indices):
            c_rank_full[feat_idx] = float(c_rank_raw[pos_cs])
    else:
        c_rank_full = np.asarray(c_rank_raw, dtype=float)

    c_tb_cs = final_models["logit_tb"].named_steps["model"].coef_[0]
    c_tb_full = np.zeros(len(active_feats), dtype=float)
    for pos_cs, feat_idx in enumerate(cs_indices):
        c_tb_full[feat_idx] = abs(float(c_tb_cs[pos_cs]))

    norm_rank = np.abs(c_rank_full) / (np.sum(np.abs(c_rank_full)) or 1.0)
    norm_tb = c_tb_full / (np.sum(c_tb_full) or 1.0)
    blended_imp = 0.70 * norm_rank + 0.30 * norm_tb
    feat_imps = sorted(
        [
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "importance_pct": round(float(imp * 100.0), 1),
                "direction": "positive" if c_rank_full[i] >= 0 else "negative",
            }
            for i, (col, imp) in enumerate(zip(active_feats, blended_imp))
        ],
        key=lambda x: x["importance_pct"],
        reverse=True,
    )

    # Live inference across all clean NSE equity symbols
    latest_df = latest_df.reset_index(drop=True)
    X_latest = latest_df[active_feats].to_numpy(dtype=float)
    p_rank_arr, p_beat_arr, p_tb_arr = _predict_r4_ensemble(final_models, X_latest)
    z_ml_live = _cs_zscore(pd.Series(p_rank_arr)).to_numpy(dtype=float)
    prior_live = latest_df["nse_factor_prior"].fillna(0.0).to_numpy(dtype=float)
    r4_live = 0.50 * prior_live + 0.50 * z_ml_live

    nifty_base_h = 0.018  # ~11.5% annualized baseline NIFTY drift over 40 trading days
    now_iso = datetime.now(timezone.utc).isoformat()
    predictions: list[dict[str, Any]] = []

    for idx, row in latest_df.iterrows():
        sym = str(row["symbol"])
        if sym in NON_EQUITY_SYMBOLS:
            continue
        close_px = float(row["close"])
        atr = float(row["atr14"]) if pd.notna(row.get("atr14")) and row["atr14"] > 0 else close_px * 0.02
        alpha_pct = float(np.clip(r4_live[idx] * 2.5, -14.0, 14.0))
        p_beat = float(np.clip(p_beat_arr[idx], 0.10, 0.90))
        p_tb = float(np.clip(p_tb_arr[idx], 0.10, 0.90))

        exp_ret_h = nifty_base_h + (alpha_pct / 100.0)
        vol_ann = float(row["vol_252d"]) if pd.notna(row.get("vol_252d")) and row["vol_252d"] > 0 else 0.24
        target_px = round(close_px + TB_UP_ATR_MULT * atr, 2)
        stop_px = round(max(0.5, close_px - TB_DOWN_ATR_MULT * atr), 2)
        rr = round((target_px - close_px) / max(0.01, close_px - stop_px), 2)

        cone = build_probability_cone(
            last_date=str(row["date"]),
            last_price=close_px,
            expected_horizon_return=exp_ret_h,
            vol_ann=vol_ann,
            horizon=horizon,
        )

        predictions.append(
            {
                "symbol": sym,
                "as_of_date": str(row["date"]),
                "pred_alpha_20d": round(alpha_pct, 2),
                "prob_outperform": round(p_beat * 100.0, 1),
                "prob_tb_win": round(p_tb * 100.0, 1),
                "expected_return_20d": round(exp_ret_h * 100.0, 2),
                "vol_20d_ann": round(vol_ann * 100.0, 1),
                "target_price": target_px,
                "stop_price": stop_px,
                "risk_reward": rr,
                "cone": cone,
                "updated_at": now_iso,
            }
        )

    oos_rows = [
        {
            "date": str(r["date"]),
            "symbol": str(r["symbol"]),
            "pred_alpha": round(float(r["r4_score"]), 5),
            "prob_tb_win": round(float(r["prob_tb_win"]) * 100.0, 2),
            "actual_fwd_20d": round(float(r["fwd_ret"]) * 100.0, 4),
            "bench_fwd_20d": round(float(r["bench_fwd_ret"]) * 100.0, 4),
            "tb_outcome": int(r["tb_label"]),
        }
        for _, r in oos_all.iterrows()
        if str(r["symbol"]) not in NON_EQUITY_SYMBOLS
    ]

    run_meta = {
        "trained_at": now_iso,
        "model_version": model_version,
        "prior_variant": prior_variant,
        "horizon_days": horizon,
        "sample_count": int(len(panel)),
        "symbol_count": int(len(predictions)),
        "hit_rate": round(hit_rate, 1),
        "ic_spearman": round(ic_overall, 3),
        "top5_spread": round(avg_top - avg_bench, 2),
        "avg_top5_return": round(avg_top, 2),
        "avg_bench_return": round(avg_bench, 2),
        "tb_auc": round(tb_auc_overall, 3),
        "tb_top5_precision": round(tb_top_prec, 1),
        "feature_importances": feat_imps[:12],
        "folds": folds_summary,
        "regime_stats": regime_stats,
    }
    if persist:
        db.save_ml_run(run_meta, predictions, oos_rows=oos_rows)
        invalidate_macro_cache()
    return {
        "run": run_meta,
        "predictions_count": len(predictions),
        "oos_rows_count": len(oos_rows),
        "predictions": predictions,
        "oos_rows": oos_rows,
    }
