"""
EquiLens India v5 Walk-Forward Quantitative ML Engine (Promoted V5 Sentiment & Macro Architecture):
  - 19 Strictly Causal NSE Institutional, Behavioral Sentiment & Intermarket Macro Features:
      * Official NSE 6M+12M Skip-1M Volatility-Adjusted Momentum & Frog-in-the-Pan (FIP) Smoothness
      * 52W High Proximity, 1Y & Downside Volatility, Sector-Relative Alpha
      * Stock-Specific Rolling Macro Sensitivities (Beta x 63D Trend for Brent Crude, USD/INR, S&P 500)
      * Behavioral MAX Retail-Lottery Hype Avoidance (Bali, Cakici & Whitelaw JFE 2011)
      * Overnight Institutional Gap Sentiment (Lou, Polk & Skouras JFE 2019)
      * India VIX, NIFTY 50 vs 200-DMA & NSE Breadth
  - 50% NSE Multi-Factor Prior + 50% Regularized 40D Cross-Sectional Rank-Target ML
    (Ridge + shallow HistGradientBoosting), evaluated across 5 expanding chronological folds
    with a strict 40-day embargo gap (zero look-ahead leakage).
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
    SP500_MACRO_SYMBOL,
    TB_DOWN_ATR_MULT,
    TB_HORIZON_DAYS,
    TB_UP_ATR_MULT,
    VOL_INDEX_SYMBOL,
)
from .database import Database
from .indicators import compute_indicator_frame

FEATURE_COLS: list[str] = [
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

FEATURE_LABELS: dict[str, str] = {
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


def compute_current_macro_regime(db: Database) -> dict[str, Any]:
    """Compute live V5 6-Gauge Indian Sentiment & Intermarket Macro regime snapshot."""
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
    gold_nifty_63d_pct = 0.0
    if len(bench_bars) >= 200:
        closes = np.array([b["close"] for b in bench_bars], dtype=float)
        sma200 = float(np.mean(closes[-200:]))
        bench_dist_200 = ((closes[-1] / sma200) - 1.0) * 100.0
        bench_above_200 = bool(closes[-1] >= sma200)

    if len(bench_bars) >= 65 and gold_map:
        bdf_tmp = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date")
        gs = pd.Series(gold_map, dtype=float).rename("gold")
        bdf_tmp = bdf_tmp.merge(gs.rename_axis("date").reset_index(), on="date", how="left")
        bdf_tmp["gold"] = bdf_tmp["gold"].ffill().bfill()
        ratio = bdf_tmp["gold"] / bdf_tmp["close"].replace(0, np.nan)
        chg = ratio.pct_change(63).dropna()
        if not chg.empty:
            gold_nifty_63d_pct = float(chg.iloc[-1] * 100.0)

    above_50 = 0
    near_hi = 0
    near_lo = 0
    counted = 0
    for sym, bars in ohlcv_map.items():
        if sym == BENCHMARK_SYMBOL or len(bars) < 50:
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

    # Exact V5 6-Gauge Stress Score
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
        "breadth_50d_pct": breadth_pct,
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


def _build_panel(
    db: Database,
    horizon: int = ML_HORIZON_DAYS,
    tb_horizon: int = TB_HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build strictly causal V5 NSE institutional, behavioral sentiment & intermarket macro panel."""
    ohlcv_map = db.get_all_ohlcv()
    tickers_meta = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    macro = db.get_macro_series()

    bench_bars = ohlcv_map.get(BENCHMARK_SYMBOL) or max(ohlcv_map.values(), key=len)
    bdf = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date").reset_index(drop=True)
    bdf["bench_fwd_ret"] = bdf["close"].shift(-horizon) / bdf["close"] - 1.0
    bdf["bench_sma200"] = bdf["close"].rolling(200, min_periods=50).mean()
    bdf["bench_dist_200"] = (bdf["close"] / bdf["bench_sma200"]) - 1.0

    for m_sym, col, default_v in [
        (VOL_INDEX_SYMBOL, "vix_level", 15.0),
        (FX_MACRO_SYMBOL, "usdinr", 85.0),
        (BRENT_MACRO_SYMBOL, "brent", 75.0),
        (SP500_MACRO_SYMBOL, "sp500", 5000.0),
    ]:
        s_dict = macro.get(m_sym, {})
        if s_dict:
            s = pd.Series(s_dict, name=col, dtype=float)
            bdf = bdf.merge(s.rename_axis("date").reset_index(), on="date", how="left")
            bdf[col] = bdf[col].ffill().bfill().fillna(default_v)
        else:
            bdf[col] = default_v

    bdf["brent_ret_1d"] = bdf["brent"].pct_change().fillna(0.0).clip(-0.15, 0.15)
    bdf["usdinr_ret_1d"] = bdf["usdinr"].pct_change().fillna(0.0).clip(-0.05, 0.05)
    bdf["sp500_ret_1d"] = bdf["sp500"].pct_change().fillna(0.0).clip(-0.10, 0.10)
    bdf["brent_63d_chg"] = bdf["brent"].pct_change(63).fillna(0.0).clip(-0.5, 0.5)
    bdf["usdinr_63d_chg"] = bdf["usdinr"].pct_change(63).fillna(0.0).clip(-0.15, 0.15)
    bdf["sp500_63d_chg"] = bdf["sp500"].pct_change(63).fillna(0.0).clip(-0.3, 0.3)

    macro_lookup = bdf[
        [
            "date",
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
        if len(bars) < 80:
            continue
        df = compute_indicator_frame(bars, bench_bars)
        df = df.merge(macro_lookup, on="date", how="left")

        close = df["close"].astype(float)
        open_p = df["open"].astype(float).replace(0, np.nan).fillna(close)
        ret_1d = close.pct_change().fillna(0.0).clip(-0.25, 0.25)
        log_ret = np.log(close / close.shift(1))

        df["symbol"] = sym
        df["sector"] = tickers_meta.get(sym, "Unknown")
        df["vol_252d"] = log_ret.rolling(252, min_periods=63).std() * np.sqrt(252)
        vol_safe = df["vol_252d"].fillna(df["vol_20d"]).fillna(0.25).clip(lower=0.12)

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

        # V5 Behavioral Sentiment 1: MAX Retail-Lottery Hype Avoidance (Bali et al. JFE 2011)
        df["max_ret_21d"] = ret_1d.rolling(21, min_periods=10).max().fillna(0.0)

        # V5 Behavioral Sentiment 2: Overnight Institutional Gap Sentiment (Lou, Polk & Skouras JFE 2019)
        prev_c = close.shift(1).replace(0, np.nan)
        overnight_r = ((open_p / prev_c) - 1.0).fillna(0.0).clip(-0.12, 0.12)
        df["overnight_sent_63d"] = overnight_r.rolling(63, min_periods=20).sum().fillna(0.0)

        # V5 Stock-Specific Rolling Macro Betas x 63D Global Macro Trend (strictly causal)
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

    stock_mask = full["symbol"] != BENCHMARK_SYMBOL
    breadth = full[stock_mask].groupby("date")["above_sma50"].mean().rename("nse_breadth_50d")
    full = full.merge(breadth.reset_index(), on="date", how="left")
    full["nse_breadth_50d"] = full["nse_breadth_50d"].fillna(0.5)

    # Sector-relative 3M & 6M momentum
    full["sec_rel_63d"] = (full["ret_63d"] - full.groupby(["date", "sector"])["ret_63d"].transform("median")).fillna(0.0)
    full["sec_rel_126d"] = (full["ret_126d"] - full.groupby(["date", "sector"])["ret_126d"].transform("median")).fillna(0.0)

    # Daily cross-sectional z-scores across the NSE universe
    for col in [
        "mom_6m1m_voladj", "mom_12m1m_voladj", "fip_smooth_6m", "prox_52w_high",
        "vol_252d", "down_vol_126d", "sec_rel_63d", "sec_rel_126d",
        "ret_21d", "rsi14", "vol_ratio_5_20",
        "macro_tailwind_brent", "macro_tailwind_fx", "macro_tailwind_sp500",
        "max_ret_21d", "overnight_sent_63d",
    ]:
        full[f"z_{col}"] = full.groupby("date")[col].transform(_cs_zscore).fillna(0.0)

    # Official-style NSE Institutional Multi-Factor Prior (zero overfitting)
    nse_mom = 0.5 * full["z_mom_6m1m_voladj"] + 0.5 * full["z_mom_12m1m_voladj"]
    nse_lowvol = -0.6 * full["z_vol_252d"] - 0.4 * full["z_down_vol_126d"]
    full["nse_factor_prior"] = (
        0.42 * nse_mom
        + 0.25 * nse_lowvol
        + 0.18 * full["z_fip_smooth_6m"]
        + 0.15 * full["z_prox_52w_high"]
    )

    full["target_alpha"] = (full["fwd_ret"] - full["bench_fwd_ret"]).clip(-0.45, 0.45)
    full["target_beat"] = (full["target_alpha"] > 0.0).astype(int)

    # Latest row per symbol for live inference
    latest_df = full.sort_values("date").groupby("symbol", as_index=False).last()
    for col in FEATURE_COLS:
        latest_df[col] = latest_df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Labeled training panel (excluding NIFTY 50 index itself)
    panel = full[full["symbol"] != BENCHMARK_SYMBOL].dropna(
        subset=["target_alpha", "fwd_ret", "bench_fwd_ret", "tb_label", "sma200", "vol_252d", "fip_smooth_6m"]
    ).copy()
    panel["fwd_rank"] = panel.groupby("date")["target_alpha"].rank(pct=True)
    for col in FEATURE_COLS:
        panel[col] = panel[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    panel["tb_label"] = panel["tb_label"].astype(int)

    return panel, latest_df


_DATE_MACRO_COLS = {"vix_level", "bench_dist_200", "nse_breadth_50d"}
CS_FEATURE_INDICES: list[int] = [i for i, c in enumerate(FEATURE_COLS) if c not in _DATE_MACRO_COLS]


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


def _fit_r4_ensemble(X: np.ndarray, y_rank: np.ndarray, y_beat: np.ndarray, y_tb: np.ndarray) -> dict[str, Any]:
    ridge_rank = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=500.0))])
    gbr_rank = HistGradientBoostingRegressor(
        max_depth=2, learning_rate=0.03, max_iter=65, min_samples_leaf=100, l2_regularization=10.0, random_state=42
    )
    # Binary payoff heads use strictly cross-sectional stock features (prevents VIX time-series inversion!)
    X_cs = X[:, CS_FEATURE_INDICES]
    logit_beat = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.05, max_iter=300))])
    gbc_beat = HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.03, max_iter=55, min_samples_leaf=120, l2_regularization=15.0, random_state=42
    )
    logit_tb = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=0.05, max_iter=300))])
    gbc_tb = HistGradientBoostingClassifier(
        max_depth=2, learning_rate=0.03, max_iter=55, min_samples_leaf=120, l2_regularization=15.0, random_state=42
    )

    ridge_rank.fit(X, y_rank)
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
    }


def _predict_r4_ensemble(models: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_cs = X[:, CS_FEATURE_INDICES]
    pred_rank = 0.60 * models["ridge_rank"].predict(X) + 0.40 * models["gbr_rank"].predict(X)
    prob_beat = 0.65 * models["logit_beat"].predict_proba(X_cs)[:, 1] + 0.35 * models["gbc_beat"].predict_proba(X_cs)[:, 1]
    prob_tb = 0.65 * models["logit_tb"].predict_proba(X_cs)[:, 1] + 0.35 * models["gbc_tb"].predict_proba(X_cs)[:, 1]
    return pred_rank, prob_beat, prob_tb


def train_and_predict_universe(
    db: Database,
    horizon: int = ML_HORIZON_DAYS,
    tb_horizon: int = TB_HORIZON_DAYS,
) -> dict[str, Any]:
    """Run strict 5-fold expanding walk-forward validation (V6 Hybrid architecture) and save predictions."""
    panel, latest_df = _build_panel(db, horizon=horizon, tb_horizon=tb_horizon)
    if panel.empty or len(panel) < 250:
        raise ValueError("Insufficient historical observations to train ML ranking model.")

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

        X_tr = train_df[FEATURE_COLS].to_numpy(dtype=float)
        models = _fit_r4_ensemble(
            X_tr,
            train_df["fwd_rank"].to_numpy(dtype=float),
            train_df["target_beat"].to_numpy(dtype=int),
            train_df["tb_label"].to_numpy(dtype=int),
        )

        X_te = test_df[FEATURE_COLS].to_numpy(dtype=float)
        p_rank, p_beat, p_tb = _predict_r4_ensemble(models, X_te)
        test_df["ml_rank"] = p_rank
        test_df["prob_beat"] = p_beat
        test_df["prob_tb_win"] = p_tb

        # Daily standardized Hybrid Score: 50% NSE Institutional Factor Prior + 50% 40D Rank-ML
        test_df["z_ml_rank"] = test_df.groupby("date")["ml_rank"].transform(_cs_zscore).fillna(0.0)
        test_df["r4_score"] = 0.50 * test_df["nse_factor_prior"] + 0.50 * test_df["z_ml_rank"]
        # Scale standardized hybrid score to realistic expected excess return % for display (~2.5% per 1 sigma)
        test_df["pred_alpha_pct"] = (test_df["r4_score"] * 2.5).clip(-12.0, 12.0)

        oos_frames.append(
            test_df[
                ["date", "symbol", "r4_score", "pred_alpha_pct", "prob_tb_win", "fwd_ret", "bench_fwd_ret", "tb_label", "vix_level"]
            ]
        )

        test_dates = sorted(test_df["date"].unique())
        f_top, f_bench, f_hits, f_tb = [], [], [], []
        for dt in test_dates[::20]:
            top_k = test_df[test_df["date"] == dt].sort_values("r4_score", ascending=False).head(8)
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
                tk = sub[sub["date"] == dt].sort_values("r4_score", ascending=False).head(8)
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
    X_all = panel[FEATURE_COLS].to_numpy(dtype=float)
    final_models = _fit_r4_ensemble(
        X_all,
        panel["fwd_rank"].to_numpy(dtype=float),
        panel["target_beat"].to_numpy(dtype=int),
        panel["tb_label"].to_numpy(dtype=int),
    )

    c_rank = final_models["ridge_rank"].named_steps["model"].coef_
    c_tb_cs = final_models["logit_tb"].named_steps["model"].coef_[0]
    c_tb_full = np.zeros(len(FEATURE_COLS), dtype=float)
    for pos_cs, feat_idx in enumerate(CS_FEATURE_INDICES):
        c_tb_full[feat_idx] = abs(c_tb_cs[pos_cs])
    norm_rank = np.abs(c_rank) / (np.sum(np.abs(c_rank)) or 1.0)
    norm_tb = c_tb_full / (np.sum(c_tb_full) or 1.0)
    blended_imp = 0.70 * norm_rank + 0.30 * norm_tb
    feat_imps = sorted(
        [
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "importance_pct": round(float(imp * 100.0), 1),
                "direction": "positive" if c_rank[i] >= 0 else "negative",
            }
            for i, (col, imp) in enumerate(zip(FEATURE_COLS, blended_imp))
        ],
        key=lambda x: x["importance_pct"],
        reverse=True,
    )

    # Live inference across all NSE symbols
    X_latest = latest_df[FEATURE_COLS].to_numpy(dtype=float)
    p_rank_arr, p_beat_arr, p_tb_arr = _predict_r4_ensemble(final_models, X_latest)
    z_ml_live = _cs_zscore(pd.Series(p_rank_arr)).to_numpy(dtype=float)
    prior_live = latest_df["nse_factor_prior"].fillna(0.0).to_numpy(dtype=float)
    r4_live = 0.50 * prior_live + 0.50 * z_ml_live

    nifty_base_h = 0.018  # ~11.5% annualized baseline NIFTY drift over 40 trading days
    now_iso = datetime.now(timezone.utc).isoformat()
    predictions: list[dict[str, Any]] = []

    for idx, row in latest_df.iterrows():
        sym = str(row["symbol"])
        close_px = float(row["close"])
        atr = float(row["atr14"]) if pd.notna(row.get("atr14")) and row["atr14"] > 0 else close_px * 0.02
        alpha_pct = float(np.clip(r4_live[idx] * 2.5, -14.0, 14.0))
        p_beat = float(np.clip(p_beat_arr[idx], 0.10, 0.90))
        p_tb = float(np.clip(p_tb_arr[idx], 0.10, 0.90))

        if sym == BENCHMARK_SYMBOL:
            alpha_pct, p_beat, p_tb = 0.0, 0.50, 0.50

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
            "pred_alpha": round(float(r["r4_score"]), 5),  # exact R4 score persisted for portfolio backtester
            "prob_tb_win": round(float(r["prob_tb_win"]) * 100.0, 2),
            "actual_fwd_20d": round(float(r["fwd_ret"]) * 100.0, 4),
            "bench_fwd_20d": round(float(r["bench_fwd_ret"]) * 100.0, 4),
            "tb_outcome": int(r["tb_label"]),
        }
        for _, r in oos_all.iterrows()
    ]

    run_meta = {
        "trained_at": now_iso,
        "horizon_days": horizon,
        "sample_count": int(len(panel)),
        "symbol_count": int(latest_df["symbol"].nunique()),
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
    db.save_ml_run(run_meta, predictions, oos_rows=oos_rows)
    invalidate_macro_cache()
    return {"run": run_meta, "predictions_count": len(predictions), "oos_rows_count": len(oos_rows)}
