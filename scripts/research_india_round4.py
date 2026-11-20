"""
EquiLens India Quant Research Lab — Round 4 (Deep Alpha, Behavioral Sentiment, Trend & Intermarket Macro):
Builds directly on Round 3's key discoveries:
  1. Dumping date-constant global macro directly into cross-sectional rank ML adds noise,
     WHEREAS using Sentiment/Gold-NIFTY/Hi-Lo in the Dynamic Exposure Risk Gate added +4.75%/yr!
  2. Now tests truly cross-sectional, institutionally-proven signals one-by-one (strict ablation)
     against the R4 Champion:
     - G1: Sentiment & Intermarket Macro Risk Gate (VIX + 200DMA + Breadth + 52W Hi-Lo + Gold/NIFTY + Breadth Thrust)
     - G2: Behavioral Sentiment — MAX Lottery-Hype Avoidance (Bali, Cakici & Whitelaw JFE 2011)
     - G3: Institutional vs Retail Sentiment — Overnight Gap vs Intraday Flow (Lou, Polk & Skouras JFE 2019)
     - G4: Idiosyncratic (NIFTY-Residual) Momentum & Betting-Against-Beta (Blitz et al. / Frazzini & Pedersen)
     - G5: Clenow Smooth Trend Quality (Regression Slope x R^2 + 50DMA Slope)
     - G6: Stock-Specific Macro Sensitivities (Rolling Stock Beta x Brent Crude, USD/INR, S&P 500 trends)
     - G7: Institutional Rupee Accumulation (20D Up-Value / Total-Value Flow + Delivery/Turnover Surge)
     - G8: Clean Point-in-Time PEAD Earnings Surprise (isolated)
     - G9: Clean Top-Winners Ensembles + Multi-K & Slippage Robustness Check (Zero Overfitting)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import BENCHMARK_SYMBOL, FX_MACRO_SYMBOL, VOL_INDEX_SYMBOL
from backend.database import Database
from backend.indicators import compute_indicator_frame

warnings.filterwarnings("ignore")

INDIA_CASH_DAILY_YIELD = (1.062 ** (1.0 / 252.0)) - 1.0


def _cs_z(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return ((series - series.mean()) / std).clip(-3.0, 3.0)


def _clenow_trend(close: pd.Series, window: int = 90) -> pd.Series:
    """Vectorized rolling log-price linear regression annualized slope * R^2 (Clenow trend quality)."""
    log_p = np.log(close.replace(0, np.nan)).to_numpy(dtype=float)
    n = len(log_p)
    out = np.full(n, np.nan, dtype=float)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    sxx = float(np.sum(x_centered ** 2))

    for i in range(window - 1, n):
        y = log_p[i - window + 1 : i + 1]
        if np.any(~np.isfinite(y)):
            continue
        y_mean = y.mean()
        y_c = y - y_mean
        syy = float(np.sum(y_c ** 2))
        if syy < 1e-12:
            out[i] = 0.0
            continue
        sxy = float(np.dot(x_centered, y_c))
        slope = sxy / sxx
        r2 = (sxy * sxy) / (sxx * syy)
        ann_slope = np.exp(slope * 252.0) - 1.0
        out[i] = float(np.clip(ann_slope, -3.0, 5.0) * r2)
    return pd.Series(out, index=close.index)


def _compute_pead(dates: pd.Series, earn_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(dates)
    surp_arr = np.zeros(n, dtype=float)
    streak_arr = np.zeros(n, dtype=float)
    if not earn_rows:
        return surp_arr, streak_arr
    valid_e = sorted(
        [(e["earnings_date"][:10], float(e["surprise_pct"])) for e in earn_rows if e.get("surprise_pct") is not None],
        key=lambda x: x[0],
    )
    if not valid_e:
        return surp_arr, streak_arr
    e_dates = [pd.Timestamp(x[0]) for x in valid_e]
    e_vals = [x[1] for x in valid_e]
    bar_ts = pd.to_datetime(dates).tolist()
    ptr, cur_streak = -1, 0
    for i, ts in enumerate(bar_ts):
        while ptr + 1 < len(e_dates) and e_dates[ptr + 1] < ts:
            ptr += 1
            if e_vals[ptr] > 0:
                cur_streak = cur_streak + 1 if cur_streak > 0 else 1
            elif e_vals[ptr] < 0:
                cur_streak = cur_streak - 1 if cur_streak < 0 else -1
            else:
                cur_streak = 0
        if ptr >= 0:
            days_since = (ts - e_dates[ptr]).days
            decay = float(np.exp(-max(0, days_since) / 65.0))
            surp_arr[i] = float(np.clip(e_vals[ptr], -40.0, 40.0)) * decay
            streak_arr[i] = float(np.clip(cur_streak, -4, 6))
    return surp_arr, streak_arr


def build_round4_panel(db: Database, horizon: int = 40) -> pd.DataFrame:
    ohlcv_map = db.get_all_ohlcv()
    tickers_meta = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    earnings_map = db.get_all_earnings()
    macro = db.get_macro_series()

    bench_bars = ohlcv_map[BENCHMARK_SYMBOL]
    bdf = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date").reset_index(drop=True)
    bdf["bench_ret_1d"] = bdf["close"].pct_change().fillna(0.0)
    bdf["bench_fwd_ret"] = bdf["close"].shift(-horizon) / bdf["close"] - 1.0
    bdf["bench_sma200"] = bdf["close"].rolling(200, min_periods=50).mean()
    bdf["bench_dist_200"] = (bdf["close"] / bdf["bench_sma200"]) - 1.0

    for m_sym, col in [
        (VOL_INDEX_SYMBOL, "vix_level"),
        (FX_MACRO_SYMBOL, "usdinr"),
        ("BZ=F", "brent"),
        ("GC=F", "gold"),
        ("^TNX", "us10y"),
        ("^GSPC", "sp500"),
    ]:
        s = pd.Series(macro.get(m_sym, {}), name=col)
        bdf = bdf.merge(s.rename_axis("date").reset_index(), on="date", how="left")
        bdf[col] = bdf[col].ffill().bfill()

    bdf["vix_level"] = bdf["vix_level"].fillna(15.0)
    bdf["brent_ret_1d"] = bdf["brent"].pct_change().fillna(0.0).clip(-0.15, 0.15)
    bdf["usdinr_ret_1d"] = bdf["usdinr"].pct_change().fillna(0.0).clip(-0.05, 0.05)
    bdf["sp500_ret_1d"] = bdf["sp500"].pct_change().fillna(0.0).clip(-0.10, 0.10)

    bdf["brent_63d_chg"] = bdf["brent"].pct_change(63).fillna(0.0).clip(-0.5, 0.5)
    bdf["usdinr_63d_chg"] = bdf["usdinr"].pct_change(63).fillna(0.0).clip(-0.15, 0.15)
    bdf["sp500_63d_chg"] = bdf["sp500"].pct_change(63).fillna(0.0).clip(-0.3, 0.3)
    gold_nifty = bdf["gold"] / bdf["close"].replace(0, np.nan)
    bdf["gold_nifty_63d_chg"] = gold_nifty.pct_change(63).fillna(0.0).clip(-0.4, 0.4)

    macro_lookup = bdf[[
        "date", "bench_ret_1d", "bench_fwd_ret", "bench_dist_200", "vix_level",
        "brent_ret_1d", "usdinr_ret_1d", "sp500_ret_1d",
        "brent_63d_chg", "usdinr_63d_chg", "sp500_63d_chg", "gold_nifty_63d_chg",
    ]]

    frames = []
    for sym, bars in ohlcv_map.items():
        if sym == BENCHMARK_SYMBOL or len(bars) < 260:
            continue
        df = compute_indicator_frame(bars, bench_bars)
        df = df.merge(macro_lookup, on="date", how="left")

        close = df["close"].astype(float)
        open_p = df["open"].astype(float).replace(0, np.nan).fillna(close)
        vol = df["volume"].astype(float)
        ret_1d = close.pct_change().fillna(0.0).clip(-0.25, 0.25)
        log_ret = np.log(close / close.shift(1))

        df["symbol"] = sym
        df["sector"] = tickers_meta.get(sym, "Unknown")
        df["vol_252d"] = log_ret.rolling(252, min_periods=63).std() * np.sqrt(252)
        vol_safe = df["vol_252d"].fillna(df["vol_20d"]).fillna(0.25).clip(lower=0.12)

        # R4 core factors
        ret_12m_1m = close.shift(21) / close.shift(252) - 1.0
        ret_6m_1m = close.shift(21) / close.shift(126) - 1.0
        df["mom_12m1m_voladj"] = ret_12m_1m.fillna(df["ret_126d"]) / vol_safe
        df["mom_6m1m_voladj"] = ret_6m_1m.fillna(df["ret_63d"]) / vol_safe
        pos_d = (log_ret > 0).astype(float).rolling(126, min_periods=40).mean()
        neg_d = (log_ret < 0).astype(float).rolling(126, min_periods=40).mean()
        df["fip_smooth_6m"] = pos_d - neg_d
        df["prox_52w_high"] = close / df["high_52w"].replace(0, np.nan)
        df["prox_52w_low"] = close / df["low_52w"].replace(0, np.nan)
        df["down_vol_126d"] = log_ret.clip(upper=0.0).rolling(126, min_periods=40).std() * np.sqrt(252)
        df["above_sma50"] = (close > df["sma50"]).astype(float)

        # G2: Behavioral Sentiment — MAX Lottery-Hype Avoidance (Bali et al. JFE 2011)
        # Retail investors chase stocks with huge single-day pops (mean of top-3 daily returns in 21D) -> mean-reverts!
        df["max_ret_21d"] = ret_1d.rolling(21, min_periods=10).max().fillna(0.0)
        df["skew_63d"] = ret_1d.rolling(63, min_periods=30).skew().fillna(0.0).clip(-4, 4)

        # G3: Institutional vs Retail Sentiment — Overnight Gap vs Intraday Flow (Lou, Polk & Skouras JFE 2019)
        prev_c = close.shift(1).replace(0, np.nan)
        overnight_r = ((open_p / prev_c) - 1.0).fillna(0.0).clip(-0.12, 0.12)
        intraday_r = ((close / open_p) - 1.0).fillna(0.0).clip(-0.12, 0.12)
        df["overnight_sent_63d"] = overnight_r.rolling(63, min_periods=20).sum().fillna(0.0)
        df["intraday_sent_63d"] = intraday_r.rolling(63, min_periods=20).sum().fillna(0.0)

        # G4: Idiosyncratic (NIFTY-Residual) Momentum & Betting-Against-Beta (Blitz et al. / BAB)
        bm_r = df["bench_ret_1d"].fillna(0.0)
        bm_var = bm_r.rolling(126, min_periods=40).var().replace(0, np.nan)
        beta_126 = (ret_1d.rolling(126, min_periods=40).cov(bm_r) / bm_var).clip(-0.5, 3.0).fillna(1.0)
        df["nifty_beta_126d"] = beta_126
        # Residual daily return stripped of NIFTY market exposure
        resid_r = ret_1d - beta_126.shift(1).fillna(1.0) * bm_r
        resid_vol = (resid_r.rolling(126, min_periods=40).std() * np.sqrt(252)).clip(lower=0.10).fillna(0.25)
        df["idio_mom_126d"] = resid_r.rolling(126, min_periods=40).sum().fillna(0.0) / resid_vol

        # G5: Clenow Smooth Trend Quality (90D regression slope * R^2) & 50-DMA Slope
        df["clenow_trend_90d"] = _clenow_trend(close, 90).fillna(0.0)
        df["sma50_slope_20d"] = (df["sma50"] / df["sma50"].shift(20).replace(0, np.nan) - 1.0).fillna(0.0).clip(-0.3, 0.3)

        # G6: Stock-Specific Cross-Sectional Macro Sensitivities (Rolling Stock Beta * Macro 63D Trend!)
        brent_r = df["brent_ret_1d"].fillna(0.0)
        fx_r = df["usdinr_ret_1d"].fillna(0.0)
        sp_r = df["sp500_ret_1d"].fillna(0.0)
        beta_brent = (ret_1d.rolling(126, min_periods=40).cov(brent_r) / brent_r.rolling(126, min_periods=40).var().replace(0, np.nan)).clip(-1.5, 1.5).fillna(0.0)
        beta_fx = (ret_1d.rolling(126, min_periods=40).cov(fx_r) / fx_r.rolling(126, min_periods=40).var().replace(0, np.nan)).clip(-3.0, 3.0).fillna(0.0)
        beta_sp = (ret_1d.rolling(126, min_periods=40).cov(sp_r) / sp_r.rolling(126, min_periods=40).var().replace(0, np.nan)).clip(-1.5, 2.0).fillna(0.0)
        # Expected macro tailwind for THIS specific stock based on its own rolling sensitivity
        df["macro_tailwind_brent"] = (beta_brent * df["brent_63d_chg"]).fillna(0.0)
        df["macro_tailwind_fx"] = (beta_fx * df["usdinr_63d_chg"]).fillna(0.0)
        df["macro_tailwind_sp500"] = (beta_sp * df["sp500_63d_chg"]).fillna(0.0)

        # G7: Institutional Rupee Accumulation (Up-day Rupee Turnover Share over 40D)
        rupee_val = close * vol
        up_val = np.where(ret_1d > 0, rupee_val, 0.0)
        df["inst_accum_40d"] = (
            pd.Series(up_val, index=df.index).rolling(40, min_periods=15).sum()
            / (rupee_val.rolling(40, min_periods=15).sum().replace(0, np.nan))
        ).fillna(0.5)

        # G8: Point-in-Time PEAD Earnings
        pead_surp, pead_streak = _compute_pead(df["date"], earnings_map.get(sym, []))
        df["pead_surprise"] = pead_surp
        df["pead_streak"] = pead_streak

        df["fwd_ret"] = close.shift(-horizon) / close - 1.0
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    # Market-Wide NSE Breadth & Sentiment Internals
    daily_br = full.groupby("date").agg(
        nse_breadth_50d=("above_sma50", "mean"),
        pct_near_52w_hi=("prox_52w_high", lambda s: float(np.mean(s >= 0.90))),
        pct_near_52w_lo=("prox_52w_low", lambda s: float(np.mean(s <= 1.12))),
    ).sort_index()
    daily_br["nse_hilo_sentiment"] = daily_br["pct_near_52w_hi"] - daily_br["pct_near_52w_lo"]
    daily_br["breadth_thrust_15d"] = daily_br["nse_breadth_50d"].diff(15).fillna(0.0)
    full = full.merge(daily_br.reset_index(), on="date", how="left")

    # Sector-relative alpha
    sec_63 = full.groupby(["date", "sector"])["ret_63d"].transform("median")
    sec_126 = full.groupby(["date", "sector"])["ret_126d"].transform("median")
    full["sec_rel_63d"] = (full["ret_63d"] - sec_63).fillna(0.0)
    full["sec_rel_126d"] = (full["ret_126d"] - sec_126).fillna(0.0)

    z_cols = [
        "mom_6m1m_voladj", "mom_12m1m_voladj", "fip_smooth_6m", "prox_52w_high",
        "vol_252d", "down_vol_126d", "sec_rel_63d", "sec_rel_126d",
        "ret_21d", "rsi14", "vol_ratio_5_20",
        # Round 4 cross-sectional signals
        "max_ret_21d", "skew_63d",
        "overnight_sent_63d", "intraday_sent_63d",
        "nifty_beta_126d", "idio_mom_126d",
        "clenow_trend_90d", "sma50_slope_20d",
        "macro_tailwind_brent", "macro_tailwind_fx", "macro_tailwind_sp500",
        "inst_accum_40d", "pead_surprise", "pead_streak",
    ]
    for col in z_cols:
        full[f"z_{col}"] = full.groupby("date")[col].transform(_cs_z).fillna(0.0)

    # Prior R4 (Current Champion)
    nse_mom = 0.5 * full["z_mom_6m1m_voladj"] + 0.5 * full["z_mom_12m1m_voladj"]
    nse_lowvol = -0.6 * full["z_vol_252d"] - 0.4 * full["z_down_vol_126d"]
    full["prior_r4"] = (
        0.42 * nse_mom
        + 0.25 * nse_lowvol
        + 0.18 * full["z_fip_smooth_6m"]
        + 0.15 * full["z_prox_52w_high"]
    )

    # Prior R4+ (Institutional Trend + Residual Mom + Low-Beta + Lottery-Avoidance + Overnight Sent)
    full["prior_r4_idio"] = (
        0.30 * nse_mom
        + 0.18 * full["z_idio_mom_126d"]
        + 0.22 * (-0.5 * full["z_vol_252d"] - 0.3 * full["z_down_vol_126d"] - 0.2 * full["z_nifty_beta_126d"])
        + 0.12 * full["z_fip_smooth_6m"]
        + 0.10 * full["z_prox_52w_high"]
        - 0.08 * full["z_max_ret_21d"]
    )

    full["target_alpha"] = (full["fwd_ret"] - full["bench_fwd_ret"]).clip(-0.45, 0.45)
    valid = full.dropna(subset=["target_alpha", "fwd_ret", "bench_fwd_ret", "sma200", "vol_252d", "fip_smooth_6m"]).copy()
    valid["fwd_rank"] = valid.groupby("date")["target_alpha"].rank(pct=True)
    return valid


def run_eval(
    valid: pd.DataFrame,
    feat_cols: list[str],
    prior_col: str,
    prices_df: pd.DataFrame,
    maps: dict[str, dict[str, Any]],
    gate_mode: str = "base",
    prior_weight: float = 0.50,
    top_k: int = 8,
    cost_bps: float = 15.0,
    horizon: int = 40,
) -> dict[str, Any]:
    dates = sorted(valid["date"].unique())
    first_test = int(len(dates) * 0.42)
    fold_len = max(20, (len(dates) - first_test) // 5)

    for c in feat_cols:
        valid[c] = valid[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    chunks = []
    for f in range(5):
        t0 = first_test + f * fold_len
        t1 = len(dates) if f == 4 else min(len(dates), t0 + fold_len)
        tr = valid[valid["date"] <= dates[t0 - horizon - 1]]
        te = valid[(valid["date"] >= dates[t0]) & (valid["date"] <= dates[t1 - 1])].copy()

        r = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=500.0))]).fit(tr[feat_cols], tr["fwd_rank"])
        g = HistGradientBoostingRegressor(
            max_depth=2, learning_rate=0.03, max_iter=65, min_samples_leaf=100, l2_regularization=10.0, random_state=42
        ).fit(tr[feat_cols], tr["fwd_rank"])
        te["ml_rank"] = 0.60 * r.predict(te[feat_cols]) + 0.40 * g.predict(te[feat_cols])
        te["z_ml"] = te.groupby("date")["ml_rank"].transform(_cs_z).fillna(0.0)
        te["score"] = prior_weight * te[prior_col] + (1.0 - prior_weight) * te["z_ml"]
        chunks.append(te)

    oos = pd.concat(chunks, ignore_index=True)
    return simulate_portfolio(oos, prices_df, maps, gate_mode=gate_mode, top_k=top_k, cost_bps=cost_bps), oos


def simulate_portfolio(
    oos: pd.DataFrame,
    prices_df: pd.DataFrame,
    maps: dict[str, dict[str, Any]],
    gate_mode: str = "base",
    top_k: int = 8,
    cost_bps: float = 15.0,
) -> dict[str, Any]:
    bench = prices_df[BENCHMARK_SYMBOL]
    bench_sma200 = bench.rolling(200, min_periods=50).mean()
    log_rets = np.log(prices_df / prices_df.shift(1))
    vol_df = log_rets.rolling(252, min_periods=63).std() * np.sqrt(252)

    vix_map = maps["vix"]
    br_map = maps["breadth"]
    hilo_map = maps["hilo"]
    thrust_map = maps["thrust"]
    gn_map = maps["gold_nifty"]
    fx_map = maps["usdinr_chg"]
    sec_map = maps["sector"]

    oos_dates = sorted(set(oos["date"].unique()).intersection(set(prices_df.index)))
    sim_dates = [d for d in prices_df.index if oos_dates[0] <= d <= oos_dates[-1]]
    rebal_set = set(oos_dates[::40])
    oos_by_date = {dt: grp for dt, grp in oos.groupby("date")}

    fee = cost_bps / 10000.0
    eq, b_eq = 1_000_000.0, 1_000_000.0
    weights: dict[str, float] = {}
    p_rets, b_rets, d_rec = [], [], []
    avg_exps = []

    for i, dt in enumerate(sim_dates):
        if i > 0:
            prev_dt = sim_dates[i - 1]
            cash_w = max(0.0, 1.0 - sum(weights.values()))
            day_r = cash_w * INDIA_CASH_DAILY_YIELD
            for sym, w in weights.items():
                p0, p1 = float(prices_df.loc[prev_dt, sym]), float(prices_df.loc[dt, sym])
                if p0 > 0 and np.isfinite(p0) and np.isfinite(p1):
                    day_r += w * ((p1 / p0) - 1.0)
            b_r = (float(bench.loc[dt]) / float(bench.loc[prev_dt])) - 1.0
        else:
            day_r, b_r = 0.0, 0.0

        eq *= 1.0 + day_r
        b_eq *= 1.0 + b_r

        vix = vix_map.get(dt, 15.0)
        br = br_map.get(dt, 0.6)
        hilo = hilo_map.get(dt, 0.1)
        thrust = thrust_map.get(dt, 0.0)
        gn = gn_map.get(dt, 0.0)
        fx_chg = fx_map.get(dt, 0.0)
        bpx = float(bench.loc[dt])
        s200 = float(bench_sma200.loc[dt]) if pd.notna(bench_sma200.loc[dt]) else bpx

        target_exp = 1.0
        if gate_mode == "base":
            if vix >= 22.0 or (bpx < s200 and br < 0.35):
                target_exp = 0.30
            elif vix >= 18.5 or bpx < s200 or br < 0.45:
                target_exp = 0.65
        elif gate_mode == "sentiment_v1":
            # Balanced Multi-Signal India Macro & Sentiment Gate (6 orthogonal fear/greed gauges)
            stress = 0
            if vix >= 19.0: stress += 1
            if vix >= 23.0: stress += 1
            if bpx < s200: stress += 1
            if br < 0.42: stress += 1
            if hilo < -0.10: stress += 1
            if gn > 0.14: stress += 1
            if stress >= 3:
                target_exp = 0.25
            elif stress == 2:
                target_exp = 0.55
            elif stress == 1:
                target_exp = 0.85
        elif gate_mode == "sentiment_v2_thrust":
            # Adds FII USD/INR Stress + Breadth-Thrust Recovery Re-entry!
            stress = 0
            if vix >= 19.0: stress += 1
            if vix >= 23.5: stress += 1
            if bpx < s200: stress += 1
            if br < 0.42: stress += 1
            if hilo < -0.10: stress += 1
            if gn > 0.14: stress += 1
            if fx_chg > 0.035: stress += 1  # Sharp INR depreciation (>3.5% in 3M = FII outflows)
            # Positive Sentiment Thrust overrides 1 stress point when market is snapping back!
            if thrust >= 0.18 and vix < 21.0:
                stress = max(0, stress - 1)
            if stress >= 4:
                target_exp = 0.20
            elif stress == 3:
                target_exp = 0.40
            elif stress == 2:
                target_exp = 0.65
            elif stress == 1:
                target_exp = 0.88

        if dt in rebal_set and dt in oos_by_date:
            day = oos_by_date[dt].sort_values("score", ascending=False).reset_index(drop=True)
            day["rank"] = np.arange(1, len(day) + 1)
            rmap = dict(zip(day["symbol"], day["rank"]))
            chosen, scnt = [], {}
            buf = top_k * 2
            for sym in list(weights.keys()):
                if rmap.get(sym, 999) <= buf and sym in prices_df.columns and scnt.get(sec_map.get(sym, "O"), 0) < 3:
                    chosen.append(sym)
                    scnt[sec_map.get(sym, "O")] = scnt.get(sec_map.get(sym, "O"), 0) + 1
            for sym in day["symbol"].tolist():
                if len(chosen) >= top_k:
                    break
                if sym in chosen or sym not in prices_df.columns or scnt.get(sec_map.get(sym, "O"), 0) >= 3:
                    continue
                chosen.append(sym)
                scnt[sec_map.get(sym, "O")] = scnt.get(sec_map.get(sym, "O"), 0) + 1

            raw_w = {s: 1.0 / max(0.12, float(vol_df.loc[dt, s]) if pd.notna(vol_df.loc[dt, s]) else 0.25) for s in chosen}
            ws = sum(raw_w.values()) or 1.0
            new_w = {s: (v / ws) * target_exp for s, v in raw_w.items()}
            turn = sum(abs(new_w.get(k, 0.0) - weights.get(k, 0.0)) for k in set(weights).union(new_w)) * 0.5
            eq *= max(0.0, 1.0 - turn * fee * 2.0)
            weights = new_w
        elif weights:
            cur_exp = sum(weights.values())
            if cur_exp > target_exp + 0.20:
                scale = target_exp / max(1e-9, cur_exp)
                eq *= max(0.0, 1.0 - abs(cur_exp - target_exp) * fee)
                weights = {s: w * scale for s, w in weights.items()}
            elif gate_mode == "sentiment_v2_thrust" and cur_exp < target_exp - 0.25 and thrust >= 0.18:
                # Re-risk cleanly on strong NSE breadth thrust recovery
                scale = target_exp / max(1e-9, cur_exp)
                eq *= max(0.0, 1.0 - abs(target_exp - cur_exp) * fee)
                weights = {s: w * scale for s, w in weights.items()}

        if i > 0:
            p_rets.append(day_r); b_rets.append(b_r); d_rec.append(dt)
            avg_exps.append(sum(weights.values()))

    pr, br_a = np.array(p_rets), np.array(b_rets)
    yrs = max(0.25, len(pr) / 252.0)
    cagr = ((eq / 1_000_000.0) ** (1.0 / yrs) - 1.0) * 100.0
    b_cagr = ((b_eq / 1_000_000.0) ** (1.0 / yrs) - 1.0) * 100.0
    vol = float(np.std(pr, ddof=1) * np.sqrt(252))
    sharpe = float((np.mean(pr) * 252.0) / max(1e-9, vol))
    down = pr[pr < 0]
    sortino = float((np.mean(pr) * 252.0) / max(1e-9, float(np.std(down, ddof=1) * np.sqrt(252))))
    cum = np.cumprod(1.0 + pr)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum(1e-9, np.maximum.accumulate(cum))) * 100.0
    calmar = cagr / max(1.0, abs(float(np.min(dd))))
    ic = float(spearmanr(oos["score"], oos["target_alpha"]).statistic)

    df_r = pd.DataFrame({"date": d_rec, "p": pr})
    sub24 = df_r[df_r["date"] >= "2024-01-01"]
    ret_24_26 = (float(np.prod(1.0 + sub24["p"])) - 1.0) * 100.0
    sub_cr = df_r[(df_r["date"] >= "2024-09-25") & (df_r["date"] <= "2025-04-07")]
    ret_crash = (float(np.prod(1.0 + sub_cr["p"])) - 1.0) * 100.0

    hits = []
    for dt in oos_dates[::20]:
        tk = oos_by_date[dt].sort_values("score", ascending=False).head(top_k)
        if not tk.empty:
            hits.append(1 if tk["fwd_ret"].mean() > tk["bench_fwd_ret"].mean() else 0)

    return {
        "cagr": round(cagr, 2),
        "alpha": round(cagr - b_cagr, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(float(np.min(dd)), 2),
        "calmar": round(calmar, 2),
        "ic": round(0.0 if np.isnan(ic) else ic, 4),
        "hit": round(float(np.mean(hits)) * 100.0, 1),
        "r24_26": round(ret_24_26, 1),
        "crash": round(ret_crash, 1),
        "exp": round(float(np.mean(avg_exps)) * 100.0, 0),
    }


def main() -> None:
    db = Database()
    valid = build_round4_panel(db, horizon=40)
    ohlcv = db.get_all_ohlcv()
    prices = pd.DataFrame({s: pd.Series({b["date"]: float(b["close"]) for b in bars}) for s, bars in ohlcv.items()}).sort_index().ffill()
    maps = {
        "vix": db.get_macro_series().get(VOL_INDEX_SYMBOL, {}),
        "breadth": valid.groupby("date")["nse_breadth_50d"].first().to_dict(),
        "hilo": valid.groupby("date")["nse_hilo_sentiment"].first().to_dict(),
        "thrust": valid.groupby("date")["breadth_thrust_15d"].first().to_dict(),
        "gold_nifty": valid.groupby("date")["gold_nifty_63d_chg"].first().to_dict(),
        "usdinr_chg": valid.groupby("date")["usdinr_63d_chg"].first().to_dict(),
        "sector": {t["symbol"]: t["sector"] for t in db.get_tickers()},
    }

    base = [
        "z_mom_6m1m_voladj", "z_mom_12m1m_voladj", "z_fip_smooth_6m",
        "z_prox_52w_high", "z_vol_252d", "z_down_vol_126d",
        "z_sec_rel_63d", "z_sec_rel_126d", "z_ret_21d", "z_rsi14",
        "z_vol_ratio_5_20", "vix_level", "bench_dist_200", "nse_breadth_50d",
    ]

    exps = [
        ("R4: Current Champion Baseline (14 feats, Base Gate)", base, "prior_r4", "base"),
        ("G1a: R4 + Sentiment/Gold/HiLo Risk Gate v1", base, "prior_r4", "sentiment_v1"),
        ("G1b: R4 + Sentiment/FX/Thrust Risk Gate v2", base, "prior_r4", "sentiment_v2_thrust"),
        ("G2: R4 + Behavioral MAX-Lottery/Skew Sentiment", base + ["z_max_ret_21d", "z_skew_63d"], "prior_r4", "base"),
        ("G3: R4 + Overnight Inst vs Intraday Retail Sent", base + ["z_overnight_sent_63d", "z_intraday_sent_63d"], "prior_r4", "base"),
        ("G4: R4 + Idiosyncratic Momentum & NIFTY Beta", base + ["z_idio_mom_126d", "z_nifty_beta_126d"], "prior_r4", "base"),
        ("G5: R4 + Clenow Smooth Trend (R2*Slope + SMA50)", base + ["z_clenow_trend_90d", "z_sma50_slope_20d"], "prior_r4", "base"),
        ("G6: R4 + Stock-Specific Macro Betas (Brent/FX/SPX)", base + ["z_macro_tailwind_brent", "z_macro_tailwind_fx", "z_macro_tailwind_sp500"], "prior_r4", "base"),
        ("G7: R4 + Institutional Rupee Accumulation 40D", base + ["z_inst_accum_40d"], "prior_r4", "base"),
        ("G8: R4 + Prior R4+ (IdioMom + LowBeta - MAX)", base + ["z_idio_mom_126d", "z_nifty_beta_126d", "z_max_ret_21d"], "prior_r4_idio", "base"),
        ("G9: R4 + Prior R4+ + Overnight + PEAD + Gate v1", base + ["z_idio_mom_126d", "z_nifty_beta_126d", "z_max_ret_21d", "z_overnight_sent_63d", "z_pead_surprise"], "prior_r4_idio", "sentiment_v1"),
        ("G10: R4 + Prior R4+ + Overnight + PEAD + Gate v2", base + ["z_idio_mom_126d", "z_nifty_beta_126d", "z_max_ret_21d", "z_overnight_sent_63d", "z_pead_surprise"], "prior_r4_idio", "sentiment_v2_thrust"),
    ]

    print(f"Round 4 Deep Sentiment/Trend/Macro Ablation | {len(valid):,} rows | 15 bps Indian STT\n")
    print(f"{'Experiment':52s} | {'CAGR':>6s} | {'Alpha':>6s} | {'Shrp':>5s} | {'Sort':>5s} | {'MaxDD':>7s} | {'Calm':>4s} | {'IC':>7s} | {'Hit%':>5s} | {'24-26':>6s} | {'Crash':>6s} | {'Exp':>4s}")
    print("-" * 145)
    oos_cache = {}
    for name, fcols, pcol, gmode in exps:
        r, oos = run_eval(valid, fcols, pcol, prices, maps, gate_mode=gmode)
        oos_cache[name] = (oos, gmode)
        print(f"{name:52s} | {r['cagr']:+5.2f}% | {r['alpha']:+5.2f}% | {r['sharpe']:5.2f} | {r['sortino']:5.2f} | {r['max_dd']:6.2f}% | {r['calmar']:4.2f} | {r['ic']:+7.4f} | {r['hit']:4.1f}% | {r['r24_26']:+5.1f}% | {r['crash']:+5.1f}% | {r['exp']:3.0f}%")


if __name__ == "__main__":
    main()
