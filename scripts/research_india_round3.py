"""
EquiLens India Quant Research Lab — Round 3 (Ablation Study):
Tests adding Sentiment, Trend Quality, Sector Rotation, Point-in-Time PEAD Earnings Surprises,
and Global Intermarket Macro (Brent Crude, Gold/NIFTY Fear Gauge, US 10Y Yield, DXY, S&P 500)
strictly out-of-sample against the R4 Production Champion (+24.57% CAGR, Sharpe 1.66).
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


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    # Signed directional trend strength: +ADX when +DI > -DI, -ADX when -DI > +DI
    return adx * np.where(plus_di >= minus_di, 1.0, -1.0)


def _compute_point_in_time_pead(dates: pd.Series, earn_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Strictly causal point-in-time PEAD: only uses earnings reported STRICTLY before bar date."""
    n = len(dates)
    surp_arr = np.zeros(n, dtype=float)
    streak_arr = np.zeros(n, dtype=float)
    if not earn_rows:
        return surp_arr, streak_arr

    # Sort earnings ascending by date, keep only reported surprises
    valid_e = sorted(
        [(e["earnings_date"][:10], float(e["surprise_pct"])) for e in earn_rows if e.get("surprise_pct") is not None],
        key=lambda x: x[0],
    )
    if not valid_e:
        return surp_arr, streak_arr

    e_dates = [pd.Timestamp(x[0]) for x in valid_e]
    e_vals = [x[1] for x in valid_e]
    bar_ts = pd.to_datetime(dates).tolist()

    ptr = -1
    cur_streak = 0
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
            # Exponential decay over ~75 calendar days (~50 trading days)
            decay = float(np.exp(-max(0, days_since) / 65.0))
            surp_arr[i] = float(np.clip(e_vals[ptr], -40.0, 40.0)) * decay
            streak_arr[i] = float(np.clip(cur_streak, -4, 6))
    return surp_arr, streak_arr


def build_round3_panel(db: Database, horizon: int = 40) -> pd.DataFrame:
    ohlcv_map = db.get_all_ohlcv()
    tickers_meta = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    earnings_map = db.get_all_earnings()
    macro = db.get_macro_series()

    bench_bars = ohlcv_map[BENCHMARK_SYMBOL]
    bdf = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date").reset_index(drop=True)
    bdf["bench_fwd_ret"] = bdf["close"].shift(-horizon) / bdf["close"] - 1.0
    bdf["bench_sma200"] = bdf["close"].rolling(200, min_periods=50).mean()
    bdf["bench_dist_200"] = (bdf["close"] / bdf["bench_sma200"]) - 1.0
    bdf["bench_ret_63d"] = bdf["close"].pct_change(63)

    # Merge Global Macro & Sentiment Intermarket Series
    macro_aliases = {
        VOL_INDEX_SYMBOL: "vix_level",
        FX_MACRO_SYMBOL: "usdinr",
        "BZ=F": "brent",
        "GC=F": "gold",
        "^TNX": "us10y",
        "DX-Y.NYB": "dxy",
        "^GSPC": "sp500",
    }
    for m_sym, col in macro_aliases.items():
        s = pd.Series(macro.get(m_sym, {}), name=col)
        bdf = bdf.merge(s.rename_axis("date").reset_index(), on="date", how="left")
        bdf[col] = bdf[col].ffill().bfill()

    bdf["vix_level"] = bdf["vix_level"].fillna(15.0)
    bdf["vix_20d_chg"] = bdf["vix_level"].pct_change(20).fillna(0.0)
    bdf["brent_63d_chg"] = bdf["brent"].pct_change(63).fillna(0.0).clip(-0.5, 0.5)
    bdf["usdinr_63d_chg"] = bdf["usdinr"].pct_change(63).fillna(0.0).clip(-0.15, 0.15)
    bdf["us10y_63d_chg"] = bdf["us10y"].diff(63).fillna(0.0).clip(-1.5, 1.5)
    bdf["dxy_63d_chg"] = bdf["dxy"].pct_change(63).fillna(0.0).clip(-0.15, 0.15)
    bdf["sp500_63d_chg"] = bdf["sp500"].pct_change(63).fillna(0.0).clip(-0.3, 0.3)
    # India Gold-to-NIFTY fear/risk-off sentiment ratio (rising gold vs NIFTY = risk-off fear)
    gold_nifty = bdf["gold"] / bdf["close"].replace(0, np.nan)
    bdf["gold_nifty_63d_chg"] = gold_nifty.pct_change(63).fillna(0.0).clip(-0.4, 0.4)

    macro_cols = [
        "date", "bench_fwd_ret", "bench_dist_200", "vix_level", "vix_20d_chg",
        "brent_63d_chg", "usdinr_63d_chg", "us10y_63d_chg", "dxy_63d_chg",
        "sp500_63d_chg", "gold_nifty_63d_chg",
    ]
    macro_lookup = bdf[macro_cols]

    frames = []
    for sym, bars in ohlcv_map.items():
        if sym == BENCHMARK_SYMBOL or len(bars) < 260:
            continue
        df = compute_indicator_frame(bars, bench_bars)
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        vol = df["volume"].astype(float)
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

        # NEW 1: Institutional Money Flow (Chaikin Money Flow 20D) & Up/Down Volume Ratio
        hl_span = (high - low).replace(0, np.nan)
        mfm = ((close - low) - (high - close)) / hl_span
        mfv = mfm.fillna(0.0) * vol
        df["cmf_20d"] = (mfv.rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)).fillna(0.0).clip(-1, 1)

        up_v = np.where(close >= close.shift(1), vol, 0.0)
        dn_v = np.where(close < close.shift(1), vol, 0.0)
        df["updown_vol_20d"] = (
            pd.Series(up_v, index=df.index).rolling(20).sum()
            / (pd.Series(dn_v, index=df.index).rolling(20).sum() + 1.0)
        ).pipe(np.log1p).clip(-3, 3)

        # NEW 2: Signed Directional Trend Strength (ADX-14)
        df["signed_adx14"] = _compute_adx(high, low, close, 14).fillna(0.0)

        # NEW 3: Point-in-Time PEAD Earnings Surprise & Beat Streak
        pead_surp, pead_streak = _compute_point_in_time_pead(df["date"], earnings_map.get(sym, []))
        df["pead_surprise"] = pead_surp
        df["pead_streak"] = pead_streak

        df["fwd_ret"] = close.shift(-horizon) / close - 1.0
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = full.merge(macro_lookup, on="date", how="left")

    # Market-Wide NSE Breadth & Sentiment Internals (strictly causal per date)
    daily_br = full.groupby("date").agg(
        nse_breadth_50d=("above_sma50", "mean"),
        pct_near_52w_hi=("prox_52w_high", lambda s: float(np.mean(s >= 0.90))),
        pct_near_52w_lo=("prox_52w_low", lambda s: float(np.mean(s <= 1.12))),
    ).sort_index()
    daily_br["nse_hilo_sentiment"] = daily_br["pct_near_52w_hi"] - daily_br["pct_near_52w_lo"]
    daily_br["breadth_thrust_20d"] = daily_br["nse_breadth_50d"].diff(20).fillna(0.0)
    full = full.merge(daily_br.reset_index(), on="date", how="left")

    # Sector-Level Momentum (Sector Rotation Trend!) + Sector-Relative Alpha
    full["sector_mom_63d"] = full.groupby(["date", "sector"])["ret_63d"].transform("median").fillna(0.0)
    full["sector_mom_126d"] = full.groupby(["date", "sector"])["ret_126d"].transform("median").fillna(0.0)
    full["sec_rel_63d"] = (full["ret_63d"] - full["sector_mom_63d"]).fillna(0.0)
    full["sec_rel_126d"] = (full["ret_126d"] - full["sector_mom_126d"]).fillna(0.0)

    # Cross-sectional z-scores
    z_cols = [
        "mom_6m1m_voladj", "mom_12m1m_voladj", "fip_smooth_6m", "prox_52w_high",
        "vol_252d", "down_vol_126d", "sec_rel_63d", "sec_rel_126d",
        "ret_21d", "rsi14", "vol_ratio_5_20",
        # New features
        "sector_mom_63d", "sector_mom_126d", "cmf_20d", "updown_vol_20d",
        "signed_adx14", "pead_surprise", "pead_streak",
    ]
    for col in z_cols:
        full[f"z_{col}"] = full.groupby("date")[col].transform(_cs_z).fillna(0.0)

    # Prior A: Current R4 NSE Factor Prior
    nse_mom = 0.5 * full["z_mom_6m1m_voladj"] + 0.5 * full["z_mom_12m1m_voladj"]
    nse_lowvol = -0.6 * full["z_vol_252d"] - 0.4 * full["z_down_vol_126d"]
    full["prior_r4"] = (
        0.42 * nse_mom
        + 0.25 * nse_lowvol
        + 0.18 * full["z_fip_smooth_6m"]
        + 0.15 * full["z_prox_52w_high"]
    )

    # Prior B: Enriched India Institutional Prior (+ Sector Rotation + Money Flow + PEAD Earnings)
    full["prior_r5_enriched"] = (
        0.34 * nse_mom
        + 0.22 * nse_lowvol
        + 0.14 * full["z_fip_smooth_6m"]
        + 0.10 * full["z_prox_52w_high"]
        + 0.10 * (0.5 * full["z_sector_mom_63d"] + 0.5 * full["z_sector_mom_126d"])
        + 0.05 * full["z_cmf_20d"]
        + 0.05 * full["z_pead_surprise"]
    )

    full["target_alpha"] = (full["fwd_ret"] - full["bench_fwd_ret"]).clip(-0.45, 0.45)
    valid = full.dropna(subset=["target_alpha", "fwd_ret", "bench_fwd_ret", "sma200", "vol_252d", "fip_smooth_6m"]).copy()
    valid["fwd_rank"] = valid.groupby("date")["target_alpha"].rank(pct=True)
    return valid


def evaluate_experiment(
    valid: pd.DataFrame,
    feat_cols: list[str],
    prior_col: str,
    prices_df: pd.DataFrame,
    vix_map: dict[str, float],
    breadth_map: dict[str, float],
    hilo_map: dict[str, float],
    gold_nifty_map: dict[str, float],
    sector_map: dict[str, str],
    enhanced_gate: bool = False,
    prior_weight: float = 0.50,
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

    # Simulate exact R4 Top-8 portfolio (40D, Buf 16, Sec 3, Inv-Vol, 15 bps STT)
    bench = prices_df[BENCHMARK_SYMBOL]
    bench_sma200 = bench.rolling(200, min_periods=50).mean()
    log_rets = np.log(prices_df / prices_df.shift(1))
    vol_df = log_rets.rolling(252, min_periods=63).std() * np.sqrt(252)

    oos_dates = sorted(set(oos["date"].unique()).intersection(set(prices_df.index)))
    sim_dates = [d for d in prices_df.index if oos_dates[0] <= d <= oos_dates[-1]]
    rebal_set = set(oos_dates[::40])
    oos_by_date = {dt: grp for dt, grp in oos.groupby("date")}

    fee = 15.0 / 10000.0
    eq, b_eq = 1_000_000.0, 1_000_000.0
    weights: dict[str, float] = {}
    p_rets, b_rets, d_rec = [], [], []

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

        # Risk Gate (Base vs Enhanced Sentiment + Gold/NIFTY + Hi-Lo Gate)
        vix = vix_map.get(dt, 15.0)
        br = breadth_map.get(dt, 0.6)
        hilo = hilo_map.get(dt, 0.1)
        gn = gold_nifty_map.get(dt, 0.0)
        bpx = float(bench.loc[dt])
        s200 = float(bench_sma200.loc[dt]) if pd.notna(bench_sma200.loc[dt]) else bpx

        target_exp = 1.0
        if enhanced_gate:
            stress_pts = 0
            if vix >= 19.0: stress_pts += 1
            if vix >= 23.0: stress_pts += 1
            if bpx < s200: stress_pts += 1
            if br < 0.42: stress_pts += 1
            if hilo < -0.10: stress_pts += 1  # More stocks near 52W low than 52W high (fear sentiment)
            if gn > 0.14: stress_pts += 1     # Gold surging >14% vs NIFTY over 3M (flight to safety)
            if stress_pts >= 3:
                target_exp = 0.25
            elif stress_pts == 2:
                target_exp = 0.55
            elif stress_pts == 1:
                target_exp = 0.85
        else:
            if vix >= 22.0 or (bpx < s200 and br < 0.35):
                target_exp = 0.30
            elif vix >= 18.5 or bpx < s200 or br < 0.45:
                target_exp = 0.65

        if dt in rebal_set and dt in oos_by_date:
            day = oos_by_date[dt].sort_values("score", ascending=False).reset_index(drop=True)
            day["rank"] = np.arange(1, len(day) + 1)
            rmap = dict(zip(day["symbol"], day["rank"]))
            chosen, scnt = [], {}
            for sym in list(weights.keys()):
                if rmap.get(sym, 999) <= 16 and sym in prices_df.columns and scnt.get(sector_map.get(sym, "O"), 0) < 3:
                    chosen.append(sym)
                    scnt[sector_map.get(sym, "O")] = scnt.get(sector_map.get(sym, "O"), 0) + 1
            for sym in day["symbol"].tolist():
                if len(chosen) >= 8:
                    break
                if sym in chosen or sym not in prices_df.columns or scnt.get(sector_map.get(sym, "O"), 0) >= 3:
                    continue
                chosen.append(sym)
                scnt[sector_map.get(sym, "O")] = scnt.get(sector_map.get(sym, "O"), 0) + 1

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

        if i > 0:
            p_rets.append(day_r); b_rets.append(b_r); d_rec.append(dt)

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
    ic = float(spearmanr(oos["score"], oos["target_alpha"]).statistic)

    df_r = pd.DataFrame({"date": d_rec, "p": pr})
    sub24 = df_r[df_r["date"] >= "2024-01-01"]
    ret_24_26 = (float(np.prod(1.0 + sub24["p"])) - 1.0) * 100.0
    sub_cr = df_r[(df_r["date"] >= "2024-09-25") & (df_r["date"] <= "2025-04-07")]
    ret_crash = (float(np.prod(1.0 + sub_cr["p"])) - 1.0) * 100.0

    hits = []
    for dt in oos_dates[::20]:
        tk = oos_by_date[dt].sort_values("score", ascending=False).head(8)
        if not tk.empty:
            hits.append(1 if tk["fwd_ret"].mean() > tk["bench_fwd_ret"].mean() else 0)

    return {
        "cagr": round(cagr, 2),
        "alpha": round(cagr - b_cagr, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(float(np.min(dd)), 2),
        "ic": round(0.0 if np.isnan(ic) else ic, 4),
        "hit": round(float(np.mean(hits)) * 100.0, 1),
        "r24_26": round(ret_24_26, 1),
        "crash": round(ret_crash, 1),
    }


def main() -> None:
    db = Database()
    valid = build_round3_panel(db, horizon=40)
    ohlcv = db.get_all_ohlcv()
    prices = pd.DataFrame({s: pd.Series({b["date"]: float(b["close"]) for b in bars}) for s, bars in ohlcv.items()}).sort_index().ffill()
    vix_map = db.get_macro_series().get(VOL_INDEX_SYMBOL, {})
    breadth_map = valid.groupby("date")["nse_breadth_50d"].first().to_dict()
    hilo_map = valid.groupby("date")["nse_hilo_sentiment"].first().to_dict()
    gn_map = valid.groupby("date")["gold_nifty_63d_chg"].first().to_dict()
    sector_map = {t["symbol"]: t["sector"] for t in db.get_tickers()}

    base_feats = [
        "z_mom_6m1m_voladj", "z_mom_12m1m_voladj", "z_fip_smooth_6m",
        "z_prox_52w_high", "z_vol_252d", "z_down_vol_126d",
        "z_sec_rel_63d", "z_sec_rel_126d", "z_ret_21d", "z_rsi14",
        "z_vol_ratio_5_20", "vix_level", "bench_dist_200", "nse_breadth_50d",
    ]
    sector_flow_feats = ["z_sector_mom_63d", "z_sector_mom_126d", "z_cmf_20d", "z_updown_vol_20d", "z_signed_adx14"]
    pead_feats = ["z_pead_surprise", "z_pead_streak"]
    macro_sent_feats = [
        "brent_63d_chg", "usdinr_63d_chg", "us10y_63d_chg", "dxy_63d_chg",
        "sp500_63d_chg", "gold_nifty_63d_chg", "nse_hilo_sentiment", "breadth_thrust_20d",
    ]

    exps = [
        ("R4: Current Champion Baseline (14 feats)", base_feats, "prior_r4", False, 0.50),
        ("R5: + Sector Rotation + MoneyFlow + ADX Trend", base_feats + sector_flow_feats, "prior_r4", False, 0.50),
        ("R6: + Point-in-Time PEAD Earnings Surprises", base_feats + pead_feats, "prior_r4", False, 0.50),
        ("R7: + Global Macro (Brent/Gold/US10Y/DXY) + Sent", base_feats + macro_sent_feats, "prior_r4", False, 0.50),
        ("R8: ALL Groups Combined (29 feats, Base Prior)", base_feats + sector_flow_feats + pead_feats + macro_sent_feats, "prior_r4", False, 0.50),
        ("R9: ALL Groups + Enriched Prior (Sec+CMF+PEAD)", base_feats + sector_flow_feats + pead_feats + macro_sent_feats, "prior_r5_enriched", False, 0.50),
        ("R10: R9 + Sentiment/Gold/HiLo Dynamic Risk Gate", base_feats + sector_flow_feats + pead_feats + macro_sent_feats, "prior_r5_enriched", True, 0.50),
        ("R11: SectorFlow + PEAD + Enhanced Sentiment Gate", base_feats + sector_flow_feats + pead_feats + ["nse_hilo_sentiment", "gold_nifty_63d_chg", "brent_63d_chg"], "prior_r5_enriched", True, 0.50),
    ]

    print(f"Round 3 Ablation Study | {len(valid):,} OOS-eligible rows | 15 bps Indian STT+Brokerage\n")
    print(f"{'Experiment':49s} | {'CAGR':>6s} | {'Alpha':>6s} | {'Shrp':>5s} | {'Sort':>5s} | {'MaxDD':>7s} | {'IC':>7s} | {'Hit%':>5s} | {'24-26':>6s} | {'Crash':>6s}")
    print("-" * 128)
    for name, fcols, pcol, egate, pw in exps:
        r = evaluate_experiment(valid, fcols, pcol, prices, vix_map, breadth_map, hilo_map, gn_map, sector_map, enhanced_gate=egate, prior_weight=pw)
        print(f"{name:49s} | {r['cagr']:+5.2f}% | {r['alpha']:+5.2f}% | {r['sharpe']:5.2f} | {r['sortino']:5.2f} | {r['max_dd']:6.2f}% | {r['ic']:+7.4f} | {r['hit']:4.1f}% | {r['r24_26']:+5.1f}% | {r['crash']:+5.1f}%")


if __name__ == "__main__":
    main()
