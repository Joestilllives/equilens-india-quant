"""
EquiLens India Quant Research Lab — Round 6 (Quant Engine & Risk Management Upgrade):
Strictly out-of-sample ablation against our promoted V5 Champion (+27.76% prod / +27.04% 4-offset avg)
across 7 institutional process upgrades:
  A. Multi-Horizon Blended Rank Target (20D + 40D + 60D, strict 60D embargo)
  B. Exponential Time-Decay Sample Weighting (2.5Y / 3.5Y half-life)
  C. Regime-Adaptive Factor Prior (smooth VIX tilt from Momentum -> Low-Vol/Quality)
  D. Correlation-Aware Portfolio Weighting (Inv-Vol*Corr Penalty, Ledoit-Wolf Shrinkage Min-Var, Sector Cap 2)
  E. Ex-Ante Portfolio Volatility Targeting (14% / 16% annualized vol cap)
  F. Stock-Level Trailing Crash Stop-Loss (-12%, -15%, -3.5xATR Chandelier -> LiquidBees cash)
  G. Payoff Probability Classifier Redesign (diagnose & fix 0.473 TB AUC -> Clean Swing Win / Symmetric TB / Meta-Label)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import (
    BENCHMARK_SYMBOL,
    BRENT_MACRO_SYMBOL,
    FX_MACRO_SYMBOL,
    GOLD_MACRO_SYMBOL,
    SP500_MACRO_SYMBOL,
    VOL_INDEX_SYMBOL,
)
from backend.database import Database
from backend.indicators import compute_indicator_frame
from backend.ml_predictor import FEATURE_COLS, _cs_zscore, compute_triple_barrier_labels

warnings.filterwarnings("ignore")

INDIA_CASH_DAILY_YIELD = (1.062 ** (1.0 / 252.0)) - 1.0


def build_v5_panel_multi_horizon(db: Database) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build strictly causal V5 19-feature panel with 20D, 40D, 60D forward targets & clean payoff labels."""
    ohlcv_map = db.get_all_ohlcv()
    tickers_meta = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    macro = db.get_macro_series()
    bench_bars = ohlcv_map[BENCHMARK_SYMBOL]

    bdf = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date").reset_index(drop=True)
    for h in (20, 40, 60):
        bdf[f"bench_fwd_{h}"] = bdf["close"].shift(-h) / bdf["close"] - 1.0
    bdf["bench_sma200"] = bdf["close"].rolling(200, min_periods=50).mean()
    bdf["bench_dist_200"] = (bdf["close"] / bdf["bench_sma200"]) - 1.0

    for m_sym, col, default_v in [
        (VOL_INDEX_SYMBOL, "vix_level", 15.0),
        (FX_MACRO_SYMBOL, "usdinr", 85.0),
        (BRENT_MACRO_SYMBOL, "brent", 75.0),
        (GOLD_MACRO_SYMBOL, "gold", 2000.0),
        (SP500_MACRO_SYMBOL, "sp500", 5000.0),
    ]:
        s = pd.Series(macro.get(m_sym, {}), name=col, dtype=float)
        bdf = bdf.merge(s.rename_axis("date").reset_index(), on="date", how="left")
        bdf[col] = bdf[col].ffill().bfill().fillna(default_v)

    bdf["brent_ret_1d"] = bdf["brent"].pct_change().fillna(0.0).clip(-0.15, 0.15)
    bdf["usdinr_ret_1d"] = bdf["usdinr"].pct_change().fillna(0.0).clip(-0.05, 0.05)
    bdf["sp500_ret_1d"] = bdf["sp500"].pct_change().fillna(0.0).clip(-0.10, 0.10)
    bdf["brent_63d_chg"] = bdf["brent"].pct_change(63).fillna(0.0).clip(-0.5, 0.5)
    bdf["usdinr_63d_chg"] = bdf["usdinr"].pct_change(63).fillna(0.0).clip(-0.15, 0.15)
    bdf["sp500_63d_chg"] = bdf["sp500"].pct_change(63).fillna(0.0).clip(-0.3, 0.3)
    bdf["gold_nifty_63d_chg"] = (bdf["gold"] / bdf["close"].replace(0, np.nan)).pct_change(63).fillna(0.0).clip(-0.4, 0.4)

    macro_lookup = bdf[
        [
            "date", "bench_fwd_20", "bench_fwd_40", "bench_fwd_60",
            "vix_level", "bench_dist_200",
            "brent_ret_1d", "usdinr_ret_1d", "sp500_ret_1d",
            "brent_63d_chg", "usdinr_63d_chg", "sp500_63d_chg", "gold_nifty_63d_chg",
        ]
    ]

    frames = []
    atr_map: dict[str, pd.Series] = {}
    for sym, bars in ohlcv_map.items():
        if sym == BENCHMARK_SYMBOL or len(bars) < 260:
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

        df["max_ret_21d"] = ret_1d.rolling(21, min_periods=10).max().fillna(0.0)
        prev_c = close.shift(1).replace(0, np.nan)
        overnight_r = ((open_p / prev_c) - 1.0).fillna(0.0).clip(-0.12, 0.12)
        df["overnight_sent_63d"] = overnight_r.rolling(63, min_periods=20).sum().fillna(0.0)

        brent_r, fx_r, sp_r = df["brent_ret_1d"].fillna(0.0), df["usdinr_ret_1d"].fillna(0.0), df["sp500_ret_1d"].fillna(0.0)
        b_brent = (ret_1d.rolling(126, min_periods=40).cov(brent_r) / brent_r.rolling(126, min_periods=40).var().replace(0, np.nan)).clip(-1.5, 1.5).fillna(0.0)
        b_fx = (ret_1d.rolling(126, min_periods=40).cov(fx_r) / fx_r.rolling(126, min_periods=40).var().replace(0, np.nan)).clip(-3.0, 3.0).fillna(0.0)
        b_sp = (ret_1d.rolling(126, min_periods=40).cov(sp_r) / sp_r.rolling(126, min_periods=40).var().replace(0, np.nan)).clip(-1.5, 2.0).fillna(0.0)
        df["macro_tailwind_brent"] = (b_brent * df["brent_63d_chg"].fillna(0.0)).fillna(0.0)
        df["macro_tailwind_fx"] = (b_fx * df["usdinr_63d_chg"].fillna(0.0)).fillna(0.0)
        df["macro_tailwind_sp500"] = (b_sp * df["sp500_63d_chg"].fillna(0.0)).fillna(0.0)

        # Multi-horizon forward returns
        for h in (20, 40, 60):
            df[f"fwd_{h}"] = close.shift(-h) / close - 1.0

        # Payoff Labels for Head G comparison:
        c_arr = close.to_numpy(dtype=float)
        h_arr = df["high"].to_numpy(dtype=float)
        l_arr = df["low"].to_numpy(dtype=float)
        a_arr = df["atr14"].to_numpy(dtype=float)
        # 1. Old Asymmetric TB (+2.5 / -1.5 ATR, 30D)
        df["tb_old"] = compute_triple_barrier_labels(c_arr, h_arr, l_arr, a_arr, horizon=30, up_mult=2.5, down_mult=1.5)
        # 2. Symmetric 40D TB (+2.0 / -2.0 ATR, 40D)
        df["tb_sym40"] = compute_triple_barrier_labels(c_arr, h_arr, l_arr, a_arr, horizon=40, up_mult=2.0, down_mult=2.0)
        # 3. Clean Swing Win (40D fwd_ret > +2% AND max adverse excursion over next 40D > -10%)
        fut_min_low = pd.Series(l_arr).iloc[::-1].rolling(40, min_periods=40).min().iloc[::-1].shift(-1).to_numpy()
        mae_40 = (fut_min_low / np.maximum(1e-9, c_arr)) - 1.0
        df["clean_swing_win"] = np.where(
            np.isfinite(df["fwd_40"]) & np.isfinite(mae_40),
            ((df["fwd_40"] > 0.0) & (mae_40 > -0.10)).astype(float),
            np.nan,
        )
        # 4. Outperform NIFTY 50 over 40D
        df["beat_nifty_40"] = np.where(
            np.isfinite(df["fwd_40"]) & np.isfinite(df["bench_fwd_40"]),
            (df["fwd_40"] > df["bench_fwd_40"]).astype(float),
            np.nan,
        )

        atr_map[sym] = pd.Series(df["atr14"].to_numpy(dtype=float), index=df["date"])
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    daily_br = full.groupby("date").agg(
        nse_breadth_50d=("above_sma50", "mean"),
        pct_near_52w_hi=("prox_52w_high", lambda s: float(np.mean(s >= 0.90))),
        pct_near_52w_lo=("prox_52w_low", lambda s: float(np.mean(s <= 1.12))),
    ).sort_index()
    daily_br["nse_hilo_sentiment"] = daily_br["pct_near_52w_hi"] - daily_br["pct_near_52w_lo"]
    full = full.merge(daily_br[["nse_breadth_50d", "nse_hilo_sentiment"]].reset_index(), on="date", how="left")

    full["sec_rel_63d"] = (full["ret_63d"] - full.groupby(["date", "sector"])["ret_63d"].transform("median")).fillna(0.0)
    full["sec_rel_126d"] = (full["ret_126d"] - full.groupby(["date", "sector"])["ret_126d"].transform("median")).fillna(0.0)

    for col in [
        "mom_6m1m_voladj", "mom_12m1m_voladj", "fip_smooth_6m", "prox_52w_high",
        "vol_252d", "down_vol_126d", "sec_rel_63d", "sec_rel_126d",
        "ret_21d", "rsi14", "vol_ratio_5_20",
        "macro_tailwind_brent", "macro_tailwind_fx", "macro_tailwind_sp500",
        "max_ret_21d", "overnight_sent_63d",
    ]:
        full[f"z_{col}"] = full.groupby("date")[col].transform(_cs_zscore).fillna(0.0)

    # Static V5 Prior
    nse_mom = 0.5 * full["z_mom_6m1m_voladj"] + 0.5 * full["z_mom_12m1m_voladj"]
    nse_lowvol = -0.6 * full["z_vol_252d"] - 0.4 * full["z_down_vol_126d"]
    full["prior_static"] = (
        0.42 * nse_mom + 0.25 * nse_lowvol + 0.18 * full["z_fip_smooth_6m"] + 0.15 * full["z_prox_52w_high"]
    )
    # Regime-Adaptive Prior: smoothly shifts up to 15% from Mom -> LowVol when India VIX rises 15 -> 25
    def_tilt = ((full["vix_level"] - 15.0) / 10.0).clip(0.0, 1.0)
    w_mom = 0.45 - 0.18 * def_tilt
    w_lv = 0.22 + 0.18 * def_tilt
    full["prior_adaptive"] = (
        w_mom * nse_mom + w_lv * nse_lowvol + 0.18 * full["z_fip_smooth_6m"] + 0.15 * full["z_prox_52w_high"]
    )

    for h in (20, 40, 60):
        full[f"alpha_{h}"] = (full[f"fwd_{h}"] - full[f"bench_fwd_{h}"]).clip(-0.45, 0.45)

    valid = full.dropna(
        subset=["alpha_40", "fwd_40", "bench_fwd_40", "tb_old", "sma200", "vol_252d", "fip_smooth_6m"]
    ).copy()

    for h in (20, 40, 60):
        valid[f"rank_{h}"] = valid.groupby("date")[f"alpha_{h}"].rank(pct=True)
    # Multi-horizon blended target (falls back cleanly to rank_40 on the very last 20 rows if 60D is NaN)
    valid["rank_mh_blend"] = (
        0.25 * valid["rank_20"].fillna(valid["rank_40"])
        + 0.50 * valid["rank_40"]
        + 0.25 * valid["rank_60"].fillna(valid["rank_40"])
    )

    for c in FEATURE_COLS:
        valid[c] = valid[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    prices = pd.DataFrame({s: pd.Series({b["date"]: float(b["close"]) for b in bars}) for s, bars in ohlcv_map.items()}).sort_index().ffill()
    atr_df = pd.DataFrame(atr_map).reindex(prices.index).ffill()
    maps = {
        "vix": macro.get(VOL_INDEX_SYMBOL, {}),
        "breadth": valid.groupby("date")["nse_breadth_50d"].first().to_dict(),
        "hilo": valid.groupby("date")["nse_hilo_sentiment"].first().to_dict(),
        "gold_nifty": valid.groupby("date")["gold_nifty_63d_chg"].first().to_dict(),
        "sector": tickers_meta,
    }
    return valid, prices, atr_df, maps


def fit_oos(
    valid: pd.DataFrame,
    target_mode: str = "40d",           # "40d", "mh_blend", "mh_ensemble"
    half_life_days: float | None = None,
    prior_col: str = "prior_static",
) -> pd.DataFrame:
    dates = sorted(valid["date"].unique())
    first_test = int(len(dates) * 0.42)
    fold_len = max(20, (len(dates) - first_test) // 5)
    embargo = 60 if target_mode in ("mh_blend", "mh_ensemble") else 40

    date_to_idx = {d: i for i, d in enumerate(dates)}
    valid_didx = valid["date"].map(date_to_idx).to_numpy(dtype=float)

    chunks = []
    for f in range(5):
        t0 = first_test + f * fold_len
        t1 = len(dates) if f == 4 else min(len(dates), t0 + fold_len)
        cut_idx = t0 - embargo - 1
        tr_mask = valid["date"] <= dates[cut_idx]
        tr = valid[tr_mask]
        te = valid[(valid["date"] >= dates[t0]) & (valid["date"] <= dates[t1 - 1])].copy()

        # Exponential time-decay sample weights
        sw = None
        if half_life_days is not None:
            age = np.maximum(0.0, cut_idx - valid_didx[tr_mask.to_numpy()])
            sw = np.maximum(0.15, np.power(0.5, age / half_life_days))

        X_tr = tr[FEATURE_COLS].to_numpy(dtype=float)
        X_te = te[FEATURE_COLS].to_numpy(dtype=float)

        if target_mode == "mh_ensemble":
            preds = []
            for h, wt in [(20, 0.20), (40, 0.50), (60, 0.30)]:
                sub_mask = tr[f"rank_{h}"].notna().to_numpy()
                X_s = X_tr[sub_mask]
                y_s = tr[f"rank_{h}"].to_numpy(dtype=float)[sub_mask]
                sw_s = sw[sub_mask] if sw is not None else None
                r = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=500.0))])
                g = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=65, min_samples_leaf=100, l2_regularization=10.0, random_state=42)
                if sw_s is not None:
                    r.fit(X_s, y_s, m__sample_weight=sw_s)
                    g.fit(X_s, y_s, sample_weight=sw_s)
                else:
                    r.fit(X_s, y_s)
                    g.fit(X_s, y_s)
                p = 0.60 * r.predict(X_te) + 0.40 * g.predict(X_te)
                z_p = pd.Series(p, index=te.index).groupby(te["date"]).transform(_cs_zscore).fillna(0.0)
                preds.append(wt * z_p)
            te["z_ml"] = sum(preds)
        else:
            tcol = "rank_mh_blend" if target_mode == "mh_blend" else "rank_40"
            sub_mask = tr[tcol].notna().to_numpy()
            X_s = X_tr[sub_mask]
            y_s = tr[tcol].to_numpy(dtype=float)[sub_mask]
            sw_s = sw[sub_mask] if sw is not None else None
            r = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=500.0))])
            g = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=65, min_samples_leaf=100, l2_regularization=10.0, random_state=42)
            if sw_s is not None:
                r.fit(X_s, y_s, m__sample_weight=sw_s)
                g.fit(X_s, y_s, sample_weight=sw_s)
            else:
                r.fit(X_s, y_s)
                g.fit(X_s, y_s)
            p = 0.60 * r.predict(X_te) + 0.40 * g.predict(X_te)
            te["z_ml"] = pd.Series(p, index=te.index).groupby(te["date"]).transform(_cs_zscore).fillna(0.0)

        te["score"] = 0.50 * te[prior_col] + 0.50 * te["z_ml"]
        chunks.append(te[["date", "symbol", "score", "alpha_40", "fwd_40", "bench_fwd_40"]])
    return pd.concat(chunks, ignore_index=True)


def sim_portfolio_v6(
    oos: pd.DataFrame,
    prices_df: pd.DataFrame,
    atr_df: pd.DataFrame,
    maps: dict[str, Any],
    weight_mode: str = "inv_vol",       # "inv_vol", "corr_pen", "min_var"
    max_per_sector: int = 3,
    vol_target: float | None = None,    # e.g. 0.15
    stop_mode: str | None = None,       # None, "pct_12", "pct_15", "atr_3.5"
    top_k: int = 8,
    cost_bps: float = 15.0,
    rebal_offset: int = 0,
) -> dict[str, float]:
    bench = prices_df[BENCHMARK_SYMBOL]
    bench_sma200 = bench.rolling(200, min_periods=50).mean()
    log_rets = np.log(prices_df / prices_df.shift(1))
    vol_df = log_rets.rolling(252, min_periods=63).std() * np.sqrt(252)

    vix_map, br_map, hilo_map, gn_map, sec_map = maps["vix"], maps["breadth"], maps["hilo"], maps["gold_nifty"], maps["sector"]

    oos_dates = sorted(set(oos["date"].unique()).intersection(set(prices_df.index)))
    sim_dates = [d for d in prices_df.index if oos_dates[rebal_offset] <= d <= oos_dates[-1]]
    rebal_set = set(oos_dates[rebal_offset::40])
    oos_by_date = {dt: grp for dt, grp in oos.groupby("date")}
    idx_loc = {d: i for i, d in enumerate(prices_df.index)}

    fee = cost_bps / 10000.0
    eq, b_eq = 1_000_000.0, 1_000_000.0
    weights: dict[str, float] = {}
    peak_px: dict[str, float] = {}
    entry_atr: dict[str, float] = {}
    p_rets, d_rec = [], []
    stops_triggered = 0

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

        # Check Stock-Level Trailing Crash Stops (strictly causal: exit at today's close into cash)
        if stop_mode and weights and dt not in rebal_set:
            to_drop = []
            for sym, w in list(weights.items()):
                px = float(prices_df.loc[dt, sym])
                if not np.isfinite(px) or px <= 0:
                    continue
                pk = max(peak_px.get(sym, px), px)
                peak_px[sym] = pk
                if stop_mode == "pct_12" and px <= pk * 0.88:
                    to_drop.append(sym)
                elif stop_mode == "pct_15" and px <= pk * 0.85:
                    to_drop.append(sym)
                elif stop_mode == "atr_3.5":
                    a = entry_atr.get(sym, px * 0.025)
                    if px <= pk - 3.5 * a:
                        to_drop.append(sym)
            for sym in to_drop:
                eq *= max(0.0, 1.0 - weights[sym] * fee)
                del weights[sym]
                stops_triggered += 1

        # V5 6-Gauge Sentiment & Macro Risk Gate
        vix = vix_map.get(dt, 15.0)
        br = br_map.get(dt, 0.6)
        hilo = hilo_map.get(dt, 0.1)
        gn = gn_map.get(dt, 0.0)
        bpx = float(bench.loc[dt])
        s200 = float(bench_sma200.loc[dt]) if pd.notna(bench_sma200.loc[dt]) else bpx
        stress = (
            int(vix >= 19.0) + int(vix >= 23.0) + int(bpx < s200)
            + int(br < 0.42) + int(hilo < -0.10) + int(gn > 0.14)
        )
        target_exp = 0.25 if stress >= 3 else (0.55 if stress == 2 else (0.85 if stress == 1 else 1.0))

        # Ex-Ante Portfolio Volatility Targeting (strictly causal trailing 60D cov)
        if vol_target is not None and weights:
            pos = idx_loc[dt]
            if pos >= 65:
                syms = list(weights.keys())
                w_vec = np.array([weights[s] for s in syms], dtype=float)
                w_sum_cur = float(np.sum(w_vec))
                if w_sum_cur > 0.1:
                    w_unit = w_vec / w_sum_cur
                    sub_r = log_rets[syms].iloc[pos - 60 : pos].fillna(0.0).to_numpy(dtype=float)
                    cov_60 = np.cov(sub_r, rowvar=False) * 252.0
                    if cov_60.ndim == 0:
                        est_vol = float(np.sqrt(max(1e-6, cov_60)))
                    else:
                        est_vol = float(np.sqrt(max(1e-6, w_unit @ cov_60 @ w_unit)))
                    vol_cap = float(np.clip(vol_target / max(0.08, est_vol), 0.25, 1.0))
                    target_exp = min(target_exp, vol_cap)

        if dt in rebal_set and dt in oos_by_date:
            day = oos_by_date[dt].sort_values("score", ascending=False).reset_index(drop=True)
            day["rank"] = np.arange(1, len(day) + 1)
            rmap = dict(zip(day["symbol"], day["rank"]))
            chosen, scnt = [], {}
            buf = top_k * 2
            for sym in list(weights.keys()):
                if rmap.get(sym, 999) <= buf and sym in prices_df.columns and scnt.get(sec_map.get(sym, "O"), 0) < max_per_sector:
                    chosen.append(sym)
                    scnt[sec_map.get(sym, "O")] = scnt.get(sec_map.get(sym, "O"), 0) + 1
            for sym in day["symbol"].tolist():
                if len(chosen) >= top_k:
                    break
                if sym in chosen or sym not in prices_df.columns or scnt.get(sec_map.get(sym, "O"), 0) >= max_per_sector:
                    continue
                chosen.append(sym)
                scnt[sec_map.get(sym, "O")] = scnt.get(sec_map.get(sym, "O"), 0) + 1

            # Weighting Scheme
            pos = idx_loc[dt]
            vols = np.array([max(0.12, float(vol_df.loc[dt, s]) if pd.notna(vol_df.loc[dt, s]) else 0.25) for s in chosen])
            if weight_mode == "inv_vol" or pos < 130 or len(chosen) < 3:
                raw_vec = 1.0 / vols
            else:
                sub_r = log_rets[chosen].iloc[pos - 126 : pos].fillna(0.0)
                if weight_mode == "corr_pen":
                    cmat = sub_r.corr().fillna(0.0).to_numpy(dtype=float, copy=True)
                    np.fill_diagonal(cmat, np.nan)
                    avg_c = np.nanmean(np.clip(cmat, 0.0, 1.0), axis=1)
                    avg_c = np.where(np.isfinite(avg_c), avg_c, 0.2)
                    raw_vec = 1.0 / (vols * (1.0 + avg_c))
                elif weight_mode == "min_var":
                    cov = sub_r.cov().fillna(0.0).to_numpy(dtype=float, copy=True) * 252.0
                    # Ledoit-Wolf style 35% shrinkage toward diagonal variance matrix
                    diag = np.diag(np.diag(cov))
                    shrunk = 0.65 * cov + 0.35 * diag + np.eye(len(chosen)) * 1e-4
                    try:
                        inv_s = np.linalg.pinv(shrunk) @ np.ones(len(chosen))
                        inv_s = np.clip(inv_s, 0.0, None)
                        if inv_s.sum() <= 1e-9:
                            raw_vec = 1.0 / vols
                        else:
                            eq_w = 1.0 / len(chosen)
                            norm_w = inv_s / inv_s.sum()
                            raw_vec = np.clip(norm_w, 0.45 * eq_w, 1.75 * eq_w)
                    except Exception:
                        raw_vec = 1.0 / vols

            raw_vec = raw_vec / max(1e-9, float(np.sum(raw_vec)))

            # If vol_target active, also cap new rebalance exposure using chosen portfolio's 60D ex-ante vol
            rebal_exp = target_exp
            if vol_target is not None and pos >= 65 and len(chosen) >= 2:
                sub60 = log_rets[chosen].iloc[pos - 60 : pos].fillna(0.0).to_numpy(dtype=float)
                cov60 = np.cov(sub60, rowvar=False) * 252.0
                est_v = float(np.sqrt(max(1e-6, raw_vec @ cov60 @ raw_vec)))
                rebal_exp = min(target_exp, float(np.clip(vol_target / max(0.08, est_v), 0.25, 1.0)))

            new_w = {s: float(w * rebal_exp) for s, w in zip(chosen, raw_vec)}
            turn = sum(abs(new_w.get(k, 0.0) - weights.get(k, 0.0)) for k in set(weights).union(new_w)) * 0.5
            eq *= max(0.0, 1.0 - turn * fee * 2.0)
            weights = new_w
            peak_px = {s: float(prices_df.loc[dt, s]) for s in chosen}
            entry_atr = {s: float(atr_df.loc[dt, s]) if (s in atr_df.columns and pd.notna(atr_df.loc[dt, s])) else float(prices_df.loc[dt, s]) * 0.025 for s in chosen}
        elif weights:
            cur_exp = sum(weights.values())
            if cur_exp > target_exp + 0.15:
                scale = target_exp / max(1e-9, cur_exp)
                eq *= max(0.0, 1.0 - abs(cur_exp - target_exp) * fee)
                weights = {s: w * scale for s, w in weights.items()}

        if i > 0:
            p_rets.append(day_r); d_rec.append(dt)

    pr = np.array(p_rets)
    yrs = max(0.25, len(pr) / 252.0)
    cagr = ((eq / 1_000_000.0) ** (1.0 / yrs) - 1.0) * 100.0
    b_cagr = ((b_eq / 1_000_000.0) ** (1.0 / yrs) - 1.0) * 100.0
    vol = float(np.std(pr, ddof=1) * np.sqrt(252))
    sharpe = float((np.mean(pr) * 252.0) / max(1e-9, vol))
    down = pr[pr < 0]
    sortino = float((np.mean(pr) * 252.0) / max(1e-9, float(np.std(down, ddof=1) * np.sqrt(252))))
    cum = np.cumprod(1.0 + pr)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum(1e-9, np.maximum.accumulate(cum))) * 100.0
    df_r = pd.DataFrame({"date": d_rec, "p": pr})
    sub_cr = df_r[(df_r["date"] >= "2024-09-25") & (df_r["date"] <= "2025-04-07")]
    ret_cr = (float(np.prod(1.0 + sub_cr["p"])) - 1.0) * 100.0
    return {
        "cagr": cagr, "alpha": cagr - b_cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": float(np.min(dd)), "calmar": cagr / max(1.0, abs(float(np.min(dd)))),
        "crash": ret_cr, "stops": stops_triggered, "_daily": pd.Series(pr, index=d_rec),
    }


def benchmark_payoff_classifiers(valid: pd.DataFrame) -> None:
    """Compare OOS ROC-AUC and Top-8 Win Precision across 4 candidate payoff probability classifiers."""
    dates = sorted(valid["date"].unique())
    first_test = int(len(dates) * 0.42)
    fold_len = max(20, (len(dates) - first_test) // 5)

    print("\n=== G. PAYOFF CLASSIFIER REDESIGN (Fixing 0.473 TB AUC | 5-Fold Strictly OOS) ===\n")
    targets = [
        ("Old TB (+2.5/-1.5 ATR, 30D)", "tb_old", 30),
        ("Symmetric TB (+2.0/-2.0 ATR, 40D)", "tb_sym40", 40),
        ("Clean Swing Win (40D > 0% & MAE > -10%)", "clean_swing_win", 40),
        ("Beat NIFTY 50 (40D Alpha > 0%)", "beat_nifty_40", 40),
    ]
    for label, col, emb in targets:
        y_true_all, y_prob_all, y_prob_meta = [], [], []
        top8_prec_raw, top8_prec_meta = [], []
        for f in range(5):
            t0 = first_test + f * fold_len
            t1 = len(dates) if f == 4 else min(len(dates), t0 + fold_len)
            tr = valid[(valid["date"] <= dates[t0 - emb - 1]) & valid[col].notna()]
            te = valid[(valid["date"] >= dates[t0]) & (valid["date"] <= dates[t1 - 1]) & valid[col].notna()].copy()
            X_tr, y_tr = tr[FEATURE_COLS].to_numpy(dtype=float), tr[col].to_numpy(dtype=int)
            X_te, y_te = te[FEATURE_COLS].to_numpy(dtype=float), te[col].to_numpy(dtype=int)

            lg = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=0.10, max_iter=300))]).fit(X_tr, y_tr)
            gb = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=60, min_samples_leaf=120, l2_regularization=15.0, random_state=42).fit(X_tr, y_tr)
            p_raw = 0.60 * lg.predict_proba(X_te)[:, 1] + 0.40 * gb.predict_proba(X_te)[:, 1]

            # Meta-Calibrated Classifier (uses V5 NSE Institutional Prior + LowVol + Momentum)
            X_tr_m = np.column_stack([X_tr, tr["prior_static"].to_numpy(dtype=float)])
            X_te_m = np.column_stack([X_te, te["prior_static"].to_numpy(dtype=float)])
            lg_m = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=0.05, max_iter=300))]).fit(X_tr_m, y_tr)
            p_meta = lg_m.predict_proba(X_te_m)[:, 1]

            y_true_all.extend(y_te.tolist())
            y_prob_all.extend(p_raw.tolist())
            y_prob_meta.extend(p_meta.tolist())

            te["p_raw"] = p_raw
            te["p_meta"] = p_meta
            for dt in sorted(te["date"].unique())[::20]:
                day = te[te["date"] == dt]
                top8_prec_raw.append(float(day.sort_values("p_raw", ascending=False).head(8)[col].mean()))
                top8_prec_meta.append(float(day.sort_values("p_meta", ascending=False).head(8)[col].mean()))

        auc_raw = roc_auc_score(y_true_all, y_prob_all)
        auc_meta = roc_auc_score(y_true_all, y_prob_meta)
        base_rate = float(np.mean(y_true_all)) * 100.0
        print(
            f"  {label:40s} | BaseRate: {base_rate:4.1f}% | "
            f"Raw AUC: {auc_raw:.3f} (Top8: {np.mean(top8_prec_raw)*100:4.1f}%) | "
            f"Meta-Logit AUC: {auc_meta:.3f} (Top8: {np.mean(top8_prec_meta)*100:4.1f}%)"
        )


def _summarize_tranched(res_list: list[dict[str, Any]]) -> dict[str, float]:
    """Blend 4 staggered overlapping 40D sleeves (25% each) — zero extra turnover, zero date luck."""
    df = pd.concat([r["_daily"] for r in res_list], axis=1).dropna()
    pr = df.mean(axis=1).to_numpy(dtype=float)
    dates = df.index.tolist()
    yrs = max(0.25, len(pr) / 252.0)
    eq = float(np.prod(1.0 + pr))
    cagr = (eq ** (1.0 / yrs) - 1.0) * 100.0
    vol = float(np.std(pr, ddof=1) * np.sqrt(252))
    sharpe = float((np.mean(pr) * 252.0) / max(1e-9, vol))
    down = pr[pr < 0]
    sortino = float((np.mean(pr) * 252.0) / max(1e-9, float(np.std(down, ddof=1) * np.sqrt(252))))
    cum = np.cumprod(1.0 + pr)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum(1e-9, np.maximum.accumulate(cum))) * 100.0
    s = pd.Series(pr, index=dates)
    cr = (float(np.prod(1.0 + s.loc[(s.index >= "2024-09-25") & (s.index <= "2025-04-07")])) - 1.0) * 100.0
    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": float(np.min(dd)), "calmar": cagr / max(1.0, abs(float(np.min(dd)))), "crash": cr,
    }


def main() -> None:
    db = Database()
    valid, prices, atr_df, maps = build_v5_panel_multi_horizon(db)
    oos_v5 = fit_oos(valid, target_mode="40d", half_life_days=None, prior_col="prior_static")

    exps = [
        ("V5 Baseline (InvVol, Sec3, NoVolTgt, NoStop)", "inv_vol", 3, None, None),
        ("D1: Correlation-Penalized Weighting (1/Vol*(1+Corr))", "corr_pen", 3, None, None),
        ("D2: Ledoit-Wolf Shrinkage Min-Variance Weighting", "min_var", 3, None, None),
        ("D3: Tighter Sector Cap = 2 (Max 2 per NSE Sector)", "inv_vol", 2, None, None),
        ("D4: Corr-Penalized Weighting + Sector Cap = 2", "corr_pen", 2, None, None),
        ("D5: Ledoit-Wolf Min-Variance + Sector Cap = 2", "min_var", 2, None, None),
        ("E1: Ex-Ante Portfolio Vol Target = 16% p.a.", "inv_vol", 3, 0.16, None),
        ("E2: Ex-Ante Portfolio Vol Target = 14% p.a.", "inv_vol", 3, 0.14, None),
        ("F1: Stock Trailing Crash Stop -12% (-> LiquidBees)", "inv_vol", 3, None, "pct_12"),
        ("F2: Stock Trailing Crash Stop -15% (-> LiquidBees)", "inv_vol", 3, None, "pct_15"),
        ("F3: Stock Chandelier ATR Stop -3.5xATR (-> LiquidBees)", "inv_vol", 3, None, "atr_3.5"),
        ("COMBO 1: CorrPen + VolTgt 16% + Stop -15%", "corr_pen", 3, 0.16, "pct_15"),
        ("COMBO 2: MinVar + VolTgt 16% + Stop -15%", "min_var", 3, 0.16, "pct_15"),
        ("COMBO 3: CorrPen + Sec2 + VolTgt 16%", "corr_pen", 2, 0.16, None),
    ]

    print(f"=== D/E/F/H PORTFOLIO CONSTRUCTION & RISK UPGRADES (4-Offset Avg & 4-Tranche Blend | 15 bps STT) ===\n")
    print(f"{'Experiment':55s} | {'AvgCAGR':>7s} | {'AvgShrp':>7s} | {'AvgDD':>7s} | {'Calm':>4s} | {'Crash':>6s} || {'TrnCAGR':>7s} | {'TrnShrp':>7s} | {'TrnDD':>7s} | {'TrnClm':>6s}")
    print("-" * 144)
    for name, wmode, scap, vtgt, smode in exps:
        res = [
            sim_portfolio_v6(oos_v5, prices, atr_df, maps, weight_mode=wmode, max_per_sector=scap, vol_target=vtgt, stop_mode=smode, rebal_offset=off)
            for off in (0, 10, 20, 30)
        ]
        tr = _summarize_tranched(res)
        print(
            f"{name:55s} | {np.mean([x['cagr'] for x in res]):+6.2f}% | {np.mean([x['sharpe'] for x in res]):7.2f} | "
            f"{np.mean([x['max_dd'] for x in res]):6.2f}% | {np.mean([x['calmar'] for x in res]):4.2f} | "
            f"{np.mean([x['crash'] for x in res]):+5.1f}% || "
            f"{tr['cagr']:+6.2f}% | {tr['sharpe']:7.2f} | {tr['max_dd']:6.2f}% | {tr['calmar']:6.2f}"
        )

    benchmark_payoff_classifiers(valid)


if __name__ == "__main__":
    main()

