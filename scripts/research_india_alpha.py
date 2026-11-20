"""
EquiLens India Quant Research Lab — Disciplined Out-of-Sample Model Iteration Harness.

Compares 6 progressively refined Indian equity ranking & portfolio architectures strictly
out-of-sample over the 120-stock NSE NIFTY 200 universe (5Y history) with realistic
15 bps Indian STT + brokerage + slippage friction.
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

INDIA_CASH_DAILY_YIELD = (1.062 ** (1.0 / 252.0)) - 1.0  # ~6.2% India T-Bill / LiquidBees yield


def build_enriched_nse_panel(db: Database, horizon: int = 40) -> pd.DataFrame:
    """Build strictly causal NSE factor + technical + macro panel across all stored stocks."""
    ohlcv_map = db.get_all_ohlcv()
    tickers_meta = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    macro = db.get_macro_series()

    bench_bars = ohlcv_map[BENCHMARK_SYMBOL]
    bdf = pd.DataFrame(bench_bars)[["date", "close"]].sort_values("date").reset_index(drop=True)
    bdf["bench_fwd_20d"] = bdf["close"].shift(-20) / bdf["close"] - 1.0
    bdf["bench_fwd_h"] = bdf["close"].shift(-horizon) / bdf["close"] - 1.0
    bdf["bench_sma200"] = bdf["close"].rolling(200, min_periods=50).mean()
    bdf["bench_dist_200"] = (bdf["close"] / bdf["bench_sma200"]) - 1.0

    vix_s = pd.Series(macro.get(VOL_INDEX_SYMBOL, {}), name="vix_level")
    fx_s = pd.Series(macro.get(FX_MACRO_SYMBOL, {}), name="usdinr")
    bdf = bdf.merge(vix_s.rename_axis("date").reset_index(), on="date", how="left")
    bdf = bdf.merge(fx_s.rename_axis("date").reset_index(), on="date", how="left")
    bdf["vix_level"] = bdf["vix_level"].ffill().bfill().fillna(15.0)
    bdf["vix_20d_chg"] = bdf["vix_level"].pct_change(20).fillna(0.0)
    bdf["usdinr_20d_chg"] = bdf["usdinr"].ffill().bfill().pct_change(20).fillna(0.0)
    macro_lookup = bdf[["date", "bench_fwd_20d", "bench_fwd_h", "bench_dist_200", "vix_level", "vix_20d_chg", "usdinr_20d_chg"]]

    frames = []
    for sym, bars in ohlcv_map.items():
        if sym == BENCHMARK_SYMBOL or len(bars) < 260:
            continue
        df = compute_indicator_frame(bars, bench_bars)
        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))

        # Proven Indian institutional factors (strictly causal)
        df["symbol"] = sym
        df["sector"] = tickers_meta.get(sym, "Unknown")
        df["vol_252d"] = log_ret.rolling(252, min_periods=126).std() * np.sqrt(252)
        df["vol_126d"] = log_ret.rolling(126, min_periods=63).std() * np.sqrt(252)
        df["ret_252d"] = close.pct_change(252)
        # Skip-1M returns (t-12M to t-1M and t-6M to t-1M) to remove Indian short-term reversal
        df["ret_12m_1m"] = close.shift(21) / close.shift(252) - 1.0
        df["ret_6m_1m"] = close.shift(21) / close.shift(126) - 1.0
        # Official NSE NIFTY200 Momentum 30 raw ratios: return / 1Y volatility
        vol_safe = df["vol_252d"].clip(lower=0.12)
        df["mom_6m_voladj"] = df["ret_126d"] / vol_safe
        df["mom_12m_voladj"] = df["ret_252d"] / vol_safe
        df["mom_12m1m_voladj"] = df["ret_12m_1m"] / vol_safe
        df["mom_6m1m_voladj"] = df["ret_6m_1m"] / vol_safe

        # Frog-in-the-Pan (FIP) smoothness: fraction of positive days - fraction of negative days over 126D
        pos_days = (log_ret > 0).astype(float).rolling(126, min_periods=63).mean()
        neg_days = (log_ret < 0).astype(float).rolling(126, min_periods=63).mean()
        df["fip_smooth_6m"] = pos_days - neg_days  # Higher = steadier gradual accumulation

        # 52W High Proximity (George & Hwang 2004 — strong in emerging markets / India)
        df["prox_52w_high"] = close / df["high_52w"].replace(0, np.nan)
        df["sma150"] = close.rolling(150, min_periods=80).mean()
        df["above_sma150"] = (close > df["sma150"]).astype(float)
        df["above_sma50"] = (close > df["sma50"]).astype(float)
        df["above_sma200"] = (close > df["sma200"]).astype(float)

        # Downside volatility & upside capture (Sortino-style quality of trend)
        neg_lr = log_ret.clip(upper=0.0)
        df["down_vol_126d"] = neg_lr.rolling(126, min_periods=63).std() * np.sqrt(252)

        # Forward targets
        df["fwd_20d"] = close.shift(-20) / close - 1.0
        df["fwd_h"] = close.shift(-horizon) / close - 1.0
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full = full.merge(macro_lookup, on="date", how="left")

    # Daily NSE breadth
    breadth = full.groupby("date")["above_sma50"].mean().rename("nse_breadth_50d")
    breadth200 = full.groupby("date")["above_sma200"].mean().rename("nse_breadth_200d")
    full = full.merge(breadth.reset_index(), on="date", how="left")
    full = full.merge(breadth200.reset_index(), on="date", how="left")

    # Daily cross-sectional z-scores (exact NSE Index methodology, clipped to [-3, 3])
    def _cs_z(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if not np.isfinite(std) or std < 1e-9:
            return pd.Series(0.0, index=series.index)
        return ((series - series.mean()) / std).clip(-3.0, 3.0)

    for col in [
        "mom_6m_voladj", "mom_12m_voladj", "mom_12m1m_voladj", "mom_6m1m_voladj",
        "fip_smooth_6m", "prox_52w_high", "vol_252d", "down_vol_126d",
        "ret_21d", "ret_63d", "rsi14", "vol_ratio_5_20",
    ]:
        full[f"z_{col}"] = full.groupby("date")[col].transform(_cs_z).fillna(0.0)

    # Sector-relative 6M & 3M momentum
    full["sec_rel_63d"] = (full["ret_63d"] - full.groupby(["date", "sector"])["ret_63d"].transform("median")).fillna(0.0)
    full["sec_rel_126d"] = (full["ret_126d"] - full.groupby(["date", "sector"])["ret_126d"].transform("median")).fillna(0.0)
    full["z_sec_rel_63d"] = full.groupby("date")["sec_rel_63d"].transform(_cs_z).fillna(0.0)
    full["z_sec_rel_126d"] = full.groupby("date")["sec_rel_126d"].transform(_cs_z).fillna(0.0)

    # NSE Official-style Multi-Factor Composite Prior (strictly causal, zero ML fitting!)
    # Combines: NSE Normalized Momentum (6M+12M Skip-1M) + FIP Smoothness + Low-Volatility + 52W High Proximity
    full["nse_mom_score"] = 0.5 * full["z_mom_6m1m_voladj"] + 0.5 * full["z_mom_12m1m_voladj"]
    full["nse_lowvol_score"] = -0.6 * full["z_vol_252d"] - 0.4 * full["z_down_vol_126d"]
    full["nse_factor_prior"] = (
        0.42 * full["nse_mom_score"]
        + 0.25 * full["nse_lowvol_score"]
        + 0.18 * full["z_fip_smooth_6m"]
        + 0.15 * full["z_prox_52w_high"]
    )

    # Targets
    full["alpha_20d"] = (full["fwd_20d"] - full["bench_fwd_20d"]).clip(-0.35, 0.35)
    full["alpha_h"] = (full["fwd_h"] - full["bench_fwd_h"]).clip(-0.45, 0.45)
    # Vol-adjusted forward excess return (forward Sharpe alpha!)
    full["voladj_alpha_h"] = (full["alpha_h"] / full["vol_252d"].clip(lower=0.15)).clip(-2.5, 2.5)
    return full


def simulate_portfolio(
    oos_df: pd.DataFrame,
    score_col: str,
    prices_df: pd.DataFrame,
    vix_map: dict[str, float],
    breadth_map: dict[str, float],
    top_k: int = 8,
    rebalance_days: int = 30,
    rank_buffer: int | None = None,
    inv_vol_weight: bool = False,
    abs_trend_filter: bool = False,
    regime_gate: bool = True,
    cost_bps: float = 15.0,
) -> dict[str, Any]:
    """Strictly out-of-sample portfolio simulation with realistic Indian STT/brokerage & LiquidBees cash yield."""
    bench = prices_df[BENCHMARK_SYMBOL]
    bench_sma200 = bench.rolling(200, min_periods=50).mean()

    oos_dates = sorted(set(oos_df["date"].unique()).intersection(set(prices_df.index)))
    start_dt, end_dt = oos_dates[0], oos_dates[-1]
    sim_dates = [d for d in prices_df.index if start_dt <= d <= end_dt]
    rebal_set = set(oos_dates[::rebalance_days])

    fee = cost_bps / 10000.0
    eq = 1_000_000.0
    b_eq = 1_000_000.0
    weights: dict[str, float] = {}
    turnovers = []
    p_rets, b_rets = [], []

    oos_by_date = {dt: grp for dt, grp in oos_df.groupby("date")}

    for i, dt in enumerate(sim_dates):
        if i > 0:
            prev_dt = sim_dates[i - 1]
            inv_exp = sum(weights.values())
            cash_w = max(0.0, 1.0 - inv_exp)
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

        # Daily India Macro Regime Exposure Cap
        target_exp = 1.0
        if regime_gate:
            vix = vix_map.get(dt, 15.0)
            br = breadth_map.get(dt, 0.6)
            bpx = float(bench.loc[dt])
            s200 = float(bench_sma200.loc[dt]) if pd.notna(bench_sma200.loc[dt]) else bpx
            if vix >= 22.0 or (bpx < s200 and br < 0.35):
                target_exp = 0.30
            elif vix >= 18.5 or bpx < s200 or br < 0.45:
                target_exp = 0.65

        if dt in rebal_set and dt in oos_by_date:
            day = oos_by_date[dt].sort_values(score_col, ascending=False).reset_index(drop=True)
            day["rank"] = np.arange(1, len(day) + 1)
            rank_map = dict(zip(day["symbol"], day["rank"]))
            vol_map = dict(zip(day["symbol"], day["vol_252d"].fillna(0.25).clip(lower=0.12)))
            trend_ok = set(day[(day["above_sma150"] > 0.5) & (day["ret_126d"] > 0)]["symbol"])

            # Apply Rank-Buffer hysteresis if enabled (retain existing holdings still within buffer)
            chosen: list[str] = []
            if rank_buffer is not None and weights:
                for sym in list(weights.keys()):
                    r = rank_map.get(sym, 999)
                    if r <= rank_buffer and sym in prices_df.columns:
                        if (not abs_trend_filter) or (sym in trend_ok):
                            chosen.append(sym)
            for sym in day["symbol"].tolist():
                if len(chosen) >= top_k:
                    break
                if sym in chosen or sym not in prices_df.columns:
                    continue
                if abs_trend_filter and sym not in trend_ok:
                    continue
                chosen.append(sym)

            if chosen:
                # If abs_trend_filter yields fewer than top_k qualifying stocks, unallocated slots stay in cash!
                slot_scale = min(1.0, len(chosen) / top_k) * target_exp
                if inv_vol_weight:
                    raw_w = {s: 1.0 / vol_map.get(s, 0.25) for s in chosen}
                    s_sum = sum(raw_w.values()) or 1.0
                    new_w = {s: (v / s_sum) * slot_scale for s, v in raw_w.items()}
                else:
                    new_w = {s: slot_scale / len(chosen) for s in chosen}
            else:
                new_w = {}

            all_k = set(weights.keys()).union(new_w.keys())
            turn = sum(abs(new_w.get(k, 0.0) - weights.get(k, 0.0)) for k in all_k) * 0.5
            eq *= max(0.0, 1.0 - turn * fee * 2.0)
            turnovers.append(turn)
            weights = new_w
        elif regime_gate and weights:
            cur_exp = sum(weights.values())
            if cur_exp > target_exp + 0.20:
                scale = target_exp / max(1e-9, cur_exp)
                turn = abs(cur_exp - target_exp) * 0.5
                eq *= max(0.0, 1.0 - turn * fee * 2.0)
                weights = {s: w * scale for s, w in weights.items()}

        if i > 0:
            p_rets.append(day_r)
            b_rets.append(b_r)

    pr = np.array(p_rets)
    br = np.array(b_rets)
    yrs = max(0.25, len(pr) / 252.0)
    cagr = ((eq / 1_000_000.0) ** (1.0 / yrs) - 1.0) * 100.0
    b_cagr = ((b_eq / 1_000_000.0) ** (1.0 / yrs) - 1.0) * 100.0
    vol = float(np.std(pr, ddof=1) * np.sqrt(252))
    sharpe = float((np.mean(pr) * 252.0) / max(1e-9, vol))
    cum = np.cumprod(1.0 + pr)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum(1e-9, np.maximum.accumulate(cum))) * 100.0
    b_cum = np.cumprod(1.0 + br)
    b_dd = ((b_cum - np.maximum.accumulate(b_cum)) / np.maximum(1e-9, np.maximum.accumulate(b_cum))) * 100.0

    # IC & Hit Rate vs NIFTY
    ic_val = float(spearmanr(oos_df[score_col], oos_df["alpha_h"]).statistic)
    hits = []
    for dt in oos_dates[::20]:
        sl = oos_by_date[dt].sort_values(score_col, ascending=False).head(top_k)
        if not sl.empty:
            hits.append(1 if sl["fwd_h"].mean() > sl["bench_fwd_h"].mean() else 0)

    return {
        "cagr": round(cagr, 2),
        "nifty_cagr": round(b_cagr, 2),
        "excess": round(cagr - b_cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(float(np.min(dd)), 2),
        "nifty_dd": round(float(np.min(b_dd)), 2),
        "ic": round(0.0 if np.isnan(ic_val) else ic_val, 4),
        "hit_pct": round(float(np.mean(hits)) * 100.0, 1) if hits else 50.0,
        "avg_turn_pct": round(float(np.mean(turnovers)) * 100.0, 1) if turnovers else 0.0,
    }


def main() -> None:
    db = Database()
    horizon = 40
    panel = build_enriched_nse_panel(db, horizon=horizon)
    valid = panel.dropna(subset=["alpha_h", "voladj_alpha_h", "vol_252d", "sma200", "fip_smooth_6m"]).copy()

    # Build aligned close price matrix + macro maps
    ohlcv_map = db.get_all_ohlcv()
    prices_df = pd.DataFrame(
        {sym: pd.Series({b["date"]: float(b["close"]) for b in bars}) for sym, bars in ohlcv_map.items()}
    ).sort_index().ffill()
    vix_map = db.get_macro_series().get(VOL_INDEX_SYMBOL, {})
    breadth_map = valid.groupby("date")["nse_breadth_50d"].first().to_dict()

    # Walk-forward 5 expanding folds with strict 40-day embargo gap
    dates = sorted(valid["date"].unique())
    first_test = int(len(dates) * 0.42)
    fold_len = max(20, (len(dates) - first_test) // 5)

    # Compact, regularized institutional feature set for ML v3
    ml_v3_cols = [
        "z_mom_6m1m_voladj", "z_mom_12m1m_voladj", "z_fip_smooth_6m",
        "z_prox_52w_high", "z_vol_252d", "z_down_vol_126d",
        "z_sec_rel_63d", "z_sec_rel_126d", "z_ret_21d", "z_rsi14",
        "z_vol_ratio_5_20", "vix_level", "bench_dist_200", "nse_breadth_50d",
    ]
    # Old v1 noisy short-term feature set
    ml_v1_cols = [
        "ret_5d", "ret_21d", "ret_63d", "ret_126d", "rs_spy_21d", "rs_spy_63d",
        "dist_sma20", "dist_sma50", "dist_sma200", "rsi14", "macd_hist_norm",
        "bb_pct_b", "vol_20d", "vol_ratio_5_20", "pos_52w", "vix_level",
    ]

    for c in set(ml_v3_cols + ml_v1_cols):
        valid[c] = valid[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    oos_chunks = []
    for f in range(5):
        t0 = first_test + f * fold_len
        t1 = len(dates) if f == 4 else min(len(dates), t0 + fold_len)
        tr_end = t0 - horizon
        if tr_end < 60 or t0 >= len(dates):
            continue
        tr = valid[valid["date"] <= dates[tr_end - 1]]
        te = valid[(valid["date"] >= dates[t0]) & (valid["date"] <= dates[t1 - 1])].copy()

        # V1 model (old features, 20D alpha)
        m1_r = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=50.0))]).fit(tr[ml_v1_cols], tr["alpha_20d"])
        m1_g = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=60, random_state=42).fit(tr[ml_v1_cols], tr["alpha_20d"])
        te["score_v1"] = 0.5 * m1_r.predict(te[ml_v1_cols]) + 0.5 * m1_g.predict(te[ml_v1_cols])

        # V3 model (NSE institutional features + strong regularization + 40D vol-adjusted alpha)
        m3_r = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=400.0))]).fit(tr[ml_v3_cols], tr["voladj_alpha_h"])
        m3_g = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=60, min_samples_leaf=80, l2_regularization=10.0, random_state=42).fit(tr[ml_v3_cols], tr["voladj_alpha_h"])
        raw_v3 = 0.65 * m3_r.predict(te[ml_v3_cols]) + 0.35 * m3_g.predict(te[ml_v3_cols])
        te["score_v3_ml"] = raw_v3

        # Standardize V3 ML per day so it blends cleanly with NSE Factor Prior
        te["z_v3_ml"] = te.groupby("date")["score_v3_ml"].transform(
            lambda s: ((s - s.mean()) / (s.std(ddof=0) + 1e-9)).clip(-3, 3)
        )
        te["score_v4_hybrid"] = 0.65 * te["nse_factor_prior"] + 0.35 * te["z_v3_ml"]
        oos_chunks.append(te)

    oos = pd.concat(oos_chunks, ignore_index=True)
    print(f"OOS Evaluation Window: {oos['date'].min()} to {oos['date'].max()} | {oos['symbol'].nunique()} NSE stocks | {len(oos):,} obs | Cost: 15 bps Indian STT+Brokerage\n")

    experiments = [
        ("V1: Old 20D ML (20D Rebal, Equal Wt, No Buffer)", "score_v1", dict(top_k=6, rebalance_days=20, rank_buffer=None, inv_vol_weight=False, abs_trend_filter=False, regime_gate=True)),
        ("V2: NSE Factor Prior Only (Mom30+FIP+LowVol, 20D)", "nse_factor_prior", dict(top_k=8, rebalance_days=20, rank_buffer=None, inv_vol_weight=False, abs_trend_filter=False, regime_gate=True)),
        ("V3: Regularized 40D Vol-Adj ML Only (30D Rebal)", "score_v3_ml", dict(top_k=8, rebalance_days=30, rank_buffer=None, inv_vol_weight=False, abs_trend_filter=False, regime_gate=True)),
        ("V4: Hybrid (65% NSE Factor + 35% 40D ML, 30D)", "score_v4_hybrid", dict(top_k=8, rebalance_days=30, rank_buffer=None, inv_vol_weight=False, abs_trend_filter=False, regime_gate=True)),
        ("V5: Hybrid + Rank Buffer (2xK) + Inv-Vol + 40D", "score_v4_hybrid", dict(top_k=8, rebalance_days=40, rank_buffer=16, inv_vol_weight=True, abs_trend_filter=False, regime_gate=True)),
        ("V6: V5 + Absolute Trend Filter (SMA150 & 6M>0)", "score_v4_hybrid", dict(top_k=8, rebalance_days=40, rank_buffer=16, inv_vol_weight=True, abs_trend_filter=True, regime_gate=True)),
        ("V7: Pure NSE Factor + Buffer + InvVol + AbsTrend", "nse_factor_prior", dict(top_k=10, rebalance_days=40, rank_buffer=20, inv_vol_weight=True, abs_trend_filter=True, regime_gate=True)),
        ("V8: Hybrid Top-10 + Buffer + InvVol + AbsTrend", "score_v4_hybrid", dict(top_k=10, rebalance_days=40, rank_buffer=20, inv_vol_weight=True, abs_trend_filter=True, regime_gate=True)),
    ]

    print(f"{'Experiment':50s} | {'CAGR':>6s} | {'NIFTY':>6s} | {'Alpha':>6s} | {'Sharpe':>6s} | {'MaxDD':>7s} | {'IC':>7s} | {'Hit%':>5s} | {'Turn%':>5s}")
    print("-" * 118)
    for name, col, kw in experiments:
        r = simulate_portfolio(oos, col, prices_df, vix_map, breadth_map, cost_bps=15.0, **kw)
        print(f"{name:50s} | {r['cagr']:+5.2f}% | {r['nifty_cagr']:+5.2f}% | {r['excess']:+5.2f}% | {r['sharpe']:6.2f} | {r['max_dd']:6.2f}% | {r['ic']:+7.4f} | {r['hit_pct']:4.1f}% | {r['avg_turn_pct']:4.1f}%")


if __name__ == "__main__":
    main()
