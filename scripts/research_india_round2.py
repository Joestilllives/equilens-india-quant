"""
EquiLens India Quant Research Lab — Round 2:
  - Top-Quartile Classifier (P(Top 25% NSE stock over 40D))
  - Rank-Target Regression (predicting cross-sectional forward rank in [0, 1])
  - Sector Cap (max 2-3 stocks per sector) + Rank-Buffer + Inverse-Vol Sizing
  - Full Sub-Period & Sep 2024–Apr 2025 Momentum Crash Audit!
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import BENCHMARK_SYMBOL, VOL_INDEX_SYMBOL
from backend.database import Database
from scripts.research_india_alpha import INDIA_CASH_DAILY_YIELD, build_enriched_nse_panel

warnings.filterwarnings("ignore")


def simulate_with_audit(
    oos_df: pd.DataFrame,
    score_col: str,
    prices_df: pd.DataFrame,
    vix_map: dict[str, float],
    breadth_map: dict[str, float],
    sector_map: dict[str, str],
    top_k: int = 8,
    rebalance_days: int = 35,
    rank_buffer: int = 16,
    max_per_sector: int = 3,
    inv_vol_weight: bool = True,
    abs_trend_filter: bool = False,
    regime_gate: bool = True,
    cost_bps: float = 15.0,
) -> dict[str, Any]:
    bench = prices_df[BENCHMARK_SYMBOL]
    bench_sma200 = bench.rolling(200, min_periods=50).mean()

    oos_dates = sorted(set(oos_df["date"].unique()).intersection(set(prices_df.index)))
    start_dt, end_dt = oos_dates[0], oos_dates[-1]
    sim_dates = [d for d in prices_df.index if start_dt <= d <= end_dt]
    rebal_set = set(oos_dates[::rebalance_days])
    oos_by_date = {dt: grp for dt, grp in oos_df.groupby("date")}

    fee = cost_bps / 10000.0
    eq, b_eq = 1_000_000.0, 1_000_000.0
    weights: dict[str, float] = {}
    turnovers, p_rets, b_rets, dates_rec = [], [], [], []

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

            chosen: list[str] = []
            sec_counts: dict[str, int] = {}

            def _can_add(sym: str) -> bool:
                sec = sector_map.get(sym, "Other")
                return sec_counts.get(sec, 0) < max_per_sector

            def _add(sym: str) -> None:
                chosen.append(sym)
                sec = sector_map.get(sym, "Other")
                sec_counts[sec] = sec_counts.get(sec, 0) + 1

            # Retain existing holdings inside rank buffer
            for sym in list(weights.keys()):
                if rank_map.get(sym, 999) <= rank_buffer and sym in prices_df.columns and _can_add(sym):
                    if (not abs_trend_filter) or (sym in trend_ok):
                        _add(sym)

            for sym in day["symbol"].tolist():
                if len(chosen) >= top_k:
                    break
                if sym in chosen or sym not in prices_df.columns or not _can_add(sym):
                    continue
                if abs_trend_filter and sym not in trend_ok:
                    continue
                _add(sym)

            if chosen:
                slot_scale = min(1.0, len(chosen) / top_k) * target_exp
                if inv_vol_weight:
                    raw_w = {s: 1.0 / vol_map.get(s, 0.25) for s in chosen}
                    ss = sum(raw_w.values()) or 1.0
                    new_w = {s: (v / ss) * slot_scale for s, v in raw_w.items()}
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
            dates_rec.append(dt)

    pr, br = np.array(p_rets), np.array(b_rets)
    yrs = max(0.25, len(pr) / 252.0)
    cagr = ((eq / 1_000_000.0) ** (1.0 / yrs) - 1.0) * 100.0
    b_cagr = ((b_eq / 1_000_000.0) ** (1.0 / yrs) - 1.0) * 100.0
    vol = float(np.std(pr, ddof=1) * np.sqrt(252))
    sharpe = float((np.mean(pr) * 252.0) / max(1e-9, vol))
    cum = np.cumprod(1.0 + pr)
    dd = ((cum - np.maximum.accumulate(cum)) / np.maximum(1e-9, np.maximum.accumulate(cum))) * 100.0

    # Sub-period breakdown: 2021-2023, 2024-2026, AND Sep 2024 - Apr 2025 crash
    df_r = pd.DataFrame({"date": dates_rec, "p": pr, "b": br})

    def _sub(d0: str, d1: str) -> tuple[float, float]:
        s = df_r[(df_r["date"] >= d0) & (df_r["date"] <= d1)]
        if s.empty:
            return 0.0, 0.0
        return round((float(np.prod(1.0 + s["p"])) - 1.0) * 100.0, 1), round((float(np.prod(1.0 + s["b"])) - 1.0) * 100.0, 1)

    p_21_23, b_21_23 = _sub("2021-01-01", "2023-12-31")
    p_24_26, b_24_26 = _sub("2024-01-01", "2026-12-31")
    p_crash, b_crash = _sub("2024-09-25", "2025-04-07")

    ic_val = float(spearmanr(oos_df[score_col], oos_df["alpha_h"]).statistic)
    hits = []
    for dt in oos_dates[::20]:
        sl = oos_by_date[dt].sort_values(score_col, ascending=False).head(top_k)
        if not sl.empty:
            hits.append(1 if sl["fwd_h"].mean() > sl["bench_fwd_h"].mean() else 0)

    return {
        "cagr": round(cagr, 2),
        "excess": round(cagr - b_cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(float(np.min(dd)), 2),
        "ic": round(0.0 if np.isnan(ic_val) else ic_val, 4),
        "hit_pct": round(float(np.mean(hits)) * 100.0, 1),
        "turn": round(float(np.mean(turnovers)) * 100.0, 1),
        "p_21_23": p_21_23, "b_21_23": b_21_23,
        "p_24_26": p_24_26, "b_24_26": b_24_26,
        "p_crash": p_crash, "b_crash": b_crash,
    }


def main() -> None:
    db = Database()
    horizon = 40
    panel = build_enriched_nse_panel(db, horizon=horizon)
    valid = panel.dropna(subset=["alpha_h", "voladj_alpha_h", "vol_252d", "sma200", "fip_smooth_6m"]).copy()

    # Cross-sectional forward rank targets (strictly per date!)
    valid["fwd_rank_h"] = valid.groupby("date")["alpha_h"].rank(pct=True)
    valid["is_top_q"] = (valid["fwd_rank_h"] >= 0.75).astype(int)

    ohlcv_map = db.get_all_ohlcv()
    prices_df = pd.DataFrame(
        {sym: pd.Series({b["date"]: float(b["close"]) for b in bars}) for sym, bars in ohlcv_map.items()}
    ).sort_index().ffill()
    vix_map = db.get_macro_series().get(VOL_INDEX_SYMBOL, {})
    breadth_map = valid.groupby("date")["nse_breadth_50d"].first().to_dict()
    sector_map = {t["symbol"]: t["sector"] for t in db.get_tickers()}

    feat_cols = [
        "z_mom_6m1m_voladj", "z_mom_12m1m_voladj", "z_fip_smooth_6m",
        "z_prox_52w_high", "z_vol_252d", "z_down_vol_126d",
        "z_sec_rel_63d", "z_sec_rel_126d", "z_ret_21d", "z_rsi14",
        "z_vol_ratio_5_20", "vix_level", "bench_dist_200", "nse_breadth_50d",
    ]
    for c in feat_cols:
        valid[c] = valid[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    dates = sorted(valid["date"].unique())
    first_test = int(len(dates) * 0.42)
    fold_len = max(20, (len(dates) - first_test) // 5)

    oos_chunks = []
    for f in range(5):
        t0 = first_test + f * fold_len
        t1 = len(dates) if f == 4 else min(len(dates), t0 + fold_len)
        tr_end = t0 - horizon
        if tr_end < 60 or t0 >= len(dates):
            continue
        tr = valid[valid["date"] <= dates[tr_end - 1]]
        te = valid[(valid["date"] >= dates[t0]) & (valid["date"] <= dates[t1 - 1])].copy()

        # 1. Rank-target Ridge + shallow GBR
        r_rank = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=500.0))]).fit(tr[feat_cols], tr["fwd_rank_h"])
        g_rank = HistGradientBoostingRegressor(max_depth=2, learning_rate=0.03, max_iter=60, min_samples_leaf=100, l2_regularization=10.0, random_state=42).fit(tr[feat_cols], tr["fwd_rank_h"])
        te["ml_rank"] = 0.6 * r_rank.predict(te[feat_cols]) + 0.4 * g_rank.predict(te[feat_cols])

        # 2. Top-Quartile Classifier (P(Finish in Top 25% of NSE over 40D))
        c_log = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=0.2, max_iter=300))]).fit(tr[feat_cols], tr["is_top_q"])
        c_gbc = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=60, min_samples_leaf=100, l2_regularization=10.0, random_state=42).fit(tr[feat_cols], tr["is_top_q"])
        te["ml_topq"] = 0.5 * c_log.predict_proba(te[feat_cols])[:, 1] + 0.5 * c_gbc.predict_proba(te[feat_cols])[:, 1]

        # Normalize daily & blend with NSE Factor Prior
        for col in ("ml_rank", "ml_topq"):
            te[f"z_{col}"] = te.groupby("date")[col].transform(lambda s: ((s - s.mean()) / (s.std(ddof=0) + 1e-9)).clip(-3, 3))

        te["blend_topq_factor"] = 0.45 * te["nse_factor_prior"] + 0.55 * te["z_ml_topq"]
        te["blend_rank_factor"] = 0.50 * te["nse_factor_prior"] + 0.50 * te["z_ml_rank"]
        te["blend_all3"] = 0.40 * te["nse_factor_prior"] + 0.30 * te["z_ml_topq"] + 0.30 * te["z_ml_rank"]
        oos_chunks.append(te)

    oos = pd.concat(oos_chunks, ignore_index=True)

    candidates = [
        ("R1: Pure Top-Q Classifier (Top8, 40D, Buf16, Sec3)", "ml_topq", dict(top_k=8, rebalance_days=40, rank_buffer=16, max_per_sector=3, inv_vol_weight=True, abs_trend_filter=False)),
        ("R2: Pure Rank-Target ML (Top8, 40D, Buf16, Sec3)", "ml_rank", dict(top_k=8, rebalance_days=40, rank_buffer=16, max_per_sector=3, inv_vol_weight=True, abs_trend_filter=False)),
        ("R3: Hybrid TopQ+Factor (Top8, 40D, Buf16, Sec3)", "blend_topq_factor", dict(top_k=8, rebalance_days=40, rank_buffer=16, max_per_sector=3, inv_vol_weight=True, abs_trend_filter=False)),
        ("R4: Hybrid Rank+Factor (Top8, 40D, Buf16, Sec3)", "blend_rank_factor", dict(top_k=8, rebalance_days=40, rank_buffer=16, max_per_sector=3, inv_vol_weight=True, abs_trend_filter=False)),
        ("R5: Hybrid All-3 Ensemble (Top8, 40D, Buf16, Sec3)", "blend_all3", dict(top_k=8, rebalance_days=40, rank_buffer=16, max_per_sector=3, inv_vol_weight=True, abs_trend_filter=False)),
        ("R6: Hybrid All-3 + AbsTrend (Top8, 40D, Buf16, Sec3)", "blend_all3", dict(top_k=8, rebalance_days=40, rank_buffer=16, max_per_sector=3, inv_vol_weight=True, abs_trend_filter=True)),
        ("R7: Hybrid All-3 (Top10, 40D, Buf20, Sec3)", "blend_all3", dict(top_k=10, rebalance_days=40, rank_buffer=20, max_per_sector=3, inv_vol_weight=True, abs_trend_filter=False)),
        ("R8: Hybrid All-3 (Top8, 30D, Buf16, Sec2)", "blend_all3", dict(top_k=8, rebalance_days=30, rank_buffer=16, max_per_sector=2, inv_vol_weight=True, abs_trend_filter=False)),
    ]

    print(f"{'Candidate':52s} | {'CAGR':>6s} | {'Alpha':>6s} | {'Shrp':>5s} | {'MaxDD':>7s} | {'IC':>7s} | {'Hit%':>5s} | {'21-23':>7s} | {'24-26':>7s} | {'Crash':>6s}")
    print("-" * 134)
    for name, col, kw in candidates:
        r = simulate_with_audit(oos, col, prices_df, vix_map, breadth_map, sector_map, cost_bps=15.0, **kw)
        print(f"{name:52s} | {r['cagr']:+5.2f}% | {r['excess']:+5.2f}% | {r['sharpe']:5.2f} | {r['max_dd']:6.2f}% | {r['ic']:+7.4f} | {r['hit_pct']:4.1f}% | {r['p_21_23']:+6.1f}% | {r['p_24_26']:+6.1f}% | {r['p_crash']:+5.1f}%")
    print(f"\nNIFTY 50 Benchmark sub-periods: 2021-23 = {r['b_21_23']:+.1f}% | 2024-26 = {r['b_24_26']:+.1f}% | Sep24-Apr25 Crash = {r['b_crash']:+.1f}%")


if __name__ == "__main__":
    main()
