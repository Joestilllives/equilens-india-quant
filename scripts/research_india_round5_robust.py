"""
EquiLens India Quant Research Lab — Round 5 (Dual-Head Sentiment/Macro Ensemble & Anti-Overfit Stress Test):
Tests clean combinations of Round 4's highest-IC / best-risk-adjusted discoveries across:
  - 4 Rebalance Start Offsets (day 0, 10, 20, 30) to eliminate rebalance-date luck
  - Portfolio Sizes (Top-6, Top-8, Top-10, Top-12)
  - Transaction Costs (0, 15, 30, 50 bps)
  - Gate Threshold Perturbation (+/-10% sensitivity check on Sentiment/Macro Gate)
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

from backend.config import BENCHMARK_SYMBOL, VOL_INDEX_SYMBOL
from backend.database import Database
from scripts.research_india_round4 import INDIA_CASH_DAILY_YIELD, _cs_z, build_round4_panel

warnings.filterwarnings("ignore")


def fit_oos_scores(
    valid: pd.DataFrame,
    core_feats: list[str],
    extra_feats: list[str] | None = None,
    dual_head: bool = False,
    prior_weight: float = 0.50,
    horizon: int = 40,
) -> pd.DataFrame:
    dates = sorted(valid["date"].unique())
    first_test = int(len(dates) * 0.42)
    fold_len = max(20, (len(dates) - first_test) // 5)

    all_cols = list(dict.fromkeys(core_feats + (extra_feats or [])))
    for c in all_cols:
        valid[c] = valid[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    chunks = []
    for f in range(5):
        t0 = first_test + f * fold_len
        t1 = len(dates) if f == 4 else min(len(dates), t0 + fold_len)
        tr = valid[valid["date"] <= dates[t0 - horizon - 1]]
        te = valid[(valid["date"] >= dates[t0]) & (valid["date"] <= dates[t1 - 1])].copy()

        if dual_head and extra_feats:
            # Head 1: Clean 14-feature NSE Institutional Core
            r1 = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=500.0))]).fit(tr[core_feats], tr["fwd_rank"])
            g1 = HistGradientBoostingRegressor(
                max_depth=2, learning_rate=0.03, max_iter=65, min_samples_leaf=100, l2_regularization=10.0, random_state=42
            ).fit(tr[core_feats], tr["fwd_rank"])
            ml1 = 0.60 * r1.predict(te[core_feats]) + 0.40 * g1.predict(te[core_feats])
            z_ml1 = pd.Series(ml1, index=te.index).groupby(te["date"]).transform(_cs_z).fillna(0.0)

            # Head 2: Dedicated Sentiment + Macro-Beta + PEAD Alpha Head
            r2 = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=600.0))]).fit(tr[extra_feats], tr["fwd_rank"])
            g2 = HistGradientBoostingRegressor(
                max_depth=2, learning_rate=0.03, max_iter=55, min_samples_leaf=120, l2_regularization=15.0, random_state=43
            ).fit(tr[extra_feats], tr["fwd_rank"])
            ml2 = 0.65 * r2.predict(te[extra_feats]) + 0.35 * g2.predict(te[extra_feats])
            z_ml2 = pd.Series(ml2, index=te.index).groupby(te["date"]).transform(_cs_z).fillna(0.0)

            te["score"] = prior_weight * te["prior_r4"] + 0.32 * z_ml1 + (0.50 - 0.32) * z_ml2
        else:
            fcols = core_feats if not extra_feats else all_cols
            r = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=500.0))]).fit(tr[fcols], tr["fwd_rank"])
            g = HistGradientBoostingRegressor(
                max_depth=2, learning_rate=0.03, max_iter=65, min_samples_leaf=100, l2_regularization=10.0, random_state=42
            ).fit(tr[fcols], tr["fwd_rank"])
            ml = 0.60 * r.predict(te[fcols]) + 0.40 * g.predict(te[fcols])
            z_ml = pd.Series(ml, index=te.index).groupby(te["date"]).transform(_cs_z).fillna(0.0)
            te["score"] = prior_weight * te["prior_r4"] + (1.0 - prior_weight) * z_ml

        chunks.append(te[["date", "symbol", "score", "target_alpha", "fwd_ret", "bench_fwd_ret"]])
    return pd.concat(chunks, ignore_index=True)


def sim_one(
    oos: pd.DataFrame,
    prices_df: pd.DataFrame,
    maps: dict[str, dict[str, Any]],
    gate_mode: str = "base",
    top_k: int = 8,
    cost_bps: float = 15.0,
    rebal_offset: int = 0,
    gate_scale: float = 1.0,
) -> dict[str, float]:
    bench = prices_df[BENCHMARK_SYMBOL]
    bench_sma200 = bench.rolling(200, min_periods=50).mean()
    log_rets = np.log(prices_df / prices_df.shift(1))
    vol_df = log_rets.rolling(252, min_periods=63).std() * np.sqrt(252)

    vix_map = maps["vix"]
    br_map = maps["breadth"]
    hilo_map = maps["hilo"]
    gn_map = maps["gold_nifty"]
    sec_map = maps["sector"]

    oos_dates = sorted(set(oos["date"].unique()).intersection(set(prices_df.index)))
    sim_dates = [d for d in prices_df.index if oos_dates[rebal_offset] <= d <= oos_dates[-1]]
    rebal_set = set(oos_dates[rebal_offset::40])
    oos_by_date = {dt: grp for dt, grp in oos.groupby("date")}

    fee = cost_bps / 10000.0
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

        vix = vix_map.get(dt, 15.0)
        br = br_map.get(dt, 0.6)
        hilo = hilo_map.get(dt, 0.1)
        gn = gn_map.get(dt, 0.0)
        bpx = float(bench.loc[dt])
        s200 = float(bench_sma200.loc[dt]) if pd.notna(bench_sma200.loc[dt]) else bpx

        target_exp = 1.0
        if gate_mode == "base":
            if vix >= 22.0 or (bpx < s200 and br < 0.35):
                target_exp = 0.30
            elif vix >= 18.5 or bpx < s200 or br < 0.45:
                target_exp = 0.65
        elif gate_mode == "sentiment_v1":
            stress = 0
            if vix >= 19.0 * gate_scale: stress += 1
            if vix >= 23.0 * gate_scale: stress += 1
            if bpx < s200: stress += 1
            if br < 0.42 / gate_scale: stress += 1
            if hilo < -0.10 * gate_scale: stress += 1
            if gn > 0.14 * gate_scale: stress += 1
            if stress >= 3:
                target_exp = 0.25
            elif stress == 2:
                target_exp = 0.55
            elif stress == 1:
                target_exp = 0.85

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

        if i > 0:
            p_rets.append(day_r); b_rets.append(b_r); d_rec.append(dt)

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
    sub24 = df_r[df_r["date"] >= "2024-01-01"]
    ret_24 = (float(np.prod(1.0 + sub24["p"])) - 1.0) * 100.0
    sub_cr = df_r[(df_r["date"] >= "2024-09-25") & (df_r["date"] <= "2025-04-07")]
    ret_cr = (float(np.prod(1.0 + sub_cr["p"])) - 1.0) * 100.0
    return {
        "cagr": cagr, "alpha": cagr - b_cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": float(np.min(dd)), "calmar": cagr / max(1.0, abs(float(np.min(dd)))),
        "r24": ret_24, "crash": ret_cr,
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
        "gold_nifty": valid.groupby("date")["gold_nifty_63d_chg"].first().to_dict(),
        "sector": {t["symbol"]: t["sector"] for t in db.get_tickers()},
    }

    core = [
        "z_mom_6m1m_voladj", "z_mom_12m1m_voladj", "z_fip_smooth_6m",
        "z_prox_52w_high", "z_vol_252d", "z_down_vol_126d",
        "z_sec_rel_63d", "z_sec_rel_126d", "z_ret_21d", "z_rsi14",
        "z_vol_ratio_5_20", "vix_level", "bench_dist_200", "nse_breadth_50d",
    ]
    sent_macro_head = [
        "z_macro_tailwind_brent", "z_macro_tailwind_fx", "z_macro_tailwind_sp500",
        "z_max_ret_21d", "z_skew_63d",
        "z_overnight_sent_63d", "z_intraday_sent_63d",
        "z_pead_surprise", "z_pead_streak",
        "z_idio_mom_126d", "z_nifty_beta_126d",
    ]
    macro_max_head = [
        "z_macro_tailwind_brent", "z_macro_tailwind_fx", "z_macro_tailwind_sp500",
        "z_max_ret_21d", "z_overnight_sent_63d", "z_pead_surprise",
    ]

    models = {
        "R4 Baseline (Old Base Gate)": (fit_oos_scores(valid, core), "base"),
        "V5a: R4 + 6-Gauge Sentiment/Macro Gate": (None, "sentiment_v1"),  # reuse R4 oos
        "V5b: Dual-Head (Core + Sent/Macro/PEAD 11f) + Base Gate": (fit_oos_scores(valid, core, sent_macro_head, dual_head=True), "base"),
        "V5c: Dual-Head (Core + Sent/Macro/PEAD 11f) + Sent Gate": (None, "sentiment_v1"),  # reuse V5b
        "V5d: Dual-Head (Core + Focused 6f Macro/MAX/ON/PEAD) + Sent Gate": (fit_oos_scores(valid, core, macro_max_head, dual_head=True), "sentiment_v1"),
        "V5e: Single-Head (Core + MacroBetas + MAX) + Sent Gate": (fit_oos_scores(valid, core, ["z_macro_tailwind_brent", "z_macro_tailwind_fx", "z_macro_tailwind_sp500", "z_max_ret_21d"], dual_head=False), "sentiment_v1"),
    }
    models["V5a: R4 + 6-Gauge Sentiment/Macro Gate"] = (models["R4 Baseline (Old Base Gate)"][0], "sentiment_v1")
    models["V5c: Dual-Head (Core + Sent/Macro/PEAD 11f) + Sent Gate"] = (models["V5b: Dual-Head (Core + Sent/Macro/PEAD 11f) + Base Gate"][0], "sentiment_v1")

    print("=== 1. STANDARD TOP-8 (Offset=0) vs 4-OFFSET AVERAGE (Days 0, 10, 20, 30) | 15 bps STT ===\n")
    print(f"{'Model':65s} | {'IC':>7s} | {'CAGR0':>6s} | {'Shrp0':>5s} | {'DD0':>7s} | {'AvgCAGR':>7s} | {'AvgShrp':>7s} | {'AvgDD':>7s} | {'AvgCrsh':>7s}")
    print("-" * 142)
    for name, (oos, gmode) in models.items():
        ic = float(spearmanr(oos["score"], oos["target_alpha"]).statistic)
        res = [sim_one(oos, prices, maps, gate_mode=gmode, top_k=8, cost_bps=15.0, rebal_offset=off) for off in (0, 10, 20, 30)]
        r0 = res[0]
        print(
            f"{name:65s} | {ic:+7.4f} | {r0['cagr']:+5.2f}% | {r0['sharpe']:5.2f} | {r0['max_dd']:6.2f}% | "
            f"{np.mean([x['cagr'] for x in res]):+6.2f}% | {np.mean([x['sharpe'] for x in res]):7.2f} | "
            f"{np.mean([x['max_dd'] for x in res]):6.2f}% | {np.mean([x['crash'] for x in res]):+6.1f}%"
        )

    print("\n=== 2. MULTI-K ROBUSTNESS (Avg across 4 Start Offsets, 15 bps) ===\n")
    for name in ["R4 Baseline (Old Base Gate)", "V5a: R4 + 6-Gauge Sentiment/Macro Gate", "V5c: Dual-Head (Core + Sent/Macro/PEAD 11f) + Sent Gate", "V5d: Dual-Head (Core + Focused 6f Macro/MAX/ON/PEAD) + Sent Gate"]:
        oos, gmode = models[name]
        parts = []
        for k in (6, 8, 10, 12):
            rs = [sim_one(oos, prices, maps, gate_mode=gmode, top_k=k, cost_bps=15.0, rebal_offset=off) for off in (0, 10, 20, 30)]
            parts.append(f"K={k:2d}: {np.mean([x['cagr'] for x in rs]):+5.2f}% (Sh {np.mean([x['sharpe'] for x in rs]):.2f}, DD {np.mean([x['max_dd'] for x in rs]):.1f}%)")
        print(f"{name:65s} | " + " | ".join(parts))

    print("\n=== 3. FRICTION STRESS & GATE PERTURBATION CHECK (V5c vs V5d vs V5a, Top-8 4-Offset Avg) ===\n")
    for name in ["V5a: R4 + 6-Gauge Sentiment/Macro Gate", "V5c: Dual-Head (Core + Sent/Macro/PEAD 11f) + Sent Gate", "V5d: Dual-Head (Core + Focused 6f Macro/MAX/ON/PEAD) + Sent Gate"]:
        oos, gmode = models[name]
        c_str = []
        for bps in (0, 15, 30, 50):
            rs = [sim_one(oos, prices, maps, gate_mode=gmode, top_k=8, cost_bps=bps, rebal_offset=off) for off in (0, 10, 20, 30)]
            c_str.append(f"{bps}bps: {np.mean([x['cagr'] for x in rs]):+5.2f}% (Sh {np.mean([x['sharpe'] for x in rs]):.2f})")
        g_str = []
        for gs in (0.90, 1.00, 1.10):
            rs = [sim_one(oos, prices, maps, gate_mode=gmode, top_k=8, cost_bps=15.0, rebal_offset=off, gate_scale=gs) for off in (0, 10, 20, 30)]
            g_str.append(f"scale={gs:.2f}: {np.mean([x['cagr'] for x in rs]):+5.2f}% (Sh {np.mean([x['sharpe'] for x in rs]):.2f})")
        print(f"{name}")
        print("   Slippage:  " + " | ".join(c_str))
        print("   Gate +/-10%: " + " | ".join(g_str))


if __name__ == "__main__":
    main()
