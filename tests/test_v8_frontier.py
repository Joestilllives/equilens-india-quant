"""
EquiLens India V8 — Unit & Invariant Test Suite Covering All 6 Frontier Pillars:
  - Pillar 1: Indian Forensic Governance Shield & Point-In-Time QMJ + Value Composite
  - Pillar 2: IIMA 4-Factor Indian Equity Alpha (CSV parser + causal NSE 4-Factor builder) & SEBI Shareholding +2d lag
  - Pillar 3: Robert Carver Forecast Scaling, FDM, Multi-Sleeve Combiner & Smooth Regime Exposure
  - Pillar 4: AFML Ch. 4 Average Label Uniqueness Sample Weights & Meta-Labeling Confidence Sizing
  - Pillar 5: Boyd et al. (2017) Convex NSE Portfolio Optimizer (cap/budget invariants, turnover & FIFO tax awareness)
  - Pillar 6: India Domestic Liquidity Stress Index & Free RBI DBIE CSV Importer (strictly causal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.convex_optimizer import solve_convex_nse_portfolio
from backend.database import Database
from backend.forensic_quality_sleeve import compute_pit_qmj_value_sleeve, evaluate_indian_forensic_shield
from backend.india_liquidity_regime import compute_india_liquidity_stress_series, parse_rbi_dbie_liquidity_csv
from backend.india_structural_signals import (
    build_causal_nse_four_factors,
    compute_iima_four_factor_alpha,
    parse_sebi_shareholding_csv,
)
from backend.ml_predictor import compute_afml_uniqueness_weights
from backend.multi_sleeve_allocator import (
    combine_carver_sleeves,
    compute_carver_fdm,
    scale_carver_forecast,
    smooth_continuous_regime_exposure,
)


def test_pillar1_forensic_shield_and_pit_qmj(tmp_path) -> None:
    clean = evaluate_indian_forensic_shield(
        "TCS.NS", "IT Services",
        {"accruals": -0.02, "fcf_yield": 0.04, "debt_to_equity": 10.0, "roe": 0.45, "gross_profit_to_assets": 0.50},
        {"promoter_pledge_pct": 0.0, "promoter_qoq_chg": 0.0},
    )
    assert clean["blocked"] is False and clean["forensic_grade"] == "CLEAN"

    toxic = evaluate_indian_forensic_shield(
        "BLOWUP.NS", "Basic Materials",
        {"accruals": 0.14, "fcf_yield": -0.08, "debt_to_equity": 240.0, "roe": 0.02, "gross_profit_to_assets": 0.04},
        {"promoter_pledge_pct": 62.0, "promoter_pledge_qoq_chg": 12.0, "promoter_qoq_chg": -5.0},
    )
    assert toxic["blocked"] is True and toxic["flag_count"] >= 4 and toxic["forensic_grade"] == "FAIL"

    # Point-in-time QMJ strict causality check (future snapshot must be invisible!)
    db = Database(tmp_path / "p1.db")
    for sym, sec in [("A.NS", "IT Services"), ("B.NS", "Pharma"), ("C.NS", "Metals"), ("D.NS", "Auto")]:
        db.upsert_ticker(sym, sym, sec, "", "NSE", "", {}, "test", "2026-01-01T00:00:00Z")
    db.insert_pit_fundamentals_snapshots([
        {"symbol": "A.NS", "as_of_date": "2025-01-10", "trailing_pe": 15.0, "ev_to_ebitda": 10.0, "fcf_yield": 0.06, "gross_profit_to_assets": 0.45, "accruals": -0.03, "roe": 0.30, "debt_to_equity": 10.0, "source": "t", "created_at": "2025-01-10"},
        {"symbol": "B.NS", "as_of_date": "2025-01-10", "trailing_pe": 30.0, "ev_to_ebitda": 20.0, "fcf_yield": 0.02, "gross_profit_to_assets": 0.25, "accruals": 0.01, "roe": 0.18, "debt_to_equity": 40.0, "source": "t", "created_at": "2025-01-10"},
        {"symbol": "C.NS", "as_of_date": "2025-01-10", "trailing_pe": 50.0, "ev_to_ebitda": 35.0, "fcf_yield": -0.05, "gross_profit_to_assets": 0.05, "accruals": 0.12, "roe": 0.03, "debt_to_equity": 220.0, "source": "t", "created_at": "2025-01-10"},
        # Future revision on 2025-06-01 must NOT appear when querying 2025-02-01
        {"symbol": "C.NS", "as_of_date": "2025-06-01", "trailing_pe": 5.0, "ev_to_ebitda": 3.0, "fcf_yield": 0.20, "gross_profit_to_assets": 0.90, "accruals": -0.10, "roe": 0.60, "debt_to_equity": 0.0, "source": "t", "created_at": "2025-06-01"},
    ])
    qmj_feb = compute_pit_qmj_value_sleeve(db, "2025-02-01")
    assert qmj_feb.iloc[0]["symbol"] == "A.NS"
    assert qmj_feb.iloc[-1]["symbol"] == "C.NS"
    assert bool(qmj_feb.loc[qmj_feb["symbol"] == "C.NS", "forensic_blocked"].iloc[0]) is True


def test_pillar2_iima_four_factors_and_sebi_lag() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=400).strftime("%Y-%m-%d").tolist()
    mkt_r = rng.normal(0.0004, 0.01, size=len(dates))
    cols = {}
    for i in range(15):
        r = 0.8 * mkt_r + rng.normal(0.0002 * (i - 7) / 7.0, 0.015, size=len(dates))
        cols[f"S{i:02d}.NS"] = 100.0 * np.cumprod(1.0 + r)
    px_df = pd.DataFrame(cols, index=dates)
    mkt_s = pd.Series(mkt_r, index=dates)

    ff = build_causal_nse_four_factors(px_df, mkt_s, annual_rf=0.062)
    assert len(ff) == len(dates)
    # Construct a portfolio with exact +10% annualized pure alpha above market
    rf_d = (1.062 ** (1.0 / 252.0)) - 1.0
    p_rets = rf_d + 0.6 * (mkt_r - rf_d) + (0.10 / 252.0) + rng.normal(0.0, 0.002, size=len(dates))
    attr = compute_iima_four_factor_alpha(dates, p_rets, ff)
    assert 6.0 < attr["alpha_4f_pct"] < 14.0
    assert 0.45 < attr["beta_mkt"] < 0.75

    # SEBI shareholding +2 NSE trading-day availability lag check
    csv_txt = "symbol,filing_date,promoter_holding_pct,promoter_pledge_pct,promoter_qoq_chg,dii_qoq_chg\nRELIANCE,2026-04-15,50.3,0.0,+0.2,+0.6\n"
    parsed = parse_sebi_shareholding_csv(csv_txt)
    assert len(parsed) == 1
    assert parsed[0]["symbol"] == "RELIANCE.NS"
    assert parsed[0]["as_of_date"] > "2026-04-15"


def test_pillar3_carver_scaling_fdm_and_smooth_regime() -> None:
    raw = {"A": 3.0, "B": 1.0, "C": -1.0, "D": -3.0}
    sc = scale_carver_forecast(raw, target_abs=10.0, cap=20.0)
    assert abs(np.mean([abs(v) for v in sc.values()]) - 10.0) < 1e-3
    assert max(abs(v) for v in sc.values()) <= 20.0

    # Two uncorrelated sleeves -> FDM = sqrt(2) ≈ 1.414
    fdm = compute_carver_fdm([0.5, 0.5], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert abs(fdm - np.sqrt(2.0)) < 1e-4

    comb = combine_carver_sleeves(
        {"price": raw, "quality": {"A": 2.0, "B": -2.0, "C": 2.0, "D": -2.0}, "empty_sleeve": {}},
        {"price": 0.6, "quality": 0.4, "empty_sleeve": 0.5},
        corr_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert comb["A"] > comb["D"]
    assert all(-20.0 <= v <= 20.0 for v in comb.values())

    calm = smooth_continuous_regime_exposure(vix_level=12.0, bench_dist_200=0.06, breadth_50d=0.72, hilo_sent=0.18, gold_nifty_63d=0.0)
    panic = smooth_continuous_regime_exposure(vix_level=28.0, bench_dist_200=-0.10, breadth_50d=0.22, hilo_sent=-0.25, gold_nifty_63d=0.22, liquidity_stress=1.0)
    assert calm > 0.85 and panic < 0.35 and calm > panic


def test_pillar4_afml_uniqueness_weights_properties() -> None:
    dates = np.repeat(np.arange(100), 10)
    syms = np.tile([f"S{i}" for i in range(10)], 100)
    w = compute_afml_uniqueness_weights(dates, syms, horizon=40)
    assert len(w) == len(dates)
    assert np.all(np.isfinite(w)) and np.all(w > 0.0)
    assert abs(float(np.mean(w)) - 1.0) < 1e-6


def test_pillar5_convex_nse_optimizer_caps_budget_and_tax_awareness() -> None:
    rng = np.random.default_rng(123)
    syms = [f"S{i}.NS" for i in range(10)]
    sectors = {f"S{i}.NS": f"Sec{i // 2}" for i in range(10)}
    rets = pd.DataFrame(rng.normal(0.0, 0.015, size=(126, 10)), columns=syms)
    alphas = {s: float(10 - i) * 0.25 for i, s in enumerate(syms)}
    adv = {s: 5e8 for s in syms}

    w = solve_convex_nse_portfolio(
        symbols=syms, alpha_scores=alphas, returns_window=rets, sector_map=sectors,
        target_exposure=0.85, max_stock_w=0.12, max_sector_w=0.20, capital_inr=1_000_000.0, adv_map=adv,
    )
    assert abs(sum(w.values()) - 0.85) < 1e-4
    assert all(-1e-6 <= val <= 0.12 + 1e-5 for val in w.values())
    for sec in set(sectors.values()):
        assert sum(w[s] for s in syms if sectors[s] == sec) <= 0.20 + 1e-5

    # Turnover & FIFO tax awareness: holding an existing STCG winner reduces sell turnover vs zero-cost
    cur_w = {s: 0.085 for s in syms}
    lots = [{"symbol": "S9.NS", "buy_date": "2026-09-01", "shares": 100.0, "cost_per_share": 50.0}]
    prices = {s: 100.0 for s in syms}
    w_no_tax = solve_convex_nse_portfolio(
        symbols=syms, alpha_scores=alphas, returns_window=rets, sector_map=sectors,
        current_weights=cur_w, target_exposure=0.85, max_stock_w=0.12, max_sector_w=0.20,
        turnover_aversion=0.0, tax_aversion=0.0,
    )
    w_with_tax = solve_convex_nse_portfolio(
        symbols=syms, alpha_scores=alphas, returns_window=rets, sector_map=sectors,
        current_weights=cur_w, target_exposure=0.85, max_stock_w=0.12, max_sector_w=0.20,
        turnover_aversion=1.5, tax_aversion=1.0, fifo_lots=lots, prices=prices, as_of_date="2026-11-20",
    )
    turn_no = sum(abs(w_no_tax[s] - cur_w[s]) for s in syms)
    turn_yes = sum(abs(w_with_tax[s] - cur_w[s]) for s in syms)
    assert turn_yes <= turn_no + 1e-4


def test_pillar6_india_liquidity_stress_causality_and_rbi_parser() -> None:
    rbi_csv = "date,india_10y_gsec,india_1y_tbill,rbi_laf_net_cr,bank_credit_yoy,amfi_sip_cr\n2025-01-15,6.85,6.65,-120000,11.2,26400\n"
    rdf = parse_rbi_dbie_liquidity_csv(rbi_csv)
    assert len(rdf) == 1 and abs(float(rdf.iloc[0]["gsec_slope_10y_1y"]) - 0.20) < 1e-6

    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2024-01-01", periods=150).strftime("%Y-%m-%d").tolist()
    cols = {"^NSEI": 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, len(dates)))}
    for sym in ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "RELIANCE.NS", "ITC.NS", "LT.NS", "SUNPHARMA.NS", "MARUTI.NS"]:
        cols[sym] = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, len(dates)))
    px_full = pd.DataFrame(cols, index=dates)
    s_full = compute_india_liquidity_stress_series(px_full, px_full["^NSEI"], vix_map={}, rbi_df=rdf)
    # Truncated-tail causality: mutating future dates after index 110 must NOT alter day 110 stress
    px_mut = px_full.copy()
    px_mut.iloc[111:] *= 0.5
    s_mut = compute_india_liquidity_stress_series(px_mut, px_mut["^NSEI"], vix_map={}, rbi_df=rdf)
    assert abs(float(s_full.iloc[110]) - float(s_mut.iloc[110])) < 1e-12
