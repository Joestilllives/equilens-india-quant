"""Comprehensive Unit & Regression Tests for EquiLens India V7 Canonical Book Upgrades (Phases 0-2)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.config import BENCHMARK_SYMBOL, DB_PATH, GOLDBEES_SYMBOL, NON_EQUITY_SYMBOLS, TRI_BENCHMARK_SYMBOL
from backend.database import Database
from backend.execution_engine import commit_paper_rebalance, generate_rebalance_ticket, get_paper_ledger_status
from backend.hrp_allocator import (
    apply_budget_normalized_no_trade_band,
    apply_concentration_caps,
    compute_hrp_weights,
)
from backend.ml_predictor import _build_panel, compute_current_macro_regime, compute_pollet_wilson_avg_corr
from backend.pit_fundamentals import (
    compute_pit_value_quality_sleeve,
    parse_nse_bhavcopy_csv,
    parse_nse_mto_dat,
    parse_xbrl_quarterly_filing,
    snapshot_current_fundamentals,
)
from backend.portfolio_backtester import compute_two_factor_alpha, run_oos_portfolio_backtest
from scripts.audit_overfitting_v6 import compute_cpcv_pbo, deflated_sharpe_ratio


def test_zero_non_equity_etf_leakage_across_pipeline(tmp_path) -> None:
    """Zero ^NSEI / NIFTYBEES.NS / GOLDBEES.NS rows in panel, latest_df, macro breadth, or backtest holdings."""
    tmp_db = tmp_path / "v7_etf_clean.db"
    shutil.copyfile(DB_PATH, tmp_db)
    db = Database(tmp_db)

    panel, latest_df = _build_panel(db, model_version="v7")
    assert set(panel["symbol"].unique()).isdisjoint(NON_EQUITY_SYMBOLS)
    assert set(latest_df["symbol"].unique()).isdisjoint(NON_EQUITY_SYMBOLS)
    for col in ("beta_vasicek", "resid_12m1m", "resid_6m1m", "mom_12_7", "cgo_126", "neg_beta", "target_resid_sharpe"):
        assert col in panel.columns
        assert np.all(np.isfinite(panel[col].to_numpy(dtype=float)))

    macro = compute_current_macro_regime(db)
    ohlcv_syms = set(db.get_all_ohlcv().keys())
    assert macro["universe_size"] <= len(ohlcv_syms - set(NON_EQUITY_SYMBOLS))
    assert "avg_corr_63" in macro and 0.0 <= macro["avg_corr_63"] <= 1.0
    assert isinstance(macro["dm_crash_damper_active"], bool)

    bt = run_oos_portfolio_backtest(db, top_k=10, rebalance_days=40, weighting_mode="hrp")
    assert bt["config"]["rank_buffer"] == 30
    assert bt["config"]["weighting_mode"] == "hrp"
    assert "excess_sharpe_rf62" in bt["portfolio_metrics"]
    assert "alpha_2f_pct" in bt["portfolio_metrics"]
    for snap in bt["_holdings_log"]:
        assert set(snap["weights"].keys()).isdisjoint(NON_EQUITY_SYMBOLS)


def test_strict_causality_truncated_tail_invariance(tmp_path) -> None:
    """
    Truncated-tail invariance canary: mutating all OHLCV & macro bars AFTER cutoff date T
    must have ZERO effect on beta_vasicek, resid_12m1m, resid_6m1m, mom_12_7, cgo_126,
    nse_factor_prior (incl. Daniel-Moskowitz damper), and avg_corr_63 computed at date <= T.
    """
    db_full = Database(DB_PATH)
    ohlcv = db_full.get_all_ohlcv()
    macro = db_full.get_macro_series()
    tickers = {t["symbol"]: t for t in db_full.get_tickers()}

    sample_syms = [BENCHMARK_SYMBOL] + [
        s for s in sorted(ohlcv.keys()) if s not in NON_EQUITY_SYMBOLS and len(ohlcv[s]) >= 600
    ][:12]
    bench_dates = [b["date"] for b in ohlcv[BENCHMARK_SYMBOL][-520:]]
    cutoff_date = bench_dates[400]

    db_orig = Database(tmp_path / "causal_orig.db")
    db_mut = Database(tmp_path / "causal_mut.db")

    for sym in sample_syms:
        bars = [dict(b) for b in ohlcv[sym] if b["date"] >= bench_dates[0]]
        mut_bars = []
        for b in bars:
            b2 = dict(b)
            if b2["date"] > cutoff_date:
                b2["open"] *= 3.1
                b2["high"] *= 3.5
                b2["low"] *= 0.25
                b2["close"] *= 3.2
                b2["volume"] = int(b2["volume"] * 40) + 1000
            mut_bars.append(b2)
        tmeta = tickers.get(sym, {"name": sym, "sector": "Technology"})
        for dbi, b_set in ((db_orig, bars), (db_mut, mut_bars)):
            dbi.upsert_ticker(
                symbol=sym, name=tmeta.get("name", sym), sector=tmeta.get("sector", "Technology"),
                industry="", exchange="NSE", summary="", fundamentals={}, data_source="test",
                updated_at="2026-11-26T00:00:00Z",
            )
            dbi.upsert_ohlcv(sym, b_set)

    for m_sym, m_series in macro.items():
        orig_m = {d: v for d, v in m_series.items() if d >= bench_dates[0]}
        mut_m = {d: (v * 3.0 if d > cutoff_date else v) for d, v in orig_m.items()}
        db_orig.upsert_macro_series(m_sym, list(orig_m.items()))
        db_mut.upsert_macro_series(m_sym, list(mut_m.items()))

    # Check truncated-tail invariance of all V7 features and DM-damped factor prior at cutoff_date
    p_orig, _ = _build_panel(db_orig, model_version="v7", prior_variant="book", enable_dm_damper=True)
    p_mut, _ = _build_panel(db_mut, model_version="v7", prior_variant="book", enable_dm_damper=True)

    s_orig = p_orig[p_orig["date"] == cutoff_date].sort_values("symbol").reset_index(drop=True)
    s_mut = p_mut[p_mut["date"] == cutoff_date].sort_values("symbol").reset_index(drop=True)
    assert not s_orig.empty and len(s_orig) == len(s_mut)

    causal_cols = [
        "beta_vasicek", "neg_beta", "resid_12m1m", "resid_6m1m",
        "mom_12_7", "cgo_126", "nse_factor_prior",
        "z_resid_12m1m", "z_mom_12_7", "z_cgo_126", "z_neg_beta",
    ]
    for col in causal_cols:
        assert np.allclose(
            s_orig[col].to_numpy(dtype=float),
            s_mut[col].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
        ), f"Future lookahead detected in {col}!"

    # Pollet-Wilson avg_corr_63 at cutoff_date must also be 100% invariant to post-cutoff mutations
    c_orig = compute_pollet_wilson_avg_corr(db_orig.get_all_ohlcv(), window=63, as_of_date=cutoff_date)
    c_mut = compute_pollet_wilson_avg_corr(db_mut.get_all_ohlcv(), window=63, as_of_date=cutoff_date)
    assert c_orig == c_mut


def test_hrp_weight_invariants_and_determinism() -> None:
    """HRP weights must be deterministic, non-negative, sum to target_exposure, and obey stock/sector caps."""
    rng = np.random.default_rng(123)
    syms = [f"S{i}.NS" for i in range(10)]
    sector_map = {s: f"Sec{i // 2}" for i, s in enumerate(syms)}
    # Create correlated cluster returns + distinct vols
    f1 = rng.normal(0.0, 0.015, size=126)
    f2 = rng.normal(0.0, 0.008, size=126)
    data = {}
    for i, s in enumerate(syms):
        if i < 5:
            data[s] = 0.8 * f1 + rng.normal(0.0, 0.012, size=126)
        else:
            data[s] = 0.8 * f2 + rng.normal(0.0, 0.005, size=126)
    rets = pd.DataFrame(data)
    alphas = {s: float(10 - i) for i, s in enumerate(syms)}

    w1 = compute_hrp_weights(
        syms, rets, alpha_scores=alphas, alpha_tilt=0.0, target_exposure=0.85,
        sector_map=sector_map, max_stock_w=0.12, max_sector_w=0.20,
    )
    w2 = compute_hrp_weights(
        syms, rets, alpha_scores=alphas, alpha_tilt=0.0, target_exposure=0.85,
        sector_map=sector_map, max_stock_w=0.12, max_sector_w=0.20,
    )
    assert w1 == w2  # Strictly deterministic
    assert all(v >= 0.0 for v in w1.values())
    assert abs(sum(w1.values()) - 0.85) < 1e-6
    assert all(v <= 0.12 + 1e-6 for v in w1.values())
    for sec in set(sector_map.values()):
        sec_w = sum(w1[s] for s in syms if sector_map[s] == sec)
        assert sec_w <= 0.20 + 1e-6

    # Lower-vol cluster (S5..S9) should receive higher total HRP allocation than high-vol cluster (S0..S4)
    w_uncapped = compute_hrp_weights(syms, rets, target_exposure=1.0)
    assert sum(w_uncapped[s] for s in syms[5:]) > sum(w_uncapped[s] for s in syms[:5])

    # Mild Grinold-Kahn tilt shifts weight toward higher-alpha assets while preserving sum
    w_tilted = compute_hrp_weights(syms, rets, alpha_scores=alphas, alpha_tilt=0.25, target_exposure=1.0)
    assert abs(sum(w_tilted.values()) - 1.0) < 1e-9
    assert w_tilted["S0.NS"] > w_uncapped["S0.NS"]


def test_budget_normalized_no_trade_band_zero_cash_leak() -> None:
    """Budget-normalized Gârleanu-Pedersen L1 band holds small drifts without leaking equity budget into idle cash."""
    cur = {"A.NS": 0.20, "B.NS": 0.20, "C.NS": 0.20, "D.NS": 0.20, "E.NS": 0.20}
    tgt = {"A.NS": 0.205, "B.NS": 0.195, "C.NS": 0.25, "D.NS": 0.15, "F.NS": 0.20}
    out = apply_budget_normalized_no_trade_band(tgt, cur, band=0.015, target_total=1.0)
    # A & B are within 1.5% band -> frozen at 0.20 exactly
    assert out["A.NS"] == pytest.approx(0.20)
    assert out["B.NS"] == pytest.approx(0.20)
    # Total equity budget preserved 100% (zero idle-cash leak)
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-9)
    assert "E.NS" not in out and out["F.NS"] > 0.0

    # All-frozen case scales proportionally to target_total (e.g., risk-gate shift to 0.85)
    out_all_frozen = apply_budget_normalized_no_trade_band(
        {"A.NS": 0.201, "B.NS": 0.199},
        {"A.NS": 0.20, "B.NS": 0.20},
        band=0.02,
        target_total=0.40,
    )
    assert sum(out_all_frozen.values()) == pytest.approx(0.40, abs=1e-9)


def test_corrected_dsr_and_cpcv_pbo_on_synthetic_series() -> None:
    """Verify Bailey-Lopez de Prado DSR (cross-trial variance) and CPCV PBO on known true-skill vs pure-noise series."""
    rng = np.random.default_rng(42)
    T = 1000

    # 1. Strong positive-alpha return series vs pure zero-alpha noise
    r_skill = rng.normal(0.0012, 0.010, size=T)
    r_noise = rng.normal(0.0, 0.010, size=T)
    trial_srs = list(rng.normal(0.2, 0.45, size=50))

    d_skill_1 = deflated_sharpe_ratio(r_skill, n_trials=1, trial_sharpes=trial_srs)
    d_skill_50 = deflated_sharpe_ratio(r_skill, n_trials=50, trial_sharpes=trial_srs)
    d_noise_50 = deflated_sharpe_ratio(r_noise, n_trials=50, trial_sharpes=trial_srs)

    assert d_skill_50["sr_threshold_ann"] > d_skill_1["sr_threshold_ann"] == 0.0
    assert d_skill_50["dsr_prob_pct"] > d_noise_50["dsr_prob_pct"]
    assert d_noise_50["dsr_prob_pct"] < 20.0
    # Cross-trial std must reflect trial_sharpes dispersion (~0.45), NOT 1/sqrt(T-1)
    assert abs(d_skill_50["trial_sr_std_ann"] - np.std(trial_srs, ddof=1)) < 0.01

    # 2. CPCV PBO: Matrix with genuine dominant strategy should have LOW PBO (< 15%)
    mat_skill = rng.normal(0.0001, 0.012, size=(T, 12))
    mat_skill[:, 0] = rng.normal(0.0016, 0.009, size=T)  # genuinely superior strategy
    pbo_skill = compute_cpcv_pbo(mat_skill, n_groups=8, k_test=4, purge_bars=5, annual_rf=0.0)
    assert pbo_skill["n_combinations"] == 70
    assert pbo_skill["pbo_pct"] < 15.0
    assert pbo_skill["pass_pbo_lt_25"] is True

    # 3. CPCV PBO: Pure i.i.d. zero-mean noise matrix should have HIGH PBO (~50%, clearly > 25%)
    mat_noise = rng.normal(0.0, 0.012, size=(T, 16))
    pbo_noise = compute_cpcv_pbo(mat_noise, n_groups=8, k_test=4, purge_bars=5, annual_rf=0.0)
    assert pbo_noise["pbo_pct"] > 30.0
    assert pbo_noise["pass_pbo_lt_25"] is False


def test_two_factor_alpha_and_phase2_pit_pipelines(tmp_path) -> None:
    """Test 2-Factor Pure Alpha math + Phase 2 PIT fundamentals snapshot, Bhavcopy parser, XBRL +2D lag & sleeve."""
    # 1. Exact 2-Factor Alpha recovery
    rng = np.random.default_rng(7)
    n = 500
    mkt = rng.normal(0.0004, 0.01, size=n)
    eqw = mkt + rng.normal(0.0001, 0.005, size=n)
    rf = 0.062 / 252.0
    true_alpha_daily = 0.08 / 252.0  # +8%/yr pure alpha
    port = rf + true_alpha_daily + 0.6 * (mkt - rf) + 0.7 * (eqw - mkt) + rng.normal(0.0, 0.001, size=n)
    tf = compute_two_factor_alpha(port, mkt, eqw, rf_daily=rf)
    assert abs(tf["alpha_2f_pct"] - 8.0) < 1.5
    assert abs(tf["beta_mkt"] - 0.6) < 0.05
    assert abs(tf["beta_size"] - 0.7) < 0.05

    # 2. Phase 2 Day-1 PIT snapshot logger (immutability + zero overwrite)
    tmp_db = tmp_path / "pit_test.db"
    shutil.copyfile(DB_PATH, tmp_db)
    db = Database(tmp_db)
    snap1 = snapshot_current_fundamentals(db, as_of_date="2026-11-26")
    assert snap1["inserted_new"] >= 50
    # Re-running on same date must insert 0 new rows (strict immutability via INSERT OR IGNORE)
    snap2 = snapshot_current_fundamentals(db, as_of_date="2026-11-26")
    assert snap2["inserted_new"] == 0

    # 3. PIT Value + Gross Profitability + Accruals sleeve causality check
    sleeve_before = compute_pit_value_quality_sleeve(db, as_of_date="2026-11-25")
    sleeve_on = compute_pit_value_quality_sleeve(db, as_of_date="2026-11-26")
    assert sleeve_before.empty
    assert len(sleeve_on) >= 50
    assert set(sleeve_on["symbol"]).isdisjoint(NON_EQUITY_SYMBOLS)

    # 4. BSE/NSE XBRL parser enforces strict +2 NSE trading-day availability lag
    sample_xbrl = """<?xml version="1.0" encoding="UTF-8"?>
    <xbrl xmlns:in-bse="http://www.bseindia.com/xbrl/fin">
      <in-bse:RevenueFromOperations>10000</in-bse:RevenueFromOperations>
      <in-bse:CostOfMaterialsConsumed>4000</in-bse:CostOfMaterialsConsumed>
      <in-bse:ProfitLossForPeriod>1200</in-bse:ProfitLossForPeriod>
      <in-bse:TotalAssets>40000</in-bse:TotalAssets>
      <in-bse:CashFlowsFromUsedInOperatingActivities>1500</in-bse:CashFlowsFromUsedInOperatingActivities>
      <in-bse:PurchaseOfPropertyPlantAndEquipment>300</in-bse:PurchaseOfPropertyPlantAndEquipment>
      <in-bse:TotalEquity>20000</in-bse:TotalEquity>
      <in-bse:Borrowings>5000</in-bse:Borrowings>
    </xbrl>"""
    parsed = parse_xbrl_quarterly_filing(sample_xbrl, "TCS.NS", filing_date="2026-11-23", market_cap_cr=50000.0)
    assert parsed["filing_date"] == "2026-11-23"
    assert parsed["as_of_date"] > "2026-11-23"  # Strict +2 NSE trading days lag
    assert parsed["gross_profit_to_assets"] == pytest.approx((6000.0 * 4.0) / 40000.0)
    assert parsed["accruals"] < 0.0  # CFO > Net Income => negative accruals (high earnings quality)

    # 5. NSE Bhavcopy CSV & MTO DAT parsers
    csv_sample = (
        "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
        "RELIANCE, EQ, 26-Nov-2026, 1300, 1305, 1320, 1295, 1315, 1315, 1310, 5000000, 65500.0, 120000, 3100000, 62.00\n"
        "RELIANCE, N1, 26-Nov-2026, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 100, 1.0, 5, 100, 100.00\n"
    )
    bhav = parse_nse_bhavcopy_csv(csv_sample, "2026-11-26", allowed_symbols={"RELIANCE.NS"})
    assert len(bhav) == 1 and bhav[0]["symbol"] == "RELIANCE.NS" and bhav[0]["deliv_pct"] == 62.0

    mto_sample = "20,1,TCS,EQ,2000000,1300000,65.00\n20,2,IGNORE,BL,1000,1000,100.0\n"
    mto = parse_nse_mto_dat(mto_sample, "2026-11-26", allowed_symbols={"TCS.NS"})
    assert len(mto) == 1 and mto[0]["symbol"] == "TCS.NS" and mto[0]["deliv_pct"] == 65.0


def test_v7_execution_4tranche_sleeves_and_adv_cap_warning(tmp_path) -> None:
    """Verify V7 HRP ticket, 4-tranche paper_sleeves persistence, and >2% 20D ADV warning at institutional capital."""
    tmp_db = tmp_path / "v7_exec.db"
    shutil.copyfile(DB_PATH, tmp_db)
    db = Database(tmp_db)

    # Existing ₹10L paper account remains intact & seamlessly exposes 4 staggered sleeves
    status_existing = get_paper_ledger_status(db)
    assert status_existing["initialized"] is True
    assert len(status_existing["sleeves"]) == 4

    # Huge ₹500 Cr ticket should trigger >2% ADV liquidity cap warning on at least one order
    tk_big = generate_rebalance_ticket(db, capital_inr=5_000_000_000.0, top_k=10, risk_profile="balanced", weighting_mode="hrp")
    assert tk_big["config"]["weighting_mode"] == "hrp"
    assert tk_big["summary"]["adv_cap_warnings"] >= 1
    assert any(o["adv_cap_warning"] for o in tk_big["orders"])

    # Commit rebalance persists 4 sleeves and FIFO lots cleanly
    db.reset_paper_account()
    res = commit_paper_rebalance(db, capital_inr=1_000_000.0, top_k=10, risk_profile="balanced", allow_stale=True, weighting_mode="hrp")
    assert len(db.get_paper_sleeves()) == 4
    assert len(db.get_paper_lots()) == 10
    assert res["ledger"]["account"]["next_scheduled_rebalance_date"] is not None
