"""
EquiLens India V8 (Pillar 1) — Fundamental Quality (QMJ) & Indian Forensic Governance Shield.

Implements:
  - Asness, Frazzini & Pedersen (2019) "Quality Minus Junk" (QMJ):
      Profitability (GP/Assets, ROE, Cash-Flow Yield) + Safety (Low Debt/Equity, Low Volatility)
  - Robert Novy-Marx (2013) "The Other Side of Value: The Gross Profitability Premium"
  - Stephen Penman "Accounting for Value" + Messod Beneish (1999) M-Score & Howard Schilit
    "Financial Shenanigans" + Saurabh Mukherjea "Diamonds in the Dust" Indian Forensic Shield:
      * Cash-conversion divergence (high accruals / weak CFO-to-PAT)
      * Promoter share pledging > 25% or QoQ pledge spike > +5%
      * Non-financial leverage & deteriorating balance-sheet red flags
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import NON_EQUITY_SYMBOLS
from .database import Database

_FINANCIAL_SECTORS = {"Financial Services", "Banking", "NBFC", "Insurance", "Capital Markets"}


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _cs_z(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if len(valid) < 3:
        return pd.Series(0.0, index=series.index)
    std = float(valid.std(ddof=0))
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return ((series - float(valid.mean())) / std).clip(-3.0, 3.0).fillna(0.0)


def evaluate_indian_forensic_shield(
    symbol: str,
    sector: str,
    pit_snapshot: dict[str, Any] | None,
    shareholding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Strictly Point-In-Time Indian Forensic Accounting & Promoter Governance Shield.

    Flags companies exhibiting classic Indian midcap blowup patterns (Mukherjea / Schilit / Beneish):
      1. Severe Accounting Accruals (Sloan 1996 / Beneish): accruals > +0.08 (paper profits without cash)
      2. Negative Free Cash Flow + High Leverage (non-financials): FCF yield < -3% & D/E > 120%
      3. Weak Profitability / Return Trap: ROE < 6% & Gross Profit/Assets < 0.08 (non-financials)
      4. Promoter Share Pledging Red Flag: promoter_pledge_pct > 25% OR pledge QoQ spike > +5%
      5. Promoter Stake Exit: promoter_qoq_chg < -3.0% (sharp promoter dumping)
    """
    flags: list[str] = []
    is_fin = str(sector or "") in _FINANCIAL_SECTORS
    snap = pit_snapshot or {}
    sh = shareholding or {}

    acc = _safe_float(snap.get("accruals"))
    fcf_y = _safe_float(snap.get("fcf_yield"))
    de = _safe_float(snap.get("debt_to_equity"))
    roe = _safe_float(snap.get("roe"))
    gp_a = _safe_float(snap.get("gross_profit_to_assets"))

    if acc is not None and acc > 0.08:
        flags.append(f"HIGH_ACCRUALS({acc:+.2f})")
    if not is_fin:
        if fcf_y is not None and de is not None and fcf_y < -0.03 and de > 120.0:
            flags.append(f"NEG_FCF_HIGH_DEBT(FCF={fcf_y*100:.1f}%,DE={de:.0f}%)")
        if de is not None and de > 200.0:
            flags.append(f"EXTREME_LEVERAGE(DE={de:.0f}%)")
        if roe is not None and gp_a is not None and roe < 0.06 and gp_a < 0.08:
            flags.append(f"WEAK_PROFITABILITY(ROE={roe*100:.1f}%,GPA={gp_a:.2f})")
    else:
        if roe is not None and roe < 0.05:
            flags.append(f"WEAK_FIN_ROE({roe*100:.1f}%)")

    pledge_pct = _safe_float(sh.get("promoter_pledge_pct"))
    pledge_chg = _safe_float(sh.get("promoter_pledge_qoq_chg"))
    prom_chg = _safe_float(sh.get("promoter_qoq_chg"))

    if pledge_pct is not None and pledge_pct > 25.0:
        flags.append(f"PROMOTER_PLEDGE_HIGH({pledge_pct:.1f}%)")
    if pledge_chg is not None and pledge_chg > 5.0:
        flags.append(f"PROMOTER_PLEDGE_SPIKE({pledge_chg:+.1f}%)")
    if prom_chg is not None and prom_chg < -3.0:
        flags.append(f"PROMOTER_DUMPING({prom_chg:+.1f}%)")

    blocked = len(flags) >= 2 or (pledge_pct is not None and pledge_pct > 50.0)
    return {
        "symbol": str(symbol).upper(),
        "flag_count": len(flags),
        "flags": flags,
        "blocked": blocked,
        "forensic_grade": "FAIL" if blocked else ("WATCH" if len(flags) == 1 else "CLEAN"),
    }


def compute_pit_qmj_value_sleeve(
    db: Database,
    as_of_date: str,
    shareholding_as_of: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build strictly causal Point-In-Time QMJ (Quality Minus Junk) + Value + Forensic Governance sleeve
    using ONLY records stamped `as_of_date <= query_date`.

    Weights (Asness-Moskowitz-Pedersen 2013 + AFP 2019 QMJ + Novy-Marx 2013):
      - 40% Value: z(E/P), z(FCF/EV), z(EBITDA/EV)
      - 35% Profitability Quality: z(GP/Assets), z(ROE), -z(Accruals)
      - 15% Safety: -z(Debt/Equity for non-financials)
      - 10% Indian Governance Bonus: +z(DII/Promoter accumulation) - forensic flag penalty
    """
    snaps = db.get_pit_fundamentals_as_of(as_of_date) if hasattr(db, "get_pit_fundamentals_as_of") else {}
    if not snaps:
        return pd.DataFrame(
            columns=[
                "symbol", "as_of_date", "value_score", "profitability_score",
                "safety_score", "forensic_flags", "forensic_blocked", "qmj_value_composite",
            ]
        )

    sector_map = {t["symbol"]: t["sector"] for t in db.get_tickers()}
    sh_map = shareholding_as_of or {}

    rows: list[dict[str, Any]] = []
    for sym, r in snaps.items():
        if sym in NON_EQUITY_SYMBOLS or sym.startswith("^"):
            continue
        sec = sector_map.get(sym, "Unknown")
        is_fin = sec in _FINANCIAL_SECTORS
        pe = _safe_float(r.get("trailing_pe"))
        ep = (1.0 / pe) if (pe is not None and pe > 1.0) else np.nan
        fcf_y = _safe_float(r.get("fcf_yield"))
        ev_eb = _safe_float(r.get("ev_to_ebitda"))
        eb_ev = (1.0 / ev_eb) if (ev_eb is not None and ev_eb > 0.5) else np.nan
        gp_a = _safe_float(r.get("gross_profit_to_assets"))
        roe = _safe_float(r.get("roe"))
        acc = _safe_float(r.get("accruals"))
        de = _safe_float(r.get("debt_to_equity")) if not is_fin else np.nan

        sh = sh_map.get(sym, {})
        shield = evaluate_indian_forensic_shield(sym, sec, r, sh)
        gov_signal = 0.0
        if _safe_float(sh.get("dii_qoq_chg")) is not None:
            gov_signal += float(sh["dii_qoq_chg"])
        if _safe_float(sh.get("promoter_qoq_chg")) is not None:
            gov_signal += float(sh["promoter_qoq_chg"])

        rows.append(
            {
                "symbol": sym,
                "as_of_date": str(r["as_of_date"])[:10],
                "ep": ep,
                "fcf_yield": fcf_y if fcf_y is not None else np.nan,
                "ebitda_ev": eb_ev,
                "gp_to_assets": gp_a if gp_a is not None else np.nan,
                "roe": roe if roe is not None else np.nan,
                "accruals": acc if acc is not None else np.nan,
                "de": de if de is not None else np.nan,
                "gov_signal": gov_signal,
                "forensic_flags": shield["flag_count"],
                "forensic_blocked": shield["blocked"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["value_score"] = (0.40 * _cs_z(df["ep"]) + 0.40 * _cs_z(df["fcf_yield"]) + 0.20 * _cs_z(df["ebitda_ev"])).round(4)
    df["profitability_score"] = (
        0.45 * _cs_z(df["gp_to_assets"])
        + 0.35 * _cs_z(df["roe"])
        - 0.20 * _cs_z(df["accruals"])
    ).round(4)
    df["safety_score"] = (-_cs_z(df["de"])).round(4)
    z_gov = _cs_z(df["gov_signal"])
    penalty = df["forensic_flags"].astype(float) * 0.50

    df["qmj_value_composite"] = (
        0.40 * df["value_score"]
        + 0.35 * df["profitability_score"]
        + 0.15 * df["safety_score"]
        + 0.10 * z_gov
        - penalty
    ).round(4)
    return df.sort_values("qmj_value_composite", ascending=False).reset_index(drop=True)
