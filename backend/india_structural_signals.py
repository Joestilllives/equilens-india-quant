"""
EquiLens India V8 (Pillar 2) — India-Specific Structural Signals & IIMA 4-Factor Alpha Attribution.

Implements:
  1. IIM Ahmedabad (Agarwalla, Jacob & Varma) 4-Factor Indian Equity Attribution (MKT, SMB, HML, WML):
     - Supports loading official free IIMA daily CSV (`MKT`, `SMB`, `HML`, `WML`, `RF`).
     - Automatically builds a strictly causal 4-Factor Indian proxy from the local NSE universe
       when official IIMA CSV is absent (MKT, Size-EqW SMB, Low-Vol/Value proxy, 12-1M WML),
       so 4-Factor Indian Alpha attribution works 100% offline with zero external dependency.
  2. Strictly Point-In-Time SEBI/NSE Quarterly Shareholding Pattern & Promoter Pledge parser
     (enforcing a hard +2 NSE trading-day availability lag after filing date).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import NON_EQUITY_SYMBOLS
from .nse_calendar import add_nse_trading_days


def load_iima_four_factor_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse official free IIM Ahmedabad Indian Market 4-Factor Daily CSV
    (https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/) into daily decimal returns.
    Columns normalized to: date, mkt_rf, smb, hml, wml, rf.
    """
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "mkt_rf", "smb", "hml", "wml", "rf"])

    text = path.read_text(encoding="utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text.strip()))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        rec = {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if k is not None}
        dt_raw = rec.get("date") or rec.get("trd_dt") or ""
        if not dt_raw:
            continue
        try:
            dt = pd.to_datetime(dt_raw).strftime("%Y-%m-%d")
        except Exception:
            continue

        def _f(keys: tuple[str, ...]) -> float:
            for k in keys:
                if k in rec and rec[k] != "":
                    try:
                        val = float(rec[k])
                        # IIMA files report percentages (e.g. 0.45 = 0.45%), auto-scale if > 0.25 typical
                        return val / 100.0 if abs(val) > 0.20 else val
                    except ValueError:
                        pass
            return 0.0

        rows.append(
            {
                "date": dt,
                "mkt_rf": _f(("mf", "mkt-rf", "mkt_rf", "rm-rf")),
                "smb": _f(("smb",)),
                "hml": _f(("hml",)),
                "wml": _f(("wml", "umd", "mom")),
                "rf": _f(("rf",)),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["date", "mkt_rf", "smb", "hml", "wml", "rf"]
    )


def build_causal_nse_four_factors(
    prices_df: pd.DataFrame,
    mkt_ret_series: pd.Series,
    annual_rf: float = 0.062,
) -> pd.DataFrame:
    """
    Build strictly causal daily Indian 4-Factor returns (MKT_RF, SMB_PROXY, REV_VAL_PROXY, WML_12_1)
    directly from the clean NSE equity price panel (T-1 lagged portfolio weights, zero lookahead).
    """
    stock_cols = [c for c in prices_df.columns if c not in NON_EQUITY_SYMBOLS and not c.startswith("^")]
    if len(stock_cols) < 10:
        return pd.DataFrame(columns=["date", "mkt_rf", "smb", "hml", "wml", "rf"])

    px = prices_df[stock_cols]
    rets = px.pct_change().clip(-0.25, 0.25)
    rf_daily = (1.0 + annual_rf) ** (1.0 / 252.0) - 1.0

    # 1. Market excess return
    mkt_rf = mkt_ret_series.reindex(px.index).fillna(0.0) - rf_daily

    # 2. Equal-weight size tilt proxy (EqW - Cap-Weighted NIFTY)
    eqw_ret = rets.mean(axis=1).fillna(0.0)
    smb_proxy = eqw_ret - mkt_ret_series.reindex(px.index).fillna(0.0)

    # 3. Strictly causal WML (12-1M cross-sectional Top-30% minus Bottom-30%, lagged 1 day!)
    mom_12_1 = (px.shift(21) / px.shift(252).replace(0, np.nan) - 1.0).shift(1)
    ranks_mom = mom_12_1.rank(axis=1, pct=True)
    w_win = (ranks_mom >= 0.70).astype(float)
    w_los = (ranks_mom <= 0.30).astype(float)
    w_win = w_win.div(w_win.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w_los = w_los.div(w_los.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    wml = (w_win * rets.fillna(0.0)).sum(axis=1) - (w_los * rets.fillna(0.0)).sum(axis=1)

    # 4. Strictly causal Long-Term Reversal / Value proxy (36M-12M loser minus winner, lagged 1 day;
    #    falls back to 252D-126D low-vol/contrarian when <3Y history)
    lt_rev = -(px.shift(126) / px.shift(504).replace(0, np.nan) - 1.0).shift(1)
    if lt_rev.notna().sum().sum() < 500:
        lt_rev = -(px.shift(21) / px.shift(252).replace(0, np.nan) - 1.0).shift(1)
    ranks_val = lt_rev.rank(axis=1, pct=True)
    w_hi = (ranks_val >= 0.70).astype(float)
    w_lo = (ranks_val <= 0.30).astype(float)
    w_hi = w_hi.div(w_hi.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w_lo = w_lo.div(w_lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    hml_proxy = (w_hi * rets.fillna(0.0)).sum(axis=1) - (w_lo * rets.fillna(0.0)).sum(axis=1)

    return pd.DataFrame(
        {
            "date": px.index.astype(str),
            "mkt_rf": mkt_rf.to_numpy(dtype=float),
            "smb": smb_proxy.to_numpy(dtype=float),
            "hml": hml_proxy.to_numpy(dtype=float),
            "wml": wml.to_numpy(dtype=float),
            "rf": np.full(len(px), rf_daily, dtype=float),
        }
    )


def compute_iima_four_factor_alpha(
    dates: list[str],
    port_rets: np.ndarray | list[float],
    factors_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Regress portfolio excess returns on Indian 4-Factor Model (MKT_RF, SMB, HML, WML):
        r_p - r_f = alpha_daily + b_mkt*MKT_RF + b_smb*SMB + b_hml*HML + b_wml*WML + eps
    Returns annualized 4-Factor Alpha (%), factor betas, alpha t-stat, and R^2.
    """
    p_df = pd.DataFrame({"date": [str(d)[:10] for d in dates], "rp": np.asarray(port_rets, dtype=float)})
    merged = p_df.merge(factors_df, on="date", how="inner").dropna()
    n = len(merged)
    if n < 30:
        return {
            "alpha_4f_pct": 0.0,
            "beta_mkt": 0.0,
            "beta_smb": 0.0,
            "beta_hml": 0.0,
            "beta_wml": 0.0,
            "alpha_tstat": 0.0,
            "r2": 0.0,
            "n_obs": n,
        }

    y = (merged["rp"] - merged["rf"]).to_numpy(dtype=float)
    X = np.column_stack(
        [
            np.ones(n, dtype=float),
            merged["mkt_rf"].to_numpy(dtype=float),
            merged["smb"].to_numpy(dtype=float),
            merged["hml"].to_numpy(dtype=float),
            merged["wml"].to_numpy(dtype=float),
        ]
    )
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(1, n - X.shape[1])
    s2 = float(np.sum(resid ** 2) / dof)
    cov_b = s2 * np.linalg.pinv(X.T @ X)
    se_alpha = float(np.sqrt(max(1e-18, cov_b[0, 0])))
    t_alpha = float(coef[0] / se_alpha)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "alpha_4f_pct": round(float(coef[0] * 252.0 * 100.0), 2),
        "beta_mkt": round(float(coef[1]), 3),
        "beta_smb": round(float(coef[2]), 3),
        "beta_hml": round(float(coef[3]), 3),
        "beta_wml": round(float(coef[4]), 3),
        "alpha_tstat": round(t_alpha, 2),
        "r2": round(float(np.clip(r2, 0.0, 1.0)), 3),
        "n_obs": n,
    }


def parse_sebi_shareholding_csv(csv_text: str) -> list[dict[str, Any]]:
    """
    Parse public NSE/BSE quarterly Shareholding Pattern CSV into strictly causal PIT rows
    stamped with `as_of_date = add_nse_trading_days(filing_date, 2)`.

    Expected columns (case-insensitive):
      symbol, filing_date, promoter_holding_pct, promoter_pledge_pct,
      promoter_qoq_chg, promoter_pledge_qoq_chg, dii_qoq_chg, fii_qoq_chg
    """
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    for raw in reader:
        rec = {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if k is not None}
        sym = rec.get("symbol", "").upper()
        fdt = rec.get("filing_date", "")[:10]
        if not sym or not fdt:
            continue
        if not sym.endswith(".NS") and not sym.startswith("^"):
            sym = f"{sym}.NS"

        def _opt(k: str) -> float | None:
            v = rec.get(k, "")
            if not v:
                return None
            try:
                f = float(v.replace("%", "").replace(",", ""))
                return f if np.isfinite(f) else None
            except ValueError:
                return None

        rows.append(
            {
                "symbol": sym,
                "filing_date": fdt,
                "as_of_date": add_nse_trading_days(fdt, 2),
                "promoter_holding_pct": _opt("promoter_holding_pct"),
                "promoter_pledge_pct": _opt("promoter_pledge_pct"),
                "promoter_qoq_chg": _opt("promoter_qoq_chg"),
                "promoter_pledge_qoq_chg": _opt("promoter_pledge_qoq_chg"),
                "dii_qoq_chg": _opt("dii_qoq_chg"),
                "fii_qoq_chg": _opt("fii_qoq_chg"),
            }
        )
    return rows
