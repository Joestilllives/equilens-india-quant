"""
EquiLens India V7 — Hierarchical Risk Parity (HRP) Portfolio Allocator.

Implements:
  - López de Prado (2016) Hierarchical Risk Parity on Ledoit-Wolf (2004) shrunk 126D covariance
    (hierarchical clustering -> quasi-diagonalization -> recursive bisection inverse-variance).
  - Optional mild Grinold-Kahn alpha tilt (gamma in {0.0, 0.25}).
  - Budget-normalized Gârleanu-Pedersen (2013) L1 no-trade band preserving 100% of target
    equity budget (zero idle-cash leak).
  - Iterative single-stock and sector concentration cap enforcement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def apply_concentration_caps(
    weights: dict[str, float],
    sector_map: dict[str, str],
    max_stock_w: float | None,
    max_sector_w: float | None,
    target_total: float,
) -> dict[str, float]:
    """Iteratively cap single-stock and sector weights, redistributing excess to uncapped positions."""
    w = {s: max(0.0, float(val)) for s, val in weights.items()}
    if not w:
        return w
    for _ in range(20):
        changed = False
        if max_stock_w is not None:
            for s in list(w.keys()):
                if w[s] > max_stock_w + 1e-9:
                    w[s] = max_stock_w
                    changed = True
        if max_sector_w is not None:
            sec_totals: dict[str, float] = {}
            for s, val in w.items():
                sec = sector_map.get(s, "Other")
                sec_totals[sec] = sec_totals.get(sec, 0.0) + val
            for sec, stot in sec_totals.items():
                if stot > max_sector_w + 1e-9:
                    scale = max_sector_w / stot
                    for s in list(w.keys()):
                        if sector_map.get(s, "Other") == sec:
                            w[s] *= scale
                    changed = True
        cur_sum = sum(w.values())
        shortfall = target_total - cur_sum
        if shortfall > 1e-7:
            sec_totals = {}
            for s, val in w.items():
                sec = sector_map.get(s, "Other")
                sec_totals[sec] = sec_totals.get(sec, 0.0) + val
            eligible = [
                s
                for s, val in w.items()
                if (max_stock_w is None or val < max_stock_w - 1e-7)
                and (max_sector_w is None or sec_totals.get(sector_map.get(s, "Other"), 0.0) < max_sector_w - 1e-7)
            ]
            if eligible:
                elig_sum = sum(w[s] for s in eligible)
                if elig_sum > 1e-12:
                    for s in eligible:
                        w[s] += shortfall * (w[s] / elig_sum)
                    changed = True
        if not changed:
            break
    if max_stock_w is not None:
        for s in list(w.keys()):
            w[s] = min(w[s], max_stock_w)
    if max_sector_w is not None:
        sec_totals = {}
        for s, val in w.items():
            sec = sector_map.get(s, "Other")
            sec_totals[sec] = sec_totals.get(sec, 0.0) + val
        for sec, stot in sec_totals.items():
            if stot > max_sector_w + 1e-9:
                scale = max_sector_w / stot
                for s in list(w.keys()):
                    if sector_map.get(s, "Other") == sec:
                        w[s] *= scale
    return w


def apply_budget_normalized_no_trade_band(
    target_weights: dict[str, float],
    current_weights: dict[str, float] | None,
    band: float,
    target_total: float | None = None,
) -> dict[str, float]:
    """
    Gârleanu-Pedersen L1 no-trade band with exact budget renormalization (zero idle-cash leak).

    Existing holdings retained in `target_weights` whose |w_target - w_cur| < band are kept at
    `w_cur`, and the remaining equity budget is renormalized across actively traded symbols so
    sum(output) == target_total.
    """
    if not target_weights:
        return {}
    budget = float(sum(target_weights.values())) if target_total is None else float(target_total)
    if budget <= 1e-12 or band <= 0.0 or not current_weights:
        return dict(target_weights)

    frozen: list[str] = []
    free: list[str] = []
    for sym, w_tgt in target_weights.items():
        w_cur = float(current_weights.get(sym, 0.0))
        if w_cur > 1e-9 and abs(w_tgt - w_cur) < band:
            frozen.append(sym)
        else:
            free.append(sym)

    if not frozen:
        return dict(target_weights)

    if not free:
        frozen_vals = {s: max(0.0, float(current_weights.get(s, 0.0))) for s in frozen}
        f_sum = sum(frozen_vals.values())
        if f_sum <= 1e-12:
            return dict(target_weights)
        scale = budget / f_sum
        return {s: float(v * scale) for s, v in frozen_vals.items()}

    frozen_sum = sum(float(current_weights[s]) for s in frozen)
    rem_budget = budget - frozen_sum
    free_tgt_sum = sum(float(target_weights[s]) for s in free)

    # Guard against degenerate case where frozen holdings exceed the new (e.g., risk-gated) budget
    if rem_budget <= 1e-9 or free_tgt_sum <= 1e-12:
        return dict(target_weights)

    out: dict[str, float] = {}
    for s in frozen:
        out[s] = float(current_weights[s])
    for s in free:
        out[s] = float(rem_budget * (float(target_weights[s]) / free_tgt_sum))
    return out


def _cluster_variance(cov: np.ndarray, indices: list[int]) -> float:
    sub_cov = cov[np.ix_(indices, indices)]
    diag = np.clip(np.diag(sub_cov), 1e-10, None)
    ivp = 1.0 / diag
    ivp /= float(np.sum(ivp))
    return float(max(1e-12, ivp @ sub_cov @ ivp))


def _recursive_bisection(cov: np.ndarray, ordered_indices: list[int]) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n, dtype=float)
    clusters: list[list[int]] = [list(ordered_indices)]
    while clusters:
        next_clusters: list[list[int]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            left = c[:mid]
            right = c[mid:]
            v_left = _cluster_variance(cov, left)
            v_right = _cluster_variance(cov, right)
            alpha = 1.0 - (v_left / (v_left + v_right))
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters
    total = float(np.sum(w))
    return w / total if total > 1e-12 else np.full(n, 1.0 / max(1, n), dtype=float)


def _estimate_shrunk_cov(returns_matrix: np.ndarray) -> np.ndarray:
    """Estimate annualized covariance via Ledoit-Wolf shrinkage with PSD safety floor."""
    clean = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    n_obs, n_assets = clean.shape
    if n_obs < 15 or n_assets < 2:
        variances = np.var(clean, axis=0, ddof=1) if n_obs >= 2 else np.full(n_assets, (0.25 ** 2) / 252.0)
        return np.diag(np.clip(variances * 252.0, 1e-4, None))
    try:
        lw = LedoitWolf(assume_centered=False)
        lw.fit(clean)
        cov = np.asarray(lw.covariance_, dtype=float) * 252.0
    except Exception:
        cov = np.cov(clean, rowvar=False) * 252.0
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    if np.any(~np.isfinite(cov)) or np.any(diag <= 1e-8):
        fallback_var = np.clip(np.var(clean, axis=0) * 252.0, 1e-4, None)
        cov = np.diag(fallback_var)
    return cov


def compute_hrp_weights(
    symbols: list[str],
    returns_window: pd.DataFrame | None,
    fallback_vols: dict[str, float] | None = None,
    alpha_scores: dict[str, float] | None = None,
    alpha_tilt: float = 0.0,
    linkage_method: str = "single",
    target_exposure: float = 1.0,
    sector_map: dict[str, str] | None = None,
    max_stock_w: float | None = None,
    max_sector_w: float | None = None,
    current_weights: dict[str, float] | None = None,
    no_trade_band: float = 0.0,
) -> dict[str, float]:
    """
    Compute López de Prado (2016) HRP weights on trailing returns (126D Ledoit-Wolf covariance).

    Returns deterministic, non-negative weights summing to `target_exposure` (up to cap feasibility),
    obeying single-stock and sector caps and optional budget-normalized no-trade band.
    """
    n = len(symbols)
    if n == 0 or target_exposure <= 0.0:
        return {}
    if n == 1:
        w_single = min(float(target_exposure), max_stock_w) if max_stock_w is not None else float(target_exposure)
        if max_sector_w is not None:
            w_single = min(w_single, max_sector_w)
        return {symbols[0]: float(w_single)}

    use_hrp = (
        returns_window is not None
        and not returns_window.empty
        and len(returns_window) >= 30
        and all(s in returns_window.columns for s in symbols)
    )

    if use_hrp:
        sub = returns_window[symbols].fillna(0.0).to_numpy(dtype=float)
        cov = _estimate_shrunk_cov(sub)
        std = np.sqrt(np.clip(np.diag(cov), 1e-10, None))
        denom = np.outer(std, std)
        corr = np.divide(cov, denom, out=np.eye(n, dtype=float), where=denom > 1e-12)
        corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        if n >= 3:
            dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
            dist = 0.5 * (dist + dist.T)
            np.fill_diagonal(dist, 0.0)
            condensed = squareform(dist, checks=False)
            method = linkage_method if linkage_method in ("single", "ward", "complete", "average") else "single"
            link = linkage(condensed, method=method)
            ordered = [int(i) for i in leaves_list(link)]
        else:
            ordered = [0, 1]
        unit_w = _recursive_bisection(cov, ordered)
    else:
        vols = np.array(
            [max(0.12, float((fallback_vols or {}).get(s, 0.25))) for s in symbols],
            dtype=float,
        )
        inv_v = 1.0 / vols
        unit_w = inv_v / max(1e-12, float(np.sum(inv_v)))

    # Optional mild Grinold-Kahn alpha tilt (gamma = 0.0 -> pure HRP)
    if abs(alpha_tilt) > 1e-9 and alpha_scores:
        raw_a = np.array([float(alpha_scores.get(s, 0.0)) for s in symbols], dtype=float)
        a_std = float(np.std(raw_a, ddof=0))
        if np.isfinite(a_std) and a_std > 1e-9:
            z_a = np.clip((raw_a - float(np.mean(raw_a))) / a_std, -2.5, 2.5)
            tilted = unit_w * np.exp(float(alpha_tilt) * z_a)
            t_sum = float(np.sum(tilted))
            if t_sum > 1e-12 and np.all(np.isfinite(tilted)):
                unit_w = tilted / t_sum

    weights = {sym: float(w * target_exposure) for sym, w in zip(symbols, unit_w)}

    if max_stock_w is not None or max_sector_w is not None:
        weights = apply_concentration_caps(
            weights,
            sector_map or {},
            max_stock_w=max_stock_w,
            max_sector_w=max_sector_w,
            target_total=float(target_exposure),
        )

    if no_trade_band > 0.0 and current_weights:
        target_sum = float(sum(weights.values()))
        weights = apply_budget_normalized_no_trade_band(
            weights,
            current_weights=current_weights,
            band=float(no_trade_band),
            target_total=target_sum,
        )
        if max_stock_w is not None or max_sector_w is not None:
            weights = apply_concentration_caps(
                weights,
                sector_map or {},
                max_stock_w=max_stock_w,
                max_sector_w=max_sector_w,
                target_total=target_sum,
            )

    return weights
