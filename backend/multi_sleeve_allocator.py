"""
EquiLens India V8 (Pillar 3) — Robert Carver Forecast Scaling & All-Weather Multi-Sleeve Allocator.

Implements:
  - Robert Carver (2015, 2023) "Systematic Trading":
      * Standardized forecast scaling (mean |forecast| = 10.0, capped at [-20.0, +20.0])
      * Forecast Diversification Multiplier (FDM = 1 / sqrt(w^T C w), capped at 2.5)
      * Orthogonal multi-sleeve combination (Sleeve A: V7/V8 Price+ML; Sleeve B: PIT Value+QMJ
        Quality activated strictly when PIT depth >= min_pit_dates; Sleeve C: Defensive Reserve)
  - Smooth continuous risk-budgeting regime exposure curve (tested head-to-head against V6/V7's
    discrete 6-Gauge stress step function).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def scale_carver_forecast(
    raw_scores: dict[str, float] | pd.Series,
    target_abs: float = 10.0,
    cap: float = 20.0,
) -> dict[str, float]:
    """
    Carver (2015) standardized cross-sectional forecast scaling:
    Centers to zero mean, scales so E[|f|] == target_abs (default 10.0), and clips to [-cap, +cap].
    """
    if isinstance(raw_scores, pd.Series):
        items = {str(k): float(v) for k, v in raw_scores.dropna().items() if np.isfinite(v)}
    else:
        items = {str(k): float(v) for k, v in raw_scores.items() if v is not None and np.isfinite(v)}
    if not items:
        return {}

    vals = np.array(list(items.values()), dtype=float)
    centered = vals - float(np.mean(vals))
    mean_abs = float(np.mean(np.abs(centered)))
    if not np.isfinite(mean_abs) or mean_abs < 1e-9:
        return {k: 0.0 for k in items.keys()}

    scaled = np.clip(centered * (target_abs / mean_abs), -cap, cap)
    return {k: round(float(v), 4) for k, v in zip(items.keys(), scaled)}


def compute_carver_fdm(
    weights: list[float] | np.ndarray,
    corr_matrix: np.ndarray | None = None,
    max_fdm: float = 2.5,
) -> float:
    """
    Compute Carver Forecast Diversification Multiplier:
        FDM = 1 / sqrt(w^T * Corr * w), bounded to [1.0, max_fdm].
    Negative correlations are floored at 0.0 per Carver's conservative anti-overfitting rule.
    """
    w = np.asarray(weights, dtype=float)
    w_sum = float(np.sum(np.abs(w)))
    if len(w) <= 1 or w_sum <= 1e-12:
        return 1.0
    w = w / w_sum
    if corr_matrix is None:
        return 1.0
    c = np.asarray(corr_matrix, dtype=float)
    if c.shape != (len(w), len(w)):
        return 1.0
    c_safe = np.clip(0.5 * (c + c.T), 0.0, 1.0)
    np.fill_diagonal(c_safe, 1.0)
    denom = float(np.sqrt(max(1e-9, float(w @ c_safe @ w))))
    return float(np.clip(1.0 / denom, 1.0, max_fdm))


def combine_carver_sleeves(
    sleeve_scores: dict[str, dict[str, float]],
    sleeve_weights: dict[str, float],
    corr_matrix: np.ndarray | None = None,
    cap: float = 20.0,
) -> dict[str, float]:
    """
    Combine multiple orthogonal alpha sleeves (e.g. 'price_v8', 'pit_qmj_value') using Carver
    forecast scaling and FDM. Automatically ignores empty sleeves and renormalizes active weights.
    """
    active_names = [
        name for name, scores in sleeve_scores.items()
        if scores and float(sleeve_weights.get(name, 0.0)) > 0.0
    ]
    if not active_names:
        return {}

    scaled_sleeves = {name: scale_carver_forecast(sleeve_scores[name], cap=cap) for name in active_names}
    raw_w = np.array([float(sleeve_weights[name]) for name in active_names], dtype=float)
    norm_w = raw_w / max(1e-12, float(np.sum(raw_w)))

    sub_corr = None
    if corr_matrix is not None and corr_matrix.shape == (len(active_names), len(active_names)):
        sub_corr = corr_matrix
    fdm = compute_carver_fdm(norm_w, sub_corr)

    all_syms = sorted({s for sc in scaled_sleeves.values() for s in sc.keys()})
    combined: dict[str, float] = {}
    for sym in all_syms:
        val = 0.0
        for name, w_i in zip(active_names, norm_w):
            val += float(w_i) * float(scaled_sleeves[name].get(sym, 0.0))
        combined[sym] = round(float(np.clip(val * fdm, -cap, cap)), 4)
    return combined


def smooth_continuous_regime_exposure(
    vix_level: float,
    bench_dist_200: float,
    breadth_50d: float,
    hilo_sent: float,
    gold_nifty_63d: float,
    liquidity_stress: float = 0.0,
    min_exp: float = 0.25,
    max_exp: float = 1.00,
) -> float:
    """
    Smooth continuous sigmoid risk-budgeting exposure curve replacing discrete step cliffs
    (100% -> 85% -> 55% -> 25%), with optional domestic liquidity stress input.
    """
    # Continuous normalized stress score centered around V6's calibrated thresholds
    s_vix = float(np.clip((vix_level - 15.0) / 8.0, 0.0, 2.5))
    s_trend = float(np.clip((-bench_dist_200) / 0.08, 0.0, 1.5))
    s_breadth = float(np.clip((0.50 - breadth_50d) / 0.20, 0.0, 1.5))
    s_hilo = float(np.clip((-0.05 - hilo_sent) / 0.20, 0.0, 1.2))
    s_gold = float(np.clip((gold_nifty_63d - 0.10) / 0.12, 0.0, 1.2))
    s_liq = float(np.clip(liquidity_stress, 0.0, 1.5))

    total_stress = 0.85 * s_vix + 0.85 * s_trend + 0.85 * s_breadth + 0.65 * s_hilo + 0.65 * s_gold + 0.50 * s_liq
    # Smooth logistic decay: stress=0 -> ~1.0, stress=1.5 -> ~0.78, stress=3.0 -> ~0.35
    sig = 1.0 / (1.0 + float(np.exp(1.15 * (total_stress - 2.15))))
    return round(float(np.clip(min_exp + (max_exp - min_exp) * sig, min_exp, max_exp)), 4)
