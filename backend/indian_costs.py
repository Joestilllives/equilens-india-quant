"""Indian NSE Equity Delivery Cost, Market Impact, Capacity & FIFO Tax Engine.

Provides a single, deterministic, side-explicit Indian transaction cost and
capital-gains tax accounting engine shared by portfolio_backtester.py,
backtester.py, and execution_engine.py:

  Statutory NSE Delivery Charges (Finance Act 2024/2025):
    - STT:              0.10% (10.0 bps) on BUY and SELL delivery value
    - Stamp Duty:       0.015% (1.5 bps) on BUY only
    - NSE Exchange Fee: 0.00297% (~0.30 bps) on BUY and SELL
    - SEBI Fee:         0.0001% (0.01 bps) on BUY and SELL
    - GST:              18% on (Brokerage + Exchange Fee + SEBI Fee)
    - DP Charge:        ₹15.93 (incl. GST) flat per SELL scrip-day (optional)
    - Spread + Impact:  Half-spread (3 bps default) + eta * sigma_d * sqrt(order_val / ADV_20d)

  Indian FIFO Capital-Gains Taxation (Finance Act 2024):
    - Holding < 365 days:  STCG taxed at 20% + 4% Health & Education Cess = 20.8%
    - Holding >= 365 days: LTCG taxed at 12.5% + 4% Cess = 13.0% above ₹1,25,000/FY exemption
    - Loss set-off:        STCL offsets STCG then LTCG; LTCL offsets LTCG only;
                           unabsorbed losses carry forward across Indian FYs (Apr 1 – Mar 31).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# Statutory Indian NSE Cash-Delivery Rates
NSE_DELIVERY_STT_RATE = 0.0010          # 0.10% on both BUY and SELL
NSE_BUY_STAMP_DUTY_RATE = 0.00015       # 0.015% on BUY only
NSE_EXCHANGE_TXN_RATE = 0.0000297       # 0.00297% on both BUY and SELL
SEBI_TURNOVER_RATE = 0.000001           # ₹10 per crore (0.0001%)
GST_RATE = 0.18                         # 18% GST on brokerage + exchange + SEBI fees
DEFAULT_DP_SELL_CHARGE_INR = 15.93      # CDSL/NSDL DP charge incl. 18% GST per sell scrip-day

# Indian Capital Gains Tax Rates (Finance Act 2024, incl. 4% Health & Education Cess)
INDIA_STCG_EFFECTIVE_RATE = 0.20 * 1.04       # 20.8% (Equity & Arbitrage Funds < 365 days)
INDIA_LTCG_EFFECTIVE_RATE = 0.125 * 1.04      # 13.0% (Equity & Arbitrage Funds >= 365 days, Gold ETF >= 365 days)
INDIA_LTCG_FY_EXEMPTION_INR = 125_000.0       # ₹1.25 Lakh annual exemption per Indian FY
INDIA_DEBT_SLAB_EFFECTIVE_RATE = 0.30 * 1.04  # 31.2% (30% marginal slab + 4% cess: LIQUIDCASE & <365d GOLDBEES)
INDIA_ARBITRAGE_STCG_RATE = INDIA_STCG_EFFECTIVE_RATE  # 20.8% (Direct Arbitrage Fund >65% equity)


def indian_financial_year(date_str: str) -> str:
    """Return Indian Financial Year label (e.g. 'FY2026-27' for Apr 2026 – Mar 2027)."""
    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
    start_yr = dt.year if dt.month >= 4 else dt.year - 1
    return f"FY{start_yr}-{str(start_yr + 1)[-2:]}"


def compute_indian_delivery_cost(
    side: str,
    gross_value_inr: float,
    adv_20d_inr: float | None = None,
    daily_vol: float = 0.02,
    brokerage_rate: float = 0.0,
    brokerage_cap_inr: float = 0.0,
    half_spread_bps: float = 3.0,
    impact_eta: float = 0.25,
    include_dp_charge: bool = True,
    extra_slippage_bps: float = 0.0,
) -> dict[str, float]:
    """Compute itemized Indian NSE delivery statutory charges + spread & ADV market impact.

    Side must be 'BUY' or 'SELL'. Always returns positive INR costs and effective one-way bps.
    """
    gross = float(max(0.0, gross_value_inr))
    if gross <= 0.0:
        return {
            "gross_value_inr": 0.0,
            "stt_inr": 0.0,
            "stamp_duty_inr": 0.0,
            "exchange_fee_inr": 0.0,
            "sebi_fee_inr": 0.0,
            "brokerage_inr": 0.0,
            "gst_inr": 0.0,
            "dp_charge_inr": 0.0,
            "statutory_inr": 0.0,
            "spread_inr": 0.0,
            "impact_inr": 0.0,
            "slippage_inr": 0.0,
            "total_cost_inr": 0.0,
            "total_cost_bps": 0.0,
            "impact_bps": 0.0,
            "participation_pct": 0.0,
        }

    is_buy = side.strip().upper() in ("BUY", "ADD")
    stt = gross * NSE_DELIVERY_STT_RATE
    stamp = gross * NSE_BUY_STAMP_DUTY_RATE if is_buy else 0.0
    exch = gross * NSE_EXCHANGE_TXN_RATE
    sebi = gross * SEBI_TURNOVER_RATE

    brok = gross * max(0.0, brokerage_rate)
    if brokerage_cap_inr > 0.0:
        brok = min(brok, brokerage_cap_inr)
    gst = GST_RATE * (brok + exch + sebi)
    dp = DEFAULT_DP_SELL_CHARGE_INR if (not is_buy and include_dp_charge) else 0.0
    statutory = stt + stamp + exch + sebi + brok + gst + dp

    # Spread + square-root market impact model
    spread_bps = max(0.0, float(half_spread_bps)) + max(0.0, float(extra_slippage_bps))
    impact_bps = 0.0
    participation_pct = 0.0
    if adv_20d_inr is not None and np.isfinite(adv_20d_inr) and adv_20d_inr > 0.0:
        part = gross / float(adv_20d_inr)
        participation_pct = part * 100.0
        vol_clamped = float(np.clip(daily_vol if np.isfinite(daily_vol) else 0.02, 0.005, 0.10))
        impact_bps = min(250.0, float(impact_eta * vol_clamped * np.sqrt(part) * 10000.0))

    spread_inr = gross * (spread_bps / 10000.0)
    impact_inr = gross * (impact_bps / 10000.0)
    slippage = spread_inr + impact_inr
    total = statutory + slippage
    total_bps = (total / gross) * 10000.0

    return {
        "gross_value_inr": round(gross, 2),
        "stt_inr": round(stt, 2),
        "stamp_duty_inr": round(stamp, 2),
        "exchange_fee_inr": round(exch, 2),
        "sebi_fee_inr": round(sebi, 4),
        "brokerage_inr": round(brok, 2),
        "gst_inr": round(gst, 2),
        "dp_charge_inr": round(dp, 2),
        "statutory_inr": round(statutory, 2),
        "spread_inr": round(spread_inr, 2),
        "impact_inr": round(impact_inr, 2),
        "slippage_inr": round(slippage, 2),
        "total_cost_inr": round(total, 2),
        "total_cost_bps": round(total_bps, 2),
        "impact_bps": round(impact_bps, 2),
        "participation_pct": round(participation_pct, 4),
    }


@dataclass
class TaxLot:
    symbol: str
    buy_date: str
    shares: float
    unit_cost_inr: float  # purchase price + pro-rata buy cost per share
    sleeve_id: int = 1


@dataclass
class FifoTaxLedger:
    """Transaction-level FIFO Indian STCG/LTCG tax ledger with FY exemptions & loss carryforward."""

    lots: dict[str, list[TaxLot]] = field(default_factory=dict)
    stcl_carry_inr: float = 0.0
    ltcl_carry_inr: float = 0.0
    fy_ltcg_exemption_used: dict[str, float] = field(default_factory=dict)
    total_stcg_realized_inr: float = 0.0
    total_ltcg_realized_inr: float = 0.0
    total_stcg_tax_inr: float = 0.0
    total_ltcg_tax_inr: float = 0.0
    total_fees_inr: float = 0.0

    def buy(
        self,
        symbol: str,
        buy_date: str,
        shares: float,
        price: float,
        buy_cost_inr: float = 0.0,
        sleeve_id: int = 1,
    ) -> None:
        if shares <= 1e-12:
            return
        unit_cost = (shares * price + max(0.0, buy_cost_inr)) / shares
        self.total_fees_inr += max(0.0, buy_cost_inr)
        self.lots.setdefault(symbol, []).append(
            TaxLot(
                symbol=symbol,
                buy_date=str(buy_date)[:10],
                shares=float(shares),
                unit_cost_inr=float(unit_cost),
                sleeve_id=sleeve_id,
            )
        )

    def sell(
        self,
        symbol: str,
        sell_date: str,
        shares: float,
        price: float,
        sell_cost_inr: float = 0.0,
    ) -> dict[str, float]:
        """Sell `shares` of `symbol` using strict FIFO; compute Indian STCG/LTCG tax."""
        qty_to_sell = float(max(0.0, shares))
        if qty_to_sell <= 1e-12:
            return {
                "realized_pnl_inr": 0.0,
                "stcg_gain_inr": 0.0,
                "ltcg_gain_inr": 0.0,
                "stcg_tax_inr": 0.0,
                "ltcg_tax_inr": 0.0,
                "total_tax_inr": 0.0,
            }

        self.total_fees_inr += max(0.0, sell_cost_inr)
        sell_dt = datetime.strptime(str(sell_date)[:10], "%Y-%m-%d").date()
        fy = indian_financial_year(sell_date)
        net_sell_px = price - (max(0.0, sell_cost_inr) / qty_to_sell)

        queue = self.lots.get(symbol, [])
        stcg_raw = 0.0
        ltcg_raw = 0.0
        rem = qty_to_sell

        while rem > 1e-12 and queue:
            lot = queue[0]
            take = min(rem, lot.shares)
            buy_dt = datetime.strptime(lot.buy_date[:10], "%Y-%m-%d").date()
            holding_days = (sell_dt - buy_dt).days
            pnl = (net_sell_px - lot.unit_cost_inr) * take
            if holding_days >= 365:
                ltcg_raw += pnl
            else:
                stcg_raw += pnl
            lot.shares -= take
            rem -= take
            if lot.shares <= 1e-12:
                queue.pop(0)

        self.total_stcg_realized_inr += stcg_raw
        self.total_ltcg_realized_inr += ltcg_raw

        # Apply Indian loss set-off & carryforward rules
        stcg_taxable = 0.0
        if stcg_raw < 0.0:
            self.stcl_carry_inr += abs(stcg_raw)
        else:
            used = min(stcg_raw, self.stcl_carry_inr)
            self.stcl_carry_inr -= used
            stcg_taxable = stcg_raw - used

        ltcg_after_loss = 0.0
        if ltcg_raw < 0.0:
            self.ltcl_carry_inr += abs(ltcg_raw)
        else:
            # First offset with LTCL, then any remaining STCL
            used_lt = min(ltcg_raw, self.ltcl_carry_inr)
            self.ltcl_carry_inr -= used_lt
            rem_lt = ltcg_raw - used_lt
            used_st = min(rem_lt, self.stcl_carry_inr)
            self.stcl_carry_inr -= used_st
            ltcg_after_loss = rem_lt - used_st

        # Apply ₹1,25,000 annual LTCG exemption per Indian Financial Year
        ex_used = self.fy_ltcg_exemption_used.get(fy, 0.0)
        ex_avail = max(0.0, INDIA_LTCG_FY_EXEMPTION_INR - ex_used)
        ex_applied = min(ltcg_after_loss, ex_avail)
        self.fy_ltcg_exemption_used[fy] = ex_used + ex_applied
        ltcg_taxable = max(0.0, ltcg_after_loss - ex_applied)

        stcg_tax = stcg_taxable * INDIA_STCG_EFFECTIVE_RATE
        ltcg_tax = ltcg_taxable * INDIA_LTCG_EFFECTIVE_RATE
        self.total_stcg_tax_inr += stcg_tax
        self.total_ltcg_tax_inr += ltcg_tax

        return {
            "realized_pnl_inr": round(stcg_raw + ltcg_raw, 2),
            "stcg_gain_inr": round(stcg_raw, 2),
            "ltcg_gain_inr": round(ltcg_raw, 2),
            "stcg_tax_inr": round(stcg_tax, 2),
            "ltcg_tax_inr": round(ltcg_tax, 2),
            "total_tax_inr": round(stcg_tax + ltcg_tax, 2),
        }

    def clone(self) -> FifoTaxLedger:
        cloned = FifoTaxLedger(
            stcl_carry_inr=self.stcl_carry_inr,
            ltcl_carry_inr=self.ltcl_carry_inr,
            fy_ltcg_exemption_used=dict(self.fy_ltcg_exemption_used),
            total_stcg_realized_inr=self.total_stcg_realized_inr,
            total_ltcg_realized_inr=self.total_ltcg_realized_inr,
            total_stcg_tax_inr=self.total_stcg_tax_inr,
            total_ltcg_tax_inr=self.total_ltcg_tax_inr,
            total_fees_inr=self.total_fees_inr,
        )
        for sym, q in self.lots.items():
            cloned.lots[sym] = [TaxLot(l.symbol, l.buy_date, l.shares, l.unit_cost_inr, l.sleeve_id) for l in q]
        return cloned

    def estimate_liquidation(
        self,
        as_of_date: str,
        latest_prices: dict[str, float],
        adv_map: dict[str, float] | None = None,
        vol_map: dict[str, float] | None = None,
        half_spread_bps: float = 3.0,
        include_dp_charge: bool = True,
    ) -> dict[str, float]:
        """Non-destructively compute terminal liquidation sell costs & accrued STCG/LTCG liability."""
        sim = self.clone()
        gross_mv = 0.0
        liq_costs = 0.0
        liq_stcg_tax = 0.0
        liq_ltcg_tax = 0.0

        for sym, q in list(sim.lots.items()):
            tot_sh = sum(l.shares for l in q)
            if tot_sh <= 1e-12:
                continue
            px = float(latest_prices.get(sym, q[-1].unit_cost_inr))
            gross = tot_sh * px
            gross_mv += gross
            c = compute_indian_delivery_cost(
                "SELL",
                gross,
                adv_20d_inr=adv_map.get(sym) if adv_map else None,
                daily_vol=vol_map.get(sym, 0.02) if vol_map else 0.02,
                half_spread_bps=half_spread_bps,
                include_dp_charge=include_dp_charge,
            )
            liq_costs += c["total_cost_inr"]
            tx = sim.sell(sym, as_of_date, tot_sh, px, sell_cost_inr=c["total_cost_inr"])
            liq_stcg_tax += tx["stcg_tax_inr"]
            liq_ltcg_tax += tx["ltcg_tax_inr"]

        return {
            "gross_equity_value_inr": round(gross_mv, 2),
            "liquidation_cost_inr": round(liq_costs, 2),
            "unrealized_stcg_tax_inr": round(liq_stcg_tax, 2),
            "unrealized_ltcg_tax_inr": round(liq_ltcg_tax, 2),
            "unrealized_total_tax_inr": round(liq_stcg_tax + liq_ltcg_tax, 2),
            "net_after_liquidation_inr": round(gross_mv - liq_costs - liq_stcg_tax - liq_ltcg_tax, 2),
        }

    def export_lots(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sym, q in self.lots.items():
            for l in q:
                if l.shares > 1e-12:
                    out.append(
                        {
                            "symbol": sym,
                            "buy_date": l.buy_date,
                            "shares_remaining": round(l.shares, 6),
                            "unit_cost_inr": round(l.unit_cost_inr, 4),
                            "sleeve_id": int(l.sleeve_id),
                        }
                    )
        return out

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        stcl_carry_inr: float = 0.0,
        ltcl_carry_inr: float = 0.0,
    ) -> FifoTaxLedger:
        ledger = cls(stcl_carry_inr=float(stcl_carry_inr), ltcl_carry_inr=float(ltcl_carry_inr))
        for r in sorted(rows, key=lambda x: (str(x["buy_date"]), int(x.get("id", 0)))):
            sh = float(r.get("shares_remaining", r.get("shares", 0.0)))
            if sh > 1e-12:
                ledger.lots.setdefault(r["symbol"], []).append(
                    TaxLot(
                        symbol=r["symbol"],
                        buy_date=str(r["buy_date"])[:10],
                        shares=sh,
                        unit_cost_inr=float(r.get("unit_cost_inr", r.get("avg_cost", 0.0))),
                        sleeve_id=int(r.get("sleeve_id", 1)),
                    )
                )
        return ledger
