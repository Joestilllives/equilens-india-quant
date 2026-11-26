"""Configuration for EquiLens India (NSE NIFTY 200 / Liquid F&O Universe) quant platform."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(os.getenv("EQUILENS_DB_PATH", str(DATA_DIR / "equilens_india.db")))
FRONTEND_DIR = BASE_DIR / "frontend"

MARKET_REGION = "IN"
CURRENCY_SYMBOL = "₹"
BENCHMARK_SYMBOL = "^NSEI"
BENCHMARK_LABEL = "NIFTY 50"
TRI_BENCHMARK_SYMBOL = os.getenv("EQUILENS_TRI_BENCHMARK_SYMBOL", "NIFTYBEES.NS")
GOLDBEES_SYMBOL = os.getenv("EQUILENS_GOLDBEES_SYMBOL", "GOLDBEES.NS")
NON_EQUITY_SYMBOLS: frozenset[str] = frozenset({
    BENCHMARK_SYMBOL,
    TRI_BENCHMARK_SYMBOL,
    GOLDBEES_SYMBOL,
})

# Live Security & Data Freshness Safety Controls
_raw_cors = os.getenv("EQUILENS_CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
CORS_ORIGINS: list[str] = [o.strip() for o in _raw_cors.split(",") if o.strip()]
API_TOKEN_ENV = os.getenv("EQUILENS_API_TOKEN", "").strip()
MAX_STALE_TRADING_DAYS = int(os.getenv("EQUILENS_MAX_STALE_TRADING_DAYS", "2"))
DATA_FRESHNESS_MIN_COVERAGE = float(os.getenv("EQUILENS_MIN_COVERAGE", "0.85"))
VOL_INDEX_SYMBOL = "^INDIAVIX"
FX_MACRO_SYMBOL = "INR=X"
BRENT_MACRO_SYMBOL = "BZ=F"
GOLD_MACRO_SYMBOL = "GC=F"
SP500_MACRO_SYMBOL = "^GSPC"
US10Y_MACRO_SYMBOL = "^TNX"
DXY_MACRO_SYMBOL = "DX-Y.NYB"
MACRO_SYMBOLS: list[str] = [
    VOL_INDEX_SYMBOL,
    FX_MACRO_SYMBOL,
    BRENT_MACRO_SYMBOL,
    GOLD_MACRO_SYMBOL,
    SP500_MACRO_SYMBOL,
    US10Y_MACRO_SYMBOL,
    DXY_MACRO_SYMBOL,
]

# ~120 liquid NSE NIFTY 50 + NIFTY Next 50 + Midcap 100 F&O leaders across all Indian sectors + ^NSEI
DEFAULT_UNIVERSE: list[str] = [
    "^NSEI",
    # Banking & Financial Services (Private, PSU, NBFC, Insurance, Capital Markets)
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "JIOFIN.NS", "SHRIRAMFIN.NS", "CHOLAFIN.NS",
    "MUTHOOTFIN.NS", "PFC.NS", "RECLTD.NS", "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS",
    "SBICARD.NS", "HDFCAMC.NS", "BSE.NS", "CDSL.NS", "MCX.NS",
    # IT Services & Digital Tech
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "KPITTECH.NS", "OFSS.NS", "NAUKRI.NS",
    # Energy, Oil, Gas & Power Utilities
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    "BPCL.NS", "IOC.NS", "GAIL.NS", "TATAPOWER.NS", "ADANIPOWER.NS", "TORNTPOWER.NS", "NHPC.NS",
    # Automobiles, EV & Ancillaries
    "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "TVSMOTOR.NS", "BHARATFORG.NS", "ASHOKLEY.NS", "BALKRISIND.NS", "MOTHERSON.NS", "BOSCHLTD.NS",
    # Pharma, Healthcare & Diagnostics
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
    "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "ALKEM.NS", "MAXHEALTH.NS", "ZYDUSLIFE.NS", "FORTIS.NS",
    # FMCG, Retail & Consumer Durables
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
    "TITAN.NS", "TRENT.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "DMART.NS",
    "VBL.NS", "COLPAL.NS", "MARICO.NS", "DABUR.NS", "GODREJCP.NS", "HAVELLS.NS", "DIXON.NS", "VOLTAS.NS", "PAGEIND.NS",
    # Capital Goods, Defence, Infrastructure, Telecom, Metals & Real Estate
    "LT.NS", "BHARTIARTL.NS", "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "SHREECEM.NS",
    "ADANIPORTS.NS", "ADANIENT.NS", "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "NMDC.NS",
    "BEL.NS", "HAL.NS", "MAZDOCK.NS", "ABB.NS", "SIEMENS.NS", "CUMMINSIND.NS", "CGPOWER.NS",
    "POLYCAB.NS", "SOLARINDS.NS", "BHEL.NS", "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS",
    "INDIGO.NS", "INDHOTEL.NS", "SRF.NS", "PIIND.NS",
]

DEFAULT_STARRED = {
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "BEL.NS", "TRENT.NS", "^NSEI",
}

HISTORY_PERIOD = "5y"
ML_HORIZON_DAYS = 40  # ~2-month horizon reduces Indian short-term mean-reversion noise & STT churn

# Symmetric 40D Triple-Barrier payoff labeling (matches 40D Indian swing horizon; eliminates asymmetric stop whipsaw)
TB_HORIZON_DAYS = 40
TB_UP_ATR_MULT = 2.0
TB_DOWN_ATR_MULT = 2.0

# Realistic Indian equity delivery/swing round-trip friction (STT + brokerage + GST + stamp + slippage)
DEFAULT_COST_BPS = 15.0

SEC_USER_AGENT = os.getenv(
    "EQUILENS_SEC_USER_AGENT",
    "EquiLensIndiaQuantResearch/3.0 (research@equilens.local)",
)
