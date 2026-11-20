"""SEC EDGAR Form 4 insider filing metadata lookup (zero-cost public SEC JSON API)."""

from __future__ import annotations

import json
import logging
import urllib.request
from functools import lru_cache
from typing import Any

from .config import SEC_USER_AGENT

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_sec_cik_map() -> dict[str, str]:
    """Fetch SEC ticker -> 10-digit zero-padded CIK map (cached in memory)."""
    url = "https://www.sec.gov/files/company_tickers.json"
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        mapping: dict[str, str] = {}
        for item in data.values():
            ticker = str(item.get("ticker", "")).upper().strip()
            cik = str(item.get("cik_str", "")).strip()
            if ticker and cik.isdigit():
                mapping[ticker] = cik.zfill(10)
        return mapping
    except Exception as exc:
        logger.warning("SEC CIK map fetch unavailable: %s", exc)
        return {}


def fetch_sec_form4_summary(symbol: str, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch recent Form 4 insider filing timestamps from SEC EDGAR submissions API."""
    cik = load_sec_cik_map().get(symbol.upper())
    if not cik:
        return []
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        recent = payload.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        out: list[dict[str, Any]] = []
        for form, fdate, acc in zip(forms, dates, accessions):
            if form == "4":
                acc_clean = str(acc).replace("-", "")
                out.append(
                    {
                        "form": "Form 4",
                        "filing_date": str(fdate),
                        "accession": str(acc),
                        "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/",
                    }
                )
                if len(out) >= limit:
                    break
        return out
    except Exception as exc:
        logger.debug("SEC Form 4 lookup failed for %s: %s", symbol, exc)
        return []
