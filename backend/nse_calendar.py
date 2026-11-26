"""NSE India Equity Market Trading Calendar (2020–2027) and Trading-Day Utilities.

Used for:
  - Accurate probability-cone future trading dates (skipping NSE holidays, not just weekends)
  - Live data freshness lag checks (in NSE trading days)
  - Scheduled 10-trading-day staggered tranche rebalance cadence
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

# Official NSE Equity segment full-day trading holidays (2020–2027).
# Extensible via EQUILENS_NSE_HOLIDAYS_FILE pointing to a JSON array of YYYY-MM-DD strings.
_BUILTIN_NSE_HOLIDAYS: frozenset[str] = frozenset(
    {
        # 2020
        "2020-02-21", "2020-03-10", "2020-04-02", "2020-04-06", "2020-04-10",
        "2020-04-14", "2020-05-01", "2020-05-25", "2020-10-02", "2020-11-16",
        "2020-11-30", "2020-12-25",
        # 2021
        "2021-01-26", "2021-03-11", "2021-03-29", "2021-04-02", "2021-04-14",
        "2021-04-21", "2021-05-13", "2021-07-21", "2021-08-19", "2021-09-10",
        "2021-10-15", "2021-11-04", "2021-11-05", "2021-11-19",
        # 2022
        "2022-01-26", "2022-03-01", "2022-03-18", "2022-04-14", "2022-04-15",
        "2022-05-03", "2022-08-09", "2022-08-15", "2022-08-31", "2022-10-05",
        "2022-10-24", "2022-10-26", "2022-11-08",
        # 2023
        "2023-01-26", "2023-03-07", "2023-03-30", "2023-04-04", "2023-04-07",
        "2023-04-14", "2023-05-01", "2023-06-29", "2023-08-15", "2023-09-19",
        "2023-10-02", "2023-10-24", "2023-11-14", "2023-11-27", "2023-12-25",
        # 2024
        "2024-01-22", "2024-01-26", "2024-03-08", "2024-03-25", "2024-03-29",
        "2024-04-11", "2024-04-17", "2024-05-01", "2024-05-20", "2024-06-17",
        "2024-07-17", "2024-08-15", "2024-10-02", "2024-11-01", "2024-11-15",
        "2024-11-20", "2024-12-25",
        # 2025
        "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
        "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
        "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
        # 2026
        "2026-01-26", "2026-03-03", "2026-03-20", "2026-03-26", "2026-03-31",
        "2026-04-03", "2026-04-14", "2026-05-01", "2026-05-28", "2026-06-26",
        "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-09", "2026-11-10",
        "2026-11-24", "2026-12-25",
        # 2027
        "2027-01-26", "2027-03-06", "2027-03-22", "2027-03-26", "2027-04-14",
        "2027-04-15", "2027-04-19", "2027-05-01", "2027-05-17", "2027-08-15",
        "2027-09-04", "2027-10-02", "2027-10-09", "2027-10-29", "2027-11-14",
        "2027-12-25",
    }
)


@lru_cache(maxsize=4)
def get_nse_holidays(extra_file: str | None = None) -> frozenset[str]:
    """Return the active set of NSE trading holidays (ISO YYYY-MM-DD)."""
    holidays = set(_BUILTIN_NSE_HOLIDAYS)
    path_str = extra_file if extra_file is not None else os.getenv("EQUILENS_NSE_HOLIDAYS_FILE", "")
    if path_str:
        p = Path(path_str)
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    holidays.update(str(x)[:10] for x in loaded if x)
            except Exception:
                pass
    return frozenset(holidays)


def _to_date(val: date | datetime | str) -> date:
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()


def is_nse_trading_day(d: date | datetime | str) -> bool:
    """Return True iff `d` is a weekday (Mon–Fri) and not an NSE full-day holiday."""
    dt = _to_date(d)
    if dt.weekday() >= 5:
        return False
    return dt.isoformat() not in get_nse_holidays()


def previous_nse_trading_day(d: date | datetime | str, inclusive: bool = True) -> date:
    """Return the latest NSE trading day on or before `d`."""
    cur = _to_date(d)
    if not inclusive:
        cur -= timedelta(days=1)
    while not is_nse_trading_day(cur):
        cur -= timedelta(days=1)
    return cur


def next_nse_trading_dates(last_date_str: str, steps: list[int]) -> list[str]:
    """Return ISO dates at `steps` NSE trading days strictly after `last_date_str`."""
    if not steps:
        return []
    cur = _to_date(last_date_str)
    max_step = max(steps)
    future: list[str] = []
    while len(future) < max_step:
        cur += timedelta(days=1)
        if is_nse_trading_day(cur):
            future.append(cur.isoformat())
    return [future[s - 1] for s in steps]


def add_nse_trading_days(start_date_str: str, n_days: int) -> str:
    """Advance (or rewind if negative) by `n_days` NSE trading days."""
    cur = _to_date(start_date_str)
    if n_days == 0:
        return cur.isoformat()
    step = 1 if n_days > 0 else -1
    remaining = abs(n_days)
    while remaining > 0:
        cur += timedelta(days=step)
        if is_nse_trading_day(cur):
            remaining -= 1
    return cur.isoformat()


def nse_trading_days_between(d1: date | datetime | str, d2: date | datetime | str) -> int:
    """Signed count of NSE trading days strictly after `d1` up to and including `d2`."""
    a = _to_date(d1)
    b = _to_date(d2)
    if a == b:
        return 0
    sign = 1 if b > a else -1
    lo, hi = (a, b) if b > a else (b, a)
    cur = lo + timedelta(days=1)
    cnt = 0
    while cur <= hi:
        if is_nse_trading_day(cur):
            cnt += 1
        cur += timedelta(days=1)
    return sign * cnt
