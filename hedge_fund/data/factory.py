"""Which market-data source a run uses.

    HEDGE_FUND_DATA=fd        Financial Datasets (default; needs FINANCIAL_DATASETS_API_KEY)
    HEDGE_FUND_DATA=yfinance  Yahoo Finance via yfinance: free, no key (see yfinance_client.py)
"""

from __future__ import annotations

import os

SOURCES = ("fd", "yfinance")


def data_source() -> str:
    """The configured source name. Raises on a typo rather than silently picking one."""
    source = (os.environ.get("HEDGE_FUND_DATA") or "fd").strip().lower()
    if source not in SOURCES:
        raise ValueError(f"HEDGE_FUND_DATA={source!r}: use one of {', '.join(SOURCES)}.")
    return source


def needs_data_key() -> bool:
    """True when the run needs FINANCIAL_DATASETS_API_KEY."""
    return data_source() == "fd"


def make_raw_client():
    """A fresh, uncached client for the configured source (wrap it in CachedDataClient)."""
    if data_source() == "yfinance":
        from hedge_fund.data.yfinance_client import YFinanceClient

        return YFinanceClient()
    from hedge_fund.data.client import FDClient

    return FDClient()
