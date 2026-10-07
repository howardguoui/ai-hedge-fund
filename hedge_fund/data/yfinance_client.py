"""Free market data from Yahoo Finance (yfinance): the local-mode DataClient.

Selected with ``HEDGE_FUND_DATA=yfinance`` (see ``hedge_fund/data/factory.py``). No API key.

What it promises compared with the Financial Datasets client:

- Prices, company facts, news and earnings surprises: close equivalents.
- Fundamentals: Yahoo publishes roughly the last five quarters and four fiscal years of
  statements. Trailing-twelve-month rows are summed from four consecutive quarters where
  they exist, then fiscal years fill in older periods, so a snapshot usually holds 4-6
  periods instead of up to 20.
- Point in time: Yahoo has no SEC filing dates. ``filing_date`` is the company's earnings
  report date for that period when Yahoo lists one, otherwise the period end plus 45 days
  (quarter) or 75 days (fiscal year), about the SEC deadlines. Restated figures replace the
  originals. Good enough for live runs; long backtests can see slightly revised numbers.
- Failures: Yahoo often answers an outage with an empty table. An empty price history for
  a range of five or more days raises ``YFinanceError`` instead of returning ``[]``, so a
  backtest cannot mistake an outage for "no data" (the DataClient contract).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Callable

import pandas as pd

from hedge_fund.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsData,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)

QUARTER_LAG_DAYS = 45
ANNUAL_LAG_DAYS = 75
REPORT_WINDOW_DAYS = 100  # an earnings report this soon after a period end belongs to it
MIN_EMPTY_RANGE_DAYS = 5  # shorter ranges can be all weekend/holiday


class YFinanceError(Exception):
    """Yahoo failed or returned nothing where data must exist. Fail loud, like FDClientError."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _day(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _row(df: pd.DataFrame | None, *names: str) -> pd.Series | None:
    """The first statement line that exists, e.g. 'Total Revenue' or 'Operating Revenue'."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def _at(series: pd.Series | None, col) -> float | None:
    if series is None:
        return None
    return _num(series.get(col))


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _cols(df: pd.DataFrame | None) -> list:
    """Statement columns (period ends), newest first."""
    if df is None or df.empty:
        return []
    return sorted(df.columns, key=lambda c: pd.Timestamp(c), reverse=True)


REVENUE = ("Total Revenue", "Operating Revenue")
GROSS = ("Gross Profit",)
OPERATING = ("Operating Income", "Total Operating Income As Reported")
NET = ("Net Income", "Net Income Common Stockholders")
EPS = ("Diluted EPS", "Basic EPS")
FCF = ("Free Cash Flow",)
EQUITY = ("Stockholders Equity", "Common Stock Equity")
DEBT = ("Total Debt",)
ASSETS = ("Total Assets",)
CUR_ASSETS = ("Current Assets",)
CUR_LIAB = ("Current Liabilities",)
SHARES = ("Ordinary Shares Number", "Share Issued")


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

class YFinanceClient:
    """DataClient backed by yfinance. ``ticker_factory`` is injectable for tests."""

    cache_namespace = "yfinance"  # keeps its disk cache apart from Financial Datasets'

    def __init__(self, ticker_factory: Callable[[str], Any] | None = None) -> None:
        if ticker_factory is None:
            import yfinance as yf

            ticker_factory = yf.Ticker
        self._factory = ticker_factory
        self._tickers: dict[str, Any] = {}
        self._history: dict[str, pd.DataFrame] = {}

    def __enter__(self) -> YFinanceClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._tickers.clear()
        self._history.clear()

    def _t(self, ticker: str):
        if ticker not in self._tickers:
            self._tickers[ticker] = self._factory(ticker)
        return self._tickers[ticker]

    def _get(self, ticker: str, attr: str):
        try:
            value = getattr(self._t(ticker), attr)
            return value() if callable(value) else value
        except Exception as exc:  # network, parsing, rate limit
            raise YFinanceError(f"Yahoo {attr} for {ticker} failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_prices(self, ticker, start_date, end_date, interval="day", interval_multiplier=1) -> list[Price]:
        if interval != "day" or interval_multiplier != 1:
            raise YFinanceError("The yfinance client serves daily bars only.")
        end_exclusive = (date.fromisoformat(end_date) + timedelta(days=1)).isoformat()
        try:
            df = self._t(ticker).history(start=start_date, end=end_exclusive, interval="1d",
                                         auto_adjust=False, actions=False)
        except Exception as exc:
            raise YFinanceError(f"Yahoo prices for {ticker} failed: {exc}") from exc
        bars = []
        if df is not None and not df.empty:
            for idx, r in df.iterrows():
                close = _num(r.get("Close"))
                if close is None:
                    continue
                bars.append(Price(open=_num(r.get("Open")) or close, close=close,
                                  high=_num(r.get("High")) or close, low=_num(r.get("Low")) or close,
                                  volume=int(_num(r.get("Volume")) or 0), time=_day(idx).isoformat()))
        span = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
        if not bars and span >= MIN_EMPTY_RANGE_DAYS:
            raise YFinanceError(f"Yahoo returned no prices for {ticker} {start_date}..{end_date}")
        return bars

    def _closes(self, ticker: str) -> pd.Series:
        """Five years of daily closes, fetched once per client, for valuing filed periods."""
        if ticker not in self._history:
            try:
                df = self._t(ticker).history(period="5y", interval="1d", auto_adjust=False, actions=False)
            except Exception as exc:
                raise YFinanceError(f"Yahoo price history for {ticker} failed: {exc}") from exc
            self._history[ticker] = df if df is not None else pd.DataFrame()
        df = self._history[ticker]
        if df.empty or "Close" not in df:
            return pd.Series(dtype=float)
        s = df["Close"].dropna()
        s.index = [_day(i) for i in s.index]
        return s

    def _close_on(self, ticker: str, day: date) -> float | None:
        s = self._closes(ticker)
        s = s[[d <= day for d in s.index]]
        return float(s.iloc[-1]) if len(s) else None

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------

    def _report_dates(self, ticker: str) -> list[date]:
        try:
            df = self._t(ticker).get_earnings_dates(limit=40)
        except Exception:
            return []  # optional input: without it filing dates fall back to the statutory lag
        if df is None or df.empty:
            return []
        return sorted({_day(i) for i in df.index})

    def _filing_date(self, period_end: date, reports: list[date], annual: bool) -> date:
        for d in reports:
            if period_end < d <= period_end + timedelta(days=REPORT_WINDOW_DAYS):
                return d
        return period_end + timedelta(days=ANNUAL_LAG_DAYS if annual else QUARTER_LAG_DAYS)

    def _periods(self, ticker: str) -> list[dict]:
        """Every TTM / fiscal-year period Yahoo has, newest first, before valuation."""
        qi, qb, qc = (self._get(ticker, a) for a in ("quarterly_income_stmt", "quarterly_balance_sheet",
                                                     "quarterly_cashflow"))
        ai, ab, ac = (self._get(ticker, a) for a in ("income_stmt", "balance_sheet", "cashflow"))
        reports = self._report_dates(ticker)

        def flows(inc, cf, cols) -> dict:
            def total(names, df):
                series = _row(df, *names)
                vals = [_at(series, c) for c in cols]
                return None if any(v is None for v in vals) else sum(vals)
            return {"revenue": total(REVENUE, inc), "gross": total(GROSS, inc), "operating": total(OPERATING, inc),
                    "net": total(NET, inc), "eps": total(EPS, inc), "fcf": total(FCF, cf)}

        def balance(bal, col) -> dict:
            cols = _cols(bal)
            if not cols:
                return {}
            near = min(cols, key=lambda c: abs((_day(c) - _day(col)).days))
            if abs((_day(near) - _day(col)).days) > 45:
                return {}
            return {k: _at(_row(bal, *names), near) for k, names in
                    (("equity", EQUITY), ("debt", DEBT), ("assets", ASSETS), ("cur_assets", CUR_ASSETS),
                     ("cur_liab", CUR_LIAB), ("shares", SHARES))}

        out = []
        qcols = _cols(qi)
        for i in range(len(qcols) - 3):
            window = qcols[i:i + 4]
            if not 250 <= (_day(window[0]) - _day(window[3])).days <= 300:
                continue  # a missing quarter: not four in a row
            end = _day(window[0])
            out.append({"end": end, "annual": False, **flows(qi, qc, window), **balance(qb, window[0]),
                        "filed": self._filing_date(end, reports, annual=False)})
        oldest = min((p["end"] for p in out), default=None)
        for col in _cols(ai):
            end = _day(col)
            if oldest is not None and end >= oldest - timedelta(days=30):
                continue  # already covered by a quarterly TTM row
            out.append({"end": end, "annual": True, **flows(ai, ac, [col]), **balance(ab, col),
                        "filed": self._filing_date(end, reports, annual=True)})
        out.sort(key=lambda p: p["end"], reverse=True)
        return out

    def _metrics(self, ticker: str, p: dict, prior: dict | None, currency: str | None) -> FinancialMetrics:
        price = self._close_on(ticker, p["filed"])
        shares, eps = p.get("shares"), p.get("eps")
        if eps is None:
            eps = _ratio(p.get("net"), shares)
        market_cap = price * shares if price is not None and shares else None
        bvps = _ratio(p.get("equity"), shares)
        fcfps = _ratio(p.get("fcf"), shares)
        return FinancialMetrics(
            ticker=ticker, report_period=p["end"].isoformat(), period="ttm", currency=currency,
            filing_date=p["filed"].isoformat(),
            market_cap=market_cap,
            price_to_earnings_ratio=_ratio(price, eps),
            price_to_book_ratio=_ratio(price, bvps),
            price_to_sales_ratio=_ratio(market_cap, p.get("revenue")),
            free_cash_flow_yield=_ratio(p.get("fcf"), market_cap),
            gross_margin=_ratio(p.get("gross"), p.get("revenue")),
            operating_margin=_ratio(p.get("operating"), p.get("revenue")),
            net_margin=_ratio(p.get("net"), p.get("revenue")),
            return_on_equity=_ratio(p.get("net"), p.get("equity")),
            return_on_assets=_ratio(p.get("net"), p.get("assets")),
            debt_to_equity=_ratio(p.get("debt"), p.get("equity")),
            current_ratio=_ratio(p.get("cur_assets"), p.get("cur_liab")),
            revenue_growth=(_ratio(p.get("revenue"), prior.get("revenue")) - 1
                            if prior and _ratio(p.get("revenue"), prior.get("revenue")) is not None else None),
            earnings_per_share_growth=(_ratio(eps, prior.get("eps")) - 1
                                       if prior and prior.get("eps") and eps is not None and prior["eps"] > 0
                                       else None),
            earnings_per_share=eps,
            book_value_per_share=bvps,
            free_cash_flow_per_share=fcfps,
        )

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10) -> list[FinancialMetrics]:
        """TTM (or fiscal-year) rows whose filing date is on or before end_date, newest first."""
        if period not in ("ttm", "annual"):
            raise YFinanceError(f"The yfinance client builds 'ttm' and 'annual' rows, not {period!r}.")
        periods = self._periods(ticker)
        if period == "annual":
            periods = [p for p in periods if p["annual"]]
        cutoff = date.fromisoformat(end_date[:10])
        currency = (self._info(ticker) or {}).get("financialCurrency")
        rows = []
        for p in periods:
            if p["filed"] > cutoff:
                continue
            prior = next((q for q in periods if 330 <= (p["end"] - q["end"]).days <= 400), None)
            rows.append(self._metrics(ticker, p, prior, currency))
            if len(rows) >= limit:
                break
        return rows

    # ------------------------------------------------------------------
    # Company facts, market cap
    # ------------------------------------------------------------------

    def _info(self, ticker: str) -> dict:
        try:
            info = self._t(ticker).info
        except Exception:
            return {}
        return info if isinstance(info, dict) else {}

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        info = self._info(ticker)
        if not info or not (info.get("longName") or info.get("shortName")):
            return None
        location = ", ".join(x for x in (info.get("city"), info.get("country")) if x) or None
        return CompanyFacts(ticker=ticker, name=info.get("longName") or info.get("shortName"),
                            sector=info.get("sector"), industry=info.get("industry"),
                            exchange=info.get("exchange"), location=location)

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        info = self._info(ticker)
        shares = _num(info.get("sharesOutstanding"))
        price = self._close_on(ticker, date.fromisoformat(end_date[:10]))
        if shares and price:
            return shares * price
        return _num(info.get("marketCap"))

    # ------------------------------------------------------------------
    # Earnings
    # ------------------------------------------------------------------

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        """Earnings announcements with EPS vs estimate, newest first. Recorded as 8-K
        rows: an earnings press release is filed on a Form 8-K the same day."""
        try:
            df = self._t(ticker).get_earnings_dates(limit=limit + 8)
        except Exception as exc:
            raise YFinanceError(f"Yahoo earnings dates for {ticker} failed: {exc}") from exc
        if df is None or df.empty:
            return []
        try:
            quarter_ends = [_day(c) for c in _cols(self._get(ticker, "quarterly_income_stmt"))]
        except YFinanceError:
            quarter_ends = []
        records = []
        for idx, r in df.sort_index(ascending=False).iterrows():
            reported, estimate = _num(r.get("Reported EPS")), _num(r.get("EPS Estimate"))
            if reported is None:
                continue  # a scheduled date, not yet reported
            filed = _day(idx)
            period = next((q for q in quarter_ends if q < filed <= q + timedelta(days=REPORT_WINDOW_DAYS)),
                          _calendar_quarter_end_before(filed))
            surprise = None
            if estimate is not None:
                surprise = "MEET" if abs(reported - estimate) < 0.005 else ("BEAT" if reported > estimate else "MISS")
            records.append(EarningsRecord(
                ticker=ticker, report_period=period.isoformat(), source_type="8-K",
                filing_date=filed.isoformat(),
                quarterly=EarningsData(earnings_per_share=reported, estimated_earnings_per_share=estimate,
                                       eps_surprise=surprise)))
            if len(records) >= limit:
                break
        return records

    def get_earnings(self, ticker: str) -> Earnings | None:
        history = self.get_earnings_history(ticker, limit=1)
        if not history:
            return None
        latest = history[0]
        return Earnings(ticker=ticker, report_period=latest.report_period, quarterly=latest.quarterly)

    # ------------------------------------------------------------------
    # News, insider trades
    # ------------------------------------------------------------------

    def get_news(self, ticker, end_date, start_date=None, limit=1000) -> list[CompanyNews]:
        try:
            items = self._t(ticker).news or []
        except Exception as exc:
            raise YFinanceError(f"Yahoo news for {ticker} failed: {exc}") from exc
        out = []
        for item in items:
            c = item.get("content", item) if isinstance(item, dict) else {}
            title = c.get("title")
            when = c.get("pubDate") or c.get("displayTime")
            if when is None and c.get("providerPublishTime"):
                when = datetime.fromtimestamp(c["providerPublishTime"]).isoformat()
            day = _day(when).isoformat() if when else None
            if not title or (day and (day > end_date[:10] or (start_date and day < start_date[:10]))):
                continue
            provider = c.get("provider")
            source = provider.get("displayName") if isinstance(provider, dict) else c.get("publisher")
            url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
            out.append(CompanyNews(ticker=ticker, title=title, source=source or "Yahoo Finance", date=day, url=url))
            if len(out) >= limit:
                break
        return out

    def get_insider_trades(self, ticker, end_date, start_date=None, limit=1000) -> list[InsiderTrade]:
        try:
            df = self._t(ticker).insider_transactions
        except Exception as exc:
            raise YFinanceError(f"Yahoo insider trades for {ticker} failed: {exc}") from exc
        if df is None or getattr(df, "empty", True):
            return []
        out = []
        for _, r in df.iterrows():
            when = r.get("Start Date")
            if when is None or (isinstance(when, float) and math.isnan(when)):
                continue
            day = _day(when).isoformat()
            if day > end_date[:10] or (start_date and day < start_date[:10]):
                continue
            title = str(r.get("Position") or "") or None
            out.append(InsiderTrade(
                ticker=ticker, name=str(r.get("Insider") or "unknown"), filing_date=day, transaction_date=day,
                title=title, is_board_director=bool(title and "director" in title.lower()),
                transaction_type=str(r.get("Transaction") or r.get("Text") or "") or None,
                transaction_shares=_num(r.get("Shares")), transaction_value=_num(r.get("Value"))))
            if len(out) >= limit:
                break
        return out


def _calendar_quarter_end_before(day: date) -> date:
    """The last calendar quarter end strictly before day (fallback fiscal period)."""
    q_month = ((day.month - 1) // 3) * 3  # month that ends the previous quarter (0 = December last year)
    if q_month == 0:
        return date(day.year - 1, 12, 31)
    end = date(day.year, q_month + 1, 1) - timedelta(days=1)
    return end if end < day else _calendar_quarter_end_before(end)
