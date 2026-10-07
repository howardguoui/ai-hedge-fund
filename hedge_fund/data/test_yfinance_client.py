"""YFinanceClient against a fake yfinance.Ticker: no network."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from hedge_fund.data import CachedDataClient
from hedge_fund.data.factory import data_source, make_raw_client, needs_data_key
from hedge_fund.data.yfinance_client import YFinanceClient, YFinanceError, _calendar_quarter_end_before
from hedge_fund.features.snapshot import build_snapshot

# Six calendar quarters, newest first: TTM rows exist for the three newest quarter ends.
Q = [pd.Timestamp(d) for d in ("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31")]
FY = [pd.Timestamp(d) for d in ("2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31")]


def _frame(rows: dict, cols) -> pd.DataFrame:
    return pd.DataFrame(rows, index=cols).T


class FakeTicker:
    def __init__(self, symbol="TEST", empty_prices=False):
        self.symbol = symbol
        self._empty = empty_prices
        quarter = {"Total Revenue": [130.0, 100.0, 100.0, 100.0, 100.0, 100.0], "Gross Profit": [60.0] * 6, "Operating Income": [30.0] * 6,
                   "Net Income": [20.0] * 6, "Diluted EPS": [0.2] * 6}
        self.quarterly_income_stmt = _frame(quarter, Q)
        self.quarterly_balance_sheet = _frame({"Stockholders Equity": [800.0] * 6, "Total Debt": [400.0] * 6,
                                               "Total Assets": [2000.0] * 6, "Current Assets": [300.0] * 6,
                                               "Current Liabilities": [150.0] * 6,
                                               "Ordinary Shares Number": [100.0] * 6}, Q)
        self.quarterly_cashflow = _frame({"Free Cash Flow": [10.0] * 6}, Q)
        self.income_stmt = _frame({"Total Revenue": [400.0, 320.0, 250.0, 200.0, float("nan")],
                                   "Gross Profit": [240.0, 190.0, 150.0, 120.0, float("nan")],
                                   "Operating Income": [120.0, 90.0, 70.0, 50.0, float("nan")],
                                   "Net Income": [80.0, 60.0, 45.0, 30.0, float("nan")],
                                   "Diluted EPS": [0.8, 0.6, 0.45, 0.3, float("nan")]}, FY + [pd.Timestamp("2021-12-31")])
        self.balance_sheet = _frame({"Stockholders Equity": [800.0, 700.0, 600.0, 500.0],
                                     "Total Debt": [400.0] * 4, "Ordinary Shares Number": [100.0] * 4}, FY)
        self.cashflow = _frame({"Free Cash Flow": [40.0, 30.0, 20.0, 10.0]}, FY)
        self.info = {"longName": "Test Corp", "sector": "Technology", "industry": "Software",
                     "exchange": "NMS", "city": "Austin", "country": "United States",
                     "sharesOutstanding": 100.0, "financialCurrency": "USD"}
        self.news = [
            {"content": {"title": "Test Corp beats", "pubDate": "2026-08-01T12:00:00Z",
                         "provider": {"displayName": "Reuters"}, "canonicalUrl": {"url": "https://x.test/a"}}},
            {"title": "Old style item", "publisher": "AP", "link": "https://x.test/b",
             "providerPublishTime": 1754000000},
        ]
        self.insider_transactions = pd.DataFrame([
            {"Insider": "Jane Doe", "Position": "Director", "Transaction": "Sale", "Shares": 1000, "Value": 5e4,
             "Start Date": pd.Timestamp("2026-05-01")},
        ])

    def history(self, start=None, end=None, period=None, **kwargs):
        if self._empty:
            return pd.DataFrame()
        idx = pd.bdate_range("2021-01-01", "2026-10-06")
        df = pd.DataFrame({"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.0, "Volume": 1000}, index=idx)
        if start:
            df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]
        return df

    def get_earnings_dates(self, limit=12):
        idx = pd.DatetimeIndex(["2026-10-28", "2026-07-29", "2026-04-29", "2026-01-28", "2025-10-29"], tz="America/New_York")
        return pd.DataFrame({"EPS Estimate": [0.25, 0.18, 0.2, 0.2, 0.21],
                             "Reported EPS": [float("nan"), 0.2, 0.2, 0.25, 0.19]}, index=idx)


def client(**kw) -> YFinanceClient:
    return YFinanceClient(ticker_factory=lambda s: FakeTicker(s, **kw))


def test_prices_are_daily_bars_in_range():
    bars = client().get_prices("TEST", "2026-09-28", "2026-10-02")
    assert [b.time for b in bars] == ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01", "2026-10-02"]
    assert bars[0].close == 10.0 and bars[0].volume == 1000


def test_empty_prices_raise_for_a_real_range_but_not_a_weekend():
    with pytest.raises(YFinanceError):
        client(empty_prices=True).get_prices("TEST", "2026-09-01", "2026-09-30")
    assert client(empty_prices=True).get_prices("TEST", "2026-10-03", "2026-10-04") == []


def test_ttm_rows_sum_four_quarters_then_fill_with_fiscal_years():
    rows = client().get_financial_metrics("TEST", "2026-10-06", limit=10)
    periods = [r.report_period for r in rows]
    # three quarterly TTM rows, then fiscal years older than the oldest TTM end (2025-12-31 is covered)
    assert periods == ["2026-06-30", "2026-03-31", "2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"]
    latest = rows[0]
    assert latest.period == "ttm"
    assert latest.gross_margin == pytest.approx(240 / 430) and latest.net_margin == pytest.approx(80 / 430)
    assert latest.earnings_per_share == pytest.approx(0.8)
    # no TTM a year earlier: latest quarter (130) vs the same quarter a year before (100)
    assert latest.revenue_growth == pytest.approx(0.3)
    assert rows[1].revenue_growth == pytest.approx(0.0)    # Q1 2026 vs Q1 2025: 100 vs 100
    assert rows[2].revenue_growth == pytest.approx(0.25)   # TTM 2025 (400) vs fiscal 2024 (320)
    assert latest.return_on_equity == pytest.approx(80 / 800)
    assert latest.debt_to_equity == pytest.approx(0.5) and latest.current_ratio == pytest.approx(2.0)
    assert latest.book_value_per_share == pytest.approx(8.0)
    assert latest.market_cap == pytest.approx(1000.0) and latest.price_to_earnings_ratio == pytest.approx(12.5)
    assert "2021-12-31" not in periods  # Yahoo's empty padding year is dropped
    # 2026-06-30 TTM revenue 400 vs the 2025-06-30 period (none) -> growth from the nearest year-earlier row
    fy2024 = next(r for r in rows if r.report_period == "2024-12-31")
    assert fy2024.revenue_growth == pytest.approx(320 / 250 - 1)


def test_filing_date_is_the_earnings_report_or_the_statutory_lag():
    rows = {r.report_period: r for r in client().get_financial_metrics("TEST", "2026-10-06", limit=10)}
    assert rows["2026-06-30"].filing_date == "2026-07-29"   # matched to the report date
    assert rows["2023-12-31"].filing_date == "2024-03-15"   # no report listed: FY end + 75 days (leap year)


def test_point_in_time_cutoff_excludes_unfiled_periods():
    rows = client().get_financial_metrics("TEST", "2026-07-15", limit=10)
    assert rows[0].report_period == "2026-03-31"  # Q2 2026 was not reported until 2026-07-29


def test_snapshot_builds_from_yfinance_rows():
    snap = build_snapshot("TEST", "2026-10-06", client())
    assert snap.sector == "Technology" and len(snap.periods) >= 4
    assert "Company: TEST" in snap.render()


def test_earnings_history_classifies_surprises_and_skips_unreported():
    records = client().get_earnings_history("TEST", limit=4)
    assert [r.filing_date for r in records] == ["2026-07-29", "2026-04-29", "2026-01-28", "2025-10-29"]
    assert [r.quarterly.eps_surprise for r in records] == ["BEAT", "MEET", "BEAT", "MISS"]
    assert records[0].report_period == "2026-06-30" and records[0].source_type == "8-K"


def test_facts_news_insiders_market_cap():
    c = client()
    facts = c.get_company_facts("TEST")
    assert facts.name == "Test Corp" and facts.location == "Austin, United States"
    news = c.get_news("TEST", "2026-10-06", start_date="2025-07-01")
    assert [n.source for n in news] == ["Reuters", "AP"] and news[0].url == "https://x.test/a"
    trades = c.get_insider_trades("TEST", "2026-10-06")
    assert trades[0].is_board_director and trades[0].transaction_shares == 1000
    assert c.get_market_cap("TEST", "2026-10-06") == pytest.approx(1000.0)


def test_unsupported_period_raises():
    with pytest.raises(YFinanceError):
        client().get_financial_metrics("TEST", "2026-10-06", period="quarterly")


def test_calendar_quarter_end_before():
    assert _calendar_quarter_end_before(date(2026, 1, 15)) == date(2025, 12, 31)
    assert _calendar_quarter_end_before(date(2026, 4, 28)) == date(2026, 3, 31)
    assert _calendar_quarter_end_before(date(2026, 10, 1)) == date(2026, 9, 30)


def test_cache_keeps_sources_apart(tmp_path):
    class Other:
        def get_company_facts(self, ticker):
            raise AssertionError("must not be called: different source, different cache entry")

    yf = CachedDataClient(client(), cache_dir=tmp_path)
    assert yf.get_company_facts("TEST").name == "Test Corp"
    fd_like = CachedDataClient(Other(), cache_dir=tmp_path)
    with pytest.raises(AssertionError):
        fd_like.get_company_facts("TEST")


def test_factory_reads_hedge_fund_data(monkeypatch):
    monkeypatch.delenv("HEDGE_FUND_DATA", raising=False)
    assert data_source() == "fd" and needs_data_key()
    monkeypatch.setenv("HEDGE_FUND_DATA", "yfinance")
    assert not needs_data_key()
    assert isinstance(make_raw_client(), YFinanceClient)
    monkeypatch.setenv("HEDGE_FUND_DATA", "yahoo")
    with pytest.raises(ValueError):
        data_source()
