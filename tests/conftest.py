"""Shared pytest fixtures and marker registration."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test needs a live QuantX Dataloader / Massive call"
    )


@pytest.fixture(scope="session")
def company_bars_slice(request: pytest.FixtureRequest):
    """A reduced slice of company bars (2 tickers, ~1 year) through the shared
    data/data_loader.py path, cached across test runs via pytest's cache."""
    if not os.environ.get("DATALOADER_API_KEY") or not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")

    from data.data_loader import load_company_bars

    cache_dir = request.config.cache.mkdir("market_data_slice")
    bars_path = cache_dir / "bars.csv"
    if bars_path.exists():
        import pandas as pd
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        return bars

    bars, _meta = load_company_bars(["MSFT", "IBM"], "2019-01-01", "2020-01-01", use_cache=True)
    bars.to_csv(bars_path, index=False)
    return bars
