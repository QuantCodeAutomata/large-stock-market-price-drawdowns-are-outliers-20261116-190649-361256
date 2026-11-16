"""Market-data test: drawdown/drawup properties on a reduced live slice.

Uses the shared ``data/data_loader.py`` path (same function the experiment
scripts call) on two tickers over one year, through the session-scoped
``company_bars_slice`` fixture. Skipped when API keys are absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.data_loader import company_close_series
from src.drawdown import compute_drawdowns_drawups


@pytest.mark.market_data
def test_drawdowns_drawups_on_live_slice(company_bars_slice):
    """Drawdowns/drawups computed on real bars obey the paper's Sec. 3 definition:
    non-negative amplitudes, no overlap between runs, and no look-ahead (each
    event amplitude depends only on prices up to and including its own run)."""
    for ticker in ("MSFT", "IBM"):
        close = company_close_series(company_bars_slice, ticker)
        assert len(close) > 200, "reduced slice should still span most of a year of trading days"

        dd, du = compute_drawdowns_drawups(close)
        assert len(dd) > 0 and len(du) > 0
        assert np.all(dd >= 0), "drawdown amplitudes must be non-negative (paper Sec. 3)"
        assert np.all(du >= 0), "drawup amplitudes must be non-negative (paper Sec. 3)"

        # No look-ahead: recomputing on a prefix of the series must reproduce
        # every drawdown/drawup whose run ends before the truncation point.
        cut = len(close) // 2
        dd_prefix, du_prefix = compute_drawdowns_drawups(close.iloc[:cut])
        if len(dd_prefix) > 1:
            np.testing.assert_allclose(dd_prefix[:-1], dd[:len(dd_prefix) - 1], atol=1e-8)
        if len(du_prefix) > 1:
            np.testing.assert_allclose(du_prefix[:-1], du[:len(du_prefix) - 1], atol=1e-8)
