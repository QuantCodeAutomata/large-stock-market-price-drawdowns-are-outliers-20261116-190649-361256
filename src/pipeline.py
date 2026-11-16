"""Shared per-instrument drawdown/drawup pipeline used by experiments 2-6.

Custom implementation: orchestrates src.drawdown and
src.stretched_exponential into the single sequence of steps the paper applies
to every instrument (Section 3): log-returns -> runs -> cumulative
distribution -> stretched-exponential fit -> error curve. Centralising this
here keeps every experiment script calling one definition of "analyze an
instrument", as required for comparability across Tables 2-7.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.drawdown import compute_drawdowns_drawups
from src.stretched_exponential import StretchedExponentialFit, fit_stretched_exponential


@dataclass
class InstrumentResult:
    label: str
    n_obs: int
    start: str
    end: str
    drawdowns: np.ndarray
    drawups: np.ndarray
    fit_drawdown: StretchedExponentialFit | None
    fit_drawup: StretchedExponentialFit | None
    fit_drawdown_error: str = ""
    fit_drawup_error: str = ""

    def to_summary(self) -> dict:
        """Machine-readable summary row (one per instrument) for the results
        tables of experiments 2-6."""
        row = {
            "label": self.label,
            "n_obs": self.n_obs,
            "start": self.start,
            "end": self.end,
            "n_drawdowns": int(len(self.drawdowns)),
            "n_drawups": int(len(self.drawups)),
        }
        if self.fit_drawdown is not None:
            f = self.fit_drawdown
            row.update({
                "drawdown_A": f.A, "drawdown_B": round(f.B, 4), "drawdown_chi": round(f.chi, 4),
                "drawdown_z": round(f.z, 4), "drawdown_sigma_chi": round(f.sigma_chi, 4),
                "drawdown_sigma_z": round(f.sigma_z, 4),
            })
        else:
            row["drawdown_fit_error"] = self.fit_drawdown_error
        if self.fit_drawup is not None:
            f = self.fit_drawup
            row.update({
                "drawup_A": f.A, "drawup_B": round(f.B, 4), "drawup_chi": round(f.chi, 4),
                "drawup_z": round(f.z, 4), "drawup_sigma_chi": round(f.sigma_chi, 4),
                "drawup_sigma_z": round(f.sigma_z, 4),
            })
        else:
            row["drawup_fit_error"] = self.fit_drawup_error
        return row


def analyze_instrument(prices: pd.Series, label: str) -> InstrumentResult:
    """Run the full drawdown/drawup + stretched-exponential pipeline on one
    closing-price series ``prices`` (paper Section 3)."""
    prices = prices.dropna().astype(float)
    drawdowns, drawups = compute_drawdowns_drawups(prices)

    fit_dd = fit_dd_err = None
    try:
        fit_dd = fit_stretched_exponential(drawdowns)
    except ValueError as exc:
        fit_dd_err = str(exc)

    fit_du = fit_du_err = None
    try:
        fit_du = fit_stretched_exponential(drawups)
    except ValueError as exc:
        fit_du_err = str(exc)

    return InstrumentResult(
        label=label,
        n_obs=len(prices),
        start=str(prices.index.min().date()) if len(prices) else "",
        end=str(prices.index.max().date()) if len(prices) else "",
        drawdowns=drawdowns,
        drawups=drawups,
        fit_drawdown=fit_dd,
        fit_drawup=fit_du,
        fit_drawdown_error=fit_dd_err or "",
        fit_drawup_error=fit_du_err or "",
    )
