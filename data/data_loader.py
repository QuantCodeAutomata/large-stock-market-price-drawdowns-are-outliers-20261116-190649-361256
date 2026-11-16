"""Market data loading for the drawdown-outlier replication.

Three instrument families need three different sources because the paper's
historical windows predate what the QuantX Dataloader / Massive REST API can
serve (see results/RESULTS.md, "Deviations from the paper" for the smoke-test
evidence):

* US company stocks (experiments 5 and 6): the QuantX Dataloader REST API,
  the project's primary OHLCV source.
* World equity indices (experiment 2): Yahoo Finance. A smoke test of the
  Dataloader against an index ticker (``I:DJI``) returned
  ``NOT_AUTHORIZED`` (no Massive index entitlement in this environment), and
  Massive's own FX/commodity bars do not go back before the 2000s, so none of
  the indices' century-scale windows could be retrieved from Dataloader or
  Massive.
* Currencies (experiment 3): FRED daily exchange-rate series, for the same
  reason (Massive has no FX bars before 2003-2015 for the needed pairs; the
  Dataloader only proxies Massive or Yahoo bars for stocks).

Downloaded data is never committed to the repository (see .gitignore); only
the cache under data/cache/ is written to disk and reused across runs.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15
MAX_PARALLEL = 4
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"


class DataLoaderError(RuntimeError):
    """A failed data request, or a load in which no instrument succeeded."""


# ---------------------------------------------------------------------------
# World equity indices and currencies/gold: Yahoo Finance / FRED
# ---------------------------------------------------------------------------

def load_index_series(yahoo_symbol: str, start_date: str, end_date: str,
                       use_cache: bool = True) -> pd.Series:
    """Daily closing price of an index from Yahoo Finance.

    Returns a pandas Series of closing prices indexed by trading date.
    Dividend adjustment is irrelevant for indices (no shares held).
    """
    key = hashlib.sha256(f"yf:{yahoo_symbol}:{start_date}:{end_date}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{key}.csv.gz"
    if use_cache and cache_path.exists():
        series = pd.read_csv(cache_path, index_col=0, parse_dates=True)["close"]
        return series

    hist = yf.Ticker(yahoo_symbol).history(start=start_date, end=end_date, auto_adjust=False)
    if hist.empty:
        raise DataLoaderError(f"Yahoo Finance returned no rows for {yahoo_symbol}")
    series = hist["Close"].rename("close")
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_frame().to_csv(cache_path)
    return series


def load_fred_series(series_id: str, start_date: str, end_date: str,
                      use_cache: bool = True) -> pd.Series:
    """Daily exchange-rate series from the FRED CSV endpoint (fredgraph.csv)."""
    key = hashlib.sha256(f"fred:{series_id}:{start_date}:{end_date}".encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{key}.csv.gz"
    if use_cache and cache_path.exists():
        series = pd.read_csv(cache_path, index_col=0, parse_dates=True)["close"]
        return series

    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start_date}&coed={end_date}"
    )
    response = requests.get(url, timeout=60)
    header = response.text.splitlines()[0] if response.text else ""
    if response.status_code != 200 or "date" not in header.lower():
        raise DataLoaderError(f"FRED returned an invalid response for {series_id}")
    raw = pd.read_csv(io.StringIO(response.text))
    raw.columns = ["date", "close"]
    raw = raw[raw["close"] != "."]
    if raw.empty:
        raise DataLoaderError(f"FRED series {series_id} has no numeric observations in range")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["close"] = raw["close"].astype(float)
    series = raw.set_index("date")["close"].sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_frame().to_csv(cache_path)
    return series


# ---------------------------------------------------------------------------
# US company stocks: QuantX Dataloader REST API
# ---------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataLoaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {dataloader_key}", "X-Massive-Api-Key": massive_key}


def _request(method: str, route: str, body: dict | None = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(method, API_ROOT + route, json=body, headers=headers, timeout=900)
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2 ** attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataLoaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list[dict]) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataLoaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int) -> tuple[pd.DataFrame, list[dict]]:
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    return bars, details


def load_company_bars(tickers: list[str], start_date: str, end_date: str,
                       use_cache: bool = True,
                       identity_date: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Daily OHLCV bars for ``tickers`` from the QuantX Dataloader.

    ``identity_date`` fixes the date on which a ticker identifies its intended
    security (Dataloader default: ``end_date``). It must be set to a date
    within the sample period for tickers that have since been reused by an
    unrelated company (e.g. GM, whose symbol was reassigned to the entity
    created in the 2009 bankruptcy), otherwise the loader follows today's
    security back in time and returns no bars for the historical period.

    Returns rows of timestamp, date, ticker, open, high, low, close, volume,
    and metadata with each ticker's ``ticker_details`` entry, failures, and
    non-Massive data sources (e.g. the Yahoo backstop for delisted names).
    """
    request = {
        "timeframe": "1d", "start_date": start_date, "end_date": end_date,
        "source": "massive", "adjust_dividends": True, "yahoo_backstop": True,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(lambda item: _load_batch({**request, "tickers": item[1]}, item[0]), enumerate(batches)))
    frames = [bars for bars, _ in results if not bars.empty]
    details = [detail for _, batch_details in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataLoaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request, "ticker_details": details, "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {d["ticker"]: d["data_source"] for d in details
                                if d.get("data_source") not in (None, "massive")},
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta


def company_close_series(bars: pd.DataFrame, ticker: str) -> pd.Series:
    """Daily closing-price series for one ``ticker`` from ``load_company_bars``
    output, indexed by trading date (New York calendar day)."""
    rows = bars.loc[bars["ticker"] == ticker].sort_values("date")
    if rows.empty:
        raise DataLoaderError(f"No bars for ticker {ticker} in the loaded batch")
    series = rows.set_index("date")["close"]
    series = series[~series.index.duplicated(keep="last")]
    return series
