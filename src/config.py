"""Module-level configuration for the drawdown-outlier replication.

Every parameter used by an experiment script lives here as a plain
module-level variable with a comment stating its meaning, unit and source,
per Johansen & Sornette, "Large Stock Market Price Drawdowns Are Outliers".
"""

# ---------------------------------------------------------------------------
# Experiment 1: toy nonlinear moving-average model (paper Section 2, Eq. 1-3)
# ---------------------------------------------------------------------------
TOY_MODEL_N = 20_000          # length of the synthetic epsilon(t) series (not specified in the paper; chosen long enough to estimate correlations precisely)
TOY_MODEL_SEED = 42           # RNG seed for reproducibility (not specified in the paper)

# ---------------------------------------------------------------------------
# Experiments 2-6: stretched-exponential fit of drawdown/drawup distributions
# (paper Section 3, Eq. 4; error bars from Thoman et al. 1970 as cited in the paper)
# ---------------------------------------------------------------------------
# A zero daily return is treated as a continuation of the ongoing run rather
# than breaking it (the paper defines a run as "a persistent decrease/increase
# over consecutive days" but does not specify the treatment of exact zeros;
# this is an interpreted choice, see results/RESULTS.md).
ZERO_RETURN_BREAKS_RUN = False

# Experiment 2: major world financial indices (paper Table 2/3, Section 3)
# ticker -> (yahoo_symbol, start_date, end_date, display_name)
# Dates follow the decimal-year windows given in the paper (e.g. 1900.0 = 1900-01-01).
WORLD_INDICES = {
    "DJIA": ("^DJI", "1900-01-01", "2000-07-01", "Dow Jones Industrial Average"),
    "SP500": ("^GSPC", "1940-12-01", "2000-07-01", "S&P 500"),
    "NASDAQ": ("^IXIC", "1971-02-01", "2000-07-01", "Nasdaq Composite"),
    "TSE300": ("^GSPTSE", "1971-02-01", "2000-07-01", "TSE 300 Composite (Toronto)"),
    "AORD": ("^AORD", "1984-08-01", "2000-07-01", "All Ordinaries (Sydney)"),
    "STI": ("^STI", "1988-01-01", "2000-07-01", "Straits Times (Singapore)"),
    "HSI": ("^HSI", "1980-02-01", "2000-07-01", "Hang Seng (Hong Kong)"),
    "N225": ("^N225", "1973-01-01", "2000-07-01", "Nikkei 225 (Tokyo)"),
    "FTSE100": ("^FTSE", "1984-04-01", "2000-07-01", "FTSE 100 (London)"),
    "CAC40": ("^FCHI", "1990-01-01", "2000-07-01", "CAC40 (Paris)"),
    "DAX": ("^GDAXI", "1970-01-01", "2000-07-01", "DAX (Frankfurt)"),
    # MIBTel itself was retired in 2009; Yahoo Finance only carries its
    # successor FTSE MIB from end-1997, shorter than the paper's 1993-1999
    # window (see results/RESULTS.md, "Deviations from the paper").
    "MIBTEL": ("FTSEMIB.MI", "1997-12-31", "1999-02-01", "MIBTel (Milan, successor FTSE MIB series)"),
}

# Experiment 3: currencies and gold (paper Table 2/3, Section 3)
# Sourced from FRED daily noon buying rates (series id) because Dataloader and
# Massive are not entitled to, or do not carry, historical FX/commodity bars
# this far back (see results/RESULTS.md).
CURRENCIES = {
    # FRED DEXSZUS: Swiss Francs to one US Dollar
    "CHF": ("DEXSZUS", "1971-01-04", "1999-06-01", "US$/CHF (Swiss Franc)"),
    # FRED DEXJPUS: Japanese Yen to one US Dollar
    "JPY": ("DEXJPUS", "1972-01-04", "1999-06-01", "US$/Yen"),
    # FRED DEXUSUK: US Dollars to one British Pound (supplementary analysis,
    # paper Section 3: "we also looked at the GBP/US$ ... exchange rate").
    "GBPUSD": ("DEXUSUK", "1971-01-04", "1999-06-01", "GBP/US$"),
}
# US$/DM (Deutschmark), Gold, and GBP/DM could not be retrieved as daily
# series from Dataloader, Massive, FRED or Yahoo Finance for this historical
# window (FRED only carries DM/US$ as a monthly average, series EXGEUS, and
# has no gold-fixing series at the documented ids); see "Not reproduced" in
# results/RESULTS.md.

# Experiment 4: robustness to added noise (paper Section 3.4)
NOISE_AMPLITUDES = [0.0, 0.5, 1.0, 2.0]   # multiples A of return std dev added as uniform[-1,1]*A*sigma (paper Eq. in Section 3.4)
NOISE_SEED = 123                           # RNG seed for the noise realization (not specified in the paper)

# Experiment 5/6: individual US companies (paper Table 4/5/6/7, Section 4)
# ticker -> (yahoo/dataloader symbol, start_date, end_date, display_name)
COMPANIES = {
    "MSFT": ("MSFT", "1986-03-01", "2000-07-01", "Microsoft"),
    "CSCO": ("CSCO", "1990-03-01", "2000-07-01", "Cisco"),
    "GE": ("GE", "1970-01-01", "2000-07-01", "General Electric"),
    "INTC": ("INTC", "1986-06-01", "2000-07-01", "Intel"),
    "XOM": ("XOM", "1970-01-01", "2000-07-01", "Exxon-Mobil"),
    "ORCL": ("ORCL", "1988-03-01", "2000-07-01", "Oracle"),
    "WMT": ("WMT", "1972-09-01", "2000-07-01", "Wall Mart"),
    "IBM": ("IBM", "1970-01-01", "2000-07-01", "IBM"),
    "T": ("T", "1970-01-01", "2000-07-01", "AT&T"),
    "C": ("C", "1977-01-01", "2000-07-01", "Citigroup"),
    "TXN": ("TXN", "1982-01-01", "2000-07-01", "Texas Instruments"),
    "HPQ": ("HPQ", "1977-01-01", "2000-07-01", "Hewlett Packard"),
    "MRK": ("MRK", "1970-01-01", "2000-07-01", "Merck"),
    "PFE": ("PFE", "1982-01-01", "2000-07-01", "Pfizer"),
    "KO": ("KO", "1970-01-01", "2000-07-01", "Coca Cola"),
    "QCOM": ("QCOM", "1992-01-01", "2000-07-01", "Qualcomm"),
    "AMAT": ("AMAT", "1985-01-01", "2000-07-01", "Applied Materials"),
    "PG": ("PG", "1970-01-01", "2000-07-01", "Procter & Gamble"),
    "GM": ("GM", "1970-01-01", "2000-07-01", "General Motors"),
    "MDT": ("MDT", "1982-01-01", "2000-07-01", "Medtronic"),
    "F": ("F", "1977-01-01", "2000-07-01", "Ford"),
    # Delisted / merged tickers not reachable via the live Dataloader/Yahoo
    # feed for this window (Lucent "LU", Sun Microsystems "SUNW", SBC
    # Communications "SBC", EMC "EMC", AOL "AOL", MCI WorldCom "WCOM",
    # JDS Uniphase "JDSU", American Home Products "AHP"); see
    # "Not reproduced" in results/RESULTS.md.
}

COMPANY_DRAWDOWN_THRESHOLD_PCT = 15.0  # amplitude threshold (%) for the annual drawdown/drawup frequency count (paper Section 4, Figure 23)

# Experiment 6: surrogate (reshuffled-returns) significance test (paper Section 3.5 / Table 6-7)
N_SURROGATES = 1000     # number of reshuffled surrogate series per instrument (paper Section 3.5)
SURROGATE_SEED = 7      # RNG seed for the permutations (not specified in the paper)

# Fixed outlier thresholds (% amplitude) read from the error plot crossover,
# reproduced from paper Table 6 for instruments it reports explicitly.
SURROGATE_THRESHOLDS_PAPER = {
    "DJIA": 18.0, "SP500": 10.0, "NASDAQ": 14.0, "TSE300": 10.0, "AORD": 14.0,
    "STI": 29.0, "HSI": 23.0, "N225": 14.0, "FTSE100": 23.0, "DAX": 13.0,
    "CHF": 9.0, "JPY": 12.0,
    # Table 7 company thresholds explicitly reported in the paper text.
    "MSFT": 30.0, "WMT": 22.0,
}
# For every other instrument (CAC40, MIBTel, GBP/US$, and companies not
# listed above) Table 6/7 gives no numeric threshold, so the crossover is
# located automatically from the error-plot rule in _auto_threshold below,
# applying the same bulk/outlier criterion the paper uses visually.
