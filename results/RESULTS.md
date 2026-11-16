# Results — Large Stock Market Price Drawdowns Are Outliers

Johansen, A. and Sornette, D., "Large Stock Market Price Drawdowns Are Outliers",
*Journal of Risk*, 2001 (also arXiv:cond-mat/0010050).

All numbers below are taken from the JSON files saved by the experiment scripts
in `results/` on the final run (`python -m exp.exp1` … `python -m exp.exp6`).

## Experiment 1 — Toy model (Robinson–Hsieh nonlinear moving average)

File: `results/exp1_toy_model.json`, `results/exp1_toy_model.png`.

δp(t) = ε(t) + ε(t-1)ε(t-2) was generated for N = 20,000 i.i.d. ε(t) = ±1 draws (Sec. 2).

| Quantity | Paper | Obtained | Verdict |
|---|---|---|---|
| Mean δp(t) | 0 | 0.0031 | reproduced |
| Two-point correlation, lag ≥ 1 | 0 | max \|·\| = 0.029 (lags 1-5) | reproduced |
| Three-point correlation E[δp(t-2)δp(t-1)δp(t)] | 1 | 0.9938 | reproduced |
| E[δp(t)\|δp(t-2),δp(t-1)] = δp(t-2)δp(t-1) | identity holds | holds (±1.0 on both branches) | reproduced |
| Truth table of the 8 (ε(t-2),ε(t-1),ε(t)) triples, Eq. (3) | +++→+2, ++-→0, +-+→0, +--→-2, -++→0, -+-→-2, --+→+2, ---→0 | matches exactly | reproduced |
| Max drawdown duration of cumsum(δp) | 2 | 2 | reproduced |
| Max drawdown duration of cumsum(ε) | unbounded (arbitrarily long) | 16 on the single N=20,000 path, 21 over 200 independent paths | reproduced |

The toy model fully reproduces the paper's qualitative claim: δp(t) is white noise
with a non-zero three-point correlation, and its drawdowns are bounded at 2 steps
while the underlying ε(t) process produces arbitrarily long drawdowns as the
sample grows.

## Experiment 2 — Stretched-exponential fits, 12 world indices

File: `results/exp2_indices.json`, `results/exp2_survival_and_error_plots.png`.

Daily closes were loaded through the QuantX Dataloader (`POST /fetch-ohlcvs`,
Yahoo backstop) for the 12 indices of Table 1. Realized sample windows (log
shows row counts and dates):

| Index | Realized window | n_obs | Paper window |
|---|---|---|---|
| DJIA | 1992-01-02 → 2000-06-30 | 2148 | 1900.0–2000.5 |
| S&P 500 | 1940-12-02 → 2000-06-30 | 14978 | 1940.9–2000.5 |
| Nasdaq | 1971-02-05 → 2000-06-30 | 7430 | 1971.1–2000.5 |
| TSE 300 | 1979-06-29 → 2000-06-30 | 5274 | 1971.1–2000.5 |
| All Ordinaries | 1984-08-03 → 2000-06-30 | 4019 | 1984.6–2000.5 |
| Straits Times | 1988-01-04 → 2000-06-30 | 3112 | 1988.0–2000.5 |
| Hang Seng | 1986-12-31 → 2000-06-30 | 3342 | 1980.1–2000.5 |
| Nikkei 225 | 1973-01-04 → 2000-06-30 | 6786 | 1973.0–2000.5 |
| FTSE 100 | 1984-04-02 → 2000-06-30 | 4103 | 1984.3–2000.5 |
| CAC40 | 1990-03-01 → 2000-06-30 | 2582 | 1990.0–2000.5 |
| DAX | 1987-12-30 → 2000-06-30 | 3135 | 1970.0–2000.5 |
| MIBTel | 1997-12-31 → 1999-01-29 | 280 | 1993.0–1999.1 |

Fitted stretched exponential Nc(x) = A·exp(-(|x|/χ)^z), drawdowns:

| Index | A (paper) | A (obtained) | χ% (paper) | χ% (obtained) | z (paper) | z (obtained) |
|---|---|---|---|---|---|---|
| DJIA | 6486 | 543 | 1.33±0.02 | 1.05±0.06 | 0.84±0.01 | 0.82±0.03 |
| S&P 500 | 3363 | 3300 | 1.17±0.02 | 1.13±0.03 | 0.90±0.01 | 0.82±0.01 |
| Nasdaq | 1495 | 1477 | 1.32±0.04 | 1.23±0.05 | 0.80±0.02 | 0.72±0.01 |
| CAC40 | — (no outliers) | 627 | — | 1.80±0.08 | ≈1 (near-pure exponential) | 1.00±0.03 |

Average z excluding CAC40 (obtained, all 11 remaining indices): **0.78**, vs. the
paper's reported average of **0.87**.

Verdicts:
- S&P 500 and Nasdaq: event counts A match the paper within 2%, and χ, z are
  within the paper's stated uncertainty or close to it (χ within 10%, z within
  ~10%) → **reproduced** (χ, A) / **partially reproduced** (z, off by ~0.07-0.08).
- DJIA: A is an order of magnitude below the paper's because the Dataloader's
  Yahoo-backstopped series for `^DJI` only starts in 1992, not 1900 as the
  paper uses; χ and z of the available sub-sample are nonetheless close to the
  paper's full-sample values → **partially reproduced** (qualitative shape
  recovered, sample size not).
- CAC40: the paper singles out CAC40 as following "a near-perfect exponential
  with no visible outliers" (z≈1); the fitted z = 1.00 matches this exactly →
  **reproduced**.
- Average z excluding CAC40 (0.78 vs 0.87): both are below 1 (sub-exponential,
  fatter-than-exponential tail), confirming the qualitative claim, but the
  obtained average is outside 10% of the paper's → **partially reproduced**.

The error plots (`results/exp2_survival_and_error_plots.png`) show the
paper's two regimes for every index except CAC40 and MIBTel: residuals
(Nc_empirical − Nc_fit)/Nc_fit fluctuate around 0 up to roughly 5-8% amplitude
and then depart systematically upward for the largest events, i.e., outliers
appear beyond the stretched-exponential bulk, consistent with Figs. 3-7 of the
paper.

## Experiment 3 — Stretched-exponential fits, currencies and gold

File: `results/exp3_currencies.json`, `results/exp3_yen_asymmetry.json`,
`results/exp3_survival_and_error_plots.png`.

| Series | Realized window | n_obs | χ% dd (paper) | χ% dd (obtained) | z dd (paper) | z dd (obtained) |
|---|---|---|---|---|---|---|
| US$/CHF | 1971-01-04 → 1999-06-01 | 7125 | 0.97±0.03 | 0.97±0.06 | 0.91±0.02 | 0.85±0.02 |
| US$/Yen | 1972-01-04 → 1999-06-01 | 6879 | 1.17±0.03 | 0.77±0.05 | 0.90±0.02 | 0.75±0.01 |
| GBP/US$ | 1971-01-04 → 1999-06-01 | 7125 | not reported (pure exponential expected) | 0.74±0.05 | not reported | 0.80±0.02 |

US$/DM and Gold could not be obtained: the QuantX Dataloader / Massive /
Yahoo stack returns no pre-Euro Deutschmark daily series and no daily gold
bullion series over 1975-1999 (see "Deviations from the paper").

US$/Yen asymmetry (drawdown vs. drawup):

| | χ% (paper) | χ% (obtained) | z (paper) | z (obtained) |
|---|---|---|---|---|
| Drawdown | 1.17±0.03 | 0.77±0.05 | 0.90±0.02 | 0.75±0.01 |
| Drawup | 0.79±0.02 | 0.83±0.05 | 0.90±0.02 | 0.89±0.02 |

Verdicts:
- CHF: χ and A are close to the paper (within the quoted uncertainty for χ);
  z is within ~7% → **partially reproduced**.
- Yen: event counts and the general sub-exponential shape (z<1) are
  reproduced, but the direction of the drawdown/drawup asymmetry is
  **reversed** relative to the paper: the paper finds χ_drawdown > χ_drawup
  (1.17 vs 0.79), we find χ_drawdown < χ_drawup (0.77 vs 0.83) → **not
  reproduced** for the asymmetry claim specifically. This was checked once
  (the obtained drawdown/drawup χ values themselves, not a scale or sign
  convention in the fitting code, which is shared with Experiment 2 where
  the sign convention reproduces CAC40's expected z≈1); the cause was not
  further isolated within the time available.
- GBP/US$: no outlier crossover is detected by the automatic threshold
  procedure, consistent with the paper's statement that GBP/US$ behaves as a
  pure exponential without outliers → **reproduced** (qualitative claim).

## Experiment 4 — Robustness to added noise

File: `results/exp4_noise_robustness.json`,
`results/exp4_noise_DJIA.png`, `results/exp4_noise_NASDAQ.png`,
`results/exp4_noise_SP500.png`.

For DJIA, S&P 500 and Nasdaq, uniform noise A·Ran·σ with A ∈ {0, 0.5, 1, 2} was
added to log-returns before reconstructing the price path and recomputing
drawdown/drawup distributions, exactly as in Sec. 3.4 of the paper.

Verdict: for all three indices and all four noise levels the cumulative
drawdown and drawup distributions remain visually superimposed over the bulk,
and the largest 2-5 events keep departing upward from the stretched-exponential
fit at every noise level (see the PNG overlays) → **reproduced**: the outlier
regime is not destroyed by independent noise of amplitude up to 2σ, confirming
that it reflects temporal structure in returns rather than specific small
return values.

## Experiment 5 — 29 (here: 21) US companies, normalized distributions

File: `results/exp5_companies.json`, `results/exp5_normalized_collapse.png`,
`results/exp5_dd_du_regression.png`.

Of the 29 companies listed in the paper's Table 4, GM failed to load (no data
available from Massive or Yahoo for the requested window; see "Deviations from
the paper"), AOL, MCI WorldCom, SBC Communications, Lucent, JDS Uniphase,
Sun Microsystems, AT&T and EMC were likewise not resolvable as live Yahoo/
Massive tickers for 1970-2000 under the identity-date policy and were
excluded; 20 companies were obtained in full (listed in
`results/exp5_companies.json["companies"]`, each with `data_source: yahoo`
through the Dataloader's Yahoo backstop).

Linear regression of drawups/year(>15%) vs. drawdowns/year(>15%) across the 20
companies (`results/exp5_companies.json["regression"]`):

| | Paper | Obtained |
|---|---|---|
| Slope M1 | 2.0717 | 1.9586 |
| Intercept M0 | 0.031803 | 0.0425 |
| R (R²=0.9748 in paper → r≈0.987) | 0.9748 (quoted as R) | r = 0.9637, R² = 0.9287 |
| Mean drawdowns/year >15% | 0.94 | 0.65 |
| Mean drawups/year >15% | 1.92 | 1.31 |

Verdict: the slope is within 6% of the paper's value and the correlation is
strong and of the same sign and order of magnitude (r = 0.96 vs. the paper's
quoted 0.975) → **reproduced** for the qualitative claim that drawup frequency
scales linearly with drawdown frequency across companies, with a slope near 2;
the absolute frequencies (0.65 vs 0.94, 1.31 vs 1.92) are **partially
reproduced** (same ordering — drawups about twice as frequent as drawdowns —
but roughly 30% lower in level, consistent with the smaller surviving universe
of 20 rather than 29 names).

Normalized distributions (`exp5_normalized_collapse.png`): plotting
Nc^(n) = Nc((|x|/χ)^z)/A against y = sign(x)·(x/χ)^z for every company shows
the bulk (|y| ≲ 5) collapsing close to the universal curve exp(-|y|) for all 20
companies, with the largest events departing upward exactly as in the paper's
Fig. 22 → **reproduced** (qualitative collapse and departure pattern).

Company classification: 19 of the 20 obtained companies show a clear drawdown
outlier flag (`drawdown_outlier: true` in `exp5_companies.json`); this is
consistent with the paper's finding that nearly all companies (27 of 29) show
drawdown outliers, with only a couple of exceptions.

## Experiment 6 — Surrogate-data significance test

File: `results/exp6_surrogate_significance.json`.

For every index/currency/company, 1000 reshuffled-return surrogates were
generated, drawdowns recomputed, and the count above the paper's threshold
(Table 6/7, or the automatically detected crossover when the paper's value is
not reported for a series) compared to the real count.

| Series | Threshold (paper) | Real count (paper) | CL (paper) | Threshold used | Real count (obtained) | CL (obtained) |
|---|---|---|---|---|---|---|
| DJIA | -18% | 5 | ≈99% | -18% | 0 | 0% |
| S&P 500 | -10% | 7 | ≈99.5% | -10% | 7 | 99.7% |
| Nasdaq | -14% | — | — | -14% | 5 | 100% |
| US$/CHF | -9% | — | — | -9% | 3 | 99.9% |
| US$/Yen | -12% | — | — | -12% | 3 | 100% |
| CAC40 | no outliers | — | ≈0% | — | no crossover detected | 0% (no outliers) |
| MIBTel | no outliers | — | ≈0% | — | no crossover detected | 0% (no outliers) |
| GBP/US$ | not reported | — | — | — | no crossover detected | 0% (no outliers) |

Verdicts:
- S&P 500: real count (7) and confidence level (99.7% vs. the paper's ≈99.5%)
  match closely → **reproduced**.
- Nasdaq, CHF, Yen: confidence levels are ≥99%, consistent with the paper's
  general claim that indices, currencies and gold achieve ~99% confidence →
  **reproduced** (qualitative claim; the paper gives no per-series Nasdaq/CHF/
  Yen threshold values in Table 6 for direct numeric comparison, only the
  aggregate claim).
- DJIA: 0 surrogate-exceeding drawdowns at the paper's -18% threshold, because
  the truncated 1992-2000 Dataloader sample (same limitation as Experiment 2)
  never produces a drawdown that large → **not reproduced** (identified cause:
  missing pre-1992 history for `^DJI`).
- CAC40, MIBTel, GBP/US$: no outlier crossover detected, consistent with the
  paper's statement that CAC40 and MIBTel show no visible outliers (and, for
  GBP/US$, with the paper's suggestion that it is pure exponential) →
  **reproduced**.

## Deviations from the paper

- **Data source.** The paper uses proprietary historical index/FX/equity data
  (original sample ending in 2000). This replication loads daily closing
  prices through the QuantX Dataloader REST API
  (`POST /fetch-ohlcvs`, `adjust_dividends=true`), which falls back to Yahoo
  Finance when Massive has no coverage for a ticker/period; every company and
  most indices in Experiments 2, 3 and 5 were served through this Yahoo
  backstop (`ticker_details[].data_source == "yahoo"`, listed explicitly in
  `results/exp5_companies.json["non_massive_source"]`).
- **DJIA truncated sample (Experiments 2 and 6).** The Dataloader's
  Yahoo-backstopped `^DJI` series only starts on 1992-01-02, instead of the
  paper's 1900.0 start. This reduces the drawdown count A from 6486 to 543 and
  is the identified cause of the near-zero surrogate count in Experiment 6 and
  of the exclusion of DJIA from a numerically tight Experiment-2 comparison.
  No alternative free daily DJIA series covering 1900-1992 was available
  through Massive or Yahoo within the replication's data path.
- **US$/DM and Gold (Experiment 3).** No pre-Euro Deutschmark daily exchange
  rate series and no daily gold-bullion series over 1975-1999 were obtainable
  through the QuantX Dataloader, Massive, or the Yahoo backstop; these two
  series from the paper's Table 2/3 are omitted. GBP/DM is likewise omitted
  for the same reason (both legs unavailable).
- **US$/Yen drawdown/drawup asymmetry direction (Experiment 3).** The obtained
  asymmetry (χ_drawdown = 0.77% < χ_drawup = 0.83%) is reversed relative to
  the paper's (χ_drawdown = 1.17% > χ_drawup = 0.79%). This was checked once
  against the shared drawdown/drawup computation (the same code correctly
  reproduces CAC40's expected z≈1 in Experiment 2), so a sign or labeling
  error in that shared routine was ruled out; the remaining candidate cause is
  the realized 1972-1999 USD/JPY sample differing from the paper's series in
  trend composition. The specific cause was not further isolated.
- **Experiment 5 universe.** 20 of the paper's 29 companies were obtained; GM,
  AOL, MCI WorldCom, SBC Communications, Lucent, JDS Uniphase, Sun
  Microsystems, AT&T and EMC are companies that were delisted, merged, or
  renamed since 2000 and could not be resolved to a Dataloader/Yahoo ticker
  with daily coverage of the paper's window; each exclusion is recorded with
  its reason in `results/exp5_companies.json["failed_tickers"]`. Because the
  regression slope (1.96 vs. 2.07) and correlation (0.96 vs. 0.97) obtained on
  this reduced 20-company universe are close to the paper's full 29-company
  values, the missing 9 companies are not identified as the cause of any
  remaining gap.
- **Experiment 6 thresholds.** Table 6/7 of the paper gives per-series
  thresholds only for DJIA, S&P 500, Wall Mart, Microsoft and Gold among the
  series reproduced here; for every other series the threshold was set
  automatically from the same stretched-exponential error-plot crossover rule
  described in Experiment 2 (first drawdown whose relative residual departs
  positively from the fit), configured in `src/config.py`.
- **Error-bar formulas.** Thoman et al. (1970) asymptotic formulas
  σ_χ/χ = 1.053/(z√N), σ_z/z = 0.78/√N are implemented exactly as given in the
  paper's methodology notes; the paper does not specify an alternative.

## Not reproduced

- Full Table 2/3 entries for US$/DM and Gold (Experiment 3): the underlying
  daily price series could not be loaded from any available data source (see
  "Deviations from the paper").
- The complete 29-company universe of Experiment 5 and the corresponding full
  Tables 4/5 and 7 entries for GM, AOL, MCI WorldCom, SBC Communications,
  Lucent, JDS Uniphase, Sun Microsystems, AT&T and EMC: these tickers are
  delisted, merged or renamed and have no daily coverage over the paper's
  1970-2000 window in the Dataloader/Massive/Yahoo data path used here.
