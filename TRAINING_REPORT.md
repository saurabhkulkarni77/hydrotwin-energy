# HydroTwin Energy — Training Report

Snapshot (retrieval) date: **2026-11-15** · fetch window 2022-01-01 .. 2026-11-14 · all figures below were computed by `pipeline/pipeline.py` from the fetched data in the same run and are read from `data/metrics.json` / `site/data.js` by the report generator.

## Design

- **Target**: hub day-ahead price at horizons +1..+7 days. Models predict the **signed-log price change** from the last observed price (`sign(p)·log1p(|p|)`), so persistence (zero change) is the anchor the features must improve on.
- **Candidates per horizon**: ridge regression (alpha 10) and an MLP (48→24 hidden, ReLU, alpha 0.1) on standardized features. The winner is selected on a **validation slice** (last 20% of the training period), refit on the full training period, and scored **once** on the holdout.
- **Holdout**: last 20% of forecast origins, time-based, no shuffling. **Persistence baseline** (forecast = last observed price) is computed on identical holdout rows.
- **Clipping**: predictions are clipped to the training period's observed price range (maximum extended 25%) — the model is never allowed to forecast an unprecedented spike magnitude.
- **Band**: ±1.2816 × per-horizon holdout residual standard deviation (~80% if residuals were normal).
- **Features** (per region, daily): price lags/rolling means (p_last, p_lag1, p_lag7, p_roll7, p_roll30), EIA Henry Hub gas (gas_lag1, gas_roll7), USGS flow (flow7 = 7-day mean discharge, flow_anom = 30-day mean / trailing-365-day mean − 1), NRCS SCAN soil moisture (soil, soil7), Open-Meteo 7-day precipitation / temperature / snowfall sums-means (precip7, temp7, snow7), NRCS SNOTEL snow water equivalent (swe, snow regions), calendar cycles (dow_sin/cos, doy_sin/cos).
- **Flow-gap handling**: short USGS gaps (≤14 days — provisional-data lags; a longer gap exists at some gauges around the Oct–Nov 2025 federal shutdown) are linearly interpolated **for feature windows only**; rolling windows use relaxed minimum periods (365-day mean needs ≥330 days). The price target series is never interpolated. Without this, a single missing flow day poisons the trailing-365-day anomaly for a full year of origins.
- **Model history**: a first design (MLP on the price *level*, signed-log target, no clipping) was far worse than persistence and exploded on Mid-Columbia price spikes (holdout RMSE in the thousands at +4d). It was replaced by the delta-target + validation-selection + clipping design above; all numbers below are from the shipped design.

## Palo Verde — Southwest (Colorado River)

- **Hub**: Palo Verde hub (Arizona) — ICE day-ahead peak index (weighted avg $/MWh), republished by EIA
- **Price source**: EIA wholesale electricity workbooks (ICE day-ahead index republished by EIA): https://www.eia.gov/electricity/wholesale/ — files ice_electric-2023final/2024final/2025final.xlsx (archive) and ice_electric-2026.xlsx (current year), product 'Palo Verde Peak', column 'Wtd avg price $/MWh' by 'Delivery start date'.
- **Price rows**: 893 priced days, 2022-12-29 → 2026-11-11
- **Origins**: 1318 (train 1054, holdout 264 from 2026-02-17); forecast issued 2026-11-07
- **Features (19)**: p_last, p_lag1, p_lag7, p_roll7, p_roll30, gas_lag1, gas_roll7, flow7, flow_anom, soil, soil7, precip7, temp7, snow7, swe, dow_sin, dow_cos, doy_sin, doy_cos
- **USGS gauges**: USGS 09380000 — Colorado River At Lees Ferry, Az; USGS 08313000 — Rio Grande At Otowi Bridge, Nm
- **SNOTEL**: 874:CO:SNTL, 542:CO:SNTL, 737:CO:SNTL
- **SCAN**: 2026:AZ:SCAN, 2172:NM:SCAN, 2171:NM:SCAN
- **Weather points (Open-Meteo)**: Page, AZ (Lake Powell); Flagstaff, AZ
- **Gas (all regions)**: EIA Henry Hub daily spot, https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDD.xls

| Horizon | Model | MAE $ | RMSE $ | R² | Persistence MAE $ | Persistence RMSE $ | Skill vs persistence | Ridge ref MAE $ | Val MAE ridge / MLP | n |
|---|---|---|---|---|---|---|---|---|---|---|
| +1d | ridge | 6.89 | 9.14 | 0.832 | 7.24 | 9.99 | 4.7% | 6.89 | 6.48 / 9.48 | 164 |
| +2d | ridge | 9.57 | 12.48 | 0.687 | 9.78 | 13.38 | 2.1% | 9.57 | 8.33 / 15.18 | 164 |
| +3d | ridge | 10.59 | 13.71 | 0.62 | 10.67 | 14.6 | 0.7% | 10.59 | 8.87 / 12.83 | 165 |
| +4d | ridge | 11.6 | 14.79 | 0.556 | 11.61 | 15.71 | 0.1% | 11.6 | 9.43 / 12.94 | 165 |
| +5d | ridge | 12.31 | 16.14 | 0.472 | 12.77 | 17.38 | 3.6% | 12.31 | 10.22 / 17.46 | 165 |
| +6d | ridge | 13.02 | 17.17 | 0.402 | 13.4 | 18.95 | 2.9% | 13.02 | 10.78 / 16.45 | 165 |
| +7d | ridge | 13.24 | 17.49 | 0.381 | 13.86 | 19.25 | 4.4% | 13.24 | 11.98 / 12.5 | 164 |

- **Forecast issued 2026-11-07**: 2026-11-08: $36.88, 2026-11-09: $41.29, 2026-11-10: $46.21, 2026-11-11: $43.41, 2026-11-12: $39.08, 2026-11-13: $36.22, 2026-11-14: $34.14

## CAISO NP15 — Northern California

- **Hub**: CAISO NP15 trading hub — day-ahead LMP, daily mean of hourly prices ($/MWh), from CAISO OASIS
- **Price source**: CAISO OASIS SingleZip API, queryname=PRC_LMP, market_run_id=DAM, node TH_NP15_GEN-APND, resultformat=6: http://oasis.caiso.com/oasisapi/SingleZip — XML_DATA_ITEM=LMP_PRC hourly values averaged to a daily mean.
- **Price rows**: 1200 priced days, 2023-08-03 → 2026-11-14
- **Origins**: 1164 (train 931, holdout 233 from 2026-03-20); forecast issued 2026-11-07
- **Features (19)**: p_last, p_lag1, p_lag7, p_roll7, p_roll30, gas_lag1, gas_roll7, flow7, flow_anom, soil, soil7, precip7, temp7, snow7, swe, dow_sin, dow_cos, doy_sin, doy_cos
- **USGS gauges**: USGS 11425500 — Sacramento R A Verona Ca; USGS 11446500 — American R A Fair Oaks Ca
- **SNOTEL**: 428:CA:SNTL, 1067:CA:SNTL, 462:CA:SNTL
- **SCAN**: 2234:CA:SCAN, 2217:CA:SCAN
- **Weather points (Open-Meteo)**: Sacramento, CA; Donner Summit, CA
- **Gas (all regions)**: EIA Henry Hub daily spot, https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDD.xls

| Horizon | Model | MAE $ | RMSE $ | R² | Persistence MAE $ | Persistence RMSE $ | Skill vs persistence | Ridge ref MAE $ | Val MAE ridge / MLP | n |
|---|---|---|---|---|---|---|---|---|---|---|
| +1d | ridge | 4.65 | 6.37 | 0.844 | 4.8 | 6.86 | 3.0% | 4.65 | 5.2 / 11.36 | 233 |
| +2d | ridge | 6.61 | 8.59 | 0.717 | 7.01 | 9.62 | 5.6% | 6.61 | 9.24 / 15.09 | 233 |
| +3d | ridge | 7.32 | 9.38 | 0.663 | 7.75 | 10.54 | 5.5% | 7.32 | 11.32 / 16.72 | 233 |
| +4d | ridge | 7.84 | 9.84 | 0.629 | 8.13 | 11.05 | 3.6% | 7.84 | 13.62 / 24.99 | 233 |
| +5d | mlp | 11.88 | 16.11 | 0.009 | 8.46 | 11.36 | -40.5% | 8.16 | 16.9 / 16.52 | 233 |
| +6d | ridge | 7.99 | 10.36 | 0.59 | 8.5 | 11.57 | 6.1% | 7.99 | 18.81 / 28.58 | 233 |
| +7d | ridge | 7.96 | 10.12 | 0.608 | 8.28 | 11.28 | 3.9% | 7.96 | 20.78 / 26.61 | 233 |

- **Forecast issued 2026-11-07**: 2026-11-08: $44.21, 2026-11-09: $51.28, 2026-11-10: $57.0, 2026-11-11: $60.2, 2026-11-12: $51.19, 2026-11-13: $52.48, 2026-11-14: $46.53

## Mid-Columbia — Pacific Northwest

- **Hub**: Mid-Columbia hub — ICE day-ahead peak index (weighted avg $/MWh), republished by EIA
- **Price source**: EIA wholesale electricity workbooks (ICE day-ahead index republished by EIA): https://www.eia.gov/electricity/wholesale/ — same files as Palo Verde, product 'Mid C Peak', column 'Wtd avg price $/MWh' by 'Delivery start date'.
- **Price rows**: 896 priced days, 2022-12-29 → 2026-11-11
- **Origins**: 1318 (train 1054, holdout 264 from 2026-02-17); forecast issued 2026-11-07
- **Features (19)**: p_last, p_lag1, p_lag7, p_roll7, p_roll30, gas_lag1, gas_roll7, flow7, flow_anom, soil, soil7, precip7, temp7, snow7, swe, dow_sin, dow_cos, doy_sin, doy_cos
- **USGS gauges**: USGS 14105700 — Columbia River At The Dalles, Or; USGS 14211720 — Willamette River At Portland, Or
- **SNOTEL**: 651:OR:SNTL, 1314:OR:SNTL, 1129:WA:SNTL
- **SCAN**: 2198:WA:SCAN, 2021:WA:SCAN
- **Weather points (Open-Meteo)**: Portland, OR; Mt Hood, OR
- **Gas (all regions)**: EIA Henry Hub daily spot, https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDD.xls

| Horizon | Model | MAE $ | RMSE $ | R² | Persistence MAE $ | Persistence RMSE $ | Skill vs persistence | Ridge ref MAE $ | Val MAE ridge / MLP | n |
|---|---|---|---|---|---|---|---|---|---|---|
| +1d | ridge | 10.49 | 14.38 | 0.703 | 10.85 | 15.06 | 3.3% | 10.49 | 11.05 / 15.24 | 164 |
| +2d | ridge | 12.14 | 16.69 | 0.6 | 12.56 | 17.7 | 3.4% | 12.14 | 17.84 / 38.02 | 164 |
| +3d | ridge | 12.74 | 17.45 | 0.56 | 13.58 | 19.2 | 6.2% | 12.74 | 14.69 / 27.65 | 165 |
| +4d | ridge | 12.89 | 17.39 | 0.563 | 13.1 | 17.71 | 1.6% | 12.89 | 15.98 / 40.92 | 165 |
| +5d | ridge | 13.4 | 17.86 | 0.539 | 13.64 | 18.33 | 1.7% | 13.4 | 16.94 / 48.47 | 165 |
| +6d | ridge | 14.59 | 18.89 | 0.484 | 14.69 | 19.52 | 0.7% | 14.59 | 17.86 / 41.87 | 165 |
| +7d | ridge | 14.85 | 19.56 | 0.45 | 14.85 | 20.3 | -0.0% | 14.85 | 22.2 / 53.63 | 164 |

- **Forecast issued 2026-11-07**: 2026-11-08: $38.21, 2026-11-09: $43.66, 2026-11-10: $45.41, 2026-11-11: $45.41, 2026-11-12: $45.43, 2026-11-13: $43.81, 2026-11-14: $42.27

## Substitutions, failures, and honest weaknesses

1. **ERCOT North → Palo Verde (region substitution).** The first-choice third region was ERCOT North (Texas; HydroTwin has Trinity/Colorado gauges there and the full Texas feature frame was built). It was dropped because no genuine multi-year ERCOT hub history is freely downloadable without an API key: the EIA ICE workbooks for 2023–2026 contain only seven products (Indiana Hub RT Peak, Mid C Peak, NP15 EZ Gen DA LMP Peak, Nepool MH DA LMP Peak, PJM WH Real Time Peak, Palo Verde Peak, SP15 EZ Gen DA LMP Peak) — ERCOT North is not among them, verified file by file; and ERCOT's own MIS archive (report 12331) retains only ~64 documents covering 2026-10-15 → 2026-11-15. Palo Verde uses two other HydroTwin stations (Colorado River at Lees Ferry 09380000, Rio Grande at Otowi Bridge 08313000).
2. **NP15 history starts 2023-08-03.** CAISO OASIS returned 'No data returned for the specified selection' for TH_NP15_GEN-APND DAM LMP in every earlier window tested (Jun 2022, Jan 2023, Mar 2023, Jun 2023); the first window with data is the one containing 2023-08-03. The NP15 series therefore covers 2023-08-03 → 2026-11-14 (~3.3 years) instead of the ~3.9 years of the ICE regions. (A probe artifact worth recording: OASIS rejects requests that use `node_id=` instead of `node=` with a generic 'Invalid Parameters' error, which briefly masqueraded as throttling during the build; the pipeline uses the correct `node=` parameter.)
3. **Freeport gauge unusable for flow.** USGS 11447650 (Sacramento River at Freeport — a HydroTwin station) is tidal and publishes **no daily-discharge series in NWIS dv at all** (verified: empty timeSeries for both a 3-month and the full window). NP15 flow therefore uses Sacramento River at Verona (11425500) and American River at Fair Oaks (11446500).
4. **Mid-Columbia price is an ICE traded index, not an ISO LMP** — Mid-C is a bilateral market with no ISO and no public LMP feed; the series is labeled as the ICE index everywhere it appears.
5. **NP15 +5d selection miss (reported, not hidden).** At +5 days the validation slice picked the MLP over ridge by a hair (validation MAE 16.52 vs 16.90); on the holdout the MLP scored MAE 11.88 (skill −40.5%) while the ridge candidate scored 8.16 (skill +3.5%). The shipped page shows the selected (MLP) model's holdout row, with the ridge reference column beside it.
6. **Skill is small — that is the honest headline.** Across the three regions the selected models beat persistence by roughly 0–6% MAE at most horizons, tie it at Mid-C +7d (−0.0%), and lose at NP15 +5d (see 5). Day-ahead hub prices are close to a random walk at daily resolution; the hydro, snow, soil and weather features add a few percent, not a breakthrough. Scarcity spikes remain systematically under-predicted (features cannot see outages or emergency dispatch), which is why Mid-C holdout RMSE is spike-dominated even when MAE is competitive.
7. **ICE price coverage ends 2026-11-11** in the EIA workbooks as retrieved on the snapshot date; CAISO NP15 runs to 2026-11-14. All regions issue their forecast from the last origin with complete features, 2026-11-07.
8. **Palo Verde hydro signal is secondary by construction** — Palo Verde prices are set at the margin mostly by natural gas and desert heat; the Colorado River features are included for completeness (Glen Canyon / Hoover hydro is real Southwest supply) but are not expected to dominate there, unlike the Sierra (NP15) and Columbia (Mid-C) regions.

## Refresh

Rerun `venv/bin/python pipeline/pipeline.py` (add `--offline` to reuse `data/raw/` caches). It refetches every source, retrains, rewrites `site/data.js` and `data/metrics.json`; then rerun `pipeline/make_report.py` to regenerate this report, push the repo, and dispatch the deploy workflow.

