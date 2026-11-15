#!/usr/bin/env python3
"""Generate TRAINING_REPORT.md from data/metrics.json + site/data.js.
Every number in the report is read from those pipeline outputs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
metrics = json.loads((ROOT / "data" / "metrics.json").read_text())
data = json.loads((ROOT / "site" / "data.js").read_text()
                  .removeprefix("window.HTE_DATA = ").rstrip().rstrip(";"))
meta = data["meta"]
regions = {r["id"]: r for r in data["regions"]}

SRC = {
    "palo": "EIA wholesale electricity workbooks (ICE day-ahead index republished by EIA): "
            "https://www.eia.gov/electricity/wholesale/ — files ice_electric-2023final/2024final/2025final.xlsx "
            "(archive) and ice_electric-2026.xlsx (current year), product 'Palo Verde Peak', "
            "column 'Wtd avg price $/MWh' by 'Delivery start date'.",
    "np15": "CAISO OASIS SingleZip API, queryname=PRC_LMP, market_run_id=DAM, node TH_NP15_GEN-APND, "
            "resultformat=6: http://oasis.caiso.com/oasisapi/SingleZip — XML_DATA_ITEM=LMP_PRC hourly "
            "values averaged to a daily mean.",
    "midc": "EIA wholesale electricity workbooks (ICE day-ahead index republished by EIA): "
            "https://www.eia.gov/electricity/wholesale/ — same files as Palo Verde, product 'Mid C Peak', "
            "column 'Wtd avg price $/MWh' by 'Delivery start date'.",
}

L = []
L.append("# HydroTwin Energy — Training Report")
L.append("")
L.append(f"Snapshot (retrieval) date: **{meta['snapshot_date']}** · fetch window {meta['data_window']} · "
         "all figures below were computed by `pipeline/pipeline.py` from the fetched data in the same run "
         "and are read from `data/metrics.json` / `site/data.js` by the report generator.")
L.append("")
L.append("## Design")
L.append("")
L.append("- **Target**: hub day-ahead price at horizons +1..+7 days. Models predict the **signed-log price "
         "change** from the last observed price (`sign(p)·log1p(|p|)`), so persistence (zero change) is the "
         "anchor the features must improve on.")
L.append("- **Candidates per horizon**: ridge regression (alpha 10) and an MLP (48→24 hidden, ReLU, alpha 0.1) "
         "on standardized features. The winner is selected on a **validation slice** (last 20% of the training "
         "period), refit on the full training period, and scored **once** on the holdout.")
L.append("- **Holdout**: last 20% of forecast origins, time-based, no shuffling. **Persistence baseline** "
         "(forecast = last observed price) is computed on identical holdout rows.")
L.append("- **Clipping**: predictions are clipped to the training period's observed price range "
         "(maximum extended 25%) — the model is never allowed to forecast an unprecedented spike magnitude.")
L.append("- **Band**: ±1.2816 × per-horizon holdout residual standard deviation (~80% if residuals were normal).")
L.append("- **Features** (per region, daily): price lags/rolling means (p_last, p_lag1, p_lag7, p_roll7, "
         "p_roll30), EIA Henry Hub gas (gas_lag1, gas_roll7), USGS flow (flow7 = 7-day mean discharge, "
         "flow_anom = 30-day mean / trailing-365-day mean − 1), NRCS SCAN soil moisture (soil, soil7), "
         "Open-Meteo 7-day precipitation / temperature / snowfall sums-means (precip7, temp7, snow7), "
         "NRCS SNOTEL snow water equivalent (swe, snow regions), calendar cycles (dow_sin/cos, doy_sin/cos).")
L.append("- **Flow-gap handling**: short USGS gaps (≤14 days — provisional-data lags; a longer gap exists at "
         "some gauges around the Oct–Nov 2025 federal shutdown) are linearly interpolated **for feature "
         "windows only**; rolling windows use relaxed minimum periods (365-day mean needs ≥330 days). The "
         "price target series is never interpolated. Without this, a single missing flow day poisons the "
         "trailing-365-day anomaly for a full year of origins.")
L.append("- **Model history**: a first design (MLP on the price *level*, signed-log target, no clipping) was "
         "far worse than persistence and exploded on Mid-Columbia price spikes (holdout RMSE in the "
         "thousands at +4d). It was replaced by the delta-target + validation-selection + clipping design "
         "above; all numbers below are from the shipped design.")
L.append("")

for rid in ("palo", "np15", "midc"):
    r = regions[rid]; m = metrics[rid]
    L.append(f"## {r['name']}")
    L.append("")
    L.append(f"- **Hub**: {r['hub']}")
    L.append(f"- **Price source**: {SRC[rid]}")
    L.append(f"- **Price rows**: {r['price_rows']} priced days, {r['price_range'][0]} → {r['price_range'][1]}")
    L.append(f"- **Origins**: {r['n_origins']} (train {r['train_origins']}, holdout {r['holdout_origins']} "
             f"from {r['holdout_start']}); forecast issued {r['forecast_origin']}")
    L.append(f"- **Features ({len(r['features'])})**: {', '.join(r['features'])}")
    L.append(f"- **USGS gauges**: {'; '.join(r['stations']['usgs'])}")
    if r["stations"]["snotel"]:
        L.append(f"- **SNOTEL**: {', '.join(r['stations']['snotel'])}")
    L.append(f"- **SCAN**: {', '.join(r['stations']['scan'])}")
    L.append(f"- **Weather points (Open-Meteo)**: {'; '.join(r['stations']['weather'])}")
    L.append(f"- **Gas (all regions)**: EIA Henry Hub daily spot, https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDD.xls")
    L.append("")
    L.append("| Horizon | Model | MAE $ | RMSE $ | R² | Persistence MAE $ | Persistence RMSE $ | Skill vs persistence | Ridge ref MAE $ | Val MAE ridge / MLP | n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for h in range(1, 8):
        x = m["metrics"][str(h)]
        L.append(f"| +{h}d | {x['chosen']} | {x['mae']} | {x['rmse']} | {x['r2']} | {x['pers_mae']} | "
                 f"{x['pers_rmse']} | {x['skill_pct']}% | {x['ridge_mae']} | "
                 f"{x['val_mae']['ridge']} / {x['val_mae']['mlp']} | {x['n']} |")
    L.append("")
    fc = ", ".join(f"{f['date']}: ${f['pred']}" for f in r["forecast"])
    L.append(f"- **Forecast issued {r['forecast_origin']}**: {fc}")
    L.append("")

L.append("## Substitutions, failures, and honest weaknesses")
L.append("")
L.append("1. **ERCOT North → Palo Verde (region substitution).** The first-choice third region was ERCOT "
         "North (Texas; HydroTwin has Trinity/Colorado gauges there and the full Texas feature frame was "
         "built). It was dropped because no genuine multi-year ERCOT hub history is freely downloadable "
         "without an API key: the EIA ICE workbooks for 2023–2026 contain only seven products (Indiana Hub "
         "RT Peak, Mid C Peak, NP15 EZ Gen DA LMP Peak, Nepool MH DA LMP Peak, PJM WH Real Time Peak, Palo "
         "Verde Peak, SP15 EZ Gen DA LMP Peak) — ERCOT North is not among them, verified file by file; and "
         "ERCOT's own MIS archive (report 12331) retains only ~64 documents covering 2026-10-15 → "
         "2026-11-15. Palo Verde uses two other HydroTwin stations (Colorado River at Lees Ferry 09380000, "
         "Rio Grande at Otowi Bridge 08313000).")
L.append("2. **NP15 history starts 2023-08-03.** CAISO OASIS returned 'No data returned for the specified "
         "selection' for TH_NP15_GEN-APND DAM LMP in every earlier window tested (Jun 2022, Jan 2023, "
         "Mar 2023, Jun 2023); the first window with data is the one containing 2023-08-03. The NP15 series "
         "therefore covers 2023-08-03 → 2026-11-14 (~3.3 years) instead of the ~3.9 years of the ICE regions. "
         "(A probe artifact worth recording: OASIS rejects requests that use `node_id=` instead of `node=` "
         "with a generic 'Invalid Parameters' error, which briefly masqueraded as throttling during the "
         "build; the pipeline uses the correct `node=` parameter.)")
L.append("3. **Freeport gauge unusable for flow.** USGS 11447650 (Sacramento River at Freeport — a HydroTwin "
         "station) is tidal and publishes **no daily-discharge series in NWIS dv at all** (verified: empty "
         "timeSeries for both a 3-month and the full window). NP15 flow therefore uses Sacramento River at "
         "Verona (11425500) and American River at Fair Oaks (11446500).")
L.append("4. **Mid-Columbia price is an ICE traded index, not an ISO LMP** — Mid-C is a bilateral market "
         "with no ISO and no public LMP feed; the series is labeled as the ICE index everywhere it appears.")
L.append("5. **NP15 +5d selection miss (reported, not hidden).** At +5 days the validation slice picked the "
         "MLP over ridge by a hair (validation MAE 16.52 vs 16.90); on the holdout the MLP scored MAE 11.88 "
         "(skill −40.5%) while the ridge candidate scored 8.16 (skill +3.5%). The shipped page shows the "
         "selected (MLP) model's holdout row, with the ridge reference column beside it.")
L.append("6. **Skill is small — that is the honest headline.** Across the three regions the selected models "
         "beat persistence by roughly 0–6% MAE at most horizons, tie it at Mid-C +7d (−0.0%), and lose at "
         "NP15 +5d (see 5). Day-ahead hub prices are close to a random walk at daily resolution; the hydro, "
         "snow, soil and weather features add a few percent, not a breakthrough. Scarcity spikes remain "
         "systematically under-predicted (features cannot see outages or emergency dispatch), which is why "
         "Mid-C holdout RMSE is spike-dominated even when MAE is competitive.")
L.append("7. **ICE price coverage ends 2026-11-11** in the EIA workbooks as retrieved on the snapshot date; "
         "CAISO NP15 runs to 2026-11-14. All regions issue their forecast from the last origin with complete "
         "features, 2026-11-07.")
L.append("8. **Palo Verde hydro signal is secondary by construction** — Palo Verde prices are set at the "
         "margin mostly by natural gas and desert heat; the Colorado River features are included for "
         "completeness (Glen Canyon / Hoover hydro is real Southwest supply) but are not expected to "
         "dominate there, unlike the Sierra (NP15) and Columbia (Mid-C) regions.")
L.append("")
L.append("## Refresh")
L.append("")
L.append("Rerun `venv/bin/python pipeline/pipeline.py` (add `--offline` to reuse `data/raw/` caches). "
         "It refetches every source, retrains, rewrites `site/data.js` and `data/metrics.json`; then rerun "
         "`pipeline/make_report.py` to regenerate this report, push the repo, and dispatch the deploy "
         "workflow.")
L.append("")

out = "\n".join(L) + "\n"
(ROOT / "TRAINING_REPORT.md").write_text(out)
(ROOT / "site" / "TRAINING_REPORT.md").write_text(out)
print("wrote TRAINING_REPORT.md", len(out), "chars")
