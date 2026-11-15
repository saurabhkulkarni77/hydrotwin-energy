#!/usr/bin/env python3
"""
HydroTwin Energy — data + training pipeline.

Fetches genuine public data (no API keys), builds daily features per price
region, trains per-horizon price forecasters (ridge / MLP on the signed-log
price change from the last observed price, selected per horizon on a
validation slice), evaluates them on a time-based holdout against a
persistence baseline, and exports the static site snapshot (site/data.js).

Sources:
  Prices  : CAISO OASIS day-ahead LMP (NP15 hub) — ISO's own data
            EIA-republished ICE day-ahead indices (Palo Verde, Mid-C)
  Gas     : EIA Henry Hub daily spot price (public EIA xls)
  Flow    : USGS NWIS daily values
  Snow    : NRCS SNOTEL via AWDB reportGenerator (SWE, start of day)
  Soil    : NRCS SCAN via AWDB reportGenerator (SMS at -8 in)
  Weather : Open-Meteo archive API

Usage:  venv/bin/python pipeline/pipeline.py [--offline]
  --offline reuses data/raw/ caches instead of refetching.
"""
import io, json, math, os, re, sys, time, zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SITE = ROOT / "site"
RAW.mkdir(parents=True, exist_ok=True)
SITE.mkdir(parents=True, exist_ok=True)

FETCH_START = "2022-01-01"          # warmup for trailing features
END = "2026-11-14"                  # last day in snapshot window
ORIGIN_START = "2023-03-01"         # first forecast origin
RETRIEVAL_DATE = date.today().isoformat()
UA = {"User-Agent": "hydrotwin-energy-pipeline/1.0 (research; contact via repo)"}

REGIONS = {
    "palo": dict(
        name="Palo Verde — Southwest (Colorado River)",
        price_kind="ice", ice_product="Palo Verde Peak",
        hub_label="Palo Verde hub (Arizona) — ICE day-ahead peak index (weighted avg $/MWh), republished by EIA",
        usgs=["09380000", "08313000"],
        snotel=["874:CO:SNTL", "542:CO:SNTL", "737:CO:SNTL"],
        scan=["2026:AZ:SCAN", "2172:NM:SCAN", "2171:NM:SCAN"],
        weather=[(36.92, -111.48), (35.20, -111.65)],
        weather_names=["Page, AZ (Lake Powell)", "Flagstaff, AZ"],
        extra_limitations=[
            "Substitution, verified during the build: this twin's first-choice third region was ERCOT North "
            "(Texas), but no genuine multi-year ERCOT hub history is freely downloadable without an API key — "
            "the EIA ICE workbooks for 2023-2026 do not contain the ERCOT North product (checked file by "
            "file), and ERCOT's own MIS archive retains only about one month of DAM Settlement Point Price "
            "files (64 documents, 2026-10-15 to 2026-11-15). The Texas feature frame was therefore replaced "
            "by Palo Verde, which uses two other HydroTwin stations (Colorado River at Lees Ferry, Rio Grande "
            "at Otowi Bridge).",
            "Palo Verde prices are set at the margin mostly by natural gas and desert heat; the Colorado "
            "River signal (Glen Canyon / Hoover hydro) is real but secondary, so expect the hydro features "
            "to matter less here than in the Sierra or Columbia regions.",
        ],
    ),
    "np15": dict(
        name="CAISO NP15 — Northern California",
        price_kind="caiso", caiso_node="TH_NP15_GEN-APND",
        hub_label="CAISO NP15 trading hub — day-ahead LMP, daily mean of hourly prices ($/MWh), from CAISO OASIS",
        usgs=["11425500", "11446500"],
        snotel=["428:CA:SNTL", "1067:CA:SNTL", "462:CA:SNTL"],
        scan=["2234:CA:SCAN", "2217:CA:SCAN"],
        weather=[(38.58, -121.49), (39.33, -120.37)],
        weather_names=["Sacramento, CA", "Donner Summit, CA"],
        extra_limitations=[],
    ),
    "midc": dict(
        name="Mid-Columbia — Pacific Northwest",
        price_kind="ice", ice_product="Mid C Peak",
        hub_label="Mid-Columbia hub — ICE day-ahead peak index (weighted avg $/MWh), republished by EIA",
        usgs=["14105700", "14211720"],
        snotel=["651:OR:SNTL", "1314:OR:SNTL", "1129:WA:SNTL"],
        scan=["2198:WA:SCAN", "2021:WA:SCAN"],
        weather=[(45.52, -122.68), (45.37, -121.70)],
        weather_names=["Portland, OR", "Mt Hood, OR"],
        extra_limitations=[
            "Mid-Columbia is a bilateral market with no ISO, so there is no public LMP feed; the price "
            "series is the ICE day-ahead peak index that EIA republishes — a traded index, not an ISO "
            "clearing price.",
        ],
    ),
}

SOURCE_URLS = dict(
    caiso="http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&market_run_id=DAM&node=TH_NP15_GEN-APND",
    eia_ice="https://www.eia.gov/electricity/wholesale/",
    henry="https://www.eia.gov/dnav/ng/hist/rngwhhdD.htm",
    usgs="https://waterservices.usgs.gov/nwis/dv/",
    nrcs="https://wcc.sc.egov.usda.gov/reportGenerator/",
    meteo="https://archive-api.open-meteo.com/v1/archive",
)

# ---------------------------------------------------------------- fetchers
def get(url, **kw):
    r = requests.get(url, headers=UA, timeout=kw.pop("timeout", 120))
    r.raise_for_status()
    return r

def fetch_ice_prices():
    """EIA ICE electricity workbooks 2023-2026 -> {product: Series(date->price)}."""
    frames = []
    for y in (2023, 2024, 2025, 2026):
        fn = RAW / f"ice_electric-{y}.xlsx"
        if not fn.exists():
            url = (f"https://www.eia.gov/electricity/wholesale/xls/ice_electric-{y}.xlsx" if y == 2026
                   else f"https://www.eia.gov/electricity/wholesale/xls/archive/ice_electric-{y}final.xlsx")
            fn.write_bytes(get(url).content)
        df = pd.read_excel(fn, sheet_name=str(y))
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    out = {}
    for product in ("Palo Verde Peak", "Mid C Peak"):
        d = all_df[all_df["Price hub"] == product].copy()
        d["date"] = pd.to_datetime(d["Delivery start date"]).dt.normalize()
        d = d.dropna(subset=["Wtd avg price $/MWh"]).sort_values("date")
        s = d.groupby("date")["Wtd avg price $/MWh"].last()
        out[product] = s
    return out

def fetch_henry_hub():
    fn = RAW / "henryhub.xls"
    if not fn.exists():
        fn.write_bytes(get("https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDD.xls").content)
    df = pd.read_excel(fn, sheet_name="Data 1", header=None, engine="xlrd")
    df = df.iloc[3:, :2]
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    return df["price"]

def fetch_caiso_daily(node):
    fn = RAW / f"caiso_{node}.csv"
    if fn.exists():
        s = pd.read_csv(fn, parse_dates=["date"]).set_index("date")["price"]
        return s.sort_index()
    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp(END)
    parts = []
    cur = start
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=29), end)
        q = (f"http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP"
             f"&startdatetime={cur.strftime('%Y%m%d')}T08:00-0000"
             f"&enddatetime={(nxt + pd.Timedelta(days=1)).strftime('%Y%m%d')}T08:00-0000"
             f"&version=1&market_run_id=DAM&node={node}&resultformat=6")
        try:
            content = get(q, timeout=180).content
            zf = zipfile.ZipFile(io.BytesIO(content))
            for name in zf.namelist():
                if name.endswith(".csv"):
                    parts.append(pd.read_csv(zf.open(name)))
        except Exception as e:
            print(f"  CAISO chunk {cur.date()} failed: {e}", file=sys.stderr)
        cur = nxt + pd.Timedelta(days=1)
        time.sleep(0.4)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["XML_DATA_ITEM"] == "LMP_PRC"]
    df["date"] = pd.to_datetime(df["OPR_DT"]).dt.normalize()
    s = df.groupby("date")["MW"].mean()
    s.rename("price").to_csv(fn, index_label="date")
    return s.sort_index()

def fetch_usgs(site):
    fn = RAW / f"usgs_{site}.csv"
    names_fn = RAW / "usgs_names.json"
    names = json.loads(names_fn.read_text()) if names_fn.exists() else {}
    if fn.exists():
        s = pd.read_csv(fn, parse_dates=["date"]).set_index("date")["flow"].sort_index()
        return s, names.get(site, site)
    url = (f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site}"
           f"&parameterCd=00060&startDT={FETCH_START}&endDT={END}")
    js = get(url).json()
    series = js["value"]["timeSeries"]
    if not series or not series[0]["values"] or not series[0]["values"][0]["value"]:
        # e.g. USGS 11447650 (Sacramento at Freeport, a HydroTwin station) is
        # tidal and publishes no daily-discharge series in NWIS dv at all.
        print(f"  USGS {site}: no daily discharge series in NWIS dv — gauge skipped")
        empty = pd.Series(dtype=float, name="flow")
        empty.to_csv(fn, index_label="date")
        return empty, f"USGS {site} (no NWIS daily discharge — skipped)"
    ts = series[0]
    name = ts["sourceInfo"]["siteName"].title()
    vals = ts["variable"]["variableName"]
    rows = [(pd.Timestamp(v["dateTime"]).normalize(), float(v["value"]))
            for v in ts["values"][0]["value"]]
    s = pd.Series(dict(rows), name="flow").sort_index()
    s[s < 0] = np.nan
    s.to_csv(fn, index_label="date")
    names[site] = name
    names_fn.write_text(json.dumps(names))
    return s, name

def fetch_nrcs(triplet, element_spec, out_col):
    safe = triplet.replace(":", "_")
    fn = RAW / f"nrcs_{safe}.csv"
    if fn.exists():
        return pd.read_csv(fn, parse_dates=["date"]).set_index("date")[out_col].sort_index()
    trip_enc = triplet.replace(":", "%3A") if False else triplet
    url = ("https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
           "customMultiTimeSeriesGroupByStationReport/daily/start_of_period/"
           f"{trip_enc}%7Cid=%22%22%7Cname/{FETCH_START},{END}/stationId,name,{element_spec}")
    txt = get(url).text
    lines = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = ["date", out_col]
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")[out_col].sort_index()
    s.to_csv(fn, index_label="date")
    return s

def fetch_meteo(lat, lon, tag):
    fn = RAW / f"meteo_{tag}.csv"
    if fn.exists():
        return pd.read_csv(fn, parse_dates=["date"]).set_index("date").sort_index()
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date={FETCH_START}&end_date={END}"
           f"&daily=precipitation_sum,temperature_2m_mean,snowfall_sum&timezone=UTC")
    js = get(url).json()["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(js["time"]),
                       "precip": js["precipitation_sum"],
                       "temp": js["temperature_2m_mean"],
                       "snowfall": js["snowfall_sum"]}).set_index("date")
    df.to_csv(fn, index_label="date")
    return df

# ---------------------------------------------------------------- features
def build_region_frame(cfg, prices_all, gas):
    idx = pd.date_range(FETCH_START, END, freq="D")
    F = pd.DataFrame(index=idx)
    # price
    if cfg["price_kind"] == "caiso":
        price = fetch_caiso_daily(cfg["caiso_node"])
    else:
        price = prices_all[cfg["ice_product"]]
    F["price"] = price.reindex(idx)
    # gas (shared)
    F["gas"] = gas.reindex(idx).ffill(limit=5)
    # flow: mean across gauges of daily discharge
    flows, gauge_names = [], []
    for site in cfg["usgs"]:
        res = fetch_usgs(site)
        s, nm = res if isinstance(res, tuple) else (res, site)
        s = s.reindex(idx)
        if s.notna().sum() < 300:
            print(f"  USGS {site}: only {int(s.notna().sum())} daily values — gauge skipped")
            continue
        flows.append(s); gauge_names.append(f"USGS {site} — {nm}")
    assert flows, "no usable USGS gauge for region"
    F["flow"] = pd.concat(flows, axis=1).mean(axis=1)
    # snow (SNOTEL SWE mean)
    if cfg["snotel"]:
        swes = [fetch_nrcs(t, "WTEQ::value", "swe").reindex(idx) for t in cfg["snotel"]]
        F["swe"] = pd.concat(swes, axis=1).mean(axis=1)
    else:
        F["swe"] = np.nan
    # soil (SCAN SMS -8in mean)
    soils = [fetch_nrcs(t, "SMS:-8:value", "soil").reindex(idx) for t in cfg["scan"]]
    F["soil"] = pd.concat(soils, axis=1).mean(axis=1)
    # weather: mean across points
    mets = [fetch_meteo(lat, lon, f"{abs(lat):.2f}_{abs(lon):.2f}") for lat, lon in cfg["weather"]]
    F["precip"] = pd.concat([m["precip"].reindex(idx) for m in mets], axis=1).mean(axis=1)
    F["temp"] = pd.concat([m["temp"].reindex(idx) for m in mets], axis=1).mean(axis=1)
    F["snowfall"] = pd.concat([m["snowfall"].reindex(idx) for m in mets], axis=1).mean(axis=1)
    return F, gauge_names

def make_features(F, has_snow):
    X = pd.DataFrame(index=F.index)
    pff = F["price"].ffill(limit=4)
    X["p_last"] = pff
    X["p_lag1"] = pff.shift(1)
    X["p_lag7"] = pff.shift(7)
    X["p_roll7"] = pff.rolling(7).mean()
    X["p_roll30"] = pff.rolling(30).mean()
    X["gas_lag1"] = F["gas"].shift(1)
    X["gas_roll7"] = F["gas"].shift(1).rolling(7).mean()
    # Short USGS gaps (<=14 d, e.g. provisional-data lags and the Oct-Nov 2025
    # federal-shutdown gap at some gauges) are linearly interpolated for the
    # FEATURE windows only; the price target series is never interpolated.
    # Without this, one missing flow day poisons the trailing-365d mean for a
    # full year of origins.
    fl = F["flow"].interpolate(limit=14)
    X["flow7"] = fl.rolling(7, min_periods=5).mean()
    flow30 = fl.rolling(30, min_periods=25).mean()
    flow365 = fl.rolling(365, min_periods=330).mean()
    X["flow_anom"] = flow30 / flow365 - 1.0
    X["soil"] = F["soil"].ffill(limit=5)
    X["soil7"] = F["soil"].rolling(7).mean()
    X["precip7"] = F["precip"].rolling(7).sum()
    X["temp7"] = F["temp"].rolling(7).mean()
    X["snow7"] = F["snowfall"].rolling(7).sum()
    if has_snow:
        X["swe"] = F["swe"].ffill(limit=5)
    dow = F.index.dayofweek.values
    doy = F.index.dayofyear.values
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7); X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return X

def slog(p):
    return np.sign(p) * np.log1p(np.abs(p))

def slog_inv(y):
    return np.sign(y) * np.expm1(np.abs(y))

# ---------------------------------------------------------------- training
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def evaluate(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if len(yt) < 10:
        return dict(n=int(len(yt)), mae=np.nan, rmse=np.nan, r2=np.nan)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - np.sum((yt - yp) ** 2) / ss) if ss > 0 else np.nan
    return dict(n=int(len(yt)), mae=mae, rmse=rmse, r2=r2)

def _fit_candidate(kind, Z, d):
    if kind == "ridge":
        return ("ridge", Ridge(alpha=10.0).fit(Z, d))
    mdl = MLPRegressor(hidden_layer_sizes=(48, 24), activation="relu", alpha=0.1,
                       learning_rate_init=0.003, max_iter=800, random_state=7)
    mdl.fit(Z, d)
    return ("mlp", mdl)

def _export_candidate(model, scaler, clip, base_slog):
    kind, m = model
    w = dict(type=kind, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
             clip=[float(clip[0]), float(clip[1])], base_slog=float(base_slog))
    if kind == "ridge":
        w["coef"] = m.coef_.tolist()
        w["intercept"] = float(m.intercept_)
    else:
        w["coefs"] = [c.tolist() for c in m.coefs_]
        w["intercepts"] = [b.tolist() for b in m.intercepts_]
    return w

def train_region(rid, cfg, F):
    """Delta-target forecaster. The model predicts the signed-log price CHANGE
    from the last observed price, so persistence (zero change) is the anchor
    the features must improve on. Per horizon, ridge vs MLP is selected on a
    validation slice (the last 20% of the training period); the winner is
    refit on the full training period and scored once on the time-based
    holdout (last 20% of origins, no shuffling). Predictions are clipped to
    the training period's observed price range (maximum extended 25%)."""
    has_snow = bool(cfg["snotel"])
    X = make_features(F, has_snow)
    origins = X.loc[ORIGIN_START:END].dropna()
    # keep origins whose +7d window exists in frame
    origins = origins[origins.index + pd.Timedelta(days=7) <= F.index.max()]
    feats = list(X.columns)
    split = origins.index[int(len(origins) * 0.8)]
    train_idx = origins.index[origins.index < split]
    hold_idx = origins.index[origins.index >= split]
    price = F["price"]
    result = dict(features=feats, n_origins=int(len(origins)),
                  train_origins=int(len(train_idx)), holdout_origins=int(len(hold_idx)),
                  holdout_start=str(split.date()), horizons={})
    Xv = origins.values
    p_last_all = origins["p_last"].values
    pos = {d: i for i, d in enumerate(origins.index)}
    final_models = {}
    for h in range(1, 8):
        y_all = price.shift(-h).reindex(origins.index)
        yv = y_all.values
        delta_all = slog(yv) - slog(p_last_all)
        ok = y_all.notna().values
        tr_pos = np.array([pos[d] for d in train_idx if ok[pos[d]]])
        ho_pos = np.array([pos[d] for d in hold_idx if ok[pos[d]]])
        n_val = max(int(len(tr_pos) * 0.2), 30)
        fit_pos, val_pos = tr_pos[:-n_val], tr_pos[-n_val:]
        clip = (float(np.min(yv[tr_pos])), float(np.max(yv[tr_pos])) * 1.25)

        def to_price(d_pred, pl, clip=clip):
            return np.clip(slog_inv(slog(pl) + d_pred), clip[0], clip[1])

        sc_fit = StandardScaler().fit(Xv[fit_pos])
        val_mae = {}
        for kind in ("ridge", "mlp"):
            mdl = _fit_candidate(kind, sc_fit.transform(Xv[fit_pos]), delta_all[fit_pos])
            pv = to_price(mdl[1].predict(sc_fit.transform(Xv[val_pos])), p_last_all[val_pos])
            val_mae[kind] = float(np.mean(np.abs(yv[val_pos] - pv)))
        chosen = min(val_mae, key=val_mae.get)
        sc_tr = StandardScaler().fit(Xv[tr_pos])
        evs = {}
        for kind in ("ridge", "mlp"):
            mdl = _fit_candidate(kind, sc_tr.transform(Xv[tr_pos]), delta_all[tr_pos])
            ph = to_price(mdl[1].predict(sc_tr.transform(Xv[ho_pos])), p_last_all[ho_pos])
            evs[kind] = evaluate(yv[ho_pos], ph)
        ev = evs[chosen]
        pers = evaluate(yv[ho_pos], p_last_all[ho_pos])
        mdl_tr = _fit_candidate(chosen, sc_tr.transform(Xv[tr_pos]), delta_all[tr_pos])
        pred_ho = to_price(mdl_tr[1].predict(sc_tr.transform(Xv[ho_pos])), p_last_all[ho_pos])
        resid = yv[ho_pos] - pred_ho
        resid = resid[np.isfinite(resid)]
        sigma = float(np.std(resid)) if len(resid) else np.nan
        skill = (1 - ev["mae"] / pers["mae"]) * 100 if pers["mae"] else np.nan
        result["horizons"][h] = dict(
            model=ev, persistence=pers, ridge=evs["ridge"], skill_pct=skill,
            resid_sigma=sigma, chosen=chosen,
            val_mae={k: round(v, 2) for k, v in val_mae.items()})
        # final model on all origins with this horizon's target present
        all_pos = np.where(ok)[0]
        clip2 = (float(np.min(yv[all_pos])), float(np.max(yv[all_pos])) * 1.25)
        sc2 = StandardScaler().fit(Xv[all_pos])
        mfin = _fit_candidate(chosen, sc2.transform(Xv[all_pos]), delta_all[all_pos])
        final_models[h] = (sc2, mfin, clip2)
    # forecast from the last origin with complete features
    last_origin = origins.index.max()
    xrow = origins.loc[[last_origin]]
    base_slog = float(slog(xrow["p_last"].iloc[0]))
    fc = []
    weights = {}
    for h in range(1, 8):
        sc, mfin, clip2 = final_models[h]
        d_pred = float(mfin[1].predict(sc.transform(xrow.values))[0])
        pred = float(np.clip(slog_inv(base_slog + d_pred), clip2[0], clip2[1]))
        sig = result["horizons"][h]["resid_sigma"]
        d = last_origin + pd.Timedelta(days=h)
        fc.append(dict(date=str(d.date()), pred=round(pred, 2),
                       lo=round(pred - 1.2816 * sig, 2), hi=round(pred + 1.2816 * sig, 2)))
        weights[h] = _export_candidate(mfin, sc, clip2, base_slog)
    result["forecast_origin"] = str(last_origin.date())
    result["forecast"] = fc
    result["weights"] = weights
    result["chosen"] = {h: result["horizons"][h]["chosen"] for h in range(1, 8)}
    result["features_last"] = {k: float(xrow[k].iloc[0]) for k in feats}
    return result, X

# ---------------------------------------------------------------- export
def spark(s, n=120):
    s = s.dropna().tail(n)
    return [dict(d=str(i.date()), v=round(float(v), 3)) for i, v in s.items()]

def main():
    offline = "--offline" in sys.argv
    if offline:
        print("offline mode: using data/raw caches")
    print("fetching ICE prices + Henry Hub ...")
    ice = fetch_ice_prices()
    gas = fetch_henry_hub()
    export = dict(meta=dict(
        title="HydroTwin Energy — Wholesale Price Forecasting Twin",
        snapshot_date=RETRIEVAL_DATE,
        data_window=f"{FETCH_START} .. {END}",
        note=("Static snapshot built " + RETRIEVAL_DATE + " by pipeline/pipeline.py. "
              "Not live data. Sibling of HydroTwin (river twin); this twin forecasts energy prices.")),
        regions=[])
    report = {}
    for rid, cfg in REGIONS.items():
        print(f"\n=== region {rid}: {cfg['name']} ===")
        F, gauge_names = build_region_frame(cfg, ice, gas)
        res, X = train_region(rid, cfg, F)
        lo = F["price"].first_valid_index(); hi = F["price"].last_valid_index()
        price_rows = int(F["price"].notna().sum())
        last_price_date = str(hi.date()); last_price = float(F["price"].loc[hi])
        o = pd.Timestamp(res["forecast_origin"])
        flow_disp = F["flow"].interpolate(limit=14).ffill(limit=3)
        latest = dict(
            last_price=round(last_price, 2), last_price_date=last_price_date,
            flow_cfs=round(float(flow_disp.loc[o]), 1),
            flow_anom_pct=round(float(X["flow_anom"].loc[o]) * 100, 1),
            soil_pct=round(float(F["soil"].ffill(limit=5).loc[o]), 1),
            gas_usd=round(float(F["gas"].loc[o]), 3),
            precip7_mm=round(float(X["precip7"].loc[o]), 1),
            temp7_c=round(float(X["temp7"].loc[o]), 1),
        )
        if cfg["snotel"]:
            latest["swe_in"] = round(float(F["swe"].ffill(limit=5).loc[o]), 2)
        hist = F["price"].dropna().tail(420)
        region_export = dict(
            id=rid, name=cfg["name"], hub=cfg["hub_label"],
            price_rows=price_rows,
            price_range=[str(lo.date()), str(hi.date())],
            history=[dict(d=str(i.date()), v=round(float(v), 2)) for i, v in hist.items()],
            latest=latest,
            sparks=dict(flow=spark(F["flow"]), swe=spark(F["swe"]) if cfg["snotel"] else [],
                        soil=spark(F["soil"]), gas=spark(F["gas"])),
            forecast_origin=res["forecast_origin"], forecast=res["forecast"],
            features=res["features"], features_last=res["features_last"],
            metrics={str(h): dict(
                mae=round(res["horizons"][h]["model"]["mae"], 2),
                rmse=round(res["horizons"][h]["model"]["rmse"], 2),
                r2=round(res["horizons"][h]["model"]["r2"], 3),
                n=res["horizons"][h]["model"]["n"],
                pers_mae=round(res["horizons"][h]["persistence"]["mae"], 2),
                pers_rmse=round(res["horizons"][h]["persistence"]["rmse"], 2),
                skill_pct=round(res["horizons"][h]["skill_pct"], 1),
                ridge_mae=round(res["horizons"][h]["ridge"]["mae"], 2),
                chosen=res["horizons"][h]["chosen"],
                val_mae=res["horizons"][h]["val_mae"],
                resid_sigma=round(res["horizons"][h]["resid_sigma"], 2))
                for h in range(1, 8)},
            n_origins=res["n_origins"], train_origins=res["train_origins"],
            holdout_origins=res["holdout_origins"], holdout_start=res["holdout_start"],
            stations=dict(usgs=gauge_names,
                          snotel=cfg["snotel"], scan=cfg["scan"],
                          weather=cfg["weather_names"]),
            weights={str(h): res["weights"][h] for h in range(1, 8)},
            limitations=cfg["extra_limitations"],
        )
        export["regions"].append(region_export)
        report[rid] = dict(cfg_name=cfg["name"], hub=cfg["hub_label"],
                           price_rows=price_rows, price_range=region_export["price_range"],
                           metrics=region_export["metrics"], n_origins=res["n_origins"],
                           holdout_start=res["holdout_start"],
                           forecast_origin=res["forecast_origin"],
                           forecast=res["forecast"], latest=latest,
                           features=res["features"])
        print(f"  price rows={price_rows} range={region_export['price_range']} "
              f"origins={res['n_origins']} holdout_from={res['holdout_start']}")
        for h in (1, 7):
            m = region_export["metrics"][str(h)]
            print(f"  h={h}: MAE {m['mae']} vs persistence {m['pers_mae']} "
                  f"(skill {m['skill_pct']}%), R2 {m['r2']}, n={m['n']}")
    (SITE / "data.js").write_text("window.HTE_DATA = " + json.dumps(export) + ";\n")
    (ROOT / "data" / "metrics.json").write_text(json.dumps(report, indent=1))
    print("\nwrote site/data.js and data/metrics.json")

if __name__ == "__main__":
    main()
