#!/usr/bin/env python3
"""
HydroTwin Energy — data + training pipeline.

Fetches genuine public data (no API keys), builds daily features per price
region, and trains per-horizon price forecasters on the signed-log price
change from the last observed price. The candidate roster is the SAME
roster as the HydroTwin (river) paper — LSTM, physics-informed LSTM,
Transformer, Temporal Fusion Transformer, PatchTST, two Gaussian process
models, a graph convolutional network over the regions, ARIMA and
seasonal ARIMA — plus this twin's original ridge and MLP and the
model-zoo winners Random Forest and MiniRocket, so the two twins are
directly comparable. Per horizon the winner is selected on a validation
slice, refit on the full training period, scored once on a time-based
holdout against a persistence baseline, and the full per-model compare
table is exported with the static site snapshot (site/data.js).

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
END = date.today().isoformat()       # last day in snapshot window (rolls forward daily)
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

# ---------------------------------------------------------------- roster
# The HydroTwin (river) paper roster, ported to the energy twin so both
# twins run the SAME model roster and their results are directly
# comparable. The abstract's ten: LSTM, physics-informed LSTM, Transformer,
# Temporal Fusion Transformer (TFT), PatchTST, two Gaussian process
# models, a graph convolutional network (river: stations linked by
# snow-regime similarity), ARIMA and seasonal ARIMA. The energy twin's
# original candidates (ridge, MLP) stay, and the two model-zoo winners
# (Random Forest, MiniRocket) join them. Per horizon the winner is
# selected on the validation slice (last 20% of the training period) —
# never on the holdout — refit on the full training period and scored
# once on the time-based holdout, exactly the shipped protocol.
#
# Energy adaptations (stated here and on the pages):
#   * PI-LSTM: the energy analogue of "physics-informed" is a
#     driver-equilibrium penalty. A ridge fit on TRAIN data only maps the
#     exogenous drivers (gas, flow, soil, weather, snow, season) to the
#     signed-log price level; the LSTM's forecast level is penalised for
#     drifting from that driver-implied equilibrium (lambda = 0.1). No
#     river-style physics is claimed.
#   * GPs: two kernel designs on the tabular features plus a day-of-year
#     input — GP-A: RBF + periodic (seasonal, on day-of-year); GP-B:
#     Matern + linear (DotProduct).
#   * GCN: nodes are the 3 price REGIONS (the river twin links stations);
#     edge weights come from train-period similarity of the regions'
#     price/feature regimes (correlation of daily signed-log price
#     changes + proximity of standardized train feature means) — the
#     energy analogue of the river twin's snow-regime similarity edges.
#   * Every family is optional at runtime: if a dependency is missing or
#     a fit fails, the family is recorded in the export as failed with
#     the reason instead of crashing the pipeline.
ROSTER_META = [
    dict(key="ridge", name="Ridge", family="Linear",
         note="The energy twin's original baseline: ridge (alpha=10) on the standardized 19 features."),
    dict(key="mlp", name="MLP", family="Neural",
         note="The energy twin's original neural candidate: MLP (48, 24) on the standardized 19 features."),
    dict(key="random_forest", name="Random Forest", family="Tree ensemble",
         note="Model-zoo winner added to the roster: 400 trees, leaf size and feature fraction selected on validation."),
    dict(key="minirocket", name="MiniRocket + ridge", family="Kernel features",
         note="Model-zoo winner added to the roster: MiniRocket features over the 28-day window, per-horizon ridge heads, alpha selected on validation."),
    dict(key="lstm", name="LSTM", family="Sequence",
         note="Abstract roster: LSTM over the 28-day x 19-feature window, direct multi-horizon head, masked loss."),
    dict(key="pi_lstm", name="Physics-informed LSTM", family="Sequence",
         note="Abstract roster, energy adaptation: LSTM plus a driver-equilibrium penalty (lambda=0.1) pulling the forecast level toward the price level implied by the exogenous drivers (gas, flow, soil, weather, snow, season), the equilibrium map fit on train data only."),
    dict(key="transformer", name="Transformer", family="Sequence",
         note="Abstract roster: 2-layer Transformer encoder over the 28-day window with sinusoidal positions, direct multi-horizon head."),
    dict(key="tft", name="Temporal Fusion Transformer", family="Sequence",
         note="Abstract roster: variable-selection weights over the 19 features, LSTM encoder, multi-head temporal attention, direct multi-horizon head (compact implementation)."),
    dict(key="patchtst", name="PatchTST", family="Sequence",
         note="Abstract roster: the 28-day window is cut into 7-day patches, embedded per feature channel, Transformer encoder, direct multi-horizon head."),
    dict(key="gp_rbf_seasonal", name="Gaussian process A", family="Gaussian process",
         note="Abstract roster GP 1: RBF + periodic seasonal kernel (periodic on the day-of-year input) + white noise, on the standardized features (length scales fixed, scale and noise fit by marginal likelihood; train capped at the most recent 600 origins)."),
    dict(key="gp_matern_linear", name="Gaussian process B", family="Gaussian process",
         note="Abstract roster GP 2: Matern (nu=1.5) + linear (DotProduct) kernel + white noise, on the standardized features (length scales fixed, scale and noise fit by marginal likelihood; train capped at the most recent 600 origins)."),
    dict(key="gcn", name="Graph convolutional network", family="Graph",
         note="Abstract roster, energy adaptation: nodes are the 3 price regions; edges are weighted by train-period regime similarity (price-change correlation + feature-mean proximity), the analogue of the river twin's snow-regime station graph. Two graph convolution layers, per-region multi-horizon heads."),
    dict(key="arima", name="ARIMA", family="Classical",
         note="Abstract roster: univariate ARIMA on the daily price series, order selected by AIC on the training period, rolled forward with frozen parameters (refit=False)."),
    dict(key="sarima", name="Seasonal ARIMA", family="Classical",
         note="Abstract roster: univariate SARIMA with a 7-day seasonal period, order selected by AIC on the training period, rolled forward with frozen parameters."),
]
ROSTER_KEYS = [m["key"] for m in ROSTER_META]
SEED = 7
WINDOW = 28            # lookback days for the window models (zoo design)
GP_MAX_TRAIN = 600     # GP fit cap: most recent train origins (stated above)
PI_LAMBDA = 0.1        # driver-equilibrium penalty weight for the PI-LSTM
DRIVER_FEATS = ["gas_lag1", "gas_roll7", "flow7", "flow_anom", "soil",
                "soil7", "precip7", "temp7", "snow7", "swe",
                "doy_sin", "doy_cos"]


def prepare_region(rid, cfg, F):
    """Build the exact data/split structure the shipped train_region used
    (same origins, same 80/20 time split, same per-horizon fit/validation
    split, same clips), packaged for the roster runners."""
    has_snow = bool(cfg["snotel"])
    X = make_features(F, has_snow)
    origins = X.loc[ORIGIN_START:END].dropna()
    origins = origins[origins.index + pd.Timedelta(days=7) <= F.index.max()]
    feats = list(X.columns)
    split = origins.index[int(len(origins) * 0.8)]
    train_idx = origins.index[origins.index < split]
    hold_idx = origins.index[origins.index >= split]
    price = F["price"]
    Xv = origins.values
    p_last_all = origins["p_last"].values
    pos = {d: i for i, d in enumerate(origins.index)}
    D = dict(rid=rid, cfg=cfg, F=F, X=X, origins=origins, feats=feats,
             split=split, train_idx=train_idx, hold_idx=hold_idx, pos=pos,
             Xv=Xv, p_last=p_last_all, horizons={}, roster=None,
             forecast_delta={}, forecasters={})
    for h in range(1, 8):
        y_all = price.shift(-h).reindex(origins.index)
        yv = y_all.values
        delta_all = slog(yv) - slog(p_last_all)
        ok = y_all.notna().values
        tr_pos = np.array([pos[d] for d in train_idx if ok[pos[d]]])
        ho_pos = np.array([pos[d] for d in hold_idx if ok[pos[d]]])
        n_val = max(int(len(tr_pos) * 0.2), 30)
        D["horizons"][h] = dict(
            yv=yv, delta=delta_all, ok=ok, tr_pos=tr_pos, ho_pos=ho_pos,
            fit_pos=tr_pos[:-n_val], val_pos=tr_pos[-n_val:],
            clip=(float(np.min(yv[tr_pos])), float(np.max(yv[tr_pos])) * 1.25))
    return D


def build_windows(D):
    """(n_origins, WINDOW, n_feats) windows ending at each origin + the
    multi-horizon delta targets (zoo design). Window NaNs are filled with
    per-feature medians over TRAIN-origin windows."""
    X = D["X"]
    dates = D["origins"].index
    W = np.full((len(dates), WINDOW, X.shape[1]), np.nan)
    for i, d in enumerate(dates):
        seg = X.loc[d - pd.Timedelta(days=WINDOW - 1): d]
        if len(seg) == WINDOW:
            W[i] = seg.values
    tr_mask = np.array([d in set(D["train_idx"]) for d in dates])
    med = np.nanmedian(W[tr_mask].reshape(-1, X.shape[1]), axis=0)
    inds = np.where(~np.isfinite(W))
    W[inds] = np.take(med, inds[2])
    T = np.full((len(dates), 7), np.nan)
    for h in range(1, 8):
        T[:, h - 1] = D["horizons"][h]["delta"]
    return W, T


def seq_positions(D, T):
    """Train-subset positions with >=1 finite horizon target, split
    fit/validation by the same last-20% (min 30) rule; holdout positions =
    all holdout origins (zoo design)."""
    pos = D["pos"]
    tr_all = np.array([pos[d] for d in D["train_idx"]])
    ho_all = np.array([pos[d] for d in D["hold_idx"]])
    tr_ok = tr_all[np.isfinite(T[tr_all]).any(axis=1)]
    n_val = max(int(len(tr_ok) * 0.2), 30)
    return tr_ok[:-n_val], tr_ok[-n_val:], tr_ok, ho_all


def _to_price(D, h, d_pred, positions):
    clip = D["horizons"][h]["clip"]
    pl = D["p_last"][positions]
    return np.clip(slog_inv(slog(pl) + d_pred), clip[0], clip[1])


def _fail(D, keys, reason):
    for h in range(1, 8):
        for k in keys:
            D["roster"][h][k] = dict(status="failed", reason=str(reason)[:200],
                                     val_mae=None, pred=None)


def _cap_rows(pos_arr, cap):
    return pos_arr[-cap:] if len(pos_arr) > cap else pos_arr


# ------------------------------------------------------- tabular families
def _roster_tabular(D):
    """ridge + MLP (the pipeline's original procedure, unchanged), Random
    Forest (zoo configs, validation-selected) and the two GPs."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (RBF, ExpSineSquared,
        WhiteKernel, Matern, DotProduct, ConstantKernel)
    from sklearn.gaussian_process.kernels import Kernel as _GPKernel

    class _RbfPlusDoyPeriodic(_GPKernel):
        """GP-A kernel: anisotropic RBF over all inputs (length scales
        fixed at 1.0, and 10.0 on the appended day-of-year column) PLUS
        a periodic kernel on the day-of-year column alone (period one
        year, circular distance so late December is close to early
        January, length scale 0.3). Written directly because sklearn's
        ExpSineSquared only supports an isotropic length scale over
        ALL inputs, which cannot express a seasonal term on one input.
        No trainable hyperparameters of its own; the output scale and
        noise level are fit through the ConstantKernel and WhiteKernel
        it is combined with."""

        def __init__(self):
            pass

        def __call__(self, X, Y=None, eval_gradient=False):
            X = np.atleast_2d(np.asarray(X, dtype=float))
            Y = X if Y is None else np.atleast_2d(np.asarray(Y, dtype=float))
            ls = np.ones(X.shape[1])
            ls[-1] = 10.0
            Xs, Ys = X / ls, Y / ls
            d2 = (np.sum(Xs ** 2, 1)[:, None] + np.sum(Ys ** 2, 1)[None, :]
                  - 2.0 * Xs @ Ys.T)
            K = np.exp(-0.5 * np.maximum(d2, 0.0))
            d = np.abs(X[:, -1][:, None] - Y[:, -1][None, :])
            d = np.minimum(d, 1.0 - d)
            K = K + np.exp(-2.0 * (np.sin(np.pi * d) / 0.3) ** 2)
            if eval_gradient:
                return K, np.empty((X.shape[0], Y.shape[0], 0))
            return K

        def diag(self, X):
            return np.full(np.atleast_2d(X).shape[0], 2.0)

        @property
        def is_stationary(self):
            return True
    Xv = D["Xv"]
    doy_frac = ((D["origins"].index.dayofyear.values - 1) / 365.25).reshape(-1, 1)
    rf_cfgs = {
        "rf_leaf2": dict(n_estimators=400, min_samples_leaf=2, max_features=0.7),
        "rf_leaf8": dict(n_estimators=400, min_samples_leaf=8, max_features=0.5),
    }

    def gp_kernel(kind, n_feat):
        # Length scales are FIXED at the stated values and only the
        # output scale and noise level are fit by marginal likelihood —
        # with ~20 inputs and up to 600 fit rows, optimising every length
        # scale makes a single GP fit take minutes, which a weekly
        # retrain cannot afford; the kernel DESIGNS (the roster point)
        # are unchanged.
        if kind == "gp_rbf_seasonal":
            return (ConstantKernel(1.0, (1e-2, 1e2)) *
                    _RbfPlusDoyPeriodic() +
                    WhiteKernel(0.1, (1e-3, 10.0)))
        return (ConstantKernel(1.0, (1e-2, 1e2)) *
                (Matern([1.0] * n_feat + [10.0], "fixed", nu=1.5) +
                 DotProduct(1.0, "fixed")) +
                WhiteKernel(0.1, (1e-3, 10.0)))

    def gp_fit_predict(kind, Z_fit, y_fit, Z_pred):
        g = GaussianProcessRegressor(kernel=gp_kernel(kind, Z_fit.shape[1] - 1),
                                     alpha=1e-6, normalize_y=True,
                                     n_restarts_optimizer=0, random_state=SEED)
        g.fit(Z_fit, y_fit)
        return g.predict(Z_pred)

    for h in range(1, 8):
        hd = D["horizons"][h]
        delta, yv = hd["delta"], hd["yv"]

        def to_price(dp, positions, h=h):
            return _to_price(D, h, dp, positions)

        # --- ridge & MLP: the shipped procedure, unchanged ---
        sc_fit = StandardScaler().fit(Xv[hd["fit_pos"]])
        sc_tr = StandardScaler().fit(Xv[hd["tr_pos"]])
        for kind in ("ridge", "mlp"):
            mdl = _fit_candidate(kind, sc_fit.transform(Xv[hd["fit_pos"]]),
                                 delta[hd["fit_pos"]])
            pv = to_price(mdl[1].predict(sc_fit.transform(Xv[hd["val_pos"]])),
                          hd["val_pos"])
            mdl_tr = _fit_candidate(kind, sc_tr.transform(Xv[hd["tr_pos"]]),
                                    delta[hd["tr_pos"]])
            ph = to_price(mdl_tr[1].predict(sc_tr.transform(Xv[hd["ho_pos"]])),
                          hd["ho_pos"])
            D["roster"][h][kind] = dict(
                status="ok",
                val_mae=float(np.mean(np.abs(yv[hd["val_pos"]] - pv))),
                pred=ph)
        # --- Random Forest: config selected on the validation slice ---
        try:
            val_scores, fitted = {}, {}
            for cname, c in rf_cfgs.items():
                m = RandomForestRegressor(n_jobs=2, random_state=SEED, **c)
                m.fit(Xv[hd["fit_pos"]], delta[hd["fit_pos"]])
                pv = to_price(m.predict(Xv[hd["val_pos"]]), hd["val_pos"])
                val_scores[cname] = float(np.mean(np.abs(yv[hd["val_pos"]] - pv)))
                fitted[cname] = m
            best_cfg = min(val_scores, key=val_scores.get)
            m_tr = RandomForestRegressor(n_jobs=2, random_state=SEED,
                                         **rf_cfgs[best_cfg])
            m_tr.fit(Xv[hd["tr_pos"]], delta[hd["tr_pos"]])
            ph = to_price(m_tr.predict(Xv[hd["ho_pos"]]), hd["ho_pos"])
            D["roster"][h]["random_forest"] = dict(
                status="ok", val_mae=val_scores[best_cfg], pred=ph,
                detail=f"config {best_cfg}")
            D.setdefault("rf_best_cfg", {})[h] = best_cfg

            if "random_forest" not in D["forecasters"]:
                def _rf_forecast(h):
                    hd2 = D["horizons"][h]
                    cfg_name = D["rf_best_cfg"][h]
                    all_pos = np.where(hd2["ok"])[0]
                    mall = RandomForestRegressor(
                        n_jobs=2, random_state=SEED, **rf_cfgs[cfg_name])
                    mall.fit(Xv[all_pos], hd2["delta"][all_pos])
                    return float(mall.predict(Xv[[-1]])[0])
                D["forecasters"]["random_forest"] = _rf_forecast
        except Exception as e:
            D["roster"][h]["random_forest"] = dict(
                status="failed", reason=str(e)[:200], val_mae=None, pred=None)
        # --- the two Gaussian processes ---
        for kind in ("gp_rbf_seasonal", "gp_matern_linear"):
            try:
                zf = np.hstack([sc_fit.transform(Xv[hd["fit_pos"]]),
                                doy_frac[hd["fit_pos"]]])
                zv = np.hstack([sc_fit.transform(Xv[hd["val_pos"]]),
                                doy_frac[hd["val_pos"]]])
                fp = _cap_rows(np.arange(len(hd["fit_pos"])), GP_MAX_TRAIN)
                pv_d = gp_fit_predict(kind, zf[fp], delta[hd["fit_pos"]][fp], zv)
                pv = to_price(pv_d, hd["val_pos"])
                zt = np.hstack([sc_tr.transform(Xv[hd["tr_pos"]]),
                                doy_frac[hd["tr_pos"]]])
                zh = np.hstack([sc_tr.transform(Xv[hd["ho_pos"]]),
                                doy_frac[hd["ho_pos"]]])
                tp = _cap_rows(np.arange(len(hd["tr_pos"])), GP_MAX_TRAIN)
                ph_d = gp_fit_predict(kind, zt[tp], delta[hd["tr_pos"]][tp], zh)
                ph = to_price(ph_d, hd["ho_pos"])
                D["roster"][h][kind] = dict(
                    status="ok",
                    val_mae=float(np.mean(np.abs(yv[hd["val_pos"]] - pv))),
                    pred=ph)

                def _gp_forecast(h=h, kind=kind):
                    hd2 = D["horizons"][h]
                    all_pos = np.where(hd2["ok"])[0]
                    sc = StandardScaler().fit(Xv[all_pos])
                    za = np.hstack([sc.transform(Xv[all_pos]), doy_frac[all_pos]])
                    ap = _cap_rows(np.arange(len(all_pos)), GP_MAX_TRAIN)
                    zl = np.hstack([sc.transform(Xv[[-1]]), doy_frac[[-1]]])
                    return float(gp_fit_predict(kind, za[ap],
                                                hd2["delta"][all_pos][ap], zl)[0])
                D["forecasters"].setdefault(kind, _gp_forecast)
            except Exception as e:
                D["roster"][h][kind] = dict(
                    status="failed", reason=str(e)[:200], val_mae=None, pred=None)


# ------------------------------------------------------- MiniRocket
def _roster_rocket(D):
    """MiniRocket features over the 28-day window (fit once), per-horizon
    ridge heads; alpha selected on validation by mean price MAE (zoo)."""
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    W, T = D["W"], D["T"]
    fit_pos, val_pos, tr_pos, ho_pos = seq_positions(D, T)
    rk = MiniRocketMultivariate(num_kernels=5000, n_jobs=2, random_state=SEED)
    F_fit = rk.fit_transform(np.transpose(W[fit_pos], (0, 2, 1)))
    F_val = rk.transform(np.transpose(W[val_pos], (0, 2, 1)))
    F_tr = rk.transform(np.transpose(W[tr_pos], (0, 2, 1)))
    F_ho = rk.transform(np.transpose(W[ho_pos], (0, 2, 1)))
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    T_fit, T_val, T_tr = T[fit_pos], T[val_pos], T[tr_pos]
    hold_list = list(D["hold_idx"])
    hold_pos_of = {p: i for i, p in enumerate(ho_pos)}

    def price_mae(h, dp, positions, src_pos):
        hd = D["horizons"][h]
        pv = _to_price(D, h, dp, positions)
        m = np.isfinite(hd["yv"][positions])
        return float(np.mean(np.abs(hd["yv"][positions][m] - pv[m])))

    val_scores = {}
    for a in alphas:
        sc = StandardScaler().fit(F_fit)
        maes = []
        for h in range(1, 8):
            okf = np.isfinite(T_fit[:, h - 1])
            m = Ridge(alpha=a).fit(sc.transform(F_fit[okf]), T_fit[okf, h - 1])
            maes.append(price_mae(h, m.predict(sc.transform(F_val)),
                                  val_pos, val_pos))
        val_scores[a] = float(np.mean(maes))
    best_a = min(val_scores, key=val_scores.get)
    # per-horizon validation MAE at the chosen alpha (fit-slice heads)
    sc_f = StandardScaler().fit(F_fit)
    sc_t = StandardScaler().fit(F_tr)
    for h in range(1, 8):
        hd = D["horizons"][h]
        okf = np.isfinite(T_fit[:, h - 1])
        mf = Ridge(alpha=best_a).fit(sc_f.transform(F_fit[okf]),
                                     T_fit[okf, h - 1])
        vm = price_mae(h, mf.predict(sc_f.transform(F_val)), val_pos, val_pos)
        okt = np.isfinite(T_tr[:, h - 1])
        mt = Ridge(alpha=best_a).fit(sc_t.transform(F_tr[okt]),
                                     T_tr[okt, h - 1])
        dp_ho = mt.predict(sc_t.transform(F_ho))
        ph_all = _to_price(D, h, dp_ho, ho_pos)
        idx_in_hold = np.array([hold_pos_of[p] for p in hd["ho_pos"]])
        D["roster"][h]["minirocket"] = dict(
            status="ok", val_mae=vm, pred=ph_all[idx_in_hold],
            detail=f"alpha {best_a:g}")

    def _rocket_forecast(h):
        all_pos = np.array([p for p in range(len(T))
                            if np.isfinite(T[p, h - 1])])
        F_all = rk.transform(np.transpose(W[all_pos], (0, 2, 1)))
        sc = StandardScaler().fit(F_all)
        m = Ridge(alpha=best_a).fit(sc.transform(F_all), T[all_pos, h - 1])
        F_last = rk.transform(np.transpose(W[[-1]], (0, 2, 1)))
        return float(m.predict(sc.transform(F_last))[0])
    D["forecasters"]["minirocket"] = _rocket_forecast


# ------------------------------------------------------- torch families
def _seq_nets(family, n_feat, DM):
    import torch
    import torch.nn as nn

    def posenc(L, DM):
        pe = torch.zeros(L, DM)
        posv = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, DM, 2).float() *
                        (-np.log(10000.0) / DM))
        pe[:, 0::2] = torch.sin(posv * div)
        pe[:, 1::2] = torch.cos(posv * div)
        return pe

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.LSTM(n_feat, DM, batch_first=True)
            self.head = nn.Linear(DM, 7)

        def forward(self, x):
            _, (hn, _) = self.body(x)
            return self.head(hn[-1])

    class TransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(n_feat, DM)
            enc = nn.TransformerEncoderLayer(d_model=DM, nhead=4,
                                             dim_feedforward=96, dropout=0.1,
                                             batch_first=True)
            self.body = nn.TransformerEncoder(enc, num_layers=2)
            self.register_buffer("pe", posenc(64, DM))
            self.head = nn.Linear(DM, 7)

        def forward(self, x):
            z = self.body(self.proj(x) + self.pe[:x.shape[1]])
            return self.head(z.mean(dim=1))

    class TFTNet(nn.Module):
        """Compact TFT: variable-selection weights over the input
        features at each step, LSTM encoder over the selected mixture,
        multi-head temporal attention, multi-horizon head."""
        def __init__(self):
            super().__init__()
            self.vsn_score = nn.Linear(n_feat, n_feat)
            self.var_embed = nn.Linear(1, DM)
            self.lstm = nn.LSTM(DM, DM, batch_first=True)
            self.attn = nn.MultiheadAttention(DM, 4, batch_first=True)
            self.head = nn.Linear(DM, 7)

        def forward(self, x):
            w = torch.softmax(self.vsn_score(x), dim=-1)       # (B,T,F)
            emb = self.var_embed(x.unsqueeze(-1))              # (B,T,F,DM)
            z = (emb * w.unsqueeze(-1)).sum(dim=2)             # (B,T,DM)
            enc, _ = self.lstm(z)
            a, _ = self.attn(enc, enc, enc)
            return self.head(a.mean(dim=1))

    class PatchTSTNet(nn.Module):
        """PatchTST: 7-day patches of the 28-day window, embedded per
        feature channel, shared Transformer encoder, pooled head."""
        def __init__(self, patch=7):
            super().__init__()
            self.patch = patch
            self.embed = nn.Linear(patch, DM)
            enc = nn.TransformerEncoderLayer(d_model=DM, nhead=4,
                                             dim_feedforward=96, dropout=0.1,
                                             batch_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=2)
            self.register_buffer("pe", posenc(16, DM))
            self.head = nn.Linear(DM, 7)

        def forward(self, x):
            B, Tlen, F = x.shape
            P = Tlen // self.patch
            xp = x.reshape(B, P, self.patch, F).permute(0, 3, 1, 2)
            xp = xp.reshape(B * F, P, self.patch)
            z = self.embed(xp) + self.pe[:P]
            z = self.enc(z).mean(dim=1).reshape(B, F, -1).mean(dim=1)
            return self.head(z)

    return {"lstm": LSTMNet, "pi_lstm": LSTMNet,
            "transformer": TransformerNet, "tft": TFTNet,
            "patchtst": PatchTSTNet}[family]()


def _roster_seq(D, family):
    """Direct multi-horizon window models with the zoo's two-phase
    protocol: phase 1 fits on the fit slice with early stopping on
    validation price MAE; phase 2 refits on the full training subset for
    the best epoch count; the issued-forecast refit (phase 3, on all
    origins) happens lazily, only if the family wins a horizon.
    PI-LSTM adds the driver-equilibrium penalty (see ROSTER_META)."""
    import torch
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    W, T = D["W"], D["T"]
    n_feat = W.shape[2]
    DM = 48
    fit_pos, val_pos, tr_pos, ho_pos = seq_positions(D, T)
    sc = StandardScaler().fit(W[fit_pos].reshape(-1, n_feat))

    def tf(P):
        return torch.tensor(
            sc.transform(W[P].reshape(-1, n_feat)).reshape(len(P), WINDOW, -1),
            dtype=torch.float32)

    def masked_mse(out, y, m):
        return ((out - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)

    # driver-equilibrium map for the PI-LSTM: ridge from the exogenous
    # driver features to the signed-log price LEVEL, fit per phase on
    # that phase's rows only (train data only, never holdout).
    eq_idx = [D["feats"].index(f) for f in DRIVER_FEATS if f in D["feats"]]

    def eq_slog(rows_fit, rows_pred):
        m = Ridge(alpha=10.0).fit(D["Xv"][rows_fit][:, eq_idx],
                                  slog(D["p_last"][rows_fit]))
        return m.predict(D["Xv"][rows_pred][:, eq_idx])

    def train_model(P, epochs, eq_fit_rows=None):
        model = _seq_nets(family, n_feat, DM)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        XP = tf(P)
        yP = torch.tensor(np.nan_to_num(T[P]), dtype=torch.float32)
        mP = torch.tensor(np.isfinite(T[P]), dtype=torch.float32)
        if family == "pi_lstm":
            baseP = torch.tensor(slog(D["p_last"][P]), dtype=torch.float32)
            eqP = torch.tensor(eq_slog(eq_fit_rows, P), dtype=torch.float32)
        n = len(P)
        bs = 64
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                out = model(XP[idx])
                loss = masked_mse(out, yP[idx], mP[idx])
                if family == "pi_lstm":
                    lvl = baseP[idx].unsqueeze(1) + out
                    pen = ((lvl - eqP[idx].unsqueeze(1)) ** 2 *
                           mP[idx]).sum() / mP[idx].sum().clamp(min=1.0)
                    loss = loss + PI_LAMBDA * pen
                loss.backward()
                opt.step()
        return model

    def predict(model, P):
        model.eval()
        with torch.no_grad():
            return model(tf(P)).numpy()

    def per_h_val_mae(model):
        dp = predict(model, val_pos)
        out = {}
        for h in range(1, 8):
            hd = D["horizons"][h]
            pv = _to_price(D, h, dp[:, h - 1], val_pos)
            m = np.isfinite(hd["yv"][val_pos])
            out[h] = float(np.mean(np.abs(hd["yv"][val_pos][m] - pv[m])))
        return out

    # phase 1: early stopping on mean validation price MAE
    model = _seq_nets(family, n_feat, DM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xfit = tf(fit_pos)
    yfit = torch.tensor(np.nan_to_num(T[fit_pos]), dtype=torch.float32)
    mfit = torch.tensor(np.isfinite(T[fit_pos]), dtype=torch.float32)
    if family == "pi_lstm":
        baseF = torch.tensor(slog(D["p_last"][fit_pos]), dtype=torch.float32)
        eqF = torch.tensor(eq_slog(fit_pos, fit_pos), dtype=torch.float32)
    n = len(fit_pos)
    bs = 64
    best_mae, best_epoch, wait = np.inf, 0, 0
    for ep_i in range(40):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xfit[idx])
            loss = masked_mse(out, yfit[idx], mfit[idx])
            if family == "pi_lstm":
                lvl = baseF[idx].unsqueeze(1) + out
                pen = ((lvl - eqF[idx].unsqueeze(1)) ** 2 *
                       mfit[idx]).sum() / mfit[idx].sum().clamp(min=1.0)
                loss = loss + PI_LAMBDA * pen
            loss.backward()
            opt.step()
        vm = float(np.mean(list(per_h_val_mae(model).values())))
        if vm < best_mae - 1e-3:
            best_mae, best_epoch, wait = vm, ep_i + 1, 0
        else:
            wait += 1
            if wait >= 8:
                break
    val_mae = per_h_val_mae(model)
    # phase 2: refit on the full training subset for best_epoch epochs
    model2 = train_model(tr_pos, max(best_epoch, 5), eq_fit_rows=tr_pos)
    dp_ho = predict(model2, ho_pos)
    hold_pos_of = {p: i for i, p in enumerate(ho_pos)}
    for h in range(1, 8):
        hd = D["horizons"][h]
        ph_all = _to_price(D, h, dp_ho[:, h - 1], ho_pos)
        idx_in_hold = np.array([hold_pos_of[p] for p in hd["ho_pos"]])
        D["roster"][h][family] = dict(
            status="ok", val_mae=val_mae[h], pred=ph_all[idx_in_hold],
            detail=f"best_epoch {best_epoch}")

    _cache = {}

    def _seq_forecast(h):
        if "model" not in _cache:
            all_pos = np.array([p for p in range(len(T))
                                if np.isfinite(T[p]).any()])
            _cache["model"] = train_model(all_pos, max(best_epoch, 5),
                                          eq_fit_rows=all_pos)
        dp = predict(_cache["model"], np.array([len(T) - 1]))
        return float(dp[0, h - 1])
    D["forecasters"][family] = _seq_forecast


# ------------------------------------------------------- ARIMA / SARIMA
def _roster_arima(D):
    """Univariate ARIMA and SARIMA on the daily price series itself
    (the zoo's protocol: raw price, NaN for non-priced days, plain
    statsmodels fits). Within each family the order is selected by AIC
    on the fit period (with the zoo's degenerate-fit guard); the chosen
    spec is scored on the validation slice by rolling the frozen state
    forward (refit=False), then refit on the full training period and
    rolled through the holdout the same operational way. (The zoo
    pooled all eight specs into one AIC competition on the full train
    series and reported a single winner; the roster keeps ARIMA and
    SARIMA as separate entries — the abstract lists both — each
    AIC-selected within its family on the fit slice so that both carry
    a validation MAE for the shared selection rule.)"""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    price = D["F"]["price"]
    h1 = D["horizons"][1]
    fit_cut = D["origins"].index[h1["fit_pos"]].max()
    split = D["split"]
    fams = {
        "arima": [((1, 1, 1), None), ((2, 1, 1), None), ((1, 1, 0), None),
                  ((0, 1, 1), None), ((2, 1, 2), None)],
        "sarima": [((1, 1, 1), (1, 0, 1, 7)), ((1, 1, 1), (0, 0, 1, 7)),
                   ((0, 1, 1), (0, 0, 1, 7))],
    }

    def fit_spec(series, order, seas):
        if seas is None:
            return ARIMA(series, order=order).fit()
        return SARIMAX(series, order=order,
                       seasonal_order=seas).fit(disp=False)

    def roll(res, start_after, upto, want_dates):
        fc_at, cur = {}, res
        days = price.index[(price.index > start_after)
                           & (price.index <= upto)]
        for d in days:
            cur = cur.apply(np.array([price.loc[d]]), refit=False)
            if d in want_dates:
                fc_at[d] = np.asarray(cur.forecast(steps=7), dtype=float)
        return fc_at, cur

    for fam, cands in fams.items():
        try:
            fit_series = price.loc[price.index <= fit_cut]
            pmax = float(np.nanmax(fit_series.values))
            scored = []
            for order, seas in cands:
                try:
                    r = fit_spec(fit_series, order, seas)
                    fc0 = np.asarray(r.forecast(steps=7), dtype=float)
                    if (not np.isfinite(fc0).all()
                            or np.max(np.abs(fc0)) > 10 * pmax):
                        continue
                    scored.append((float(r.aic), order, seas, r))
                except Exception:
                    pass
            if not scored:
                raise RuntimeError("all candidate orders failed or diverged")
            scored.sort(key=lambda t: t[0])
            _aic, order, seas, res_fit = scored[0]
            val_dates = set()
            for h in range(1, 8):
                val_dates |= set(D["origins"].index[D["horizons"][h]["val_pos"]])
            fc_val, _ = roll(res_fit, fit_cut,
                             max(val_dates) if val_dates else fit_cut, val_dates)
            train_series = price.loc[price.index < split]
            res_tr = fit_spec(train_series, order, seas)
            hold_dates = set(D["origins"].index)
            last_origin = D["origins"].index.max()
            fc_ho, _ = roll(res_tr, split - pd.Timedelta(days=1),
                            last_origin, hold_dates)
            spec_name = f"{fam}{order}" + (f"x{seas}" if seas else "")
            fc_last = fc_ho.get(last_origin)
            if fc_last is not None:
                base = slog(float(D["origins"]["p_last"].iloc[-1]))
                D["forecast_delta"][fam] = {
                    h: float(slog(float(np.clip(
                        fc_last[h - 1],
                        D["horizons"][h]["clip"][0],
                        D["horizons"][h]["clip"][1]))) - base)
                    for h in range(1, 8)}
            for h in range(1, 8):
                hd = D["horizons"][h]
                clip = hd["clip"]
                vm_rows, vm_true = [], []
                for p in hd["val_pos"]:
                    d = D["origins"].index[p]
                    if d in fc_val and np.isfinite(hd["yv"][p]):
                        vm_rows.append(float(np.clip(
                            fc_val[d][h - 1], clip[0], clip[1])))
                        vm_true.append(hd["yv"][p])
                val_mae = (float(np.mean(np.abs(np.array(vm_true) -
                                               np.array(vm_rows))))
                           if len(vm_rows) >= 10 else None)
                ph = np.array([float(np.clip(
                    fc_ho[D["origins"].index[p]][h - 1],
                    clip[0], clip[1])) for p in hd["ho_pos"]])
                D["roster"][h][fam] = dict(
                    status="ok", val_mae=val_mae, pred=ph,
                    detail=f"AIC-selected {spec_name}")
        except Exception as e:
            _fail(D, [fam], e)


def run_gcn(Ds):
    """Graph convolutional network over the 3 regions. Node features are
    the origin day's 19 features (standardized per region on its train
    origins); edges are train-period regime similarities (see
    ROSTER_META). Training follows the same two-phase protocol as the
    other window models, with per-region fit/validation/train/holdout
    target masks taken from each region's own splits."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rids = list(Ds.keys())
    R = len(rids)
    # ---- adjacency from train-period regime similarity ----
    def slog_diff(rid):
        p = Ds[rid]["F"]["price"]
        s = slog(p.ffill(limit=4)).diff()
        return s
    mu = {}
    for rid in rids:
        D = Ds[rid]
        tr_rows = np.array([D["pos"][d] for d in D["train_idx"]])
        mu[rid] = np.nanmean(D["Xv"][tr_rows], axis=0)
    A = np.eye(R)
    for i, ri in enumerate(rids):
        for j, rj in enumerate(rids):
            if i == j:
                continue
            si, sj = slog_diff(ri), slog_diff(rj)
            common = si.index.intersection(sj.index)
            common = common[common < min(Ds[ri]["split"], Ds[rj]["split"])]
            a, b = si.loc[common].values, sj.loc[common].values
            m = np.isfinite(a) & np.isfinite(b)
            corr = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else 0.0
            corr = max(corr, 0.0)
            dist = float(np.linalg.norm(mu[ri] - mu[rj]))
            sim = float(np.exp(-(dist ** 2) / (2 * 25.0 ** 2)))
            A[i, j] = 0.5 * corr + 0.5 * sim
    deg = A.sum(axis=1)
    Dm12 = np.diag(deg ** -0.5)
    A_norm = Dm12 @ A @ Dm12
    adj = {rids[i]: {rids[j]: round(float(A[i, j]), 3) for j in range(R)}
           for i in range(R)}
    for rid in rids:
        Ds[rid]["gcn_adjacency"] = adj
    # ---- aligned dataset over the union of origin dates ----
    all_dates = sorted(set().union(*[set(Ds[r]["origins"].index) for r in rids]))
    date_i = {d: i for i, d in enumerate(all_dates)}
    n_feat = len(Ds[rids[0]]["feats"])
    Xg = np.zeros((len(all_dates), R, n_feat))
    Tg = np.full((len(all_dates), R, 7), np.nan)
    masks = {name: np.zeros((len(all_dates), R, 7), dtype=bool)
             for name in ("fit", "val", "train", "hold", "all")}
    for j, rid in enumerate(rids):
        D = Ds[rid]
        tr_rows = np.array([D["pos"][d] for d in D["train_idx"]])
        med = np.nanmedian(D["X"].loc[D["train_idx"]].values, axis=0)
        sc = StandardScaler().fit(D["Xv"][tr_rows])
        Xall = D["X"].copy()
        vals = Xall.values.copy()
        inds = np.where(~np.isfinite(vals))
        vals[inds] = np.take(med, inds[1])
        Xall = pd.DataFrame(vals, index=Xall.index, columns=Xall.columns)
        for d in all_dates:
            if d in Xall.index:
                Xg[date_i[d], j] = sc.transform(
                    Xall.loc[[d]].values)[0]
        for h in range(1, 8):
            hd = D["horizons"][h]
            for name, pos_arr in (("fit", hd["fit_pos"]), ("val", hd["val_pos"]),
                                  ("train", hd["tr_pos"]), ("hold", hd["ho_pos"])):
                for p in pos_arr:
                    d = D["origins"].index[p]
                    Tg[date_i[d], j, h - 1] = hd["delta"][p]
                    masks[name][date_i[d], j, h - 1] = True
            for p in np.where(hd["ok"])[0]:
                d = D["origins"].index[p]
                Tg[date_i[d], j, h - 1] = hd["delta"][p]
                masks["all"][date_i[d], j, h - 1] = True
    DM = 48

    class GCNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("A", torch.tensor(A_norm, dtype=torch.float32))
            self.lin1 = nn.Linear(n_feat, DM)
            self.lin2 = nn.Linear(DM, DM)
            self.head = nn.Linear(DM, 7)

        def forward(self, x):
            h = torch.relu(torch.matmul(self.A, self.lin1(x)))
            h = torch.relu(torch.matmul(self.A, self.lin2(h)))
            return self.head(h)

    Xt = torch.tensor(Xg, dtype=torch.float32)
    Tt = torch.tensor(np.nan_to_num(Tg), dtype=torch.float32)

    def mask_t(name):
        return torch.tensor(masks[name], dtype=torch.float32)

    def train_model(mask_name, epochs):
        model = GCNNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        Mk = mask_t(mask_name)
        rows = np.where(masks[mask_name].any(axis=(1, 2)))[0]
        n = len(rows)
        bs = 64
        for _ in range(epochs):
            model.train()
            perm = rows[torch.randperm(n).numpy()]
            for i in range(0, n, bs):
                idx = torch.tensor(perm[i:i + bs])
                opt.zero_grad()
                out = model(Xt[idx])
                loss = ((out - Tt[idx]) ** 2 * Mk[idx]).sum() / \
                    Mk[idx].sum().clamp(min=1.0)
                loss.backward()
                opt.step()
        return model

    def predict(model, dates):
        model.eval()
        idx = torch.tensor([date_i[d] for d in dates])
        with torch.no_grad():
            return model(Xt[idx]).numpy()

    def val_maes(model):
        out = {}
        dp = predict(model, all_dates)
        for j, rid in enumerate(rids):
            D = Ds[rid]
            per_h = {}
            for h in range(1, 8):
                hd = D["horizons"][h]
                preds, trues = [], []
                for p in hd["val_pos"]:
                    d = D["origins"].index[p]
                    if np.isfinite(hd["yv"][p]):
                        preds.append(dp[date_i[d], j, h - 1])
                        trues.append(p)
                if len(trues) >= 10:
                    pv = _to_price(D, h, np.array(preds), np.array(trues))
                    per_h[h] = float(np.mean(np.abs(
                        hd["yv"][np.array(trues)] - pv)))
                else:
                    per_h[h] = None
            out[rid] = per_h
        return out

    # phase 1: early stopping on pooled validation price MAE
    model = GCNNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Mk = mask_t("fit")
    rows = np.where(masks["fit"].any(axis=(1, 2)))[0]
    n = len(rows)
    bs = 64
    best_mae, best_epoch, wait = np.inf, 0, 0
    for ep_i in range(40):
        model.train()
        perm = rows[torch.randperm(n).numpy()]
        for i in range(0, n, bs):
            idx = torch.tensor(perm[i:i + bs])
            opt.zero_grad()
            out = model(Xt[idx])
            loss = ((out - Tt[idx]) ** 2 * Mk[idx]).sum() / \
                Mk[idx].sum().clamp(min=1.0)
            loss.backward()
            opt.step()
        vm = val_maes(model)
        vals = [v for per in vm.values() for v in per.values() if v is not None]
        cur = float(np.mean(vals)) if vals else np.inf
        if cur < best_mae - 1e-3:
            best_mae, best_epoch, wait = cur, ep_i + 1, 0
        else:
            wait += 1
            if wait >= 8:
                break
    vm = val_maes(model)
    # phase 2: refit on the train masks
    model2 = train_model("train", max(best_epoch, 5))
    dp_ho = predict(model2, all_dates)
    for j, rid in enumerate(rids):
        D = Ds[rid]
        for h in range(1, 8):
            hd = D["horizons"][h]
            dp = np.array([dp_ho[date_i[D["origins"].index[p]], j, h - 1]
                           for p in hd["ho_pos"]])
            D["roster"][h]["gcn"] = dict(
                status="ok", val_mae=vm[rid][h],
                pred=_to_price(D, h, dp, hd["ho_pos"]),
                detail=f"best_epoch {best_epoch}")

    _cache = {}

    def _gcn_forecast(rid):
        if "model" not in _cache:
            _cache["model"] = train_model("all", max(best_epoch, 5))
        return _cache["model"]

    for j, rid in enumerate(rids):
        D = Ds[rid]

        def _fc(h, rid=rid, j=j):
            model = _gcn_forecast(rid)
            last = D["origins"].index.max()
            dp = predict(model, [last])
            return float(dp[0, j, h - 1])
        D["forecasters"]["gcn"] = _fc


# ------------------------------------------------------- orchestration
def roster_region(D):
    """Run every per-region roster family (the GCN runs separately, it
    needs all regions at once). Each family records, per horizon, its
    validation MAE and its holdout price predictions, or a failure with
    the reason."""
    D["roster"] = {h: {} for h in range(1, 8)}
    D["W"], D["T"] = build_windows(D)
    try:
        _roster_tabular(D)
    except Exception as e:
        _fail(D, ["ridge", "mlp", "random_forest",
                  "gp_rbf_seasonal", "gp_matern_linear"], e)
    try:
        _roster_rocket(D)
    except Exception as e:
        _fail(D, ["minirocket"], e)
    for fam in ("lstm", "pi_lstm", "transformer", "tft", "patchtst"):
        try:
            _roster_seq(D, fam)
        except Exception as e:
            _fail(D, [fam], e)
    try:
        _roster_arima(D)
    except Exception as e:
        _fail(D, ["arima", "sarima"], e)
    return D


def _forecast_delta(D, key, h):
    if key in D.get("forecast_delta", {}) and h in D["forecast_delta"][key]:
        return float(D["forecast_delta"][key][h])
    fn = D.get("forecasters", {}).get(key)
    if fn is not None:
        return float(fn(h))
    return 0.0


def finalize_region(D):
    """Per-horizon validation selection over the full roster, headline
    holdout metrics, the full model-compare table, and the issued
    forecast (the winner refit on all origins; ridge/MLP winners also
    export their weights for the site's in-browser recompute, other
    winners ship the pipeline-computed forecast with weights=None)."""
    result = dict(features=D["feats"], n_origins=int(len(D["origins"])),
                  train_origins=int(len(D["train_idx"])),
                  holdout_origins=int(len(D["hold_idx"])),
                  holdout_start=str(D["split"].date()), horizons={})
    Xv = D["Xv"]
    last_origin = D["origins"].index.max()
    xrow = D["origins"].loc[[last_origin]]
    base_slog = float(slog(xrow["p_last"].iloc[0]))
    fc, weights = [], {}
    for h in range(1, 8):
        hd = D["horizons"][h]
        yv, ho = hd["yv"], hd["ho_pos"]
        pers = evaluate(yv[ho], D["p_last"][ho])
        entries = D["roster"][h]
        val_mae = {k: e["val_mae"] for k, e in entries.items()
                   if e.get("status") == "ok" and e.get("val_mae") is not None
                   and np.isfinite(e["val_mae"])}
        chosen = min(val_mae, key=val_mae.get)
        ev = evaluate(yv[ho], entries[chosen]["pred"])
        resid = yv[ho] - entries[chosen]["pred"]
        resid = resid[np.isfinite(resid)]
        sigma = float(np.std(resid)) if len(resid) else np.nan
        skill = (1 - ev["mae"] / pers["mae"]) * 100 if pers["mae"] else np.nan
        models = {}
        for k in ROSTER_KEYS:
            e = entries.get(k)
            if e is None:
                models[k] = dict(status="failed", reason="not run")
            elif e.get("status") == "ok":
                evk = evaluate(yv[ho], e["pred"])
                models[k] = dict(
                    status="ok",
                    mae=round(evk["mae"], 2), rmse=round(evk["rmse"], 2),
                    r2=round(evk["r2"], 3), n=evk["n"],
                    val_mae=(round(e["val_mae"], 2)
                             if e.get("val_mae") is not None else None),
                    detail=e.get("detail"))
            else:
                models[k] = dict(status="failed",
                                 reason=e.get("reason", ""))
        models["persistence"] = dict(
            status="ok",
            mae=round(pers["mae"], 2), rmse=round(pers["rmse"], 2),
            r2=round(pers["r2"], 3), n=pers["n"], val_mae=None)
        ridge_ev = (evaluate(yv[ho], entries["ridge"]["pred"])
                    if entries.get("ridge", {}).get("status") == "ok" else ev)
        result["horizons"][h] = dict(
            model=ev, persistence=pers, ridge=ridge_ev, skill_pct=skill,
            resid_sigma=sigma, chosen=chosen,
            val_mae={k: round(v, 2) for k, v in val_mae.items()},
            models=models)
        # issued forecast from the winner, refit on all origins
        all_pos = np.where(hd["ok"])[0]
        clip2 = (float(np.min(yv[all_pos])), float(np.max(yv[all_pos])) * 1.25)
        if chosen in ("ridge", "mlp"):
            sc2 = StandardScaler().fit(Xv[all_pos])
            mfin = _fit_candidate(chosen, sc2.transform(Xv[all_pos]),
                                  hd["delta"][all_pos])
            d_pred = float(mfin[1].predict(sc2.transform(xrow.values))[0])
            weights[h] = _export_candidate(mfin, sc2, clip2, base_slog)
        else:
            weights[h] = None
            try:
                d_pred = _forecast_delta(D, chosen, h)
            except Exception as e:
                print(f"  forecast refit for {chosen} h={h} failed ({e}); "
                      f"falling back to persistence for this horizon")
                d_pred = 0.0
        pred = float(np.clip(slog_inv(base_slog + d_pred), clip2[0], clip2[1]))
        sig = result["horizons"][h]["resid_sigma"]
        d = last_origin + pd.Timedelta(days=h)
        fc.append(dict(date=str(d.date()), pred=round(pred, 2),
                       lo=round(pred - 1.2816 * sig, 2),
                       hi=round(pred + 1.2816 * sig, 2)))
    result["forecast_origin"] = str(last_origin.date())
    result["forecast"] = fc
    result["weights"] = weights
    result["chosen"] = {h: result["horizons"][h]["chosen"] for h in range(1, 8)}
    result["features_last"] = {k: float(xrow[k].iloc[0]) for k in D["feats"]}
    result["roster"] = ROSTER_META
    result["gcn_adjacency"] = D.get("gcn_adjacency")
    return result, D["X"]

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
              "Not live data. Sibling of HydroTwin (river twin); this twin forecasts energy prices. "
              "It runs the SAME model roster as the HydroTwin paper (LSTM, physics-informed LSTM, "
              "Transformer, TFT, PatchTST, two Gaussian processes, a graph convolutional network, "
              "ARIMA, SARIMA) plus ridge, MLP, Random Forest and MiniRocket, so the two twins are "
              "directly comparable.")),
        regions=[])
    report = {}
    # pass 1: build frames, prepare data, run every per-region roster
    # family; pass 2 (after the cross-region GCN): finalize + export.
    prepared = {}
    for rid, cfg in REGIONS.items():
        print(f"\n=== region {rid}: {cfg['name']} ===")
        F, gauge_names = build_region_frame(cfg, ice, gas)
        D = prepare_region(rid, cfg, F)
        roster_region(D)
        prepared[rid] = (F, gauge_names, D)
    try:
        run_gcn({rid: prepared[rid][2] for rid in REGIONS})
    except Exception as e:
        print(f"GCN failed ({e}); roster continues without it")
        for rid in REGIONS:
            _fail(prepared[rid][2], ["gcn"], e)
    for rid, cfg in REGIONS.items():
        F, gauge_names, D = prepared[rid]
        res, X = finalize_region(D)
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
                models=res["horizons"][h]["models"],
                resid_sigma=round(res["horizons"][h]["resid_sigma"], 2))
                for h in range(1, 8)},
            roster=res["roster"],
            gcn_adjacency=res.get("gcn_adjacency"),
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
