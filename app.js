// HydroTwin Energy — static snapshot viewer. All numbers come from data.js,
// which pipeline/pipeline.py computes from fetched public data. Nothing here
// invents values; charts draw exactly the exported series.
(function () {
  const D = window.HTE_DATA;
  const $ = (s) => document.querySelector(s);
  document.getElementById("snap").textContent =
    "Data snapshot: " + D.meta.snapshot_date + " · window " + D.meta.data_window + " · not live data";
  document.getElementById("foot").textContent = D.meta.note;

  // Recompute each region's forecast in the browser from the exported model
  // weights + the last origin's feature vector (same math as the pipeline:
  // standardize, predict signed-log change, invert, clip). If anything is
  // missing, the pipeline-computed forecast shipped in data.js is used.
  function slog(p) { return Math.sign(p) * Math.log1p(Math.abs(p)); }
  function slogInv(y) { return Math.sign(y) * Math.expm1(Math.abs(y)); }
  D.regions.forEach(r => {
    try {
      if (!r.weights || !r.features_last) return;
      const x = r.features.map(f => r.features_last[f]);
      r.forecast = r.forecast.map((f, i) => {
        const h = String(i + 1), w = r.weights[h];
        if (!w) return f;
        const z = x.map((v, j) => (v - w.mean[j]) / w.scale[j]);
        let d;
        if (w.type === "ridge") {
          d = w.intercept + z.reduce((s, v, j) => s + v * w.coef[j], 0);
        } else {
          let a = z;
          w.coefs.forEach((C, li) => {
            const out = w.intercepts[li].slice();
            for (let j = 0; j < out.length; j++)
              for (let k = 0; k < a.length; k++) out[j] += a[k] * C[k][j];
            a = li < w.coefs.length - 1 ? out.map(v => Math.max(v, 0)) : out;
          });
          d = a[0];
        }
        const pred = Math.min(Math.max(slogInv(w.base_slog + d), w.clip[0]), w.clip[1]);
        const sig = r.metrics[h].resid_sigma;
        return { date: f.date, pred: Math.round(pred * 100) / 100,
                 lo: Math.round((pred - 1.2816 * sig) * 100) / 100,
                 hi: Math.round((pred + 1.2816 * sig) * 100) / 100 };
      });
    } catch (e) { /* keep pipeline forecast */ }
  });

  let current = D.regions[0].id;

  function fmt(n, d) { return (n === null || n === undefined || isNaN(n)) ? "—" : Number(n).toFixed(d === undefined ? 2 : d); }
  function skillCls(v) { return v >= 0 ? "pos" : "neg"; }

  // ---------- overview cards ----------
  function renderCards() {
    $("#cards").innerHTML = D.regions.map(r => {
      const f1 = r.forecast[0], f7 = r.forecast[6];
      const m1 = r.metrics["1"];
      return `<div class="card ${r.id === current ? "active" : ""}" data-id="${r.id}">
        <h3>${r.name}</h3>
        <div class="hub">${r.hub}</div>
        <div class="big">$${fmt(r.latest.last_price)}<span style="font-size:13px;color:var(--mut)"> /MWh · ${r.latest.last_price_date}</span></div>
        <div class="row"><span>Forecast +1d (${f1.date})</span><b>$${fmt(f1.pred)}</b></div>
        <div class="row"><span>Forecast +7d (${f7.date})</span><b>$${fmt(f7.pred)}</b></div>
        <div class="row"><span>Holdout skill vs persistence (+1d)</span><b class="${skillCls(m1.skill_pct)}">${fmt(m1.skill_pct, 1)}%</b></div>
      </div>`;
    }).join("");
    document.querySelectorAll(".card").forEach(c =>
      c.addEventListener("click", () => { current = c.dataset.id; renderCards(); renderRegion(); }));
  }

  // ---------- canvas helpers ----------
  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
    return [ctx, w, h];
  }
  function lineChart(cv, series, opts) {
    // series: [{pts:[[xLabel, y], ...], color, width, fill:[lo,hi] arrays optional}]
    const [ctx, W, H] = setupCanvas(cv);
    const padL = 52, padR = 10, padT = 10, padB = 22;
    let ymin = Infinity, ymax = -Infinity;
    series.forEach(s => s.pts.forEach(p => { if (p[1] < ymin) ymin = p[1]; if (p[1] > ymax) ymax = p[1]; }));
    if (opts.band) opts.band.forEach(p => { if (p[1] < ymin) ymin = p[1]; if (p[2] > ymax) ymax = p[2]; });
    if (!isFinite(ymin)) return;
    const pad = (ymax - ymin) * 0.08 + 1e-6; ymin -= pad; ymax += pad;
    const n = opts.n;
    const X = i => padL + (W - padL - padR) * (i / (n - 1));
    const Y = v => padT + (H - padT - padB) * (1 - (v - ymin) / (ymax - ymin));
    ctx.strokeStyle = "#D8E2EA"; ctx.fillStyle = "#5A6B7A"; ctx.font = "11px sans-serif"; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const v = ymin + (ymax - ymin) * g / 4, y = Y(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(fmt(v, Math.abs(ymax) >= 100 ? 0 : 1), 6, y + 4);
    }
    ctx.fillText(opts.x0, padL, H - 6);
    const xr = opts.x1; ctx.fillText(xr, W - padR - ctx.measureText(xr).width, H - 6);
    if (opts.band) {
      ctx.beginPath();
      opts.band.forEach((p, i) => { const x = X(p[0]); i ? ctx.lineTo(x, Y(p[2])) : ctx.moveTo(x, Y(p[2])); });
      for (let i = opts.band.length - 1; i >= 0; i--) ctx.lineTo(X(opts.band[i][0]), Y(opts.band[i][1]));
      ctx.closePath(); ctx.fillStyle = "rgba(6,179,199,.18)"; ctx.fill();
    }
    series.forEach(s => {
      ctx.beginPath();
      s.pts.forEach((p, i) => { const x = X(p[0]), y = Y(p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.stroke();
    });
    if (opts.splitX !== undefined) {
      ctx.beginPath(); ctx.moveTo(X(opts.splitX), padT); ctx.lineTo(X(opts.splitX), H - padB);
      ctx.strokeStyle = "#B3261E"; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = "#B3261E"; ctx.fillText("forecast →", X(opts.splitX) + 4, padT + 10);
    }
  }
  function spark(cv, pts, color) {
    if (!pts || !pts.length) return;
    const [ctx, W, H] = setupCanvas(cv);
    let ymin = Infinity, ymax = -Infinity;
    pts.forEach(p => { if (p.v < ymin) ymin = p.v; if (p.v > ymax) ymax = p.v; });
    const pad = (ymax - ymin) * 0.1 + 1e-9; ymin -= pad; ymax += pad;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = 2 + (W - 4) * i / (pts.length - 1), y = 2 + (H - 4) * (1 - (p.v - ymin) / (ymax - ymin));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.stroke();
  }

  // ---------- region panel ----------
  function featCard(lab, val, anom, sparkId) {
    return `<div class="feat"><div class="lab">${lab}</div><div class="val">${val}</div>
      <div class="anom">${anom || ""}</div><canvas id="${sparkId}"></canvas></div>`;
  }
  function renderRegion() {
    const r = D.regions.find(x => x.id === current);
    const L = r.latest;
    const rows = [1, 2, 3, 4, 5, 6, 7].map(h => {
      const m = r.metrics[String(h)];
      return `<tr><td>+${h} day${h > 1 ? "s" : ""}</td><td>${m.chosen}</td><td>${fmt(m.mae)}</td><td>${fmt(m.rmse)}</td>
        <td>${fmt(m.r2, 3)}</td><td>${fmt(m.pers_mae)}</td><td>${fmt(m.pers_rmse)}</td>
        <td class="${skillCls(m.skill_pct)}">${fmt(m.skill_pct, 1)}%</td><td>${m.n}</td></tr>`;
    }).join("");
    const lims = [
      "Static snapshot of " + D.meta.snapshot_date + " — the page does not pull live data; rerun pipeline/pipeline.py to refresh.",
      "Prices are day-ahead values; extreme scarcity spikes are driven by heat waves, outages and emergency conditions the features cannot fully see, and are systematically under-predicted.",
      "The forecast band is ±1.28 × the holdout residual standard deviation per horizon (an ~80% band if residuals were normal); it is a measured holdout spread, not a guarantee."
    ].concat(r.limitations || []);
    $("#region").innerHTML = `
    <div class="panel">
      <h2>${r.name}</h2>
      <div class="hub">${r.hub}</div>
      <h3>Price history (last ${r.history.length} priced days, $${""}/MWh)</h3>
      <canvas class="chart" id="hist"></canvas>
      <h3>Forecast — issued ${r.forecast_origin}, next 7 days with holdout band</h3>
      <canvas class="chart" id="fc"></canvas>
      <h3>Five-source features — latest values at the forecast origin</h3>
      <div class="feats">
        ${featCard("USGS river flow (gauge mean)", fmt(L.flow_cfs, 0) + " cfs",
          "30d flow vs trailing-year mean: " + (L.flow_anom_pct >= 0 ? "+" : "") + fmt(L.flow_anom_pct, 1) + "%", "sp_flow")}
        ${L.swe_in !== undefined ? featCard("NRCS SNOTEL snowpack (SWE)", fmt(L.swe_in) + " in",
          "mean of the region's SNOTEL stations", "sp_swe") : ""}
        ${featCard("NRCS SCAN soil moisture (−8 in)", fmt(L.soil_pct, 1) + " %",
          "mean of the region's SCAN stations", "sp_soil")}
        ${featCard("EIA Henry Hub natural gas", "$" + fmt(L.gas_usd, 2) + " /MMBtu",
          "daily spot price, EIA", "sp_gas")}
        ${featCard("Open-Meteo weather (7-day)", fmt(L.precip7_mm, 1) + " mm precip",
          "mean temp " + fmt(L.temp7_c, 1) + " °C over the region's weather points", "sp_none")}
      </div>
      <h3>Model card</h3>
      <p style="font-size:13.5px">
        One model per forecast day. Each model predicts the <b>signed-log price change</b> from the last observed
        price, so persistence ("no change") is the anchor the features must beat. Two candidates are trained per
        horizon — ridge regression and an MLP (48→24 hidden units, ReLU) — on standardized engineered features;
        the winner is picked on a validation slice (the last 20% of the training period), refit on the full
        training period, and scored once on the holdout. Predictions are clipped to the training period's
        observed price range (maximum extended 25%). The forecast above is recomputed in your browser from the
        exported weights. Features (${r.features.length}):
        <span class="flist">${r.features.join(", ")}</span>.<br>
        Training origins: ${r.train_origins} · Holdout origins: ${r.holdout_origins} (time-based, from ${r.holdout_start},
        no shuffling) · Price observations used: ${r.price_rows} (${r.price_range[0]} to ${r.price_range[1]}).
        Persistence baseline = "price stays at its last observed value". Skill = 1 − MAE<sub>model</sub>/MAE<sub>persistence</sub>.
        Where skill is negative, the honest reading is that the features did not beat "tomorrow = today" on the holdout.
      </p>
      <table>
        <tr><th>Horizon</th><th>Model</th><th>MAE $</th><th>RMSE $</th><th>R²</th><th>Persistence MAE $</th><th>Persistence RMSE $</th><th>Skill</th><th>n</th></tr>
        ${rows}
      </table>
      <h3>Stations &amp; sources</h3>
      <ul class="tight">
        <li>USGS gauges: ${r.stations.usgs.join("; ")}</li>
        ${r.stations.snotel.length ? `<li>SNOTEL: ${r.stations.snotel.join(", ")}</li>` : ""}
        <li>SCAN: ${r.stations.scan.join(", ")}</li>
        <li>Weather points: ${r.stations.weather.join("; ")}</li>
      </ul>
      <p class="srcs">
        Price: ${r.hub}. Gas: EIA Henry Hub daily spot (eia.gov/dnav/ng/hist).
        Flow: USGS NWIS daily values (waterservices.usgs.gov). Snow/soil: NRCS AWDB report generator (wcc.sc.egov.usda.gov).
        Weather: Open-Meteo archive (archive-api.open-meteo.com). Retrieved ${D.meta.snapshot_date}.
      </p>
      <h3>Limitations</h3>
      <ul class="tight">${lims.map(x => `<li>${x}</li>`).join("")}</ul>
    </div>`;
    // charts
    const hist = r.history;
    lineChart($("#hist"), [{ pts: hist.map((p, i) => [i, p.v]), color: "#0B5F8A" }],
      { n: hist.length, x0: hist[0].d, x1: hist[hist.length - 1].d });
    const tail = hist.slice(-45);
    const nAll = tail.length + r.forecast.length;
    const fcPts = tail.map((p, i) => [i, p.v]);
    const all = fcPts.concat(r.forecast.map((f, j) => [tail.length + j, f.pred]));
    const band = r.forecast.map((f, j) => [tail.length + j, f.lo, f.hi]);
    // connect forecast to last actual point
    const bandFull = [[tail.length - 1, tail[tail.length - 1].v, tail[tail.length - 1].v]].concat(band);
    lineChart($("#fc"), [
      { pts: all.slice(0, tail.length), color: "#0B5F8A" },
      { pts: [[tail.length - 1, tail[tail.length - 1].v]].concat(r.forecast.map((f, j) => [tail.length + j, f.pred])), color: "#B3261E", width: 2 }
    ], { n: nAll, x0: tail[0].d, x1: r.forecast[r.forecast.length - 1].date, band: bandFull, splitX: tail.length - 1 });
    spark($("#sp_flow"), r.sparks.flow, "#0B5F8A");
    if ($("#sp_swe")) spark($("#sp_swe"), r.sparks.swe, "#06B3C7");
    spark($("#sp_soil"), r.sparks.soil, "#7A5C2E");
    spark($("#sp_gas"), r.sparks.gas, "#B3261E");
  }

  window.addEventListener("resize", () => renderRegion());
  renderCards();
  renderRegion();
})();
