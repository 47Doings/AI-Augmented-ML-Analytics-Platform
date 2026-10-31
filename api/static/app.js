// Front end for the platform. No framework. Every value that came from a user or the server
// goes through esc() before it is put into the page, so nothing typed by a user can run as code.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let me = null;
let pollTimer = null;
let mode = "demo";  // which way of getting data is showing: "demo" or "own"
let profile = null; // what the uploaded file contains

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  return { ok: res.ok, status: res.status, data };
}

function errorText(r) {
  const d = r.data && r.data.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ");
  return `Something went wrong (${r.status})`;
}

function say(id, text, kind) {
  const el = $(id);
  el.textContent = text || "";
  el.className = "msg" + (kind ? " " + kind : "");
}

// ---------------------------------------------------------------- start / auth
async function start() {
  const r = await api("GET", "/api/me");
  if (r.status === 401 || !r.ok) return showAuth();
  me = r.data;
  showApp();
}

function showAuth() {
  clearInterval(pollTimer);
  $("auth").hidden = false; $("app").hidden = true; $("who").hidden = true;
}

function setTab(which) {
  $("form-signup").hidden = which !== "signup";
  $("form-login").hidden = which !== "login";
  $("tab-signup").classList.toggle("active", which === "signup");
  $("tab-login").classList.toggle("active", which === "login");
  say("auth-msg", "");
}

async function submitAuth(ev, path) {
  ev.preventDefault();
  const form = ev.target;
  const body = Object.fromEntries(new FormData(form).entries());
  const btn = form.querySelector("button[type=submit]");
  btn.disabled = true;
  const r = await api("POST", path, body);
  btn.disabled = false;
  if (!r.ok) return say("auth-msg", errorText(r), "err");
  form.reset();
  await start();
}

async function logout() {
  await api("POST", "/api/logout");
  me = null;
  showAuth();
}

// ---------------------------------------------------------------- main app
async function showApp() {
  $("auth").hidden = true; $("app").hidden = false;
  const who = $("who");
  who.hidden = false;
  who.innerHTML = `<span>${esc(me.business_name)} · ${esc(me.email)}</span><button id="btn-logout" type="button">Log out</button>`;
  $("btn-logout").addEventListener("click", logout);
  await renderUseCases();
  setMode(me.use_case_id === "custom" || (me.upload && !me.use_case_id) ? "own" : "demo");
  await renderOwn();
  renderData();
  renderSteps(me.latest_run);
  if (me.latest_run && me.latest_run.status === "running") poll();
  await loadResults();
}

async function renderUseCases() {
  const r = await api("GET", "/api/use-cases");
  const box = $("usecases");
  box.innerHTML = "";
  for (const u of r.data || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "choice" + (me.use_case_id === u.id ? " selected" : "");
    b.innerHTML = `<b>${esc(u.name)}</b><span>${esc(u.business_question)} (${esc(u.problem_type)})</span>`;
    b.addEventListener("click", async () => {
      const res = await api("PUT", "/api/workspace/use-case", { use_case_id: u.id });
      if (!res.ok) return say("data-msg", errorText(res), "err");
      await refreshMe();
      await renderUseCases();
      await loadResults();
    });
    box.appendChild(b);
  }
}

async function refreshMe() {
  const r = await api("GET", "/api/me");
  if (r.ok) me = r.data;
}

function renderData() {
  const rows = me.data_rows;
  const done = me.data_loaded && rows;
  if (me.use_case_id !== "custom") {
    say("data-msg", done ? `Loaded: ${Object.entries(rows).map(([t, n]) => `${t} ${n.toLocaleString()}`).join(" · ")}` : "", done ? "ok" : "");
  }
  $("btn-run").disabled = !(me.use_case_id && me.data_loaded) || (me.latest_run && me.latest_run.status === "running");
}

async function generateData() {
  const n = parseInt($("customers").value, 10);
  $("btn-data").disabled = true;
  say("data-msg", "Generating and loading data…");
  const r = await api("POST", "/api/workspace/data", { customers: n });
  $("btn-data").disabled = false;
  if (!r.ok) return say("data-msg", errorText(r), "err");
  await refreshMe();
  renderData();
  $("results").hidden = true;
}

// ---------------------------------------------------------------- the business's own data
const KIND_LABEL = { numeric: "number", categorical: "category", date: "date", identifier: "unique id", text: "free text", empty: "empty" };
const MAX_UPLOAD = 10 * 1024 * 1024;

function setMode(m) {
  mode = m;
  $("panel-demo").hidden = m !== "demo";
  $("panel-own").hidden = m !== "own";
  $("mode-demo").classList.toggle("active", m === "demo");
  $("mode-own").classList.toggle("active", m === "own");
}

async function uploadFile() {
  const file = $("file").files[0];
  if (!file) return say("upload-msg", "Choose a CSV file first.", "err");
  if (file.size > MAX_UPLOAD) return say("upload-msg", "The file is larger than 10 MB.", "err");
  $("btn-upload").disabled = true;
  say("upload-msg", "Uploading and checking…");
  let res, data = null;
  try {
    // the file is the raw request body; the server checks size, encoding and shape before anything else
    res = await fetch("/api/workspace/upload?filename=" + encodeURIComponent(file.name), {
      method: "PUT", credentials: "same-origin", headers: { "Content-Type": "text/csv" }, body: file,
    });
    try { data = await res.json(); } catch { /* empty body */ }
  } catch {
    $("btn-upload").disabled = false;
    return say("upload-msg", "Could not reach the server.", "err");
  }
  $("btn-upload").disabled = false;
  if (!res.ok) return say("upload-msg", errorText({ data, status: res.status }), "err");
  await refreshMe();
  await renderUseCases();
  renderData();
  $("results").hidden = true;
  await renderOwn();
}

async function renderOwn() {
  if (!me.upload) { $("profile").hidden = true; return; }
  const r = await api("GET", "/api/workspace/upload");
  if (!r.ok) { $("profile").hidden = true; return; }
  profile = r.data;
  const name = me.upload.filename ? me.upload.filename + ": " : "";
  say("upload-msg", `${name}${profile.rows.toLocaleString()} rows, ${profile.columns.length} columns.`, "ok");
  const shown = (c) => (c.original && c.original !== c.name ? `${c.original} (${c.name})` : c.name);
  $("profile-table").innerHTML = table({
    columns: ["column", "looks like", "missing %", "different values", "examples"],
    rows: profile.columns.map((c) => [shown(c), KIND_LABEL[c.kind] || c.kind, c.missing_pct, c.distinct, c.samples.join(", ")]),
    total_rows: profile.columns.length,
  });
  fillMapping();
  $("profile").hidden = false;
}

const colLabel = (c) => (c.original && c.original !== c.name ? `${c.original} (${c.name})` : c.name);
const opt = (v, text, sel) => `<option value="${esc(v)}"${sel ? " selected" : ""}>${esc(text)}</option>`;

function targetCandidates(type) {
  return profile.columns.filter((c) => type === "classification"
    ? c.top_values && c.distinct >= 2 && c.distinct <= 20 && ["categorical", "numeric"].includes(c.kind)
    : c.kind === "numeric" && c.distinct >= 10);
}

function fillMapping() {
  const f = $("form-map");
  $("features").innerHTML = "";  // a new file: forget which boxes were ticked for the old one
  const saved = me.custom_mapping;
  const sug = profile.suggested || {};
  if (saved) {
    f.elements.analysis_name.value = saved.name;
    f.elements.business_question.value = saved.business_question;
    f.elements.problem_type.value = saved.problem_type;
  } else if (sug.target_column && !targetCandidates("classification").some((c) => c.name === sug.target_column)) {
    f.elements.problem_type.value = "regression";  // the suggested target has too many values to be a yes/no outcome
  }
  const pickFrom = (name, items, none, want) => {
    f.elements[name].innerHTML = (none ? opt("", none, !want) : "") + items.map((c) => opt(c.name, colLabel(c), c.name === want)).join("");
  };
  pickFrom("id_column", profile.columns, "(none: use the row number)", saved ? saved.id_column : sug.id_column);
  pickFrom("date_column", profile.columns.filter((c) => c.kind === "date"), "(none)", saved ? saved.date_column : sug.date_column);
  refreshTarget(saved ? saved.target_column : sug.target_column, saved ? saved.positive_value : null, saved ? new Set(saved.features) : null);
}

function refreshTarget(wantTarget, wantPositive, wantFeatures) {
  const f = $("form-map");
  const cands = targetCandidates(f.elements.problem_type.value);
  const want = wantTarget || f.elements.target_column.value;
  f.elements.target_column.innerHTML = cands.length
    ? cands.map((c) => opt(c.name, colLabel(c), c.name === want)).join("")
    : opt("", "(no column in your file suits this kind of answer)");
  refreshPositive(wantPositive);
  refreshFeatures(wantFeatures);
}

function refreshPositive(want) {
  const f = $("form-map");
  const classification = f.elements.problem_type.value === "classification";
  $("pos-wrap").hidden = !classification;
  const t = profile.columns.find((c) => c.name === f.elements.target_column.value);
  const vals = (t && t.top_values) || [];
  let pick = want || f.elements.positive_value.value;
  if (!vals.some((v) => v.value === pick) && vals.length) {
    pick = [...vals].sort((a, b) => a.count - b.count)[0].value;  // the rarer value is usually the one of interest
  }
  f.elements.positive_value.innerHTML = vals.map((v) => opt(v.value, `${v.value} (${v.count.toLocaleString()} rows)`, v.value === pick)).join("");
}

function refreshFeatures(keep) {
  const f = $("form-map");
  const taken = new Set([f.elements.target_column.value, f.elements.id_column.value, f.elements.date_column.value].filter(Boolean));
  const boxes = document.querySelectorAll("#features input");
  const previous = keep || (boxes.length ? new Set([...document.querySelectorAll("#features input:checked")].map((i) => i.value)) : null);
  const usable = profile.columns.filter((c) => c.usable_as_feature && !taken.has(c.name));
  const skipped = profile.columns.filter((c) => !c.usable_as_feature && !taken.has(c.name));
  $("features").innerHTML = usable.map((c) =>
    `<label><input type="checkbox" value="${esc(c.name)}"${!previous || previous.has(c.name) ? " checked" : ""}> ${esc(colLabel(c))} <span class="hint">${esc(KIND_LABEL[c.kind] || c.kind)}</span></label>`).join("")
    + (skipped.length ? `<p class="note skipped">Not offered: ${skipped.map((c) => `${esc(colLabel(c))} (${esc(KIND_LABEL[c.kind] || c.kind)})`).join(", ")}</p>` : "");
}

async function saveMapping(ev) {
  ev.preventDefault();
  const f = ev.target;
  const classification = f.elements.problem_type.value === "classification";
  const body = {
    name: f.elements.analysis_name.value,
    business_question: f.elements.business_question.value,
    problem_type: f.elements.problem_type.value,
    target_column: f.elements.target_column.value,
    positive_value: classification ? f.elements.positive_value.value || null : null,
    id_column: f.elements.id_column.value || null,
    date_column: f.elements.date_column.value || null,
    features: [...document.querySelectorAll("#features input:checked")].map((i) => i.value),
  };
  const btn = f.querySelector("button[type=submit]");
  btn.disabled = true;
  const r = await api("PUT", "/api/workspace/custom-use-case", body);
  btn.disabled = false;
  if (!r.ok) return say("map-msg", errorText(r), "err");
  await refreshMe();
  await renderUseCases();
  renderData();
  $("results").hidden = true;
  say("map-msg", `Saved: ${r.data.numeric_inputs} number inputs and ${r.data.category_inputs} category inputs. Now run the analysis.`, "ok");
}

// ---------------------------------------------------------------- run + progress
const ICON = { done: "✓", failed: "✕", running: "…", pending: "·", skipped: "–" };

function renderSteps(run) {
  const ol = $("steps");
  ol.innerHTML = "";
  if (!run) return say("run-msg", "");
  for (const s of run.steps) {
    const li = document.createElement("li");
    li.className = "s-" + s.status;
    const extra = [s.detail, s.seconds != null ? s.seconds + "s" : null].filter(Boolean).join(" · ");
    li.innerHTML = `<span class="dot">${ICON[s.status] || "·"}</span><span>${esc(s.label)}</span> <small>${esc(extra)}</small>`;
    ol.appendChild(li);
  }
  if (run.status === "failed") say("run-msg", "Run failed: " + (run.error || "unknown error"), "err");
  else if (run.status === "succeeded") say("run-msg", "Finished.", "ok");
  else say("run-msg", "Running…");
}

async function startRun() {
  $("btn-run").disabled = true;
  const r = await api("POST", "/api/workspace/run");
  if (!r.ok) { say("run-msg", errorText(r), "err"); return renderData(); }
  poll();
}

function poll() {
  clearInterval(pollTimer);
  const tick = async () => {
    const r = await api("GET", "/api/workspace/run");
    if (!r.ok) return clearInterval(pollTimer);
    renderSteps(r.data.run);
    if (r.data.run && r.data.run.status !== "running") {
      clearInterval(pollTimer);
      await refreshMe();
      renderData();
      await loadResults();
    } else {
      $("btn-run").disabled = true;
    }
  };
  tick();
  pollTimer = setInterval(tick, 1500);
}

// ---------------------------------------------------------------- results
function fmt(v) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  return esc(v);
}

function table(t, opts = {}) {
  if (!t || !t.columns) return "";
  const head = t.columns.map((c) => `<th>${esc(c)}</th>`).join("");
  const body = t.rows.map((row) => {
    const sel = opts.selectedBy && row[t.columns.indexOf(opts.selectedBy)] === true ? ' class="sel"' : "";
    return `<tr${sel}>${row.map((v, i) => {
      if (opts.pillCol === t.columns[i] && v) return `<td><span class="pill ${esc(v)}">${esc(v)}</span></td>`;
      return `<td${typeof v === "number" ? ' class="num"' : ""}>${fmt(v)}</td>`;
    }).join("")}</tr>`;
  }).join("");
  const more = t.total_rows > t.rows.length ? `<p class="note">Showing ${t.rows.length} of ${t.total_rows} rows.</p>` : "";
  return `<div class="scroll"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>${more}`;
}

function col(t, name) { const i = t.columns.indexOf(name); return t.rows.map((r) => r[i]); }

function bars(labels, values, opts = {}) {
  const max = Math.max(...values.map((v) => Math.abs(v || 0)), 1e-9);
  return `<div class="bars">${labels.map((l, i) => {
    const v = values[i] || 0;
    const weak = opts.weak && opts.weak[i] ? " weak" : "";
    return `<div class="bar"><span>${esc(l)}</span><div class="track"><div class="fill${weak}" data-w="${Math.max(0, (v / max) * 100).toFixed(1)}"></div></div><span class="n">${esc(opts.format ? opts.format(v) : v.toFixed(3))}</span></div>`;
  }).join("")}</div>`;
}

async function loadResults() {
  const box = $("results");
  if (!me.use_case_id || !me.results_available.includes(me.use_case_id)) { box.hidden = true; return; }
  const r = await api("GET", "/api/workspace/results");
  if (!r.ok) { box.hidden = true; return; }
  renderResults(r.data);
}

function renderResults(res) {
  const m = res.model, card = m.card;
  const kpis = res.kpis.map((k) => `<div class="kpi"><div class="v">${esc(k.shown)}</div><div class="l">${esc(k.label)}</div></div>`).join("");
  const imp = m.importance;
  const impBars = bars(col(imp, "feature").slice(0, 10), col(imp, "importance").slice(0, 10), {
    weak: col(imp, "clearly_matters").slice(0, 10).map((x) => !x),
  });
  let deciles = "";
  if (m.deciles) {
    deciles = `<h3>Does it help? Actual outcome rate by risk decile (10 = riskiest)</h3>` +
      bars(col(m.deciles, "risk_decile").map((d) => "Decile " + d), col(m.deciles, "rate"), { format: (v) => (v * 100).toFixed(1) + "%" });
  }
  const questions = Object.entries(res.business_questions).map(([name, t]) =>
    `<details><summary>${esc(name.replaceAll("_", " "))}</summary>${table(t)}</details>`).join("");
  const metric = res.use_case.problem_type === "classification"
    ? `Flagging the riskiest ${(card.operating_point.flag_top_fraction * 100).toFixed(0)}% of entities: precision ${(card.test_metrics.precision * 100).toFixed(1)}%, recall ${(card.test_metrics.recall * 100).toFixed(1)}%, lift ${card.test_metrics.lift.toFixed(1)}x.`
    : `Test error: MAE ${card.test_metrics.mae.toFixed(2)}, RMSE ${card.test_metrics.rmse.toFixed(2)}, R² ${card.test_metrics.r2.toFixed(3)}.`;

  $("results").innerHTML = `
    <section class="card">
      <h2>${esc(res.use_case.name)}</h2>
      <p class="muted">${esc(res.use_case.business_question)}</p>
      <div class="kpis">${kpis}</div>
      <p><a class="primary-link" href="/api/workspace/report" target="_blank" rel="noopener">Open business report (print it to PDF)</a></p>
      <div id="insights" class="insights"></div>

      <h3>Baseline vs better models</h3>
      <p class="note">The baseline predicts the average for everyone. Models are picked by cross-validation on the training split; the test columns are on rows the model never saw.</p>
      ${table(m.comparison, { selectedBy: "selected" })}
      <p>Selected <b>${esc(m.selected)}</b>. ${esc(metric)}</p>

      <h3>What drives the prediction</h3>
      <p class="note">Permutation importance on the test set. Faded bars are within noise. This shows association in the data, not cause.</p>
      ${impBars}
      <details><summary>Importance table with direction</summary>${table(imp)}</details>
      ${deciles}

      <h3>Business questions (SQL)</h3>${questions}

      <h3>Data quality</h3>
      <details><summary>${res.data_quality ? res.data_quality.rows.length : 0} checks, worst first</summary>${res.data_quality ? table(res.data_quality) : ""}</details>

      <h3>Drift check</h3>
      ${res.drift
        ? `<p class="note">Same SQL at a later date (${esc(res.drift.later_cutoff)}) compared with the training data. PSI below 0.10 stable, 0.10–0.25 watch, above 0.25 alert.</p>`
        : `<p class="note">Not run for this data: it needs a later export date to compare against. See the monitoring plan for how it would be done in production.</p>`}
      ${res.drift ? `<details><summary>${res.drift.rows.length} variables</summary>${table(res.drift, { pillCol: "status" })}</details>` : ""}
      <p class="note">Rows used: ${card.rows.train.toLocaleString()} train, ${card.rows.test.toLocaleString()} test. Seed ${card.seed}. Library versions are recorded in the model card.</p>
    </section>`;
  // The page's Content-Security-Policy forbids style="..." attributes in markup, so bar widths
  // are applied here through the DOM instead (that is allowed).
  document.querySelectorAll("#results .fill[data-w]").forEach((el) => { el.style.width = el.dataset.w + "%"; });
  $("results").hidden = false;
  loadInsights();
}

// ---------------------------------------------------------------- AI insights
const SECTIONS = [
  ["executive_summary", "Executive summary"],
  ["key_findings", "Key findings"],
  ["recommendations", "Recommendations"],
  ["caveats", "Caveats"],
];

async function loadInsights() {
  const box = $("insights");
  if (!box) return;
  const r = await api("GET", "/api/workspace/insights");
  renderInsights(r.ok ? r.data : null);
}

function renderInsights(ins) {
  const box = $("insights");
  if (!box) return;
  if (!ins) { box.innerHTML = ""; return; }
  const label = { ai: "Written by AI, every number checked", fallback: "Template summary, no AI used", none: "Not available" }[ins.mode] || ins.mode;
  let html = `<h3>Insights <span class="badge ${esc(ins.mode)}">${esc(label)}</span></h3>`;
  if (ins.reason) html += `<p class="note">${esc(ins.reason)}</p>`;
  if (ins.document) {
    for (const [key, title] of SECTIONS) {
      const items = ins.document[key] || [];
      if (!items.length) continue;
      html += `<h4>${esc(title)}</h4><ul class="insight-list">` + items.map((b) =>
        `<li>${b.type ? `<span class="pill">${esc(b.type)}</span> ` : ""}${esc(b.text)} <small class="cites">${b.facts.map(esc).join(", ")}</small></li>`).join("") + "</ul>";
    }
    const v = ins.validation;
    if (v) html += `<p class="note">Checks: ${v.numbers_checked} numbers matched to facts, ${v.facts_cited} facts cited. Provider: ${esc(ins.provider)}.</p>`;
    html += `<details><summary>Facts behind this text (${ins.facts.length})</summary><ul class="facts">` +
      ins.facts.map((f) => `<li><b>${esc(f.id)}</b> ${esc(f.text)}</li>`).join("") + "</ul></details>";
  }
  if (ins.attempts && ins.attempts.some((a) => a.outcome !== "accepted")) {
    html += `<details><summary>Rejected or failed AI attempts</summary><ul class="facts">` +
      ins.attempts.map((a) => `<li>Attempt ${a.n}: ${esc(a.outcome)}${a.errors.length ? " - " + a.errors.map(esc).join("; ") : ""}</li>`).join("") + "</ul></details>";
  }
  html += `<p><button id="btn-insights" type="button">Write insights again</button> <span id="insights-msg" class="msg"></span></p>`;
  box.innerHTML = html;
  $("btn-insights").addEventListener("click", async () => {
    $("btn-insights").disabled = true;
    say("insights-msg", "Working…");
    const res = await api("POST", "/api/workspace/insights");
    if (!res.ok) { $("btn-insights").disabled = false; return say("insights-msg", errorText(res), "err"); }
    renderInsights(res.data);
  });
}

// ---------------------------------------------------------------- wire up
$("tab-signup").addEventListener("click", () => setTab("signup"));
$("tab-login").addEventListener("click", () => setTab("login"));
$("form-signup").addEventListener("submit", (e) => submitAuth(e, "/api/signup"));
$("form-login").addEventListener("submit", (e) => submitAuth(e, "/api/login"));
$("btn-data").addEventListener("click", generateData);
$("mode-demo").addEventListener("click", () => setMode("demo"));
$("mode-own").addEventListener("click", () => setMode("own"));
$("btn-upload").addEventListener("click", uploadFile);
$("form-map").addEventListener("submit", saveMapping);
$("form-map").elements.problem_type.addEventListener("change", () => refreshTarget());
$("form-map").elements.target_column.addEventListener("change", () => { refreshPositive(); refreshFeatures(); });
$("form-map").elements.id_column.addEventListener("change", () => refreshFeatures());
$("form-map").elements.date_column.addEventListener("change", () => refreshFeatures());
$("btn-run").addEventListener("click", startRun);
start();
