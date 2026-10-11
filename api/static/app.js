// Front end for the platform. No framework. Every value that came from a user or the server
// goes through esc() before it is put into the page, so nothing typed by a user can run as code.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let me = null;
let pollTimer = null;

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
  say("data-msg", done ? `Loaded: ${Object.entries(rows).map(([t, n]) => `${t} ${n.toLocaleString()}`).join(" · ")}` : "", done ? "ok" : "");
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
      <p class="note">Same SQL at a later date (${esc(res.drift ? res.drift.later_cutoff : "n/a")}) compared with the training data. PSI below 0.10 stable, 0.10–0.25 watch, above 0.25 alert.</p>
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
$("btn-run").addEventListener("click", startRun);
start();
