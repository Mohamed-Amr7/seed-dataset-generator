"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = (u, o) => fetch(u, o).then(r => r.ok ? r.json() : r.json().then(e => Promise.reject(e)));

let STATE = {};          // current Run-form config, keyed by Config field name
let SCHEMA = null;
let PRESETS = {};        // name -> config dict (recommended + dataset recipes)
let WS = null;

// ---- tabs ----
$$(".tab").forEach(t => t.onclick = () => {
  $$(".tab").forEach(x => x.classList.remove("active"));
  $$(".panel").forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  $("#" + t.dataset.tab).classList.add("active");
});

// ---- form ----
function fieldRow(f) {
  const row = document.createElement("div");
  row.className = "field";
  const lab = document.createElement("label");
  lab.innerHTML = `<code>${f.name}</code>`;
  lab.title = f.help || "";
  row.appendChild(lab);
  let inp;
  if (f.widget === "bool") {
    inp = document.createElement("input"); inp.type = "checkbox";
    inp.checked = !!f.default; STATE[f.name] = !!f.default;
    inp.onchange = () => STATE[f.name] = inp.checked;
  } else if (f.widget === "choice") {
    inp = document.createElement("select");
    f.choices.forEach(c => { const o = document.createElement("option"); o.value = o.textContent = c; inp.appendChild(o); });
    inp.value = f.default ?? f.choices[0]; STATE[f.name] = inp.value;
    inp.onchange = () => STATE[f.name] = inp.value;
  } else {
    inp = document.createElement("input");
    inp.type = (f.widget === "int" || f.widget === "float") ? "number" : "text";
    if (f.widget === "float") inp.step = "any";
    const dv = Array.isArray(f.default) ? f.default.join(",") : (f.default ?? "");
    inp.value = dv; STATE[f.name] = f.default;
    inp.onchange = () => {
      let v = inp.value;
      if (f.widget === "int") v = v === "" ? null : parseInt(v);
      else if (f.widget === "float") v = v === "" ? null : parseFloat(v);
      else if (f.name === "backgrounds") v = v.split(",").map(s => s.trim()).filter(Boolean);
      STATE[f.name] = v;
    };
  }
  inp.dataset.field = f.name;
  row.appendChild(inp);
  // path fields get a "Browse…" button that opens the server-side picker modal
  if (f.path_kind && f.widget === "text") {
    const b = document.createElement("button");
    b.type = "button"; b.className = "browse-btn"; b.textContent = "Browse…";
    b.onclick = () => openPicker(inp, f.path_kind);
    row.appendChild(b);
  }
  if (f.help) { const h = document.createElement("span"); h.className = "hint"; h.textContent = f.help; row.appendChild(h); }
  return row;
}

function buildForm() {
  const host = $("#form"); host.innerHTML = "";
  SCHEMA.groups.forEach(g => {
    const fs = document.createElement("fieldset");
    const lg = document.createElement("legend"); lg.textContent = g.title; fs.appendChild(lg);
    g.fields.forEach(f => fs.appendChild(fieldRow(f)));
    host.appendChild(fs);
  });
}

function setForm(values) {        // fill inputs + STATE from a config dict
  $$("#form [data-field]").forEach(inp => {
    const name = inp.dataset.field;
    if (!(name in values)) return;
    let v = values[name];
    if (inp.type === "checkbox") { inp.checked = !!v; STATE[name] = !!v; }
    else { inp.value = Array.isArray(v) ? v.join(",") : (v ?? ""); STATE[name] = v; }
  });
  syncSegControls();              // keep the seg tuner aligned with the loaded recipe
}

// Seed the Segmentation-tuner controls from the loaded Run-form config so a check
// evaluates the SAME method the dataset will use (e.g. the Healthy preset -> rembg),
// not the standalone "auto" default. Controls stay editable for experimentation.
function syncSegControls() {
  if (STATE.sources != null)         $("#seg-sources").value  = STATE.sources || "";
  if (STATE.segment)                 $("#seg-method").value   = STATE.segment;
  if (STATE.seg_conf_thresh != null) $("#seg-thresh").value   = STATE.seg_conf_thresh;
  if (STATE.rembg_provider)          $("#seg-provider").value = STATE.rembg_provider;
}

function collectConfig() {
  const c = {};
  for (const [k, v] of Object.entries(STATE)) {
    if (v === null || v === "" || v === undefined) continue;
    c[k] = v;
  }
  return c;
}

// ---- path picker (server-side filesystem browser) ----
let PICKER = null;          // { inp, kind, cwd }

function openPicker(inp, kind, displayName) {
  PICKER = { inp, kind, cwd: null };
  const name = displayName || inp.dataset.field || inp.id || "path";
  $("#fs-title").textContent = `Select a ${kind === "file" ? "file" : "folder"} for `;
  $("#fs-title").appendChild(Object.assign(document.createElement("code"), { textContent: name }));
  // The new-name row works in both modes: name a new subfolder (dir) or a new file that
  // doesn't exist yet — e.g. a config path to save (file). Leave it blank to pick an
  // existing entry (click a file in file mode, or "Select this folder" in dir mode).
  $("#fs-newlabel").textContent = kind === "file" ? "New file name" : "New subfolder";
  $("#fs-newname").placeholder = kind === "file"
    ? "(optional) e.g. run_config.yaml" : "(optional) e.g. soybean_healthy";
  $("#fs-select").textContent = kind === "file" ? "Use this name" : "Select this folder";
  $("#fs-newname").value = "";
  $("#fs-overlay").style.display = "flex";
  // start at the field's current value if set, else the data root
  fsLoad((inp.value && inp.value.trim()) || SCHEMA.data_root);
}

function closePicker() { $("#fs-overlay").style.display = "none"; PICKER = null; }

async function fsLoad(path) {
  if (!PICKER) return;
  const list = $("#fs-list"); list.textContent = "loading…";
  let r;
  try {
    r = await api(`/api/fs/list?path=${encodeURIComponent(path)}&kind=${PICKER.kind}`);
  } catch (e) {
    list.textContent = "ERROR: " + (e.detail || JSON.stringify(e)); return;
  }
  PICKER.cwd = r.path;
  $("#fs-path").textContent = r.path;
  const up = $("#fs-up");
  up.disabled = !r.parent;
  up.onclick = r.parent ? () => fsLoad(r.parent) : null;
  list.textContent = "";
  if (!r.entries.length) { list.appendChild(Object.assign(document.createElement("div"), { className: "fs-empty muted", textContent: "(empty)" })); }
  r.entries.forEach(e => {
    const row = document.createElement("div");
    row.className = "fs-row" + (e.is_dir ? " dir" : " file");
    row.textContent = (e.is_dir ? "📁 " : "📄 ") + e.name;
    row.title = e.path;
    row.onclick = e.is_dir ? () => fsLoad(e.path)            // navigate into folders
                           : () => fsPick(e.path);           // pick a file (file mode)
    list.appendChild(row);
  });
}

function fsPick(path) {
  if (!PICKER) return;
  PICKER.inp.value = path;
  const f = PICKER.inp.dataset.field;          // Run-form fields drive STATE; the seg /
  if (f) STATE[f] = path;                       // browser / config inputs are read by id
  closePicker();
}

// boot-time wiring for the picker chrome (elements exist: script is at end of body)
$("#fs-cancel").onclick = closePicker;
$("#fs-overlay").onclick = (ev) => { if (ev.target === $("#fs-overlay")) closePicker(); };
$("#fs-select").onclick = () => {
  if (!PICKER || !PICKER.cwd) return;
  const extra = $("#fs-newname").value.trim().replace(/^\/+|\/+$/g, "");
  if (PICKER.kind === "file" && !extra) return;   // file mode: type a name, or click a file
  fsPick(extra ? PICKER.cwd.replace(/\/+$/, "") + "/" + extra : PICKER.cwd);
};

// ---- run ----
function setStatus(s) {
  const b = $("#run-status"); b.textContent = s; b.className = "badge " + s;
}
$("#run-btn").onclick = async () => {
  $("#run-log").textContent = ""; setStatus("running");
  $("#run-btn").disabled = true; $("#stop-btn").disabled = false;
  try {
    const { run_id } = await api("/api/runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: collectConfig() }),
    });
    openWS(run_id);
  } catch (e) {
    $("#run-log").textContent = "ERROR: " + (e.detail || JSON.stringify(e));
    setStatus("error"); $("#run-btn").disabled = false; $("#stop-btn").disabled = true;
  }
};
let CURRENT_RUN = null;
$("#stop-btn").onclick = () => CURRENT_RUN && api("/api/runs/" + CURRENT_RUN + "/stop", { method: "POST" });

function openWS(rid) {
  CURRENT_RUN = rid;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  WS = new WebSocket(`${proto}://${location.host}/ws/runs/${rid}`);
  const log = $("#run-log");
  const finish = (status) => { setStatus(status); $("#run-btn").disabled = false; $("#stop-btn").disabled = true; };
  WS.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "line") { log.textContent += m.line + "\n"; log.scrollTop = log.scrollHeight; }
    else if (m.type === "done") {
      finish(m.status);
      log.textContent += `\n--- finished (exit ${m.returncode}) ---\n`; log.scrollTop = log.scrollHeight;
    } else if (m.type === "error") {
      log.textContent += `\n--- error: ${m.msg} ---\n`; finish("error");
    }
  };
  WS.onerror = () => finish("error");
}

// ---- seg tuner ----
$("#seg-btn").onclick = async () => {
  $("#seg-report").innerHTML = "checking…"; $("#seg-grid").innerHTML = ""; $("#seg-log").textContent = "";
  const cfg = {
    ...STATE,                                 // inherit loaded preset / Run-form config
    sources: $("#seg-sources").value,         // (strip_shadow, shadow_sat_max, bg_tolerance,
    segment: $("#seg-method").value,          //  rembg_fallback ... so the check matches the
    seg_conf_thresh: parseFloat($("#seg-thresh").value),  // dataset); 4 controls override.
    rembg_provider: $("#seg-provider").value,
  };
  try {
    const r = await api("/api/seg/check", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: cfg, n: parseInt($("#seg-n").value) }),
    });
    renderSegReport(r.report);
    $("#seg-grid").innerHTML = r.images.map(u =>
      `<figure><img src="${u}"><figcaption>${u.split("/").pop()}</figcaption></figure>`).join("");
    $("#seg-log").textContent = r.log || "";
  } catch (e) { $("#seg-report").textContent = "ERROR: " + (e.detail || JSON.stringify(e)); }
};

function renderSegReport(rep) {
  const el = $("#seg-report");
  if (!rep) { el.textContent = "(no seg_report.json produced)"; return; }
  const s = rep.summary || {};
  let h = "";
  h += `<div class="muted">scanned ${s.scanned ?? "?"} · kept ${s.kept ?? "?"} · skipped ${s.skipped ?? "?"} · cache hits ${s.cache_hits ?? 0}</div>`;
  if (s.method_counts) h += "<div><b>kept by method:</b> " +
    Object.entries(s.method_counts).map(([k, v]) => `<span class="pill">${k}: ${v}</span>`).join("") + "</div>";
  if (s.skip_counts && Object.keys(s.skip_counts).length) h += "<div><b>skipped by reason:</b> " +
    Object.entries(s.skip_counts).map(([k, v]) => `<span class="pill skip">${k}: ${v}</span>`).join("") + "</div>";
  if (s.confidence) h += `<div class="muted">confidence min/median/mean: ${s.confidence.min} / ${s.confidence.median} / ${s.confidence.mean}</div>`;
  if (Array.isArray(rep.skipped) && rep.skipped.length)
    h += "<table><tr><th>source</th><th>reason</th><th>conf</th></tr>" +
      rep.skipped.slice(0, 40).map(x =>
        `<tr><td>${x.source}</td><td>${x.reason}</td><td>${x.confidence}</td></tr>`).join("") + "</table>";
  el.innerHTML = h;
}

// ---- browser ----
let DS_IDX = 0, DS_TOTAL = 0;
$("#ds-scan").onclick = async () => {
  const root = $("#ds-root").value.trim();
  const r = await api("/api/datasets" + (root ? "?root=" + encodeURIComponent(root) : ""));
  const sel = $("#ds-list"); sel.innerHTML = "";
  r.datasets.forEach(d => { const o = document.createElement("option"); o.value = o.textContent = d; sel.appendChild(o); });
  if (r.datasets.length) loadDataset();
};
$("#ds-list").onchange = loadDataset;
$("#ds-split").onchange = () => { DS_IDX = 0; showImage(); };
$("#ds-prev").onclick = () => { DS_IDX = Math.max(0, DS_IDX - 1); showImage(); };
$("#ds-next").onclick = () => { DS_IDX = Math.min(DS_TOTAL - 1, DS_IDX + 1); showImage(); };

async function loadDataset() {
  const path = $("#ds-list").value; if (!path) return;
  const info = await api("/api/datasets/info?path=" + encodeURIComponent(path));
  let h = `<b>${info.path}</b><br>`;
  h += "splits: " + Object.entries(info.splits).map(([k, v]) => `${k}=${v}`).join(", ") + "<br>";
  h += "classes (" + info.classes.length + "): " + info.classes.slice(0, 30).map(c => `<span class="pill">${c}</span>`).join("");
  $("#ds-info").innerHTML = h;
  $("#ds-qa").innerHTML = (info.qa || []).map(u =>
    `<figure><img src="${u}"><figcaption>${u.split("/").pop()}</figcaption></figure>`).join("");
  $("#ds-seg").innerHTML = info.seg_report ? "<b>seg_report:</b><br>" + segSummary(info.seg_report) : "";
  DS_IDX = 0; showImage();
}
function segSummary(rep) {
  const s = rep.summary || {};
  let h = "";
  if (s.method_counts) h += Object.entries(s.method_counts).map(([k, v]) => `<span class="pill">${k}: ${v}</span>`).join("");
  if (s.skip_counts) h += " " + Object.entries(s.skip_counts).map(([k, v]) => `<span class="pill skip">${k}: ${v}</span>`).join("");
  return h;
}
function showImage() {
  const path = $("#ds-list").value; if (!path) return;
  const split = $("#ds-split").value;
  const url = `/api/datasets/image?path=${encodeURIComponent(path)}&split=${split}&idx=${DS_IDX}`;
  const img = $("#ds-img");
  fetch(url).then(r => {
    if (!r.ok) throw new Error("no image");        // fetch() does not reject on 404
    DS_TOTAL = parseInt(r.headers.get("X-Total") || "0");
    const name = r.headers.get("X-Name") || "";
    $("#ds-caption").textContent = `${name}  (${DS_IDX + 1} / ${DS_TOTAL})`;
    return r.blob();
  }).then(b => img.src = URL.createObjectURL(b)).catch(() => $("#ds-caption").textContent = "(no image)");
}

// ---- config ----
$("#cfg-save").onclick = async () => {
  try {
    const r = await api("/api/config/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("#cfg-path").value, config: collectConfig() }),
    });
    $("#cfg-msg").textContent = "Saved -> " + r.saved;
  } catch (e) { $("#cfg-msg").textContent = "ERROR: " + (e.detail || JSON.stringify(e)); }
};
$("#cfg-load").onclick = async () => {
  try {
    const v = await api("/api/config/load?path=" + encodeURIComponent($("#cfg-path").value));
    setForm(v);
    $("#cfg-msg").textContent = "Loaded " + Object.keys(v).length + " keys into the Run form.";
  } catch (e) { $("#cfg-msg").textContent = "ERROR: " + (e.detail || JSON.stringify(e)); }
};

// ---- boot ----
api("/api/schema").then(s => {
  SCHEMA = s; buildForm();
  PRESETS = s.presets || {};
  // open the form in the recommended state so it's ready to generate a quality set
  if (s.recommended) setForm(s.recommended);
  // preset picker: switch the whole form to a named recipe
  const sel = $("#preset-sel");
  if (sel) {
    Object.keys(PRESETS).forEach(name => {
      const o = document.createElement("option"); o.value = o.textContent = name; sel.appendChild(o);
    });
    sel.onchange = () => { const v = PRESETS[sel.value]; if (v) setForm(v); };
  }
  $("#ds-root").placeholder = s.data_root;
  // seed the seg-tuner controls from the recommended/loaded config (method, thresh, source)
  syncSegControls();
  // Browse buttons on the other tabs' path inputs (read by id, not part of STATE)
  const wireBrowse = (btn, inp, kind) => {
    const b = $(btn), i = $(inp);
    if (b && i) b.onclick = () => openPicker(i, kind);
  };
  wireBrowse("#seg-sources-browse", "#seg-sources", "dir");
  wireBrowse("#ds-root-browse", "#ds-root", "dir");
  wireBrowse("#cfg-path-browse", "#cfg-path", "file");
});
