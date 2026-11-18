// app.js - "Evaluating AI-Generated Image Captions for Blind and Low-Vision Users" (Group 6, Naija Jollof)
// Reads window.DEMO_DATA (made by build_site_data.py). Runs the top navigation (one view at a time,
// driven by the address bar hash so Back/Forward work), draws the results tables and runs the explorer.
// Every score is written as text with a screen-reader label; colour is never the only signal.

const photos = window.DEMO_DATA.photos;
const RATERS = ["Success", "Ebere", "Yvonne", "Samuel"];
const MODELS = ["BLIP-2", "LLaVA-1.5"];
const MODEL_VAR = { "BLIP-2": "--blip", "LLaVA-1.5": "--llava" };
const GROUPS = { clear: "Clear", blurry: "Blurry", low_light: "Low light", text: "Contains text" };
const EH_WORDS = ["made something up", "nothing false, but inadequate", "accurate or honestly unsure"];
const TR_WORDS = ["ignored the text", "mentioned it, didn't read it right", "read the key text"];
const FAILURES = [
  ["Confident description of an unreadable photo", "Describes a photo too dark or blurry to support it"],
  ["Invented person or body part", "Invented a person or body part"],
  ["Invented setting or furniture", "Invented a setting or furniture"],
  ["Wrong or invented main object", "Wrong or invented main object"],
  ["Wrong colour, count or position", "Wrong colour, count or position"],
  ["Wrong text, brand or label", "Wrong text, brand or label"],
  ["Degenerate output", "Repeating (looping) output"],
  ["Vague or over-hedged (3+ of 4 gave 1)", "Vague or over-hedged"],
];
const FAILURE_NAME = Object.fromEntries(FAILURES);

const $ = (id) => document.getElementById(id);
const safe = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const words = (t) => String(t).trim().split(/\s+/).length;
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const allCaps = photos.flatMap((p) => p.captions.map((c) => ({ ...c, stratum: p.stratum })));

function announce(msg) {
  $("announcer").textContent = "";
  setTimeout(() => { $("announcer").textContent = msg; }, 50);
}

// ---------- results tables (computed from the data, so they always match the explorer) ----------
function barRow(label, values, max, fmt) {
  return `<tr><th scope="row">${label}</th>${MODELS.map((m, k) =>
    `<td class="bar-cell"><span class="hbar" style="--model: var(${MODEL_VAR[m]})"><i style="width:${(values[k] / max) * 80}%"></i>${fmt(values[k])}</span></td>`).join("")}</tr>`;
}
function drawResults() {
  const counts = FAILURES.map(([key]) => MODELS.map((m) => allCaps.filter((c) => c.failure === key && c.model === m).length));
  const max1 = Math.max(...counts.flat());
  $("taxonomy-table").tBodies[0].innerHTML = FAILURES.map(([key, name], i) =>
    barRow(name, counts[i], max1, (v) => v).replace("</tr>",
      `<td><button type="button" class="link-button" data-failure="${safe(key)}">See examples<span class="visually-hidden"> of ${safe(name)}</span></button></td></tr>`)).join("");
  $("taxonomy-table").querySelectorAll("[data-failure]").forEach((b) => b.addEventListener("click", () => {
    $("group-select").value = "all";
    $("show-select").value = b.dataset.failure;
    location.hash = "#explore";
    applyFilters();
  }));
  const strata = Object.keys(GROUPS).map((g) => MODELS.map((m) => mean(allCaps.filter((c) => c.stratum === g && c.model === m).map((c) => mean(c.human.eh)))));
  $("strata-table").tBodies[0].innerHTML = Object.values(GROUPS).map((name, i) => barRow(name, strata[i], 2, (v) => v.toFixed(2))).join("");
}

// ---------- how one caption's scores are written out ----------
function chips(list, wordsList) {
  const label = list.map((v, i) => `${RATERS[i]}: ${v}, ${wordsList[v]}`).join("; ");
  return `<span class="chips" role="img" aria-label="Scores out of 2. ${label}.">${list.map((v, i) =>
    `<span class="chip s${v}" title="${RATERS[i]}: ${wordsList[v]}">${v}</span>`).join("")}</span>
    <span class="score-sub">out of 2, one score per reviewer (${RATERS.join(", ")})</span>`;
}
function judgeScore(value, wordsList, reason) {
  if (value == null) return `<span class="score-sub">No score (the judge found no readable text)</span>`;
  return `<span class="score-text">${value} / 2</span> <span class="score-sub">${wordsList[value]}</span>${reason ? `<span class="judge-reason">“${safe(reason)}”</span>` : ""}`;
}
function scoreList(c, hasText) {
  let html = `<dl class="scores">
    <div><dt>Error Honesty · our reviewers</dt><dd>${chips(c.human.eh, EH_WORDS)}</dd></div>
    <div><dt>Error Honesty · Gemini judge</dt><dd>${judgeScore(c.judge.eh, EH_WORDS, c.judge.eh_reason)}</dd></div>`;
  if (hasText && c.human.tr) {
    html += `<div><dt>Text Recognition · our reviewers</dt><dd>${chips(c.human.tr, TR_WORDS)}</dd></div>
      <div><dt>Text Recognition · Gemini judge</dt><dd>${judgeScore(c.judge.tr, TR_WORDS, c.judge.tr_reason)}</dd></div>`;
  }
  html += `<div><dt>CIDEr · standard metric</dt><dd>${c.cider == null
    ? `<span class="score-sub">Not computed: this photo has no usable VizWiz reference captions</span>`
    : `<span class="score-text">${(c.cider * 100).toFixed(1)}</span> <span class="score-sub">(×100) similarity to the captions sighted people wrote</span>`}</dd></div>`;
  return html + `</dl>`;
}
function flags(c) {
  const out = [];
  if (c.failure) out.push(`<span class="flag fail">${safe(FAILURE_NAME[c.failure] || c.failure)}</span>`);
  if (c.tr_outcome) out.push(`<span class="flag">${safe(c.tr_outcome)}</span>`);
  if (c.cut_off) out.push(`<span class="flag">Cut off at the 64-token limit</span>`);
  return out.length ? `<p class="flags"><span class="visually-hidden">Labels: </span>${out.join("")}</p>` : "";
}

// ---------- explorer ----------
const disagrees = (c) => c.judge.eh != null && Math.abs(c.judge.eh - mean(c.human.eh)) >= 1;
const SHOW = [
  ["all", "All photos", () => true],
  ["disagree", "Judge disagrees with reviewers", (p) => p.captions.some(disagrees)],
  ...FAILURES.map(([key, name]) => [key, name, (p) => p.captions.some((c) => c.failure === key)]),
];
let shown = photos;
let index = 0;

function applyFilters() {
  const g = $("group-select").value;
  const test = SHOW.find((s) => s[0] === $("show-select").value)[2];
  shown = photos.filter((p) => (g === "all" || p.stratum === g) && test(p));
  fillPhotos();
  if (shown.length) showPhoto(0);
  else $("photo").innerHTML = `<p class="photo-missing" style="color:var(--ink-3)">No photos match these filters.</p>`;
  announce(`${shown.length} photos match.`);
}
function fillFilters() {
  const g = $("group-select");
  g.add(new Option(`All groups (${photos.length})`, "all"));
  for (const [k, label] of Object.entries(GROUPS)) g.add(new Option(`${label} (${photos.filter((p) => p.stratum === k).length})`, k));
  const s = $("show-select");
  for (const [k, label, test] of SHOW) s.add(new Option(`${label} (${photos.filter(test).length})`, k));
  g.addEventListener("change", applyFilters);
  s.addEventListener("change", applyFilters);
}
function fillPhotos() {
  const sel = $("photo-select");
  sel.innerHTML = "";
  shown.forEach((p, i) => sel.add(new Option(`${i + 1} of ${shown.length} · ${GROUPS[p.stratum]} · ${p.file_name}`, i)));
}

function showPhoto(i, focus = false) {
  index = Math.max(0, Math.min(i, shown.length - 1));
  const p = shown[index];
  $("photo-select").value = index;
  $("prev").disabled = index === 0;
  $("next").disabled = index === shown.length - 1;
  const hasText = Boolean(p.expected_text);
  const alt = `Photo taken by a blind or low-vision person, ${GROUPS[p.stratum].toLowerCase()} group. Its content is described by the captions below.`;

  $("photo").innerHTML = `
    <div class="photo-top">
      <figure class="photo-frame"><img src="${safe(p.image)}" alt="${alt}"></figure>
      <div class="photo-info">
        <span class="tag">${GROUPS[p.stratum]}</span>
        <h2 id="photo-heading">Photo ${index + 1} of ${shown.length}</h2>
        <p class="file">${safe(p.file_name)}</p>
        ${hasText ? `<div class="expected"><strong>Agreed key text in this photo</strong>${safe(p.expected_text)}</div>` : ""}
      </div>
    </div>
    <div class="captions">
      ${p.captions.map((c) => `
        <section class="caption" style="--model: var(${MODEL_VAR[c.model]})" aria-label="${c.model} caption">
          <h3><i aria-hidden="true"></i>${c.model}</h3>
          <blockquote>${safe(c.caption)}</blockquote>
          <p class="words">${words(c.caption)} words · caption ${safe(c.id)}</p>
          ${flags(c)}
          ${scoreList(c, hasText)}
          <button type="button" class="speak" data-text="${safe(c.caption)}">Read aloud</button>
        </section>`).join("")}
    </div>`;

  const img = $("photo").querySelector("img");
  img.addEventListener("error", () => {
    img.closest("figure").innerHTML = `<p class="photo-missing">Photo not available. Copy the 250 sample photos into the images folder.</p>`;
  });
  $("photo").querySelectorAll(".speak").forEach((b) => b.addEventListener("click", () => {
    if (!("speechSynthesis" in window)) return;
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(b.dataset.text));
  }));
  if (focus) { $("photo").focus(); announce(`Photo ${index + 1} of ${shown.length}, ${GROUPS[p.stratum]}.`); }
}

$("photo-select").addEventListener("change", () => showPhoto(Number($("photo-select").value)));
$("prev").addEventListener("click", () => showPhoto(index - 1, true));
$("next").addEventListener("click", () => showPhoto(index + 1, true));
$("random").addEventListener("click", () => showPhoto(Math.floor(Math.random() * shown.length), true));

// ---------- top navigation: one view at a time ----------
const VIEWS = ["overview", "findings", "failures", "explore", "method", "team"];
function route(focus) {
  const name = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== name; });
  document.querySelectorAll("#site-nav a").forEach((a) => {
    if (a.getAttribute("href") === `#${name}`) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  $("site-nav").classList.remove("open");
  $("menu-button").setAttribute("aria-expanded", "false");
  const heading = $(`h-${name}`);
  document.title = `${heading.textContent} · Evaluating AI-Generated Image Captions for BLV Users`;
  if (focus) { window.scrollTo(0, 0); heading.setAttribute("tabindex", "-1"); heading.focus({ preventScroll: true }); }
}
window.addEventListener("hashchange", () => route(true));
$("menu-button").addEventListener("click", () => {
  const open = $("site-nav").classList.toggle("open");
  $("menu-button").setAttribute("aria-expanded", String(open));
});

drawResults();
fillFilters();
fillPhotos();
showPhoto(0);
route(false);
