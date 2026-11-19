// app.js
// Everything the website does is in this file.
// data.js (made by build_site_data.py) loads first and puts all our results in window.DEMO_DATA.
// This file reads that data and:
//   1. builds the two results tables (failure types, and honesty by photo group)
//   2. runs the photo explorer (filters, photo list, Previous/Next, Read aloud)
//   3. runs the top menu so only one section shows at a time
// Rule we kept everywhere: a score is always written out in words, never shown by
// colour only, so someone using a screen reader gets the same information.

// ---------- setup: the data and the labels we reuse ----------

// all 250 photos, each with its 2 captions and every score
const photos = window.DEMO_DATA.photos;
// same order as the score columns in our review sheets
const RATERS = ["Success", "Ebere", "Yvonne", "Samuel"];
const MODELS = ["BLIP-2", "LLaVA-1.5"];
// each model has its own colour, set as a CSS variable in style.css
const MODEL_VAR = { "BLIP-2": "--blip", "LLaVA-1.5": "--llava" };
// our 4 photo groups: name in the data -> name shown on the site
const GROUPS = { clear: "Clear", blurry: "Blurry", low_light: "Low light", text: "Contains text" };
// what each score means in plain words, taken from our rubric
// (position 0 = score 0, position 1 = score 1, position 2 = score 2)
const EH_WORDS = ["made something up", "nothing false, but inadequate", "accurate or honestly unsure"];
const TR_WORDS = ["ignored the text", "mentioned it, didn't read it right", "read the key text"];
// failure types from the taxonomy: [name in the data, shorter name for the site]
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
// turns that list into a lookup: data name -> site name
const FAILURE_NAME = Object.fromEntries(FAILURES);

// ---------- small helpers ----------

// shortcut so we can write $("prev") instead of document.getElementById("prev")
const $ = (id) => document.getElementById(id);
// escapes & < > and " so caption text can't break the page's HTML
// (some captions have quotes and symbols in them)
const safe = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// counts the words in a caption (BLIP-2 averages about 9, LLaVA about 48)
const words = (t) => String(t).trim().split(/\s+/).length;
// average of a list of numbers, used on the 4 reviewer scores
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
// one flat list of all 500 captions, each tagged with its photo group,
// so the tables can count things without going photo by photo
const allCaps = photos.flatMap((p) => p.captions.map((c) => ({ ...c, stratum: p.stratum })));

// tells screen readers about a change, like "12 photos match".
// #announcer is an aria-live area in index.html. We empty it first and add the new text
// a moment later, because some screen readers won't repeat a message that looks unchanged.
function announce(msg) {
  $("announcer").textContent = "";
  setTimeout(() => { $("announcer").textContent = msg; }, 50);
}

// ---------- results tables ----------
// These get calculated from data.js when the page loads instead of being typed in by hand,
// so they can never disagree with what the explorer shows.

// builds one table row: the label, then a bar and a number for each model.
// bar width = value divided by the biggest value (max), so the longest bar fills 80%
function barRow(label, values, max, fmt) {
  return `<tr><th scope="row">${label}</th>${MODELS.map((m, k) =>
    `<td class="bar-cell"><span class="hbar" style="--model: var(${MODEL_VAR[m]})"><i style="width:${(values[k] / max) * 80}%"></i>${fmt(values[k])}</span></td>`).join("")}</tr>`;
}
function drawResults() {
// failure table: how many captions from each model got each failure type
  const counts = FAILURES.map(([key]) => MODELS.map((m) => allCaps.filter((c) => c.failure === key && c.model === m).length));
// the biggest count, used to size the bars
  const max1 = Math.max(...counts.flat());
// one row per failure type, with a "See examples" button at the end
  $("taxonomy-table").tBodies[0].innerHTML = FAILURES.map(([key, name], i) =>
    barRow(name, counts[i], max1, (v) => v).replace("</tr>",
      `<td><button type="button" class="link-button" data-failure="${safe(key)}">See examples<span class="visually-hidden"> of ${safe(name)}</span></button></td></tr>`)).join("");
// "See examples" jumps to the explorer with that failure type already picked
  $("taxonomy-table").querySelectorAll("[data-failure]").forEach((b) => b.addEventListener("click", () => {
    $("group-select").value = "all";
    $("show-select").value = b.dataset.failure;
    location.hash = "#explore";
    applyFilters();
  }));
// honesty by photo group: for each group and model, average our mean EH scores
  const strata = Object.keys(GROUPS).map((g) => MODELS.map((m) => mean(allCaps.filter((c) => c.stratum === g && c.model === m).map((c) => mean(c.human.eh)))));
// scores are out of 2, so 2 is the max for these bars
  $("strata-table").tBodies[0].innerHTML = Object.values(GROUPS).map((name, i) => barRow(name, strata[i], 2, (v) => v.toFixed(2))).join("");
}

// ---------- how one caption's scores are written out ----------
// the 4 reviewer scores shown as small boxes ("chips").
// role="img" with an aria-label makes a screen reader read all 4 scores in words
// (e.g. "Success: 0, made something up; ...") instead of just reading four numbers.
function chips(list, wordsList) {
  const label = list.map((v, i) => `${RATERS[i]}: ${v}, ${wordsList[v]}`).join("; ");
  return `<span class="chips" role="img" aria-label="Scores out of 2. ${label}.">${list.map((v, i) =>
    `<span class="chip s${v}" title="${RATERS[i]}: ${wordsList[v]}">${v}</span>`).join("")}</span>
    <span class="score-sub">out of 2, one score per reviewer (${RATERS.join(", ")})</span>`;
}
// the Gemini judge's score, what it means, and the judge's written reason.
// value is null when the judge decided a photo has no readable text (Text Recognition only)
function judgeScore(value, wordsList, reason) {
  if (value == null) return `<span class="score-sub">No score (the judge found no readable text)</span>`;
  return `<span class="score-text">${value} / 2</span> <span class="score-sub">${wordsList[value]}</span>${reason ? `<span class="judge-reason">“${safe(reason)}”</span>` : ""}`;
}
// the full score panel for one caption, in this order:
//   Error Honesty (us, then the judge)
//   Text Recognition (only on text photos where all 4 of us gave a TR score)
//   CIDEr, stored as a raw number (0 to about 2.5 in our data) and multiplied by 100 to
//   match our slides and poster.
// 33 photos have no CIDEr because all 5 VizWiz annotators marked them unrecognisable.
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
// small labels under a caption: its failure type, how it did on the key text,
// and whether it got cut off by our 64-token generation limit (mostly LLaVA)
function flags(c) {
  const out = [];
  if (c.failure) out.push(`<span class="flag fail">${safe(FAILURE_NAME[c.failure] || c.failure)}</span>`);
  if (c.tr_outcome) out.push(`<span class="flag">${safe(c.tr_outcome)}</span>`);
  if (c.cut_off) out.push(`<span class="flag">Cut off at the 64-token limit</span>`);
  return out.length ? `<p class="flags"><span class="visually-hidden">Labels: </span>${out.join("")}</p>` : "";
}

// ---------- photo explorer ----------
// a caption counts as a "disagreement" when the judge's EH score is 1 or more
// away from our average, e.g. the judge gave 2 and we gave 0, 0, 0, 0
const disagrees = (c) => c.judge.eh != null && Math.abs(c.judge.eh - mean(c.human.eh)) >= 1;
// options for the "Show" dropdown: [value, label, test that says if a photo matches]
const SHOW = [
  ["all", "All photos", () => true],
  ["disagree", "Judge disagrees with reviewers", (p) => p.captions.some(disagrees)],
  ...FAILURES.map(([key, name]) => [key, name, (p) => p.captions.some((c) => c.failure === key)]),
];
// shown = photos that match the current filters, index = the one on screen
let shown = photos;
let index = 0;

// runs when a filter changes: keep the matching photos, refill the photo list,
// show the first match, and tell screen readers how many matched
function applyFilters() {
  const g = $("group-select").value;
  const test = SHOW.find((s) => s[0] === $("show-select").value)[2];
  shown = photos.filter((p) => (g === "all" || p.stratum === g) && test(p));
  fillPhotos();
  if (shown.length) showPhoto(0);
  else $("photo").innerHTML = `<p class="photo-missing" style="color:var(--ink-3)">No photos match these filters.</p>`;
  announce(`${shown.length} photos match.`);
}
// fills the "Photo group" and "Show" dropdowns, with a count next to each option
function fillFilters() {
  const g = $("group-select");
  g.add(new Option(`All groups (${photos.length})`, "all"));
  for (const [k, label] of Object.entries(GROUPS)) g.add(new Option(`${label} (${photos.filter((p) => p.stratum === k).length})`, k));
  const s = $("show-select");
  for (const [k, label, test] of SHOW) s.add(new Option(`${label} (${photos.filter(test).length})`, k));
  g.addEventListener("change", applyFilters);
  s.addEventListener("change", applyFilters);
}
// fills the "Photo" dropdown with the photos that are currently shown
function fillPhotos() {
  const sel = $("photo-select");
  sel.innerHTML = "";
  shown.forEach((p, i) => sel.add(new Option(`${i + 1} of ${shown.length} · ${GROUPS[p.stratum]} · ${p.file_name}`, i)));
}

// draws one photo with both captions and all their scores.
// focus is true when the user moved with Previous, Next or Random. Then we move keyboard
// focus to the photo and announce it (we skip that on first load).
function showPhoto(i, focus = false) {
// keep i in range so we never go before the first or past the last photo
  index = Math.max(0, Math.min(i, shown.length - 1));
  const p = shown[index];
  $("photo-select").value = index;
// turn off Previous on the first photo and Next on the last one
  $("prev").disabled = index === 0;
  $("next").disabled = index === shown.length - 1;
// only the 51 text photos have agreed key text
  const hasText = Boolean(p.expected_text);
// alt text: the captions are exactly what we're testing, so instead of writing our own
// description the alt text says what the photo is and points to the captions
  const alt = `Photo taken by a blind or low-vision person, ${GROUPS[p.stratum].toLowerCase()} group. Its content is described by the captions below.`;

// build the photo card: photo and info on top, then one box per model's caption
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

// if a photo file is missing, show a message instead of a broken image
  const img = $("photo").querySelector("img");
  img.addEventListener("error", () => {
    img.closest("figure").innerHTML = `<p class="photo-missing">Photo not available. Copy the 250 sample photos into the images folder.</p>`;
  });
// Read aloud uses the browser's built-in speech (Web Speech API), no outside service.
// cancel() first so clicking twice doesn't play two voices on top of each other
  $("photo").querySelectorAll(".speak").forEach((b) => b.addEventListener("click", () => {
    if (!("speechSynthesis" in window)) return;
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(b.dataset.text));
  }));
  if (focus) { $("photo").focus(); announce(`Photo ${index + 1} of ${shown.length}, ${GROUPS[p.stratum]}.`); }
}

// hook up the photo dropdown and the Previous / Next / Random buttons
$("photo-select").addEventListener("change", () => showPhoto(Number($("photo-select").value)));
$("prev").addEventListener("click", () => showPhoto(index - 1, true));
$("next").addEventListener("click", () => showPhoto(index + 1, true));
$("random").addEventListener("click", () => showPhoto(Math.floor(Math.random() * shown.length), true));

// ---------- top menu: one section at a time ----------
// Each section has its own address, like .../#findings. Changing the part after the #
// doesn't reload the page. It fires a "hashchange" event and route() shows that section.
// So the whole site is one page, but the Back button and bookmarks still work.
const VIEWS = ["overview", "findings", "failures", "explore", "method", "team"];
function route(focus) {
// read the section name from the address bar, use overview if it isn't one of ours
  const name = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "overview";
// hide every section except the one we want
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.dataset.view !== name; });
// mark the current menu link with aria-current so screen readers say "current page"
  document.querySelectorAll("#site-nav a").forEach((a) => {
    if (a.getAttribute("href") === `#${name}`) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
// on phones, close the menu after a section is picked
  $("site-nav").classList.remove("open");
  $("menu-button").setAttribute("aria-expanded", "false");
// update the tab title and move focus to the section heading,
// so a screen reader user hears where they landed
  const heading = $(`h-${name}`);
  document.title = `${heading.textContent} · Evaluating AI-Generated Image Captions for BLV Users`;
  if (focus) { window.scrollTo(0, 0); heading.setAttribute("tabindex", "-1"); heading.focus({ preventScroll: true }); }
}
window.addEventListener("hashchange", () => route(true));
// the Menu button (only shows on small screens) opens and closes the menu
$("menu-button").addEventListener("click", () => {
  const open = $("site-nav").classList.toggle("open");
  $("menu-button").setAttribute("aria-expanded", String(open));
});

// ---------- start ----------
// build everything once when the page loads
drawResults();
fillFilters();
fillPhotos();
showPhoto(0);
route(false);
