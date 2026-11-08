// The profile menu and the Preferences dialog (style guide layout profile-menu, STANDARD.md section 5).
// A choice applies at once and is saved on the server; a failed save reverts it.
"use strict";

window.Preferences = (() => {
  const $ = (id) => document.getElementById(id);
  const SCALES = [0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2];
  let cat = null;       // type.json
  let themes = [];      // theme ids from vendor.json
  let prefs = null;     // the stored object

  const themeName = (id) => (id.charAt(0).toUpperCase() + id.slice(1)).replace(/-/g, " ");

  async function catalogue() {
    if (cat) return;
    const [vendor, type] = await Promise.all([
      fetch("style/vendor.json").then((r) => r.json()),
      fetch("style/type.json").then((r) => r.json()),
    ]);
    themes = Object.keys(vendor.themes);
    cat = type;
  }

  // ---- The profile menu (WAI-ARIA APG menu button) ------------------------

  const items = () => [...$("profileMenu").querySelectorAll('[role="menuitem"]')];

  function openMenu(focusLast) {
    $("profileMenu").hidden = false;
    $("avatar").setAttribute("aria-expanded", "true");
    const list = items();
    (focusLast ? list[list.length - 1] : list[0]).focus();
  }

  function closeMenu(returnFocus) {
    if ($("profileMenu").hidden) return;
    $("profileMenu").hidden = true;
    $("avatar").setAttribute("aria-expanded", "false");
    if (returnFocus) $("avatar").focus();
  }

  $("avatar").addEventListener("click", () => {
    if ($("profileMenu").hidden) openMenu(false);
    else closeMenu(true);
  });
  $("avatar").addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      openMenu(ev.key === "ArrowUp");
    }
  });
  $("profileMenu").addEventListener("keydown", (ev) => {
    const list = items();
    const i = list.indexOf(document.activeElement);
    const go = (n) => { ev.preventDefault(); list[(n + list.length) % list.length].focus(); };
    if (ev.key === "ArrowDown") go(i + 1);
    else if (ev.key === "ArrowUp") go(i - 1);
    else if (ev.key === "Home") go(0);
    else if (ev.key === "End") go(list.length - 1);
    else if (ev.key === "Escape") { ev.preventDefault(); closeMenu(true); }
    else if (ev.key === "Tab") closeMenu(false);
  });
  document.addEventListener("click", (ev) => {
    if (!$("profile").contains(ev.target)) closeMenu(false);
  });

  $("openPrefs").addEventListener("click", () => {
    closeMenu(false);
    openDialog();
  });

  // ---- The Preferences dialog ---------------------------------------------

  function option(select, value, label) {
    const o = document.createElement("option");
    o.value = String(value);
    o.textContent = label;
    select.append(o);
  }

  // The theme's own value of a token, read from a hidden preview element
  // (the inline overrides on <html> do not reach it).
  function themeValue(token) {
    const probe = $("themeProbe");
    probe.dataset.preset = window.Appearance.resolveTheme(prefs.theme);
    return getComputedStyle(probe).getPropertyValue(token).trim();
  }

  function fontLabel(stack) {
    const first = stack.split(",")[0].replace(/["']/g, "").trim();
    const choice = cat.choices.find((c) => c.stack.split(",")[0].replace(/["']/g, "").trim() === first);
    return choice ? choice.label : first;
  }

  function fillType() {
    const t = prefs.type;
    const head = $("typeHead");
    const body = $("typeBody");
    const text = $("typeText");
    const heading = $("typeHeading");
    const label = $("typeLabel");
    [head, body, text, heading, label].forEach((s) => s.replaceChildren());
    option(head, "theme", `Theme default (${fontLabel(themeValue("--font-head"))})`);
    option(body, "theme", `Theme default (${fontLabel(themeValue("--font-sans"))})`);
    cat.choices.forEach((c) => {
      if (c.roles.includes("head")) option(head, c.id, c.label);
      if (c.roles.includes("body")) option(body, c.id, c.label);
    });
    option(text, 0, `Theme default (${themeValue("--body-size")})`);
    for (let n = cat.sizes.text.min; n <= cat.sizes.text.max; n++) option(text, n, `${n} px`);
    option(heading, "theme", `Theme default (${themeValue("--h1-size")} / ${themeValue("--h2-size")} / ${themeValue("--h3-size")})`);
    cat.headings.forEach((h) => option(heading, h.id, `${h.label} (${h.h1} / ${h.h2} / ${h.h3} px)`));
    option(label, 0, `Theme default (${themeValue("--label-size")})`);
    for (let n = cat.sizes.label.min; n <= cat.sizes.label.max; n++) option(label, n, `${n} px`);
    head.value = t.head;
    body.value = t.body;
    text.value = String(t.text);
    heading.value = t.heading;
    label.value = String(t.label);
  }

  function themeCard(id, title, previewId) {
    const label = document.createElement("label");
    label.className = "theme-card";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "theme";
    input.value = id;
    const preview = document.createElement("span");
    preview.className = "preset-preview swatch";
    preview.dataset.preset = previewId;
    preview.setAttribute("aria-hidden", "true");
    const head = document.createElement("span");
    head.className = "swatch-head";
    head.textContent = "Aa";
    const bar = document.createElement("span");
    bar.className = "swatch-bar";
    preview.append(head, bar);
    const name = document.createElement("span");
    name.className = "theme-name";
    name.textContent = title;
    label.append(input, preview, name);
    return label;
  }

  function fill() {
    const cards = $("themeCards");
    cards.replaceChildren(themeCard("auto", "Automatic", window.Appearance.resolveTheme("auto")));
    themes.forEach((id) => cards.append(themeCard(id, themeName(id), id)));
    cards.querySelector(`input[value="${CSS.escape(prefs.theme)}"]`).checked = true;
    const autoTheme = themeName(window.Appearance.resolveTheme("auto"));
    $("themeAuto").textContent = window.Appearance.contrastQuery.matches
      ? `Automatic uses ${autoTheme}: your system asks for more contrast.`
      : `Automatic uses ${autoTheme}, or High contrast when your system asks for more contrast.`;
    document.querySelector(`#modeGroup input[value="${prefs.mode}"]`).checked = true;
    const scale = $("scale");
    scale.replaceChildren();
    SCALES.forEach((s) => option(scale, s, `${Math.round(s * 100)}%`));
    scale.value = String(prefs.scale);
    fillType();
  }

  function status(text) { $("prefsStatus").textContent = text; }

  async function save(change) {
    const before = prefs;
    const next = { ...prefs, ...change, type: { ...prefs.type, ...(change.type || {}) } };
    prefs = next;
    window.Appearance.apply(prefs, cat);
    try {
      const res = await fetch("/api/preferences", {
        method: "PUT",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(change),
      });
      if (!res.ok) {
        let detail = `error ${res.status}`;
        try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
        throw new Error(detail);
      }
      prefs = await res.json();
      window.Appearance.apply(prefs, cat);
      status("Saved on this server.");
    } catch (e) {
      prefs = before;
      window.Appearance.apply(prefs, cat);
      status(`Not saved: ${e.message}`);
    }
    fill();
  }

  $("themeCards").addEventListener("change", (ev) => save({ theme: ev.target.value }));
  $("modeGroup").addEventListener("change", (ev) => save({ mode: ev.target.value }));
  $("scale").addEventListener("change", (ev) => save({ scale: Number(ev.target.value) }));
  $("typeHead").addEventListener("change", (ev) => save({ type: { head: ev.target.value } }));
  $("typeBody").addEventListener("change", (ev) => save({ type: { body: ev.target.value } }));
  $("typeText").addEventListener("change", (ev) => save({ type: { text: Number(ev.target.value) } }));
  $("typeHeading").addEventListener("change", (ev) => save({ type: { heading: ev.target.value } }));
  $("typeLabel").addEventListener("change", (ev) => save({ type: { label: Number(ev.target.value) } }));

  async function openDialog() {
    if (!prefs) await load();
    if (!prefs) {
      $("warning").textContent = "Could not read the preferences from the server.";
      $("warning").hidden = false;
      return;
    }
    fill();
    status("Changes apply at once and are saved on this server.");
    $("prefs").showModal();
    $("prefsTitle").focus();
  }

  $("closePrefs").addEventListener("click", () => $("prefs").close());
  $("prefs").addEventListener("close", () => $("avatar").focus());

  // After sign-in: show the menu and apply the server's record.
  async function load() {
    $("profile").hidden = false;
    try {
      await catalogue();
      const res = await fetch("/api/preferences", { credentials: "same-origin" });
      if (!res.ok) return;
      prefs = await res.json();
      window.Appearance.apply(prefs, cat);
    } catch (_) { /* keep the cached look; the dialog retries */ }
  }

  function signedOut() {
    closeMenu(false);
    $("profile").hidden = true;
  }

  return { load, signedOut };
})();
