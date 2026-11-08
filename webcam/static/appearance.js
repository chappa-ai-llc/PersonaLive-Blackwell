// Appearance: theme, light or dark, UI scale and type (style guide sections 4, 5, 5.1, 7, 8.1).
// Loaded in <head> without defer, so the cached choice paints before the page shows.
// The server holds the record (GET/PUT /api/preferences); localStorage is only that cache.
"use strict";

window.Appearance = (() => {
  const CACHE = "personalive.appearance";
  const DEFAULT_THEME = "modernist";
  const TYPE_PROPS = ["--font-head", "--font-sans", "--label-size", "--label-size-lg",
    "--h1-size", "--h2-size", "--h3-size", "--body-size"];
  const contrastQuery = window.matchMedia("(prefers-contrast: more)");
  const root = document.documentElement;

  // "auto" (no stored theme) follows the system's contrast request.
  function resolveTheme(theme) {
    if (theme !== "auto") return theme;
    return contrastQuery.matches ? "high-contrast" : DEFAULT_THEME;
  }

  // The inline custom properties the type choices set; {} for "theme default".
  function typeProps(type, cat) {
    const props = {};
    if (!cat || !type) return props;
    const stack = (id) => (cat.choices.find((c) => c.id === id) || {}).stack;
    if (type.head !== "theme" && stack(type.head)) props["--font-head"] = stack(type.head);
    if (type.body !== "theme" && stack(type.body)) props["--font-sans"] = stack(type.body);
    if (type.label) {
      props["--label-size"] = `${type.label}px`;
      props["--label-size-lg"] = `${type.label + 1}px`;
    }
    const h = cat.headings.find((x) => x.id === type.heading);
    if (h) {
      props["--h1-size"] = `${h.h1}px`;
      props["--h2-size"] = `${h.h2}px`;
      props["--h3-size"] = `${h.h3}px`;
    }
    if (type.text) props["--body-size"] = `${type.text}px`;
    return props;
  }

  function paint(prefs, props) {
    root.dataset.preset = resolveTheme(prefs.theme || "auto");
    if (prefs.mode === "light" || prefs.mode === "dark") root.dataset.theme = prefs.mode;
    else delete root.dataset.theme;
    const scale = Number(prefs.scale) || 1;
    root.style.setProperty("--zoom", String(scale));
    root.style.zoom = scale === 1 ? "" : String(scale);
    for (const p of TYPE_PROPS) {
      if (props[p]) root.style.setProperty(p, props[p]);
      else root.style.removeProperty(p);
    }
  }

  let current = { theme: "auto", mode: "system", scale: 1 };

  function apply(prefs, cat) {
    current = prefs;
    const props = typeProps(prefs.type, cat);
    paint(prefs, props);
    try {
      localStorage.setItem(CACHE, JSON.stringify({ prefs, props }));
    } catch (_) { /* storage blocked: the server copy still applies on sign-in */ }
  }

  try {
    const cached = JSON.parse(localStorage.getItem(CACHE) || "null");
    if (cached && cached.prefs) {
      current = cached.prefs;
      paint(cached.prefs, cached.props || {});
    }
  } catch (_) { /* no cache: the default theme in the markup applies */ }

  contrastQuery.addEventListener("change", () => {
    if ((current.theme || "auto") === "auto") root.dataset.preset = resolveTheme("auto");
  });

  return { apply, resolveTheme, contrastQuery, DEFAULT_THEME };
})();
