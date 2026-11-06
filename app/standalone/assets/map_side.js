/* The map's side (style.css reads --map-side): the height under the top bar, measured on load, once the layout
   stands, once the fonts are in and on resize, so the map is a square at the full height of the workspace and the
   right column takes the same height; in portrait and narrow windows (style.css's rules) the map takes the window's
   width instead. CSS cannot size a column by a row's height, so Dash serves this with the page's assets. A change of
   the side dispatches a resize event, so plotly's responsive figures follow it. */
(function () {
  "use strict";

  const NARROW = 1000;
  const LEAST = 200;
  let last = null;
  let pending = null;

  function setMapSide() {
    const root = document.documentElement;
    const workspace = document.querySelector(".workspace");
    const shell = document.querySelector(".app-shell");
    const height = root.clientHeight;  // the layout viewport, what 100vh measures
    const width = root.clientWidth;
    const portrait = height > width || width <= NARROW;
    let side;
    if (portrait || !workspace || !shell) {
      side = Math.round(width - 32);
    } else {
      const top = workspace.getBoundingClientRect().top + window.scrollY;
      const bottom = parseFloat(getComputedStyle(shell).paddingBottom) || 0;
      side = Math.max(LEAST, Math.floor(height - top - bottom));
    }
    if (side === last) {
      return;
    }
    last = side;
    root.style.setProperty("--map-side", `${side}px`);
    window.dispatchEvent(new Event("resize"));
    window.requestAnimationFrame(function () {
      const plotly = window.Plotly;
      if (plotly && plotly.Plots && typeof plotly.Plots.resize === "function") {
        document.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
          plotly.Plots.resize(plot);
        });
      }
    });
  }

  function soon() {
    if (pending !== null) {
      return;
    }
    pending = window.setTimeout(function () {
      pending = null;
      setMapSide();
    }, 50);
  }

  window.addEventListener("resize", function () {
    const workspace = document.querySelector(".workspace");
    if (workspace) {
      setMapSide();
    }
  });
  window.addEventListener("load", soon);
  document.addEventListener("DOMContentLoaded", soon);
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(soon);
  }
  new MutationObserver(soon).observe(document.documentElement, { childList: true, subtree: true });
  if (window.ResizeObserver) {
    new ResizeObserver(soon).observe(document.documentElement);  // the viewport's size, resize event or not
  }
  soon();
})();
