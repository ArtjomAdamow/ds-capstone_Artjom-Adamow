/* The results tables' rows (layout.results_table, class measure-row): a click on a row sends its key
   (data-measure) to the page's callback through the hidden input #measure-click, the close box of the open
   description (measure-close) sends null; the server keeps the one open row in the view store (app.Page.step: a
   second click on the open row closes it too). The value is a JSON of the key and a sequence number, so every click
   is a new value. The transport is the viewer's: dash_clientside.set_props, else the native setter and an input
   event. Dash serves this with the page's assets. */
(function () {
  "use strict";

  const INPUT_ID = "measure-click";
  let seq = 0;

  function send(measure) {
    seq += 1;
    const value = JSON.stringify({ measure: measure, seq: seq });
    const dash = window.dash_clientside;
    if (dash && typeof dash.set_props === "function") {
      dash.set_props(INPUT_ID, { value: value });
      return;
    }
    const input = document.querySelector("input#" + INPUT_ID + ", #" + INPUT_ID + " input");
    if (!input) {
      return;
    }
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  document.addEventListener("click", function (event) {
    const target = event.target;
    if (!target || typeof target.closest !== "function") {
      return;
    }
    if (target.closest(".measure-close")) {
      event.preventDefault();
      send(null);
      return;
    }
    const row = target.closest(".measure-row");
    if (row && row.dataset && row.dataset.measure) {
      send(row.dataset.measure);
    }
  });
})();
