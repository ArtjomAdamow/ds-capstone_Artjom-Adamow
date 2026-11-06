/* The page's clickable rows: a generation's tab (layout.level_tabs, class level-tab) sends its generation
   (data-level), a results table's row in Details (layout.results_table, class measure-row) its key (data-measure),
   the close box of the open description (measure-close) a null measure, one of P1's targets (layout.prompt_answer,
   class prompt-target) the deck, the topic and the card it shows (data-target); all go to the page's callback through the
   hidden input #row-click, which the server reads (app.Page.step: the open generation and the one open row live
   in the view store; a second click on the open row closes it too). The value is a JSON of the generation or the key
   and a sequence number, so every click is a new value. The transport is the viewer's:
   dash_clientside.set_props, else the native setter and an input event. Dash serves this with the page's assets. */
(function () {
  "use strict";

  const INPUT_ID = "row-click";
  let seq = 0;

  function send(payload) {
    seq += 1;
    const value = JSON.stringify(Object.assign({ seq: seq }, payload));
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
    const level = target.closest(".level-tab");
    if (level && level.dataset && level.dataset.level) {
      send({ level: level.dataset.level });
      return;
    }
    if (target.closest(".measure-close")) {
      event.preventDefault();
      send({ measure: null });
      return;
    }
    const row = target.closest(".measure-row");
    if (row && row.dataset && row.dataset.measure) {
      send({ measure: row.dataset.measure });
      return;
    }
    const goal = target.closest(".prompt-target");
    if (goal && goal.dataset && goal.dataset.target) {
      send({ target: goal.dataset.target });
    }
  });
})();
