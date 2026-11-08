/* A click in the dashboard's diagram: on an item (.diagram-item, its row in data-row, its id in data-dot, a topic's
   deck in data-deck), the item goes to the page's callback through the hidden input #diagram-click
   (app.dashboard.layout.DIAGRAM_INPUT) as JSON, which the server reads as the selection of a card or the opening of
   a deck or a topic (app.dashboard.app.Dashboard.step). The JSON carries a count, so every click is a new value. The
   listener sits on the document, since the callback draws the diagram's items anew. The transport is keys.js's:
   dash_clientside.set_props, else the native setter and an input event. Dash serves this with the dashboard's
   assets. */
(function () {
  "use strict";

  const INPUT_ID = "diagram-click";
  let count = 0;

  function send(value) {
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
    const item = event.target && event.target.closest ? event.target.closest(".diagram-item") : null;
    if (!item || !item.closest("#diagram")) {
      return;
    }
    count += 1;
    send(JSON.stringify({ row: item.dataset.row, id: item.dataset.dot, deck: item.dataset.deck || null, count: count }));
  });
})();
