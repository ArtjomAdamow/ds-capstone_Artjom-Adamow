/* Enter on the dashboard: outside the request box (its own Enter submits the request), the buttons and the links,
   Enter goes to the page's callback through the hidden input #key-enter (app.dashboard.layout.KEY_INPUT), which the
   server reads as the turn of the card or a quiz's Confirm (app.dashboard.app.Dashboard.step). A focused checkbox
   counts as outside: the browser does nothing with Enter on it, so it confirms the quiz. The value counts up, so
   every press is a new value. The transport: dash_clientside.set_props, else the native setter and
   an input event. Dash serves this with the dashboard's assets. */
(function () {
  "use strict";

  const INPUT_ID = "key-enter";
  const OWN_KEYS = { INPUT: true, TEXTAREA: true, SELECT: true, BUTTON: true, A: true };
  let count = 0;

  function send() {
    count += 1;
    const value = String(count);
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

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" || event.ctrlKey || event.altKey || event.metaKey || event.repeat) {
      return;
    }
    const target = event.target;
    const tag = target && target.tagName ? target.tagName.toUpperCase() : "";
    const checkbox = tag === "INPUT" && target.type === "checkbox";
    if (OWN_KEYS[tag] && !checkbox) {
      return;
    }
    if (target && typeof target.isContentEditable === "boolean" && target.isContentEditable) {
      return;
    }
    event.preventDefault();
    send();
  }, true);
})();
