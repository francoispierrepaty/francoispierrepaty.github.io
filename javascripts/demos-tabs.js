/* ============================================================
   Demos page — internal tab switcher.
   Buttons carry data-tab; panels carry data-panel. Only one
   panel is shown at a time. On switch we fire a resize event so
   the canvas demos (srw.js / regularity.js) re-fit to the now
   visible panel. The active tab is reflected in the URL hash so
   links like demos.html#reg1d open the right demo.
   ============================================================ */
(function () {
  "use strict";

  var tabs = Array.prototype.slice.call(document.querySelectorAll(".demo-tab-btn"));
  var panels = Array.prototype.slice.call(document.querySelectorAll(".demo-panel"));
  if (!tabs.length) return;

  function activate(id) {
    tabs.forEach(function (t) {
      var on = t.getAttribute("data-tab") === id;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    panels.forEach(function (p) {
      p.hidden = p.getAttribute("data-panel") !== id;
    });
    // the freshly shown canvas needs to measure its real width
    window.dispatchEvent(new Event("resize"));
  }

  tabs.forEach(function (t) {
    t.addEventListener("click", function () {
      var id = t.getAttribute("data-tab");
      activate(id);
      if (history.replaceState) history.replaceState(null, "", "#" + id);
    });
  });

  var hash = (location.hash || "").replace("#", "");
  var valid = tabs.some(function (t) { return t.getAttribute("data-tab") === hash; });
  activate(valid ? hash : tabs[0].getAttribute("data-tab"));
})();
