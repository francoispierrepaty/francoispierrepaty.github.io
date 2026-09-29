/* ============================================================
   The easter egg: type the word "forecast" and the cursor turns into a
   crosshair. Then each click on the page plants a point. From the second
   point a line joins them; from the third, a dashed line continues the
   recent trend inside a band whose width comes from how scattered the
   points are. It starts over after eight points, switches itself off
   after a quiet moment or on Esc, and writes nothing on the page.
   ============================================================ */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var SKIP = "a, button, input, select, textarea, label, summary, canvas, iframe, img, video, mjx-container, .dr-thumb, .dualrange";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var WORD = "forecast", typed = "";
  var svg = null, pts = [], timer = null, offTimer = null, armed = false;

  function ensure() {
    if (svg) return;
    svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "fc-plot");
    svg.setAttribute("aria-hidden", "true");
    document.body.appendChild(svg);
  }

  // Two-sided 80% Student t for small samples, indexed by degrees of freedom.
  var T80 = [0, 3.08, 1.89, 1.64, 1.53, 1.48, 1.44, 1.41];

  /* A simple, honest forecast of the drawn series (points sorted by x).
     - trend: least squares slope, with recent points weighted more (decay 0.7)
     - noise: weighted residual spread around that trend, with a small-sample correction
     - path: starts at the last point and follows the trend
     - band: t * noise * sqrt(steps ahead), so it opens like a random walk
       and is wide for scattered points, thin for tidy ones
     - horizon: 70% of the drawn span, limited by the room left on the right */
  function forecast(s, room) {
    var n = s.length, w = [], W = 0, W2 = 0, i;
    for (i = 0; i < n; i++) { w[i] = Math.pow(0.7, n - 1 - i); W += w[i]; W2 += w[i] * w[i]; }
    var mx = 0, my = 0;
    for (i = 0; i < n; i++) { mx += w[i] * s[i].x / W; my += w[i] * s[i].y / W; }
    var sxx = 0, sxy = 0;
    for (i = 0; i < n; i++) { sxx += w[i] * (s[i].x - mx) * (s[i].x - mx); sxy += w[i] * (s[i].x - mx) * (s[i].y - my); }
    var span = s[n - 1].x - s[0].x;
    if (sxx < 1 || span < 30) return null;
    var slope = sxy / sxx, rss = 0;
    for (i = 0; i < n; i++) { var r = s[i].y - (my + slope * (s[i].x - mx)); rss += w[i] * r * r; }
    var nEff = (W * W) / W2, df = Math.max(1, nEff - 2);
    var sigma = Math.max(1.5, Math.sqrt(rss / W * nEff / df));
    var t = T80[Math.min(7, Math.max(1, Math.round(df)))];
    var L = Math.min(0.7 * span, room);
    if (L < 24) return null;
    var step = span / (n - 1);
    return {
      L: L,
      y: function (dx) { return s[n - 1].y + slope * dx; },
      half: function (dx) { return Math.min(150, t * sigma * Math.sqrt(dx / step)); }
    };
  }

  function draw() {
    var W = window.innerWidth, H = window.innerHeight;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var s = pts.slice().sort(function (a, b) { return a.x - b.x; });
    var out = "";
    if (s.length >= 2) {
      out += '<path class="hist" d="' + s.map(function (p, i) { return (i ? "L" : "M") + p.x.toFixed(1) + " " + p.y.toFixed(1); }).join(" ") + '"/>';
    }
    if (s.length >= 3) {
      var last = s[s.length - 1], f = forecast(s, W - last.x - 12);
      if (f) {
        var up = "", down = "", k, dx;
        for (k = 0; k <= 24; k++) {
          dx = f.L * k / 24;
          up   += (k ? " L" : "M") + (last.x + dx).toFixed(1) + " " + (f.y(dx) - f.half(dx)).toFixed(1);
          down  = " L" + (last.x + dx).toFixed(1) + " " + (f.y(dx) + f.half(dx)).toFixed(1) + down;
        }
        out += '<path class="cone" d="' + up + down + ' Z"/>' +
               '<path class="fc" d="M' + last.x + " " + last.y + " L" + (last.x + f.L).toFixed(1) + " " + f.y(f.L).toFixed(1) + '"/>';
      }
    }
    s.forEach(function (p) { out += '<circle class="dot" cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" r="5"/>'; });
    svg.innerHTML = out;
  }

  function clear() { pts = []; if (svg) svg.innerHTML = ""; }

  function arm(on) {
    armed = on;
    document.documentElement.classList.toggle("fc-on", on);
    clearTimeout(offTimer);
    if (on) offTimer = setTimeout(function () { arm(false); }, 25000);
    else { typed = ""; clear(); }
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { if (armed) arm(false); return; }
    var t = e.target;
    if (e.metaKey || e.ctrlKey || e.altKey || e.key.length !== 1) return;
    if (t && /^(input|textarea|select)$/i.test(t.tagName)) return;
    typed = (typed + e.key.toLowerCase()).slice(-WORD.length);
    if (typed === WORD) arm(!armed);
  });

  document.addEventListener("click", function (e) {
    if (!armed) return;
    if (e.target.closest && e.target.closest(SKIP)) return;
    var sel = window.getSelection && window.getSelection();
    if (sel && String(sel).length) return;
    ensure();
    if (pts.length >= 8) pts = [];
    pts.push({ x: e.clientX, y: e.clientY });
    svg.style.transition = "none";
    svg.style.opacity = 1;
    draw();
    clearTimeout(timer);
    clearTimeout(offTimer);
    offTimer = setTimeout(function () { arm(false); }, 25000);
    timer = setTimeout(function () {
      if (!reduce) svg.style.transition = "opacity 1.2s";
      svg.style.opacity = 0;
      setTimeout(clear, reduce ? 0 : 1300);
    }, 6000);
  });
})();
