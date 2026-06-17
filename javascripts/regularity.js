/* ============================================================
   Regularity as Regularization — interactive demo (1D)
   Paty, d'Aspremont & Cuturi, AISTATS 2020.

   In 1D, an L-smooth, mu-strongly-convex Brenier potential f is
   convex with f'' in [mu, L]; its gradient (the OT map T = f')
   is monotone with slope in [mu, L]. Estimating T from noisy
   samples under this constraint is the convex QP

        min_T  sum_i ( T_i - y_i )^2
        s.t.   mu*(x_{i+1}-x_i) <= T_{i+1}-T_i <= L*(x_{i+1}-x_i),

   a generalized isotonic regression (plain isotonic = mu=0,
   L=inf). Here the *ground-truth* map is itself drawn from a
   (mu*, L*) potential, and a grid search recovers the best
   (mu, L) band. Everything is solved exactly in the browser.
   ============================================================ */
(function () {
  "use strict";

  var canvas = document.getElementById("reg-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");

  var COL = {
    data:  "#8a857b",
    fit:   "#a4452f",   // regularized map
    raw:   "#c6b9a8",   // unconstrained (overfit)
    truth: "#4e9b5b",   // ground-truth map
    iso:   "#7d6fa3",   // plain isotonic regression
    src:   "#3a6ea5",   // source measure P
    tgt:   "#c47d33",   // target measure Q
    axis:  "#d9d3c7"
  };

  var N = 46, M = 181;          // sample count, fine-grid count
  var mu = 0.15, L = 2.2;       // fit band  (dual slider)
  var muStar = 0.2, Lstar = 3.0;// true band (dual slider)
  var sigma = 0.14;

  var xs = [], zs = [], ys = [];        // samples + frozen N(0,1) noise
  var xf = [], gBase = [], TstarFine = [];

  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function cumtrapz(xa, ya) {           // cumulative integral by the trapezoid rule
    var F = new Float64Array(xa.length);
    for (var i = 1; i < xa.length; i++)
      F[i] = F[i - 1] + 0.5 * (ya[i] + ya[i - 1]) * (xa[i] - xa[i - 1]);
    return F;
  }

  // ---- ground truth drawn from a (mu*, L*) potential ----
  function resample() {
    xs = []; zs = [];
    for (var i = 0; i < N; i++) { xs.push(i / (N - 1)); zs.push(randn()); }
    // smooth random slope shape in [0,1] from a few sine components
    var freqs = [0.8, 1.7, 2.6, 3.9], amp = [], ph = [];
    for (var k = 0; k < freqs.length; k++) { amp.push(0.4 + Math.random()); ph.push(Math.random() * 2 * Math.PI); }
    xf = []; var raw = new Float64Array(M), lo = Infinity, hi = -Infinity, j;
    for (j = 0; j < M; j++) {
      var x = j / (M - 1); xf.push(x);
      var s = 0;
      for (k = 0; k < freqs.length; k++) s += amp[k] * Math.sin(2 * Math.PI * freqs[k] * x + ph[k]);
      raw[j] = s; if (s < lo) lo = s; if (s > hi) hi = s;
    }
    gBase = new Float64Array(M);
    for (j = 0; j < M; j++) gBase[j] = (raw[j] - lo) / (hi - lo + 1e-9);  // in [0,1]
    recomputeTrue();
    rebuildY();
  }
  function recomputeTrue() {
    var slope = new Float64Array(M);
    for (var j = 0; j < M; j++) slope[j] = muStar + (Lstar - muStar) * gBase[j];
    TstarFine = cumtrapz(xf, slope);
  }
  function Tstar(x) {
    var t = x * (M - 1), i = Math.floor(t);
    if (i >= M - 1) return TstarFine[M - 1];
    if (i < 0) return TstarFine[0];
    var fr = t - i;
    return TstarFine[i] * (1 - fr) + TstarFine[i + 1] * fr;
  }
  function rebuildY() {
    ys = [];
    for (var i = 0; i < N; i++) ys.push(Tstar(xs[i]) + sigma * zs[i]);
  }

  // ---- solver: box-constrained QP by coordinate descent ----
  function solve(muu, Lu) {
    var n = xs.length;
    var d = new Float64Array(n - 1), l = new Float64Array(n - 1), u = new Float64Array(n - 1);
    var T = new Float64Array(n), r = new Float64Array(n), k, i;
    T[0] = ys[0];
    for (k = 0; k < n - 1; k++) {
      var dx = xs[k + 1] - xs[k];
      l[k] = muu * dx; u[k] = Lu * dx;
      d[k] = Math.min(u[k], Math.max(l[k], ys[k + 1] - ys[k]));
      T[k + 1] = T[k] + d[k];
    }
    for (i = 0; i < n; i++) r[i] = T[i] - ys[i];
    for (var sweep = 0; sweep < 220; sweep++) {
      var mr = 0; for (i = 0; i < n; i++) mr += r[i]; mr /= n;
      for (i = 0; i < n; i++) { T[i] -= mr; r[i] -= mr; }
      for (k = 0; k < n - 1; k++) {
        var cnt = n - (k + 1), sum = 0;
        for (i = k + 1; i < n; i++) sum += r[i];
        var nd = Math.min(u[k], Math.max(l[k], d[k] - sum / cnt));
        var applied = nd - d[k];
        if (applied !== 0) { for (i = k + 1; i < n; i++) { T[i] += applied; r[i] += applied; } d[k] = nd; }
      }
    }
    return T;
  }
  function rmse(Tarr) {
    var s = 0; for (var i = 0; i < Tarr.length; i++) { var e = Tarr[i] - Tstar(xs[i]); s += e * e; }
    return Math.sqrt(s / Tarr.length);
  }
  function bestFit() {
    var muGrid = [0, 0.15, 0.3, 0.5, 0.75, 1.05, 1.4, 1.8, 2.3];
    var lGrid = [0.4, 0.7, 1.0, 1.4, 1.8, 2.2, 2.7, 3.2, 3.8];
    var best = Infinity, bm = mu, bl = L;
    for (var a = 0; a < muGrid.length; a++)
      for (var b = 0; b < lGrid.length; b++) {
        if (lGrid[b] < muGrid[a]) continue;
        var e = rmse(solve(muGrid[a], lGrid[b]));
        if (e < best) { best = e; bm = muGrid[a]; bl = lGrid[b]; }
      }
    fitDR.set(bm, bl);
  }

  // ---------------- canvas / plotting ----------------
  var dpr = Math.max(1, window.devicePixelRatio || 1), W = 0, H = 0;
  function fit() {
    var cssW = canvas.clientWidth || 900;
    var cssH = Math.round(Math.min(720, Math.max(480, cssW * 0.72)));
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    W = cssW; H = cssH;
  }
  function Panel(y0, h, ymin, ymax) {
    var padL = 8, padR = 10, padT = 26, padB = 8;
    var ix0 = padL, iy0 = y0 + padT, iw = W - padL - padR, ih = h - padT - padB;
    return {
      px: function (x) { return ix0 + x * iw; },
      py: function (y) { return iy0 + ih - (y - ymin) / (ymax - ymin) * ih; },
      frame: function (label) {
        ctx.strokeStyle = COL.axis; ctx.lineWidth = 1; ctx.strokeRect(ix0, iy0, iw, ih);
        ctx.fillStyle = "#6f6a62"; ctx.font = "600 13px Spectral, Georgia, serif";
        ctx.fillText(label, ix0, y0 + 16);
      }
    };
  }
  function poly(P, xa, ya, color, width, dash) {
    ctx.save(); if (dash) ctx.setLineDash(dash);
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    for (var i = 0; i < xa.length; i++) {
      var X = P.px(xa[i]), Y = P.py(ya[i]);
      if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    }
    ctx.stroke(); ctx.restore();
  }
  function range(arrs) {
    var lo = Infinity, hi = -Infinity;
    for (var a = 0; a < arrs.length; a++)
      for (var i = 0; i < arrs[a].length; i++) { var v = arrs[a][i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    var pad = 0.08 * (hi - lo || 1);
    return [lo - pad, hi + pad];
  }

  function kde(samples, v, bw) {
    var s = 0;
    for (var i = 0; i < samples.length; i++) { var z = (v - samples[i]) / bw; s += Math.exp(-0.5 * z * z); }
    return s / (samples.length * bw * Math.sqrt(2 * Math.PI));
  }
  function hexA(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }

  // top panel: the two measures, and the map carrying the source onto the target
  function drawMeasures(y0, h) {
    var padL = 8, padR = 10, ix0 = padL, iw = W - padL - padR, i;
    var vlo = 0, vhi = 1;
    for (i = 0; i < ys.length; i++) { if (ys[i] < vlo) vlo = ys[i]; if (ys[i] > vhi) vhi = ys[i]; }
    if (TstarFine[M - 1] > vhi) vhi = TstarFine[M - 1];
    var vp = 0.05 * (vhi - vlo); vlo -= vp; vhi += vp;
    function vx(v) { return ix0 + (v - vlo) / (vhi - vlo) * iw; }
    var midY = y0 + h * 0.56;
    var halfTop = midY - (y0 + 24), halfBot = (y0 + h - 6) - midY;

    // target density Q = pushforward of the uniform source by the true map
    var nuSamp = [], NP = 300;
    for (i = 0; i < NP; i++) nuSamp.push(Tstar(i / (NP - 1)));
    var bw = 0.03 * (vhi - vlo), G = 160, nuv = new Float64Array(G), maxd = 1;
    for (i = 0; i < G; i++) { var v = vlo + (vhi - vlo) * i / (G - 1); nuv[i] = kde(nuSamp, v, bw); if (nuv[i] > maxd) maxd = nuv[i]; }
    var sTop = halfTop * 0.9 / maxd, sBot = halfBot * 0.9 / maxd;

    ctx.fillStyle = "#6f6a62"; ctx.font = "600 13px Spectral, Georgia, serif";
    ctx.fillText("The two measures: the map T carries the source P onto the target Q", ix0, y0 + 16);

    ctx.strokeStyle = COL.axis; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ix0, midY); ctx.lineTo(ix0 + iw, midY); ctx.stroke();

    // source P: uniform on [0,1], filled above the axis
    ctx.fillStyle = hexA(COL.src, 0.18); ctx.strokeStyle = COL.src; ctx.lineWidth = 2;
    var muH = sTop;
    ctx.beginPath();
    ctx.moveTo(vx(0), midY); ctx.lineTo(vx(0), midY - muH); ctx.lineTo(vx(1), midY - muH); ctx.lineTo(vx(1), midY);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = COL.src;
    for (i = 0; i < xs.length; i++) ctx.fillRect(vx(xs[i]) - 0.5, midY - 5, 1, 5);

    // target Q: pushforward density, filled below the axis
    ctx.fillStyle = hexA(COL.tgt, 0.18); ctx.strokeStyle = COL.tgt; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(vx(vlo), midY);
    for (i = 0; i < G; i++) ctx.lineTo(vx(vlo + (vhi - vlo) * i / (G - 1)), midY + nuv[i] * sBot);
    ctx.lineTo(vx(vhi), midY); ctx.fill(); ctx.stroke();
    ctx.fillStyle = COL.tgt;
    for (i = 0; i < ys.length; i++) ctx.fillRect(vx(ys[i]) - 0.5, midY, 1, 5);

    ctx.font = "italic 13px Spectral, Georgia, serif";
    ctx.fillStyle = COL.src; ctx.fillText("P  source", ix0 + 4, midY - muH - 6);
    ctx.fillStyle = COL.tgt; ctx.fillText("Q  target", ix0 + 4, midY + 18);
  }

  function draw() {
    var Treg = solve(mu, L), Tiso = solve(0, 1e9);
    ctx.clearRect(0, 0, W, H);
    var gap = 18, ph1 = (H - gap) * 0.42, ph2 = (H - gap) * 0.58;

    drawMeasures(0, ph1);

    var mr = range([TstarFine, Treg, Tiso, ys]);
    var bot = Panel(ph1 + gap, ph2, mr[0], mr[1]);
    bot.frame("The transport map  T   (its slope stays between μ and L)");
    poly(bot, xf, TstarFine, COL.truth, 2.6);
    ctx.fillStyle = COL.data;
    for (var i = 0; i < xs.length; i++) { ctx.beginPath(); ctx.arc(bot.px(xs[i]), bot.py(ys[i]), 2.6, 0, 2 * Math.PI); ctx.fill(); }
    poly(bot, xs, ys, COL.raw, 1.5, [4, 4]);
    poly(bot, xs, Tiso, COL.iso, 1.8);
    poly(bot, xs, Treg, COL.fit, 2.6);

    set("reg-rmse", rmse(Treg).toFixed(3));
    set("reg-rmse-iso", rmse(Tiso).toFixed(3));
    set("reg-rmse-raw", rmse(ys).toFixed(3));
  }
  function set(id, v) { var el = document.getElementById(id); if (el) el.textContent = v; }
  function fmt(lo, hi) { return "(" + lo.toFixed(2) + ", " + hi.toFixed(2) + ")"; }

  // ---------------- loop ----------------
  var dirty = true;
  function frame() {
    if (dirty) { dirty = false; draw(); }
    requestAnimationFrame(frame);
  }
  function invalidate() { dirty = true; }

  // ---------------- dual-thumb slider ----------------
  function makeDualRange(container, min, max, lo, hi, step, color, onInput) {
    container.style.setProperty("--dr", color);
    container.innerHTML = '<div class="dr-track"><div class="dr-fill"></div></div>' +
      '<div class="dr-thumb dr-lo"></div><div class="dr-thumb dr-hi"></div>';
    var track = container.querySelector(".dr-track");
    var fill = container.querySelector(".dr-fill");
    var loEl = container.querySelector(".dr-lo");
    var hiEl = container.querySelector(".dr-hi");
    var st = { lo: lo, hi: hi }, drag = null;
    function pct(v) { return (v - min) / (max - min) * 100; }
    function render() {
      var pl = pct(st.lo), ph2 = pct(st.hi);
      loEl.style.left = pl + "%"; hiEl.style.left = ph2 + "%";
      fill.style.left = pl + "%"; fill.style.right = (100 - ph2) + "%";
    }
    function valAt(clientX) {
      var r = track.getBoundingClientRect();
      var t = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
      return Math.round((min + t * (max - min)) / step) * step;
    }
    function apply(clientX, which) {
      var v = valAt(clientX);
      if (which === "lo") st.lo = Math.min(v, st.hi);
      else st.hi = Math.max(v, st.lo);
      render(); onInput(st.lo, st.hi);
    }
    function down(e, which) { drag = which; e.preventDefault(); apply(cx(e), which); }
    function moveH(e) { if (drag) { apply(cx(e), drag); e.preventDefault(); } }
    function cx(e) { return e.touches ? e.touches[0].clientX : e.clientX; }
    loEl.addEventListener("mousedown", function (e) { down(e, "lo"); });
    hiEl.addEventListener("mousedown", function (e) { down(e, "hi"); });
    loEl.addEventListener("touchstart", function (e) { down(e, "lo"); }, { passive: false });
    hiEl.addEventListener("touchstart", function (e) { down(e, "hi"); }, { passive: false });
    track.addEventListener("mousedown", function (e) {
      if (e.target === loEl || e.target === hiEl) return;
      var v = valAt(cx(e));
      down(e, Math.abs(v - st.lo) <= Math.abs(v - st.hi) ? "lo" : "hi");
    });
    window.addEventListener("mousemove", moveH);
    window.addEventListener("touchmove", moveH, { passive: false });
    window.addEventListener("mouseup", function () { drag = null; });
    window.addEventListener("touchend", function () { drag = null; });
    render();
    return { set: function (l, h) { st.lo = l; st.hi = h; render(); onInput(l, h); }, get: function () { return st; } };
  }

  // ---------------- controls ----------------
  var fitDR, trueDR;
  function onFit(lo, hi) { mu = lo; L = hi; set("reg-fit-val", fmt(lo, hi)); invalidate(); }
  function onTrue(lo, hi) { muStar = lo; Lstar = hi; set("reg-true-val", fmt(lo, hi)); recomputeTrue(); rebuildY(); invalidate(); }

  var elSigma = document.getElementById("reg-sigma");
  var elResample = document.getElementById("reg-resample");
  var elBest = document.getElementById("reg-bestfit");
  if (elSigma) elSigma.addEventListener("input", function () { sigma = +elSigma.value; rebuildY(); invalidate(); });
  if (elResample) elResample.addEventListener("click", function () { resample(); invalidate(); });
  if (elBest) elBest.addEventListener("click", bestFit);
  window.addEventListener("resize", function () { fit(); invalidate(); });

  // ---------------- go ----------------
  fit();
  resample();
  fitDR = makeDualRange(document.getElementById("reg-fit-dr"), 0, 4, mu, L, 0.05, COL.fit, onFit);
  trueDR = makeDualRange(document.getElementById("reg-true-dr"), 0.1, 4, muStar, Lstar, 0.05, COL.truth, onTrue);
  requestAnimationFrame(frame);
})();
