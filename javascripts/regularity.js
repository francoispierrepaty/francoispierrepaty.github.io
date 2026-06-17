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
  var sigma = 0.09;
  var showRaw = true;

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

    var mr = range(showRaw ? [TstarFine, Treg, Tiso, ys] : [TstarFine, Treg, Tiso]);
    var bot = Panel(ph1 + gap, ph2, mr[0], mr[1]);
    bot.frame("The transport map  T   (its slope stays between μ and L)");
    poly(bot, xf, TstarFine, COL.truth, 2.6);
    ctx.fillStyle = COL.data;
    for (var i = 0; i < xs.length; i++) { ctx.beginPath(); ctx.arc(bot.px(xs[i]), bot.py(ys[i]), 2.6, 0, 2 * Math.PI); ctx.fill(); }
    if (showRaw) poly(bot, xs, ys, COL.raw, 1.5, [4, 4]);
    poly(bot, xs, Tiso, COL.iso, 1.8, [6, 4]);
    poly(bot, xs, Treg, COL.fit, 2.6);

    set("reg-rmse", rmse(Treg).toFixed(3));
    set("reg-rmse-iso", rmse(Tiso).toFixed(3));
    set("reg-rmse-raw", rmse(ys).toFixed(3));
  }
  function set(id, v) { var el = document.getElementById(id); if (el) el.textContent = v; }
  function fmt(lo, hi) { return "(" + lo.toFixed(2) + ", " + hi.toFixed(2) + ")"; }

  // ---------------- loop ----------------
  var dirty = true, reg2d = null;
  function frame() {
    if (dirty) { dirty = false; draw(); }
    if (reg2d && reg2d.dirty) { reg2d.dirty = false; draw2D(); }
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
  var elToggle = document.getElementById("reg-toggle");
  if (elSigma) elSigma.addEventListener("input", function () { sigma = +elSigma.value; rebuildY(); invalidate(); });
  if (elResample) elResample.addEventListener("click", function () { resample(); invalidate(); });
  if (elBest) elBest.addEventListener("click", bestFit);
  if (elToggle) elToggle.addEventListener("click", function () {
    showRaw = !showRaw;
    elToggle.textContent = showRaw ? "Hide unconstrained fit" : "Show unconstrained fit";
    elToggle.classList.toggle("ghost", !showRaw);
    invalidate();
  });
  window.addEventListener("resize", function () { fit(); fit2D(); invalidate(); invalidate2D(); });

  // ---------------- two-dimensional alternating demo ----------------
  var REG2D_MIN_GAP = 0.16;
  var REG2D_MAX_COORD = 4;
  function make2DPoint(x, y) { return { x: x, y: y }; }
  function dot2(a, b) { return a.x * b.x + a.y * b.y; }
  function sub2(a, b) { return make2DPoint(a.x - b.x, a.y - b.y); }
  function addScaled2(a, b, s) { a.x += s * b.x; a.y += s * b.y; }
  function norm2(a) { return a.x * a.x + a.y * a.y; }
  function finite2DPoint(p) { return p && isFinite(p.x) && isFinite(p.y); }
  function clamp2DPoint(p) {
    p.x = Math.max(-REG2D_MAX_COORD, Math.min(REG2D_MAX_COORD, p.x));
    p.y = Math.max(-REG2D_MAX_COORD, Math.min(REG2D_MAX_COORD, p.y));
  }
  function regularityParams2D() {
    var ell = reg2d.mu;
    var L2 = Math.max(reg2d.L, ell + REG2D_MIN_GAP);
    var oneMinus = Math.max(1 - ell / L2, 1e-4);
    return { ell: ell, L2: L2, oneMinus: oneMinus, denom: 2 * oneMinus };
  }
  function trueWarp2D() {
    var r = reg2d;
    var gap = Math.max(r.Lstar - r.muStar, REG2D_MIN_GAP);
    var margin = 0.22 * gap;
    var kx = 5.0, ky = 4.0;
    return {
      ax: r.Lstar - margin,
      ay: r.muStar + margin,
      alpha: margin / (kx * kx),
      beta: 0.75 * margin / (ky * ky),
      kx: kx,
      ky: ky,
      phase: r.warpPhase
    };
  }
  function rotate2D(p, theta) {
    var c = Math.cos(theta), s = Math.sin(theta);
    return make2DPoint(c * p.x - s * p.y, s * p.x + c * p.y);
  }
  function trueMap2D(x) {
    var r = reg2d;
    var q = rotate2D(x, -r.theta);
    var w = trueWarp2D();
    var gradQ = make2DPoint(
      w.ax * q.x + w.alpha * w.kx * Math.sin(w.kx * q.x + w.phase),
      w.ay * q.y + w.beta * w.ky * Math.sin(w.ky * q.y - 0.7 * w.phase)
    );
    var out = rotate2D(gradQ, r.theta);
    return make2DPoint(r.trueShift.x + out.x, r.trueShift.y + out.y);
  }
  function truePotential2D(x) {
    var r = reg2d;
    var q = rotate2D(x, -r.theta);
    var w = trueWarp2D();
    return dot2(r.trueShift, x) +
      0.5 * (w.ax * q.x * q.x + w.ay * q.y * q.y) -
      w.alpha * Math.cos(w.kx * q.x + w.phase) -
      w.beta * Math.cos(w.ky * q.y - 0.7 * w.phase);
  }
  function init2D() {
    var c = document.getElementById("reg2d-canvas");
    if (!c) return;
    reg2d = {
      canvas: c,
      ctx: c.getContext("2d"),
      n: 50,
      muStar: 1.25,
      Lstar: 2.35,
      mu: 1.25,
      L: 2.35,
      theta: 0,
      warpPhase: 0,
      sourceCenter: make2DPoint(0, 0),
      trueShift: make2DPoint(0.16, 0.06),
      iter: 0,
      maxIter: 100,
      epsilon: 0.00015,
      delta: Infinity,
      status: "ready",
      running: false,
      dirty: true,
      W: 0,
      H: 0,
      x: [],
      targetSource: [],
      y: [],
      z: [],
      u: [],
      w: [],
      P: [],
      cost: 0,
      violation: 0,
      timer: null
    };
    fit2D();
    resample2D();
    var resampleEl = document.getElementById("reg2d-resample");
    var runEl = document.getElementById("reg2d-run");
    var resetEl = document.getElementById("reg2d-reset");
    var dr = document.getElementById("reg2d-fit-dr");
    var trueDr = document.getElementById("reg2d-true-dr");
    var fitDR2D = null;
    if (dr) fitDR2D = makeDualRange(dr, 0.4, 3.2, reg2d.mu, reg2d.L, 0.05, COL.fit, function (lo, hi) {
      stop2D();
      reg2d.mu = lo; reg2d.L = Math.max(hi, lo + 0.05);
      set("reg2d-fit-val", fmt(reg2d.mu, reg2d.L));
      resetMap2D();
    });
    if (trueDr) makeDualRange(trueDr, 0.4, 3.2, reg2d.muStar, reg2d.Lstar, 0.05, COL.truth, function (lo, hi) {
      stop2D();
      reg2d.muStar = lo;
      reg2d.Lstar = Math.max(hi, lo + 0.05);
      set("reg2d-true-val", fmt(reg2d.muStar, reg2d.Lstar));
      if (fitDR2D) fitDR2D.set(reg2d.muStar, reg2d.Lstar);
      else { reg2d.mu = reg2d.muStar; reg2d.L = reg2d.Lstar; set("reg2d-fit-val", fmt(reg2d.mu, reg2d.L)); }
      rebuildTrue2D();
      resetMap2D();
    });
    if (resampleEl) resampleEl.addEventListener("click", function () { stop2D(); resample2D(); });
    if (resetEl) resetEl.addEventListener("click", function () { stop2D(); resetMap2D(); });
    if (runEl) runEl.addEventListener("click", function () {
      if (reg2d.running) stop2D();
      else {
        if (reg2d.status === "stable" || reg2d.status === "max steps") resetMap2D();
        reg2d.running = true;
        reg2d.status = "running";
        runEl.textContent = "Pause";
        invalidate2D();
        reg2d.timer = window.setInterval(function () { alternation2D(true); }, 280);
      }
    });
  }
  function stop2D() {
    if (!reg2d) return;
    reg2d.running = false;
    if (reg2d.timer) window.clearInterval(reg2d.timer);
    reg2d.timer = null;
    var runEl = document.getElementById("reg2d-run");
    if (runEl) runEl.textContent = "Run";
  }
  function fit2D() {
    if (!reg2d) return;
    var c = reg2d.canvas, cx = reg2d.ctx;
    var cssW = c.clientWidth || 900;
    var cssH = cssW < 640
      ? Math.round(Math.min(920, Math.max(660, cssW * 1.55)))
      : Math.round(Math.min(560, Math.max(420, cssW * 0.54)));
    c.width = Math.round(cssW * dpr);
    c.height = Math.round(cssH * dpr);
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    reg2d.W = cssW; reg2d.H = cssH;
  }
  function invalidate2D() { if (reg2d) reg2d.dirty = true; }
  function resample2D() {
    var r = reg2d, i;
    r.x = []; r.targetSource = []; r.y = []; r.z = []; r.u = []; r.w = []; r.P = [];
    r.theta = -0.6 + Math.random() * 1.2;
    r.warpPhase = Math.random() * 2 * Math.PI;
    set("reg2d-true-val", fmt(r.muStar, r.Lstar));
    var golden = Math.PI * (3 - Math.sqrt(5));
    var phase = Math.random() * 2 * Math.PI;
    var targetPhase = phase + 0.57 * golden;
    for (i = 0; i < r.n; i++) {
      var rad = Math.sqrt((i + 0.5) / r.n) * 0.46;
      var ang = phase + i * golden;
      var x = make2DPoint(r.sourceCenter.x + rad * Math.cos(ang), r.sourceCenter.y + rad * Math.sin(ang));
      r.x.push(x);
      var targetRad = Math.sqrt((i + 0.5) / r.n) * 0.46;
      var targetAng = targetPhase + i * golden;
      r.targetSource.push(make2DPoint(r.sourceCenter.x + targetRad * Math.cos(targetAng), r.sourceCenter.y + targetRad * Math.sin(targetAng)));
    }
    rebuildTrue2D();
    resetMap2D();
  }
  function rebuildTrue2D() {
    var r = reg2d;
    r.y = [];
    for (var i = 0; i < r.targetSource.length; i++) r.y.push(trueMap2D(r.targetSource[i]));
  }
  function resetMap2D() {
    var r = reg2d;
    r.iter = 0; r.z = []; r.u = [];
    r.delta = Infinity;
    r.status = "ready";
    for (var i = 0; i < r.n; i++) {
      r.z.push(make2DPoint(r.x[i].x, r.x[i].y));
      r.u.push(0.5 * norm2(r.x[i]));
    }
    computeOT2D();
    for (i = 0; i < r.n; i++) {
      r.z[i] = make2DPoint(r.w[i].x, r.w[i].y);
      r.u[i] = 0.5 * dot2(r.x[i], r.z[i]);
    }
    computeOT2D();
    maxViolation2D();
    invalidate2D();
  }
  function computeOT2D() {
    var r = reg2d, n = r.n, i, j, it;
    var eps = 0.035, K = [], uS = new Float64Array(n), vS = new Float64Array(n);
    for (i = 0; i < n; i++) {
      K[i] = [];
      uS[i] = 1;
      vS[i] = 1;
      for (j = 0; j < n; j++) {
        var d = sub2(r.z[i], r.y[j]);
        var kval = Math.exp(-Math.min(norm2(d) / eps, 700)) + 1e-12;
        K[i][j] = isFinite(kval) ? kval : 1e-12;
      }
    }
    for (it = 0; it < 80; it++) {
      for (i = 0; i < n; i++) {
        var sv = 0; for (j = 0; j < n; j++) sv += K[i][j] * vS[j];
        uS[i] = (1 / n) / Math.max(sv, 1e-12);
      }
      for (j = 0; j < n; j++) {
        var su = 0; for (i = 0; i < n; i++) su += K[i][j] * uS[i];
        vS[j] = (1 / n) / Math.max(su, 1e-12);
      }
    }
    r.P = []; r.w = []; r.cost = 0;
    for (i = 0; i < n; i++) {
      r.P[i] = [];
      var wx = 0, wy = 0, row = 0;
      for (j = 0; j < n; j++) {
        var p = uS[i] * K[i][j] * vS[j];
        if (!isFinite(p)) p = 0;
        r.P[i][j] = p; row += p;
        wx += p * r.y[j].x; wy += p * r.y[j].y;
        var cd = sub2(r.z[i], r.y[j]);
        r.cost += p * norm2(cd);
      }
      var wp = make2DPoint(wx / Math.max(row, 1e-12), wy / Math.max(row, 1e-12));
      if (!finite2DPoint(wp)) wp = make2DPoint(r.z[i].x, r.z[i].y);
      r.w.push(wp);
    }
  }
  function qcqpConstraint2D(i, j) {
    var r = reg2d, prm = regularityParams2D(), ell = prm.ell, L2 = prm.L2;
    var dx = sub2(r.x[i], r.x[j]);
    var dz = sub2(r.z[i], r.z[j]);
    var a = 1 / prm.denom;
    var b = norm2(dz) / L2 + ell * norm2(dx) - 2 * ell * dot2(dz, dx) / L2;
    var c = r.u[j] + dot2(r.z[j], dx) + a * b - r.u[i];
    return isFinite(c) ? c : 0;
  }
  function maxViolation2D() {
    var r = reg2d, m = 0;
    for (var i = 0; i < r.n; i++)
      for (var j = 0; j < r.n; j++)
        if (i !== j) m = Math.max(m, qcqpConstraint2D(i, j));
    r.violation = m;
  }
  function qcqp2D(target, sweeps) {
    var r = reg2d, n = r.n, prm = regularityParams2D(), ell = prm.ell, L2 = prm.L2;
    var rho = 7.5, step = 0.018 / (1 + 0.5 * L2);
    for (var sweep = 0; sweep < sweeps; sweep++) {
      var gz = [], gu = new Float64Array(n), i, j;
      for (i = 0; i < n; i++) {
        gz.push(make2DPoint((r.z[i].x - target[i].x) / n, (r.z[i].y - target[i].y) / n));
        gu[i] = 0.0004 * r.u[i];
      }
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          if (i === j) continue;
          var c = qcqpConstraint2D(i, j);
          if (c <= 0) continue;
          var dx = sub2(r.x[i], r.x[j]);
          var dz = sub2(r.z[i], r.z[j]);
          var a = 1 / prm.denom;
          var gi = make2DPoint(a * (2 * dz.x / L2 - 2 * ell * dx.x / L2),
                               a * (2 * dz.y / L2 - 2 * ell * dx.y / L2));
          var gj = make2DPoint(dx.x - gi.x, dx.y - gi.y);
          addScaled2(gz[i], gi, rho * c);
          addScaled2(gz[j], gj, rho * c);
          gu[i] -= rho * c;
          gu[j] += rho * c;
        }
      }
      for (i = 0; i < n; i++) {
        var gzn = Math.hypot(gz[i].x, gz[i].y);
        if (!isFinite(gzn)) return false;
        if (gzn > 16) {
          gz[i].x *= 16 / gzn;
          gz[i].y *= 16 / gzn;
        }
        if (!isFinite(gu[i])) return false;
        gu[i] = Math.max(-32, Math.min(32, gu[i]));
        r.z[i].x -= step * gz[i].x;
        r.z[i].y -= step * gz[i].y;
        r.u[i] -= step * gu[i];
        if (!finite2DPoint(r.z[i]) || !isFinite(r.u[i])) return false;
        clamp2DPoint(r.z[i]);
      }
      if (sweep === 80) { rho *= 1.8; step *= 0.6; }
    }
    maxViolation2D();
    if (!isFinite(r.violation)) r.violation = 0;
    return true;
  }
  function alternation2D(fromRun) {
    var before = [];
    for (var i = 0; i < reg2d.z.length; i++) before.push(make2DPoint(reg2d.z[i].x, reg2d.z[i].y));
    computeOT2D();
    if (!qcqp2D(reg2d.w, 160)) {
      resetMap2D();
      reg2d.status = "numerical reset";
      if (fromRun) stop2D();
      invalidate2D();
      return;
    }
    computeOT2D();
    reg2d.iter += 1;
    var s = 0;
    for (var j = 0; j < reg2d.z.length; j++) {
      var d = sub2(reg2d.z[j], before[j]);
      s += norm2(d);
    }
    reg2d.delta = Math.sqrt(s / Math.max(1, reg2d.z.length));
    if (!isFinite(reg2d.delta) || !isFinite(reg2d.cost)) {
      resetMap2D();
      reg2d.status = "numerical reset";
      if (fromRun) stop2D();
      invalidate2D();
      return;
    }
    if (reg2d.delta <= reg2d.epsilon) {
      reg2d.status = "stable";
      if (fromRun) stop2D();
    } else if (reg2d.iter >= reg2d.maxIter) {
      reg2d.status = "max steps";
      if (fromRun) stop2D();
    } else {
      reg2d.status = fromRun ? "running" : "stepped";
    }
    invalidate2D();
  }
  function bounds2D() {
    var r = reg2d, pts = r.x.concat(r.y, r.z), loX = Infinity, hiX = -Infinity, loY = Infinity, hiY = -Infinity;
    for (var i = 0; i < pts.length; i++) {
      loX = Math.min(loX, pts[i].x); hiX = Math.max(hiX, pts[i].x);
      loY = Math.min(loY, pts[i].y); hiY = Math.max(hiY, pts[i].y);
    }
    var padX = 0.12 * (hiX - loX || 1), padY = 0.12 * (hiY - loY || 1);
    return [loX - padX, hiX + padX, loY - padY, hiY + padY];
  }
  function drawDot2D(ctx2, P, p, color, r, stroke) {
    ctx2.beginPath(); ctx2.arc(P.x(p.x), P.y(p.y), r, 0, 2 * Math.PI);
    ctx2.fillStyle = color; ctx2.fill();
    if (stroke) { ctx2.strokeStyle = stroke; ctx2.lineWidth = 1.2; ctx2.stroke(); }
  }
  function estimatePotentialAt2D(x) {
    var r = reg2d, prm = regularityParams2D(), ell = prm.ell, L2 = prm.L2;
    var denom = prm.denom;
    var g = make2DPoint(0, 0), i;
    for (i = 0; i < r.n; i++) { g.x += r.z[i].x; g.y += r.z[i].y; }
    g.x /= r.n; g.y /= r.n;
    function phiAndGrad(idx) {
      var dx = sub2(x, r.x[idx]);
      var h = sub2(g, r.z[idx]);
      var phi = r.u[idx] + dot2(r.z[idx], dx) +
        (norm2(h) / L2 + ell * norm2(dx) - 2 * ell * dot2(h, dx) / L2) / denom;
      return {
        value: phi,
        grad: make2DPoint((h.x - ell * dx.x) / (prm.oneMinus * L2),
                          (h.y - ell * dx.y) / (prm.oneMinus * L2))
      };
    }
    var lip = 1 / (prm.oneMinus * L2);
    var step = 0.42 / Math.max(lip, 1e-6);
    for (var it = 0; it < 55; it++) {
      var best = phiAndGrad(0);
      for (i = 1; i < r.n; i++) {
        var pg = phiAndGrad(i);
        if (pg.value > best.value) best = pg;
      }
      g.x -= step * best.grad.x;
      g.y -= step * best.grad.y;
      step *= 0.94;
    }
    var v = -Infinity;
    for (i = 0; i < r.n; i++) {
      var p = phiAndGrad(i).value;
      if (p > v) v = p;
    }
    return v;
  }
  function drawContour2D(ctx2, P, fn, color, alpha, dash) {
    var grid = 44, vals = [], lo = Infinity, hi = -Infinity, i, j;
    var bx0 = -0.58, bx1 = 0.58, by0 = -0.58, by1 = 0.58;
    for (i = 0; i <= grid; i++) {
      vals[i] = [];
      for (j = 0; j <= grid; j++) {
        var gx = bx0 + (bx1 - bx0) * i / grid;
        var gy = by0 + (by1 - by0) * j / grid;
        var v = fn(make2DPoint(gx, gy));
        if (!isFinite(v)) v = 0;
        vals[i][j] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    var levels = [];
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return;
    for (i = 1; i <= 9; i++) levels.push(lo + (hi - lo) * i / 10);
    ctx2.save();
    if (dash) ctx2.setLineDash(dash);
    ctx2.strokeStyle = hexA(color, alpha);
    ctx2.lineWidth = 1;
    function interp(p1, p2, v1, v2, level) {
      var t = (level - v1) / (v2 - v1 || 1e-9);
      return make2DPoint(p1.x + (p2.x - p1.x) * t, p1.y + (p2.y - p1.y) * t);
    }
    for (var li = 0; li < levels.length; li++) {
      var level = levels[li];
      for (i = 0; i < grid; i++) {
        for (j = 0; j < grid; j++) {
          var p0 = make2DPoint(bx0 + (bx1 - bx0) * i / grid, by0 + (by1 - by0) * j / grid);
          var p1 = make2DPoint(bx0 + (bx1 - bx0) * (i + 1) / grid, p0.y);
          var p2 = make2DPoint(p1.x, by0 + (by1 - by0) * (j + 1) / grid);
          var p3 = make2DPoint(p0.x, p2.y);
          var v0 = vals[i][j], v1 = vals[i + 1][j], v2 = vals[i + 1][j + 1], v3 = vals[i][j + 1];
          var hits = [];
          if ((v0 <= level && level < v1) || (v1 <= level && level < v0)) hits.push(interp(p0, p1, v0, v1, level));
          if ((v1 <= level && level < v2) || (v2 <= level && level < v1)) hits.push(interp(p1, p2, v1, v2, level));
          if ((v2 <= level && level < v3) || (v3 <= level && level < v2)) hits.push(interp(p2, p3, v2, v3, level));
          if ((v3 <= level && level < v0) || (v0 <= level && level < v3)) hits.push(interp(p3, p0, v3, v0, level));
          if (hits.length >= 2) {
            ctx2.beginPath();
            ctx2.moveTo(P.x(hits[0].x), P.y(hits[0].y));
            ctx2.lineTo(P.x(hits[1].x), P.y(hits[1].y));
            ctx2.stroke();
            if (hits.length === 4) {
              ctx2.beginPath();
              ctx2.moveTo(P.x(hits[2].x), P.y(hits[2].y));
              ctx2.lineTo(P.x(hits[3].x), P.y(hits[3].y));
              ctx2.stroke();
            }
          }
        }
      }
    }
    ctx2.restore();
  }
  function drawTrueLevelSets2D(ctx2, P) {
    drawContour2D(ctx2, P, truePotential2D, "#6f6a62", 0.18);
  }
  function drawEstimatedLevelSets2D(ctx2, P) {
    drawContour2D(ctx2, P, estimatePotentialAt2D, COL.fit, 0.32, [5, 4]);
  }
  function draw2D() {
    var r = reg2d, ctx2 = r.ctx, n = r.n, i, j;
    ctx2.clearRect(0, 0, r.W, r.H);
    var outerPad = r.W < 640 ? 18 : 14, gap = r.W < 640 ? 20 : 14, titleH = 20, stacked = r.W < 640;
    var pointBounds = bounds2D(), levelBounds = [-0.62, 0.62, -0.62, 0.62];
    var pointBox, levelBox;
    if (stacked) {
      var panelH = (r.H - 2 * outerPad - gap) / 2;
      pointBox = { x: outerPad, y: outerPad, w: r.W - 2 * outerPad, h: panelH };
      levelBox = { x: outerPad, y: outerPad + panelH + gap, w: r.W - 2 * outerPad, h: panelH };
    } else {
      var panelW = (r.W - 2 * outerPad - gap) / 2;
      pointBox = { x: outerPad, y: outerPad, w: panelW, h: r.H - 2 * outerPad };
      levelBox = { x: outerPad + panelW + gap, y: outerPad, w: panelW, h: r.H - 2 * outerPad };
    }
    function projection(bounds, box) {
      var pad = 4, iw = box.w - 2 * pad, ih = box.h - titleH - pad;
      var sx = iw / (bounds[1] - bounds[0]), sy = ih / (bounds[3] - bounds[2]), sc = Math.min(sx, sy);
      var ox = box.x + pad + 0.5 * (iw - sc * (bounds[1] - bounds[0]));
      var oy = box.y + titleH + 0.5 * (ih - sc * (bounds[3] - bounds[2]));
      return {
        x: function (v) { return ox + (v - bounds[0]) * sc; },
        y: function (v) { return oy + sc * (bounds[3] - v); }
      };
    }
    var Ppoints = projection(pointBounds, pointBox);
    var Plevels = projection(levelBounds, levelBox);
    ctx2.fillStyle = "#6f6a62"; ctx2.font = "600 13px Spectral, Georgia, serif";
    function panelTitle(text, box) {
      ctx2.fillText(text, box.x + 0.5 * (box.w - ctx2.measureText(text).width), box.y + 14);
    }
    panelTitle("Unpaired samples and current mapped points", pointBox);
    if (r.P.length) {
      for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
          var p = r.P[i][j];
          if (p < 0.0012) continue;
          ctx2.strokeStyle = "rgba(164,69,47," + Math.min(0.28, p * n * 0.24).toFixed(3) + ")";
          ctx2.lineWidth = 1;
          ctx2.beginPath(); ctx2.moveTo(Ppoints.x(r.z[i].x), Ppoints.y(r.z[i].y)); ctx2.lineTo(Ppoints.x(r.y[j].x), Ppoints.y(r.y[j].y)); ctx2.stroke();
        }
      }
    }
    for (i = 0; i < n; i++) drawDot2D(ctx2, Ppoints, r.y[i], COL.tgt, 5.2, "#fff");
    for (i = 0; i < n; i++) drawDot2D(ctx2, Ppoints, r.x[i], COL.src, 4.8, "#fff");
    for (i = 0; i < n; i++) drawDot2D(ctx2, Ppoints, r.z[i], COL.fit, 5.4, "#fff");
    ctx2.fillStyle = "#6f6a62"; ctx2.font = "600 13px Spectral, Georgia, serif";
    panelTitle("Potential level sets on the source domain", levelBox);
    drawTrueLevelSets2D(ctx2, Plevels);
    drawEstimatedLevelSets2D(ctx2, Plevels);
    set("reg2d-iter", String(r.iter));
    set("reg2d-delta", isFinite(r.delta) ? r.delta.toFixed(5) : "…");
    set("reg2d-cost", isFinite(r.cost) ? r.cost.toFixed(4) : "…");
    set("reg2d-viol", isFinite(r.violation) ? Math.max(0, r.violation).toFixed(4) : "…");
    set("reg2d-status", r.status);
  }

  // ---------------- go ----------------
  fit();
  resample();
  fitDR = makeDualRange(document.getElementById("reg-fit-dr"), 0, 4, mu, L, 0.05, COL.fit, onFit);
  trueDR = makeDualRange(document.getElementById("reg-true-dr"), 0.1, 4, muStar, Lstar, 0.05, COL.truth, onTrue);
  init2D();
  requestAnimationFrame(frame);
})();
