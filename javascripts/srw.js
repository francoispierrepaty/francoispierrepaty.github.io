/* ============================================================
   Subspace Robust Wasserstein — interactive demo
   Faithful port of Paty & Cuturi, "Subspace Robust Wasserstein
   Distances", ICML 2019, for k = 1 in 2D.

   The SRW distance maximizes, over k-dimensional projections
   P in  Omega = { P symmetric : 0 <= P <= I, tr P = k },
   the transport cost of the projected measures:

       SRW^2 = max_{P in Omega} min_{pi in Pi(a,b)}
                 sum_ij pi_ij (x_i - y_j)^T P (x_i - y_j).

   The algorithm alternates (projected supergradient ascent):
     1. given P, solve OT  ->  plan pi          (here: Sinkhorn)
     2. form V_pi = sum_ij pi_ij d_ij d_ij^T
     3. P <- projection onto top-k eigenvectors of V_pi
   For k = 1 the optimal subspace is the leading eigenvector
   of V_pi: the direction along which the two clouds are
   hardest to align.
   ============================================================ */
(function () {
  "use strict";

  var canvas = document.getElementById("srw-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");

  // colours (kept in sync with the page legend)
  var COL = { x: "#2a6f7a", y: "#a4452f", axis: "#caa84a", plan: "#9a958c", pca: "#6a6258" };

  var N = 24;                 // points per cloud
  var SINKHORN_ITERS = 100;
  var OUTER_ITERS = 15;
  var eps = 0.003;            // entropic regularization (slider)
  var spread = 0.18;          // spread of cloud Y (slider)
  var showPlan = false;

  // model state: base standard-normal samples + a transform per cloud
  var X = [], Y = [];         // current world coords, each [x, y] in [0,1]^2
  var zX = [], zY = [];       // frozen N(0,I) samples
  var cX = [0.47, 0.54], cY = [0.55, 0.49];
  var angX = 0, angY = 0;     // cloud shape angles

  // -------- small linear algebra ----------------------------
  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // leading eigenvector of symmetric [[a,b],[b,c]]
  function topEig(a, b, c) {
    var tr = a + c, det = a * c - b * b;
    var disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
    var l1 = tr / 2 + disc;
    var vx, vy;
    if (Math.abs(b) > 1e-12) { vx = l1 - c; vy = b; }
    else if (a >= c) { vx = 1; vy = 0; }
    else { vx = 0; vy = 1; }
    var n = Math.hypot(vx, vy) || 1;
    return { u: [vx / n, vy / n], lambda: l1 };
  }

  // -------- cloud generation --------------------------------
  function resample() {
    zX = []; zY = [];
    for (var i = 0; i < N; i++) {
      zX.push([randn(), randn()]);
      zY.push([randn(), randn()]);
    }
    angX = -0.35 + Math.random() * 0.7;
    angY = angX + Math.PI / 2 + (Math.random() - 0.5) * 0.45;
    cX = [0.46 + Math.random() * 0.06, 0.51 + (Math.random() - 0.5) * 0.08];
    cY = [cX[0] + 0.06 + (Math.random() - 0.5) * 0.05, cX[1] - 0.04 + (Math.random() - 0.5) * 0.06];
    rebuild();
  }

  function rebuild() {
    var cax = Math.cos(angX), sax = Math.sin(angX);
    var cay = Math.cos(angY), say = Math.sin(angY);
    var xMaj = 0.055, xMin = 0.04, xGap = 0.11; // X: two compact lobes
    var yMaj = spread, yMin = spread * 0.20;    // Y: curved, elongated cloud
    X = []; Y = [];
    for (var i = 0; i < N; i++) {
      var side = i % 2 === 0 ? -1 : 1;
      var xx = side * xGap + xMaj * zX[i][0], xy = xMin * zX[i][1];
      X.push([cX[0] + cax * xx - sax * xy, cX[1] + sax * xx + cax * xy]);

      var yx = yMaj * zY[i][0];
      var yy = yMin * zY[i][1] + 0.9 * yx * yx - 0.025; // bend the support
      Y.push([cY[0] + cay * yx - say * yy, cY[1] + say * yx + cay * yy]);
    }
  }
  function clampC(t) { return t < 0.12 ? 0.12 : t > 0.88 ? 0.88 : t; }  // keep cloud centres in view

  // -------- optimal transport (Sinkhorn) --------------------
  // cost C_ij = d_ij^T P d_ij, with P = [[p00,p01],[p01,p11]]
  function sinkhorn(p00, p01, p11) {
    var n = X.length, m = Y.length;
    var K = new Array(n);
    var dx = new Array(n * m), dy = new Array(n * m);
    for (var i = 0; i < n; i++) {
      K[i] = new Float64Array(m);
      for (var j = 0; j < m; j++) {
        var ddx = X[i][0] - Y[j][0], ddy = X[i][1] - Y[j][1];
        dx[i * m + j] = ddx; dy[i * m + j] = ddy;
        var cost = p00 * ddx * ddx + 2 * p01 * ddx * ddy + p11 * ddy * ddy;
        K[i][j] = Math.exp(-cost / eps);
      }
    }
    var u = new Float64Array(n).fill(1);
    var v = new Float64Array(m).fill(1);
    var a = 1 / n, b = 1 / m;
    for (var it = 0; it < SINKHORN_ITERS; it++) {
      for (var i2 = 0; i2 < n; i2++) {
        var s = 0; for (var j2 = 0; j2 < m; j2++) s += K[i2][j2] * v[j2];
        u[i2] = a / (s + 1e-300);
      }
      for (var j3 = 0; j3 < m; j3++) {
        var s2 = 0; for (var i3 = 0; i3 < n; i3++) s2 += K[i3][j3] * u[i3];
        v[j3] = b / (s2 + 1e-300);
      }
    }
    return { K: K, u: u, v: v, dx: dx, dy: dy, n: n, m: m };
  }

  // displacement matrix V_pi and total cost, given a Sinkhorn result
  function moments(S, p00, p01, p11) {
    var V00 = 0, V01 = 0, V11 = 0, cost = 0;
    for (var i = 0; i < S.n; i++) {
      for (var j = 0; j < S.m; j++) {
        var pi = S.u[i] * S.K[i][j] * S.v[j];
        var ddx = S.dx[i * S.m + j], ddy = S.dy[i * S.m + j];
        V00 += pi * ddx * ddx;
        V01 += pi * ddx * ddy;
        V11 += pi * ddy * ddy;
        cost += pi * (p00 * ddx * ddx + 2 * p01 * ddx * ddy + p11 * ddy * ddy);
      }
    }
    return { V00: V00, V01: V01, V11: V11, cost: cost };
  }

  // full SRW (k=1) iteration; also returns plain W2 and the plan
  function solveSRW() {
    // P initialised at the projection onto principal displacement axis
    var p00 = 0.5, p01 = 0, p11 = 0.5, dir = [1, 0];
    var S, M;
    for (var t = 0; t < OUTER_ITERS; t++) {
      S = sinkhorn(p00, p01, p11);
      M = moments(S, p00, p01, p11);
      var e = topEig(M.V00, M.V01, M.V11);
      dir = e.u;
      p00 = dir[0] * dir[0];
      p01 = dir[0] * dir[1];
      p11 = dir[1] * dir[1];
    }
    var srw = Math.sqrt(Math.max(0, moments(S, p00, p01, p11).cost));

    // plain W2: OT under the identity (full-space squared distance)
    var Sfull = sinkhorn(1, 0, 1);
    var w2 = Math.sqrt(Math.max(0, moments(Sfull, 1, 0, 1).cost));

    return { dir: dir, srw: srw, w2: w2, S: S };
  }

  // -------- canvas geometry ---------------------------------
  var dpr = Math.max(1, window.devicePixelRatio || 1);
  var W = 0, H = 0, pad = 26, scale = 1, ox = 0, oy = 0;
  var ptR = 5;                // dot radius, rescaled with the drawing area

  function fit() {
    var cssW = canvas.clientWidth || 900;
    // On narrow screens make the canvas closer to square and trim the
    // padding so the inner drawing square (and thus the clouds) keeps as
    // many pixels as possible — otherwise the points look cramped.
    var aspect = cssW < 560 ? 0.92 : 0.56;
    pad = cssW < 560 ? 12 : 26;
    var cssH = Math.round(cssW * aspect);
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    W = cssW; H = cssH;
    var s = Math.min(W - 2 * pad, H - 2 * pad);
    scale = s; ox = (W - s) / 2; oy = (H - s) / 2;
    ptR = Math.max(2.6, scale * 0.011);   // keep dots proportional to the cloud
  }
  function px(p) { return [ox + p[0] * scale, oy + (1 - p[1]) * scale]; }
  function unpx(qx, qy) { return [(qx - ox) / scale, 1 - (qy - oy) / scale]; }

  // -------- rendering ---------------------------------------
  function dot(p, color, r) {
    var q = px(p);
    ctx.beginPath();
    ctx.arc(q[0], q[1], r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }

  // principal axis (PCA) of one cloud, drawn dashed in its colour
  function drawPCA(pts, color) {
    var n = pts.length, cx = 0, cy = 0, i;
    for (i = 0; i < n; i++) { cx += pts[i][0]; cy += pts[i][1]; }
    cx /= n; cy /= n;
    var c00 = 0, c01 = 0, c11 = 0;
    for (i = 0; i < n; i++) {
      var ax = pts[i][0] - cx, ay = pts[i][1] - cy;
      c00 += ax * ax; c01 += ax * ay; c11 += ay * ay;
    }
    c00 /= n; c01 /= n; c11 /= n;
    var e = topEig(c00, c01, c11);
    var L = 1.4;
    var a = px([cx - e.u[0] * L, cy - e.u[1] * L]);
    var b = px([cx + e.u[0] * L, cy + e.u[1] * L]);
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.75;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.restore();
  }

  function draw(res) {
    ctx.clearRect(0, 0, W, H);

    // PCA of all points (X and Y treated as one cloud), dashed
    drawPCA(X.concat(Y), COL.pca);

    // mean of all points -> anchor for the worst-case axis
    var mx = 0, my = 0;
    for (var i = 0; i < X.length; i++) { mx += X[i][0]; my += X[i][1]; }
    for (var j = 0; j < Y.length; j++) { mx += Y[j][0]; my += Y[j][1]; }
    mx /= (X.length + Y.length); my /= (X.length + Y.length);

    // worst-case 1-D subspace, drawn across the frame
    var d = res.dir, L = 1.4;
    var aPt = px([mx - d[0] * L, my - d[1] * L]);
    var bPt = px([mx + d[0] * L, my + d[1] * L]);
    ctx.strokeStyle = COL.axis;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(aPt[0], aPt[1]); ctx.lineTo(bPt[0], bPt[1]); ctx.stroke();

    // projections of all points onto the subspace (small ticks)
    ctx.fillStyle = COL.axis;
    function proj(p) {
      var t = (p[0] - mx) * d[0] + (p[1] - my) * d[1];
      return [mx + t * d[0], my + t * d[1]];
    }
    var all = X.concat(Y);
    var tickR = Math.max(1, ptR * 0.32);
    for (var k = 0; k < all.length; k++) {
      var pr = px(proj(all[k]));
      ctx.beginPath(); ctx.arc(pr[0], pr[1], tickR, 0, 2 * Math.PI); ctx.fill();
    }

    // transport plan (optional)
    if (showPlan && res.S) {
      var S = res.S, maxpi = 0, pis = [];
      for (var a = 0; a < S.n; a++) for (var b = 0; b < S.m; b++) {
        var pi = S.u[a] * S.K[a][b] * S.v[b];
        pis.push([a, b, pi]); if (pi > maxpi) maxpi = pi;
      }
      ctx.strokeStyle = COL.plan; ctx.lineWidth = 1;
      for (var e = 0; e < pis.length; e++) {
        var w = pis[e][2] / (maxpi || 1);
        if (w < 0.08) continue;
        var pa = px(X[pis[e][0]]), pb = px(Y[pis[e][1]]);
        ctx.globalAlpha = 0.06 + 0.5 * w;
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // points
    for (var p2 = 0; p2 < X.length; p2++) dot(X[p2], COL.x, ptR);
    for (var q2 = 0; q2 < Y.length; q2++) dot(Y[q2], COL.y, ptR);
  }

  // -------- readout -----------------------------------------
  function setText(id, v) { var el = document.getElementById(id); if (el) el.textContent = v; }
  function report(res) {
    setText("srw-val", res.srw.toFixed(3));
    setText("srw-w2", res.w2.toFixed(3));
    var deg = (Math.atan2(res.dir[1], res.dir[0]) * 180 / Math.PI);
    if (deg < 0) deg += 180;
    setText("srw-dir", deg.toFixed(0) + "°");
  }

  // -------- main loop (recompute only when dirty) -----------
  var dirty = true;
  function frame() {
    if (dirty) {
      dirty = false;
      var res = solveSRW();
      draw(res);
      report(res);
    }
    requestAnimationFrame(frame);
  }
  function invalidate() { dirty = true; }

  // -------- interaction: drag a whole cloud -----------------
  var dragCloud = null, dragPrev = null;
  function pointerPos(ev) {
    var r = canvas.getBoundingClientRect();
    var cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    var cy = (ev.touches ? ev.touches[0].clientY : ev.clientY) - r.top;
    return unpx(cx, cy);
  }
  function nearest(p) {
    var best = null, bd = Infinity;
    function scan(arr, name) {
      for (var i = 0; i < arr.length; i++) {
        var dd = (arr[i][0] - p[0]) * (arr[i][0] - p[0]) + (arr[i][1] - p[1]) * (arr[i][1] - p[1]);
        if (dd < bd) { bd = dd; best = name; }
      }
    }
    scan(X, "X"); scan(Y, "Y");
    return bd < 0.02 ? best : null;
  }
  function down(ev) {
    var p = pointerPos(ev);
    dragCloud = nearest(p);
    if (dragCloud) { dragPrev = p; ev.preventDefault(); }
  }
  function move(ev) {
    if (!dragCloud) return;
    var p = pointerPos(ev);
    var dxp = p[0] - dragPrev[0], dyp = p[1] - dragPrev[1];
    dragPrev = p;
    var c = dragCloud === "X" ? cX : cY;
    c[0] = clampC(c[0] + dxp); c[1] = clampC(c[1] + dyp);
    rebuild(); invalidate(); ev.preventDefault();
  }
  function up() { dragCloud = null; }

  canvas.addEventListener("mousedown", down);
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
  canvas.addEventListener("touchstart", down, { passive: false });
  canvas.addEventListener("touchmove", move, { passive: false });
  window.addEventListener("touchend", up);

  // -------- controls ----------------------------------------
  var elResample = document.getElementById("srw-resample");
  var elPlan = document.getElementById("srw-toggle-plan");
  var elSpread = document.getElementById("srw-spread");
  var elEps = document.getElementById("srw-eps");
  if (elResample) elResample.addEventListener("click", function () { resample(); invalidate(); });
  if (elPlan) elPlan.addEventListener("click", function () {
    showPlan = !showPlan;
    elPlan.textContent = showPlan ? "Hide transport plan" : "Show transport plan";
    elPlan.classList.toggle("ghost", !showPlan);
    invalidate();
  });
  if (elSpread) elSpread.addEventListener("input", function () { spread = +elSpread.value; rebuild(); invalidate(); });
  if (elEps) elEps.addEventListener("input", function () { eps = +elEps.value; invalidate(); });

  window.addEventListener("resize", function () { fit(); invalidate(); });

  // -------- go ----------------------------------------------
  fit();
  resample();
  requestAnimationFrame(frame);
})();
