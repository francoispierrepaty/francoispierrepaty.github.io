/* ============================================================
   Regularized OT is Ground Cost Adversarial - Algorithm 1 demo.

   We solve the nonnegative adversarial-cost problem from the
   paper on a small discrete problem:

     max_{c >= 0} T_c(mu,nu) - eps R*((c-c0)/eps).

   Following the paper's Algorithm 1 and its smoothed variant, each
   ascent step computes a Sinkhorn plan pi_eta for the current cost c
   and updates

     c <- Proj_{R_+}(c + lr * (pi_eta - grad R*((c-c0)/eps))).

   The regularizer selector changes the conjugate R* and therefore
   changes the adversary acting on the ground cost.
   ============================================================ */
(function () {
  "use strict";

  var canvas = document.getElementById("adv-canvas");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");

  var NX = 8;
  var NY = 8;
  var a = new Array(NX).fill(1 / NX);
  var b = new Array(NY).fill(1 / NY);
  var X = [], Y = [], c0 = [], c = [], plan = [];
  var dXX = [], dYY = [];
  var geomScale = 1;
  var eps = 0.06;
  var eta = 0.004;
  var mode = "entropy";
  var dpr = Math.max(1, window.devicePixelRatio || 1);
  var W = 0, H = 0;

  var COL = {
    x: "#2a6f7a",
    y: "#a4452f",
    high: "#123f4a",
    mid: "#b85d3b",
    low: "#efe4cb",
    ink: "#33312e",
    muted: "#6f6a62",
    grid: "#d8d1c2"
  };

  var MODES = {
    entropy: {
      name: "Entropy",
      eps: 0.06,
      epsMin: 0.001,
      epsStep: 0.001,
      epsMax: 0.30,
      equation: "$$\\min_{\\pi\\in\\Pi(\\mu,\\nu)}\\langle c_0,\\pi\\rangle+\\varepsilon\\sum_{ij}\\pi_{ij}(\\log\\pi_{ij}-1)=\\max_c \\mathcal{T}_c(\\mu,\\nu)-\\varepsilon\\sum_{ij}\\exp\\!\\left({c_{ij}-{c_0}_{ij}\\over\\varepsilon}\\right).$$",
      note: "Entropic regularization uses \\(R(\\pi)=\\sum_{ij}\\pi_{ij}(\\log\\pi_{ij}-1)\\), with conjugate \\(R^*(s)=\\sum_{ij}\\exp(s_{ij})\\). The penalty grows exponentially as a cost rises above \\(c_0\\), so the adversary nudges every cost a little but pays steeply for large moves: a soft perturbation spread around \\(c_0\\)."
    },
    quadratic: {
      name: "Quadratic",
      p: 2,
      eps: 2,
      epsMax: 20,
      epsStep: 0.10,
      equation: "$$\\min_{\\pi\\in\\Pi(\\mu,\\nu)}\\langle c_0,\\pi\\rangle+{\\varepsilon\\over 2}\\sum_{ij}\\pi_{ij}^2=\\max_c \\mathcal{T}_c(\\mu,\\nu)-{\\varepsilon\\over 2}\\sum_{ij}\\left[\\left({c_{ij}-{c_0}_{ij}\\over\\varepsilon}\\right)_+\\right]^2.$$",
      note: "Quadratic regularization \\(R(\\pi)=\\tfrac12\\sum_{ij}\\pi_{ij}^2\\) (the \\(p=2\\) case) has conjugate the squared positive part, \\(R^*(s)=\\tfrac12\\sum_{ij}[(s_{ij})_+]^2\\). The penalty only switches on once a cost rises above \\(c_0\\), then grows linearly, so the adversary leaves many costs untouched: the plan and the deformation look sparser than for entropy."
    },
    tsallis: {
      name: "Tsallis",
      q: 0.5,
      eps: 0.05,
      epsMin: 0.004,
      epsMax: 0.20,
      equation: "$$\\min_{\\pi\\in\\Pi(\\mu,\\nu)}\\langle c_0,\\pi\\rangle-2\\varepsilon\\sum_{ij}(\\sqrt{\\pi_{ij}}-\\pi_{ij})=\\max_{c\\le c_0}\\mathcal{T}_c(\\mu,\\nu)-\\varepsilon^2\\sum_{ij}{1\\over {c_0}_{ij}-c_{ij}}+2\\varepsilon.$$",
      note: "Tsallis regularization (here \\(q=\\tfrac12\\)) has a conjugate whose domain forces \\(c\\le c_0\\): the adversary may only lower costs, never raise them. A barrier \\(\\propto 1/(c_0-c)\\) keeps \\(c\\) strictly below \\(c_0\\), so some pairs become cheaper to transport while the rest stay near their prior."
    },
    capacity: {
      name: "Capacity",
      eps: 0.025,
      epsMin: 0.002,
      epsMax: 0.08,
      epsStep: 0.0005,
      equation: "$$\\min_{\\pi\\in\\Pi(\\mu,\\nu),\\ \\|\\pi\\|_\\infty\\le\\varepsilon}\\langle c_0,\\pi\\rangle=\\max_c \\mathcal{T}_c(\\mu,\\nu)-\\varepsilon\\sum_{ij}|c_{ij}-{c_0}_{ij}|.$$",
      note: "Capacity-constrained OT caps the plan, \\(\\|\\pi\\|_\\infty\\le\\varepsilon\\) (the \\(p=\\infty\\) case). Its conjugate is an \\(L^1\\) norm, giving penalty \\(\\varepsilon\\sum_{ij}|c_{ij}-{c_0}_{ij}|\\), and \\(L^1\\) favours sparse deformations: the adversary changes a few costs a lot and leaves the rest at \\(c_0\\). Note the \\(\\varepsilon\\) slider runs in reverse here: small \\(\\varepsilon\\) tightens the cap and forces the plan to spread out, while large \\(\\varepsilon\\) lifts it and recovers the sharp unregularized OT plan, the opposite of the other regularizers, where larger \\(\\varepsilon\\) means more blur."
    }
  };

  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function normalizeCost(M) {
    var maxv = 0;
    for (var i = 0; i < M.length; i++) {
      for (var j = 0; j < M[i].length; j++) maxv = Math.max(maxv, M[i][j]);
    }
    maxv = maxv || 1;
    for (var i2 = 0; i2 < M.length; i2++) {
      for (var j2 = 0; j2 < M[i2].length; j2++) M[i2][j2] /= maxv;
    }
  }

  function makePoints() {
    X = []; Y = [];
    for (var i = 0; i < NX; i++) {
      var theta = 2 * Math.PI * i / NX;
      var radius = 0.165;
      X.push([0.34 + radius * Math.cos(theta), 0.51 + radius * Math.sin(theta)]);
    }
    for (var j = 0; j < 5; j++) {
      Y.push([0.58 + 0.060 * j, 0.665]);
    }
    for (var k = 0; k < 3; k++) {
      Y.push([0.70, 0.555 - 0.095 * k]);
    }
    c0 = new Array(NX);
    dXX = new Array(NX);
    dYY = new Array(NY);
    var maxCross = 0;
    for (var r = 0; r < NX; r++) {
      c0[r] = new Array(NY);
      for (var s = 0; s < NY; s++) {
        var dx = X[r][0] - Y[s][0];
        var dy = X[r][1] - Y[s][1];
        c0[r][s] = dx * dx + dy * dy;
        maxCross = Math.max(maxCross, c0[r][s]);
      }
    }
    maxCross = maxCross || 1;
    geomScale = Math.sqrt(maxCross);
    for (var i2 = 0; i2 < NX; i2++) {
      dXX[i2] = new Array(NX);
      for (var j2 = 0; j2 < NX; j2++) {
        var dxx = X[i2][0] - X[j2][0];
        var dxy = X[i2][1] - X[j2][1];
        dXX[i2][j2] = (dxx * dxx + dxy * dxy) / maxCross;
      }
    }
    for (var i3 = 0; i3 < NY; i3++) {
      dYY[i3] = new Array(NY);
      for (var j3 = 0; j3 < NY; j3++) {
        var dyx = Y[i3][0] - Y[j3][0];
        var dyy = Y[i3][1] - Y[j3][1];
        dYY[i3][j3] = (dyx * dyx + dyy * dyy) / maxCross;
      }
    }
    normalizeCost(c0);
    resetCost();
  }

  function resetCost() {
    c = new Array(NX);
    for (var i = 0; i < NX; i++) {
      c[i] = new Array(NY);
      for (var j = 0; j < NY; j++) c[i][j] = mode === "tsallis" ? 0.72 * c0[i][j] : c0[i][j];
    }
    runAscent();
  }

  function sinkhorn(C, reg, iters) {
    var n = C.length, m = C[0].length;
    var logK = new Array(n);
    for (var i = 0; i < n; i++) {
      logK[i] = new Array(m);
      for (var j = 0; j < m; j++) logK[i][j] = -C[i][j] / reg;
    }
    var logU = new Array(n).fill(0);
    var logV = new Array(m).fill(0);
    for (var it = 0; it < iters; it++) {
      for (var r = 0; r < n; r++) {
        var row = new Array(m);
        for (var j2 = 0; j2 < m; j2++) row[j2] = logK[r][j2] + logV[j2];
        logU[r] = Math.log(a[r]) - logSumExp(row);
      }
      for (var s = 0; s < m; s++) {
        var col = new Array(n);
        for (var i2 = 0; i2 < n; i2++) col[i2] = logK[i2][s] + logU[i2];
        logV[s] = Math.log(b[s]) - logSumExp(col);
      }
    }
    var P = new Array(n);
    for (var i3 = 0; i3 < n; i3++) {
      P[i3] = new Array(m);
      for (var j3 = 0; j3 < m; j3++) P[i3][j3] = Math.exp(logU[i3] + logK[i3][j3] + logV[j3]);
    }
    return P;
  }

  function logSumExp(vals) {
    var mx = -Infinity;
    for (var i = 0; i < vals.length; i++) mx = Math.max(mx, vals[i]);
    if (!isFinite(mx)) return mx;
    var s = 0;
    for (var j = 0; j < vals.length; j++) s += Math.exp(vals[j] - mx);
    return mx + Math.log(s);
  }

  function gradConjugate(s) {
    if (mode === "entropy") return Math.exp(Math.max(-40, Math.min(40, s)));
    if (mode === "tsallis") {
      var tq = MODES.tsallis.q;
      var tp = tq / (tq - 1);
      var neg = Math.min(-1e-3, s);
      return Math.pow(neg / tp, tp - 1);
    }
    if (mode === "capacity") {
      var weight = eps;
      if (s > 1e-4) return weight;
      if (s < -1e-4) return -weight;
      return 0;
    }
    var p = MODES[mode].p;
    var q = p / (p - 1);
    return s > 0 ? Math.pow(s, q - 1) : 0;
  }

  function projectCost(value, prior) {
    value = Math.max(0, value);
    if (mode === "tsallis") return Math.min(value, prior * (1 - 1e-3));
    return value;
  }

  function runAscent() {
    if (!c0.length) return;
    var lr = mode === "capacity" ? 0.16 : 0.42;
    var iters = mode === "capacity" ? 340 : 180;
    var planEta = mode === "entropy"
      ? Math.max(0.00001, Math.min(0.010, eps * 0.20))
      : Math.max(0.0007, Math.min(0.010, eps * 0.20));
    for (var t = 0; t < iters; t++) {
      var P = sinkhorn(c, planEta, 70);
      for (var i = 0; i < NX; i++) {
        for (var j = 0; j < NY; j++) {
          var s = (c[i][j] - c0[i][j]) / eps;
          c[i][j] = projectCost(c[i][j] + lr * (P[i][j] - gradConjugate(s)), c0[i][j]);
        }
      }
      if (t % 20 === 0) lr *= 0.82;
    }
    plan = sinkhorn(c, planEta, 120);
    draw();
  }

  function fit() {
    var cssW = canvas.clientWidth || 900;
    var cssH = cssW < 560 ? Math.round(cssW * 1.08) : Math.round(cssW * 0.78);
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    W = cssW; H = cssH;
    draw();
  }

  function lerp(a0, b0, t) { return a0 + (b0 - a0) * t; }
  function hex(n) { var s = Math.round(n).toString(16); return s.length === 1 ? "0" + s : s; }
  function colorScale(v) {
    v = Math.max(0, Math.min(1, v));
    var aCol = v < 0.55 ? [239, 228, 203] : [184, 93, 59];
    var bCol = v < 0.55 ? [184, 93, 59] : [18, 63, 74];
    var t = v < 0.55 ? v / 0.55 : (v - 0.55) / 0.45;
    return "#" + hex(lerp(aCol[0], bCol[0], t)) + hex(lerp(aCol[1], bCol[1], t)) + hex(lerp(aCol[2], bCol[2], t));
  }

  function signedColorScale(v) {
    v = Math.max(-1, Math.min(1, v));
    var neutral = [247, 243, 234];
    var end = v < 0 ? [42, 111, 122] : [164, 69, 47];
    var t = Math.abs(v);
    return "#" + hex(lerp(neutral[0], end[0], t)) + hex(lerp(neutral[1], end[1], t)) + hex(lerp(neutral[2], end[2], t));
  }

  function matrixRange(M) {
    var mn = Infinity, mx = -Infinity;
    for (var i = 0; i < M.length; i++) {
      for (var j = 0; j < M[i].length; j++) {
        mn = Math.min(mn, M[i][j]);
        mx = Math.max(mx, M[i][j]);
      }
    }
    return { min: mn, max: mx, span: Math.max(1e-12, mx - mn) };
  }

  function drawMatrix(M, x, y, size, title, fixedMax, range) {
    var r = range || matrixRange(M);
    var span = range ? Math.max(1e-12, range.max - range.min) : r.span;
    var rows = M.length, cols = M[0].length;
    var cell = size / Math.max(rows, cols);
    var w = cols * cell, h = rows * cell;
    ctx.fillStyle = COL.ink;
    ctx.font = "600 14px Spectral, Georgia, serif";
    ctx.fillText(title, x, y - 12);
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var v = fixedMax ? M[i][j] / fixedMax : (M[i][j] - r.min) / span;
        ctx.fillStyle = colorScale(v);
        ctx.fillRect(x + j * cell, y + i * cell, cell - 1, cell - 1);
      }
    }
    ctx.strokeStyle = COL.grid;
    ctx.strokeRect(x, y, w, h);
  }

  function combinedRange(A, B) {
    var ra = matrixRange(A), rb = matrixRange(B);
    var mn = Math.min(ra.min, rb.min), mx = Math.max(ra.max, rb.max);
    return { min: mn, max: mx, span: Math.max(1e-12, mx - mn) };
  }

  function drawSignedMatrix(M, x, y, size, title) {
    var maxAbs = 0;
    var rows = M.length, cols = M[0].length;
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) maxAbs = Math.max(maxAbs, Math.abs(M[i][j]));
    }
    maxAbs = maxAbs || 1;
    var cell = size / Math.max(rows, cols);
    var w = cols * cell, h = rows * cell;
    ctx.fillStyle = COL.ink;
    ctx.font = "600 14px Spectral, Georgia, serif";
    ctx.fillText(title, x, y - 12);
    for (var r = 0; r < rows; r++) {
      for (var s = 0; s < cols; s++) {
        ctx.fillStyle = signedColorScale(M[r][s] / maxAbs);
        ctx.fillRect(x + s * cell, y + r * cell, cell - 1, cell - 1);
      }
    }
    ctx.strokeStyle = COL.grid;
    ctx.strokeRect(x, y, w, h);
  }

  function costDelta() {
    var D = new Array(NX);
    for (var i = 0; i < NX; i++) {
      D[i] = new Array(NY);
      for (var j = 0; j < NY; j++) D[i][j] = c[i][j] - c0[i][j];
    }
    return D;
  }

  function jacobiEigen(A) {
    var n = A.length;
    var V = new Array(n);
    for (var i = 0; i < n; i++) {
      V[i] = new Array(n).fill(0);
      V[i][i] = 1;
    }
    for (var sweep = 0; sweep < 80; sweep++) {
      var p = 0, q = 1, max = 0;
      for (var r = 0; r < n; r++) {
        for (var s = r + 1; s < n; s++) {
          var v = Math.abs(A[r][s]);
          if (v > max) { max = v; p = r; q = s; }
        }
      }
      if (max < 1e-10) break;
      var app = A[p][p], aqq = A[q][q], apq = A[p][q];
      var tau = (aqq - app) / (2 * apq);
      var t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
      var cos = 1 / Math.sqrt(1 + t * t);
      var sin = t * cos;
      for (var k = 0; k < n; k++) {
        if (k !== p && k !== q) {
          var akp = A[k][p], akq = A[k][q];
          A[k][p] = A[p][k] = cos * akp - sin * akq;
          A[k][q] = A[q][k] = sin * akp + cos * akq;
        }
      }
      A[p][p] = cos * cos * app - 2 * sin * cos * apq + sin * sin * aqq;
      A[q][q] = sin * sin * app + 2 * sin * cos * apq + cos * cos * aqq;
      A[p][q] = A[q][p] = 0;
      for (var k2 = 0; k2 < n; k2++) {
        var vkp = V[k2][p], vkq = V[k2][q];
        V[k2][p] = cos * vkp - sin * vkq;
        V[k2][q] = sin * vkp + cos * vkq;
      }
    }
    var eig = [];
    for (var e = 0; e < n; e++) eig.push({ value: A[e][e], vector: V.map(function (row) { return row[e]; }) });
    eig.sort(function (u, v) { return v.value - u.value; });
    return eig;
  }

  function adversarialEmbedding() {
    var m = NX + NY;
    var D = new Array(m);
    for (var i = 0; i < m; i++) D[i] = new Array(m).fill(0);
    for (var rx = 0; rx < NX; rx++) {
      for (var sx = 0; sx < NX; sx++) D[rx][sx] = dXX[rx][sx];
    }
    for (var ry = 0; ry < NY; ry++) {
      for (var sy = 0; sy < NY; sy++) D[NX + ry][NX + sy] = dYY[ry][sy];
    }
    for (var r = 0; r < NX; r++) {
      for (var s = 0; s < NY; s++) {
        D[r][NX + s] = c[r][s];
        D[NX + s][r] = c[r][s];
      }
    }
    var rowMean = new Array(m).fill(0);
    var colMean = new Array(m).fill(0);
    var totalMean = 0;
    for (var i2 = 0; i2 < m; i2++) {
      for (var j2 = 0; j2 < m; j2++) {
        rowMean[i2] += D[i2][j2];
        colMean[j2] += D[i2][j2];
        totalMean += D[i2][j2];
      }
    }
    for (var a2 = 0; a2 < m; a2++) {
      rowMean[a2] /= m;
      colMean[a2] /= m;
    }
    totalMean /= m * m;
    var B = new Array(m);
    for (var br = 0; br < m; br++) {
      B[br] = new Array(m);
      for (var bs = 0; bs < m; bs++) {
        B[br][bs] = -0.5 * (D[br][bs] - rowMean[br] - colMean[bs] + totalMean);
      }
    }
    var eig = jacobiEigen(B);
    var coords = new Array(m);
    for (var k = 0; k < m; k++) {
      coords[k] = [
        eig[0].vector[k] * Math.sqrt(Math.max(0, eig[0].value)),
        eig[1].vector[k] * Math.sqrt(Math.max(0, eig[1].value))
      ];
    }
    return alignEmbedding(coords);
  }

  function originalEmbeddingTarget() {
    var target = [];
    for (var i = 0; i < NX; i++) target.push([X[i][0] / geomScale, X[i][1] / geomScale]);
    for (var j = 0; j < NY; j++) target.push([Y[j][0] / geomScale, Y[j][1] / geomScale]);
    return target;
  }

  function centerPoints(points) {
    var n = points.length, cx = 0, cy = 0;
    for (var i = 0; i < n; i++) {
      cx += points[i][0];
      cy += points[i][1];
    }
    cx /= n; cy /= n;
    var out = new Array(n);
    for (var j = 0; j < n; j++) {
      var x = points[j][0] - cx;
      var y = points[j][1] - cy;
      out[j] = [x, y];
    }
    return out;
  }

  function rmsScale(points) {
    var s = 0;
    for (var i = 0; i < points.length; i++) s += points[i][0] * points[i][0] + points[i][1] * points[i][1];
    return Math.sqrt(s / points.length) || 1;
  }

  function rotatePoints(points, theta, reflectX) {
    var cth = Math.cos(theta), sth = Math.sin(theta);
    return points.map(function (p) {
      var x = reflectX ? -p[0] : p[0];
      var y = p[1];
      return [x * cth - y * sth, x * sth + y * cth];
    });
  }

  function alignmentError(A, B) {
    var err = 0;
    for (var i = 0; i < A.length; i++) {
      var dx = A[i][0] - B[i][0];
      var dy = A[i][1] - B[i][1];
      err += dx * dx + dy * dy;
    }
    return err;
  }

  function bestRotation(A, B, reflectX) {
    var h00 = 0, h01 = 0, h10 = 0, h11 = 0;
    for (var i = 0; i < A.length; i++) {
      var ax = reflectX ? -A[i][0] : A[i][0];
      var ay = A[i][1];
      h00 += ax * B[i][0];
      h01 += ax * B[i][1];
      h10 += ay * B[i][0];
      h11 += ay * B[i][1];
    }
    return Math.atan2(h01 - h10, h00 + h11);
  }

  function alignEmbedding(coords) {
    var Araw = centerPoints(coords);
    var A = Araw.map(function (p) { return [p[0], p[1]]; });
    var B = centerPoints(originalEmbeddingTarget());
    var bScale = rmsScale(B);
    var aScale = rmsScale(A);
    for (var i = 0; i < A.length; i++) {
      A[i][0] /= aScale;
      A[i][1] /= aScale;
    }
    for (var j = 0; j < B.length; j++) {
      B[j][0] /= bScale;
      B[j][1] /= bScale;
    }
    var theta0 = bestRotation(A, B, false);
    var C0 = rotatePoints(A, theta0, false);
    var theta1 = bestRotation(A, B, true);
    var C1 = rotatePoints(A, theta1, true);
    return alignmentError(C0, B) <= alignmentError(C1, B)
      ? rotatePoints(Araw, theta0, false)
      : rotatePoints(Araw, theta1, true);
  }

  function dot(x, y, color) {
    ctx.beginPath();
    ctx.arc(x, y, 5.5, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }

  function drawCenteredCloud(points, x, y, w, h, title, linesFromPlan) {
    var inset = 7;
    var px0 = x + inset, py0 = y + inset, pw = w - 2 * inset, ph = h - 2 * inset;
    var side = Math.min(pw, ph);
    var ox = px0 + (pw - side) / 2;
    var oy = py0 + (ph - side) / 2;
    var target = centerPoints(originalEmbeddingTarget());
    var maxAbs = 0;
    for (var i = 0; i < target.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(target[i][0]), Math.abs(target[i][1]));
    }
    maxAbs = maxAbs || 1;
    var centered = centerPoints(points);
    var sx = function (p) { return ox + 0.5 * side + (p[0] / maxAbs) * 0.42 * side; };
    var sy = function (p) { return oy + 0.5 * side - (p[1] / maxAbs) * 0.42 * side; };
    ctx.fillStyle = COL.ink;
    ctx.font = "600 14px Spectral, Georgia, serif";
    ctx.textAlign = "center";
    ctx.fillText(title, x + w / 2, y - 12);
    ctx.textAlign = "left";
    ctx.strokeStyle = COL.grid;
    ctx.strokeRect(px0, py0, pw, ph);
    if (linesFromPlan) {
      var maxp = 0;
      for (var r = 0; r < NX; r++) for (var s = 0; s < NY; s++) maxp = Math.max(maxp, plan[r][s]);
      for (var r2 = 0; r2 < NX; r2++) {
        for (var s2 = 0; s2 < NY; s2++) {
          var alpha = plan[r2][s2] / (maxp || 1);
          if (alpha < 0.015) continue;
          ctx.beginPath();
          ctx.moveTo(sx(centered[r2]), sy(centered[r2]));
          ctx.lineTo(sx(centered[NX + s2]), sy(centered[NX + s2]));
          ctx.strokeStyle = "rgba(106,98,88," + (0.10 + 0.45 * alpha).toFixed(3) + ")";
          ctx.lineWidth = 0.6 + 2.8 * alpha;
          ctx.stroke();
        }
      }
    }
    for (var i2 = 0; i2 < NX; i2++) {
      dot(sx(centered[i2]), sy(centered[i2]), COL.x);
    }
    for (var j2 = 0; j2 < NY; j2++) {
      dot(sx(centered[NX + j2]), sy(centered[NX + j2]), COL.y);
    }
  }

  function drawOriginalGeometry(x, y, w, h) {
    var inset = 7;
    var px0 = x + inset, py0 = y + inset, pw = w - 2 * inset, ph = h - 2 * inset;
    var side = Math.min(pw, ph);
    var ox = px0 + (pw - side) / 2;
    var oy = py0 + (ph - side) / 2;
    var sx = function (p) { return ox + p[0] * side; };
    var sy = function (p) { return oy + (1 - p[1]) * side; };
    ctx.fillStyle = COL.ink;
    ctx.font = "600 14px Spectral, Georgia, serif";
    ctx.textAlign = "center";
    ctx.fillText("true geometry (prior cost c0)", x + w / 2, y - 12);
    ctx.textAlign = "left";
    ctx.strokeStyle = COL.grid;
    ctx.strokeRect(px0, py0, pw, ph);
    var maxp = 0;
    for (var r = 0; r < NX; r++) for (var s = 0; s < NY; s++) maxp = Math.max(maxp, plan[r][s]);
    for (var r2 = 0; r2 < NX; r2++) {
      for (var s2 = 0; s2 < NY; s2++) {
        var alpha = plan[r2][s2] / (maxp || 1);
        if (alpha < 0.015) continue;
        ctx.beginPath();
        ctx.moveTo(sx(X[r2]), sy(X[r2]));
        ctx.lineTo(sx(Y[s2]), sy(Y[s2]));
        ctx.strokeStyle = "rgba(106,98,88," + (0.10 + 0.45 * alpha).toFixed(3) + ")";
        ctx.lineWidth = 0.6 + 2.8 * alpha;
        ctx.stroke();
      }
    }
    for (var i = 0; i < NX; i++) dot(sx(X[i]), sy(X[i]), COL.x);
    for (var j = 0; j < NY; j++) dot(sx(Y[j]), sy(Y[j]), COL.y);
  }

  function drawWarpedGeometry(x, y, w, h) {
    var coords = adversarialEmbedding();
    drawCenteredCloud(coords, x, y, w, h, "adversarial geometry (best 2-D MDS fit)", true);
  }

  function draw() {
    if (!W || !plan.length) return;
    ctx.clearRect(0, 0, W, H);
    var pad = 24;
    var top = 42;
    var gap = 22;
    var geomW = Math.min(390, W * 0.46);
    var geomGap = 14;
    var smallGeomW = (geomW - geomGap) / 2;
    var matrixSize = Math.min(118, (W - 2 * pad - geomW - 4 * gap) / 4);
    var costRange = combinedRange(c0, c);
    if (W < 560) {
      matrixSize = Math.min(118, (W - 2 * pad - 3 * gap) / 4);
      var mx0 = Math.max(pad, (W - 4 * matrixSize - 3 * gap) / 2);
      drawMatrix(plan, mx0, top, matrixSize, "plan", 1 / NX);
      drawMatrix(c0, mx0 + matrixSize + gap, top, matrixSize, "prior c0", null, costRange);
      drawMatrix(c, mx0 + 2 * (matrixSize + gap), top, matrixSize, "adv c", null, costRange);
      drawSignedMatrix(costDelta(), mx0 + 3 * (matrixSize + gap), top, matrixSize, "change");
      var py = top + matrixSize + 54;
      var compactW = W - 2 * pad;
      var compactH = Math.max(150, Math.min(compactW, (H - py - 34) / 2));
      drawOriginalGeometry(pad, py, compactW, compactH);
      drawWarpedGeometry(pad, py + compactH + 34, compactW, compactH);
    } else {
      matrixSize = Math.min(128, (W - 2 * pad - 3 * gap) / 4);
      var mx = Math.max(pad, (W - 4 * matrixSize - 3 * gap) / 2);
      drawMatrix(plan, mx, top, matrixSize, "transport plan", 1 / NX);
      drawMatrix(c0, mx + matrixSize + gap, top, matrixSize, "prior cost c0", null, costRange);
      drawMatrix(c, mx + 2 * (matrixSize + gap), top, matrixSize, "adversarial cost c", null, costRange);
      drawSignedMatrix(costDelta(), mx + 3 * (matrixSize + gap), top, matrixSize, "cost change c-c0");
      var gy = top + matrixSize + 58;
      var geomPanelW = (W - 2 * pad - geomGap) / 2;
      var geomH = Math.max(180, Math.min(geomPanelW, H - gy - 24));
      drawOriginalGeometry(pad, gy, geomPanelW, geomH);
      drawWarpedGeometry(pad + geomPanelW + geomGap, gy, geomPanelW, geomH);
    }
    updateReadout();
  }

  function updateReadout() {
    var entropy = 0, active = 0;
    for (var i = 0; i < NX; i++) {
      for (var j = 0; j < NY; j++) {
        var p = plan[i][j];
        if (p > 1e-5) entropy -= p * Math.log(p);
        if (p > 0.01) active++;
      }
    }
    var cr = matrixRange(c);
    document.getElementById("adv-reg-name").textContent = MODES[mode].name;
    document.getElementById("adv-plan-entropy").textContent = entropy.toFixed(2);
    document.getElementById("adv-active-links").textContent = String(active);
    document.getElementById("adv-cost-spread").textContent = cr.span.toFixed(2);
    updateEquation();
  }

  function updateEquation() {
    var eq = document.getElementById("adv-equation");
    if (!eq || eq.getAttribute("data-mode") === mode) return;
    eq.setAttribute("data-mode", mode);
    eq.innerHTML = MODES[mode].equation;
    var ex = document.getElementById("adv-explainer");
    if (ex) ex.innerHTML = MODES[mode].note;
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise(ex ? [eq, ex] : [eq]);
    }
  }

  function setMode(next) {
    mode = next;
    var slider = document.getElementById("adv-eps");
    if (slider) {
      slider.min = MODES[mode].epsMin || 0.01;
      slider.max = MODES[mode].epsMax || 0.20;
      slider.step = MODES[mode].epsStep || 0.002;
    }
    if (MODES[mode].eps) {
      eps = MODES[mode].eps;
      if (slider) slider.value = eps.toFixed(3);
    }
    Array.prototype.forEach.call(document.querySelectorAll(".adv-mode"), function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-regularizer") === mode);
    });
    resetCost();
  }

  Array.prototype.forEach.call(document.querySelectorAll(".adv-mode"), function (btn) {
    btn.addEventListener("click", function () {
      setMode(btn.getAttribute("data-regularizer"));
    });
  });

  var epsSlider = document.getElementById("adv-eps");
  if (epsSlider) {
    epsSlider.addEventListener("input", function () {
      eps = parseFloat(epsSlider.value);
      resetCost();
    });
  }
  window.addEventListener("resize", fit);
  makePoints();
  fit();
})();
