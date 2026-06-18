/* ============================================================
   Weak OT in Economics - small mirror-ascent demo.

   The discrete solvers follow Algorithm 1 of Paty, Chone and
   Kramarz (2022):
     P <- P * exp(gamma * grad f(P))
   followed by the KL projection:
     - OT/WOT: Sinkhorn projection onto Pi(a,b)
     - WOTUK: closed-form column scaling onto {P : P^T 1 = b}
   Entropic OT is the usual Sinkhorn solve on exp(F / epsilon).
   ============================================================ */
(function () {
  "use strict";

  var canvas = document.getElementById("econ-canvas");
  if (!canvas) return;

  var ctx = canvas.getContext("2d");
  var workerSlider = document.getElementById("econ-workers");
  var firmSlider = document.getElementById("econ-firms");
  var modelButtons = Array.prototype.slice.call(document.querySelectorAll(".econ-model"));
  var modelName = document.getElementById("econ-model-name");
  var concentrationEl = document.getElementById("econ-concentration");
  var sizeSpreadEl = document.getElementById("econ-size-spread");
  var explainerEl = document.getElementById("econ-explainer");

  var model = "ot";
  var N = 13;
  var MIRROR_ITERS = 90;
  var SINKHORN_ITERS = 70;
  var CES_RHO = 0.55;
  var POS = 1e-8;
  var dpr = Math.max(1, window.devicePixelRatio || 1);
  var W = 900;
  var H = 620;

  var TEXT = {
    ot: {
      name: "OT",
      note: "Each firm hires essentially one kind of worker: skill-1 firms take skill-1 specialists, skill-2 firms take skill-2 specialists. The matching is sharp and runs along the diagonal."
    },
    eot: {
      name: "Entropic OT",
      note: "The same pairwise logic, blurred by an entropy term. Firms still favour nearby worker types, but the match spreads out into a soft band around the diagonal."
    },
    wot: {
      name: "WOT",
      note: ""
    },
    wotuk: {
      name: "WOTUK",
      note: "Firm size is chosen by the model rather than fixed in advance. The worker population is still fully employed, but some firm types grow large and others shrink, so a firm-size distribution emerges."
    }
  };

  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  function gaussian(x, mu, sigma) {
    var z = (x - mu) / sigma;
    return Math.exp(-0.5 * z * z);
  }

  function mixWorkerSupply(generalists) {
    var b = [];
    var sum = 0;
    for (var j = 0; j < N; j++) {
      var t = j / (N - 1);
      var specialists = gaussian(t, 0.08, 0.12) + gaussian(t, 0.92, 0.12);
      var middle = gaussian(t, 0.50, 0.18);
      var v = (1 - generalists) * specialists + generalists * 1.8 * middle + 0.08;
      b.push(v);
      sum += v;
    }
    return b.map(function (v) { return v / sum; });
  }

  function firmDemand(i, firmSpecialization) {
    var centered = i / (N - 1) - 0.5;
    return clamp(0.5 + centered * (0.35 + 0.65 * firmSpecialization), 0.02, 0.98);
  }

  function workerSkill(j) {
    var t = j / (N - 1);
    return [1 - t, t];
  }

  function firmSkill(i, firmSpecialization) {
    var alpha = firmDemand(i, firmSpecialization);
    return [1 - alpha, alpha];
  }

  function uniformFirmMass() {
    var a = [];
    for (var i = 0; i < N; i++) a.push(1 / N);
    return a;
  }

  function zeros() {
    var M = [];
    for (var i = 0; i < N; i++) {
      var row = [];
      for (var j = 0; j < N; j++) row.push(0);
      M.push(row);
    }
    return M;
  }

  function initialCoupling(a, b) {
    var P = zeros();
    for (var i = 0; i < N; i++) {
      for (var j = 0; j < N; j++) P[i][j] = a[i] * b[j];
    }
    return P;
  }

  function pairwiseProduction(firmSpecialization) {
    var F = zeros();
    for (var i = 0; i < N; i++) {
      var x = firmSkill(i, firmSpecialization);
      for (var j = 0; j < N; j++) {
        F[i][j] = cesProduction(x, workerSkill(j));
      }
    }
    return F;
  }

  function cesProduction(alpha, z) {
    var z0 = Math.max(POS, z[0]);
    var z1 = Math.max(POS, z[1]);
    var s = alpha[0] * Math.pow(z0, CES_RHO) + alpha[1] * Math.pow(z1, CES_RHO);
    return Math.pow(Math.max(POS, s), 1 / CES_RHO);
  }

  function cesGradient(alpha, z) {
    var z0 = Math.max(POS, z[0]);
    var z1 = Math.max(POS, z[1]);
    var s = alpha[0] * Math.pow(z0, CES_RHO) + alpha[1] * Math.pow(z1, CES_RHO);
    s = Math.max(POS, s);
    var factor = Math.pow(s, 1 / CES_RHO - 1);
    return [
      alpha[0] * Math.pow(z0, CES_RHO - 1) * factor,
      alpha[1] * Math.pow(z1, CES_RHO - 1) * factor
    ];
  }

  function sinkhornProject(K, a, b) {
    var u = new Array(N).fill(1);
    var v = new Array(N).fill(1);
    for (var it = 0; it < SINKHORN_ITERS; it++) {
      for (var i = 0; i < N; i++) {
        var rs = 0;
        for (var j = 0; j < N; j++) rs += K[i][j] * v[j];
        u[i] = a[i] / Math.max(rs, 1e-300);
      }
      for (var j2 = 0; j2 < N; j2++) {
        var cs = 0;
        for (var i2 = 0; i2 < N; i2++) cs += u[i2] * K[i2][j2];
        v[j2] = b[j2] / Math.max(cs, 1e-300);
      }
    }

    var P = zeros();
    for (var i = 0; i < N; i++) {
      for (var j = 0; j < N; j++) P[i][j] = u[i] * K[i][j] * v[j];
    }
    return P;
  }

  function columnProject(P, b) {
    for (var j = 0; j < N; j++) {
      var cs = 0;
      for (var i = 0; i < N; i++) cs += P[i][j];
      var s = b[j] / Math.max(cs, 1e-300);
      for (var i2 = 0; i2 < N; i2++) P[i2][j] *= s;
    }
  }

  function gradientWOT(P, a, firmSpecialization) {
    var G = zeros();
    for (var i = 0; i < N; i++) {
      var x = firmSkill(i, firmSpecialization);
      var z0 = 0, z1 = 0;
      for (var j = 0; j < N; j++) {
        var y = workerSkill(j);
        z0 += P[i][j] * y[0] / a[i];
        z1 += P[i][j] * y[1] / a[i];
      }
      var grad = cesGradient(x, [z0, z1]);
      for (var j2 = 0; j2 < N; j2++) {
        var y2 = workerSkill(j2);
        G[i][j2] = grad[0] * y2[0] + grad[1] * y2[1];
      }
    }
    return G;
  }

  function gradientWOTUK(P, a, firmSpecialization) {
    var G = zeros();
    for (var i = 0; i < N; i++) {
      var x = firmSkill(i, firmSpecialization);
      var z0 = 0, z1 = 0;
      for (var j = 0; j < N; j++) {
        var y = workerSkill(j);
        var q = P[i][j] / a[i];
        z0 += q * y[0];
        z1 += q * y[1];
      }
      var grad = cesGradient(x, [z0, z1]);
      for (var j2 = 0; j2 < N; j2++) {
        var y2 = workerSkill(j2);
        G[i][j2] = grad[0] * y2[0] + grad[1] * y2[1];
      }
    }
    return G;
  }

  function mirrorUpdate(P, G, gamma) {
    var maxG = -Infinity;
    for (var i = 0; i < N; i++) {
      for (var j = 0; j < N; j++) maxG = Math.max(maxG, G[i][j]);
    }
    for (var i2 = 0; i2 < N; i2++) {
      for (var j2 = 0; j2 < N; j2++) {
        var exponent = clamp(gamma * (G[i2][j2] - maxG), -40, 0);
        P[i2][j2] = Math.max(1e-300, P[i2][j2] * Math.exp(exponent));
      }
    }
  }

  function solveOT(a, b, firmSpecialization) {
    var F = pairwiseProduction(firmSpecialization);
    var P = initialCoupling(a, b);
    var gamma = 0.55;
    for (var it = 0; it < MIRROR_ITERS; it++) {
      mirrorUpdate(P, F, gamma);
      P = sinkhornProject(P, a, b);
    }
    return P;
  }

  function solveEntropicOT(a, b, firmSpecialization) {
    var F = pairwiseProduction(firmSpecialization);
    var epsilon = 0.035;
    var K = zeros();
    for (var i = 0; i < N; i++) {
      var rowMax = Math.max.apply(null, F[i]);
      for (var j = 0; j < N; j++) K[i][j] = Math.exp((F[i][j] - rowMax) / epsilon);
    }
    return sinkhornProject(K, a, b);
  }

  function solveWOT(a, b, firmSpecialization) {
    var P = initialCoupling(a, b);
    var gamma = 0.10;
    for (var it = 0; it < MIRROR_ITERS; it++) {
      mirrorUpdate(P, gradientWOT(P, a, firmSpecialization), gamma);
      P = sinkhornProject(P, a, b);
    }
    return P;
  }

  function solveWOTUK(a, b, firmSpecialization) {
    var P = initialCoupling(a, b);
    var gamma = 0.16;
    for (var it = 0; it < MIRROR_ITERS; it++) {
      mirrorUpdate(P, gradientWOTUK(P, a, firmSpecialization), gamma);
      columnProject(P, b);
    }
    return P;
  }

  function summarize(P, firms, supply) {
    var sizes = [];
    var concentration = 0;
    var maxSize = 0, minSize = Infinity;
    for (var i = 0; i < N; i++) {
      var rowSum = 0, rowMax = 0;
      for (var j = 0; j < N; j++) {
        rowSum += P[i][j];
        rowMax = Math.max(rowMax, P[i][j]);
      }
      var size = rowSum * N;
      sizes.push(size);
      maxSize = Math.max(maxSize, size);
      minSize = Math.min(minSize, size);
      concentration += rowMax / Math.max(rowSum, 1e-300);
    }
    return {
      supply: supply,
      firms: firms,
      sizes: sizes,
      matrix: P,
      concentration: concentration / N,
      sizeSpread: maxSize - minSize
    };
  }

  function build() {
    var generalists = parseFloat(workerSlider.value);
    var firmSpecialization = parseFloat(firmSlider.value);
    var a = uniformFirmMass();
    var b = mixWorkerSupply(generalists);
    var P;

    if (model === "ot") P = solveOT(a, b, firmSpecialization);
    else if (model === "eot") P = solveEntropicOT(a, b, firmSpecialization);
    else if (model === "wot") P = solveWOT(a, b, firmSpecialization);
    else P = solveWOTUK(a, b, firmSpecialization);

    return summarize(P, a, b);
  }

  // Layout geometry (CSS px). The figure is a coupling shown with its two
  // marginals on the margins: the heatmap is the matching, firm sizes are the
  // row sums (right), the worker distribution is the column sums (below).
  // The desktop branch keeps the original fixed geometry untouched. Below a
  // narrow canvas width (phones such as an iPhone 12 mini, where the canvas is
  // only ~290 CSS px wide) the fixed margins would swallow the whole figure, so
  // a compact branch shrinks margins, bars, fonts and label text instead.
  var COMPACT_W = 440;
  var LAY;

  function setLayout(cssW) {
    if (cssW >= COMPACT_W) {
      LAY = {
        compact: false,
        marginL: 104,  // rotated firm axis + "more skill" labels
        marginR: 150,  // firm-size bars + label
        top: 56,
        histGap: 30,   // heatmap bottom -> histogram title
        histTitle: 18, // space reserved for the histogram title
        histH: 150,    // worker distribution height
        bottomPad: 56, // shared worker-type axis labels
        barMax: 78,    // longest firm-size bar
        barGap: 26,    // heatmap right edge -> firm-size bars
        firmAxisOff: 74, // rotated "firm technology" label offset
        gridMin: 150,
        fs: 1
      };
    } else {
      LAY = {
        compact: true,
        marginL: 40,
        marginR: 72,
        top: 44,
        histGap: 22,
        histTitle: 16,
        histH: 74,
        bottomPad: 46,
        barMax: 48,
        barGap: 12,
        firmAxisOff: 27,
        gridMin: 96,
        fs: 0.86
      };
    }
  }

  var INK = "#33312e";
  var MUTED = "#6f6a62";
  var FAINT = "#9a948a";

  function fit() {
    var cssW = canvas.clientWidth || 900;
    setLayout(cssW);
    var grid = Math.max(LAY.gridMin, cssW - LAY.marginL - LAY.marginR);
    var cssH = LAY.top + grid + LAY.histGap + LAY.histTitle + LAY.histH + LAY.bottomPad;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    W = cssW;
    H = cssH;
    draw();
  }

  function heatColor(t) {
    t = clamp(t, 0, 1);
    var light = [247, 241, 227];
    var mid = [193, 124, 60];
    var dark = [16, 63, 74];
    var a = t < 0.55 ? t / 0.55 : (t - 0.55) / 0.45;
    var from = t < 0.55 ? light : mid;
    var to = t < 0.55 ? mid : dark;
    var r = Math.round(from[0] + (to[0] - from[0]) * a);
    var g = Math.round(from[1] + (to[1] - from[1]) * a);
    var b = Math.round(from[2] + (to[2] - from[2]) * a);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function label(txt, x, y, opts) {
    opts = opts || {};
    ctx.save();
    ctx.fillStyle = opts.color || MUTED;
    ctx.font = (opts.weight || "400") + " " + (opts.size || 14) + "px Spectral, Georgia, serif";
    ctx.textAlign = opts.align || "left";
    ctx.textBaseline = opts.baseline || "alphabetic";
    if (opts.rotate) {
      ctx.translate(x, y);
      ctx.rotate(opts.rotate);
      ctx.fillText(txt, 0, 0);
    } else {
      ctx.fillText(txt, x, y);
    }
    ctx.restore();
  }

  function fsz(s) {
    return s * LAY.fs;
  }

  function roundRect(x, y, w, h, r) {
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function draw() {
    var data = build();
    ctx.clearRect(0, 0, W, H);

    var left = LAY.marginL;
    var top = LAY.top;
    var grid = Math.min(W - LAY.marginL - LAY.marginR,
                        H - LAY.top - LAY.histGap - LAY.histTitle - LAY.histH - LAY.bottomPad);
    grid = Math.max(LAY.gridMin, grid);
    var cell = grid / N;
    var right = left + grid;
    var bottom = top + grid;

    var maxCell = 0;
    for (var i = 0; i < N; i++) {
      for (var j = 0; j < N; j++) maxCell = Math.max(maxCell, data.matrix[i][j]);
    }

    // ---- heatmap cells (the matching) ----
    for (var row = 0; row < N; row++) {
      for (var col = 0; col < N; col++) {
        var strength = data.matrix[row][col] / (maxCell || 1);
        ctx.fillStyle = heatColor(Math.pow(strength, 0.72));
        ctx.fillRect(left + col * cell + 0.6, top + row * cell + 0.6, cell - 1.2, cell - 1.2);
      }
    }
    ctx.strokeStyle = "#d9d3c6";
    ctx.lineWidth = 1;
    ctx.strokeRect(left + 0.5, top + 0.5, grid, grid);

    // ---- firm axis (rows) ----
    label("firm technology", left - LAY.firmAxisOff, top + grid / 2, { rotate: -Math.PI / 2, align: "center", size: fsz(13.5), color: INK, weight: "600" });
    if (!LAY.compact) {
      label("more skill 1", left - 14, top + 5, { align: "right", size: 11.5, color: FAINT });
      label("more skill 2", left - 14, bottom - 1, { align: "right", size: 11.5, color: FAINT });
    }

    // ---- firm sizes (row marginal, on the right) ----
    // Drawn on a common, absolute scale across all models. Firm size is the
    // share of total employment, normalized so the average firm is 1.0; the
    // dashed line marks that equal-share size. In OT, EOT and WOT every firm
    // has size 1, so all bars reach the line; only WOTUK reallocates mass,
    // pushing some firms past it and leaving others short.
    var barX = right + LAY.barGap;
    var unit = LAY.barMax / 2.6;   // pixels per unit of firm size
    var refX = barX + unit;        // equal-share reference (size = 1)
    label("firm size", barX, top - 16, { weight: "600", color: INK, size: fsz(14) });

    ctx.save();
    ctx.strokeStyle = "#c2bbac";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(refX, top - 4);
    ctx.lineTo(refX, bottom + 4);
    ctx.stroke();
    ctx.restore();
    label(LAY.compact ? "avg" : "equal share", refX, bottom + 16, { align: "center", size: fsz(10.5), color: FAINT, baseline: "top" });

    for (var s = 0; s < N; s++) {
      var bw = clamp(unit * data.sizes[s], 1.5, LAY.barMax);
      var by = top + s * cell + cell * 0.2;
      var bh = cell * 0.6;
      ctx.fillStyle = "#8f3b2d";
      roundRect(barX, by, bw, bh, bh / 2);
      ctx.fill();
    }

    // ---- worker distribution (column marginal, below, aligned to columns) ----
    var histTop = bottom + LAY.histGap + LAY.histTitle;
    var histH = LAY.histH;
    label("worker distribution  ν", left, bottom + LAY.histGap + 2, { weight: "600", color: INK, size: fsz(14) });

    var maxSupply = Math.max.apply(null, data.supply) || 1;
    // soft guide line at the top of the tallest bar
    ctx.strokeStyle = "#ece6d9";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, histTop);
    ctx.lineTo(right, histTop);
    ctx.stroke();

    for (var k = 0; k < N; k++) {
      var h = (histH - 4) * data.supply[k] / maxSupply;
      var bx = left + k * cell + cell * 0.16;
      var bwk = cell * 0.68;
      ctx.fillStyle = "#d7b46a";
      roundRect(bx, histTop + histH - h, bwk, h, Math.min(3, bwk / 2));
      ctx.fill();
    }
    ctx.strokeStyle = "#b9b2a4";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(left, histTop + histH + 0.5);
    ctx.lineTo(right, histTop + histH + 0.5);
    ctx.stroke();

    // ---- shared worker-type axis (under heatmap + histogram) ----
    var axisY = histTop + histH + 22;
    var leftLab = LAY.compact ? "skill 1" : "skill-1 specialists";
    var rightLab = LAY.compact ? "skill 2" : "skill-2 specialists";
    label(leftLab, left, axisY, { align: "left", size: fsz(11.5), color: FAINT });
    label(rightLab, right, axisY, { align: "right", size: fsz(11.5), color: FAINT });

    // The centered "worker type" caption is only drawn when it clears both edge
    // labels; on a narrow grid (phones) it would overlap them, and the "worker
    // distribution ν" title above already names the axis, so it is dropped.
    ctx.font = "600 " + fsz(13.5) + "px Spectral, Georgia, serif";
    var centerHalf = ctx.measureText("worker type").width / 2;
    ctx.font = "400 " + fsz(11.5) + "px Spectral, Georgia, serif";
    var edgeRoom = Math.max(ctx.measureText(leftLab).width, ctx.measureText(rightLab).width);
    if (!LAY.compact && edgeRoom + centerHalf + 16 <= grid / 2) {
      label("worker type", left + grid / 2, axisY, { align: "center", size: fsz(13.5), color: INK, weight: "600" });
    }

    modelName.textContent = TEXT[model].name;
    concentrationEl.textContent = Math.round(data.concentration * 100) + "%";
    sizeSpreadEl.textContent = data.sizeSpread.toFixed(2);
    explainerEl.textContent = TEXT[model].note;
  }

  modelButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      model = button.getAttribute("data-model");
      modelButtons.forEach(function (b) {
        b.classList.toggle("active", b === button);
      });
      draw();
    });
  });

  workerSlider.addEventListener("input", draw);
  firmSlider.addEventListener("input", draw);
  window.addEventListener("resize", fit);
  fit();
})();
