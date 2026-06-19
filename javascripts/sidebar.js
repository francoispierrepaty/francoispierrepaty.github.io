/* ============================================================
   Shared sidebar — single source of truth across all pages.
   Each page sets <body data-page="home|research|demos|teaching">.
   This script renders the portrait, name, social links and the
   tab navigation into <aside class="sidebar">, marking the
   current page active.
   ============================================================ */
(function () {
  "use strict";

  var CV = "assets/cv/cv.pdf";

  var TABS = [
    { id: "home",     label: "Home",     href: "index.html" },
    { id: "research", label: "Research", href: "research.html" },
    { id: "demos",    label: "Demos",    href: "demos.html" },
    { id: "teaching", label: "Teaching", href: "teaching.html" }
  ];

  var active = (document.body.getAttribute("data-page") || "home").toLowerCase();

  var navItems = TABS.map(function (t) {
    var cls = t.id === active ? ' class="active" aria-current="page"' : "";
    return '<li><a href="' + t.href + '"' + cls + ">" + t.label + "</a></li>";
  }).join("");

  var html =
    '<a class="portrait-link" href="index.html">' +
      '<img src="francoispierre.jpg" alt="François-Pierre Paty" class="portrait" />' +
    "</a>" +
    '<h1 class="name"><a href="index.html">François-Pierre Paty</a></h1>' +
    '<p class="tagline">Applied mathematics · ML · optimal transport · time series · operations research</p>' +
    '<p class="social">' +
      '<a href="mailto:francoispierre.paty@gmail.com" title="Email"><i class="fas fa-envelope"></i></a>' +
      '<a href="https://scholar.google.fr/citations?user=brDW8E8AAAAJ" title="Google Scholar"><i class="ai ai-google-scholar"></i></a>' +
      '<a href="https://arxiv.org/a/paty_f_1" title="arXiv"><i class="ai ai-arxiv"></i></a>' +
      '<a href="https://github.com/francoispierrepaty" title="GitHub"><i class="fab fa-github"></i></a>' +
      '<a href="https://twitter.com/fpierrepaty" title="Twitter"><i class="fab fa-twitter"></i></a>' +
      '<a href="' + CV + '" title="Resume / CV"><i class="ai ai-cv"></i></a>' +
    "</p>" +
    "<nav><ul class=\"nav\">" + navItems +
    "</ul></nav>";

  var aside = document.querySelector("aside.sidebar");
  if (aside) aside.innerHTML = html;
})();
