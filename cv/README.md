# CV

Single source of truth for my CV. Edit data, run one command, get a PDF.

## How it works

```
cv.json          all the content (the only file you normally edit)
template.html.j2 Jinja2 template: turns the data into HTML
cv.css           print styling, in the spirit of the website
build_cv.py       cv.json + template -> assets/cv/cv.pdf
```

Pipeline: `cv.json` → (Jinja2) → HTML → (WeasyPrint) → `../assets/cv/cv.pdf`.
No LaTeX, no Overleaf.

## Update the CV

1. Edit `cv.json` (add a publication, change a title, etc.).
2. Rebuild:
   ```bash
   python3 cv/build_cv.py
   ```
3. The PDF is written to `assets/cv/cv.pdf`. A `cv.preview.html` is also
   written next to this README for a quick browser preview (it is gitignored).

## Conventions baked into the data

- `period` strings use an en dash with spaces, e.g. `"Oct 2023 – Sept 2025"`.
  The template splits on the dash to stack the start/end dates in the rail.
  A value with no dash (e.g. `"Dec 2024"`) shows as a single line.
- `publications` are listed newest first; they are auto-numbered with the
  newest getting the highest number.
- In `talks` and `teaching`, the list key is `entries` (not `items`, which
  collides with a dict method in Jinja2).

## Dependencies

`python3` with `jinja2` and `weasyprint`. Both already installed locally.
Fonts (Spectral) are fetched from Google Fonts at build time; if offline,
the CV falls back to Georgia/serif.
