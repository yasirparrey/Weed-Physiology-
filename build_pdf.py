#!/usr/bin/env python3
"""Build the CME 295 lecture notes PDF from the markdown sources in notes/.

Usage:  python3 build_pdf.py
Output: CME295-Transformers-and-LLMs-Notes.pdf
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).parent
NOTES_DIR = ROOT / "notes"
OUTPUT = ROOT / "CME295-Transformers-and-LLMs-Notes.pdf"

CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 18mm 18mm;
    @bottom-center {
        content: counter(page);
        font-family: "DejaVu Serif", serif;
        font-size: 8.5pt;
        color: #8a8a8a;
    }
    @top-right {
        content: "CME 295 — Transformers & Large Language Models";
        font-family: "DejaVu Sans", sans-serif;
        font-size: 7.5pt;
        color: #a8a8a8;
    }
}
@page :first {
    @top-right { content: ""; }
    @bottom-center { content: ""; }
}

html { font-size: 10pt; }
body {
    font-family: "DejaVu Serif", serif;
    line-height: 1.5;
    color: #1c1c1c;
    text-align: left;
    hyphens: auto;
}

/* ---- title page ---- */
.titlepage {
    page-break-after: always;
    text-align: center;
    padding-top: 55mm;
}
.titlepage h1 {
    font-size: 25pt;
    border: none;
    margin: 0 0 6mm 0;
    padding: 0;
    color: #8c1515;           /* Stanford cardinal */
    line-height: 1.2;
}
.titlepage h3 {
    font-size: 13pt;
    font-weight: normal;
    color: #333;
    margin: 0 0 14mm 0;
    border: none;
    padding: 0;
}
.titlepage em { color: #555; font-size: 11pt; }
.titlepage hr {
    width: 40%;
    margin: 12mm auto;
    border: none;
    border-top: 1px solid #c9a0a0;
}

/* ---- headings ---- */
h1 {
    font-family: "DejaVu Sans", sans-serif;
    font-size: 17pt;
    color: #8c1515;
    border-bottom: 2px solid #8c1515;
    padding-bottom: 2mm;
    margin: 0 0 5mm 0;
    page-break-before: always;
    page-break-after: avoid;
}
.titlepage h1, #toc h1 { page-break-before: auto; }
h2 {
    font-family: "DejaVu Sans", sans-serif;
    font-size: 13pt;
    color: #16324f;
    margin: 8mm 0 2.5mm 0;
    page-break-after: avoid;
}
h3 {
    font-family: "DejaVu Sans", sans-serif;
    font-size: 11pt;
    color: #16324f;
    margin: 6mm 0 2mm 0;
    page-break-after: avoid;
}
h4 {
    font-family: "DejaVu Sans", sans-serif;
    font-size: 10pt;
    margin: 4mm 0 1.5mm 0;
    page-break-after: avoid;
}

p { margin: 0 0 2.6mm 0; }
strong { color: #000; }

/* ---- lists ---- */
ul, ol { margin: 0 0 2.8mm 0; padding-left: 6mm; }
li { margin-bottom: 1.1mm; }
li > ul, li > ol { margin-top: 1.1mm; margin-bottom: 0; }

/* ---- code ---- */
code {
    font-family: "DejaVu Sans Mono", monospace;
    font-size: 8.4pt;
    background: #f2f2f0;
    padding: 0.3mm 0.9mm;
    border-radius: 1.5px;
}
pre {
    font-family: "DejaVu Sans Mono", monospace;
    background: #f7f7f5;
    border-left: 3px solid #9aa5ae;
    padding: 2.4mm 3mm;
    margin: 0 0 3.2mm 0;
    font-size: 8.2pt;
    line-height: 1.35;
    white-space: pre-wrap;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 8.2pt; }

/* ---- callout boxes (blockquotes) ---- */
blockquote {
    margin: 0 0 3.4mm 0;
    padding: 2.6mm 3.4mm;
    background: #f4f7fa;
    border-left: 3.5px solid #2c6fa6;
    font-size: 9.3pt;
    page-break-inside: avoid;
}
blockquote p { margin: 0 0 2mm 0; }
blockquote p:last-child { margin-bottom: 0; }
blockquote.warn {
    background: #fdf6ec;
    border-left-color: #c98b16;
}
blockquote.quote {
    background: #f6f6f4;
    border-left-color: #b9b9b4;
    font-style: italic;
}

/* ---- tables ---- */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0 0 3.6mm 0;
    font-size: 8.8pt;
    page-break-inside: avoid;
}
th {
    background: #edeff2;
    text-align: left;
    font-family: "DejaVu Sans", sans-serif;
    font-size: 8.4pt;
    border-bottom: 1.4px solid #8f9aa5;
    padding: 1.5mm 2mm;
}
td {
    border-bottom: 0.5px solid #dcdcdc;
    padding: 1.5mm 2mm;
    vertical-align: top;
}

hr { border: none; border-top: 0.6px solid #d8d8d8; margin: 5mm 0; }

/* ---- table of contents ---- */
#toc { page-break-after: always; }
#toc h1 { page-break-before: auto; }
#toc ul { list-style: none; padding-left: 0; }
#toc > ul > li { margin-top: 2.6mm; font-weight: bold; font-size: 10.5pt; }
#toc ul ul { padding-left: 6mm; margin-top: 1mm; }
#toc ul ul li { font-weight: normal; font-size: 9.3pt; color: #333; }
#toc a { text-decoration: none; color: #16324f; }
#toc a::after {
    content: " ····· " target-counter(attr(href), page);
    color: #999;
    font-size: 8.5pt;
}
"""


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return re.sub(r"\s+", "-", text.strip())[:70]


def classify_blockquotes(html: str) -> str:
    """Tag blockquotes as warning / quote callouts based on their lead-in text."""

    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        head = re.sub(r"<[^>]+>", "", body)[:40].lower()
        if "watch out" in head:
            cls = "warn"
        elif "intuition" in head:
            cls = ""
        else:
            cls = "quote"
        attr = f' class="{cls}"' if cls else ""
        return f"<blockquote{attr}>{body}</blockquote>"

    return re.sub(r"<blockquote>(.*?)</blockquote>", repl, html, flags=re.S)


def add_heading_ids(html: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Give every h1/h2 an id and collect them for the table of contents."""
    entries: list[tuple[int, str, str]] = []
    seen: dict[str, int] = {}

    def repl(match: re.Match[str]) -> str:
        level = int(match.group(1))
        inner = match.group(2)
        base = slugify(inner) or f"h{level}"
        seen[base] = seen.get(base, 0) + 1
        anchor = base if seen[base] == 1 else f"{base}-{seen[base]}"
        entries.append((level, anchor, inner))
        return f'<h{level} id="{anchor}">{inner}</h{level}>'

    html = re.sub(r"<h([12])>(.*?)</h\1>", repl, html, flags=re.S)
    return html, entries


def build_toc(entries: list[tuple[int, str, str]]) -> str:
    parts = ['<div id="toc"><h1>Contents</h1><ul>']
    open_sub = False
    for level, anchor, text in entries:
        label = re.sub(r"<[^>]+>", "", text)
        if level == 1:
            if open_sub:
                parts.append("</ul></li>")
                open_sub = False
            parts.append(f'<li><a href="#{anchor}">{label}</a><ul>')
            open_sub = True
        else:
            if not open_sub:
                parts.append("<li><ul>")
                open_sub = True
            parts.append(f'<li><a href="#{anchor}">{label}</a></li>')
    if open_sub:
        parts.append("</ul></li>")
    parts.append("</ul></div>")
    return "".join(parts)


def main() -> int:
    sources = sorted(NOTES_DIR.glob("*.md"))
    if not sources:
        print(f"no markdown found in {NOTES_DIR}", file=sys.stderr)
        return 1

    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])

    front = md.convert(sources[0].read_text(encoding="utf-8"))
    # The first document becomes the title page plus a "how to read" section.
    split = front.find("<h2")
    title_html, front_rest = (front, "") if split == -1 else (front[:split], front[split:])

    bodies = []
    for path in sources[1:]:
        md.reset()
        bodies.append(md.convert(path.read_text(encoding="utf-8")))

    body_html = front_rest + "\n".join(bodies)
    body_html, entries = add_heading_ids(body_html)
    body_html = classify_blockquotes(body_html)

    # The front-matter sections belong under a heading of their own in the TOC.
    entries.insert(0, (1, "how-to-read", "About these notes"))
    body_html = f'<h1 id="how-to-read">About these notes</h1>{body_html}'

    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>CME 295 — Transformers &amp; Large Language Models: Lecture Notes</title>
<style>{CSS}</style></head>
<body>
<div class="titlepage">{title_html}</div>
{build_toc(entries)}
{body_html}
</body></html>"""

    HTML(string=document, base_url=str(ROOT)).write_pdf(OUTPUT)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
