# -*- coding: utf-8 -*-
"""Render every page in headless Chrome and measure what actually breaks.

check-figures.py estimates text width from character counts. That estimate is
wrong for Cyrillic by enough to miss real spills, and it cannot see CSS at all
— it never noticed a component whose markup did not match its stylesheet.
This one asks the browser.

    python3 tools/check-render.py                 # all pages, 1400px
    python3 tools/check-render.py --widths 1920,900,420
    python3 tools/check-render.py language/en/09-prompting-rag.html
"""
import argparse
import glob
import html
import io
import json
import os
import re
import subprocess
import sys
import tempfile

CHROME = os.environ.get(
    "CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

MEASURE = r"""
var out = { spills: [], ovf: [] };
document.querySelectorAll('svg').forEach(function (svg) {
  var id = svg.getAttribute('aria-labelledby') || '?';
  var rects = [].slice.call(svg.querySelectorAll('rect')).map(function (r) {
    return { x: +r.getAttribute('x'), y: +r.getAttribute('y'),
             w: +r.getAttribute('width'), h: +r.getAttribute('height') };
  }).filter(function (r) { return !isNaN(r.x); });
  svg.querySelectorAll('text').forEach(function (t) {
    var body = t.textContent.trim();
    if (!body) return;
    var bb; try { bb = t.getBBox(); } catch (e) { return; }
    var cx = bb.x + bb.width / 2;
    var holders = rects.filter(function (r) {
      return bb.y >= r.y - 2 && bb.y + bb.height <= r.y + r.h + 8 &&
             cx >= r.x - 2 && cx <= r.x + r.w + 2;
    });
    if (!holders.length) return;
    var r = holders.reduce(function (a, b) { return a.w * a.h <= b.w * b.h ? a : b; });
    var over = Math.max((r.x + 1) - bb.x, 0) +
               Math.max((bb.x + bb.width) - (r.x + r.w - 1), 0);
    if (over > 0.5) out.spills.push({ id: id, body: body.slice(0, 50),
                                      over: +over.toFixed(1), boxW: r.w,
                                      size: +(t.getAttribute('font-size') || 12) });
  });
});
document.querySelectorAll('.card,.challenge,.definition,.callout,.note,.figure,.breakdown,.takeaways li')
  .forEach(function (el) {
    if (el.scrollWidth > el.clientWidth + 2)
      out.ovf.push((el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 60));
  });
out.pageWidth = document.documentElement.scrollWidth;
out.vw = document.documentElement.clientWidth;
"""

PROBE = ('\n<pre id="RESULT" style="display:none"></pre>\n<script>'
         'window.addEventListener("load",function(){setTimeout(function(){'
         '%s document.getElementById("RESULT").textContent='
         'JSON.stringify(out);},250);});</script>\n' % MEASURE)


def measure(path, width):
    src = io.open(path, encoding="utf-8").read()
    probe = path.replace(".html", "._render_probe.html")
    io.open(probe, "w", encoding="utf-8").write(src.replace("</body>", PROBE + "</body>"))
    try:
        dom = subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--window-size=%d,1200" % width,
             "--virtual-time-budget=4000", "--dump-dom",
             "file://" + os.path.abspath(probe)],
            capture_output=True, text=True, timeout=90).stdout
    finally:
        os.remove(probe)
    m = re.search(r'<pre id="RESULT"[^>]*>(.*?)</pre>', dom, re.S)
    return json.loads(html.unescape(m.group(1))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--widths", default="1400")
    args = ap.parse_args()
    files = args.files or sorted(f for f in glob.glob("language/*/[0-9][0-9]-*.html")
                                 if "_probe" not in f)
    if not os.path.exists(CHROME):
        sys.exit("Chrome not found at %s — set CHROME=..." % CHROME)

    problems = 0
    for width in [int(w) for w in args.widths.split(",")]:
        print("== width %d ==" % width)
        for f in files:
            o = measure(f, width)
            if o is None:
                print("  ?? %s — no result" % f)
                continue
            issues = []
            for s in o["spills"]:
                issues.append("SPILL  %-15s over %4.1f (box %d, size %g)  %s"
                              % (s["id"], s["over"], s["boxW"], s["size"], s["body"]))
            for t in o["ovf"]:
                issues.append("OVERFLOW  %s" % t)
            if o["pageWidth"] > o["vw"]:
                issues.append("PAGE SCROLLS  %d > %d" % (o["pageWidth"], o["vw"]))
            if issues:
                problems += len(issues)
                print("  %s" % f)
                for i in issues:
                    print("     ", i)
        print()
    print("%d render problems" % problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
