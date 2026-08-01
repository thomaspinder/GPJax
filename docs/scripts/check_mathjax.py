"""Catch MathJax rendering failures in the built Sphinx site.

Sphinx cannot see this class of defect. It writes LaTeX into the HTML verbatim
and MathJax parses it in the browser, so `sphinx-build -W` can report zero
warnings while a page displays "Erroneous nesting of equation structures" to
every reader. That shipped to a docs preview once already.

Three distinct failure modes reach the reader, and only the first announces
itself:

1. `mjx-merror` -- MathJax typeset the block but flagged it, e.g. "Erroneous
   nesting of equation structures".
2. Untypeset blocks -- an unbalanced brace makes MathJax 4 leave the node
   *completely alone*: no `mjx-container`, no `mjx-merror`, just the raw
   `\\[...\\]` source on the page.
3. Undefined macros -- the `noundefined` package (on by default) renders an
   unknown control sequence such as `\\bm` as red literal text rather than an
   error, so a missing macro ships silently.

The only reliable detector is a real browser loading the real page. Extracting
the LaTeX out of `div.math` / `span.math` and typesetting it in a scratch
element DOES NOT WORK -- it reports OK for markup that visibly fails in situ,
because the failure depends on the page's own MathJax configuration and on how
Sphinx wrapped the block. Do not "optimise" this script into that shortcut.

What this does NOT cover, so a clean run is not read as more than it is. The
trust boundary is *authored maths nodes* -- anything Sphinx marked up as maths.
Outside it:

- LaTeX embedded in notebook HTML *outputs* (a printed string, a matplotlib
  label) is not a maths node, so it is never inspected.
- A delimiter typo that degrades before Sphinx sees it -- `$x$$` closing wrong,
  say -- produces ordinary prose rather than a maths node, so there is nothing
  here to flag. It renders as visible dollar signs instead.

Usage::

    python docs/scripts/check_mathjax.py [SITE_DIR]

Exits 0 when every page with maths typesets cleanly, 1 otherwise.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
import threading
import time

from playwright.sync_api import (
    Error as PlaywrightError,
    sync_playwright,
)

# Pages that contain no maths cannot produce a MathJax failure, and loading them
# costs a browser navigation each, so this is the cheap pre-filter.
#
# It must match `math` anywhere in the class list, not just at its start: MyST's
# amsmath extension emits `class="amsmath math notranslate nohighlight"`, so the
# old `'class="math'` prefix test skipped any page whose only maths was a bare
# `\begin{align*}` block -- exactly what the MyST migration produced.
MATH_MARKER = re.compile(r'class="[^"]*\bmath\b')

# Returns "ok", "timeout" or "no-mathjax". `MathJax.startup.promise` is the
# documented signal that typesetting has finished. The short poll in front of it
# is only there because the MathJax script tag is deferred: by the time the load
# event fires it has run, but `MathJax.startup` is populated a tick later.
# Anything that has not appeared within a few seconds is never going to.
WAIT_FOR_MATHJAX = """
async (timeoutMs) => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const start = Date.now();
  while (!(window.MathJax && window.MathJax.startup
           && window.MathJax.startup.promise)) {
    if (Date.now() - start > 5000) return 'no-mathjax';
    await sleep(50);
  }
  return Promise.race([
    window.MathJax.startup.promise.then(() => 'ok'),
    sleep(timeoutMs).then(() => 'timeout'),
  ]);
}
"""

# One entry per rendering failure in the DOM, covering all three modes listed in
# the module docstring. The original TeX is recovered from MathJax's own math
# list (each MathItem keeps `.math`, the source string, and `.typesetRoot`, the
# container it produced), which is far more useful than the rendered error text.
# Where that lookup fails we fall back to whatever text surrounds the failure so
# a human can still find it on the page.
#
# Note this is a *raw* string: the regex escapes below are JavaScript's, and
# Python must not eat them.
COLLECT_PROBLEMS = r"""
() => {
  const containerToTex = new Map();
  try {
    for (const item of window.MathJax.startup.document.math) {
      if (item.typesetRoot) containerToTex.set(item.typesetRoot, item.math);
    }
  } catch (e) { /* older/absent math list: fall back to nearby text */ }

  const squash = (s) => (s || '').replace(/\s+/g, ' ').trim();

  const heading = (node) => {
    let el = node;
    while (el && el !== document.body) {
      let sib = el.previousElementSibling;
      while (sib) {
        const h = sib.matches('h1,h2,h3,h4,h5,h6')
          ? sib
          : sib.querySelector('h1,h2,h3,h4,h5,h6');
        if (h) return squash(h.textContent).replace(/\u00b6$/, '');
        sib = sib.previousElementSibling;
      }
      el = el.parentElement;
    }
    const title = document.querySelector('h1');
    return title ? squash(title.textContent).replace(/\u00b6$/, '') : '';
  };

  const problems = [];

  // 1. Blocks MathJax typeset but flagged.
  for (const err of document.querySelectorAll('mjx-merror')) {
    const container = err.closest('mjx-container');
    const tex = container ? containerToTex.get(container) : undefined;
    const near = squash((container || err).parentElement?.textContent);
    problems.push({
      kind: 'merror',
      message: err.getAttribute('data-mjx-error') || squash(err.textContent)
               || '(no message)',
      tex: squash(tex),
      near: tex ? '' : near.slice(0, 200),
      section: heading(container || err),
    });
  }

  // 2. Blocks MathJax never typeset at all. A TeX error such as an unbalanced
  //    brace makes MathJax 4 abandon the node instead of erroring on it: no
  //    mjx-container is produced and the raw \[...\] is left on the page for
  //    the reader to see. The TeX test keeps genuinely empty or already-plain
  //    math nodes from being reported.
  const LOOKS_LIKE_TEX = /\\\[|\\\(|\\begin\{|\\[a-zA-Z]/;
  for (const el of document.querySelectorAll('div.math, span.math')) {
    if (el.querySelector('mjx-container')) continue;
    const raw = squash(el.textContent);
    if (!LOOKS_LIKE_TEX.test(raw)) continue;
    problems.push({
      kind: 'untypeset',
      message: 'not typeset (raw LaTeX left on page)',
      tex: '',
      near: raw.slice(0, 300),
      section: heading(el),
    });
  }

  // 3. Undefined control sequences. MathJax's `noundefined` package is on by
  //    default and renders an unknown macro as red literal text rather than an
  //    error, which is how \bm, \mathbbm and missing custom macros ship
  //    unnoticed. The renderer emits them as <mjx-mtext> carrying the macro
  //    name in data-latex and an inline red colour. Requiring the text to be a
  //    bare control sequence keeps deliberately red \textcolor{red}{...} prose
  //    out of the report.
  const CONTROL_SEQUENCE = /^\\[a-zA-Z]+\*?$/;
  for (const el of document.querySelectorAll('mjx-mtext')) {
    const style = el.getAttribute('style') || '';
    const red = el.getAttribute('mathcolor') === 'red'
                || /(^|;)\s*color:\s*red\b/i.test(style);
    if (!red) continue;
    const name = squash(el.getAttribute('data-latex') || el.textContent);
    if (!CONTROL_SEQUENCE.test(name)) continue;
    const container = el.closest('mjx-container');
    const tex = container ? containerToTex.get(container) : undefined;
    problems.push({
      kind: 'undefined-macro',
      message: 'undefined macro: ' + name,
      tex: squash(tex),
      near: tex ? '' : squash(el.parentElement?.textContent).slice(0, 200),
      section: heading(container || el),
    });
  }

  return problems;
}
"""


@dataclass
class PageResult:
    """Outcome of loading one built page in the browser."""

    path: str
    status: str  # "ok", "timeout", "no-mathjax" or "crashed"
    problems: list[dict]
    detail: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.problems) or self.status != "ok"


def find_math_pages(site: Path) -> tuple[list[Path], int]:
    """Return the HTML files containing maths, and how many were skipped."""
    with_math: list[Path] = []
    skipped = 0
    for html in sorted(site.rglob("*.html")):
        if MATH_MARKER.search(html.read_text(encoding="utf-8", errors="replace")):
            with_math.append(html)
        else:
            skipped += 1
    return with_math, skipped


@contextlib.contextmanager
def serve(site: Path) -> Iterator[str]:
    """Serve `site` on a background thread; yield its base URL.

    The pages must be fetched over http://, not file://: they pull MathJax from
    a CDN and resolve relative assets, and file:// gives them a different origin
    and different CORS behaviour from the deployed site.
    """
    handler = functools.partial(QuietHandler, directory=str(site))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


class QuietHandler(SimpleHTTPRequestHandler):
    """`SimpleHTTPRequestHandler` without the per-request stderr chatter."""

    def log_message(self, format: str, *args: object) -> None:
        pass


def check_pages(
    site: Path, pages: list[Path], base_url: str, timeout_s: float
) -> list[PageResult]:
    """Load every page in a headless browser and collect its MathJax problems."""
    results: list[PageResult] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.set_default_navigation_timeout(timeout_s * 1000)
        # Playwright's own 30s action cap would otherwise abort `evaluate`
        # before the in-page race below can report a clean "timeout".
        page.set_default_timeout((timeout_s + 10) * 1000)
        for index, html in enumerate(pages, start=1):
            rel = html.relative_to(site).as_posix()
            print(f"  [{index}/{len(pages)}] {rel}", flush=True)
            try:
                page.goto(f"{base_url}/{rel}", wait_until="load")
                status = page.evaluate(WAIT_FOR_MATHJAX, timeout_s * 1000)
                problems = page.evaluate(COLLECT_PROBLEMS) if status == "ok" else []
                results.append(PageResult(rel, status, problems))
            except PlaywrightError as exc:
                first_line = str(exc).strip().splitlines()[0]
                results.append(PageResult(rel, "crashed", [], first_line))
        context.close()
        browser.close()
    return results


def report(results: list[PageResult], skipped: int, elapsed: float) -> int:
    """Print a per-page report and return the process exit code."""
    failures = [r for r in results if r.failed]

    print()
    print("=" * 72)
    print("MathJax render check")
    print("=" * 72)
    print(f"pages with maths : {len(results)}")
    print(f"pages skipped    : {skipped} (no 'math' class in the HTML)")
    print(f"elapsed          : {elapsed:.1f}s")

    if not failures:
        print(
            "\nNo MathJax errors, untypeset blocks or undefined macros found. "
            "All maths typeset cleanly."
        )
        return 0

    print(f"\n{len(failures)} page(s) with problems:\n")
    total = 0
    for result in failures:
        print(f"{result.path}")
        if result.status != "ok":
            # Not a rendering error but not a pass either: the page was never
            # actually checked, which must not be reported as clean.
            note = {
                "timeout": "MathJax did not finish typesetting before the timeout",
                "no-mathjax": "MathJax never initialised (CDN blocked? script missing?)",
                "crashed": "the browser failed to load the page",
            }[result.status]
            print(f"  ! {note}")
            if result.detail:
                print(f"    {result.detail}")
        # Identical messages are collapsed, but the count is always shown.
        seen: dict[tuple[str, str], dict] = {}
        counts: dict[tuple[str, str], int] = {}
        for problem in result.problems:
            key = (problem["message"], problem["tex"] or problem["near"])
            seen.setdefault(key, problem)
            counts[key] = counts.get(key, 0) + 1
            total += 1
        for key, problem in seen.items():
            print(f"  x{counts[key]} {problem['message']}")
            if problem["section"]:
                print(f"      {'section:':9}{problem['section']}")
            if problem["tex"]:
                print(f"      {'tex:':9}{problem['tex'][:300]}")
            elif problem["near"]:
                # For an untypeset block the surrounding text *is* the raw
                # LaTeX, which is what the author needs in order to find it.
                label = "raw:" if problem["kind"] == "untypeset" else "near:"
                print(f"      {label:9}{problem['near']}")
        print()

    print(f"{total} problem(s) across {len(failures)} page(s).")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "site",
        nargs="?",
        default="docs/_build/html",
        type=Path,
        help="built site directory (default: docs/_build/html)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait for MathJax on each page (default: 30)",
    )
    args = parser.parse_args()

    site = args.site.resolve()
    if not site.is_dir():
        print(f"error: no such directory: {site}", file=sys.stderr)
        return 2

    pages, skipped = find_math_pages(site)
    if not pages:
        print(f"No pages contain maths under {site} ({skipped} skipped).")
        return 0

    print(f"Checking {len(pages)} page(s) with maths ({skipped} skipped)...")
    start = time.monotonic()
    with serve(site) as base_url:
        results = check_pages(site, pages, base_url, args.timeout)
    return report(results, skipped, time.monotonic() - start)


if __name__ == "__main__":
    raise SystemExit(main())
