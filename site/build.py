r"""Results site generator — the experiments, published, no server.

One command turns an experiment folder into a static HTML page:

    python site/build.py logistic-regression      # one page
    python site/build.py                          # everything registered

For each registered experiment the builder
  1. RUNS the script headless (Agg), exactly as F5 would, capturing
     stdout — the script is not modified, and a page can only be built
     if every assert in it passes (an exception fails the build for
     that page, loudly);
  2. collects every matplotlib Figure and FuncAnimation the script
     left behind: animations become in-page players (mp4 if ffmpeg is
     installed, else matplotlib's self-contained jshtml player with
     play/pause/scrub), figures become PNGs;
  3. compiles the companion .tex (pdflatex, two passes) and embeds the
     PDF;
  4. writes one HTML page with tabs:
        Report     — verification checklist (the ": OK" lines), meta
        Animations — the players
        Plots      — the static figures
        LaTeX      — the embedded PDF
        Terminal   — the captured stdout, as run
        Source     — the script itself
     plus an index page linking all built experiments.

Output goes to site/out/ (gitignored — the site is an artifact,
rebuilt by one command, never hand-edited). Open site/out/index.html
in a browser, or `python -m http.server -d site/out`.
"""

import contextlib
import html
import io
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.animation import (FuncAnimation,     # noqa: E402
                                  FFMpegWriter)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"
PY = sys.executable

# registry: slug -> (script path, tex path or None, title, blurb)
EXPERIMENTS = {
    "logistic-regression": dict(
        script="logistic-regression/logistic_animation.py",
        tex="logistic-regression/logistic-regression.tex",
        title="Logistic Regression",
        blurb="A Bernoulli whose parameter is a function of X: "
              "Newton/IRLS vs GD vs SGD vs autograd, the Bayes floor, "
              "and SGD filmed one sample at a time."),
}

ANIM_DPI = 65          # animation frames re-render at this dpi (size!)
ANIM_FPS = 12


def run_experiment(folder: Path, script: Path):
    """Exec the script as __main__ under Agg, capturing stdout and
    everything it draws. Returns (ok, stdout, figures, animations,
    error_traceback)."""
    plt.close("all")
    buf = io.StringIO()
    ns = {"__name__": "__main__", "__file__": str(script)}
    old_cwd, old_path = Path.cwd(), list(sys.path)
    ok, tb = True, ""
    try:
        import os
        os.chdir(folder)
        sys.path.insert(0, str(folder))
        code = compile(script.read_text(encoding="utf-8"),
                       str(script), "exec")
        with contextlib.redirect_stdout(buf):
            exec(code, ns)
    except Exception:
        ok, tb = False, traceback.format_exc()
    finally:
        import os
        os.chdir(old_cwd)
        sys.path[:] = old_path

    anims = [v for v in ns.values() if isinstance(v, FuncAnimation)]
    anim_figs = {id(a._fig) for a in anims}
    figs = [plt.figure(n) for n in plt.get_fignums()
            if id(plt.figure(n)) not in anim_figs]
    return ok, buf.getvalue(), figs, anims, tb


def export_animation(ani, path_base: Path) -> str:
    """mp4 when ffmpeg exists (small), else jshtml (self-contained).
    Returns an HTML snippet embedding the player."""
    ani._fig.set_dpi(ANIM_DPI)
    if FFMpegWriter.isAvailable():
        mp4 = path_base.with_suffix(".mp4")
        ani.save(mp4, writer=FFMpegWriter(fps=ANIM_FPS, bitrate=1800))
        return (f'<video controls loop style="max-width:100%">'
                f'<source src="{mp4.name}" type="video/mp4"></video>')
    return f'<div class="jsanim">{ani.to_jshtml(fps=ANIM_FPS)}</div>'


def compile_tex(tex: Path) -> Path | None:
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             tex.name], cwd=tex.parent, capture_output=True)
        if r.returncode != 0:
            return None
    return tex.with_suffix(".pdf")


STYLE = """
body{font-family:Segoe UI,system-ui,sans-serif;margin:0;
     background:#f6f5f2;color:#1c1c1c}
header{background:#101418;color:#eee;padding:1.1rem 2rem}
header h1{margin:0;font-size:1.35rem} header a{color:#7fb3e8}
.meta{color:#889;font-size:.8rem;margin-top:.3rem}
nav{display:flex;gap:.4rem;background:#101418;padding:0 2rem .8rem}
nav button{background:#232a33;color:#ccc;border:0;border-radius:6px 6px 0 0;
     padding:.5rem 1.1rem;cursor:pointer;font-size:.9rem}
nav button.on{background:#f6f5f2;color:#111;font-weight:600}
main{padding:1.4rem 2rem;max-width:1200px;margin:auto}
.tab{display:none}.tab.on{display:block}
.check{color:#1a7a3d;font-family:Consolas,monospace;font-size:.9rem;
     margin:.25rem 0}
.badge{display:inline-block;padding:.25rem .7rem;border-radius:99px;
     font-weight:600;font-size:.85rem}
.badge.ok{background:#d9f2e0;color:#176633}
.badge.fail{background:#fbdcd6;color:#8f2011}
pre.term{background:#101418;color:#d7e0d9;padding:1rem;border-radius:8px;
     overflow-x:auto;font-size:.82rem;line-height:1.45}
pre.src{background:#fff;border:1px solid #ddd;padding:1rem;
     border-radius:8px;overflow-x:auto;font-size:.8rem;line-height:1.4}
img.plot{max-width:100%;border:1px solid #ddd;border-radius:8px;
     margin:.6rem 0;background:#fff}
embed.pdf{width:100%;height:82vh;border:1px solid #ddd;border-radius:8px}
.card{background:#fff;border:1px solid #ddd;border-radius:10px;
     padding:1rem 1.3rem;margin:.7rem 0}
a.expcard{display:block;text-decoration:none;color:inherit}
a.expcard:hover .card{border-color:#2a78d6}
.jsanim{background:#fff;border:1px solid #ddd;border-radius:8px;
     padding:.5rem;overflow-x:auto}
"""

TABS_JS = """
function show(t){
 document.querySelectorAll('.tab').forEach(e=>e.classList.remove('on'));
 document.querySelectorAll('nav button').forEach(e=>e.classList.remove('on'));
 document.getElementById('tab-'+t).classList.add('on');
 document.getElementById('btn-'+t).classList.add('on');}
"""


def page_html(slug, exp, ok, stdout, anim_snips, fig_names, pdf_name,
              source, elapsed, commit):
    checks = [ln.strip() for ln in stdout.splitlines()
              if ln.strip().endswith("OK") or ": OK" in ln]
    badge = ('<span class="badge ok">all asserts passed</span>' if ok
             else '<span class="badge fail">FAILED — see Terminal</span>')
    check_html = "".join(f'<div class="check">&#10003; {html.escape(c)}'
                         f'</div>' for c in checks) or "<p>—</p>"
    anims = "".join(f'<div class="card">{s}</div>'
                    for s in anim_snips) or "<p>no animations.</p>"
    plots = "".join(f'<img class="plot" src="{n}">'
                    for n in fig_names) or "<p>no static figures.</p>"
    pdf = (f'<embed class="pdf" src="{pdf_name}" '
           f'type="application/pdf">' if pdf_name else
           "<p>no companion document.</p>")
    tabs = ["report", "animations", "plots", "latex", "terminal",
            "source"]
    nav = "".join(f'<button id="btn-{t}" onclick="show(\'{t}\')" '
                  f'{"class=on" if t == "report" else ""}>'
                  f'{t.capitalize()}</button>' for t in tabs)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{exp['title']} — ML-RESEARCH</title>
<style>{STYLE}</style><script>{TABS_JS}</script></head><body>
<header><h1><a href="../index.html">ML-RESEARCH</a> / {exp['title']}</h1>
<div class="meta">{html.escape(exp['blurb'])}</div>
<div class="meta">built {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC
 &middot; commit {commit} &middot; run {elapsed:.0f}s &middot; {badge}
</div></header>
<nav>{nav}</nav><main>
<div class="tab on" id="tab-report"><h2>Verification</h2>
<div class="card">{check_html}</div>
<p>Every line above is an assert-backed claim printed by the script;
the page only builds if the script runs to completion.</p></div>
<div class="tab" id="tab-animations">{anims}</div>
<div class="tab" id="tab-plots">{plots}</div>
<div class="tab" id="tab-latex">{pdf}</div>
<div class="tab" id="tab-terminal">
<pre class="term">{html.escape(stdout)}</pre></div>
<div class="tab" id="tab-source">
<pre class="src">{html.escape(source)}</pre></div>
</main></body></html>"""


def index_html(built, commit):
    cards = "".join(
        f'<a class="expcard" href="{slug}/index.html"><div class="card">'
        f'<h3>{EXPERIMENTS[slug]["title"]}</h3>'
        f'<p>{html.escape(EXPERIMENTS[slug]["blurb"])}</p>'
        f'</div></a>' for slug in built)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>ML-RESEARCH — results</title><style>{STYLE}</style></head><body>
<header><h1>ML-RESEARCH — results</h1>
<div class="meta">generated static site &middot; commit {commit} &middot;
 rebuilt by <code>python site/build.py</code></div></header>
<main>{cards}</main></body></html>"""


def build(slug: str, commit: str) -> bool:
    exp = EXPERIMENTS[slug]
    script = ROOT / exp["script"]
    folder = script.parent
    out = OUT / slug
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{slug}] running {script.name} ...", flush=True)

    t0 = time.perf_counter()
    ok, stdout, figs, anims, tb = run_experiment(folder, script)
    if not ok:
        stdout += "\n\n=== BUILD FAILED ===\n" + tb
        print(f"[{slug}] FAILED:\n{tb}", flush=True)

    fig_names = []
    for i, f in enumerate(figs):
        name = f"fig{i}.png"
        f.savefig(out / name, dpi=110, bbox_inches="tight")
        fig_names.append(name)
    anim_snips = []
    for i, a in enumerate(anims):
        print(f"[{slug}] exporting animation {i} "
              f"({'mp4' if FFMpegWriter.isAvailable() else 'jshtml'})"
              " ...", flush=True)
        anim_snips.append(export_animation(a, out / f"anim{i}"))
    plt.close("all")

    pdf_name = None
    if exp.get("tex"):
        pdf = compile_tex(ROOT / exp["tex"])
        if pdf:
            shutil.copy(pdf, out / pdf.name)
            pdf_name = pdf.name

    elapsed = time.perf_counter() - t0
    (out / "index.html").write_text(
        page_html(slug, exp, ok, stdout, anim_snips, fig_names,
                  pdf_name, script.read_text(encoding="utf-8"),
                  elapsed, commit),
        encoding="utf-8")
    print(f"[{slug}] -> {out / 'index.html'}  ({elapsed:.0f}s)",
          flush=True)
    return ok


if __name__ == "__main__":
    targets = sys.argv[1:] or list(EXPERIMENTS)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=ROOT, capture_output=True,
                            text=True).stdout.strip() or "?"
    results = {t: build(t, commit) for t in targets}
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(
        index_html([t for t in EXPERIMENTS if (OUT / t).exists()],
                   commit), encoding="utf-8")
    print(f"\nindex -> {OUT / 'index.html'}")
    if not all(results.values()):
        sys.exit(1)
