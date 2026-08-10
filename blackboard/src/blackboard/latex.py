"""LaTeX compilation for the theory board: two pdflatex passes, no perl."""

import subprocess
from pathlib import Path


def compile_tex(tex_path: Path) -> tuple[bool, str]:
    """Compile tex_path in its own directory; return (ok, log_tail)."""
    tex_path = tex_path.resolve()
    log = ""
    for _ in (1, 2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            timeout=180,
        )
        log = proc.stdout[-4000:]
        if proc.returncode != 0:
            errors = [
                line for line in proc.stdout.splitlines() if line.startswith("!")
            ]
            return False, "\n".join(errors) or log
    return True, log
