"""Python lint/format for board cells, powered by ruff.

``fix_code`` runs ruff's formatter plus its safe autofixes and returns the
cleaned source with any remaining diagnostics. Cells share a kernel
namespace, so undefined-name checks (F821) and import-position checks (E402)
are suppressed — names legitimately come from other cells.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_CELL_IGNORES = "F821,E402"


def _ruff(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def fix_code(code: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cell.py"
        path.write_text(code, encoding="utf-8")

        _ruff("check", "--fix-only", "--ignore", _CELL_IGNORES, str(path))
        _ruff("format", str(path))

        check = _ruff(
            "check", "--ignore", _CELL_IGNORES, "--output-format", "json",
            "--exit-zero", str(path),
        )
        try:
            raw = json.loads(check.stdout or "[]")
        except json.JSONDecodeError:
            raw = []
        diagnostics = [
            {
                "line": d.get("location", {}).get("row"),
                "code": d.get("code"),
                "message": d.get("message"),
            }
            for d in raw
        ]
        fixed = path.read_text(encoding="utf-8")

    # cells are fragments: don't force the trailing newline files get
    if fixed.endswith("\n") and not code.endswith("\n"):
        fixed = fixed[:-1]
    return {"code": fixed, "diagnostics": diagnostics}
