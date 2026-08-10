"""Blackboard's execution engine: one persistent namespace, REPL semantics.

Code cells execute in a shared namespace that survives across runs (like a
notebook kernel). The last statement, if it is an expression, is evaluated
and its repr returned — matplotlib figures created during execution are
captured as base64 PNGs.
"""

import ast
import base64
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field

_STARTUP = """\
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
"""


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    value: str | None = None  # repr of the trailing expression, if any
    error: str | None = None
    images: list[str] = field(default_factory=list)  # base64 PNGs


class Kernel:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.namespace: dict = {"__name__": "__blackboard__"}
        startup = self.run(_STARTUP)
        if not startup.ok:  # library import problems should be loud
            raise RuntimeError(f"kernel startup failed: {startup.error}")

    def run(self, code: str) -> ExecResult:
        buf = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            return ExecResult(ok=False, error=traceback.format_exc(limit=0))

        trailing_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing_expr = ast.Expression(tree.body.pop(-1).value)

        try:
            with redirect_stdout(buf), redirect_stderr(buf):
                if tree.body:
                    exec(compile(tree, "<cell>", "exec"), self.namespace)
                value = None
                if trailing_expr is not None:
                    result = eval(compile(trailing_expr, "<cell>", "eval"), self.namespace)
                    if result is not None:
                        value = repr(result)
        except Exception:
            return ExecResult(
                ok=False,
                stdout=buf.getvalue(),
                error=traceback.format_exc(),
                images=self._grab_figures(),
            )
        return ExecResult(
            ok=True, stdout=buf.getvalue(), value=value, images=self._grab_figures()
        )

    def _grab_figures(self) -> list[str]:
        plt = self.namespace.get("plt")
        if plt is None:
            return []
        images = []
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
            images.append(base64.b64encode(png.getvalue()).decode("ascii"))
        plt.close("all")
        return images
