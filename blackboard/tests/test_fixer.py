from blackboard.fixer import fix_code


def test_formats_spacing():
    result = fix_code("x=1+2\ny  =  x*3")
    assert result["code"] == "x = 1 + 2\ny = x * 3"


def test_applies_safe_fixes():
    # unused import is a safe autofix (F401)
    result = fix_code("import os\nx = 1")
    assert "import os" not in result["code"]


def test_reports_remaining_diagnostics():
    # undefined name in a *definition* position ruff can't fix: E741 ambiguous name
    result = fix_code("l = 1")
    codes = [d["code"] for d in result["diagnostics"]]
    assert "E741" in codes
    assert all({"line", "code", "message"} <= set(d) for d in result["diagnostics"])


def test_kernel_names_not_flagged():
    # np comes from the kernel namespace — F821 must be suppressed
    result = fix_code("z = np.mean([1, 2, 3])")
    codes = [d["code"] for d in result["diagnostics"]]
    assert "F821" not in codes


def test_no_trailing_newline_added():
    result = fix_code("x = 1")
    assert result["code"] == "x = 1"
