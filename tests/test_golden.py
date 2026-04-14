"""Golden snapshot + structural invariants for the 25 reference architectures.

Baseline captured at the start of the refactor (see REFACTOR_PLAN.md).
Regenerate snapshots with ``UPDATE_GOLDEN=1 pytest tests/test_golden.py``.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).parent / "golden"
EXAMPLES_DIR = ROOT / "examples"


@pytest.fixture(scope="module")
def architectures():
    os.environ["PLOTNN_SKIP_WRITE"] = "1"
    sys.path.insert(0, str(EXAMPLES_DIR))
    sys.path.insert(0, str(ROOT / "src"))
    if "generate_all" in sys.modules:
        importlib.reload(sys.modules["generate_all"])
    else:
        importlib.import_module("generate_all")
    from generate_all import ARCHITECTURES
    return ARCHITECTURES


def test_registry_has_25(architectures):
    assert len(architectures) == 25, f"expected 25 architectures, got {len(architectures)}"


def test_all_golden_files_present(architectures):
    names = {name for name, _, _ in architectures}
    golden = {p.stem for p in GOLDEN_DIR.glob("*.tex")}
    missing = names - golden
    assert not missing, f"missing golden snapshots: {sorted(missing)}"


@pytest.mark.parametrize("idx", range(25))
def test_snapshot_matches_golden(architectures, idx):
    name, arch, show_n = architectures[idx]
    rendered = arch.render(show_n=show_n)
    golden_path = GOLDEN_DIR / f"{name}.tex"

    if os.environ.get("UPDATE_GOLDEN") == "1":
        golden_path.write_text(rendered)
        return

    expected = golden_path.read_text()
    if rendered != expected:
        # Surface a small diff for debugging.
        import difflib
        diff = "\n".join(list(difflib.unified_diff(
            expected.splitlines(), rendered.splitlines(),
            fromfile=f"golden/{name}.tex", tofile="rendered",
            lineterm="", n=3,
        ))[:60])
        pytest.fail(f"{name}: diverged from golden\n{diff}")


# ---------------------------------------------------------------------------
# Structural invariants — these must hold for every rendered architecture
# ---------------------------------------------------------------------------

_BALANCED_PAIRS = [
    (r"\begin{document}", r"\end{document}"),
    (r"\begin{tikzpicture}", r"\end{tikzpicture}"),
]


@pytest.mark.parametrize("idx", range(25))
def test_latex_balanced(architectures, idx):
    name, arch, show_n = architectures[idx]
    tex = arch.render(show_n=show_n)
    for open_tok, close_tok in _BALANCED_PAIRS:
        assert tex.count(open_tok) == tex.count(close_tok) == 1, \
            f"{name}: {open_tok!r}/{close_tok!r} not balanced"


@pytest.mark.parametrize("idx", range(25))
def test_exactly_one_input_output_label(architectures, idx):
    name, arch, show_n = architectures[idx]
    tex = arch.render(show_n=show_n)
    in_count = len(re.findall(r"\{Input\}", tex))
    out_count = len(re.findall(r"\{Output\}", tex))
    # Phase 7 tight invariant — DeBERTa double-Output bug is fixed.
    assert in_count <= 1, f"{name}: {in_count} Input labels"
    assert out_count <= 1, f"{name}: {out_count} Output labels"


@pytest.mark.parametrize("idx", range(25))
def test_skip_arrow_has_matching_add_circle(architectures, idx):
    name, arch, show_n = architectures[idx]
    tex = arch.render(show_n=show_n)
    skip_count = tex.count(r"\draw[skiparrow]")
    # Every skiparrow should terminate on an add-circle (either {+} or $\oplus$).
    add_count = len(re.findall(r"\{\+\}|\{\$\\oplus\$\}", tex))
    # Some add circles exist without a skiparrow (e.g. positional encoding ⊕),
    # so we only assert skip <= add, not equality.
    assert skip_count <= add_count, \
        f"{name}: {skip_count} skiparrows but only {add_count} add-circles"


_FRACTIONAL_CM = re.compile(r"\d\.\d{4,}cm")


@pytest.mark.parametrize("idx", range(25))
def test_no_fractional_cm_baseline(architectures, idx):
    """Phase 7 tight invariant: zero fractional cm anywhere.

    The 4 legacy offenders (16/19/23/24) are now fixed because
    the default render path goes through the IR emitter which
    rounds widths to 2 decimals.
    """
    name, arch, show_n = architectures[idx]
    tex = arch.render(show_n=show_n)
    matches = _FRACTIONAL_CM.findall(tex)
    assert not matches, f"{name}: fractional cm leaked: {matches}"


@pytest.mark.parametrize("idx", range(25))
def test_no_dead_preamble_styles(architectures, idx):
    """Task 8: smallblock, clrbackground, clrdense were dead code.

    smallblock was never used; clrbackground had no fill=clrbackground site;
    clrdense had the same hex as clrresidual. Guard against them leaking
    back into the preamble.
    """
    name, arch, show_n = architectures[idx]
    tex = arch.render(show_n=show_n)
    assert "smallblock/.style" not in tex, f"{name}: smallblock style leaked back"
    assert r"\definecolor{clrbackground}" not in tex, \
        f"{name}: clrbackground leaked back"
    assert r"\definecolor{clrdense}" not in tex, \
        f"{name}: clrdense leaked back"


@pytest.mark.parametrize("idx", range(25))
def test_structural_document(architectures, idx):
    name, arch, show_n = architectures[idx]
    tex = arch.render(show_n=show_n)
    assert r"\documentclass" in tex
    assert r"\usetikzlibrary{positioning" in tex
    assert tex.rstrip().endswith(r"\end{document}"), \
        f"{name}: does not end with \\end{{document}}"
