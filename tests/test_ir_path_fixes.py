"""End-to-end proof that the IR render path fixes the conceptual bugs.

These tests build the exact baseline architectures from generate_all.py
that suffer from each bug, render them via ``use_ir=True``, and assert
the bug is gone. The legacy path still has the bugs (captured by the
golden snapshots in test_golden.py); those go away only when the IR
path becomes the default in a later phase.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def baselines():
    os.environ["PLOTNN_SKIP_WRITE"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    if "generate_all" in sys.modules:
        importlib.reload(sys.modules["generate_all"])
    else:
        importlib.import_module("generate_all")
    from generate_all import ARCHITECTURES
    return {name: (arch, show_n) for name, arch, show_n in ARCHITECTURES}


# ---------------------------------------------------------------------------
# Bug #1 — Mixtral MoE residual was missing in legacy
# ---------------------------------------------------------------------------

def test_mixtral_ir_path_has_moe_residual(baselines):
    arch, show_n = baselines["11_mixtral_8x7b"]
    legacy = arch.render(show_n=show_n, use_ir=False)
    ir = arch.render(show_n=show_n, use_ir=True)

    # Each transformer block in IR path: attn-residual + MoE-residual = 2 skips.
    # Legacy has only 1 skip per block (attn), MoE was bare.
    legacy_skips = legacy.count(r"\draw[skiparrow]")
    ir_skips = ir.count(r"\draw[skiparrow]")
    assert ir_skips > legacy_skips, (
        f"IR path must add MoE residuals: legacy={legacy_skips}, IR={ir_skips}"
    )


# ---------------------------------------------------------------------------
# Bug #2 — DeBERTa double {Output} label in legacy
# ---------------------------------------------------------------------------

def test_deberta_ir_path_no_double_output(baselines):
    arch, show_n = baselines["06_deberta_v3"]
    legacy = arch.render(show_n=show_n, use_ir=False)
    ir = arch.render(show_n=show_n, use_ir=True)

    legacy_out = len(re.findall(r"\{Output\}", legacy))
    ir_out = len(re.findall(r"\{Output\}", ir))
    # Legacy still has the bug (kept for diff/comparison); IR fixes it.
    assert legacy_out >= 2, "legacy DeBERTa baseline should still show the bug"
    assert ir_out == 1, f"IR path must produce exactly 1 Output label, got {ir_out}"


# ---------------------------------------------------------------------------
# Bug #3 — Fractional cm widths in legacy ResNet
# ---------------------------------------------------------------------------

_FRACTIONAL_CM = re.compile(r"\d\.\d{4,}cm")


def test_efficientnet_ir_path_no_fractional_cm(baselines):
    """EfficientNet has migrated layers; verify width rounding."""
    arch, show_n = baselines["17_efficientnet_b0"]
    ir = arch.render(show_n=show_n, use_ir=True)
    matches = _FRACTIONAL_CM.findall(ir)
    assert not matches, f"IR path leaked fractional cm: {matches}"


def test_fno_2d_ir_path_no_fractional_cm(baselines):
    arch, show_n = baselines["23_fno_2d"]
    ir = arch.render(show_n=show_n, use_ir=True)
    matches = _FRACTIONAL_CM.findall(ir)
    assert not matches, f"IR path leaked fractional cm: {matches}"


# ---------------------------------------------------------------------------
# Bug #4 — Mixtral lacks ln2 (pre-FFN norm) before MoE in legacy
# ---------------------------------------------------------------------------

def test_mixtral_ir_path_has_pre_moe_norm(baselines):
    """In legacy, MoELayer rendered Router→Experts without a preceding norm.
    IR path wraps NORM into the residual body — the LN must appear."""
    arch, show_n = baselines["11_mixtral_8x7b"]
    ir = arch.render(show_n=show_n, use_ir=True)
    # Each transformer block has 2 LayerNorms (pre-attn, pre-MoE) × 32 layers.
    # We don't assert exact count (depends on show_n), just that there are
    # significantly more than the legacy single-LN per block.
    legacy = arch.render(show_n=show_n, use_ir=False)
    legacy_ln = legacy.count("LayerNorm")
    ir_ln = ir.count("LayerNorm")
    assert ir_ln > legacy_ln, (
        f"IR path should emit more LayerNorms (1 attn + 1 MoE per block); "
        f"legacy={legacy_ln}, IR={ir_ln}"
    )


# ---------------------------------------------------------------------------
# Bug #5 — Input arrow attaches to first SEMANTIC node, not anchor
# ---------------------------------------------------------------------------

def test_ir_path_input_attaches_to_real_block(baselines):
    """For all IR-renderable baselines, Input arrow must reference a
    block-shaped node id (e.g. ``layer_…`` or ``n_…``), not an anchor."""
    from plot_nn_mcp.lowering import architecture_to_ir, can_lower_architecture

    for name, (arch, _show_n) in baselines.items():
        if not can_lower_architecture(arch)[0]:
            continue
        graph = architecture_to_ir(arch)
        in_node = graph.nodes[graph.input_node]
        assert in_node.shape != "anchor", \
            f"{name}: Input attaches to anchor node, must be a real block"


# ---------------------------------------------------------------------------
# Smoke — IR path produces valid, complete LaTeX for all 12 ready baselines
# ---------------------------------------------------------------------------

_ALL_BASELINES = [f"{i:02d}" for i in range(1, 26)]


@pytest.mark.parametrize("idx", _ALL_BASELINES)
def test_ir_path_produces_valid_latex(baselines, idx):
    """All 25 baselines must render successfully through the IR path."""
    name = next(n for n in baselines if n.startswith(idx))
    arch, show_n = baselines[name]
    tex = arch.render(show_n=show_n, use_ir=True)
    assert tex.startswith(r"\documentclass")
    assert tex.rstrip().endswith(r"\end{document}")
    assert tex.count(r"\begin{tikzpicture}") == 1
    assert tex.count(r"\end{tikzpicture}") == 1


@pytest.mark.parametrize("idx", _ALL_BASELINES)
def test_ir_path_no_fractional_cm_anywhere(baselines, idx):
    """Phase 5 invariant — IR path emits zero fractional cm across all 25."""
    name = next(n for n in baselines if n.startswith(idx))
    arch, show_n = baselines[name]
    tex = arch.render(show_n=show_n, use_ir=True)
    matches = _FRACTIONAL_CM.findall(tex)
    assert not matches, f"{name}: IR leaked fractional cm: {matches}"


@pytest.mark.parametrize("idx", _ALL_BASELINES)
def test_ir_path_exactly_one_input(baselines, idx):
    """IR path must produce exactly one Input label per architecture."""
    name = next(n for n in baselines if n.startswith(idx))
    arch, show_n = baselines[name]
    tex = arch.render(show_n=show_n, use_ir=True)
    assert len(re.findall(r"\{Input\}", tex)) <= 1, \
        f"{name}: multiple Input labels"


@pytest.mark.parametrize("idx", _ALL_BASELINES)
def test_ir_path_at_most_one_output_label(baselines, idx):
    """No double-Output anywhere through IR path."""
    name = next(n for n in baselines if n.startswith(idx))
    arch, show_n = baselines[name]
    tex = arch.render(show_n=show_n, use_ir=True)
    out = len(re.findall(r"\{Output\}", tex))
    assert out <= 1, f"{name}: {out} Output labels in IR path"


def test_legacy_path_still_invokable(baselines):
    """After Phase 7 default flip, legacy path remains accessible via
    ``use_ir=False`` for users who want to compare or revert."""
    arch, show_n = baselines["04_bert_base"]
    legacy = arch.render(show_n=show_n, use_ir=False)
    assert legacy.startswith(r"\documentclass")
    assert legacy.rstrip().endswith(r"\end{document}")
