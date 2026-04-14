"""Tests for the IR → TikZ emitter (Phase 4 foundation)."""

from plot_nn_mcp.ir import IRBlockOp, IRBuilder, IRResidualOp
from plot_nn_mcp.render import emit_tikz
from plot_nn_mcp.themes import Role, get_theme


def test_emit_roundtrip_mini():
    """Smallest possible graph emits valid LaTeX shell."""
    b = IRBuilder(title="Mini")
    emb = b.add_block(Role.EMBED, "Embedding")
    out = b.add_block(Role.OUTPUT, "Classifier")
    b.set_input(emb)
    b.set_output(out)
    tex = emit_tikz(b.build(), get_theme("modern"))

    assert tex.startswith(r"\documentclass")
    assert tex.rstrip().endswith(r"\end{document}")
    assert r"\begin{tikzpicture}" in tex
    assert "Embedding" in tex and "Classifier" in tex
    assert r"{Input}" in tex and r"{Output}" in tex
    assert tex.count(r"\draw[arrow]") >= 3  # data edge + Input + Output


def test_emit_residual_has_skiparrow_and_add():
    """Emitting an IRResidualOp must produce a skiparrow and an add-circle."""
    b = IRBuilder()
    b.add_op(IRResidualOp(body=[
        IRBlockOp(Role.NORM, "LayerNorm"),
        IRBlockOp(Role.ATTENTION, "Self-Attn"),
    ]))
    tex = emit_tikz(b.build(), get_theme("modern"))
    assert r"\draw[skiparrow]" in tex
    # add circle uses {+}
    assert r"{+}" in tex


def test_no_fractional_cm_after_rounding():
    """Fractional widths from size hints must be rounded to 2 decimals."""
    b = IRBuilder()
    b.add_block(Role.ATTENTION, "Conv",
                size_hint=(1.8666666666666667, 0.85))
    tex = emit_tikz(b.build(), get_theme("modern"))
    assert "1.87cm" in tex
    assert "1.8666" not in tex


def test_input_skips_section_rule_node():
    """If the first semantic node is preceded by a section_rule anchor,
    the Input arrow must attach to the semantic node, not the rule.

    This is the structural fix for the DCGAN/YOLOv8 Input→SectionHeader bug.
    """
    b = IRBuilder()
    # Simulate SectionHeader as first element.
    b.add_block(Role.OUTPUT, "Generator", shape="section_rule", connect=False)
    real = b.add_block(Role.EMBED, "Noise z", connect=False)
    g = b.build()
    tex = emit_tikz(g, get_theme("modern"))
    assert g.input_node == real
    # Input arrow text should reference the real node, not the section header.
    assert f"({real}.south)" in tex


def test_skip_xshift_scales_with_block_width():
    """Task 4: wide blocks push skip-arrow xshift wider to clear group frame."""
    from plot_nn_mcp.ir import IRResidualOp, IRBlockOp
    import re

    b_narrow = IRBuilder()
    b_narrow.add_op(IRResidualOp(body=[
        IRBlockOp(Role.NORM, "narrow", size_hint=(2.0, 0.5)),
        IRBlockOp(Role.FFN, "x", size_hint=(2.0, 0.5)),
    ]))
    narrow_tex = emit_tikz(b_narrow.build(), get_theme("modern"))

    b_wide = IRBuilder()
    b_wide.add_op(IRResidualOp(body=[
        IRBlockOp(Role.ATTENTION, "very wide block", size_hint=(6.0, 0.9)),
        IRBlockOp(Role.FFN, "x", size_hint=(2.0, 0.5)),
    ]))
    wide_tex = emit_tikz(b_wide.build(), get_theme("modern"))

    def _xshift(tex: str) -> float:
        m = re.search(r"skiparrow\].*?\+\+\(([\d.]+),0\)", tex)
        assert m, f"no skiparrow xshift found in:\n{tex[:400]}"
        return float(m.group(1))

    narrow_x = _xshift(narrow_tex)
    wide_x = _xshift(wide_tex)
    assert wide_x > narrow_x, \
        f"wide block must get wider xshift: narrow={narrow_x}, wide={wide_x}"
    # Wide block (6cm) / 2 + safety = 3.4cm; must clear this.
    assert wide_x >= 3.0


def test_horizontal_layout_uses_right_of_positioning():
    """Task 3: horizontal layout routes blocks west→east."""
    from plot_nn_mcp.dsl import (
        Architecture, ClassificationHead, ConvBlock, Embedding,
    )
    import re

    arch = Architecture("h", layout="horizontal", theme="modern")
    arch.add(Embedding(d_model=64, label="in"))
    arch.add(ConvBlock(filters=32, pool=None, label="c1"))
    arch.add(ClassificationHead(label="out"))
    tex = arch.render()

    # Sequential blocks must use right-of, not above-of.
    assert re.search(r"right=[\d.]+cm of ", tex), \
        "horizontal layout must position blocks with right=...of"
    assert "{Input}" in tex and "{Output}" in tex
    # Horizontal flow uses east→west data arrows for edges.
    assert ".east) -- (" in tex


def test_vertical_layout_still_uses_above():
    from plot_nn_mcp.dsl import (
        Architecture, ClassificationHead, Embedding,
    )
    arch = Architecture("v", layout="vertical", theme="modern")
    arch.add(Embedding(d_model=64, label="in"))
    arch.add(ClassificationHead(label="out"))
    tex = arch.render()
    assert "above=" in tex
    # And NOT the horizontal-only east-to-west arrow pattern.
    assert ".east) -- (" not in tex or "right=" in tex


def test_horizontal_layout_no_fractional_cm():
    from plot_nn_mcp.dsl import (
        Architecture, ClassificationHead, ConvBlock, Embedding,
    )
    import re
    arch = Architecture("h", layout="horizontal")
    arch.add(Embedding(d_model=64, label="in"))
    arch.add(ConvBlock(filters=64, pool="max", label="c1"))
    arch.add(ConvBlock(filters=128, pool="max", label="c2"))
    arch.add(ClassificationHead(label="out"))
    tex = arch.render()
    assert not re.findall(r"\d\.\d{4,}cm", tex), \
        "horizontal layout must inherit the no-fractional-cm invariant"


def test_parallel_op_renders_branches_side_by_side():
    """Task 2: IRParallelOp branches must use right=of positioning,
    not stack vertically via above=of."""
    from plot_nn_mcp.ir import IRBlockOp, IRParallelOp
    import re

    b = IRBuilder(title="YOLO-like")
    b.add_block(Role.ATTENTION, "Backbone")
    b.add_op(IRParallelOp(branches=[
        IRBlockOp(Role.OUTPUT, "Head P3"),
        IRBlockOp(Role.OUTPUT, "Head P4"),
        IRBlockOp(Role.OUTPUT, "Head P5"),
    ], merge="none"))
    tex = emit_tikz(b.build(), get_theme("modern"))
    # At least 2 "right=" placements for the 2nd and 3rd heads.
    right_count = len(re.findall(r"right=[\d.]+cm of ", tex))
    assert right_count >= 2, \
        f"parallel branches must use right-of positioning: got {right_count}"
    assert all(h in tex for h in ["Head P3", "Head P4", "Head P5"])


def test_parallel_op_first_branch_goes_above_split():
    from plot_nn_mcp.ir import IRBlockOp, IRParallelOp
    b = IRBuilder()
    b.add_block(Role.ATTENTION, "root")
    b.add_op(IRParallelOp(branches=[
        IRBlockOp(Role.OUTPUT, "alpha"),
        IRBlockOp(Role.OUTPUT, "beta"),
    ]))
    g = b.build()
    # alpha should have parallel_index=0, beta parallel_index=1.
    alpha = next(n for n in g.nodes.values() if n.label == "alpha")
    beta = next(n for n in g.nodes.values() if n.label == "beta")
    assert alpha.meta["parallel_index"] == 0
    assert beta.meta["parallel_index"] == 1
    # Their previous-sibling pointer should differ — alpha → split, beta → alpha.
    assert alpha.meta["parallel_prev_sibling"] != beta.meta["parallel_prev_sibling"]
    assert beta.meta["parallel_prev_sibling"] == alpha.id


def test_skip_xshift_rounded_to_two_decimals():
    from plot_nn_mcp.ir import IRResidualOp, IRBlockOp
    import re

    b = IRBuilder()
    b.add_op(IRResidualOp(body=[
        IRBlockOp(Role.ATTENTION, "w", size_hint=(5.333, 0.7)),
        IRBlockOp(Role.FFN, "x"),
    ]))
    tex = emit_tikz(b.build(), get_theme("modern"))
    fractional = re.findall(r"\d\.\d{4,}", tex)
    assert not fractional, f"unrounded xshift/width leaked: {fractional}"


def test_skip_and_data_edges_both_rendered():
    b = IRBuilder()
    b.add_op(IRResidualOp(body=[IRBlockOp(Role.FFN, "MLP")]))
    tex = emit_tikz(b.build(), get_theme("modern"))
    assert r"\draw[arrow]" in tex
    assert r"\draw[skiparrow]" in tex
