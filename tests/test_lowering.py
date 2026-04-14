"""Tests for DSL → IR lowering (Phase 4b).

These prove the structural fixes from REFACTOR_PLAN.md are baked in:
- TransformerBlock pre-LN produces 2 residuals (attn + FFN).
- MoELayer produces 1 residual (NOT bare Router→Experts) — Mixtral fix.
- skip_ffn=True produces 1 residual (no second FFN one).
- ConvBlock with pool produces 2 nodes in sequence.
"""

from plot_nn_mcp import dsl
from plot_nn_mcp.ir import (
    IRBlockOp,
    IRBuilder,
    IRResidualOp,
    IRSequenceOp,
)
from plot_nn_mcp.lowering import layer_to_ir
from plot_nn_mcp.render import emit_tikz
from plot_nn_mcp.themes import Role, get_theme


def _count_residuals(op):
    if isinstance(op, IRResidualOp):
        return 1 + sum(_count_residuals(c) for c in op.body)
    if isinstance(op, IRSequenceOp):
        return sum(_count_residuals(c) for c in op.ops)
    return 0


def test_transformer_pre_ln_has_two_residuals():
    block = dsl.TransformerBlock(attention="self", norm="pre_ln")
    op = layer_to_ir(block)
    assert _count_residuals(op) == 2, \
        "Pre-LN transformer must wrap attn AND FFN in residuals"


def test_transformer_skip_ffn_has_one_residual():
    block = dsl.TransformerBlock(attention="self", skip_ffn=True)
    op = layer_to_ir(block)
    assert _count_residuals(op) == 1, \
        "skip_ffn must omit the FFN residual"


def test_moe_layer_has_residual_fixes_mixtral():
    """The CORE Mixtral fix: MoELayer is now wrapped in a residual.

    Pre-refactor, the legacy renderer drew Router→Experts WITHOUT add2,
    producing a transformer block whose MoE half had no skip connection.
    """
    layer = dsl.MoELayer(num_experts=8, top_k=2, d_ff=14336)
    op = layer_to_ir(layer)
    assert isinstance(op, IRResidualOp), \
        "MoELayer must lower to IRResidualOp (was bare in legacy)"


def test_mixtral_pattern_emits_two_skip_arrows():
    """End-to-end: lower attn + MoE, render, count skips.

    This is the structural guarantee that the Mixtral diagram now has
    BOTH the attention skip and the MoE skip — impossible to forget."""
    b = IRBuilder()
    b.add_op(layer_to_ir(dsl.TransformerBlock(attention="gqa", kv_heads=8,
                                              skip_ffn=True)))
    b.add_op(layer_to_ir(dsl.MoELayer(num_experts=8, top_k=2)))
    tex = emit_tikz(b.build(), get_theme("modern"))
    assert tex.count(r"\draw[skiparrow]") == 2, \
        "Mixtral block must produce 2 skip arrows (attn + MoE)"


def test_conv_block_with_pool_lowers_to_sequence():
    block = dsl.ConvBlock(filters=64, kernel_size=7, pool="max")
    op = layer_to_ir(block)
    assert isinstance(op, IRSequenceOp)
    assert len(op.ops) == 2  # conv + pool


def test_conv_block_no_pool_lowers_to_single_block():
    block = dsl.ConvBlock(filters=128, pool=None)
    op = layer_to_ir(block)
    assert isinstance(op, IRBlockOp)
    assert op.dim == 128


def test_residual_block_lowers_to_residual():
    block = dsl.ResidualBlock(filters=64, kernel_size=3)
    op = layer_to_ir(block)
    assert isinstance(op, IRResidualOp)
    assert len(op.body) == 2


def test_unregistered_layer_raises():
    class FakeLayer:
        pass
    try:
        layer_to_ir(FakeLayer())
    except NotImplementedError as e:
        assert "FakeLayer" in str(e)
        return
    raise AssertionError("expected NotImplementedError")


def test_coverage_progress():
    """Track refactor progress — fail if coverage regresses below threshold."""
    from plot_nn_mcp.lowering import coverage
    cov = coverage()
    assert cov["total"] == 47, f"layer-type inventory drifted: {cov}"
    # Task 5-6 complete: full migration, zero legacy fallback.
    assert cov["migrated"] == cov["total"], f"migration regressed: {cov}"
    assert cov["legacy_only"] == 0, f"legacy fallback resurfaced: {cov}"
    # Every layer must be either migrated or explicitly legacy_only — the
    # invariant that catches "forgotten" new types.
    assert cov["migrated"] + cov["legacy_only"] == cov["total"], \
        f"some layer types neither migrated nor declared legacy: {cov}"


def test_swin_block_has_two_residuals():
    """Swin uses the same composable shape as transformer."""
    op = layer_to_ir(dsl.SwinBlock(window_type="shifted"))
    assert _count_residuals(op) == 2


def test_mbconv_has_residual_with_se():
    op = layer_to_ir(dsl.MBConvBlock(filters=64, se=True))
    assert isinstance(op, IRResidualOp)
    # 4 children: expand, dw-conv, SE, project
    assert len(op.body) == 4


def test_mbconv_has_residual_without_se():
    op = layer_to_ir(dsl.MBConvBlock(filters=64, se=False))
    assert isinstance(op, IRResidualOp)
    assert len(op.body) == 3  # no SE


def test_lstm_gates_style_has_residual_and_gates():
    """LSTM in gates mode: concat + forget + input wrapped in a residual
    (⊕ is the cell update), followed by output gate and h_t."""
    op = layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="gates"))
    assert isinstance(op, IRSequenceOp)
    # First op is the residual around the cell update
    assert isinstance(op.ops[0], IRResidualOp)
    body_labels = [b.label for b in op.ops[0].body if isinstance(b, IRBlockOp)]
    assert any("Forget" in l for l in body_labels)
    assert any("Input" in l for l in body_labels)
    # Output gate + h_t after the residual
    assert any(isinstance(o, IRBlockOp) and "Output Gate" in o.label
               for o in op.ops[1:])


def test_lstm_compact_style_falls_back_to_single_box():
    op = layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="compact"))
    assert isinstance(op, IRBlockOp)


def test_lstm_bidirectional_adds_merge_node():
    op = layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="gates",
                                   bidirectional=True))
    assert isinstance(op, IRSequenceOp)
    merge = op.ops[-1]
    assert isinstance(merge, IRBlockOp)
    assert "Bi-Merge" in merge.label


def test_gru_gates_style_has_residual_with_four_gates():
    op = layer_to_ir(dsl.GRUBlock(hidden_size=128, style="gates"))
    # Without bidirectional, op is just the IRResidualOp
    residual = op if isinstance(op, IRResidualOp) else op.ops[0]
    assert isinstance(residual, IRResidualOp)
    # concat + reset + update + candidate
    assert len(residual.body) == 4


def test_mamba_lowers_to_three_block_sequence():
    op = layer_to_ir(dsl.MambaBlock(d_model=256, d_state=16))
    assert isinstance(op, IRSequenceOp)
    assert len(op.ops) == 3
    assert all(isinstance(o, IRBlockOp) for o in op.ops)


def test_lstm_gates_end_to_end_renders_skiparrow():
    b = IRBuilder()
    b.add_op(layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="gates")))
    tex = emit_tikz(b.build(), get_theme("modern"))
    assert tex.count(r"\draw[skiparrow]") == 1
    assert r"\sigma" in tex


def test_bottleneck_block_has_residual():
    op = layer_to_ir(dsl.BottleneckBlock(filters=256, expansion=4))
    assert isinstance(op, IRResidualOp)
    assert len(op.body) == 3  # 1x1 → 3x3 → 1x1


def test_lstm_gates_style_has_residual():
    """Task 1: LSTM default style produces residual (conveyor belt)."""
    from plot_nn_mcp.ir import IRSequenceOp
    op = layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="gates"))
    assert isinstance(op, IRSequenceOp)
    # First op in sequence is the residual wrap
    assert isinstance(op.ops[0], IRResidualOp)
    # Body has 3 blocks: concat, forget, input
    assert len(op.ops[0].body) == 3


def test_lstm_compact_style_is_single_box():
    op = layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="compact"))
    assert isinstance(op, IRBlockOp)


def test_gru_gates_style_has_residual_with_four_children():
    from plot_nn_mcp.ir import IRSequenceOp
    op = layer_to_ir(dsl.GRUBlock(hidden_size=128, style="gates"))
    # Non-bidirectional: the sequence-of-one collapses to the residual itself
    res = op.ops[0] if isinstance(op, IRSequenceOp) else op
    assert isinstance(res, IRResidualOp)
    assert len(res.body) == 4  # concat, reset, update, candidate


def test_lstm_bidirectional_adds_merge():
    from plot_nn_mcp.ir import IRSequenceOp
    op = layer_to_ir(dsl.LSTMBlock(hidden_size=128, style="gates",
                                    bidirectional=True))
    assert isinstance(op, IRSequenceOp)
    last_label = op.ops[-1].label if isinstance(op.ops[-1], IRBlockOp) else ""
    assert "Bi-Merge" in last_label


def test_mamba_block_lowers_to_proj_sandwich():
    from plot_nn_mcp.ir import IRSequenceOp
    op = layer_to_ir(dsl.MambaBlock(d_model=256, d_state=16))
    assert isinstance(op, IRSequenceOp)
    assert len(op.ops) == 3  # in-proj, SSM, out-proj


def test_lstm_baseline_renders_with_gate_symbols():
    """End-to-end: the LSTM NER baseline contains $\\sigma$ gate labels."""
    import os
    os.environ["PLOTNN_SKIP_WRITE"] = "1"
    import importlib, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    if "generate_all" in sys.modules:
        importlib.reload(sys.modules["generate_all"])
    else:
        importlib.import_module("generate_all")
    from generate_all import ARCHITECTURES
    name, arch, show_n = next(a for a in ARCHITECTURES if a[0] == "01_lstm_ner")
    tex = arch.render(show_n=show_n)
    assert r"\sigma" in tex, "LSTM gates style must render $\\sigma$ labels"
    assert "Forget Gate" in tex or "Input Gate" in tex


def test_side_by_side_lowers_to_parallel():
    """Task 5: SideBySide becomes IRParallelOp with 2 branches."""
    from plot_nn_mcp.ir import IRParallelOp
    layer = dsl.SideBySide(
        left=[dsl.Embedding(d_model=64, label="Enc")],
        right=[dsl.Embedding(d_model=64, label="Dec")],
    )
    op = layer_to_ir(layer)
    assert isinstance(op, IRParallelOp)
    assert len(op.branches) == 2


def test_encoder_decoder_lowers_to_parallel():
    from plot_nn_mcp.ir import IRParallelOp
    layer = dsl.EncoderDecoder(
        encoder=[dsl.TransformerBlock(), dsl.TransformerBlock()],
        decoder=[dsl.TransformerBlock()],
    )
    op = layer_to_ir(layer)
    assert isinstance(op, IRParallelOp)
    assert len(op.branches) == 2


def test_spinn_has_three_branches():
    from plot_nn_mcp.ir import IRParallelOp
    op = layer_to_ir(dsl.SPINNBlock())
    assert isinstance(op, IRParallelOp)
    assert len(op.branches) == 3


def test_unet_level_has_encoder_and_decoder_branches():
    from plot_nn_mcp.ir import IRParallelOp
    layer = dsl.UNetLevel(
        encoder=[dsl.ConvBlock(filters=64, pool=None)],
        decoder=[dsl.ConvBlock(filters=64, pool=None)],
    )
    op = layer_to_ir(layer)
    assert isinstance(op, IRParallelOp)
    assert len(op.branches) == 2


def test_bottleneck_lowers_to_block():
    op = layer_to_ir(dsl.Bottleneck(label="1x1x512"))
    assert isinstance(op, IRBlockOp)


def test_render_pipeline_no_fractional_cm():
    """Resnet bottleneck width 64 → 1.866...cm raw, must round to 1.87."""
    b = IRBuilder()
    b.add_op(layer_to_ir(dsl.ConvBlock(filters=64, kernel_size=7, pool="max")))
    tex = emit_tikz(b.build(), get_theme("modern"))
    import re
    fractional = re.findall(r"\d\.\d{4,}cm", tex)
    assert not fractional, f"unrounded cm leaked: {fractional}"


def test_architecture_to_ir_smoke():
    """A fully-migrated mini architecture builds end-to-end."""
    from plot_nn_mcp.lowering import architecture_to_ir

    arch = dsl.Architecture("Mini-BERT")
    arch.add(dsl.Embedding(d_model=128, label="Token Emb"))
    arch.add(dsl.TransformerBlock(d_model=128, heads=4))
    arch.add(dsl.TransformerBlock(d_model=128, heads=4))
    arch.add(dsl.ClassificationHead(label="[CLS]"))
    g = architecture_to_ir(arch)

    # 2 transformer blocks × 2 residuals each = 4 skip edges.
    skip_count = sum(1 for e in g.edges if e.kind == "skip")
    assert skip_count == 4
    # Input is the embedding (first semantic node), Output is the head.
    assert g.input_node is not None
    assert g.output_node is not None
    assert g.input_node != g.output_node


def test_can_lower_architecture_detects_missing_types():
    """Unregistered custom class must be flagged by can_lower_architecture."""
    from plot_nn_mcp.lowering import can_lower_architecture
    from dataclasses import dataclass

    @dataclass
    class UnregisteredLayer:
        label: str = "unknown"

    arch = dsl.Architecture("Has-Unknown")
    arch.add(dsl.Embedding(d_model=64, label="E"))
    arch.add(UnregisteredLayer())
    ok, missing = can_lower_architecture(arch)
    assert not ok
    assert "UnregisteredLayer" in missing


def test_count_baseline_architectures_lowerable():
    """How many of the 25 baselines can already use the new IR path?

    Establishes a refactor-progress metric. Not a hard threshold yet
    because most baselines use SectionHeader (Phase 6).
    """
    import os
    os.environ["PLOTNN_SKIP_WRITE"] = "1"
    import importlib, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    if "generate_all" in sys.modules:
        importlib.reload(sys.modules["generate_all"])
    else:
        importlib.import_module("generate_all")
    from generate_all import ARCHITECTURES
    from plot_nn_mcp.lowering import can_lower_architecture

    lowerable = [name for name, arch, _ in ARCHITECTURES
                 if can_lower_architecture(arch)[0]]
    # Document the current count — pinning lets us see progress between
    # phases. Bump as Phase 4c migrates more types.
    assert len(lowerable) >= 1, \
        f"Expected at least 1 architecture lowerable; got {lowerable}"
