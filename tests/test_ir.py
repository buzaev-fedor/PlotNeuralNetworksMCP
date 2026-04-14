"""Tests for the IR builder (Phase 3 scaffolding + Phase 4 ops)."""

from plot_nn_mcp.ir import (
    IRBlockOp,
    IRBuilder,
    IRNode,
    IRParallelOp,
    IRResidualOp,
    IRSequenceOp,
)
from plot_nn_mcp.themes import Role


def test_sequential_add_auto_connects():
    b = IRBuilder(title="test")
    a = b.add_block(Role.EMBED, "Embedding")
    c = b.add_block(Role.ATTENTION, "Attention")
    graph = b.build()

    assert graph.order == [a, c]
    assert len(graph.edges) == 1
    assert graph.edges[0].src == a and graph.edges[0].dst == c
    assert graph.edges[0].kind == "data"


def test_input_defaults_to_first_semantic_node():
    """SectionHeader anchors must NOT become the Input endpoint."""
    b = IRBuilder()
    header = b.add_block(Role.OUTPUT, "Encoder", shape="section_rule",
                          connect=False)
    real = b.add_block(Role.EMBED, "Token Embedding", connect=False)
    graph = b.build()

    assert graph.input_node == real, \
        "Input must skip structural section_rule nodes"


def test_explicit_input_output():
    b = IRBuilder()
    a = b.add_block(Role.EMBED, "A")
    c = b.add_block(Role.OUTPUT, "C")
    b.set_input(a)
    b.set_output(c)
    g = b.build()
    assert g.input_node == a
    assert g.output_node == c


def test_skip_edge_kind():
    b = IRBuilder()
    entry = b.add_block(Role.NORM, "entry")
    body = b.add_block(Role.ATTENTION, "body")
    add = b.add_block(Role.RESIDUAL, "+", shape="circle")
    b.connect(entry, add, kind="skip")
    g = b.build()

    kinds = [e.kind for e in g.edges]
    assert "skip" in kinds
    assert "data" in kinds


def test_duplicate_node_id_rejected():
    b = IRBuilder()
    b.add_block(Role.EMBED, "a", id="x")
    try:
        b.add_block(Role.EMBED, "b", id="x")
    except ValueError:
        return
    raise AssertionError("expected ValueError on duplicate id")


def test_residual_op_lowering_generates_skip_edge():
    """An IRResidualOp must produce exactly one skip-edge from entry to add."""
    b = IRBuilder()
    b.add_op(IRResidualOp(body=[
        IRBlockOp(Role.NORM, "LayerNorm"),
        IRBlockOp(Role.ATTENTION, "Self-Attn"),
    ]))
    g = b.build()

    skip_edges = [e for e in g.edges if e.kind == "skip"]
    assert len(skip_edges) == 1, f"expected 1 skip edge, got {len(skip_edges)}"
    # Must terminate at the add-circle (last circle-shaped node).
    circles = [n for n in g.nodes.values() if n.shape == "circle"]
    assert len(circles) == 1, "expected exactly one add-circle"
    assert skip_edges[0].dst == circles[0].id


def test_sequence_of_two_residuals_fixes_mixtral_pattern():
    """Transformer attn+FFN and Mixtral attn+MoE share the SAME lowering:
    two sequential IRResidualOp. Both get 2 skip edges — no bug possible.
    """
    b = IRBuilder()
    b.add_op(IRSequenceOp(ops=[
        IRResidualOp(body=[IRBlockOp(Role.NORM, "LN"),
                           IRBlockOp(Role.ATTENTION, "Attn")]),
        IRResidualOp(body=[IRBlockOp(Role.NORM, "LN"),
                           IRBlockOp(Role.FFN, "MoE Experts")]),
    ]))
    g = b.build()
    skip_count = sum(1 for e in g.edges if e.kind == "skip")
    assert skip_count == 2, \
        f"expected 2 residual skips (attn + MoE), got {skip_count}"


def test_parallel_op_preserves_branch_edges():
    """YOLOv8-style multi-head output — branches must be marked as such."""
    b = IRBuilder()
    b.add_block(Role.ATTENTION, "Backbone")
    b.add_op(IRParallelOp(
        branches=[
            IRBlockOp(Role.OUTPUT, "Detect P3"),
            IRBlockOp(Role.OUTPUT, "Detect P4"),
            IRBlockOp(Role.OUTPUT, "Detect P5"),
        ],
        merge="none",
    ))
    g = b.build()
    branch_edges = [e for e in g.edges if e.kind == "branch"]
    assert len(branch_edges) == 3, \
        f"expected 3 branch edges, got {len(branch_edges)}"


def test_group_assignment():
    b = IRBuilder()
    n = b.add_block(Role.ATTENTION, "Attn")
    grp = b.new_group(title="Encoder", repeat_count=6)
    grp.children.append(n)
    g = b.build()
    assert g.groups[grp.id].title == "Encoder"
    assert g.groups[grp.id].repeat_count == 6
    assert n in g.groups[grp.id].children
