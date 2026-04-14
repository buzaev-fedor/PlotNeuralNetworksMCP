"""Intermediate Representation (IR) for architecture diagrams.

This is a DAG-based intermediate layer between the user-facing DSL
(``Architecture.add(LayerType(...))``) and TikZ emission. It replaces the
implicit single-pointer ``prev: str`` model used by the legacy renderer.

Design goals (see REFACTOR_PLAN.md Phase 3):
- Explicit ``input_node`` / ``output_node`` so IO-arrows can never attach
  to a section header.
- Typed edges (``data`` / ``skip`` / ``branch`` / ``merge``) so parallel
  branches and residual connections are first-class.
- ``IRGroup`` with optional ``title`` so section headers and pattern
  grouping become one concept (fixes DCGAN/YOLO regressions).

Phase 3 scope: types and builder only. Wiring into ``_render_vertical``
happens in Phase 4 together with composable ``Residual`` / ``Parallel``
primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .themes import Role

EdgeKind = Literal["data", "skip", "branch", "merge"]
ShapeHint = Literal["block", "circle", "anchor", "section_rule"]


@dataclass
class IRNode:
    """A single visual element in the diagram.

    ``size_hint`` is advisory — the layout pass (Phase 5) decides final
    geometry and must round to avoid ugly fractional cm in the emission.
    """
    id: str
    role: Role
    label: str = ""
    shape: ShapeHint = "block"
    size_hint: tuple[float, float] | None = None   # (width, height) in cm
    dim: int | None = None                          # side-label for tensor dim
    meta: dict = field(default_factory=dict)


@dataclass
class IREdge:
    src: str
    dst: str
    kind: EdgeKind = "data"
    label: str = ""
    style: dict = field(default_factory=dict)


@dataclass
class IRGroup:
    """Grouping construct. Unifies ``SectionHeader`` and pattern ``×N``.

    - ``title`` is set → renders a section header rule + label.
    - ``repeat_count`` is set → renders a ``×N`` badge beside the frame.
    - Both can coexist (a section that happens to be a repeating block).
    - ``children`` is a list of node IDs that belong directly to this group.
    - ``subgroups`` enables nesting (e.g. sections of sections).
    """
    id: str
    children: list[str] = field(default_factory=list)
    subgroups: list[str] = field(default_factory=list)
    title: str | None = None
    subtitle: str | None = None
    repeat_count: int | None = None


@dataclass
class IRGraph:
    nodes: dict[str, IRNode] = field(default_factory=dict)
    edges: list[IREdge] = field(default_factory=list)
    groups: dict[str, IRGroup] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # nodes in insertion order
    input_node: str | None = None                    # explicit IO endpoints
    output_node: str | None = None
    title: str = ""
    subtitle: str = ""

    def add_node(self, node: IRNode) -> str:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id: {node.id!r}")
        self.nodes[node.id] = node
        self.order.append(node.id)
        return node.id

    def add_edge(self, src: str, dst: str, kind: EdgeKind = "data",
                 label: str = "") -> None:
        if src not in self.nodes:
            raise KeyError(f"unknown src node: {src!r}")
        if dst not in self.nodes:
            raise KeyError(f"unknown dst node: {dst!r}")
        self.edges.append(IREdge(src=src, dst=dst, kind=kind, label=label))

    def add_group(self, group: IRGroup) -> str:
        if group.id in self.groups:
            raise ValueError(f"duplicate group id: {group.id!r}")
        self.groups[group.id] = group
        return group.id


# ---------------------------------------------------------------------------
# Composable operators (Phase 4)
#
# An ``IROp`` is a declarative spec that the builder "lowers" into concrete
# ``IRNode`` + ``IREdge`` entries. The key insight: Residual is expressed
# *once* and every block that wraps it (Transformer attn, Transformer FFN,
# Mixtral MoE, ResNet bottleneck) reuses the same lowering. This closes the
# Mixtral-MoE-without-add2 regression structurally, not by patching cases.
# ---------------------------------------------------------------------------


@dataclass
class IRBlockOp:
    """Leaf op — becomes a single :class:`IRNode`."""
    role: Role
    label: str
    shape: ShapeHint = "block"
    size_hint: tuple[float, float] | None = None
    dim: int | None = None


@dataclass
class IRSequenceOp:
    """Ops chained with ``data`` edges."""
    ops: list["IROp"]


@dataclass
class IRResidualOp:
    """Wraps ``body`` with a skip connection: entry → body → add ← skip(entry).

    Lowers to: anchor(entry), body nodes, add-circle, data-edges through
    body, skip-edge from entry to add. Used for attention residuals, FFN
    residuals, MoE residuals (post-Phase-4) — one implementation.
    """
    body: list["IROp"]


@dataclass
class IRParallelOp:
    """Branches split from one source, optionally merge into one sink.

    For YOLOv8 multi-scale detection heads, Inception modules, encoder/
    decoder parallelism.
    """
    branches: list["IROp"]
    merge: Literal["none", "concat", "add"] = "none"


@dataclass
class IRCustomOp:
    """Escape hatch — owns a list of pre-built nodes and edges.

    Used during gradual migration for layers that have complex internal
    layouts (LSTM gates, U-Net cross-shape, GAN generator/discriminator
    pairs) that don't yet have a clean composable lowering. The custom
    op participates in the IR DAG (entry/exit are exposed) so it can be
    sequenced with normal ops, but its internal rendering goes through
    the legacy code path.
    """
    nodes: list[IRNode]
    edges: list[IREdge]
    entry_id: str   # node id where data flow enters
    exit_id: str    # node id where data flow exits


IROp = IRBlockOp | IRSequenceOp | IRResidualOp | IRParallelOp | IRCustomOp


class IRBuilder:
    """Fluent builder for an :class:`IRGraph`.

    Maintains a ``cursor`` pointing at the most recently added node, so
    sequential ``add_block`` calls auto-connect with a ``data`` edge. For
    residuals/branches, callers use explicit :meth:`residual` /
    :meth:`parallel` helpers (added in Phase 4).
    """

    def __init__(self, title: str = "", subtitle: str = "") -> None:
        self.graph = IRGraph(title=title, subtitle=subtitle)
        self.cursor: str | None = None
        self._id_counter = 0
        self._group_counter = 0

    def _next_id(self, prefix: str) -> str:
        self._id_counter += 1
        return f"{prefix}_{self._id_counter}"

    def add_block(
        self,
        role: Role,
        label: str,
        *,
        shape: ShapeHint = "block",
        size_hint: tuple[float, float] | None = None,
        dim: int | None = None,
        id: str | None = None,
        connect: bool = True,
    ) -> str:
        nid = id or self._next_id("n")
        self.graph.add_node(IRNode(
            id=nid, role=role, label=label, shape=shape,
            size_hint=size_hint, dim=dim,
        ))
        if connect and self.cursor is not None:
            self.graph.add_edge(self.cursor, nid, kind="data")
        self.cursor = nid
        return nid

    def set_input(self, node_id: str) -> None:
        if node_id not in self.graph.nodes:
            raise KeyError(node_id)
        self.graph.input_node = node_id

    def set_output(self, node_id: str) -> None:
        if node_id not in self.graph.nodes:
            raise KeyError(node_id)
        self.graph.output_node = node_id

    def connect(self, src: str, dst: str, kind: EdgeKind = "data",
                label: str = "") -> None:
        self.graph.add_edge(src, dst, kind=kind, label=label)

    def new_group(
        self, *, title: str | None = None, subtitle: str | None = None,
        repeat_count: int | None = None,
    ) -> IRGroup:
        gid = f"g_{self._group_counter}"
        self._group_counter += 1
        grp = IRGroup(id=gid, title=title, subtitle=subtitle,
                      repeat_count=repeat_count)
        self.graph.add_group(grp)
        return grp

    def lower(self, op: "IROp") -> tuple[str, str]:
        """Lower a composable op into nodes/edges, return (entry_id, exit_id).

        ``entry_id`` is the id of the first node in the op (useful for skip
        connections from outside). ``exit_id`` is the last node (this is
        what the builder's cursor advances to).

        The caller is responsible for connecting ``entry_id`` to whatever
        came before (or not, if this is the graph's input).
        """
        if isinstance(op, IRBlockOp):
            nid = self.add_block(
                op.role, op.label, shape=op.shape,
                size_hint=op.size_hint, dim=op.dim, connect=False,
            )
            return nid, nid

        if isinstance(op, IRSequenceOp):
            if not op.ops:
                raise ValueError("IRSequenceOp requires at least one op")
            entry, last = self.lower(op.ops[0])
            for child in op.ops[1:]:
                child_entry, child_exit = self.lower(child)
                self.connect(last, child_entry, kind="data")
                last = child_exit
            return entry, last

        if isinstance(op, IRResidualOp):
            # Insert an anchor so the skip-edge has a stable origin even
            # when the body's first node is inline-positioned.
            entry_id = self._next_id("res_entry")
            self.graph.add_node(IRNode(
                id=entry_id, role=Role.RESIDUAL, label="",
                shape="anchor",
            ))
            # body
            if not op.body:
                raise ValueError("IRResidualOp requires non-empty body")
            body_entry, body_exit = self.lower(IRSequenceOp(op.body))
            self.connect(entry_id, body_entry, kind="data")
            # add node
            add_id = self._next_id("add")
            self.graph.add_node(IRNode(
                id=add_id, role=Role.RESIDUAL, label="+",
                shape="circle",
            ))
            self.connect(body_exit, add_id, kind="data")
            self.connect(entry_id, add_id, kind="skip")
            return entry_id, add_id

        if isinstance(op, IRCustomOp):
            for n in op.nodes:
                self.graph.add_node(n)
            self.graph.edges.extend(op.edges)
            return op.entry_id, op.exit_id

        if isinstance(op, IRParallelOp):
            if not op.branches:
                raise ValueError("IRParallelOp requires at least one branch")
            # For now emit branches sequentially as anchors; proper parallel
            # layout lands in Phase 5. Preserve branch/merge edge kinds so
            # future layout can identify the split.
            split_id = self._next_id("split")
            self.graph.add_node(IRNode(
                id=split_id, role=Role.RESIDUAL, label="",
                shape="anchor",
            ))
            exits: list[str] = []
            entries: list[str] = []
            for branch in op.branches:
                b_entry, b_exit = self.lower(branch)
                self.connect(split_id, b_entry, kind="branch")
                exits.append(b_exit)
                entries.append(b_entry)
            # Annotate siblings so the emitter can place branches side-by-side.
            # First branch entry anchors to the split (vertical), later
            # entries anchor to the previous sibling (right-of horizontal).
            for i, eid in enumerate(entries):
                self.graph.nodes[eid].meta["parallel_prev_sibling"] = (
                    split_id if i == 0 else entries[i - 1]
                )
                self.graph.nodes[eid].meta["parallel_index"] = i
            if op.merge == "none":
                return split_id, exits[-1]
            merge_id = self._next_id("merge")
            symbol = "+" if op.merge == "add" else "||"
            self.graph.add_node(IRNode(
                id=merge_id, role=Role.RESIDUAL, label=symbol,
                shape="circle",
            ))
            for exit_id in exits:
                self.connect(exit_id, merge_id, kind="merge")
            return split_id, merge_id

        raise TypeError(f"unknown IROp: {type(op).__name__}")

    def add_op(self, op: "IROp") -> str:
        """Lower an op and auto-connect to cursor. Returns the exit node id."""
        entry, exit_ = self.lower(op)
        if self.cursor is not None:
            self.connect(self.cursor, entry, kind="data")
        self.cursor = exit_
        return exit_

    def build(self) -> IRGraph:
        # Auto-assign IO endpoints if caller forgot.
        if self.graph.input_node is None and self.graph.order:
            self.graph.input_node = self._first_semantic_node()
        if self.graph.output_node is None and self.graph.order:
            self.graph.output_node = self.graph.order[-1]
        return self.graph

    def _first_semantic_node(self) -> str:
        """Return the first node that is NOT a structural anchor/rule.

        This is where the ``Input`` arrow must attach — fixes the bug where
        ``first_node = prev`` in the legacy renderer caused Input to point
        at a ``SectionHeader`` (see REFACTOR_PLAN.md §"Карта соответствия").
        """
        for nid in self.graph.order:
            node = self.graph.nodes[nid]
            if node.shape not in ("anchor", "section_rule"):
                return nid
        return self.graph.order[0]
