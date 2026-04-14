"""IR → TikZ emitter.

Takes an :class:`IRGraph` (built by :class:`IRBuilder` or by the upcoming
DSL-to-IR lowering in Phase 4) and produces the same-style TikZ output as
the legacy ``_render_vertical`` path.

Design split (see REFACTOR_PLAN.md Phase 5):
- This module only *emits strings* given a layout. It does NOT compute
  positions — that is the responsibility of a layout stage.
- For Phase 3/4 we ship a trivial "vertical stack" layout that mirrors
  current behavior; Phase 5 replaces it with a proper layout solver.
"""

from __future__ import annotations

from .flat_renderer import (
    flat_add_circle,
    flat_arrow,
    flat_begin,
    flat_block,
    flat_colors,
    flat_end,
    flat_head,
    flat_op_circle,
    flat_skip_arrow,
)
from .ir import IRGraph, IRNode
from .themes import Theme


def _branch_position(node: IRNode) -> str | None:
    """If node is a sibling in an IRParallelOp, return its horizontal anchor.

    Returns a full ``right=... of ...`` option string or ``None`` if the
    node isn't a parallel sibling. First sibling goes above the split
    anchor (default vertical flow); later siblings go right of the
    previous sibling.
    """
    sibling = node.meta.get("parallel_prev_sibling") if node.meta else None
    idx = node.meta.get("parallel_index") if node.meta else None
    if sibling is None or idx is None:
        return None
    if idx == 0:
        return f"above=0.5cm of {sibling}"
    return f"right=1.2cm of {sibling}"


def _emit_node(node: IRNode, prev_id: str | None, is_first: bool,
               direction: str = "vertical") -> str:
    """Emit a single IR node as TikZ.

    ``prev_id`` is used for relative positioning (``above=... of prev``);
    ``is_first`` triggers absolute ``at (0,0)`` for the root.
    """
    if node.shape == "anchor":
        # ×N badge — emitted next to the anchor of the last visible block
        # in a collapsed pattern group.
        if node.label and "times" in node.label:
            target = node.meta.get("badge_for") or prev_id
            return (
                rf"\node[right=8pt of {target}, font=\sffamily\Large\bfseries, "
                rf"text=clrborder] ({node.id}) {{{node.label}}};" "\n"
            )
        branch_pos = _branch_position(node)
        if branch_pos is not None:
            return (
                rf"\node[inner sep=0, minimum size=0, {branch_pos}] ({node.id}) {{}};"
                "\n"
            )
        if is_first:
            pos = "at (0,0)"
        elif direction == "horizontal":
            pos = f"right=0.4cm of {prev_id}"
        else:
            pos = f"above=0.4cm of {prev_id}"
        return (
            rf"\node[inner sep=0, minimum size=0, {pos}] ({node.id}) {{}};"
            "\n"
        )

    if node.shape == "section_rule":
        # Thin horizontal rule + bold title text — used for SectionHeader
        # and Separator. No data flow through it; positioning is relative
        # to the previous node so the visual gap is preserved.
        # If this is the first element (no prev) we anchor at (0,0) — using
        # ``current bounding box.south`` before any node exists causes a
        # "Dimension too large" error in TikZ.
        rule_id = f"{node.id}_rule"
        if prev_id is None:
            out = (
                rf"\node[inner sep=0, minimum height=0] ({rule_id}) "
                rf"at (0,0) {{}};" "\n"
                rf"\draw[color=clrgroup_frame!50, line width=0.4pt] "
                rf"([xshift=-2.2cm]{rule_id}.center) -- "
                rf"([xshift=2.2cm]{rule_id}.center);" "\n"
            )
            if node.label:
                out += (
                    rf"\node[above=4pt of {rule_id}, "
                    rf"font=\sffamily\normalsize\bfseries, text=clrborder] "
                    rf"({node.id}) {{{node.label}}};" "\n"
                )
            else:
                out += (
                    rf"\node[above=4pt of {rule_id}, inner sep=0, "
                    rf"minimum size=0] ({node.id}) {{}};" "\n"
                )
            return out
        ref = prev_id
        out = (
            rf"\node[above=0.6cm of {ref}, inner sep=0, minimum height=0] "
            rf"({rule_id}) {{}};" "\n"
            rf"\draw[color=clrgroup_frame!50, line width=0.4pt] "
            rf"([xshift=-2.2cm]{rule_id}.center) -- "
            rf"([xshift=2.2cm]{rule_id}.center);" "\n"
        )
        if node.label:
            out += (
                rf"\node[above=4pt of {rule_id}, "
                rf"font=\sffamily\normalsize\bfseries, text=clrborder] "
                rf"({node.id}) {{{node.label}}};" "\n"
            )
        else:
            # Empty separator — still need a node anchor with the IR id.
            out += (
                rf"\node[above=4pt of {rule_id}, inner sep=0, minimum size=0] "
                rf"({node.id}) {{}};" "\n"
            )
        return out

    width = node.size_hint[0] if node.size_hint else 3.8
    height = node.size_hint[1] if node.size_hint else 0.85
    # Phase 5 rounding: never emit >2 decimal places — fixes the
    # "1.8666666666666667cm" regression from width_from_dim.
    width = round(width, 2)
    height = round(height, 2)

    branch_pos = _branch_position(node)
    if node.shape == "circle":
        kwargs: dict = {}
        if branch_pos is not None:
            symbol = node.label or "+"
            return (
                rf"\node[circle, draw=clrborder, fill=clrresidual!30, "
                rf"inner sep=2pt, font=\sffamily\scriptsize\bfseries, "
                rf"{branch_pos}] ({node.id}) {{{symbol}}};" "\n"
            )
        if prev_id is not None and not is_first:
            if direction == "horizontal":
                kwargs["right_of"] = prev_id
            else:
                kwargs["above_of"] = prev_id
        symbol = node.label or "+"
        return flat_op_circle(node.id, symbol=symbol, **kwargs)

    block_kwargs: dict = {
        "width": width,
        "height": height,
    }
    if branch_pos is not None:
        idx = node.meta.get("parallel_index", 0) if node.meta else 0
        sibling = node.meta.get("parallel_prev_sibling")
        if idx == 0:
            block_kwargs["above_of"] = sibling
            block_kwargs["node_distance"] = 0.5
        else:
            block_kwargs["right_of"] = sibling
            block_kwargs["node_distance"] = 1.2
    elif is_first or prev_id is None:
        block_kwargs["position"] = "(0,0)"
    elif direction == "horizontal":
        block_kwargs["right_of"] = prev_id
    else:
        block_kwargs["above_of"] = prev_id
    return flat_block(node.id, node.label, node.role, **block_kwargs)


def _safe_skip_xshift(graph: IRGraph, edge, order_index: dict[str, int],
                       safety: float = 0.4, default: float = 2.2) -> float:
    """Compute xshift that clears the widest block between src and dst.

    Fix for Task 4 (see REFACTOR_HANDOFF.md): the legacy hardcoded 2.2cm
    overlapped group frames when the residual contained a wide block
    (e.g. 4.2cm attention block in Swin). We take ``max_width / 2`` plus
    a safety margin; this is also rounded to 2 decimals to preserve the
    "no fractional cm" invariant.
    """
    src_i = order_index.get(edge.src)
    dst_i = order_index.get(edge.dst)
    if src_i is None or dst_i is None or dst_i <= src_i:
        return default
    widths: list[float] = []
    for nid in graph.order[src_i + 1:dst_i + 1]:
        node = graph.nodes[nid]
        if node.size_hint and node.shape == "block":
            widths.append(node.size_hint[0])
    if not widths:
        return default
    return round(max(widths) / 2 + safety, 2)


def _normalize_widths_within_runs(graph: IRGraph) -> None:
    """Layout pass: equalize widths of consecutive same-role blocks.

    Phase 5 fix for the YOLOv8/ResNet jitter where a Stem Conv (1.87cm)
    sits next to Stage1 Conv (3.8cm) creating visual chaos. Within a
    run of nodes sharing the same Role and shape, take the max width
    so the run reads as a coherent strip.
    """
    if not graph.order:
        return
    run: list[str] = []
    prev_role = None
    prev_shape = None

    def _flush(run_ids: list[str]) -> None:
        if len(run_ids) < 2:
            return
        widths = [graph.nodes[i].size_hint[0] for i in run_ids
                  if graph.nodes[i].size_hint]
        if not widths:
            return
        max_w = max(widths)
        for i in run_ids:
            if graph.nodes[i].size_hint:
                _, h = graph.nodes[i].size_hint
                graph.nodes[i].size_hint = (max_w, h)

    for nid in graph.order:
        node = graph.nodes[nid]
        if node.role == prev_role and node.shape == prev_shape == "block":
            run.append(nid)
        else:
            _flush(run)
            run = [nid]
            prev_role = node.role
            prev_shape = node.shape
    _flush(run)


def emit_tikz(graph: IRGraph, theme: Theme,
              direction: str = "vertical") -> str:
    """Emit a complete .tex document for the given IR graph.

    This is the straightforward "vertical stack" emitter used as the baseline
    for Phase 4 migration. Nodes are rendered in insertion order; edges are
    drawn as arrows (data) or skip arrows (skip). Branch/merge edges are
    drawn as data arrows for now — proper parallel layout comes in Phase 5.
    """
    _normalize_widths_within_runs(graph)
    parts: list[str] = [flat_head(), flat_colors(theme), flat_begin()]

    prev: str | None = None
    for i, nid in enumerate(graph.order):
        node = graph.nodes[nid]
        parts.append(_emit_node(node, prev_id=prev, is_first=(i == 0),
                                 direction=direction))
        prev = nid

    # Edges. In horizontal mode data arrows use east→west, skip arrows
    # loop north/south instead of east/west.
    order_index = {nid: i for i, nid in enumerate(graph.order)}
    for edge in graph.edges:
        if edge.kind == "skip":
            xshift = _safe_skip_xshift(graph, edge, order_index)
            parts.append(flat_skip_arrow(edge.src, edge.dst, xshift=xshift))
        elif direction == "horizontal":
            parts.append(flat_arrow(edge.src, edge.dst,
                                     from_anchor="east", to_anchor="west"))
        else:
            parts.append(flat_arrow(edge.src, edge.dst))

    # Input/Output arrows. When the IO endpoint already carries the same
    # label text (e.g. ClassificationHead("Output")), suppress the duplicate
    # arrow annotation — fixes the legacy "double Output" bug in DeBERTa.
    def _redundant(node_id: str | None, label_text: str) -> bool:
        if node_id is None:
            return False
        node_label = graph.nodes[node_id].label or ""
        return node_label.strip().lower() == label_text.lower()

    io_start_anchor = "west" if direction == "horizontal" else "south"
    io_end_anchor = "east" if direction == "horizontal" else "north"
    io_start_offset = "(-0.5,0)" if direction == "horizontal" else "(0,-0.5)"
    io_end_offset = "(0.5,0)" if direction == "horizontal" else "(0,0.5)"

    if graph.input_node:
        if _redundant(graph.input_node, "Input"):
            parts.append(
                rf"\draw[arrow] ({graph.input_node}.{io_start_anchor}) "
                rf"++ {io_start_offset} -- "
                rf"({graph.input_node}.{io_start_anchor});" "\n"
            )
        else:
            parts.append(
                rf"\draw[arrow] ({graph.input_node}.{io_start_anchor}) "
                rf"++ {io_start_offset} -- "
                rf"node[right, yshift=-2pt, font=\sffamily\scriptsize, "
                rf"text=clrtext] {{Input}} ({graph.input_node}.{io_start_anchor});"
                "\n"
            )
    if graph.output_node:
        if _redundant(graph.output_node, "Output"):
            parts.append(
                rf"\draw[arrow] ({graph.output_node}.{io_end_anchor}) -- "
                rf"++ {io_end_offset};" "\n"
            )
        else:
            parts.append(
                rf"\draw[arrow] ({graph.output_node}.{io_end_anchor}) -- "
                rf"node[right, yshift=-2pt, font=\sffamily\scriptsize, "
                rf"text=clrtext] {{Output}} ++ {io_end_offset};" "\n"
            )

    # Group frames (fit boxes around children) with optional ×N badge.
    for gid, group in graph.groups.items():
        if not group.children:
            continue
        fit = " ".join(f"({c})" for c in group.children)
        parts.append(
            rf"\node[draw=clrgroup_frame, rounded corners=6pt, dashed, "
            rf"line width=0.8pt, fill=clrgroup_fill, fill opacity=0.4, "
            rf"inner sep=0.35cm, fit={fit}] ({gid}) {{}};" "\n"
        )
        if group.title:
            parts.append(
                rf"\node[above=4pt of {gid}.north, font=\sffamily\small\bfseries, "
                rf"text=clrborder] ({gid}_title) {{{group.title}}};" "\n"
            )
        if group.repeat_count and group.repeat_count > 1:
            parts.append(
                rf"\node[right=12pt of {gid}.east, anchor=west, "
                rf"font=\sffamily\Large\bfseries, text=clrborder] "
                rf"({gid}_badge) {{$\times{group.repeat_count}$}};" "\n"
            )

    if graph.title:
        parts.append(
            rf"\node[below=0.6cm of current bounding box.south, "
            rf"subtitle, text width=14cm, align=center] "
            rf"{{\textbf{{{graph.title}}}}};" "\n"
        )

    parts.append(flat_end())
    return "".join(parts)
