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

# Spacing around SectionHeader rules. Tuned so that the bold title + optional
# subtitle sit clear of both the preceding block and the next block; the
# legacy 0.6/0.4 cm pair let wide title labels clip into neighbour frames.
_SECTION_GAP_BEFORE_CM = 1.0
_SECTION_GAP_AFTER_CM = 0.4  # applied to next block via node `above` distance


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
               direction: str = "vertical",
               prev_shape: str | None = None) -> str:
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
        gap = "0.4cm"
        if prev_shape == "section_rule":
            gap = f"{_SECTION_GAP_AFTER_CM + 0.4:g}cm"
        if is_first:
            pos = "at (0,0)"
        elif direction == "horizontal":
            pos = f"right={gap} of {prev_id}"
        else:
            pos = f"above={gap} of {prev_id}"
        return (
            rf"\node[inner sep=0, minimum size=0, {pos}] ({node.id}) {{}};"
            "\n"
        )

    if node.shape == "section_rule":
        # Thin rule + bold title text — used for SectionHeader and
        # Separator. In vertical flow the rule is horizontal and the
        # title sits above it; in horizontal flow the rule is a vertical
        # tick and the title sits above it floating over the chain.
        # If this is the first element (no prev) we anchor at (0,0) to
        # avoid a "Dimension too large" TikZ error from empty bbox refs.
        rule_id = f"{node.id}_rule"
        horizontal = direction == "horizontal"
        if horizontal:
            rule_stroke = (
                rf"\draw[color=clrgroup_frame!50, line width=0.4pt] "
                rf"([yshift=-1.8cm]{rule_id}.center) -- "
                rf"([yshift=1.8cm]{rule_id}.center);" "\n"
            )
        else:
            rule_stroke = (
                rf"\draw[color=clrgroup_frame!50, line width=0.4pt] "
                rf"([xshift=-2.2cm]{rule_id}.center) -- "
                rf"([xshift=2.2cm]{rule_id}.center);" "\n"
            )
        # In horizontal mode the next block positions itself relative to
        # node.id, so node.id must sit on the flow line (not above the
        # rule where the title floats). We emit a secondary title label.
        # Positioning options must live INSIDE the [] — placing
        # "above=... of ..." outside the bracket is invalid TikZ syntax
        # and breaks compilation (idea 3 in the fix plan).
        if prev_id is None:
            rule_opts = "inner sep=0, minimum height=0"
            rule_tail = " at (0,0)"
        elif horizontal:
            rule_opts = (
                f"inner sep=0, minimum height=0, "
                f"right={_SECTION_GAP_BEFORE_CM}cm of {prev_id}"
            )
            rule_tail = ""
        else:
            rule_opts = (
                f"inner sep=0, minimum height=0, "
                f"above={_SECTION_GAP_BEFORE_CM}cm of {prev_id}"
            )
            rule_tail = ""
        out = (
            rf"\node[{rule_opts}] ({rule_id}){rule_tail} {{}};" "\n"
        ) + rule_stroke
        anchor_id = node.id
        title_id = f"{node.id}_title" if horizontal else node.id
        if horizontal:
            # node.id = invisible anchor on the flow line (rule center)
            out += (
                rf"\node[right=0pt of {rule_id}, inner sep=0, "
                rf"minimum size=0] ({anchor_id}) {{}};" "\n"
            )
            if node.label:
                out += (
                    rf"\node[above=6pt of {rule_id}, "
                    rf"font=\sffamily\normalsize\bfseries, text=clrborder] "
                    rf"({title_id}) {{{node.label}}};" "\n"
                )
        else:
            if node.label:
                out += (
                    rf"\node[above=4pt of {rule_id}, "
                    rf"font=\sffamily\normalsize\bfseries, text=clrborder] "
                    rf"({anchor_id}) {{{node.label}}};" "\n"
                )
            else:
                out += (
                    rf"\node[above=4pt of {rule_id}, inner sep=0, "
                    rf"minimum size=0] ({anchor_id}) {{}};" "\n"
                )
        return out

    width = node.size_hint[0] if node.size_hint else 3.8
    height = node.size_hint[1] if node.size_hint else 0.85
    # Phase 5 rounding: never emit >2 decimal places — fixes the
    # "1.8666666666666667cm" regression from width_from_dim.
    width = round(width, 2)
    height = round(height, 2)

    section_bump = prev_shape == "section_rule"
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
            if section_bump:
                kwargs["node_distance"] = 0.4 + _SECTION_GAP_AFTER_CM
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
        if section_bump:
            block_kwargs["node_distance"] = 0.4 + _SECTION_GAP_AFTER_CM
    else:
        block_kwargs["above_of"] = prev_id
        if section_bump:
            block_kwargs["node_distance"] = 0.4 + _SECTION_GAP_AFTER_CM
    return flat_block(node.id, node.label, node.role, **block_kwargs)


def _safe_skip_xshift(graph: IRGraph, edge, order_index: dict[str, int],
                       safety: float = 0.6, default: float = 2.4) -> float:
    """Compute xshift that clears the widest block between src and dst.

    Fix for Task 4 (see REFACTOR_HANDOFF.md): the legacy hardcoded 2.2cm
    overlapped group frames when the residual contained a wide block
    (e.g. 4.2cm attention block in Swin). We take ``max_width / 2`` plus
    a safety margin; this is also rounded to 2 decimals to preserve the
    "no fractional cm" invariant.
    """
    src_i = order_index.get(edge.src)
    dst_i = order_index.get(edge.dst)
    if src_i is None or dst_i is None or dst_i == src_i:
        return default
    lo, hi = (src_i, dst_i) if src_i < dst_i else (dst_i, src_i)
    widths: list[float] = []
    for nid in graph.order[lo + 1:hi + 1]:
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
    prev_shape: str | None = None
    for i, nid in enumerate(graph.order):
        node = graph.nodes[nid]
        parts.append(_emit_node(node, prev_id=prev, is_first=(i == 0),
                                 direction=direction,
                                 prev_shape=prev_shape))
        prev = nid
        prev_shape = node.shape

    # Edges. In horizontal mode data arrows use east→west, skip arrows
    # loop north/south instead of east/west.
    order_index = {nid: i for i, nid in enumerate(graph.order)}
    for edge in graph.edges:
        # Skip self-loops: the IR occasionally produces edges where src == dst
        # (e.g. when a passthrough op like Dropout reuses the cursor). Drawing
        # them emits zero-length arrows that render as smudges on the block.
        if edge.src == edge.dst:
            continue
        src_i = order_index.get(edge.src)
        dst_i = order_index.get(edge.dst)
        # Idea 4: edges that go BACKWARD in flow order (dst positioned before
        # src) — e.g. RNN recurrence h_t→entry — must never be drawn as a
        # straight line: they would stab through every block in between.
        # Route them as a skip arrow on the LEFT side (opposite of the normal
        # residual skip which routes right), so they read as a loop.
        is_backward = (
            src_i is not None and dst_i is not None and dst_i < src_i
        )
        if edge.kind == "skip":
            xshift = _safe_skip_xshift(graph, edge, order_index)
            parts.append(flat_skip_arrow(edge.src, edge.dst, xshift=xshift))
        elif is_backward:
            # swap src/dst endpoints so the arrow head sits on the incoming
            # side; route via the left gutter.
            xshift = _safe_skip_xshift(graph, edge, order_index,
                                        safety=0.8, default=2.6)
            parts.append(flat_skip_arrow(edge.src, edge.dst,
                                           xshift=xshift, direction="left"))
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

    # IO stubs: short arrow pointing INTO input (from outside) and OUT of
    # output (to outside). When the endpoint block already contains the word
    # "Input"/"Output", the label is redundant and we omit the arrow entirely
    # — this fixes the zero-length self-arrow that rendered as a visual
    # smudge on the block edge (ideas 1, 2 in the fix plan).
    # Start offset directions: inputs come from "outside" (below in vertical,
    # left in horizontal) and point TOWARD the block.
    if direction == "vertical":
        in_shift = "yshift=-0.6cm"
        out_shift = "yshift=0.6cm"
    else:
        in_shift = "xshift=-0.6cm"
        out_shift = "xshift=0.6cm"

    if graph.input_node and not _redundant(graph.input_node, "Input"):
        parts.append(
            rf"\draw[arrow] ([{in_shift}]{graph.input_node}.{io_start_anchor}) "
            rf"-- node[right, yshift=-2pt, font=\sffamily\scriptsize, "
            rf"text=clrtext] {{Input}} ({graph.input_node}.{io_start_anchor});"
            "\n"
        )
    if graph.output_node and not _redundant(graph.output_node, "Output"):
        parts.append(
            rf"\draw[arrow] ({graph.output_node}.{io_end_anchor}) "
            rf"-- node[right, yshift=-2pt, font=\sffamily\scriptsize, "
            rf"text=clrtext] {{Output}} "
            rf"([{out_shift}]{graph.output_node}.{io_end_anchor});" "\n"
        )

    # Group frames (fit boxes around children) with optional ×N badge.
    # Idea 6+7: increase inner padding so children don't touch the border,
    # and soften the stroke (densely dashed + lower opacity) so the frame
    # recedes visually and doesn't compete with arrows. Asymmetric xsep
    # gives residual skiparrows room to route outside the frame.
    for gid, group in graph.groups.items():
        if not group.children:
            continue
        fit = " ".join(f"({c})" for c in group.children)
        parts.append(
            rf"\node[draw=clrgroup_frame!60, rounded corners=8pt, "
            rf"densely dashed, line width=0.6pt, "
            rf"fill=clrgroup_fill, fill opacity=0.35, "
            rf"inner ysep=0.55cm, inner xsep=0.7cm, fit={fit}] "
            rf"({gid}) {{}};" "\n"
        )
        if group.title:
            parts.append(
                rf"\node[above=4pt of {gid}.north, font=\sffamily\small\bfseries, "
                rf"text=clrborder] ({gid}_title) {{{group.title}}};" "\n"
            )
        if group.repeat_count and group.repeat_count > 1:
            if direction == "horizontal":
                badge_pos = (
                    rf"below=10pt of {gid}.south, anchor=north"
                )
            else:
                badge_pos = (
                    rf"right=12pt of {gid}.east, anchor=west"
                )
            parts.append(
                rf"\node[{badge_pos}, "
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
    output = "".join(parts)
    _lint_tikz(output)
    return output


# ---------------------------------------------------------------------------
# Output linter (idea 10)
# ---------------------------------------------------------------------------

import re as _re

# Zero-length self-arrow: \draw[...] (X.anchor) -- (X.anchor);
_RE_SELF_ARROW = _re.compile(
    r"\\draw\[[^\]]*\]\s*\(([^.)]+)\.(\w+)\)\s*--\s*\(\1\.\2\)"
)
# Positioning (above=/below=/left=/right=) written AFTER the closing bracket
# of a \node — invalid TikZ syntax. Skip "at (x,y)" which IS valid outside [].
_RE_BAD_POS = _re.compile(
    r"\\node\[[^\]]*\]\s*\([^)]+\)\s+(above|below|left|right)\s*="
)


def _lint_tikz(tex: str) -> None:
    """Fail fast on known emission bugs.

    Guards against regressions: zero-length self-arrows and
    out-of-bracket positioning options. Both were recurring bugs in
    generated output before the fix campaign.
    """
    m = _RE_SELF_ARROW.search(tex)
    if m:
        raise AssertionError(
            f"emit produced zero-length self-arrow for node "
            f"'{m.group(1)}.{m.group(2)}' — this renders as a visual smudge"
        )
    m = _RE_BAD_POS.search(tex)
    if m:
        raise AssertionError(
            f"emit produced '{m.group(1)}=' outside \\node[...] brackets — "
            f"invalid TikZ positioning syntax"
        )
