"""
Semantic DSL for describing neural network architectures.

Users describe *what* the architecture is, not *where* to place boxes.
The DSL auto-computes layout, groups repeated blocks, and renders via
the flat 2D renderer with themed colors.

Usage:
    arch = Architecture("ModernBERT", theme="modern")
    arch.add(Embedding(768, rope=True))
    for i in range(22):
        attn = "global" if (i+1) % 3 == 0 else "local"
        arch.add(TransformerBlock(attention=attn, norm="pre_ln", ffn="geglu", heads=12))
    arch.add(ClassificationHead())
    tex = arch.render()  # returns complete LaTeX string
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .themes import get_theme, Theme
from .flat_renderer import (
    flat_head, flat_colors, flat_begin, flat_end,
    flat_block, flat_arrow, flat_skip_arrow, flat_add_circle,
    group_frame, flat_title, flat_side_label, flat_dim_label,
    flat_separator, flat_io_arrow,
)


# ---------------------------------------------------------------------------
# Layer dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Embedding:
    d_model: int = 512
    rope: bool = False
    label: str = "Embedding"


@dataclass
class TransformerBlock:
    attention: Literal["self", "masked", "cross", "local", "global"] = "self"
    norm: Literal["pre_ln", "post_ln"] = "pre_ln"
    ffn: Literal["gelu", "geglu", "swiglu", "relu"] = "gelu"
    d_ff: int = 2048
    heads: int = 8
    d_model: int = 512
    label: str | None = None


@dataclass
class ConvBlock:
    filters: int = 64
    kernel_size: int = 3
    activation: str = "relu"
    pool: str | None = "max"
    label: str | None = None


@dataclass
class DenseLayer:
    units: int = 256
    activation: str = "relu"
    label: str | None = None


@dataclass
class ClassificationHead:
    n_classes: int = 10
    label: str = "Output"


@dataclass
class FourierBlock:
    modes: int = 16
    width: int = 64
    label: str | None = None


@dataclass
class CustomBlock:
    """Freeform block with explicit text and color role."""
    text: str
    color_role: str = "dense"
    label: str | None = None


Layer = Embedding | TransformerBlock | ConvBlock | DenseLayer | ClassificationHead | FourierBlock | CustomBlock


# ---------------------------------------------------------------------------
# Architecture container
# ---------------------------------------------------------------------------

class Architecture:
    """Declarative neural network architecture builder."""

    def __init__(
        self,
        name: str,
        theme: str = "modern",
        layout: Literal["vertical", "horizontal"] = "vertical",
    ):
        self.name = name
        self.theme_name = theme
        self.layout = layout
        self.layers: list[Layer] = []

    def add(self, layer: Layer) -> Architecture:
        self.layers.append(layer)
        return self

    def render(self, show_n: int = 4) -> str:
        """Render the architecture to a complete LaTeX string.

        Args:
            show_n: For repeated blocks, show this many explicitly
                    and collapse the rest with a ×N bracket.
        """
        theme = get_theme(self.theme_name)
        groups = _detect_groups(self.layers)
        return _render_vertical(self.name, self.layers, groups, theme, show_n)

    def render_to_file(self, path: str, show_n: int = 4) -> None:
        with open(path, "w") as f:
            f.write(self.render(show_n=show_n))


# ---------------------------------------------------------------------------
# Auto-grouping: detect consecutive runs of same-type layers
# ---------------------------------------------------------------------------

@dataclass
class _Group:
    start: int
    end: int  # exclusive
    count: int
    layer_type: type


def _detect_groups(layers: list[Layer]) -> list[_Group]:
    """Find consecutive runs of the same layer type for ×N grouping."""
    groups = []
    i = 0
    while i < len(layers):
        layer_type = type(layers[i])
        j = i + 1
        while j < len(layers) and type(layers[j]) is layer_type:
            j += 1
        if j - i > 1:
            groups.append(_Group(start=i, end=j, count=j - i, layer_type=layer_type))
        i = j
    return groups


def _find_group(idx: int, groups: list[_Group]) -> _Group | None:
    for g in groups:
        if g.start <= idx < g.end:
            return g
    return None


# ---------------------------------------------------------------------------
# Vertical renderer
# ---------------------------------------------------------------------------

def _attention_label(block: TransformerBlock) -> str:
    labels = {
        "self": "Self-Attention",
        "masked": "Masked Self-Attention",
        "cross": "Cross-Attention",
        "local": "Local Attention",
        "global": "Global Attention",
    }
    return labels.get(block.attention, "Attention")


def _ffn_label(block: TransformerBlock) -> str:
    labels = {"gelu": "FFN (GeLU)", "geglu": "FFN (GeGLU)", "swiglu": "FFN (SwiGLU)", "relu": "FFN (ReLU)"}
    return labels.get(block.ffn, "FFN")


def _render_transformer_block(
    block: TransformerBlock,
    prefix: str,
    prev: str,
    theme: Theme,
    node_dist: float = 0.35,
) -> tuple[list[str], str, list[str]]:
    """Render one TransformerBlock. Returns (tikz_lines, last_node_name, all_node_names)."""
    lines: list[str] = []
    nodes: list[str] = []
    attn_color = "attention" if block.attention == "global" else "attention_alt"
    attn_label = _attention_label(block)
    ffn_label = _ffn_label(block)
    is_pre_ln = block.norm == "pre_ln"

    if is_pre_ln:
        # Pre-LN: Entry → Norm → Attn → Add(+skip) → Norm → FFN → Add(+skip)
        entry = f"{prefix}_in"
        n1 = f"{prefix}_ln1"
        at = f"{prefix}_attn"
        a1 = f"{prefix}_add1"
        n2 = f"{prefix}_ln2"
        ff = f"{prefix}_ffn"
        a2 = f"{prefix}_add2"

        # Invisible entry anchor — skip connections start here, not from prev
        entry_dist = node_dist + 0.15
        lines.append(
            rf"\node[inner sep=0, minimum size=0, above={entry_dist}cm of {prev}] ({entry}) {{}};" "\n"
        )
        lines.append(flat_arrow(prev, entry))

        lines.append(flat_block(n1, "LayerNorm", "norm", above_of=entry, node_distance=0.12))
        lines.append(flat_arrow(entry, n1))
        attn_opacity = 1.0 if block.attention == "global" else 0.65
        attn_height = 1.05 if block.attention == "global" else 0.9
        lines.append(flat_block(at, attn_label, attn_color, above_of=n1, node_distance=node_dist,
                                width=3.8, height=attn_height, opacity=attn_opacity))
        lines.append(flat_arrow(n1, at))
        lines.append(flat_dim_label(f"{block.heads}h", at, side="left", distance=0.15))

        lines.append(flat_add_circle(a1, above_of=at, node_distance=0.25))
        lines.append(flat_arrow(at, a1))
        lines.append(flat_skip_arrow(entry, a1))  # skip from block entry, not from prev

        lines.append(flat_block(n2, "LayerNorm", "norm", above_of=a1, node_distance=node_dist))
        lines.append(flat_arrow(a1, n2))
        lines.append(flat_block(ff, ffn_label, "ffn", above_of=n2, node_distance=node_dist))
        lines.append(flat_arrow(n2, ff))
        lines.append(flat_dim_label(str(block.d_ff), ff, side="left", distance=0.15))

        lines.append(flat_add_circle(a2, above_of=ff, node_distance=0.25))
        lines.append(flat_arrow(ff, a2))
        lines.append(flat_skip_arrow(a1, a2))  # skip from after first add

        # exclude entry from group frame nodes (it's a spacer near prev layer)
        nodes = [n1, at, a1, n2, ff, a2]
        return lines, a2, nodes
    else:
        # Post-LN: Entry → Attn → Add(+skip) → Norm → FFN → Add(+skip) → Norm
        entry = f"{prefix}_in"
        at = f"{prefix}_attn"
        a1 = f"{prefix}_add1"
        n1 = f"{prefix}_ln1"
        ff = f"{prefix}_ffn"
        a2 = f"{prefix}_add2"
        n2 = f"{prefix}_ln2"

        lines.append(
            rf"\node[inner sep=0, minimum size=0, above={node_dist}cm of {prev}] ({entry}) {{}};" "\n"
        )
        lines.append(flat_arrow(prev, entry))

        lines.append(flat_block(at, attn_label, attn_color, above_of=entry, node_distance=0.15))
        lines.append(flat_arrow(entry, at))
        lines.append(flat_add_circle(a1, above_of=at, node_distance=0.25))
        lines.append(flat_arrow(at, a1))
        lines.append(flat_skip_arrow(entry, a1))
        lines.append(flat_block(n1, "LayerNorm", "norm", above_of=a1, node_distance=node_dist))
        lines.append(flat_arrow(a1, n1))
        lines.append(flat_block(ff, ffn_label, "ffn", above_of=n1, node_distance=node_dist))
        lines.append(flat_arrow(n1, ff))
        lines.append(flat_add_circle(a2, above_of=ff, node_distance=0.25))
        lines.append(flat_arrow(ff, a2))
        lines.append(flat_skip_arrow(n1, a2))
        lines.append(flat_block(n2, "LayerNorm", "norm", above_of=a2, node_distance=node_dist))
        lines.append(flat_arrow(a2, n2))

        nodes = [at, a1, n1, ff, a2, n2]
        return lines, n2, nodes


def _render_vertical(
    name: str,
    layers: list[Layer],
    groups: list[_Group],
    theme: Theme,
    show_n: int,
) -> str:
    """Render full architecture as vertical stack."""
    parts: list[str] = []
    parts.append(flat_head())
    parts.append(flat_colors(theme))
    parts.append(flat_begin())

    prev = None
    node_idx = 0
    rendered_in_group: dict[int, list[str]] = {}  # group_id → list of node names
    group_show_count: dict[int, int] = {}

    # Pre-compute which layers to show vs collapse
    visible: dict[int, bool] = {}
    for i in range(len(layers)):
        grp = _find_group(i, groups)
        if grp is None:
            visible[i] = True
        else:
            grp_id = id(grp)
            if grp_id not in group_show_count:
                group_show_count[grp_id] = 0
            if group_show_count[grp_id] < show_n:
                visible[i] = True
                group_show_count[grp_id] += 1
            else:
                visible[i] = False

    # Track group frame data
    group_frame_data: dict[int, dict] = {}

    for i, layer in enumerate(layers):
        if not visible[i]:
            grp = _find_group(i, groups)
            if grp:
                grp_id = id(grp)
                if grp_id not in group_frame_data:
                    group_frame_data[grp_id] = {"count": grp.count, "nodes": []}
            continue

        grp = _find_group(i, groups)
        grp_id = id(grp) if grp else None

        if isinstance(layer, Embedding):
            nid = f"layer_{node_idx}"
            label = layer.label
            if layer.rope:
                label += " + RoPE"
            parts.append(flat_block(
                nid, label, "embed",
                position="(0,0)" if prev is None else None,
                above_of=prev, node_distance=0.6 if prev else 0,
            ))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.d_model), nid, side="left"))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        elif isinstance(layer, TransformerBlock):
            # Add separator between consecutive transformer blocks
            if i > 0 and isinstance(layers[i - 1], TransformerBlock) and visible.get(i - 1, False):
                sep_name = f"sep_{node_idx}"
                parts.append(flat_separator(prev, sep_name))

            prefix = f"tb_{node_idx}"
            block_lines, last_node, block_nodes = _render_transformer_block(
                layer, prefix, prev, theme,
            )
            parts.extend(block_lines)
            prev = last_node
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].extend(block_nodes)

        elif isinstance(layer, ConvBlock):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Conv{layer.kernel_size}×{layer.kernel_size}"
            parts.append(flat_block(
                nid, label, "attention",
                above_of=prev, node_distance=0.4,
            ))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.filters), nid, side="left"))
            prev = nid
            if layer.pool:
                pid = f"pool_{node_idx}"
                pool_label = f"{'Max' if layer.pool == 'max' else 'Avg'}Pool"
                parts.append(flat_block(pid, pool_label, "ffn",
                                        above_of=nid, node_distance=0.25,
                                        width=3.0, height=0.6))
                parts.append(flat_arrow(nid, pid))
                prev = pid

        elif isinstance(layer, DenseLayer):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Dense ({layer.units})"
            parts.append(flat_block(nid, label, "dense",
                                    above_of=prev, node_distance=0.4))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        elif isinstance(layer, ClassificationHead):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "output",
                                    above_of=prev, node_distance=0.8))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        elif isinstance(layer, FourierBlock):
            label = layer.label or f"Fourier Layer (modes={layer.modes})"
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, label, "spectral",
                                    above_of=prev, node_distance=0.4))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        elif isinstance(layer, CustomBlock):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.text, layer.color_role,
                                    above_of=prev, node_distance=0.4))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        node_idx += 1

    # Input/output arrows
    if layers:
        parts.append(flat_io_arrow("layer_0", direction="below", label="Input"))
    if prev:
        parts.append(flat_io_arrow(prev, direction="above", label=""))

    # Draw group frames
    for gid, gdata in group_frame_data.items():
        if gdata["nodes"]:
            parts.append(group_frame(
                f"grp_{hash(gid) % 10000}",
                gdata["nodes"],
                repeat=gdata["count"],
                padding=0.7,
            ))

    # Title
    parts.append(flat_title(rf"\textbf{{{name}}}"))

    parts.append(flat_end())
    return "".join(parts)
