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
    flat_separator, flat_io_arrow, flat_separator_label, flat_section_header,
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
class PositionalEncoding:
    encoding_type: Literal["rope", "learned", "sinusoidal", "alibi"] = "learned"
    d_model: int = 512
    label: str | None = None


@dataclass
class TransformerBlock:
    attention: Literal["self", "masked", "cross", "local", "global",
                       "gqa", "mqa"] = "self"
    norm: Literal["pre_ln", "post_ln"] = "pre_ln"
    ffn: Literal["gelu", "geglu", "swiglu", "relu"] = "gelu"
    d_ff: int = 2048
    heads: int = 8
    kv_heads: int | None = None  # for GQA/MQA
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


# --- Regularization ---

@dataclass
class Dropout:
    rate: float = 0.1
    label: str | None = None


@dataclass
class Activation:
    function: Literal["relu", "gelu", "swish", "mish", "sigmoid",
                      "tanh", "softmax", "leaky_relu", "selu"] = "relu"
    label: str | None = None


# --- Normalization variants ---

@dataclass
class BatchNorm:
    label: str = "BatchNorm"


@dataclass
class RMSNorm:
    label: str = "RMSNorm"


@dataclass
class AdaptiveLayerNorm:
    label: str = "AdaLN"
    condition: str = "timestep"


# --- Residual / Bottleneck composites ---

@dataclass
class ResidualBlock:
    filters: int = 64
    kernel_size: int = 3
    label: str | None = None


@dataclass
class BottleneckBlock:
    filters: int = 64
    expansion: int = 4
    label: str | None = None


# --- GAN ---

@dataclass
class Generator:
    channels: int = 64
    label: str = "Generator"


@dataclass
class Discriminator:
    channels: int = 64
    label: str = "Discriminator"


# --- VAE / Autoencoder ---

@dataclass
class EncoderBlock:
    d_model: int = 512
    label: str = "Encoder"


@dataclass
class DecoderBlock:
    d_model: int = 512
    label: str = "Decoder"


@dataclass
class SamplingLayer:
    label: str = r"Sampling ($\mu, \sigma$)"


# --- Diffusion ---

@dataclass
class UNetBlock:
    filters: int = 64
    with_attention: bool = True
    label: str | None = None


@dataclass
class NoiseHead:
    label: str = "Noise Prediction"


# --- State Space Models (Mamba) ---

@dataclass
class MambaBlock:
    d_model: int = 512
    d_state: int = 16
    label: str | None = None


@dataclass
class SelectiveSSM:
    d_model: int = 512
    label: str = "Selective SSM"


# --- Mixture of Experts ---

@dataclass
class MoELayer:
    num_experts: int = 8
    top_k: int = 2
    d_ff: int = 2048
    label: str | None = None


@dataclass
class Router:
    num_experts: int = 8
    top_k: int = 2
    label: str | None = None


@dataclass
class Expert:
    d_ff: int = 2048
    label: str | None = None


# --- Graph Neural Networks ---

@dataclass
class GraphConv:
    channels: int = 64
    label: str = "Graph Conv"


@dataclass
class MessagePassing:
    aggregation: Literal["sum", "mean", "max"] = "sum"
    label: str | None = None


@dataclass
class GraphAttention:
    heads: int = 4
    label: str = "Graph Attention"


@dataclass
class GraphPooling:
    pool_type: Literal["max", "mean", "topk"] = "mean"
    label: str | None = None


@dataclass
class CustomBlock:
    """Freeform block with explicit text and color role."""
    text: str
    color_role: str = "dense"
    label: str | None = None


# --- Structural / Visual ---

@dataclass
class Separator:
    """Prominent labeled divider line between architecture sections."""
    label: str = ""
    style: Literal["line", "thick", "double"] = "thick"


@dataclass
class SectionHeader:
    """Section title (e.g. 'Encoder', 'Decoder') — text + thin rule, no box."""
    title: str = ""
    subtitle: str = ""


# Union of all layer types
Layer = (
    Embedding | PositionalEncoding | TransformerBlock | ConvBlock | DenseLayer
    | ClassificationHead | FourierBlock | Dropout | Activation | BatchNorm
    | RMSNorm | AdaptiveLayerNorm | ResidualBlock | BottleneckBlock
    | Generator | Discriminator | EncoderBlock | DecoderBlock | SamplingLayer
    | UNetBlock | NoiseHead | MambaBlock | SelectiveSSM | MoELayer | Router
    | Expert | GraphConv | MessagePassing | GraphAttention | GraphPooling
    | CustomBlock | Separator | SectionHeader
)


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
# Auto-grouping: detect consecutive runs AND repeating patterns
# ---------------------------------------------------------------------------

@dataclass
class _Group:
    start: int
    end: int  # exclusive
    count: int          # number of repeat units
    layer_type: type    # single type for simple runs
    pattern_len: int = 1  # 1 = single-type run, >1 = multi-type pattern


def _detect_groups(layers: list[Layer]) -> list[_Group]:
    """Find consecutive runs of same-type layers AND repeating multi-type patterns.

    E.g. [A, B, A, B, A, B] → pattern [A, B] × 3 (pattern_len=2, count=3).
    Structural layers (Separator, SectionHeader) break groups.
    """
    structural = (Separator, SectionHeader)
    groups: list[_Group] = []

    # Phase 1: detect single-type runs
    i = 0
    covered = set()
    while i < len(layers):
        if isinstance(layers[i], structural):
            i += 1
            continue
        layer_type = type(layers[i])
        j = i + 1
        while j < len(layers) and type(layers[j]) is layer_type:
            j += 1
        if j - i > 1:
            groups.append(_Group(start=i, end=j, count=j - i, layer_type=layer_type))
            covered.update(range(i, j))
        i = j

    # Phase 2: detect repeating patterns of length 2-4 in uncovered regions
    i = 0
    while i < len(layers):
        if i in covered or isinstance(layers[i], structural):
            i += 1
            continue
        found_pattern = False
        for pat_len in (2, 3, 4):
            if i + pat_len > len(layers):
                break
            pattern = [type(layers[i + k]) for k in range(pat_len)]
            if any(issubclass(t, structural) for t in pattern):
                continue
            repeats = 1
            pos = i + pat_len
            while pos + pat_len <= len(layers):
                next_pat = [type(layers[pos + k]) for k in range(pat_len)]
                if next_pat == pattern:
                    repeats += 1
                    pos += pat_len
                else:
                    break
            if repeats >= 2:
                total = repeats * pat_len
                groups.append(_Group(
                    start=i, end=i + total, count=repeats,
                    layer_type=type(layers[i]), pattern_len=pat_len,
                ))
                covered.update(range(i, i + total))
                i = i + total
                found_pattern = True
                break
        if not found_pattern:
            i += 1

    groups.sort(key=lambda g: g.start)
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
        "gqa": f"GQA ({block.kv_heads or block.heads // 4}kv)",
        "mqa": "Multi-Query Attn",
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

        lines.append(flat_block(n1, "LayerNorm", "norm", above_of=entry, node_distance=0.12,
                                width=3.0, height=0.6))
        lines.append(flat_arrow(entry, n1))
        attn_opacity = 1.0 if block.attention == "global" else 0.65
        attn_height = 1.1 if block.attention == "global" else 0.95
        lines.append(flat_block(at, attn_label, attn_color, above_of=n1, node_distance=node_dist,
                                width=4.2, height=attn_height, opacity=attn_opacity))
        lines.append(flat_arrow(n1, at))
        lines.append(flat_dim_label(f"{block.heads}h", at, side="left", distance=0.15))

        lines.append(flat_add_circle(a1, above_of=at, node_distance=0.25))
        lines.append(flat_arrow(at, a1))
        lines.append(flat_skip_arrow(entry, a1))  # skip from block entry, not from prev

        lines.append(flat_block(n2, "LayerNorm", "norm", above_of=a1, node_distance=node_dist,
                                width=3.0, height=0.6))
        lines.append(flat_arrow(a1, n2))
        lines.append(flat_block(ff, ffn_label, "ffn", above_of=n2, node_distance=node_dist,
                                width=3.8, height=0.85))
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
    # For pattern groups (pattern_len > 1), show_n counts pattern repetitions, not layers
    visible: dict[int, bool] = {}
    for i in range(len(layers)):
        grp = _find_group(i, groups)
        if grp is None:
            visible[i] = True
        else:
            grp_id = id(grp)
            if grp_id not in group_show_count:
                group_show_count[grp_id] = 0
            # Which repetition does this layer belong to?
            offset_in_group = i - grp.start
            rep_index = offset_in_group // grp.pattern_len
            if rep_index < show_n:
                visible[i] = True
            else:
                visible[i] = False
            # Count unique repetitions shown
            if offset_in_group % grp.pattern_len == 0:
                group_show_count[grp_id] += 1

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
            parts.append(flat_block(
                nid, label, "embed",
                position="(0,0)" if prev is None else None,
                above_of=prev, node_distance=0.6 if prev else 0,
            ))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.d_model), nid, side="left"))
            prev = nid
            # If rope=True and next layer is NOT PositionalEncoding, auto-add RoPE block
            next_is_pos = (i + 1 < len(layers) and isinstance(layers[i + 1], PositionalEncoding))
            if layer.rope and not next_is_pos:
                rope_nid = f"layer_{node_idx}_rope"
                parts.append(flat_block(
                    rope_nid, "RoPE", "residual",
                    above_of=nid, node_distance=0.25,
                    width=3.0, height=0.65, opacity=0.5,
                ))
                parts.append(flat_arrow(nid, rope_nid))
                prev = rope_nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        elif isinstance(layer, PositionalEncoding):
            nid = f"layer_{node_idx}"
            type_labels = {
                "rope": "RoPE",
                "learned": "Learned Pos. Enc.",
                "sinusoidal": "Sinusoidal Pos. Enc.",
                "alibi": "ALiBi",
            }
            label = layer.label or type_labels.get(layer.encoding_type, "Pos. Encoding")
            parts.append(flat_block(
                nid, label, "residual",
                above_of=prev, node_distance=0.25,
                width=3.0, height=0.65, opacity=0.5,
            ))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.d_model), nid, side="left"))
            prev = nid

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

        # --- Regularization / Activation ---

        elif isinstance(layer, Dropout):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Dropout ({layer.rate})"
            parts.append(flat_block(nid, label, "border", above_of=prev,
                                    node_distance=0.2, width=3.0, height=0.55, opacity=0.25))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        elif isinstance(layer, Activation):
            nid = f"layer_{node_idx}"
            fn_labels = {"relu": "ReLU", "gelu": "GELU", "swish": "Swish/SiLU",
                         "mish": "Mish", "sigmoid": "Sigmoid", "tanh": "Tanh",
                         "softmax": "Softmax", "leaky_relu": "LeakyReLU", "selu": "SELU"}
            label = layer.label or fn_labels.get(layer.function, layer.function)
            parts.append(flat_block(nid, label, "ffn", above_of=prev,
                                    node_distance=0.2, width=3.0, height=0.6, opacity=0.6))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        # --- Normalization variants ---

        elif isinstance(layer, (BatchNorm, RMSNorm)):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "norm", above_of=prev,
                                    node_distance=0.2, width=3.4, height=0.65))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        elif isinstance(layer, AdaptiveLayerNorm):
            nid = f"layer_{node_idx}"
            label = f"{layer.label} ({layer.condition})"
            parts.append(flat_block(nid, label, "norm", above_of=prev,
                                    node_distance=0.3, width=3.4, height=0.7))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        # --- Residual / Bottleneck composites ---

        elif isinstance(layer, ResidualBlock):
            nid = f"layer_{node_idx}"
            label = layer.label or f"ResBlock ({layer.filters})"
            entry = f"{nid}_in"
            conv1 = f"{nid}_c1"
            conv2 = f"{nid}_c2"
            add = f"{nid}_add"
            parts.append(
                rf"\node[inner sep=0, minimum size=0, above=0.4cm of {prev}] ({entry}) {{}};" "\n"
            )
            if prev:
                parts.append(flat_arrow(prev, entry))
            parts.append(flat_block(conv1, f"Conv{layer.kernel_size}x{layer.kernel_size} + BN + ReLU",
                                    "attention", above_of=entry, node_distance=0.12, height=0.7))
            parts.append(flat_arrow(entry, conv1))
            parts.append(flat_block(conv2, f"Conv{layer.kernel_size}x{layer.kernel_size} + BN",
                                    "attention_alt", above_of=conv1, node_distance=0.25, height=0.7))
            parts.append(flat_arrow(conv1, conv2))
            parts.append(flat_dim_label(str(layer.filters), conv2, side="left"))
            parts.append(flat_add_circle(add, above_of=conv2, node_distance=0.25))
            parts.append(flat_arrow(conv2, add))
            parts.append(flat_skip_arrow(entry, add))
            prev = add
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].extend([conv1, conv2, add])

        elif isinstance(layer, BottleneckBlock):
            nid = f"layer_{node_idx}"
            entry = f"{nid}_in"
            c1 = f"{nid}_1x1a"
            c2 = f"{nid}_3x3"
            c3 = f"{nid}_1x1b"
            add = f"{nid}_add"
            mid = layer.filters
            out = layer.filters * layer.expansion
            parts.append(
                rf"\node[inner sep=0, minimum size=0, above=0.4cm of {prev}] ({entry}) {{}};" "\n"
            )
            if prev:
                parts.append(flat_arrow(prev, entry))
            parts.append(flat_block(c1, f"1x1 Conv ({mid})", "attention_alt",
                                    above_of=entry, node_distance=0.12, height=0.6))
            parts.append(flat_arrow(entry, c1))
            parts.append(flat_block(c2, f"3x3 Conv ({mid})", "attention",
                                    above_of=c1, node_distance=0.2, height=0.7))
            parts.append(flat_arrow(c1, c2))
            parts.append(flat_block(c3, f"1x1 Conv ({out})", "attention_alt",
                                    above_of=c2, node_distance=0.2, height=0.6))
            parts.append(flat_arrow(c2, c3))
            parts.append(flat_add_circle(add, above_of=c3, node_distance=0.25))
            parts.append(flat_arrow(c3, add))
            parts.append(flat_skip_arrow(entry, add))
            prev = add
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].extend([c1, c2, c3, add])

        # --- GAN ---

        elif isinstance(layer, Generator):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "embed", above_of=prev,
                                    node_distance=0.4, height=1.0))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.channels), nid, side="left"))
            prev = nid

        elif isinstance(layer, Discriminator):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "physics", above_of=prev,
                                    node_distance=0.4, height=1.0))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.channels), nid, side="left"))
            prev = nid

        # --- VAE / Autoencoder ---

        elif isinstance(layer, EncoderBlock):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "attention", above_of=prev,
                                    node_distance=0.4, height=1.0))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.d_model), nid, side="left"))
            prev = nid

        elif isinstance(layer, DecoderBlock):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "attention_alt", above_of=prev,
                                    node_distance=0.4, height=1.0))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.d_model), nid, side="left"))
            prev = nid

        elif isinstance(layer, SamplingLayer):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "residual", above_of=prev,
                                    node_distance=0.3, width=3.2, height=0.7, opacity=0.6))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        # --- Diffusion ---

        elif isinstance(layer, UNetBlock):
            nid = f"layer_{node_idx}"
            label = layer.label or f"UNet Block ({layer.filters})"
            if layer.with_attention:
                label += " + Attn"
            parts.append(flat_block(nid, label, "spectral", above_of=prev,
                                    node_distance=0.4, height=0.9))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.filters), nid, side="left"))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        elif isinstance(layer, NoiseHead):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "physics", above_of=prev,
                                    node_distance=0.4, width=3.4, height=0.8))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        # --- State Space Models (Mamba) ---

        elif isinstance(layer, MambaBlock):
            nid = f"layer_{node_idx}"
            label = layer.label or "Mamba Block"
            entry = f"{nid}_in"
            proj = f"{nid}_proj"
            conv = f"{nid}_conv"
            ssm = f"{nid}_ssm"
            out = f"{nid}_out"
            parts.append(
                rf"\node[inner sep=0, minimum size=0, above=0.5cm of {prev}] ({entry}) {{}};" "\n"
            )
            if prev:
                parts.append(flat_arrow(prev, entry))
            parts.append(flat_block(proj, "Linear Proj", "dense", above_of=entry,
                                    node_distance=0.12, height=0.65))
            parts.append(flat_arrow(entry, proj))
            parts.append(flat_block(conv, "Conv1D", "attention_alt", above_of=proj,
                                    node_distance=0.25, height=0.65))
            parts.append(flat_arrow(proj, conv))
            parts.append(flat_block(ssm, "Selective SSM", "spectral", above_of=conv,
                                    node_distance=0.25, height=0.85))
            parts.append(flat_arrow(conv, ssm))
            parts.append(flat_dim_label(str(layer.d_model), ssm, side="left"))
            parts.append(flat_block(out, "Linear Out", "dense", above_of=ssm,
                                    node_distance=0.25, height=0.65))
            parts.append(flat_arrow(ssm, out))
            prev = out
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].extend([proj, conv, ssm, out])

        elif isinstance(layer, SelectiveSSM):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "spectral", above_of=prev,
                                    node_distance=0.4, height=0.9))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.d_model), nid, side="left"))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        # --- Mixture of Experts ---

        elif isinstance(layer, MoELayer):
            nid = f"layer_{node_idx}"
            label = layer.label or f"MoE (top-{layer.top_k}/{layer.num_experts})"
            router = f"{nid}_router"
            experts = f"{nid}_experts"
            parts.append(flat_block(router, f"Router (top-{layer.top_k})", "output",
                                    above_of=prev, node_distance=0.4, width=3.0, height=0.7))
            if prev:
                parts.append(flat_arrow(prev, router))
            parts.append(flat_block(experts, f"{layer.num_experts} Experts (FFN {layer.d_ff})",
                                    "dense", above_of=router, node_distance=0.25, height=0.9))
            parts.append(flat_arrow(router, experts))
            prev = experts
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].extend([router, experts])

        elif isinstance(layer, Router):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Router (top-{layer.top_k}/{layer.num_experts})"
            parts.append(flat_block(nid, label, "output", above_of=prev,
                                    node_distance=0.3, width=3.0, height=0.7))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        elif isinstance(layer, Expert):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Expert FFN ({layer.d_ff})"
            parts.append(flat_block(nid, label, "dense", above_of=prev,
                                    node_distance=0.3, height=0.8))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        # --- Graph Neural Networks ---

        elif isinstance(layer, GraphConv):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "attention_alt", above_of=prev,
                                    node_distance=0.4, height=0.85))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(str(layer.channels), nid, side="left"))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        elif isinstance(layer, MessagePassing):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Message Passing ({layer.aggregation})"
            parts.append(flat_block(nid, label, "spectral", above_of=prev,
                                    node_distance=0.4, height=0.85))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        elif isinstance(layer, GraphAttention):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "attention", above_of=prev,
                                    node_distance=0.4, height=0.9))
            if prev:
                parts.append(flat_arrow(prev, nid))
            parts.append(flat_dim_label(f"{layer.heads}h", nid, side="left"))
            prev = nid
            if grp_id is not None:
                group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
                group_frame_data[grp_id]["nodes"].append(nid)

        elif isinstance(layer, GraphPooling):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Graph {layer.pool_type.title()}Pool"
            parts.append(flat_block(nid, label, "ffn", above_of=prev,
                                    node_distance=0.3, width=3.0, height=0.65))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        # --- Freeform ---

        elif isinstance(layer, CustomBlock):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.text, layer.color_role,
                                    above_of=prev, node_distance=0.4))
            if prev:
                parts.append(flat_arrow(prev, nid))
            prev = nid

        # --- Structural ---

        elif isinstance(layer, Separator):
            nid = f"sep_label_{node_idx}"
            if prev:
                parts.append(flat_separator_label(nid, layer.label, prev,
                                                   style=layer.style))
                prev = nid
            # Separator is not added to groups

        elif isinstance(layer, SectionHeader):
            nid = f"section_{node_idx}"
            if prev:
                parts.append(flat_section_header(nid, layer.title, prev,
                                                  subtitle=layer.subtitle))
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
