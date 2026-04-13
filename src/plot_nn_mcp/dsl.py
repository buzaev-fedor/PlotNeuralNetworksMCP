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

import copy
from dataclasses import dataclass, fields as _dc_fields, is_dataclass as _is_dataclass
from typing import Literal

from .themes import get_theme, Theme
from .flat_renderer import (
    _latex_escape, width_from_dim,
    flat_head, flat_colors, flat_begin, flat_end,
    flat_block, flat_arrow, flat_arrow_h,
    flat_skip_arrow,
    flat_add_circle, flat_op_circle,
    group_frame, flat_title, flat_side_label, flat_dim_label,
    flat_separator, flat_io_arrow, flat_separator_label, flat_section_header,
    flat_cross_attention_arrow,
    olah_gate_node, olah_cell_highway, olah_thin_arrow, olah_curved_arrow,
    olah_label, olah_copy_dot, olah_bus_line, olah_branch_tap,
    OLAH_SIGMA, OLAH_TANH, OLAH_MULTIPLY, OLAH_ADD,
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
    skip_ffn: bool = False  # True = stop after Attn+Add (for MoE replacement)
    label: str | None = None


@dataclass
class ConvBlock:
    filters: int = 64
    kernel_size: int = 3
    activation: str = "relu"
    pool: str | None = "max"
    label: str | None = None


@dataclass
class PatchEmbedding:
    """ViT-style patch embedding: split image into patches + linear projection."""
    patch_size: int = 16
    d_model: int = 768
    label: str | None = None


@dataclass
class MBConvBlock:
    """EfficientNet-style MBConv (inverted residual) block."""
    filters: int = 64
    expansion: int = 6
    kernel_size: int = 3
    se: bool = True  # squeeze-excitation
    label: str | None = None


@dataclass
class SwinBlock:
    """Swin Transformer block with W-MSA / SW-MSA."""
    window_type: Literal["regular", "shifted"] = "regular"
    heads: int = 4
    d_model: int = 96
    window_size: int = 7
    label: str | None = None


@dataclass
class PatchMerging:
    """Swin-style patch merging (spatial downsampling 2x)."""
    d_model: int = 192
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


# --- Recurrent (LSTM / GRU) ---

@dataclass
class LSTMBlock:
    hidden_size: int = 256
    bidirectional: bool = False
    style: Literal["gates", "olah", "compact"] = "gates"
    # "gates" = vertical gate stack (default)
    # "olah" = horizontal conveyor belt (Chris Olah blog style)
    # "compact" = single block (for pipeline views)
    unroll: int = 1  # >1 = show N timesteps side by side
    label: str | None = None


@dataclass
class GRUBlock:
    hidden_size: int = 256
    bidirectional: bool = False
    style: Literal["gates", "olah", "compact"] = "gates"
    unroll: int = 1
    label: str | None = None


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


# --- Parallel / Branching ---

@dataclass
class EncoderDecoder:
    """Encoder-Decoder architecture with cross-attention (Vaswani, T5, Seq2Seq).

    Unlike SideBySide, this explicitly models the encoder→decoder data flow:
    - Separate input embeddings for encoder and decoder
    - Cross-attention connections from encoder to decoder layers
    - Output only from decoder (not merged)
    """
    encoder: list  # list of Layer
    decoder: list  # list of Layer
    encoder_input: list | None = None  # embedding layers for encoder
    decoder_input: list | None = None  # embedding layers for decoder
    cross_attention: Literal["all", "last", "none"] = "all"
    # "all" = every decoder layer receives encoder output (T5, Vaswani)
    # "last" = only last encoder hidden state → first decoder (Sutskever Seq2Seq)
    # "none" = no cross-connections
    encoder_label: str = "Encoder"
    decoder_label: str = "Decoder"
    cross_attention_label: str | None = None  # override arrow label ("Cross-Attn"/"Context")
    encoder_input_label: str = "Input"
    decoder_input_label: str = "Shifted Output"
    label: str = ""


@dataclass
class SPINNBlock:
    """SPINN (Stack-augmented Parser-Interpreter Neural Network, Bowman 2016).

    Renders a three-zone layout: Buffer | Stack | Tracker.
    """
    hidden_size: int = 300
    tracking_size: int = 128
    buffer_tokens: list[str] | None = None  # e.g. ["The", "cat", "sat"]
    label: str = "SPINN"


@dataclass
class SideBySide:
    """Two parallel vertical stacks rendered side by side.

    Used for encoder-decoder (Vaswani), GAN (generator/discriminator),
    DeepONet (branch/trunk), etc.
    """
    left: list  # list of Layer
    right: list  # list of Layer
    left_label: str = "Encoder"
    right_label: str = "Decoder"
    connections: list[tuple[int, int]] | None = None  # cross-connections (left_idx → right_idx)
    label: str = ""


@dataclass
class UNetLevel:
    """One encoder-decoder level of a U-Net architecture.

    width encodes spatial resolution relative to input (1.0 = full res).
    """
    encoder: Layer | None = None
    decoder: Layer | None = None
    filters: int = 64
    resolution: float = 1.0  # relative spatial resolution (1.0 → 0.5 → 0.25 ...)
    label: str | None = None


@dataclass
class Bottleneck:
    """U-Net bottleneck (lowest resolution level)."""
    layer: Layer | None = None
    filters: int = 512
    label: str | None = None


@dataclass
class DetailPanel:
    """Zoom-in panel showing internal structure of a block.

    Renders a compact main block with a dashed leader line to an expanded
    detail view on the right (Vaswani-style Multi-Head Attention detail).
    """
    summary_label: str  # compact label for the main diagram
    detail_layers: list  # list of Layer to render in the detail panel
    summary_color: str = "attention"
    title: str = ""  # optional title above detail panel
    label: str = ""


@dataclass
class ForkLoss:
    """Fork output into multiple loss branches (PINN-style).

    Renders as a branching point after the network with separate
    loss computation paths (e.g., data loss + physics loss).
    """
    branches: list[tuple[str, str, str]]  # list of (label, color_role, annotation)
    # e.g. [("Data Loss", "embed", "$\\mathcal{L}_{data}$"),
    #        ("PDE Residual", "physics", "$\\mathcal{L}_{PDE}$")]
    merge_label: str = r"$\mathcal{L}_{total}$"
    label: str = ""


@dataclass
class BidirectionalFlow:
    """Markov chain / diffusion process: nodes with forward and reverse arrows.

    Used for DDPM (Ho et al. 2020) style visualizations.
    """
    steps: list[str]  # node labels, e.g. ["$x_0$", "$x_1$", "...", "$x_T$"]
    forward_label: str = ""  # label on forward arrows
    reverse_label: str = ""  # label on reverse arrows
    label: str = ""


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
    | PatchEmbedding | MBConvBlock | SwinBlock | PatchMerging
    | Generator | Discriminator | EncoderBlock | DecoderBlock | SamplingLayer
    | UNetBlock | NoiseHead | MambaBlock | SelectiveSSM | LSTMBlock | GRUBlock
    | MoELayer | Router | Expert
    | GraphConv | MessagePassing | GraphAttention | GraphPooling
    | CustomBlock | EncoderDecoder | SPINNBlock
    | SideBySide | BidirectionalFlow | ForkLoss | DetailPanel
    | UNetLevel | Bottleneck
    | Separator | SectionHeader
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
        layout: Literal["vertical", "horizontal", "unet"] = "vertical",
    ):
        if layout not in ("vertical", "horizontal", "unet"):
            raise ValueError(f"Unsupported layout: {layout!r}. Use 'vertical', 'horizontal', or 'unet'.")
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
        if self.layout == "horizontal":
            return _render_horizontal(self.name, self.layers, groups, theme, show_n)
        if self.layout == "unet":
            return _render_unet(self.name, self.layers, theme)
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


def _group_signature(layer) -> tuple:
    """Hashable signature for grouping. Same signature = same visual group.

    Compares all dataclass fields except ``label`` so that layers with
    different configurations (e.g. local vs global attention) are NOT
    merged into a single ×N group.
    """
    if _is_dataclass(layer) and not isinstance(layer, type):
        return (type(layer), *(
            (f.name, getattr(layer, f.name))
            for f in _dc_fields(layer) if f.name != "label"
        ))
    return (type(layer),)


def _detect_groups(layers: list[Layer]) -> list[_Group]:
    """Find consecutive runs of same-type layers AND repeating multi-type patterns.

    E.g. [A, B, A, B, A, B] → pattern [A, B] × 3 (pattern_len=2, count=3).
    Structural layers (Separator, SectionHeader) break groups.
    """
    structural = (Separator, SectionHeader)
    groups: list[_Group] = []

    # Phase 1: detect single-type runs (type-only, ignores config differences)
    # This keeps diagrams compact for heterogeneous architectures like ModernBERT
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

    # Phase 2: detect repeating multi-type patterns (signature-aware)
    # Uses config-aware signatures so [TB(skip_ffn)+MoE] patterns work correctly
    i = 0
    while i < len(layers):
        if i in covered or isinstance(layers[i], structural):
            i += 1
            continue
        found_pattern = False
        for pat_len in (2, 3, 4):
            if i + pat_len > len(layers):
                break
            pattern = [_group_signature(layers[i + k]) for k in range(pat_len)]
            if any(isinstance(layers[i + k], structural) for k in range(pat_len)):
                continue
            repeats = 1
            pos = i + pat_len
            while pos + pat_len <= len(layers):
                next_pat = [_group_signature(layers[pos + k]) for k in range(pat_len)]
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


# ---------------------------------------------------------------------------
# Vertical renderer
# ---------------------------------------------------------------------------

def _format_dim(value: int | str) -> str:
    """Format dimension values, abbreviating large numbers (14336 → 14.3K).

    Numbers >= 10K are shown in K notation.  If the fractional part is < 0.05
    the value is rounded (e.g. 14000 → "14K"), otherwise one decimal is kept.
    """
    if isinstance(value, str):
        try:
            n = int(value)
        except ValueError:
            return value
    else:
        n = value
    if n >= 10000:
        k = n / 1000
        if abs(k - round(k)) < 0.05:
            return f"{round(k):.0f}K"
        return f"{k:.1f}K"
    return str(n)


def _attention_label(block: TransformerBlock) -> str:
    labels = {
        "self": "Self-Attention",
        "masked": "Masked Self-Attention",
        "cross": "Cross-Attention",
        "local": "Local Attention",
        "global": "Global Attention",
        "gqa": f"GQA ({block.kv_heads or '?'}kv)",
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
    block_idx: int = 0,
) -> tuple[list[str], str, list[str]]:
    """Render one TransformerBlock. Returns (tikz_lines, last_node_name, all_node_names)."""
    lines: list[str] = []
    nodes: list[str] = []
    # Alternate skip xshift per block to prevent overlap
    skip_xshift = 2.0 + (block_idx % 5) * 0.15
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
                                width=3.0, height=0.7))
        lines.append(flat_arrow(entry, n1))
        attn_opacity = 1.0 if block.attention == "global" else 0.65
        attn_height = 1.1 if block.attention == "global" else 0.95
        lines.append(flat_block(at, attn_label, attn_color, above_of=n1, node_distance=node_dist,
                                width=4.2, height=attn_height, opacity=attn_opacity))
        lines.append(flat_arrow(n1, at))
        lines.append(flat_dim_label(f"{block.heads}h", at, side="left", distance=0.25))

        lines.append(flat_add_circle(a1, above_of=at, node_distance=0.25))
        lines.append(flat_arrow(at, a1))
        lines.append(flat_skip_arrow(entry, a1, xshift=skip_xshift))

        if block.skip_ffn:
            nodes = [n1, at, a1]
            return lines, a1, nodes

        lines.append(flat_block(n2, "LayerNorm", "norm", above_of=a1, node_distance=node_dist,
                                width=3.0, height=0.7))
        lines.append(flat_arrow(a1, n2))
        lines.append(flat_block(ff, ffn_label, "ffn", above_of=n2, node_distance=node_dist,
                                width=3.8, height=0.85))
        lines.append(flat_arrow(n2, ff))
        lines.append(flat_dim_label(_format_dim(block.d_ff), ff, side="left", distance=0.25))

        lines.append(flat_add_circle(a2, above_of=ff, node_distance=0.25))
        lines.append(flat_arrow(ff, a2))
        lines.append(flat_skip_arrow(a1, a2, xshift=skip_xshift))

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
        lines.append(flat_dim_label(f"{block.heads}h", at, side="left", distance=0.25))
        lines.append(flat_add_circle(a1, above_of=at, node_distance=0.25))
        lines.append(flat_arrow(at, a1))
        lines.append(flat_skip_arrow(entry, a1, xshift=skip_xshift))
        lines.append(flat_block(n1, "LayerNorm", "norm", above_of=a1, node_distance=node_dist))
        lines.append(flat_arrow(a1, n1))

        if block.skip_ffn:
            nodes = [at, a1, n1]
            return lines, n1, nodes

        lines.append(flat_block(ff, ffn_label, "ffn", above_of=n1, node_distance=node_dist))
        lines.append(flat_arrow(n1, ff))
        lines.append(flat_dim_label(_format_dim(block.d_ff), ff, side="left", distance=0.25))
        lines.append(flat_add_circle(a2, above_of=ff, node_distance=0.25))
        lines.append(flat_arrow(ff, a2))
        lines.append(flat_skip_arrow(n1, a2, xshift=skip_xshift))
        lines.append(flat_block(n2, "LayerNorm", "norm", above_of=a2, node_distance=node_dist))
        lines.append(flat_arrow(a2, n2))

        nodes = [at, a1, n1, ff, a2, n2]
        return lines, n2, nodes


def _render_lstm_block(
    block: LSTMBlock,
    prefix: str,
    prev: str,
    theme: Theme,
    node_dist: float = 0.25,
    block_idx: int = 0,
) -> tuple[list[str], str, list[str]]:
    """Render one LSTMBlock showing gate internals. Returns (tikz_lines, last_node, all_node_names)."""
    lines: list[str] = []
    nodes: list[str] = []
    skip_xshift = 2.0 + (block_idx % 5) * 0.15

    entry = f"{prefix}_in"
    concat = f"{prefix}_concat"
    fg = f"{prefix}_forget"
    ig = f"{prefix}_input"
    cu = f"{prefix}_cell_upd"
    og = f"{prefix}_output"
    ht = f"{prefix}_ht"

    # Invisible entry anchor
    entry_dist = node_dist + 0.15
    lines.append(
        rf"\node[inner sep=0, minimum size=0, above={entry_dist}cm of {prev}] ({entry}) {{}};" "\n"
    )
    lines.append(flat_arrow(prev, entry))

    # Concat [h_{t-1}, x_t]
    lines.append(flat_block(concat, r"Concat [$h_{t-1}$, $x_t$]", "norm",
                            above_of=entry, node_distance=0.12,
                            width=3.4, height=0.6))
    lines.append(flat_arrow(entry, concat))

    # Forget Gate
    lines.append(flat_block(fg, r"Forget Gate ($\sigma$)", "attention",
                            above_of=concat, node_distance=node_dist,
                            width=3.8, height=0.7))
    lines.append(flat_arrow(concat, fg))
    lines.append(flat_dim_label(str(block.hidden_size), fg, side="left", distance=0.25))

    # Input Gate
    lines.append(flat_block(ig, r"Input Gate ($\sigma$ + tanh)", "attention_alt",
                            above_of=fg, node_distance=node_dist,
                            width=3.8, height=0.7))
    lines.append(flat_arrow(fg, ig))

    # Cell Update (⊕)
    lines.append(flat_op_circle(cu, r"$\oplus$", above_of=ig, node_distance=0.25))
    lines.append(flat_arrow(ig, cu))
    # Skip arrow: cell state "conveyor belt" from entry to cell update
    lines.append(flat_skip_arrow(entry, cu, xshift=skip_xshift))

    # Output Gate
    lines.append(flat_block(og, r"Output Gate ($\sigma$)", "attention",
                            above_of=cu, node_distance=node_dist,
                            width=3.8, height=0.7))
    lines.append(flat_arrow(cu, og))

    # Hidden state output
    lines.append(flat_block(ht, r"$h_t = o_t \odot \tanh(C_t)$", "ffn",
                            above_of=og, node_distance=node_dist,
                            width=3.4, height=0.65))
    lines.append(flat_arrow(og, ht))

    nodes = [concat, fg, ig, cu, og, ht]
    last = ht

    # Bidirectional merge
    if block.bidirectional:
        merge = f"{prefix}_merge"
        lines.append(flat_block(merge, r"Bi-Merge (concat $\rightarrow \leftarrow$)", "residual",
                                above_of=ht, node_distance=node_dist,
                                width=3.4, height=0.65))
        lines.append(flat_arrow(ht, merge))
        lines.append(flat_dim_label(f"2x{block.hidden_size}", merge, side="left", distance=0.25))
        nodes.append(merge)
        last = merge

    return lines, last, nodes


def _render_gru_block(
    block: GRUBlock,
    prefix: str,
    prev: str,
    theme: Theme,
    node_dist: float = 0.25,
    block_idx: int = 0,
) -> tuple[list[str], str, list[str]]:
    """Render one GRUBlock showing gate internals. Returns (tikz_lines, last_node, all_node_names)."""
    lines: list[str] = []
    nodes: list[str] = []
    skip_xshift = 2.0 + (block_idx % 5) * 0.15

    entry = f"{prefix}_in"
    concat = f"{prefix}_concat"
    rg = f"{prefix}_reset"
    ug = f"{prefix}_update"
    cand = f"{prefix}_candidate"
    interp = f"{prefix}_interp"

    # Invisible entry anchor
    entry_dist = node_dist + 0.15
    lines.append(
        rf"\node[inner sep=0, minimum size=0, above={entry_dist}cm of {prev}] ({entry}) {{}};" "\n"
    )
    lines.append(flat_arrow(prev, entry))

    # Concat [h_{t-1}, x_t]
    lines.append(flat_block(concat, r"Concat [$h_{t-1}$, $x_t$]", "norm",
                            above_of=entry, node_distance=0.12,
                            width=3.4, height=0.6))
    lines.append(flat_arrow(entry, concat))

    # Reset Gate
    lines.append(flat_block(rg, r"Reset Gate ($\sigma$)", "attention_alt",
                            above_of=concat, node_distance=node_dist,
                            width=3.8, height=0.7))
    lines.append(flat_arrow(concat, rg))
    lines.append(flat_dim_label(str(block.hidden_size), rg, side="left", distance=0.25))

    # Update Gate
    lines.append(flat_block(ug, r"Update Gate ($\sigma$)", "attention",
                            above_of=rg, node_distance=node_dist,
                            width=3.8, height=0.7))
    lines.append(flat_arrow(rg, ug))

    # Candidate hidden state
    lines.append(flat_block(cand, r"Candidate $\tilde{h}_t$ (tanh)", "ffn",
                            above_of=ug, node_distance=node_dist,
                            width=3.8, height=0.7))
    lines.append(flat_arrow(ug, cand))

    # Interpolation (⊕)
    lines.append(flat_op_circle(interp, r"$\oplus$", above_of=cand, node_distance=0.25))
    lines.append(flat_arrow(cand, interp))
    # Skip arrow: update gate interpolation with previous hidden state
    lines.append(flat_skip_arrow(entry, interp, xshift=skip_xshift))

    nodes = [concat, rg, ug, cand, interp]
    last = interp

    # Bidirectional merge
    if block.bidirectional:
        merge = f"{prefix}_merge"
        lines.append(flat_block(merge, r"Bi-Merge (concat $\rightarrow \leftarrow$)", "residual",
                                above_of=interp, node_distance=node_dist,
                                width=3.4, height=0.65))
        lines.append(flat_arrow(interp, merge))
        lines.append(flat_dim_label(f"2x{block.hidden_size}", merge, side="left", distance=0.25))
        nodes.append(merge)
        last = merge

    return lines, last, nodes


def _render_mbconv_block(
    block: MBConvBlock,
    prefix: str,
    prev: str,
    theme: Theme,
    node_dist: float = 0.25,
    block_idx: int = 0,
) -> tuple[list[str], str, list[str]]:
    """Render one MBConv (inverted residual) block. Returns (tikz_lines, last_node, all_node_names)."""
    lines: list[str] = []
    nodes: list[str] = []
    skip_xshift = 2.0 + (block_idx % 5) * 0.15
    expanded = block.filters * block.expansion

    entry = f"{prefix}_in"
    expand = f"{prefix}_expand"
    dw = f"{prefix}_dw"
    se_node = f"{prefix}_se"
    proj = f"{prefix}_proj"
    add = f"{prefix}_add"

    # Invisible entry anchor
    lines.append(
        rf"\node[inner sep=0, minimum size=0, above=0.4cm of {prev}] ({entry}) {{}};" "\n"
    )
    lines.append(flat_arrow(prev, entry))

    # 1×1 Expand
    lines.append(flat_block(expand, f"1x1 Conv Expand ({expanded})", "attention_alt",
                            above_of=entry, node_distance=0.12, height=0.6, width=3.6))
    lines.append(flat_arrow(entry, expand))

    # Depthwise Conv
    lines.append(flat_block(dw, f"DW Conv {block.kernel_size}x{block.kernel_size}", "attention",
                            above_of=expand, node_distance=node_dist, height=0.7))
    lines.append(flat_arrow(expand, dw))
    lines.append(flat_dim_label(str(expanded), dw, side="left", distance=0.25))

    last_before_proj = dw

    # Squeeze-Excitation
    if block.se:
        lines.append(flat_block(se_node, "SE (Squeeze-Excitation)", "norm",
                                above_of=dw, node_distance=node_dist,
                                width=3.4, height=0.6, opacity=0.7))
        lines.append(flat_arrow(dw, se_node))
        nodes.append(se_node)
        last_before_proj = se_node

    # 1×1 Project
    lines.append(flat_block(proj, f"1x1 Conv Project ({block.filters})", "attention_alt",
                            above_of=last_before_proj, node_distance=node_dist, height=0.6, width=3.6))
    lines.append(flat_arrow(last_before_proj, proj))
    lines.append(flat_dim_label(str(block.filters), proj, side="left", distance=0.25))

    # Residual add
    lines.append(flat_add_circle(add, above_of=proj, node_distance=0.25))
    lines.append(flat_arrow(proj, add))
    lines.append(flat_skip_arrow(entry, add, xshift=skip_xshift))

    nodes.extend([expand, dw, proj, add])
    return lines, add, nodes


def _render_swin_block(
    block: SwinBlock,
    prefix: str,
    prev: str,
    theme: Theme,
    node_dist: float = 0.3,
    block_idx: int = 0,
) -> tuple[list[str], str, list[str]]:
    """Render one SwinBlock (W-MSA or SW-MSA). Returns (tikz_lines, last_node, all_node_names)."""
    lines: list[str] = []
    nodes: list[str] = []
    skip_xshift = 2.0 + (block_idx % 5) * 0.15

    is_shifted = block.window_type == "shifted"
    msa_label = "SW-MSA (shifted)" if is_shifted else "W-MSA (regular)"
    msa_color = "attention_alt" if is_shifted else "attention"

    entry = f"{prefix}_in"
    n1 = f"{prefix}_ln1"
    msa = f"{prefix}_msa"
    a1 = f"{prefix}_add1"
    n2 = f"{prefix}_ln2"
    mlp = f"{prefix}_mlp"
    a2 = f"{prefix}_add2"

    # Invisible entry anchor
    entry_dist = node_dist + 0.1
    lines.append(
        rf"\node[inner sep=0, minimum size=0, above={entry_dist}cm of {prev}] ({entry}) {{}};" "\n"
    )
    lines.append(flat_arrow(prev, entry))

    # LN → W-MSA/SW-MSA → Add → LN → MLP → Add (same as pre-LN Transformer)
    lines.append(flat_block(n1, "LayerNorm", "norm", above_of=entry, node_distance=0.12,
                            width=3.0, height=0.6))
    lines.append(flat_arrow(entry, n1))

    lines.append(flat_block(msa, msa_label, msa_color, above_of=n1, node_distance=node_dist,
                            width=4.2, height=0.95))
    lines.append(flat_arrow(n1, msa))
    lines.append(flat_dim_label(f"{block.heads}h", msa, side="left", distance=0.25))
    lines.append(flat_dim_label(f"w{block.window_size}", msa, side="right", distance=0.25))

    lines.append(flat_add_circle(a1, above_of=msa, node_distance=0.25))
    lines.append(flat_arrow(msa, a1))
    lines.append(flat_skip_arrow(entry, a1, xshift=skip_xshift))

    lines.append(flat_block(n2, "LayerNorm", "norm", above_of=a1, node_distance=node_dist,
                            width=3.0, height=0.6))
    lines.append(flat_arrow(a1, n2))

    lines.append(flat_block(mlp, "MLP", "ffn", above_of=n2, node_distance=node_dist,
                            width=3.8, height=0.85))
    lines.append(flat_arrow(n2, mlp))
    lines.append(flat_dim_label(str(block.d_model), mlp, side="left", distance=0.25))

    lines.append(flat_add_circle(a2, above_of=mlp, node_distance=0.25))
    lines.append(flat_arrow(mlp, a2))
    lines.append(flat_skip_arrow(a1, a2, xshift=skip_xshift))

    nodes = [n1, msa, a1, n2, mlp, a2]
    return lines, a2, nodes


def _h_block_params(layer: Layer) -> tuple[str, str, float, float]:
    """Return (label, fill_role, width, height) for a layer in horizontal mode."""
    match layer:
        case Embedding():
            return (layer.label, "embed", 2.0, 1.2)
        case PositionalEncoding():
            type_labels = {"rope": "RoPE", "learned": "Learned PE",
                           "sinusoidal": "Sinusoidal PE", "alibi": "ALiBi"}
            return (layer.label or type_labels.get(layer.encoding_type, "Pos. Enc."), "embed", 1.6, 0.7)
        case TransformerBlock():
            attn_short = {"self": "Self-Attn", "masked": "Masked Attn", "cross": "Cross-Attn",
                          "local": "Local Attn", "global": "Global Attn",
                          "gqa": "GQA", "mqa": "MQA"}
            label = layer.label or f"Transformer-{attn_short.get(layer.attention, 'Attn')}"
            fill = "attention" if layer.attention == "global" else "attention_alt"
            return (label, fill, 2.4, 1.4)
        case ConvBlock():
            label = layer.label or f"Conv{layer.kernel_size}x{layer.kernel_size}"
            return (label, "attention", 2.0, 1.2)
        case PatchEmbedding():
            label = layer.label or f"Patch {layer.patch_size}x{layer.patch_size}"
            return (label, "embed", 2.2, 1.2)
        case DenseLayer():
            label = layer.label or f"Dense ({layer.units})"
            return (label, "dense", 2.0, 1.0)
        case ClassificationHead():
            return (layer.label, "output", 2.0, 1.2)
        case FourierBlock():
            label = layer.label or f"Fourier (m={layer.modes})"
            return (label, "spectral", 2.2, 1.2)
        case EncoderBlock():
            return (layer.label, "attention", 2.2, 1.4)
        case DecoderBlock():
            return (layer.label, "attention_alt", 2.2, 1.4)
        case BatchNorm() | RMSNorm():
            return (layer.label, "norm", 1.6, 0.9)
        case Activation():
            fn_labels = {"relu": "ReLU", "gelu": "GELU", "swish": "Swish",
                         "softmax": "Softmax", "tanh": "Tanh", "sigmoid": "Sigmoid"}
            return (layer.label or fn_labels.get(layer.function, layer.function), "ffn", 1.6, 0.9)
        case Dropout():
            return (layer.label or f"Drop({layer.rate})", "border", 1.4, 0.8)
        case Generator():
            return (layer.label, "embed", 2.2, 1.4)
        case Discriminator():
            return (layer.label, "physics", 2.2, 1.4)
        case UNetBlock():
            label = layer.label or f"UNet ({layer.filters})"
            return (label, "spectral", 2.2, 1.2)
        case NoiseHead():
            return (layer.label, "physics", 2.0, 1.0)
        case LSTMBlock():
            base = "Bi-LSTM" if layer.bidirectional else "LSTM"
            label = layer.label or f"{base} ({layer.hidden_size})"
            return (label, "attention", 2.2, 1.2)
        case GRUBlock():
            base = "Bi-GRU" if layer.bidirectional else "GRU"
            label = layer.label or f"{base} ({layer.hidden_size})"
            return (label, "attention_alt", 2.2, 1.2)
        case MambaBlock():
            label = layer.label or "Mamba"
            return (label, "spectral", 2.0, 1.2)
        case MoELayer():
            label = layer.label or f"MoE top-{layer.top_k}/{layer.num_experts}"
            return (label, "dense", 2.2, 1.2)
        case GraphConv():
            return (layer.label, "attention_alt", 2.0, 1.0)
        case GraphAttention():
            return (layer.label, "attention", 2.0, 1.2)
        case MessagePassing():
            label = layer.label or f"MsgPass ({layer.aggregation})"
            return (label, "spectral", 2.0, 1.0)
        case GraphPooling():
            label = layer.label or f"Graph {layer.pool_type.title()}Pool"
            return (label, "ffn", 1.8, 1.0)
        case CustomBlock():
            return (layer.text, layer.color_role, 2.0, 1.2)
        case ResidualBlock():
            label = layer.label or f"ResBlock ({layer.filters})"
            return (label, "attention", 2.0, 1.2)
        case BottleneckBlock():
            label = layer.label or f"Bottleneck ({layer.filters})"
            return (label, "attention", 2.2, 1.2)
        case SwinBlock():
            wt = "SW-MSA" if layer.window_type == "shifted" else "W-MSA"
            label = layer.label or f"Swin-{wt}"
            fill = "attention_alt" if layer.window_type == "shifted" else "attention"
            return (label, fill, 2.2, 1.2)
        case PatchMerging():
            label = layer.label or "Patch Merge"
            return (label, "embed", 1.8, 1.0)
        case SamplingLayer():
            return (layer.label, "residual", 2.0, 1.0)
        case AdaptiveLayerNorm():
            return (f"{layer.label}", "norm", 1.8, 1.0)
        case SelectiveSSM():
            return (layer.label, "spectral", 2.0, 1.0)
        case _:
            return (str(type(layer).__name__), "dense", 2.0, 1.0)


def _render_horizontal(
    name: str,
    layers: list[Layer],
    groups: list[_Group],
    theme: Theme,
    show_n: int,
) -> str:
    """Render architecture as horizontal (west→east) pipeline.

    Blocks are compact single nodes — no expanded internals.
    Ideal for ViT, DETR, FNO pipeline-style diagrams.
    """
    parts: list[str] = []
    parts.append(flat_head())
    parts.append(flat_colors(theme))
    parts.append(flat_begin())

    # Precompute group index
    group_index: dict[int, _Group] = {}
    for grp in groups:
        for idx in range(grp.start, grp.end):
            group_index[idx] = grp

    # Visibility
    visible: dict[int, bool] = {}
    for i in range(len(layers)):
        grp = group_index.get(i)
        if grp is None:
            visible[i] = True
        else:
            offset_in_group = i - grp.start
            rep_index = offset_in_group // grp.pattern_len
            visible[i] = rep_index < show_n

    prev: str | None = None
    first_node: str | None = None
    node_idx = 0
    group_frame_data: dict[int, dict] = {}
    node_dist = 0.5

    for i, layer in enumerate(layers):
        if not visible[i]:
            grp = group_index.get(i)
            if grp:
                gid = id(grp)
                if gid not in group_frame_data:
                    group_frame_data[gid] = {"count": grp.count, "nodes": []}
            continue

        grp = group_index.get(i)
        grp_id = id(grp) if grp else None

        # Structural elements
        if isinstance(layer, Separator):
            nid = f"sep_h_{node_idx}"
            if prev:
                parts.append(
                    rf"\node[inner sep=0, minimum size=0, right=0.4cm of {prev}] ({nid}) {{}};" "\n"
                    rf"\draw[densely dashed, color=clrgroup_frame!70, line width=0.6pt] "
                    rf"([yshift=-0.8cm]{nid}.center) -- ([yshift=0.8cm]{nid}.center);" "\n"
                )
            prev = nid
            node_idx += 1
            continue
        elif isinstance(layer, SectionHeader):
            nid = f"section_h_{node_idx}"
            if prev:
                parts.append(
                    rf"\node[right=0.6cm of {prev}, font=\sffamily\small\bfseries, "
                    rf"text=clrborder, rotate=0] ({nid}) {{{_latex_escape(layer.title)}}};" "\n"
                )
            else:
                parts.append(
                    rf"\node[font=\sffamily\small\bfseries, text=clrborder] ({nid}) "
                    rf"at (0,0) {{{_latex_escape(layer.title)}}};" "\n"
                )
            prev = nid
            node_idx += 1
            continue

        # SideBySide in horizontal mode — render as nested vertical stacks
        if isinstance(layer, SideBySide):
            prefix = f"sbs_h_{node_idx}"
            left_last, right_last, sbs_nodes = _render_side_by_side(
                layer, prefix, prev, theme, parts, show_n=show_n,
            )
            merge = f"{prefix}_merge"
            parts.append(
                rf"\node[inner sep=0, minimum size=0] ({merge}) "
                rf"at ($({left_last}.north)!0.5!({right_last}.north) + (0,0.8)$) {{}};" "\n"
            )
            parts.append(flat_arrow(left_last, merge, from_anchor="north", to_anchor="south"))
            parts.append(flat_arrow(right_last, merge, from_anchor="north", to_anchor="south"))
            if first_node is None:
                first_node = sbs_nodes[0] if sbs_nodes else merge
            prev = merge
            node_idx += 1
            continue

        # BidirectionalFlow in horizontal mode — render inline
        if isinstance(layer, BidirectionalFlow):
            prefix = f"bidir_h_{node_idx}"
            last_step, bf_nodes, _ = _render_bidir_flow(layer, prefix, prev, parts)
            if first_node is None and bf_nodes:
                first_node = bf_nodes[0]
            prev = last_step
            node_idx += 1
            continue

        # Get compact block params
        label, fill, w, h = _h_block_params(layer)

        # Add gap after group ends
        extra_dist = node_dist
        if prev and i > 0 and grp_id is None:
            prev_grp = group_index.get(i - 1)
            if prev_grp is not None:
                extra_dist = node_dist + 0.3

        nid = f"h_{node_idx}"
        if prev is None:
            parts.append(flat_block(nid, label, fill, position="(0,0)",
                                    width=w, height=h, style="block",
                                    text_color="clrtext"))
        else:
            parts.append(flat_block(nid, label, fill, right_of=prev,
                                    node_distance=extra_dist, width=w, height=h,
                                    style="block", text_color="clrtext"))
            parts.append(flat_arrow_h(prev, nid))

        # Dimension label below block (avoid overlap with inter-block arrows)
        dim_text = _get_dim_text(layer)
        if dim_text:
            parts.append(
                rf"\node[below=6pt of {nid}, font=\sffamily\scriptsize, text=clrborder] {{{dim_text}}};" "\n"
            )

        if first_node is None:
            first_node = nid
        prev = nid

        # Track group
        if grp_id is not None:
            group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
            group_frame_data[grp_id]["nodes"].append(nid)

        node_idx += 1

    # I/O arrows
    if first_node:
        parts.append(flat_io_arrow(first_node, direction="left", label="Input"))
    if prev:
        parts.append(flat_io_arrow(prev, direction="right", label="Output"))

    # Group frames
    for idx, (gid, gdata) in enumerate(group_frame_data.items()):
        if gdata["nodes"]:
            parts.append(group_frame(
                f"grp_h_{idx}", gdata["nodes"],
                repeat=gdata["count"], padding=0.25, horizontal=True,
            ))

    # Title below
    parts.append(flat_title(rf"\textbf{{{name}}}"))

    parts.append(flat_end())
    return "".join(parts)


def _get_dim_text(layer: Layer) -> str:
    """Extract dimension annotation text for a layer."""
    match layer:
        case Embedding():
            return _format_dim(layer.d_model)
        case TransformerBlock():
            return f"{layer.heads}h, {_format_dim(layer.d_model)}"
        case ConvBlock():
            return str(layer.filters)
        case PatchEmbedding():
            return _format_dim(layer.d_model)
        case DenseLayer():
            return str(layer.units)
        case FourierBlock():
            return f"m={layer.modes}"
        case LSTMBlock() | GRUBlock():
            return str(layer.hidden_size)
        case UNetBlock():
            return str(layer.filters)
        case MambaBlock():
            return _format_dim(layer.d_model)
        case SwinBlock():
            return f"{layer.heads}h"
        case EncoderBlock() | DecoderBlock():
            return _format_dim(layer.d_model)
        case _:
            return ""


def _render_unet(
    name: str,
    layers: list[Layer],
    theme: Theme,
) -> str:
    """Render U-Net-style architecture with iconic U-shaped layout.

    Expects layers to be UNetLevel instances followed by a Bottleneck.
    Block width encodes spatial resolution (wider = higher resolution).
    """
    parts: list[str] = []
    parts.append(flat_head())
    parts.append(flat_colors(theme))
    parts.append(flat_begin())

    # Separate levels and bottleneck
    levels: list[UNetLevel] = []
    bottleneck: Bottleneck | None = None
    pre_layers: list[Layer] = []  # layers before U-Net structure
    post_layers: list[Layer] = []  # layers after U-Net structure
    unet_started = False

    for layer in layers:
        if isinstance(layer, UNetLevel):
            unet_started = True
            levels.append(layer)
        elif isinstance(layer, Bottleneck):
            unet_started = True
            bottleneck = layer
        elif not unet_started:
            pre_layers.append(layer)
        else:
            post_layers.append(layer)

    n_levels = len(levels)
    if n_levels == 0:
        # Fallback to vertical if no U-Net structure found
        groups = _detect_groups(layers)
        return _render_vertical(name, layers, groups, theme, 4)

    # Layout parameters
    level_y_step = 1.8   # vertical distance between levels
    col_x_sep = 6.0      # horizontal separation between encoder and decoder columns
    max_width = 4.5       # block width at full resolution
    min_width = 2.0       # block width at lowest resolution

    # Compute block widths based on resolution (copy to avoid mutating caller's data)
    levels = [copy.copy(lv) for lv in levels]
    for i, level in enumerate(levels):
        if level.resolution <= 0:
            level.resolution = 1.0 / (2 ** i)

    # Render encoder (descending on left)
    enc_nodes: list[str] = []
    prev_enc: str | None = None

    for i, level in enumerate(levels):
        nid = f"enc_{i}"
        w = min_width + level.resolution * (max_width - min_width)
        label = level.label or f"Conv ({level.filters})"
        if level.encoder:
            label, fill, _, _ = _h_block_params(level.encoder)
        else:
            fill = "attention"

        x = -col_x_sep / 2
        y = -i * level_y_step
        parts.append(flat_block(nid, label, fill, position=f"({x},{y})",
                                width=w, height=1.0))
        parts.append(flat_dim_label(str(level.filters), nid, side="left", distance=0.15))

        if prev_enc:
            # Red downward arrow
            parts.append(
                rf"\draw[arrow, color=red!70!black] ({prev_enc}.south) -- ({nid}.north);" "\n"
            )

        enc_nodes.append(nid)
        prev_enc = nid

    # Render bottleneck (bottom center)
    bn_nid = "bottleneck"
    bn_y = -n_levels * level_y_step
    bn_filters = bottleneck.filters if bottleneck else levels[-1].filters * 2
    bn_label = (bottleneck.label if bottleneck and bottleneck.label
                else f"Bottleneck ({bn_filters})")
    bn_w = min_width * 0.9
    parts.append(flat_block(bn_nid, bn_label, "spectral",
                            position=f"(0,{bn_y})", width=bn_w, height=1.0))
    parts.append(flat_dim_label(str(bn_filters), bn_nid, side="left", distance=0.15))

    # Arrow from last encoder to bottleneck
    if prev_enc:
        parts.append(
            rf"\draw[arrow, color=red!70!black] ({prev_enc}.south) -- ({bn_nid}.north west);" "\n"
        )

    # Render decoder (ascending on right)
    dec_nodes: list[str] = []
    prev_dec = bn_nid

    for i in range(n_levels - 1, -1, -1):
        level = levels[i]
        nid = f"dec_{i}"
        w = min_width + level.resolution * (max_width - min_width)
        label = level.label or f"UpConv ({level.filters})"
        if level.decoder:
            label, fill, _, _ = _h_block_params(level.decoder)
        else:
            fill = "attention_alt"

        x = col_x_sep / 2
        y = -i * level_y_step
        parts.append(flat_block(nid, label, fill, position=f"({x},{y})",
                                width=w, height=1.0))
        parts.append(flat_dim_label(str(level.filters), nid, side="right", distance=0.15))

        # Green upward arrow
        if prev_dec == bn_nid:
            parts.append(
                rf"\draw[arrow, color=green!60!black] ({prev_dec}.north east) -- ({nid}.south);" "\n"
            )
        else:
            parts.append(
                rf"\draw[arrow, color=green!60!black] ({prev_dec}.north) -- ({nid}.south);" "\n"
            )

        dec_nodes.append(nid)
        prev_dec = nid

    dec_nodes.reverse()  # now dec_nodes[i] corresponds to levels[i]

    # Skip connections (gray horizontal arrows from encoder to decoder)
    for i in range(n_levels):
        enc = enc_nodes[i]
        dec = dec_nodes[i]
        parts.append(
            rf"\draw[skiparrow, densely dashed, color=clrborder!50, line width=0.7pt] "
            rf"({enc}.east) -- node[above, font=\sffamily\scriptsize, text=clrborder!70] {{copy}} "
            rf"({dec}.west);" "\n"
        )

    # Title
    parts.append(flat_title(rf"\textbf{{{name}}}"))

    parts.append(flat_end())
    return "".join(parts)


def _render_encoder_decoder(
    ed: EncoderDecoder,
    prefix: str,
    prev: str | None,
    theme: Theme,
    parts: list[str],
    show_n: int = 2,
) -> tuple[str, list[str]]:
    """Render EncoderDecoder with proper cross-attention flow.

    Returns (decoder_last_node, all_node_names).
    Unlike SideBySide: output only from decoder, cross-attention arrows,
    separate input embeddings.
    """
    col_sep = 5.5
    all_nodes: list[str] = []

    # Encoder/decoder anchors
    enc_anchor = f"{prefix}_enc_anchor"
    dec_anchor = f"{prefix}_dec_anchor"
    if prev:
        parts.append(
            rf"\node[inner sep=0, minimum size=0, above=0.8cm of {prev}, xshift=-{col_sep/2}cm] "
            rf"({enc_anchor}) {{}};" "\n"
        )
        parts.append(
            rf"\node[inner sep=0, minimum size=0, above=0.8cm of {prev}, xshift={col_sep/2}cm] "
            rf"({dec_anchor}) {{}};" "\n"
        )
    else:
        parts.append(rf"\node[inner sep=0, minimum size=0] ({enc_anchor}) at (-{col_sep/2},0) {{}};" "\n")
        parts.append(rf"\node[inner sep=0, minimum size=0] ({dec_anchor}) at ({col_sep/2},0) {{}};" "\n")

    # Column labels below anchors — placed below I/O arrow + label to avoid overlap
    parts.append(
        rf"\node[below=0.95cm of {enc_anchor}, font=\sffamily\small\bfseries, text=clrborder] "
        rf"{{{_latex_escape(ed.encoder_label)}}};" "\n"
    )
    parts.append(
        rf"\node[below=0.95cm of {dec_anchor}, font=\sffamily\small\bfseries, text=clrborder] "
        rf"{{{_latex_escape(ed.decoder_label)}}};" "\n"
    )

    # Encoder input embeddings
    enc_prev = enc_anchor
    if ed.encoder_input:
        for j, layer in enumerate(ed.encoder_input):
            nid = f"{prefix}_enc_in_{j}"
            label, fill, w, h = _h_block_params(layer)
            parts.append(flat_block(nid, label, fill, above_of=enc_prev,
                                    node_distance=0.35, width=min(w + 0.8, 3.8), height=max(h, 0.75)))
            parts.append(flat_arrow(enc_prev, nid))
            enc_prev = nid
            all_nodes.append(nid)

    # Decoder input embeddings
    dec_prev = dec_anchor
    if ed.decoder_input:
        for j, layer in enumerate(ed.decoder_input):
            nid = f"{prefix}_dec_in_{j}"
            label, fill, w, h = _h_block_params(layer)
            parts.append(flat_block(nid, label, fill, above_of=dec_prev,
                                    node_distance=0.35, width=min(w + 0.8, 3.8), height=max(h, 0.75)))
            parts.append(flat_arrow(dec_prev, nid))
            dec_prev = nid
            all_nodes.append(nid)

    # Render encoder column
    enc_nodes: dict[int, str] = {}
    enc_visible: list[str] = []
    n_enc = len(ed.encoder)
    for j, layer in enumerate(ed.encoder):
        if j >= show_n:
            continue
        nid = f"{prefix}_enc_{j}"
        label, fill, w, h = _h_block_params(layer)
        parts.append(flat_block(nid, label, fill, above_of=enc_prev,
                                node_distance=0.4, width=min(w + 0.8, 3.8), height=max(h, 0.85)))
        parts.append(flat_arrow(enc_prev, nid))
        dim = _get_dim_text(layer)
        if dim:
            parts.append(flat_dim_label(dim, nid, side="left"))
        enc_prev = nid
        enc_nodes[j] = nid
        enc_visible.append(nid)
        all_nodes.append(nid)

    # Encoder group frame with ×N
    if n_enc > show_n and enc_visible:
        parts.append(group_frame(f"{prefix}_enc_grp", enc_visible, repeat=n_enc, padding=0.25))

    # Render decoder column
    dec_nodes: dict[int, str] = {}
    dec_visible: list[str] = []
    n_dec = len(ed.decoder)
    for j, layer in enumerate(ed.decoder):
        if j >= show_n:
            continue
        nid = f"{prefix}_dec_{j}"
        label, fill, w, h = _h_block_params(layer)
        parts.append(flat_block(nid, label, fill, above_of=dec_prev,
                                node_distance=0.4, width=min(w + 0.8, 3.8), height=max(h, 0.85)))
        parts.append(flat_arrow(dec_prev, nid))
        dim = _get_dim_text(layer)
        if dim:
            parts.append(flat_dim_label(dim, nid, side="right"))
        dec_prev = nid
        dec_nodes[j] = nid
        dec_visible.append(nid)
        all_nodes.append(nid)

    # Decoder group frame with ×N
    if n_dec > show_n and dec_visible:
        parts.append(group_frame(f"{prefix}_dec_grp", dec_visible, repeat=n_dec, padding=0.25))

    # Cross-attention arrows
    if ed.cross_attention != "none" and enc_visible and dec_visible:
        default_lbl = "Cross-Attn" if ed.cross_attention == "all" else "Context"
        lbl = ed.cross_attention_label if ed.cross_attention_label is not None else default_lbl
        top_enc = enc_visible[-1]
        first_dec = dec_visible[0]
        parts.append(flat_cross_attention_arrow(top_enc, first_dec, label=lbl))

    # Input arrows for each column (from below) — labels suppressed if empty string
    if ed.encoder_input_label:
        parts.append(flat_io_arrow(enc_anchor, direction="below", label=ed.encoder_input_label))
    if ed.decoder_input_label:
        parts.append(flat_io_arrow(dec_anchor, direction="below", label=ed.decoder_input_label))

    # Output arrow above the decoder column — anchor on group frame if collapsed
    if n_dec > show_n and dec_visible:
        parts.append(
            rf"\draw[arrow] ({prefix}_dec_grp.north) -- ++ (0,0.5) "
            rf"node[right, yshift=-2pt, font=\sffamily\scriptsize, text=clrtext] {{Output}};" "\n"
        )
    else:
        parts.append(flat_io_arrow(dec_prev, direction="above", label="Output"))

    # Return: output from decoder only (NOT merged with encoder)
    return dec_prev, all_nodes


def _render_lstm_olah(
    block: LSTMBlock,
    prefix: str,
    prev: str | None,
    parts: list[str],
    x_offset: float = 0,
    y_offset: float = 0,
) -> tuple[str, list[str]]:
    """Render LSTM cell in Chris Olah 'conveyor belt' style.

    Architecture:
    - Top: Cell state highway C_{t-1} → [×forget] → [+input] → branch → C_t
    - Middle: Gate computation σ/tanh nodes
    - Bottom: Concat [h_{t-1}, x_t] → fan-out bus → branch taps to each gate
    - Right: branch_ct → tanh(C_t) → [×output] with o_t → h_t

    Returns (last_node, all_node_names).
    """
    all_nodes: list[str] = []
    CW = 7.0  # cell width

    # Base position
    base = f"{prefix}_base"
    if prev:
        parts.append(rf"\node[inner sep=0, minimum size=0, above=1.5cm of {prev}] ({base}) {{}};" "\n")
        parts.append(flat_arrow(prev, base))
    else:
        parts.append(rf"\node[inner sep=0, minimum size=0] ({base}) at ({x_offset},{y_offset}) {{}};" "\n")

    # =========================================================================
    # TOP LEVEL: Cell state highway (y=+3.0 relative to base)
    # =========================================================================
    cs_in = f"{prefix}_cs_in"
    cs_out = f"{prefix}_cs_out"
    parts.append(rf"\node[inner sep=0] ({cs_in}) at ([xshift=-{CW/2}cm, yshift=3.0cm]{base}) {{}};" "\n")
    parts.append(rf"\node[inner sep=0] ({cs_out}) at ([xshift={CW/2}cm, yshift=3.0cm]{base}) {{}};" "\n")
    parts.append(olah_label(r"$C_{t-1}$", relative_to=cs_in, xshift=-0.6, anchor="east"))
    parts.append(olah_label(r"$C_t$", relative_to=cs_out, xshift=0.6, anchor="west"))

    # Operations on cell state line (left to right)
    forget_x = f"{prefix}_fg_x"      # × (forget gate action)
    input_add = f"{prefix}_in_add"    # + (input gate action)
    parts.append(olah_gate_node(forget_x, r"$\times$", OLAH_MULTIPLY,
                                relative_to=cs_in, xshift=1.5, yshift=0))
    parts.append(olah_gate_node(input_add, "$+$", OLAH_ADD,
                                relative_to=cs_in, xshift=3.5, yshift=0))

    # Branch point where cell state splits down to output
    branch_ct = f"{prefix}_br_ct"
    parts.append(olah_copy_dot(branch_ct, cs_in, xshift=5.0, yshift=0))

    # Cell state highway arrows (thick blue)
    parts.append(olah_cell_highway(cs_in, forget_x))
    parts.append(olah_cell_highway(forget_x, input_add))
    parts.append(olah_cell_highway(input_add, branch_ct))
    parts.append(
        rf"\draw[line width=2.5pt, color=clrresidual, ->, >=Stealth] "
        rf"({branch_ct}) -- ({cs_out});" "\n"
    )

    # =========================================================================
    # OUTPUT PATH: branch_ct → tanh(C_t) → ×output with o_t → h_t
    # =========================================================================
    tanh_ct = f"{prefix}_tanh_ct"
    output_x = f"{prefix}_out_x"
    parts.append(olah_gate_node(tanh_ct, "tanh", OLAH_TANH,
                                relative_to=branch_ct, xshift=0.6, yshift=-1.0))
    parts.append(olah_gate_node(output_x, r"$\times$", OLAH_MULTIPLY,
                                relative_to=branch_ct, xshift=1.4, yshift=-2.1))

    # branch_ct → tanh_ct → output_x (no overlap; arrows route diagonally)
    parts.append(olah_thin_arrow(branch_ct, tanh_ct, color="clrborder!50"))
    parts.append(
        rf"\draw[->, color=clrborder!50, line width=0.5pt] "
        rf"({tanh_ct}.south) -- ({output_x}.north west);" "\n"
    )

    # h_t output (horizontal right from output_x)
    h_out = f"{prefix}_h_out"
    parts.append(rf"\node[inner sep=0] ({h_out}) at ([xshift={CW/2}cm, yshift=0.5cm]{base}) {{}};" "\n")
    parts.append(
        rf"\draw[->, color=clrborder!80, line width=0.8pt] ({output_x}.east) -- ({h_out});" "\n"
    )

    # h_t branch point and label
    branch_ht = f"{prefix}_br_ht"
    parts.append(olah_copy_dot(branch_ht, h_out, xshift=-0.3))
    parts.append(olah_label(r"$h_t$", relative_to=h_out, xshift=0.6, anchor="west"))

    # =========================================================================
    # MIDDLE LEVEL: Gate computation nodes (y=+1.5)
    # =========================================================================
    fg_sigma = f"{prefix}_fg_s"   # forget gate σ
    ig_sigma = f"{prefix}_ig_s"   # input gate σ
    ig_tanh = f"{prefix}_ig_t"    # input candidate tanh
    og_sigma = f"{prefix}_og_s"   # output gate σ

    parts.append(olah_gate_node(fg_sigma, r"$\sigma$", OLAH_SIGMA,
                                relative_to=cs_in, xshift=1.5, yshift=-1.5))
    parts.append(olah_gate_node(ig_sigma, r"$\sigma$", OLAH_SIGMA,
                                relative_to=cs_in, xshift=3.0, yshift=-1.5))
    parts.append(olah_gate_node(ig_tanh, "tanh", OLAH_TANH,
                                relative_to=cs_in, xshift=4.0, yshift=-1.5))
    parts.append(olah_gate_node(og_sigma, r"$\sigma$", OLAH_SIGMA,
                                relative_to=cs_in, xshift=5.9, yshift=-1.5))

    # Gate → cell state operation arrows
    parts.append(olah_thin_arrow(fg_sigma, forget_x))          # σ_f → ×
    parts.append(olah_thin_arrow(ig_sigma, input_add, to_anchor="south west"))  # σ_i → +
    parts.append(olah_thin_arrow(ig_tanh, input_add, to_anchor="south east"))   # tanh_cand → +
    parts.append(
        rf"\draw[->, color=clrborder!50, line width=0.5pt] "
        rf"({og_sigma}.north) |- ({output_x}.east);" "\n"
    )

    # =========================================================================
    # BOTTOM LEVEL: Concat + fan-out bus (y=-0.3)
    # =========================================================================
    h_in = f"{prefix}_h_in"
    x_in = f"{prefix}_x_in"
    concat = f"{prefix}_concat"

    # Input positions: h on left, x further right and below — clean diagonal in
    parts.append(rf"\node[inner sep=0] ({h_in}) at ([xshift=-{CW/2}cm, yshift=-0.1cm]{base}) {{}};" "\n")
    parts.append(rf"\node[inner sep=0] ({x_in}) at ([xshift=-{CW/2 - 1.5}cm, yshift=-1.5cm]{base}) {{}};" "\n")
    parts.append(olah_label(r"$h_{t-1}$", relative_to=h_in, xshift=-0.6, anchor="east"))
    parts.append(olah_label(r"$x_t$", relative_to=x_in, xshift=0, yshift=-0.35, anchor="north"))

    # Concat point (where h and x merge)
    parts.append(olah_copy_dot(concat, base, xshift=-2.0, yshift=-0.1))

    # Arrows: h_{t-1} → concat (horizontal), x_t → concat (diagonal from south-east)
    parts.append(
        rf"\draw[->, color=clrborder!60, line width=0.6pt] ({h_in}) -- ({concat});" "\n"
        rf"\draw[->, color=clrborder!60, line width=0.6pt] ({x_in}) -- ({concat});" "\n"
    )

    # Fan-out bus line from concat to rightmost gate
    bus_end = f"{prefix}_bus_end"
    parts.append(rf"\node[inner sep=0] ({bus_end}) at ([xshift=5.9cm, yshift=-1.5cm]{cs_in}) {{}};" "\n")
    parts.append(
        rf"\draw[-, color=clrborder!70, line width=1pt] ({concat}) -- ({concat} -| {bus_end});" "\n"
    )

    # Branch taps from bus to each gate (copy dots + vertical arrows)
    for gate_name, gate_label in [(fg_sigma, "f"), (ig_sigma, "i"), (ig_tanh, "it"), (og_sigma, "o")]:
        tap = f"{prefix}_tap_{gate_label}"
        parts.append(
            rf"\node[circle, fill=clrborder, minimum size=2.5pt, inner sep=0pt] "
            rf"({tap}) at ({gate_name} |- {concat}) {{}};" "\n"
        )
        parts.append(olah_branch_tap(tap, gate_name))

    # =========================================================================
    # BOUNDING BOX + LABEL
    # =========================================================================
    cell_box = f"{prefix}_cell"
    parts.append(
        rf"\node[draw=clrborder!25, rounded corners=8pt, dashed, line width=0.5pt, "
        rf"inner sep=0.35cm, fit=({cs_in}) ({cs_out}) ({x_in}) ({concat}) ({output_x})] ({cell_box}) {{}};" "\n"
    )
    parts.append(
        rf"\node[below=0.35cm of {cell_box}.south, font=\sffamily\scriptsize\bfseries, text=clrborder] "
        rf"{{LSTM Cell ({block.hidden_size})}};" "\n"
    )

    all_nodes = [cs_in, cs_out, forget_x, input_add, branch_ct,
                 tanh_ct, output_x, fg_sigma, ig_sigma, ig_tanh, og_sigma,
                 concat, h_in, x_in, h_out, cell_box]

    return cell_box, all_nodes


def _render_spinn(
    spinn: SPINNBlock,
    prefix: str,
    prev: str | None,
    parts: list[str],
) -> tuple[str, list[str]]:
    """Render SPINN architecture: Buffer | Stack | Tracker.

    Returns (last_node, all_node_names).
    """
    all_nodes: list[str] = []
    tokens = spinn.buffer_tokens or ["$w_1$", "$w_2$", "$w_3$", "$w_4$"]

    # Base position
    if prev:
        base = f"{prefix}_base"
        parts.append(rf"\node[inner sep=0, above=1.0cm of {prev}] ({base}) {{}};" "\n")
        parts.append(flat_arrow(prev, base))
    else:
        base = f"{prefix}_base"
        parts.append(rf"\node[inner sep=0] ({base}) at (0,0) {{}};" "\n")

    # === Buffer (left) — horizontal row of tokens ===
    buf_label = f"{prefix}_buf_label"
    parts.append(
        rf"\node[font=\sffamily\small\bfseries, text=clrborder, above=0.3cm of {base}, xshift=-4cm] "
        rf"({buf_label}) {{Buffer}};" "\n"
    )
    buf_prev = buf_label
    buf_nodes = []
    for j, tok in enumerate(tokens):
        nid = f"{prefix}_buf_{j}"
        parts.append(flat_block(nid, tok, "embed", above_of=buf_prev if j == 0 else None,
                                right_of=buf_prev if j > 0 else None,
                                node_distance=0.2, width=1.2, height=0.7,
                                style="smallblock"))
        buf_prev = nid
        buf_nodes.append(nid)
        all_nodes.append(nid)

    # === Stack (center) — vertical stack of slots ===
    stack_label = f"{prefix}_stack_label"
    parts.append(
        rf"\node[font=\sffamily\small\bfseries, text=clrborder, "
        rf"right=2.5cm of {buf_nodes[-1]}] ({stack_label}) {{Stack}};" "\n"
    )
    stack_slots = []
    stack_prev = stack_label
    for j in range(3):
        nid = f"{prefix}_stack_{j}"
        labels = ["$s_1$", "$s_2$", "$s_3$"]
        parts.append(flat_block(nid, labels[j], "attention", above_of=stack_prev,
                                node_distance=0.2, width=1.5, height=0.7, style="smallblock"))
        stack_prev = nid
        stack_slots.append(nid)
        all_nodes.append(nid)

    # Push arrow from buffer to stack
    parts.append(
        rf"\draw[->, thick, color=clrembed, rounded corners=3pt] "
        rf"({buf_nodes[-1]}.east) -- node[above, font=\sffamily\tiny] {{push}} ({stack_slots[0]}.west);" "\n"
    )

    # === Composition (below stack) — binary merge ===
    comp = f"{prefix}_compose"
    parts.append(
        rf"\node[circle, draw=clrborder, fill=clrffn!50, inner sep=3pt, "
        rf"font=\sffamily\scriptsize\bfseries, above=0.6cm of {stack_slots[-1]}] "
        rf"({comp}) {{Reduce}};" "\n"
    )
    parts.append(
        rf"\draw[->, color=clrborder, line width=0.9pt] ({stack_slots[-1]}.north) -- ({comp}.south west);" "\n"
        rf"\draw[->, color=clrborder, line width=0.9pt] ({stack_slots[-2]}.north east) -- ({comp}.south east);" "\n"
    )
    all_nodes.append(comp)

    # === Tracker RNN (right) ===
    tracker = f"{prefix}_tracker"
    parts.append(
        rf"\node[block, fill=clrspectral!85, minimum width=2.2cm, minimum height=1.2cm, "
        rf"text=clrtext, right=2.0cm of {stack_label}, yshift=1.0cm] "
        rf"({tracker}) {{Tracker RNN ({spinn.tracking_size})}};" "\n"
    )
    all_nodes.append(tracker)

    # Tracker ↔ Stack arrows
    parts.append(
        rf"\draw[<->, thick, color=clrspectral, densely dashed] "
        rf"({stack_slots[1]}.east) -- node[above, font=\sffamily\tiny] {{read}} ({tracker}.west);" "\n"
    )

    # Self-loop on tracker
    parts.append(
        rf"\draw[->, thick, color=clrspectral!70, rounded corners=3pt] "
        rf"({tracker}.north) .. controls ++(0,0.5) and ++(0.8,0.5) .. ({tracker}.east);" "\n"
    )

    # Frame around everything
    frame = f"{prefix}_frame"
    fit_list = " ".join(f"({n})" for n in [buf_label] + buf_nodes + stack_slots + [comp, tracker])
    parts.append(
        rf"\node[draw=clrgroup_frame, rounded corners=6pt, dashed, line width=0.8pt, "
        rf"fill=clrgroup_fill, fill opacity=0.3, inner sep=0.4cm, fit={fit_list}] ({frame}) {{}};" "\n"
    )
    parts.append(
        rf"\node[above=2pt of {frame}.north west, anchor=south west, "
        rf"font=\sffamily\scriptsize\bfseries, text=clrgroup_frame] "
        rf"{{{spinn.label} ({spinn.hidden_size})}};" "\n"
    )
    all_nodes.append(frame)

    return frame, all_nodes


def _render_side_by_side(
    sbs: SideBySide,
    prefix: str,
    prev: str | None,
    theme: Theme,
    parts: list[str],
    show_n: int = 2,
) -> tuple[str, str, list[str]]:
    """Render SideBySide as two vertical columns with show_n collapsing.

    Returns (left_last, right_last, all_node_names).
    """
    col_sep = 5.0
    all_nodes: list[str] = []

    # Column starting points
    left_anchor = f"{prefix}_L_anchor"
    right_anchor = f"{prefix}_R_anchor"
    if prev:
        parts.append(
            rf"\node[inner sep=0, minimum size=0, above=0.8cm of {prev}, xshift=-{col_sep/2}cm] "
            rf"({left_anchor}) {{}};" "\n"
        )
        parts.append(
            rf"\node[inner sep=0, minimum size=0, above=0.8cm of {prev}, xshift={col_sep/2}cm] "
            rf"({right_anchor}) {{}};" "\n"
        )
    else:
        parts.append(
            rf"\node[inner sep=0, minimum size=0] ({left_anchor}) at (-{col_sep/2},0) {{}};" "\n"
        )
        parts.append(
            rf"\node[inner sep=0, minimum size=0] ({right_anchor}) at ({col_sep/2},0) {{}};" "\n"
        )

    # Column labels below anchors (avoid overlap with upward arrows)
    parts.append(
        rf"\node[below=0.95cm of {left_anchor}, font=\sffamily\small\bfseries, text=clrborder] "
        rf"{{{_latex_escape(sbs.left_label)}}};" "\n"
    )
    parts.append(
        rf"\node[below=0.95cm of {right_anchor}, font=\sffamily\small\bfseries, text=clrborder] "
        rf"{{{_latex_escape(sbs.right_label)}}};" "\n"
    )

    def _render_column(layers, col_prefix, anchor, dim_side, show_n_col):
        """Render one column with show_n collapsing. Returns (last_node, node_map, visible_nodes)."""
        col_prev = anchor
        node_map: dict[int, str] = {}
        visible: list[str] = []
        n = len(layers)
        for j, layer in enumerate(layers):
            # Show first show_n, skip rest
            if show_n_col < n and j >= show_n_col:
                continue
            nid = f"{col_prefix}_{j}"
            label, fill, w, h = _h_block_params(layer)
            parts.append(flat_block(nid, label, fill, above_of=col_prev,
                                    node_distance=0.4, width=min(w + 1.0, 3.8), height=max(h, 0.85)))
            parts.append(flat_arrow(col_prev, nid))
            dim = _get_dim_text(layer)
            if dim:
                parts.append(flat_dim_label(dim, nid, side=dim_side))
            col_prev = nid
            node_map[j] = nid
            visible.append(nid)
        # Add ×N bracket if collapsed
        if n > show_n_col and visible:
            parts.append(group_frame(
                f"{col_prefix}_grp", visible,
                repeat=n, padding=0.25,
            ))
        return col_prev, node_map, visible

    left_prev, left_nodes, left_visible = _render_column(
        sbs.left, f"{prefix}_L", left_anchor, "left", show_n)
    right_prev, right_nodes, right_visible = _render_column(
        sbs.right, f"{prefix}_R", right_anchor, "right", show_n)
    all_nodes.extend(left_visible)
    all_nodes.extend(right_visible)

    # Cross-connections — draw only those referencing visible nodes
    if sbs.connections:
        for li, ri in sbs.connections:
            if li in left_nodes and ri in right_nodes:
                ln = left_nodes[li]
                rn = right_nodes[ri]
                parts.append(
                    rf"\draw[arrow, densely dashed, color=clrresidual] "
                    rf"({ln}.east) -- ({rn}.west);" "\n"
                )

    return left_prev, right_prev, all_nodes


def _render_bidir_flow(
    bf: BidirectionalFlow,
    prefix: str,
    prev: str | None,
    parts: list[str],
) -> tuple[str, list[str], bool]:
    """Render BidirectionalFlow as horizontal chain with forward/reverse arrows.

    Returns (last_node, all_node_names, has_own_io).
    has_own_io=True means the caller should skip default I/O arrows.
    """
    all_nodes: list[str] = []
    step_prev: str | None = None

    n_steps = len(bf.steps)
    for j, step_label in enumerate(bf.steps):
        nid = f"{prefix}_s{j}"
        # Gradient opacity + width: x_0=0.40 (clean, narrow) → x_T=1.0 (noisy, wide)
        frac = j / max(n_steps - 1, 1)
        step_opacity = 0.40 + 0.60 * frac
        step_width = 1.4 + 0.4 * frac
        if step_prev is None:
            if prev:
                parts.append(flat_block(nid, step_label, "embed", above_of=prev,
                                        node_distance=0.8, width=step_width, height=1.0,
                                        opacity=step_opacity))
                parts.append(flat_arrow(prev, nid))
            else:
                parts.append(flat_block(nid, step_label, "embed", position="(0,0)",
                                        width=step_width, height=1.0, opacity=step_opacity))
        else:
            parts.append(flat_block(nid, step_label, "embed", right_of=step_prev,
                                    node_distance=0.7, width=step_width, height=1.0,
                                    opacity=step_opacity))

        all_nodes.append(nid)
        step_prev = nid

    # Forward arrows (above)
    for j in range(len(bf.steps) - 1):
        fn = f"{prefix}_s{j}"
        tn = f"{prefix}_s{j+1}"
        label_part = ""
        if j == 0 and bf.forward_label:
            label_part = rf" node[pos=0.3, above=8pt, font=\sffamily\scriptsize, text=clrborder] {{{bf.forward_label}}}"
        parts.append(
            rf"\draw[arrow, color=clrattention] "
            rf"([yshift=3pt]{fn}.east) --{label_part} ([yshift=3pt]{tn}.west);" "\n"
        )

    # Reverse arrows (below)
    for j in range(len(bf.steps) - 1, 0, -1):
        fn = f"{prefix}_s{j}"
        tn = f"{prefix}_s{j-1}"
        label_part = ""
        if j == len(bf.steps) - 1 and bf.reverse_label:
            label_part = rf" node[pos=0.7, below=8pt, font=\sffamily\scriptsize, text=clrborder] {{{bf.reverse_label}}}"
        parts.append(
            rf"\draw[arrow, color=clrphysics] "
            rf"([yshift=-3pt]{fn}.west) --{label_part} ([yshift=-3pt]{tn}.east);" "\n"
        )

    # Add horizontal I/O arrows if this is a standalone flow (no prev)
    has_own_io = prev is None and len(all_nodes) >= 2
    if has_own_io:
        first = all_nodes[0]
        last = all_nodes[-1]
        parts.append(flat_io_arrow(first, direction="left", label="Input"))
        parts.append(flat_io_arrow(last, direction="right", label="Output"))

    return step_prev or "", all_nodes, has_own_io


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
    first_node: str | None = None
    node_idx = 0
    skip_default_io = False

    # Precomputed index: layer index → Group (replaces O(n) _find_group calls)
    group_index: dict[int, _Group] = {}
    for grp in groups:
        for idx in range(grp.start, grp.end):
            group_index[idx] = grp

    # DRY helpers -------------------------------------------------------
    def _connect(from_node: str | None, to_node: str) -> None:
        if from_node:
            parts.append(flat_arrow(from_node, to_node))

    # Track group frame data
    group_frame_data: dict[int, dict] = {}

    def _track_group(grp_id: int | None, grp: _Group | None, nodes: list[str]) -> None:
        if grp_id is not None:
            group_frame_data.setdefault(grp_id, {"count": grp.count, "nodes": []})
            group_frame_data[grp_id]["nodes"].extend(nodes)

    # Pre-compute which layers to show vs collapse
    # For pattern groups (pattern_len > 1), show_n counts pattern repetitions, not layers
    visible: dict[int, bool] = {}
    for i in range(len(layers)):
        grp = group_index.get(i)
        if grp is None:
            visible[i] = True
        else:
            # Which repetition does this layer belong to?
            offset_in_group = i - grp.start
            rep_index = offset_in_group // grp.pattern_len
            visible[i] = rep_index < show_n

    for i, layer in enumerate(layers):
        if not visible[i]:
            grp = group_index.get(i)
            if grp:
                grp_id = id(grp)
                if grp_id not in group_frame_data:
                    group_frame_data[grp_id] = {"count": grp.count, "nodes": []}
            continue

        grp = group_index.get(i)
        grp_id = id(grp) if grp else None

        # Add extra gap after a group ends (so frame doesn't overlap next layer)
        if prev and i > 0 and grp_id is None:
            prev_grp = group_index.get(i - 1)
            if prev_grp is not None:
                parts.append(
                    rf"\node[inner sep=0, minimum size=0, above=0.5cm of {prev}] (gap_{node_idx}) {{}};" "\n"
                )
                prev = f"gap_{node_idx}"

        if isinstance(layer, Embedding):
            nid = f"layer_{node_idx}"
            label = layer.label
            parts.append(flat_block(
                nid, label, "embed",
                position="(0,0)" if prev is None else None,
                above_of=prev, node_distance=0.6 if prev else 0,
            ))
            _connect(prev, nid)
            parts.append(flat_dim_label(_format_dim(layer.d_model), nid, side="left"))
            prev = nid
            # If rope=True and next layer is NOT PositionalEncoding, auto-add RoPE ⊕
            next_is_pos = (i + 1 < len(layers) and isinstance(layers[i + 1], PositionalEncoding))
            if layer.rope and not next_is_pos:
                rope_nid = f"layer_{node_idx}_rope"
                parts.append(flat_op_circle(
                    rope_nid, r"$\oplus$", above_of=nid, node_distance=0.3,
                    fill="clrresidual!30",
                ))
                parts.append(flat_arrow(nid, rope_nid))
                parts.append(flat_side_label("RoPE", rope_nid, side="right", distance=0.35))
                prev = rope_nid
            group_nodes = [nid]
            if layer.rope and not next_is_pos:
                group_nodes.append(rope_nid)
            _track_group(grp_id, grp, group_nodes)

        elif isinstance(layer, PositionalEncoding):
            nid = f"layer_{node_idx}"
            type_labels = {
                "rope": "RoPE",
                "learned": "Learned PE",
                "sinusoidal": "Sinusoidal PE",
                "alibi": "ALiBi",
            }
            label = layer.label or type_labels.get(layer.encoding_type, "Pos. Enc.")
            # ⊕ circle with side label (paper-style positional embedding addition)
            parts.append(flat_op_circle(
                nid, r"$\oplus$", above_of=prev, node_distance=0.3,
                fill="clrresidual!30",
            ))
            _connect(prev, nid)
            parts.append(flat_side_label(label, nid, side="right", distance=0.35))
            prev = nid

        elif isinstance(layer, TransformerBlock):
            # Add separator between consecutive transformer blocks
            if i > 0 and isinstance(layers[i - 1], TransformerBlock) and visible.get(i - 1, False):
                sep_name = f"sep_{node_idx}"
                parts.append(flat_separator(prev, sep_name))

            prefix = f"tb_{node_idx}"
            block_lines, last_node, block_nodes = _render_transformer_block(
                layer, prefix, prev, theme, block_idx=node_idx,
            )
            parts.extend(block_lines)
            prev = last_node
            _track_group(grp_id, grp, block_nodes)

        elif isinstance(layer, ConvBlock):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Conv{layer.kernel_size}×{layer.kernel_size}"
            conv_width = width_from_dim(layer.filters)
            parts.append(flat_block(
                nid, label, "attention",
                above_of=prev, node_distance=0.4, width=conv_width,
            ))
            _connect(prev, nid)
            parts.append(flat_dim_label(str(layer.filters), nid, side="left"))
            prev = nid
            if layer.pool:
                pid = f"pool_{node_idx}"
                pool_label = f"{'Max' if layer.pool == 'max' else 'Avg'}Pool"
                pool_width = max(conv_width * 0.7, 2.0)
                parts.append(flat_block(pid, pool_label, "ffn",
                                        above_of=nid, node_distance=0.25,
                                        width=pool_width, height=0.6))
                parts.append(flat_arrow(nid, pid))
                prev = pid

        elif isinstance(layer, PatchEmbedding):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Patch {layer.patch_size}x{layer.patch_size} + Linear Proj"
            parts.append(flat_block(
                nid, label, "embed",
                position="(0,0)" if prev is None else None,
                above_of=prev, node_distance=0.5 if prev else 0,
                width=4.0, height=0.9,
            ))
            _connect(prev, nid)
            parts.append(flat_dim_label(_format_dim(layer.d_model), nid, side="left"))
            prev = nid

        elif isinstance(layer, MBConvBlock):
            if i > 0 and isinstance(layers[i - 1], MBConvBlock) and visible.get(i - 1, False):
                sep_name = f"sep_{node_idx}"
                parts.append(flat_separator(prev, sep_name))
            prefix = f"mbconv_{node_idx}"
            block_lines, last_node, block_nodes = _render_mbconv_block(
                layer, prefix, prev, theme, block_idx=node_idx,
            )
            parts.extend(block_lines)
            prev = last_node
            _track_group(grp_id, grp, block_nodes)

        elif isinstance(layer, SwinBlock):
            if i > 0 and isinstance(layers[i - 1], SwinBlock) and visible.get(i - 1, False):
                sep_name = f"sep_{node_idx}"
                parts.append(flat_separator(prev, sep_name))
            prefix = f"swin_{node_idx}"
            block_lines, last_node, block_nodes = _render_swin_block(
                layer, prefix, prev, theme, block_idx=node_idx,
            )
            parts.extend(block_lines)
            prev = last_node
            _track_group(grp_id, grp, block_nodes)

        elif isinstance(layer, PatchMerging):
            nid = f"layer_{node_idx}"
            label = layer.label or "Patch Merging (2x downsample)"
            parts.append(flat_block(nid, label, "embed",
                                    above_of=prev, node_distance=0.4,
                                    width=3.6, height=0.7))
            _connect(prev, nid)
            parts.append(flat_dim_label(_format_dim(layer.d_model), nid, side="left"))
            prev = nid

        elif isinstance(layer, DenseLayer):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Dense ({layer.units})"
            parts.append(flat_block(nid, label, "dense",
                                    above_of=prev, node_distance=0.4))
            _connect(prev, nid)
            prev = nid
            _track_group(grp_id, grp, [nid])

        elif isinstance(layer, ClassificationHead):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "output",
                                    above_of=prev, node_distance=0.8))
            _connect(prev, nid)
            prev = nid

        elif isinstance(layer, FourierBlock):
            # Dual-path FNO visualization: spectral path + spatial bypass → merge
            prefix = f"fno_{node_idx}"
            entry = f"{prefix}_in"
            fft_node = f"{prefix}_fft"
            spectral = f"{prefix}_spectral"
            ifft_node = f"{prefix}_ifft"
            bypass = f"{prefix}_bypass"
            merge = f"{prefix}_merge"
            act = f"{prefix}_act"

            skip_xshift = 2.0 + (node_idx % 5) * 0.15

            # Entry anchor
            parts.append(
                rf"\node[inner sep=0, minimum size=0, above=0.5cm of {prev}] ({entry}) {{}};" "\n"
            )
            _connect(prev, entry)

            # Spectral path (main column)
            parts.append(flat_block(fft_node, r"$\mathcal{F}$ (FFT)", "spectral",
                                    above_of=entry, node_distance=0.15, width=3.2, height=0.65))
            parts.append(flat_arrow(entry, fft_node))
            parts.append(flat_block(spectral, f"Spectral Conv (modes={layer.modes})", "spectral",
                                    above_of=fft_node, node_distance=0.25, width=3.8, height=0.85))
            parts.append(flat_arrow(fft_node, spectral))
            parts.append(flat_dim_label(_format_dim(layer.width), spectral, side="left", distance=0.25))
            parts.append(flat_block(ifft_node, r"$\mathcal{F}^{-1}$ (iFFT)", "spectral",
                                    above_of=spectral, node_distance=0.25, width=3.2, height=0.65))
            parts.append(flat_arrow(spectral, ifft_node))

            # Merge (⊕) — bypass joins here
            parts.append(flat_op_circle(merge, r"$\oplus$", above_of=ifft_node, node_distance=0.25))
            parts.append(flat_arrow(ifft_node, merge))
            # Spatial bypass skip arrow
            parts.append(flat_skip_arrow(entry, merge, xshift=skip_xshift))
            parts.append(flat_side_label(r"$W \cdot x$ (bypass)", merge, side="right", distance=skip_xshift + 0.3))

            # Activation
            parts.append(flat_block(act, r"$\sigma$ (GeLU)", "ffn",
                                    above_of=merge, node_distance=0.25, width=2.8, height=0.6, opacity=0.7))
            parts.append(flat_arrow(merge, act))

            prev = act
            group_nodes = [fft_node, spectral, ifft_node, merge, act]
            _track_group(grp_id, grp, group_nodes)

        # --- Regularization / Activation ---

        elif isinstance(layer, Dropout):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Dropout ({layer.rate})"
            parts.append(flat_block(nid, label, "border", above_of=prev,
                                    node_distance=0.2, width=3.0, height=0.55, opacity=0.25))
            _connect(prev, nid)
            prev = nid

        elif isinstance(layer, Activation):
            nid = f"layer_{node_idx}"
            fn_labels = {"relu": "ReLU", "gelu": "GELU", "swish": "Swish/SiLU",
                         "mish": "Mish", "sigmoid": "Sigmoid", "tanh": "Tanh",
                         "softmax": "Softmax", "leaky_relu": "LeakyReLU", "selu": "SELU"}
            label = layer.label or fn_labels.get(layer.function, layer.function)
            parts.append(flat_block(nid, label, "ffn", above_of=prev,
                                    node_distance=0.2, width=3.0, height=0.6, opacity=0.6))
            _connect(prev, nid)
            prev = nid

        # --- Normalization variants ---

        elif isinstance(layer, (BatchNorm, RMSNorm)):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "norm", above_of=prev,
                                    node_distance=0.2, width=3.4, height=0.65))
            _connect(prev, nid)
            prev = nid

        elif isinstance(layer, AdaptiveLayerNorm):
            nid = f"layer_{node_idx}"
            label = f"{layer.label} ({layer.condition})"
            parts.append(flat_block(nid, label, "norm", above_of=prev,
                                    node_distance=0.3, width=3.4, height=0.7))
            _connect(prev, nid)
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
            _connect(prev, entry)
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
            _track_group(grp_id, grp, [conv1, conv2, add])

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
            _connect(prev, entry)
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
            _track_group(grp_id, grp, [c1, c2, c3, add])

        # --- GAN ---

        elif isinstance(layer, Generator):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "embed", above_of=prev,
                                    node_distance=0.4, height=1.0))
            _connect(prev, nid)
            parts.append(flat_dim_label(str(layer.channels), nid, side="left"))
            prev = nid

        elif isinstance(layer, Discriminator):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "physics", above_of=prev,
                                    node_distance=0.4, height=1.0))
            _connect(prev, nid)
            parts.append(flat_dim_label(str(layer.channels), nid, side="left"))
            prev = nid

        # --- VAE / Autoencoder ---

        elif isinstance(layer, EncoderBlock):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "attention", above_of=prev,
                                    node_distance=0.4, height=1.0))
            _connect(prev, nid)
            parts.append(flat_dim_label(_format_dim(layer.d_model), nid, side="left"))
            prev = nid

        elif isinstance(layer, DecoderBlock):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "attention_alt", above_of=prev,
                                    node_distance=0.4, height=1.0))
            _connect(prev, nid)
            parts.append(flat_dim_label(_format_dim(layer.d_model), nid, side="left"))
            prev = nid

        elif isinstance(layer, SamplingLayer):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "residual", above_of=prev,
                                    node_distance=0.3, width=3.2, height=0.7, opacity=0.6))
            _connect(prev, nid)
            prev = nid

        # --- Diffusion ---

        elif isinstance(layer, UNetBlock):
            nid = f"layer_{node_idx}"
            label = layer.label or f"UNet Block ({layer.filters})"
            if layer.with_attention:
                label += " + Attn"
            parts.append(flat_block(nid, label, "spectral", above_of=prev,
                                    node_distance=0.4, height=0.9))
            _connect(prev, nid)
            parts.append(flat_dim_label(str(layer.filters), nid, side="left"))
            prev = nid
            _track_group(grp_id, grp, [nid])

        elif isinstance(layer, NoiseHead):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "physics", above_of=prev,
                                    node_distance=0.4, width=3.4, height=0.8))
            _connect(prev, nid)
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
            _connect(prev, entry)
            parts.append(flat_block(proj, "Linear Proj", "dense", above_of=entry,
                                    node_distance=0.12, height=0.65))
            parts.append(flat_arrow(entry, proj))
            parts.append(flat_block(conv, "Conv1D", "attention_alt", above_of=proj,
                                    node_distance=0.25, height=0.65))
            parts.append(flat_arrow(proj, conv))
            parts.append(flat_block(ssm, "Selective SSM", "spectral", above_of=conv,
                                    node_distance=0.25, height=0.85))
            parts.append(flat_arrow(conv, ssm))
            parts.append(flat_dim_label(_format_dim(layer.d_model), ssm, side="left"))
            parts.append(flat_block(out, "Linear Out", "dense", above_of=ssm,
                                    node_distance=0.25, height=0.65))
            parts.append(flat_arrow(ssm, out))
            prev = out
            _track_group(grp_id, grp, [proj, conv, ssm, out])

        elif isinstance(layer, SelectiveSSM):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "spectral", above_of=prev,
                                    node_distance=0.4, height=0.9))
            _connect(prev, nid)
            parts.append(flat_dim_label(_format_dim(layer.d_model), nid, side="left"))
            prev = nid
            _track_group(grp_id, grp, [nid])

        # --- Recurrent (LSTM / GRU) ---

        elif isinstance(layer, LSTMBlock):
            if layer.style == "olah":
                prefix = f"lstm_olah_{node_idx}"
                last_node, olah_nodes = _render_lstm_olah(layer, prefix, prev, parts)
                prev = last_node
                # Olah cell renders its own h_t output horizontally — no default I/O
                skip_default_io = True
            elif layer.style == "compact":
                nid = f"layer_{node_idx}"
                label = layer.label or f"LSTM ({layer.hidden_size})"
                parts.append(flat_block(nid, label, "attention", above_of=prev,
                                        node_distance=0.4, height=1.0))
                _connect(prev, nid)
                parts.append(flat_dim_label(str(layer.hidden_size), nid, side="left"))
                prev = nid
                _track_group(grp_id, grp, [nid])
            else:  # "gates" — original vertical style
                if i > 0 and isinstance(layers[i - 1], LSTMBlock) and visible.get(i - 1, False):
                    sep_name = f"sep_{node_idx}"
                    parts.append(flat_separator(prev, sep_name))
                prefix = f"lstm_{node_idx}"
                block_lines, last_node, block_nodes = _render_lstm_block(
                    layer, prefix, prev, theme, block_idx=node_idx,
                )
                parts.extend(block_lines)
                prev = last_node
                _track_group(grp_id, grp, block_nodes)

        elif isinstance(layer, GRUBlock):
            if layer.style == "compact":
                nid = f"layer_{node_idx}"
                label = layer.label or f"GRU ({layer.hidden_size})"
                parts.append(flat_block(nid, label, "attention_alt", above_of=prev,
                                        node_distance=0.4, height=1.0))
                _connect(prev, nid)
                parts.append(flat_dim_label(str(layer.hidden_size), nid, side="left"))
                prev = nid
                _track_group(grp_id, grp, [nid])
            else:  # "gates" or "olah"
                if i > 0 and isinstance(layers[i - 1], GRUBlock) and visible.get(i - 1, False):
                    sep_name = f"sep_{node_idx}"
                    parts.append(flat_separator(prev, sep_name))
                prefix = f"gru_{node_idx}"
                block_lines, last_node, block_nodes = _render_gru_block(
                    layer, prefix, prev, theme, block_idx=node_idx,
                )
            parts.extend(block_lines)
            prev = last_node
            _track_group(grp_id, grp, block_nodes)

        # --- Mixture of Experts ---

        elif isinstance(layer, MoELayer):
            nid = f"layer_{node_idx}"
            label = layer.label or f"MoE (top-{layer.top_k}/{layer.num_experts})"
            router = f"{nid}_router"
            experts = f"{nid}_experts"
            parts.append(flat_block(router, f"Router (top-{layer.top_k})", "output",
                                    above_of=prev, node_distance=0.4, width=3.0, height=0.7))
            _connect(prev, router)
            parts.append(flat_block(experts, f"{layer.num_experts} Experts (FFN {layer.d_ff})",
                                    "dense", above_of=router, node_distance=0.25, height=0.9))
            parts.append(flat_arrow(router, experts))
            prev = experts
            _track_group(grp_id, grp, [router, experts])

        elif isinstance(layer, Router):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Router (top-{layer.top_k}/{layer.num_experts})"
            parts.append(flat_block(nid, label, "output", above_of=prev,
                                    node_distance=0.3, width=3.0, height=0.7))
            _connect(prev, nid)
            prev = nid

        elif isinstance(layer, Expert):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Expert FFN ({layer.d_ff})"
            parts.append(flat_block(nid, label, "dense", above_of=prev,
                                    node_distance=0.3, height=0.8))
            _connect(prev, nid)
            prev = nid
            _track_group(grp_id, grp, [nid])

        # --- Graph Neural Networks ---

        elif isinstance(layer, GraphConv):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "attention_alt", above_of=prev,
                                    node_distance=0.4, height=0.85))
            _connect(prev, nid)
            parts.append(flat_dim_label(str(layer.channels), nid, side="left"))
            prev = nid
            _track_group(grp_id, grp, [nid])

        elif isinstance(layer, MessagePassing):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Message Passing ({layer.aggregation})"
            parts.append(flat_block(nid, label, "spectral", above_of=prev,
                                    node_distance=0.4, height=0.85))
            _connect(prev, nid)
            prev = nid
            _track_group(grp_id, grp, [nid])

        elif isinstance(layer, GraphAttention):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.label, "attention", above_of=prev,
                                    node_distance=0.4, height=0.9))
            _connect(prev, nid)
            parts.append(flat_dim_label(f"{layer.heads}h", nid, side="left"))
            prev = nid
            _track_group(grp_id, grp, [nid])

        elif isinstance(layer, GraphPooling):
            nid = f"layer_{node_idx}"
            label = layer.label or f"Graph {layer.pool_type.title()}Pool"
            parts.append(flat_block(nid, label, "ffn", above_of=prev,
                                    node_distance=0.3, width=3.0, height=0.65))
            _connect(prev, nid)
            prev = nid

        # --- Freeform ---

        elif isinstance(layer, CustomBlock):
            nid = f"layer_{node_idx}"
            parts.append(flat_block(nid, layer.text, layer.color_role,
                                    above_of=prev, node_distance=0.4))
            _connect(prev, nid)
            prev = nid

        # --- Parallel / Branching ---

        elif isinstance(layer, EncoderDecoder):
            prefix = f"encdec_{node_idx}"
            dec_last, ed_nodes = _render_encoder_decoder(
                layer, prefix, prev, theme, parts, show_n=show_n,
            )
            prev = dec_last
            # EncoderDecoder always emits its own column I/O arrows
            skip_default_io = True

        elif isinstance(layer, SPINNBlock):
            prefix = f"spinn_{node_idx}"
            spinn_last, spinn_nodes = _render_spinn(layer, prefix, prev, parts)
            prev = spinn_last

        elif isinstance(layer, SideBySide):
            prefix = f"sbs_{node_idx}"
            left_last, right_last, sbs_nodes = _render_side_by_side(
                layer, prefix, prev, theme, parts, show_n=show_n,
            )
            # Merge the two columns into a single node
            merge = f"{prefix}_merge"
            parts.append(
                rf"\node[inner sep=0, minimum size=0] ({merge}) "
                rf"at ($({left_last}.north)!0.5!({right_last}.north) + (0,0.8)$) {{}};" "\n"
            )
            parts.append(flat_arrow(left_last, merge, from_anchor="north", to_anchor="south"))
            parts.append(flat_arrow(right_last, merge, from_anchor="north", to_anchor="south"))
            prev = merge

        elif isinstance(layer, BidirectionalFlow):
            prefix = f"bidir_{node_idx}"
            last_step, bf_nodes, bf_has_io = _render_bidir_flow(layer, prefix, prev, parts)
            prev = last_step
            if bf_has_io:
                skip_default_io = True

        elif isinstance(layer, ForkLoss):
            prefix = f"fork_{node_idx}"
            n_branches = len(layer.branches)
            branch_nodes: list[str] = []
            # Fork point
            fork_pt = f"{prefix}_fork"
            parts.append(
                rf"\node[inner sep=0, minimum size=0, above=0.4cm of {prev}] ({fork_pt}) {{}};" "\n"
            )
            _connect(prev, fork_pt)
            # Render branches side by side
            branch_sep = 4.5
            total_width = (n_branches - 1) * branch_sep
            for bi, (blabel, bcolor, bannot) in enumerate(layer.branches):
                x_offset = -total_width / 2 + bi * branch_sep
                bnid = f"{prefix}_b{bi}"
                parts.append(
                    rf"\node[block, fill=clr{bcolor}!85, minimum width=3.0cm, minimum height=0.85cm, "
                    rf"text=clrtext, above=1.2cm of {fork_pt}, xshift={x_offset}cm] "
                    rf"({bnid}) {{{_latex_escape(blabel)}}};" "\n"
                )
                parts.append(
                    rf"\draw[arrow] ({fork_pt}.north) -- ({bnid}.south);" "\n"
                )
                if bannot:
                    parts.append(flat_dim_label(bannot, bnid, side="right", distance=0.2))
                branch_nodes.append(bnid)
            # Merge node
            merge = f"{prefix}_merge"
            parts.append(
                rf"\node[circle, draw=clrborder, fill=clrresidual!30, inner sep=3pt, "
                rf"font=\sffamily\scriptsize\bfseries, "
                rf"above=0.8cm of {fork_pt}, yshift=2.0cm] ({merge}) "
                rf"{{{layer.merge_label}}};" "\n"
            )
            for bnid in branch_nodes:
                parts.append(
                    rf"\draw[arrow] ({bnid}.north) -- ({merge}.south);" "\n"
                )
            prev = merge

        elif isinstance(layer, DetailPanel):
            prefix = f"detail_{node_idx}"
            # Compact summary block in main column
            summary = f"{prefix}_summary"
            parts.append(flat_block(summary, layer.summary_label, layer.summary_color,
                                    above_of=prev, node_distance=0.5, width=3.8, height=1.0))
            _connect(prev, summary)
            prev = summary

            # Detail panel — shifted right of main diagram
            panel_x = 5.5
            detail_anchor = f"{prefix}_detail_anchor"
            parts.append(
                rf"\node[inner sep=0, minimum size=0, "
                rf"right={panel_x}cm of {summary}] ({detail_anchor}) {{}};" "\n"
            )

            # Title above detail panel
            if layer.title:
                parts.append(
                    rf"\node[above=0.2cm of {detail_anchor}, "
                    rf"font=\sffamily\small\bfseries, text=clrborder] "
                    rf"{{{_latex_escape(layer.title)}}};" "\n"
                )

            # Render detail layers vertically in panel
            detail_prev = detail_anchor
            detail_nodes: list[str] = []
            for di, dlayer in enumerate(layer.detail_layers):
                dnid = f"{prefix}_d{di}"
                dlabel, dfill, dw, dh = _h_block_params(dlayer)
                parts.append(flat_block(dnid, dlabel, dfill, above_of=detail_prev,
                                        node_distance=0.3, width=3.0, height=0.7))
                parts.append(flat_arrow(detail_prev, dnid))
                detail_prev = dnid
                detail_nodes.append(dnid)

            # Dashed leader line from summary to detail panel
            parts.append(
                rf"\draw[densely dashed, color=clrgroup_frame, line width=0.6pt, ->] "
                rf"({summary}.east) -- ({detail_anchor}.west);" "\n"
            )

            # Frame around detail panel
            if detail_nodes:
                parts.append(group_frame(
                    f"{prefix}_detail_frame", detail_nodes,
                    title=layer.title or "Detail", padding=0.25,
                ))

        # --- Structural ---

        elif isinstance(layer, Separator):
            nid = f"sep_label_{node_idx}"
            if prev is None:
                # First element — place at origin
                parts.append(
                    rf"\node[inner sep=0, minimum size=0] ({nid}_anchor) at (0,0) {{}};" "\n"
                )
                parts.append(flat_separator_label(nid, layer.label, f"{nid}_anchor",
                                                   style=layer.style))
            else:
                parts.append(flat_separator_label(nid, layer.label, prev,
                                                   style=layer.style))
            prev = nid

        elif isinstance(layer, SectionHeader):
            nid = f"section_{node_idx}"
            if prev is None:
                parts.append(
                    rf"\node[inner sep=0, minimum size=0] ({nid}_anchor) at (0,0) {{}};" "\n"
                )
                parts.append(flat_section_header(nid, layer.title, f"{nid}_anchor",
                                                  subtitle=layer.subtitle))
            else:
                parts.append(flat_section_header(nid, layer.title, prev,
                                                  subtitle=layer.subtitle))
            prev = nid

        # Track the first rendered node for input arrow (BUG-01/02 fix)
        if first_node is None and prev is not None:
            first_node = prev

        node_idx += 1

    # Input/output arrows (skip if a component already added its own)
    if not skip_default_io:
        if first_node:
            parts.append(flat_io_arrow(first_node, direction="below", label="Input"))
        if prev:
            parts.append(flat_io_arrow(prev, direction="above", label="Output"))

    # Draw group frames
    for idx, (gid, gdata) in enumerate(group_frame_data.items()):
        if gdata["nodes"]:
            parts.append(group_frame(
                f"grp_{idx}",
                gdata["nodes"],
                repeat=gdata["count"],
                padding=0.35,
            ))

    # Title
    parts.append(flat_title(rf"\textbf{{{name}}}"))

    parts.append(flat_end())
    return "".join(parts)
