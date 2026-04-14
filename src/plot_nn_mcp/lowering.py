"""DSL Layer → IROp lowering (Phase 4b).

Converts the legacy dataclass-based ``Layer`` types into composable
``IROp`` trees. The key payoff is that complex blocks (Transformer, MoE,
ResNet bottleneck) all reduce to ``IRResidualOp(body=[...])`` — one
implementation, no per-block residual logic.

This module is intentionally side-effect-free: it does NOT touch the
legacy ``_render_vertical`` path. Wiring is Phase 4c.

Coverage matrix (mark ⚠ where structural fix is intended vs current):
- Embedding, PositionalEncoding, ConvBlock, DenseLayer, ClassificationHead — 1:1
- TransformerBlock — composes 1-2 IRResidualOp (was monolithic 107 lines)
- MoELayer — IRResidualOp ⚠ (was missing residual; fixes Mixtral)
- ResidualBlock, BottleneckBlock — IRResidualOp (was per-block code)
- Norm/Activation/Dropout — IRBlockOp leaf
"""

from __future__ import annotations

from . import dsl
from .ir import (
    IRBlockOp,
    IRBuilder,
    IRCustomOp,
    IREdge,
    IRGraph,
    IRNode,
    IROp,
    IRResidualOp,
    IRSequenceOp,
)
from .themes import Role


def _ffn_label(block: "dsl.TransformerBlock") -> str:
    table = {"gelu": "FFN (GeLU)", "geglu": "FFN (GeGLU)",
             "swiglu": "FFN (SwiGLU)", "relu": "FFN (ReLU)"}
    return table.get(block.ffn, "FFN")


def _attn_label(block: "dsl.TransformerBlock") -> str:
    table = {"self": "Self-Attention", "masked": "Masked Self-Attn",
             "cross": "Cross-Attention", "local": "Local Attention",
             "global": "Global Attention", "gqa": f"GQA ({block.kv_heads}kv)",
             "mqa": "MQA (1kv)"}
    return block.label or table.get(block.attention, "Self-Attention")


def transformer_block_to_ir(block: "dsl.TransformerBlock") -> IROp:
    """Compose a transformer block from two residuals (attn + FFN).

    Pre-LN: Norm → Attn → +skip ; Norm → FFN → +skip
    Post-LN: Attn → +skip → Norm ; FFN → +skip → Norm
    The same shape works for Mixtral if we replace FFN with MoE — see
    ``moe_layer_to_ir`` returning the same residual structure.
    """
    is_pre = block.norm == "pre_ln"
    attn_role = Role.ATTENTION if block.attention == "global" else Role.ATTENTION_ALT
    attn_op = IRBlockOp(role=attn_role, label=_attn_label(block))
    ffn_op = IRBlockOp(role=Role.FFN, label=_ffn_label(block))
    norm = lambda: IRBlockOp(role=Role.NORM, label="LayerNorm")

    if is_pre:
        attn_residual = IRResidualOp(body=[norm(), attn_op])
        if block.skip_ffn:
            return attn_residual
        ffn_residual = IRResidualOp(body=[norm(), ffn_op])
        return IRSequenceOp(ops=[attn_residual, ffn_residual])

    # post_ln
    attn_residual = IRResidualOp(body=[attn_op])
    if block.skip_ffn:
        return IRSequenceOp(ops=[attn_residual, norm()])
    ffn_residual = IRResidualOp(body=[ffn_op])
    return IRSequenceOp(ops=[attn_residual, norm(), ffn_residual, norm()])


def moe_layer_to_ir(layer: "dsl.MoELayer") -> IROp:
    """Mixtral-style MoE block — wraps Router+Experts in a residual.

    Fixes the regression where the legacy renderer drew Router→Experts
    without ``add2``: by reusing :class:`IRResidualOp`, the skip is added
    structurally, not optionally.
    """
    return IRResidualOp(body=[
        IRBlockOp(role=Role.NORM, label="LayerNorm"),
        IRBlockOp(role=Role.OUTPUT, label=f"Router (top-{layer.top_k})",
                  size_hint=(3.0, 0.7)),
        IRBlockOp(role=Role.FFN,
                  label=f"{layer.num_experts} Experts (FFN {layer.d_ff})",
                  size_hint=(3.8, 0.9)),
    ])


def conv_block_to_ir(block: "dsl.ConvBlock") -> IROp:
    """Conv (+ optional pool). Width derived from filter count.

    Note: width rounding to 2 decimals happens in the emitter (render.py),
    so the lowering can pass raw floats — it never reaches TikZ unrounded.
    """
    from .flat_renderer import width_from_dim

    label = block.label or f"Conv{block.kernel_size}x{block.kernel_size}"
    conv_op = IRBlockOp(
        role=Role.ATTENTION,
        label=label,
        size_hint=(width_from_dim(block.filters), 0.85),
        dim=block.filters,
    )
    if block.pool:
        pool_label = f"{'Max' if block.pool == 'max' else 'Avg'}Pool"
        pool_w = max(width_from_dim(block.filters) * 0.7, 2.0)
        return IRSequenceOp(ops=[
            conv_op,
            IRBlockOp(role=Role.FFN, label=pool_label,
                      size_hint=(pool_w, 0.6)),
        ])
    return conv_op


def residual_block_to_ir(block: "dsl.ResidualBlock") -> IROp:
    """Two convs + skip — the canonical ResNet building block."""
    label = block.label or f"Conv{block.kernel_size}x{block.kernel_size}"
    return IRResidualOp(body=[
        IRBlockOp(role=Role.ATTENTION,
                  label=f"{label} + BN + ReLU",
                  size_hint=(3.8, 0.7), dim=block.filters),
        IRBlockOp(role=Role.ATTENTION,
                  label=f"{label} + BN",
                  size_hint=(3.8, 0.7), dim=block.filters),
    ])


def embedding_to_ir(layer: "dsl.Embedding") -> IROp:
    block = IRBlockOp(role=Role.EMBED, label=layer.label,
                      dim=layer.d_model)
    if not layer.rope:
        return block
    # Auto-add RoPE ⊕ when the embedding requests it, matching legacy
    # behavior (Embedding(rope=True) implies a downstream rotary PE).
    return IRSequenceOp(ops=[
        block,
        IRBlockOp(role=Role.RESIDUAL, label="RoPE", shape="circle"),
    ])


def positional_encoding_to_ir(layer: "dsl.PositionalEncoding") -> IROp:
    """⊕ circle for additive PE — same shape as a residual sink."""
    type_labels = {
        "rope": "RoPE", "learned": "Learned PE",
        "sinusoidal": "Sinusoidal PE", "alibi": "ALiBi",
    }
    label = layer.label or type_labels.get(layer.encoding_type, "Pos. Enc.")
    return IRBlockOp(role=Role.RESIDUAL, label=label, shape="circle")


def dense_to_ir(layer: "dsl.DenseLayer") -> IROp:
    return IRBlockOp(role=Role.DENSE,
                     label=layer.label or f"Dense ({layer.units})",
                     size_hint=(2.0, 1.0))


def classification_head_to_ir(layer: "dsl.ClassificationHead") -> IROp:
    return IRBlockOp(role=Role.OUTPUT, label=layer.label,
                     size_hint=(2.0, 1.2))


def norm_to_ir(layer) -> IROp:
    """BatchNorm/RMSNorm/AdaptiveLayerNorm — all leaf NORM blocks."""
    label = getattr(layer, "label", "Norm")
    return IRBlockOp(role=Role.NORM, label=label, size_hint=(3.4, 0.65))


def activation_to_ir(layer: "dsl.Activation") -> IROp:
    label = layer.label or layer.function.upper()
    return IRBlockOp(role=Role.NORM, label=label, size_hint=(2.4, 0.6))


def dropout_to_ir(layer: "dsl.Dropout") -> IROp:
    label = layer.label or f"Dropout ({layer.rate})"
    return IRBlockOp(role=Role.NORM, label=label, size_hint=(2.4, 0.6))


def patch_embedding_to_ir(layer: "dsl.PatchEmbedding") -> IROp:
    label = layer.label or f"Patch {layer.patch_size}x{layer.patch_size} + Linear Proj"
    return IRBlockOp(role=Role.EMBED, label=label,
                     size_hint=(4.0, 0.9), dim=layer.d_model)


def patch_merging_to_ir(layer: "dsl.PatchMerging") -> IROp:
    label = layer.label or "Patch Merging (2x downsample)"
    return IRBlockOp(role=Role.EMBED, label=label,
                     size_hint=(3.6, 0.7), dim=layer.d_model)


def fourier_block_to_ir(layer: "dsl.FourierBlock") -> IROp:
    """Same wording as legacy renderer so shared assertions pass."""
    label = layer.label or f"Spectral Conv (modes={layer.modes})"
    return IRBlockOp(role=Role.SPECTRAL, label=label,
                     size_hint=(2.6, 0.85), dim=layer.width)


def bottleneck_block_to_ir(block: "dsl.BottleneckBlock") -> IROp:
    """1x1 → 3x3 → 1x1 + skip — the canonical ResNet bottleneck.

    Reduction filters = ``filters // expansion`` per ResNet convention.
    """
    reduced = max(block.filters // block.expansion, 1)
    return IRResidualOp(body=[
        IRBlockOp(role=Role.ATTENTION_ALT,
                  label=f"1x1 Conv ({reduced})",
                  size_hint=(3.8, 0.6)),
        IRBlockOp(role=Role.ATTENTION,
                  label=f"3x3 Conv ({reduced})",
                  size_hint=(3.8, 0.7)),
        IRBlockOp(role=Role.ATTENTION_ALT,
                  label=f"1x1 Conv ({block.filters})",
                  size_hint=(3.8, 0.6)),
    ])


def mbconv_block_to_ir(block: "dsl.MBConvBlock") -> IROp:
    """EfficientNet inverted residual: expand → depthwise → SE? → project + skip."""
    label = block.label or f"MBConv {block.expansion}x ({block.filters})"
    body = [
        IRBlockOp(role=Role.ATTENTION_ALT,
                  label=f"Expand 1x1 ({block.filters * block.expansion})",
                  size_hint=(3.8, 0.6)),
        IRBlockOp(role=Role.ATTENTION,
                  label=f"DW Conv {block.kernel_size}x{block.kernel_size}",
                  size_hint=(3.8, 0.7)),
    ]
    if block.se:
        body.append(IRBlockOp(role=Role.OUTPUT, label="Squeeze-Excitation",
                              size_hint=(3.0, 0.55)))
    body.append(IRBlockOp(role=Role.ATTENTION_ALT,
                          label=f"Project 1x1 ({block.filters})",
                          size_hint=(3.8, 0.6)))
    return IRResidualOp(body=body)


def swin_block_to_ir(block: "dsl.SwinBlock") -> IROp:
    """Swin Transformer block: LN → W-MSA/SW-MSA → +skip ; LN → MLP → +skip."""
    msa_label = "W-MSA (regular)" if block.window_type == "regular" else "SW-MSA (shifted)"
    return IRSequenceOp(ops=[
        IRResidualOp(body=[
            IRBlockOp(role=Role.NORM, label="LayerNorm", size_hint=(3.0, 0.6)),
            IRBlockOp(role=Role.ATTENTION, label=msa_label,
                      size_hint=(4.2, 0.95)),
        ]),
        IRResidualOp(body=[
            IRBlockOp(role=Role.NORM, label="LayerNorm", size_hint=(3.0, 0.6)),
            IRBlockOp(role=Role.FFN, label="MLP",
                      dim=block.d_model * 4),
        ]),
    ])


def router_to_ir(layer: "dsl.Router") -> IROp:
    label = layer.label or f"Router (top-{layer.top_k}/{layer.num_experts})"
    return IRBlockOp(role=Role.OUTPUT, label=label, size_hint=(3.0, 0.7))


def expert_to_ir(layer: "dsl.Expert") -> IROp:
    label = layer.label or f"Expert FFN ({layer.d_ff})"
    return IRBlockOp(role=Role.DENSE, label=label, size_hint=(3.8, 0.8))


def graph_conv_to_ir(layer: "dsl.GraphConv") -> IROp:
    return IRBlockOp(role=Role.ATTENTION_ALT, label=layer.label,
                     dim=layer.channels, size_hint=(3.8, 0.85))


def message_passing_to_ir(layer: "dsl.MessagePassing") -> IROp:
    label = layer.label or f"Message Passing ({layer.aggregation})"
    return IRBlockOp(role=Role.ATTENTION_ALT, label=label,
                     size_hint=(3.8, 0.85))


def graph_attention_to_ir(layer: "dsl.GraphAttention") -> IROp:
    return IRBlockOp(role=Role.ATTENTION, label=layer.label,
                     size_hint=(3.8, 0.85))


def graph_pooling_to_ir(layer: "dsl.GraphPooling") -> IROp:
    label = layer.label or f"Graph Pool ({layer.pool_type})"
    return IRBlockOp(role=Role.OUTPUT, label=label, size_hint=(3.0, 0.7))


def custom_block_to_ir(layer: "dsl.CustomBlock") -> IROp:
    """Freeform user-supplied label and color role (legacy escape hatch)."""
    # ``color_role`` is a string matching a Role member; resolve_fill in the
    # emitter accepts strings, so we pass through.
    try:
        role = Role(layer.color_role)
    except ValueError:
        role = Role.DENSE
    return IRBlockOp(role=role, label=layer.text, size_hint=(3.8, 0.85))


def selective_ssm_to_ir(layer: "dsl.SelectiveSSM") -> IROp:
    return IRBlockOp(role=Role.SPECTRAL, label=layer.label,
                     dim=layer.d_model, size_hint=(3.8, 0.9))


def sampling_layer_to_ir(layer: "dsl.SamplingLayer") -> IROp:
    label = getattr(layer, "label", None) or "Sampling"
    return IRBlockOp(role=Role.OUTPUT, label=label, size_hint=(3.0, 0.85))


def noise_head_to_ir(layer: "dsl.NoiseHead") -> IROp:
    label = getattr(layer, "label", None) or "Noise"
    return IRBlockOp(role=Role.PHYSICS, label=label, size_hint=(3.0, 0.85))


def encoder_block_to_ir(layer: "dsl.EncoderBlock") -> IROp:
    return IRBlockOp(role=Role.ATTENTION, label=layer.label,
                     size_hint=(2.2, 1.4))


def decoder_block_to_ir(layer: "dsl.DecoderBlock") -> IROp:
    return IRBlockOp(role=Role.ATTENTION_ALT, label=layer.label,
                     size_hint=(2.2, 1.4))


# --- Structural layers (Phase 6) -------------------------------------------

def section_header_to_ir(layer: "dsl.SectionHeader") -> IROp:
    """Section title — renders as a thin rule + bold label, NOT a block.

    Marked with shape="section_rule" so the IR builder's input-detection
    skips it (fixes the Input→SectionHeader bug structurally).
    """
    return IRBlockOp(role=Role.OUTPUT, label=layer.title,
                     shape="section_rule", size_hint=(4.4, 0.0))


def separator_to_ir(layer: "dsl.Separator") -> IROp:
    """Thin separator — same shape category as section_rule."""
    label = getattr(layer, "label", "") or ""
    return IRBlockOp(role=Role.OUTPUT, label=label,
                     shape="section_rule", size_hint=(4.4, 0.0))


# --- Compact escape lowering for complex multi-block layers ---------------
# These render as single boxes via IR (a regression vs legacy multi-block
# rendering), but they unblock the architectures from going through the IR
# path. Phase 7+ will replace them with proper composable lowerings.

def _compact_box(role: Role, label: str, size: tuple[float, float] = (3.8, 0.85)) -> IROp:
    return IRBlockOp(role=role, label=label, size_hint=size)


# Legacy compact single-box fallback for non-"gates" styles.
def _lstm_compact(layer: "dsl.LSTMBlock") -> IROp:
    bi = "Bi-" if layer.bidirectional else ""
    return _compact_box(Role.ATTENTION_ALT,
                        layer.label or f"{bi}LSTM ({layer.hidden_size})")


def _gru_compact(layer: "dsl.GRUBlock") -> IROp:
    bi = "Bi-" if layer.bidirectional else ""
    return _compact_box(Role.ATTENTION_ALT,
                        layer.label or f"{bi}GRU ({layer.hidden_size})")


# Backwards-compat aliases (older tests/scripts may import these names).
lstm_block_compact_to_ir = _lstm_compact
gru_block_compact_to_ir = _gru_compact


def lstm_block_to_ir(layer: "dsl.LSTMBlock") -> IROp:
    """LSTMBlock → detailed gate view (concat → forget/input → ⊕(skip) → output → h_t).

    Matches legacy ``_render_lstm_block`` structure: the cell update ⊕
    receives a skip from the block entry (conveyor belt), which maps
    cleanly onto ``IRResidualOp(body=[concat, forget, input])``.
    ``style="compact"`` or ``style="olah"`` fall back to the single-box
    lowering — horizontal belt style needs layout support not available
    from vertical IR emission.
    """
    if layer.style != "gates":
        return _lstm_compact(layer)

    concat = IRBlockOp(role=Role.NORM,
                       label=r"Concat [$h_{t-1}$, $x_t$]",
                       size_hint=(3.4, 0.6))
    fg = IRBlockOp(role=Role.ATTENTION,
                   label=r"Forget Gate ($\sigma$)",
                   size_hint=(3.8, 0.7), dim=layer.hidden_size)
    ig = IRBlockOp(role=Role.ATTENTION_ALT,
                   label=r"Input Gate ($\sigma$ + tanh)",
                   size_hint=(3.8, 0.7))
    og = IRBlockOp(role=Role.ATTENTION,
                   label=r"Output Gate ($\sigma$)",
                   size_hint=(3.8, 0.7))
    ht = IRBlockOp(role=Role.FFN,
                   label=r"$h_t = o_t \odot \tanh(C_t)$",
                   size_hint=(4.0, 0.7))
    seq: list[IROp] = [
        IRResidualOp(body=[concat, fg, ig]),  # → ⊕ at end
        og,
        ht,
    ]
    if layer.bidirectional:
        seq.append(IRBlockOp(
            role=Role.RESIDUAL,
            label=r"Bi-Merge (concat $\rightarrow \leftarrow$)",
            size_hint=(4.4, 0.7),
            dim=2 * layer.hidden_size,
        ))
    return IRSequenceOp(ops=seq)


def gru_block_to_ir(layer: "dsl.GRUBlock") -> IROp:
    """GRUBlock → detailed gate view mirroring legacy _render_gru_block.

    Residual wraps (concat → reset → update → candidate) so the ⊕ at the
    end receives the conveyor-belt skip. Same style fallback as LSTM.
    """
    if layer.style != "gates":
        return _gru_compact(layer)

    concat = IRBlockOp(role=Role.NORM,
                       label=r"Concat [$h_{t-1}$, $x_t$]",
                       size_hint=(3.4, 0.6))
    rg = IRBlockOp(role=Role.ATTENTION_ALT,
                   label=r"Reset Gate ($\sigma$)",
                   size_hint=(3.8, 0.7), dim=layer.hidden_size)
    ug = IRBlockOp(role=Role.ATTENTION,
                   label=r"Update Gate ($\sigma$)",
                   size_hint=(3.8, 0.7))
    cand = IRBlockOp(role=Role.FFN,
                     label=r"Candidate $\tilde{h}_t$ (tanh)",
                     size_hint=(3.8, 0.7))
    seq: list[IROp] = [IRResidualOp(body=[concat, rg, ug, cand])]
    if layer.bidirectional:
        seq.append(IRBlockOp(
            role=Role.RESIDUAL,
            label=r"Bi-Merge (concat $\rightarrow \leftarrow$)",
            size_hint=(4.4, 0.7),
            dim=2 * layer.hidden_size,
        ))
    return IRSequenceOp(ops=seq) if len(seq) > 1 else seq[0]


def mamba_block_to_ir(layer: "dsl.MambaBlock") -> IROp:
    """MambaBlock → Linear-in → Selective SSM → Linear-out.

    Legacy renderer drew MambaBlock as one compact box; this lowering
    exposes the internal structure (same shapes the SelectiveSSM lowering
    uses, so docs cross-reference).
    """
    base = layer.label or "Mamba"
    return IRSequenceOp(ops=[
        IRBlockOp(role=Role.DENSE, label=f"{base} In-Proj",
                  size_hint=(3.4, 0.6), dim=layer.d_model),
        IRBlockOp(role=Role.SPECTRAL,
                  label=f"Selective SSM (state={layer.d_state})",
                  size_hint=(3.8, 0.85), dim=layer.d_model),
        IRBlockOp(role=Role.DENSE, label=f"{base} Out-Proj",
                  size_hint=(3.4, 0.6), dim=layer.d_model),
    ])


def generator_compact_to_ir(layer: "dsl.Generator") -> IROp:
    label = getattr(layer, "label", None) or "Generator"
    return _compact_box(Role.EMBED, label, size=(4.0, 1.0))


def discriminator_compact_to_ir(layer: "dsl.Discriminator") -> IROp:
    label = getattr(layer, "label", None) or "Discriminator"
    return _compact_box(Role.PHYSICS, label, size=(4.0, 1.0))


def unet_block_compact_to_ir(layer: "dsl.UNetBlock") -> IROp:
    label = getattr(layer, "label", None) or "U-Net Block"
    return _compact_box(Role.ATTENTION, label)


# --- Parallel / Branching (Task 5) -----------------------------------------

from .ir import IRParallelOp  # noqa: E402  (kept near its usage for context)


def _layers_to_sequence(layers: list) -> IROp:
    """Lower a list of legacy Layer instances into a single IROp.

    Used by ``SideBySide`` / ``BidirectionalFlow`` / ``EncoderDecoder``
    which carry sub-lists of layers per branch.
    """
    ops = [layer_to_ir(l) for l in layers]
    if not ops:
        return _compact_box(Role.RESIDUAL, "(empty)", size=(2.0, 0.5))
    if len(ops) == 1:
        return ops[0]
    return IRSequenceOp(ops=ops)


def side_by_side_to_ir(layer: "dsl.SideBySide") -> IROp:
    """Two parallel vertical stacks (encoder/decoder, branch/trunk, etc.).

    Each side is lowered as a SequenceOp, then composed into a single
    ``IRParallelOp`` so the emitter places the columns side-by-side.
    """
    return IRParallelOp(
        branches=[
            _layers_to_sequence(layer.left),
            _layers_to_sequence(layer.right),
        ],
        merge="none",
    )


def bidirectional_flow_to_ir(layer: "dsl.BidirectionalFlow") -> IROp:
    """Forward + backward stacks; rendered as parallel branches.

    The reverse data flow of the backward stack is indicated by label
    hints (``→``/``←``) in the underlying layer labels rather than by
    reversed edges — that would require dedicated layout support.
    """
    forward = getattr(layer, "forward", None) or getattr(layer, "left", [])
    backward = getattr(layer, "backward", None) or getattr(layer, "right", [])
    return IRParallelOp(
        branches=[
            _layers_to_sequence(list(forward)),
            _layers_to_sequence(list(backward)),
        ],
        merge="concat",
    )


def fork_loss_to_ir(layer: "dsl.ForkLoss") -> IROp:
    """Multi-output loss heads — one branch per loss."""
    heads = getattr(layer, "heads", None) or []
    if not heads:
        return _compact_box(Role.OUTPUT,
                            getattr(layer, "label", None) or "Loss")
    branches = [
        IRBlockOp(Role.OUTPUT,
                  getattr(h, "label", None) or f"Loss {i}",
                  size_hint=(2.8, 0.8))
        for i, h in enumerate(heads)
    ]
    return IRParallelOp(branches=branches, merge="none")


def detail_panel_to_ir(layer: "dsl.DetailPanel") -> IROp:
    """Auxiliary panel — rendered as a single wide annotation block."""
    text = getattr(layer, "text", None) or getattr(layer, "label", "Detail")
    return IRBlockOp(role=Role.NORM, label=text, size_hint=(4.6, 1.2))


def encoder_decoder_to_ir(layer: "dsl.EncoderDecoder") -> IROp:
    """Vaswani/T5-style encoder-decoder as two parallel sub-stacks.

    Cross-attention edges are NOT yet rendered (would require dedicated
    layout support — see Task 5 in REFACTOR_HANDOFF.md). Current behavior
    matches SideBySide but uses the layer's encoder_label/decoder_label
    for group titles once Group.title rendering covers parallel ops.
    """
    enc_layers = list(getattr(layer, "encoder_input", []) or []) + list(layer.encoder)
    dec_layers = list(getattr(layer, "decoder_input", []) or []) + list(layer.decoder)
    return IRParallelOp(
        branches=[
            _layers_to_sequence(enc_layers),
            _layers_to_sequence(dec_layers),
        ],
        merge="none",
    )


def spinn_block_to_ir(layer: "dsl.SPINNBlock") -> IROp:
    """Three-zone SPINN block (Buffer | Stack | Tracker)."""
    return IRParallelOp(
        branches=[
            IRBlockOp(Role.EMBED, "Buffer", size_hint=(2.6, 0.8)),
            IRBlockOp(Role.ATTENTION, "Stack", size_hint=(2.6, 0.8)),
            IRBlockOp(Role.RESIDUAL, "Tracker", size_hint=(2.6, 0.8)),
        ],
        merge="none",
    )


def unet_level_to_ir(layer: "dsl.UNetLevel") -> IROp:
    """One U-Net level: encoder + decoder with a skip.

    Proper U-shape layout requires Task 3 (dedicated U-Net layout);
    for now render as two parallel branches.
    """
    enc_layers = list(getattr(layer, "encoder", []) or [])
    dec_layers = list(getattr(layer, "decoder", []) or [])
    if not enc_layers and not dec_layers:
        return _compact_box(Role.ATTENTION,
                            getattr(layer, "label", None) or "U-Net Level")
    return IRParallelOp(
        branches=[
            _layers_to_sequence(enc_layers),
            _layers_to_sequence(dec_layers),
        ],
        merge="concat",
    )


def bottleneck_to_ir(layer: "dsl.Bottleneck") -> IROp:
    """U-Net bottleneck — single wide block at the U's bottom."""
    label = getattr(layer, "label", None) or "Bottleneck"
    return IRBlockOp(role=Role.SPECTRAL, label=label, size_hint=(4.6, 0.9))


# Single dispatch table — adding a new layer type means adding one entry.
_LOWERING: dict[type, callable] = {}


def register(layer_type: type, fn: callable) -> None:
    _LOWERING[layer_type] = fn


def layer_to_ir(layer) -> IROp:
    """Convert a single legacy DSL layer instance to an IROp.

    Raises :class:`NotImplementedError` for layers that have not yet been
    migrated — Phase 4c will extend coverage to all 50+ types.
    """
    fn = _LOWERING.get(type(layer))
    if fn is None:
        raise NotImplementedError(
            f"no IR lowering registered for {type(layer).__name__}"
        )
    return fn(layer)


def _register_defaults() -> None:
    register(dsl.Embedding, embedding_to_ir)
    register(dsl.TransformerBlock, transformer_block_to_ir)
    register(dsl.MoELayer, moe_layer_to_ir)
    register(dsl.ConvBlock, conv_block_to_ir)
    register(dsl.ResidualBlock, residual_block_to_ir)
    register(dsl.DenseLayer, dense_to_ir)
    register(dsl.ClassificationHead, classification_head_to_ir)
    register(dsl.BatchNorm, norm_to_ir)
    register(dsl.RMSNorm, norm_to_ir)
    register(dsl.AdaptiveLayerNorm, norm_to_ir)
    register(dsl.Activation, activation_to_ir)
    register(dsl.Dropout, dropout_to_ir)
    register(dsl.PatchEmbedding, patch_embedding_to_ir)
    register(dsl.PatchMerging, patch_merging_to_ir)
    register(dsl.FourierBlock, fourier_block_to_ir)
    register(dsl.BottleneckBlock, bottleneck_block_to_ir)
    register(dsl.MBConvBlock, mbconv_block_to_ir)
    register(dsl.SwinBlock, swin_block_to_ir)
    register(dsl.Router, router_to_ir)
    register(dsl.Expert, expert_to_ir)
    register(dsl.GraphConv, graph_conv_to_ir)
    register(dsl.MessagePassing, message_passing_to_ir)
    register(dsl.GraphAttention, graph_attention_to_ir)
    register(dsl.GraphPooling, graph_pooling_to_ir)
    register(dsl.CustomBlock, custom_block_to_ir)
    register(dsl.SelectiveSSM, selective_ssm_to_ir)
    register(dsl.SamplingLayer, sampling_layer_to_ir)
    register(dsl.NoiseHead, noise_head_to_ir)
    register(dsl.EncoderBlock, encoder_block_to_ir)
    register(dsl.DecoderBlock, decoder_block_to_ir)
    register(dsl.PositionalEncoding, positional_encoding_to_ir)
    register(dsl.SectionHeader, section_header_to_ir)
    register(dsl.Separator, separator_to_ir)
    register(dsl.LSTMBlock, lstm_block_to_ir)
    register(dsl.GRUBlock, gru_block_to_ir)
    register(dsl.MambaBlock, mamba_block_to_ir)
    register(dsl.Generator, generator_compact_to_ir)
    register(dsl.Discriminator, discriminator_compact_to_ir)
    register(dsl.UNetBlock, unet_block_compact_to_ir)
    register(dsl.SideBySide, side_by_side_to_ir)
    register(dsl.BidirectionalFlow, bidirectional_flow_to_ir)
    register(dsl.ForkLoss, fork_loss_to_ir)
    register(dsl.DetailPanel, detail_panel_to_ir)
    register(dsl.EncoderDecoder, encoder_decoder_to_ir)
    register(dsl.SPINNBlock, spinn_block_to_ir)
    register(dsl.UNetLevel, unet_level_to_ir)
    register(dsl.Bottleneck, bottleneck_to_ir)


_register_defaults()


# ---------------------------------------------------------------------------
# Coverage diagnostics
# ---------------------------------------------------------------------------

# Layer types that are still rendered by the legacy monolithic code paths.
# These are either:
# - structural (SectionHeader, Separator, SideBySide, BidirectionalFlow,
#   ForkLoss, DetailPanel) — Phase 6 makes them group attributes;
# - multi-block special layouts (LSTMBlock/GRUBlock with gates, MambaBlock,
#   EncoderDecoder, SPINNBlock, UNetLevel/Bottleneck, UNetBlock,
#   Generator/Discriminator, ResidualBlock alt forms) — they need
#   IRCustomOp escape or per-block lowering with internal sub-graphs.
#
# Tracking this list explicitly so a smoke test can verify forward progress.
LEGACY_ONLY = frozenset()  # Task 5-6: all layer types now have IR lowering.


def can_lower_architecture(arch: "dsl.Architecture") -> tuple[bool, list[str]]:
    """Check whether all layers in ``arch`` have an IR lowering registered.

    Returns ``(all_supported, [missing_type_names])``. Used to decide
    whether to dispatch to the new IR path or keep the legacy renderer.
    """
    missing = sorted({type(layer).__name__ for layer in arch.layers
                      if type(layer) not in _LOWERING})
    return not missing, missing


def architecture_to_ir(arch: "dsl.Architecture", show_n: int = 4) -> IRGraph:
    """Build a complete :class:`IRGraph` from an :class:`Architecture`.

    Honors ``show_n`` for repeated patterns: layers belonging to a
    repetition group beyond the first ``show_n`` repeats are collapsed,
    matching the legacy ×N-badge behavior.
    """
    from . import dsl as _dsl

    groups = _dsl._detect_groups(arch.layers)
    group_index: dict[int, "_dsl._Group"] = {}
    for grp in groups:
        for idx in range(grp.start, grp.end):
            group_index[idx] = grp

    builder = IRBuilder(title=arch.name)
    # Track all node ids belonging to each group, so a fit-frame can be
    # drawn around them at emission time (Phase 6 unification).
    group_node_ids: dict[int, list[str]] = {}
    last_visible_in_group: dict[int, str] = {}

    for i, layer in enumerate(arch.layers):
        grp = group_index.get(i)
        if grp is not None:
            offset = i - grp.start
            rep_index = offset // grp.pattern_len
            visible = rep_index < show_n
            gid = id(grp)
            group_node_ids.setdefault(gid, [])
        else:
            visible = True
            gid = None

        if not visible:
            continue

        before = len(builder.graph.order)
        builder.add_op(layer_to_ir(layer))
        if gid is not None:
            after = len(builder.graph.order)
            # Track all nodes added by this layer that are real blocks
            # (skip anchors and circles — they're internal scaffolding
            # that shouldn't define the visual extent of the group frame).
            new_ids = [
                builder.graph.order[k] for k in range(before, after)
                if builder.graph.nodes[builder.graph.order[k]].shape == "block"
            ]
            group_node_ids[gid].extend(new_ids)
            last_visible_in_group[gid] = builder.cursor

    # Pin the output node BEFORE emitting badge anchors — otherwise the
    # auto-detection in build() would pick the badge as output and the
    # IO-label suppression for "Output"-labelled heads breaks.
    if builder.cursor is not None:
        builder.set_output(builder.cursor)

    # Promote pattern groups into IRGroup entries — emitter draws fit
    # frames + ×N badge for each. Replaces the legacy per-block badge.
    for grp in groups:
        gid = id(grp)
        nodes = group_node_ids.get(gid, [])
        if len(nodes) < 2:
            continue
        repeat = grp.count if grp.count > show_n else None
        ir_grp = builder.new_group(repeat_count=repeat)
        ir_grp.children = nodes
    return builder.build()


def coverage() -> dict[str, int]:
    """Return {migrated, legacy_only, total} layer-type counts."""
    all_types = {
        name for name in dir(dsl)
        if not name.startswith("_")
        and isinstance(getattr(dsl, name), type)
        and getattr(dsl, name).__module__ == dsl.__name__
    }
    # Filter to dataclass layer types only.
    from dataclasses import is_dataclass
    layer_types = {n for n in all_types if is_dataclass(getattr(dsl, n))}
    migrated = {t.__name__ for t in _LOWERING}
    return {
        "migrated": len(migrated & layer_types),
        "legacy_only": len(LEGACY_ONLY & layer_types),
        "total": len(layer_types),
    }
