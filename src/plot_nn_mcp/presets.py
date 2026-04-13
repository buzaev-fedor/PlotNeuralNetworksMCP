"""Preset neural network architecture definitions.

CNN presets (simple_cnn, vgg16, unet, resnet) are zero-parameter.
New presets (transformer, bert, gpt, vit, pinn, fno) accept keyword
arguments for configurable architecture dimensions.
"""

from __future__ import annotations

from typing import Callable

from .pycore.tikzeng import (
    _render_ball, to_begin, to_branch, to_connection, to_Conv,
    to_ConvConvRelu, to_ConvSoftMax, to_cor, to_Dense, to_Embed,
    to_end, to_head, to_Lifting, to_merge, to_MultiHeadAttn, to_Norm,
    to_Pool, to_repeat_bracket, to_skip, to_SoftMax, to_Sum,
)
from .pycore.blocks import block_2ConvPool, block_Res, block_Unconv
from .pycore.blocks_transformer import (
    block_EmbeddingStack, block_FourierLayer, block_MLPStack,
    block_TransformerDecoderLayer, block_TransformerEncoderLayer,
)


# ---------------------------------------------------------------------------
# CNN presets (original, zero-parameter)
# ---------------------------------------------------------------------------

def simple_cnn() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]
    arch.append(to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)",
                        height=64, depth=64, width=2, caption="Conv1"))
    arch.append(to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
                        height=48, depth=48, width=1, caption=" "))
    arch.append(to_Conv("conv2", 256, 128, offset="(1,0,0)", to="(pool1-east)",
                        height=48, depth=48, width=3, caption="Conv2"))
    arch.append(to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)",
                        height=32, depth=32, width=1, caption=" "))
    arch.append(to_Conv("conv3", 128, 256, offset="(1,0,0)", to="(pool2-east)",
                        height=32, depth=32, width=4, caption="Conv3"))
    arch.append(to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)",
                        height=16, depth=16, width=1, caption=" "))
    arch.append(to_SoftMax("soft1", 10, offset="(2,0,0)", to="(pool3-east)",
                           caption="Softmax"))
    arch.append(to_connection("pool1", "conv2"))
    arch.append(to_connection("pool2", "conv3"))
    arch.append(to_connection("pool3", "soft1"))
    arch.append(to_end())
    return arch


def vgg16() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]
    blocks = [
        ("b1", 224, 64, "(0,0,0)", "(0,0,0)", 2, 64, "Block1", None),
        ("b2", 112, 128, "(1,0,0)", "(pool1-east)", 3, 48, "Block2", "pool1"),
        ("b3", 56, 256, "(1,0,0)", "(pool2-east)", 4, 36, "Block3", "pool2"),
        ("b4", 28, 512, "(1,0,0)", "(pool3-east)", 5, 24, "Block4", "pool3"),
        ("b5", 14, 512, "(1,0,0)", "(pool4-east)", 5, 16, "Block5", "pool4"),
    ]
    for i, (name, sf, nf, off, to, w, size, cap, conn_from) in enumerate(blocks, 1):
        arch.append(to_ConvConvRelu(name, sf, (nf, nf), offset=off, to=to,
                                    width=(w, w), height=size, depth=size, caption=cap))
        pool_h = size - int(size / 4) if i < 5 else size // 2
        arch.append(to_Pool(f"pool{i}", offset="(0,0,0)", to=f"({name}-east)",
                            height=pool_h, depth=pool_h))
        if conn_from:
            arch.append(to_connection(conn_from, name))

    arch.append(to_SoftMax("soft1", 1000, offset="(2,0,0)", to="(pool5-east)",
                           width=1.5, height=3, depth=25, caption="Softmax"))
    arch.append(to_connection("pool5", "soft1"))
    arch.append(to_end())
    return arch


def unet() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]

    encoder_cfg = [
        ("b1", "start", "pool_b1", 512, 64, "(0,0,0)", (64, 64, 3.5)),
        ("b2", "pool_b1", "pool_b2", 256, 128, "(1,0,0)", (48, 48, 4.5)),
        ("b3", "pool_b2", "pool_b3", 128, 256, "(1,0,0)", (32, 32, 6)),
        ("b4", "pool_b3", "pool_b4", 64, 512, "(1,0,0)", (16, 16, 8)),
    ]
    for name, bot, top, sf, nf, off, sz in encoder_cfg:
        arch.extend(block_2ConvPool(name, bot, top, s_filer=sf, n_filer=nf,
                                    offset=off, size=sz))

    arch.append(to_ConvConvRelu("bneck", 32, (1024, 1024), offset="(2,0,0)",
                                to="(pool_b4-east)", width=(10, 10), height=8,
                                depth=8, caption="Bottleneck"))
    arch.append(to_connection("pool_b4", "bneck"))

    decoder_cfg = [
        ("b5", "bneck", "end_b5", 64, 512, (16, 16, 8)),
        ("b6", "end_b5", "end_b6", 128, 256, (32, 32, 6)),
        ("b7", "end_b6", "end_b7", 256, 128, (48, 48, 4.5)),
        ("b8", "end_b7", "end_b8", 512, 64, (64, 64, 3.5)),
    ]
    for name, bot, top, sf, nf, sz in decoder_cfg:
        arch.extend(block_Unconv(name, bot, top, s_filer=sf, n_filer=nf,
                                 offset="(2,0,0)", size=sz))

    arch.append(to_ConvSoftMax("soft1", 512, offset="(1,0,0)", to="(end_b8-east)",
                               width=1, height=64, depth=64, caption="Output"))
    arch.append(to_connection("end_b8", "soft1"))

    skip_pairs = [("ccr_b4", "unpool_b5"), ("ccr_b3", "unpool_b6"),
                  ("ccr_b2", "unpool_b7"), ("ccr_b1", "unpool_b8")]
    for src, dst in skip_pairs:
        arch.append(to_skip(src, dst, pos=1.25))

    arch.append(to_end())
    return arch


def resnet() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]

    arch.append(to_Conv("conv1", 224, 64, offset="(0,0,0)", to="(0,0,0)",
                        height=64, depth=64, width=3, caption="Conv1"))
    arch.append(to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
                        height=48, depth=48))

    res_blocks = [
        (3, "res_a", "pool1", "res_a_top", 56, 64, (48, 48, 3.5)),
        (4, "res_b", "res_a_top", "res_b_top", 28, 128, (32, 32, 4.5)),
        (3, "res_c", "res_b_top", "res_c_top", 14, 256, (16, 16, 6)),
    ]
    for num, name, bot, top, sf, nf, sz in res_blocks:
        arch.extend(block_Res(num, name, bot, top, s_filer=sf, n_filer=nf,
                              offset="(1,0,0)", size=sz))

    arch.append(to_SoftMax("soft1", 1000, offset="(2,0,0)", to="(res_c_top-east)",
                           width=1.5, height=3, depth=25, caption="Softmax"))
    arch.append(to_connection("res_c_top", "soft1"))
    arch.append(to_end())
    return arch


# ---------------------------------------------------------------------------
# Transformer-family presets (parameterizable)
# ---------------------------------------------------------------------------

def transformer(
    n_enc: int = 6,
    n_dec: int = 6,
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
) -> list[str]:
    """Full encoder-decoder Transformer (Vaswani et al.)."""
    arch = [to_head("."), to_cor(), to_begin()]
    size = (32, 32, 5)

    # Input embedding
    arch.append(to_Embed("input_start", d_model, offset="(0,0,0)", to="(0,0,0)",
                         width=1, height=32, depth=10, caption="Input"))
    arch.extend(block_EmbeddingStack("enc_emb", "input_start", "enc_emb_out",
                                     d_model=d_model, size=(32, 32, 2)))

    # Encoder layers
    prev = "enc_emb_out"
    first_enc = last_enc = None
    for i in range(n_enc):
        top = f"enc_{i}_out"
        arch.extend(block_TransformerEncoderLayer(
            f"enc_{i}", prev, top, d_model=d_model, num_heads=n_heads,
            d_ff=d_ff, offset="(1.5,0,0)", size=size,
        ))
        if first_enc is None:
            first_enc = f"enc_{i}_attn"
        last_enc = top
        prev = top

    if n_enc > 1:
        arch.append(to_repeat_bracket(first_enc, last_enc, rf"\times {n_enc}"))

    # Decoder embedding
    arch.append(to_Embed("dec_input_start", d_model, offset="(3,0,0)", to=f"({last_enc}-east)",
                         width=1, height=32, depth=10, caption="Output (shifted)"))
    arch.extend(block_EmbeddingStack("dec_emb", "dec_input_start", "dec_emb_out",
                                     d_model=d_model, size=(32, 32, 2)))
    arch.append(to_connection(last_enc, "dec_input_start"))

    # Decoder layers
    prev = "dec_emb_out"
    first_dec = last_dec = None
    for i in range(n_dec):
        top = f"dec_{i}_out"
        arch.extend(block_TransformerDecoderLayer(
            f"dec_{i}", prev, top, encoder_out=last_enc,
            d_model=d_model, num_heads=n_heads, d_ff=d_ff,
            offset="(1.5,0,0)", size=size,
        ))
        if first_dec is None:
            first_dec = f"dec_{i}_mattn"
        last_dec = top
        prev = top

    if n_dec > 1:
        arch.append(to_repeat_bracket(first_dec, last_dec, rf"\times {n_dec}"))

    # Output
    arch.append(to_SoftMax("output", d_model, offset="(2,0,0)", to=f"({last_dec}-east)",
                           caption="Softmax"))
    arch.append(to_connection(last_dec, "output"))
    arch.append(to_end())
    return arch


def bert(
    n_layers: int = 12,
    d_model: int = 768,
    n_heads: int = 12,
    d_ff: int = 3072,
) -> list[str]:
    """BERT encoder-only Transformer."""
    arch = [to_head("."), to_cor(), to_begin()]
    size = (32, 32, 5)

    # Embedding with segment
    arch.append(to_Embed("input_start", d_model, offset="(0,0,0)", to="(0,0,0)",
                         width=1, height=32, depth=10, caption="Input"))
    arch.extend(block_EmbeddingStack("emb", "input_start", "emb_out",
                                     d_model=d_model, size=(32, 32, 2),
                                     include_segment=True))

    # Encoder layers
    prev = "emb_out"
    first_enc = last_enc = None
    for i in range(n_layers):
        top = f"enc_{i}_out"
        arch.extend(block_TransformerEncoderLayer(
            f"enc_{i}", prev, top, d_model=d_model, num_heads=n_heads,
            d_ff=d_ff, offset="(1.5,0,0)", size=size,
        ))
        if first_enc is None:
            first_enc = f"enc_{i}_attn"
        last_enc = top
        prev = top

    if n_layers > 1:
        arch.append(to_repeat_bracket(first_enc, last_enc, rf"\times {n_layers}"))

    # Classification head
    arch.append(to_Dense("cls_dense", d_model, offset="(2,0,0)", to=f"({last_enc}-east)",
                         width=2, height=3, depth=25, caption="CLS Head"))
    arch.append(to_connection(last_enc, "cls_dense"))
    arch.append(to_SoftMax("output", d_model, offset="(1,0,0)", to="(cls_dense-east)",
                           caption="Softmax"))
    arch.append(to_connection("cls_dense", "output"))
    arch.append(to_end())
    return arch


def gpt(
    n_layers: int = 12,
    d_model: int = 768,
    n_heads: int = 12,
    d_ff: int = 3072,
) -> list[str]:
    """GPT decoder-only Transformer (masked self-attention)."""
    arch = [to_head("."), to_cor(), to_begin()]
    size = (32, 32, 5)

    # Embedding
    arch.append(to_Embed("input_start", d_model, offset="(0,0,0)", to="(0,0,0)",
                         width=1, height=32, depth=10, caption="Input"))
    arch.extend(block_EmbeddingStack("emb", "input_start", "emb_out",
                                     d_model=d_model, size=(32, 32, 2)))

    # Decoder layers (masked self-attention only, no cross-attention)
    # We use encoder layers with "Masked MHA" caption to represent this
    prev = "emb_out"
    first_layer = last_layer = None
    for i in range(n_layers):
        top = f"dec_{i}_out"
        attn = f"dec_{i}_attn"
        add1 = f"dec_{i}_add1"
        norm1 = f"dec_{i}_norm1"
        ff1 = f"dec_{i}_ff1"
        ff2 = f"dec_{i}_ff2"
        add2 = f"dec_{i}_add2"

        h, d, w = size
        arch.extend([
            to_MultiHeadAttn(
                name=attn, num_heads=n_heads, d_model=d_model,
                offset="(1.5,0,0)", to=f"({prev}-east)",
                width=w, height=h, depth=d, caption="Masked MHA",
            ),
            to_connection(prev, attn),
            to_Sum(name=add1, offset="(0.8,0,0)", to=f"({attn}-east)", radius=1.5, opacity=0.6),
            to_connection(attn, add1),
            to_skip(prev, add1, pos=1.25),
            to_Norm(name=norm1, offset="(0.5,0,0)", to=f"({add1}-east)",
                    width=0.3, height=h, depth=d, caption="LN"),
            to_connection(add1, norm1),
            to_Dense(name=ff1, n_units=d_ff, offset="(0.8,0,0)", to=f"({norm1}-east)",
                     width=w / 2, height=h * 0.7, depth=d, caption="FFN"),
            to_connection(norm1, ff1),
            to_Dense(name=ff2, n_units=d_model, offset="(0,0,0)", to=f"({ff1}-east)",
                     width=w / 2, height=h * 0.5, depth=d, caption=" "),
            to_connection(ff1, ff2),
            to_Sum(name=add2, offset="(0.8,0,0)", to=f"({ff2}-east)", radius=1.5, opacity=0.6),
            to_connection(ff2, add2),
            to_skip(norm1, add2, pos=1.25),
            to_Norm(name=top, offset="(0.5,0,0)", to=f"({add2}-east)",
                    width=0.3, height=h, depth=d, caption="LN"),
            to_connection(add2, top),
        ])
        if first_layer is None:
            first_layer = attn
        last_layer = top
        prev = top

    if n_layers > 1:
        arch.append(to_repeat_bracket(first_layer, last_layer, rf"\times {n_layers}"))

    # Output head
    arch.append(to_Dense("lm_head", d_model, offset="(2,0,0)", to=f"({last_layer}-east)",
                         width=2, height=3, depth=25, caption="LM Head"))
    arch.append(to_connection(last_layer, "lm_head"))
    arch.append(to_SoftMax("output", d_model, offset="(1,0,0)", to="(lm_head-east)",
                           caption="Softmax"))
    arch.append(to_connection("lm_head", "output"))
    arch.append(to_end())
    return arch


def vit(
    n_layers: int = 12,
    d_model: int = 768,
    n_heads: int = 12,
    d_ff: int = 3072,
    patch_size: int = 16,
) -> list[str]:
    """Vision Transformer (ViT)."""
    arch = [to_head("."), to_cor(), to_begin()]
    size = (32, 32, 5)

    # Patch embedding (shown as Conv)
    arch.append(to_Conv("patch_embed", patch_size, d_model, offset="(0,0,0)", to="(0,0,0)",
                        width=3, height=48, depth=48, caption=f"Patch {patch_size}x{patch_size}"))

    # CLS token + position embedding
    arch.append(to_Embed("cls_token", d_model, offset="(1,0,0)", to="(patch_embed-east)",
                         width=1, height=32, depth=10, caption="CLS Token"))
    arch.append(to_connection("patch_embed", "cls_token"))
    arch.append(to_Embed("pos_embed", d_model, offset="(0,0,0)", to="(cls_token-east)",
                         width=1, height=32, depth=10, caption="Pos Emb"))
    arch.append(to_connection("cls_token", "pos_embed"))
    arch.append(to_Sum("pos_add", offset="(0.8,0,0)", to="(pos_embed-east)",
                       radius=1.5, opacity=0.6))
    arch.append(to_connection("pos_embed", "pos_add"))

    # Encoder layers
    prev = "pos_add"
    first_enc = last_enc = None
    for i in range(n_layers):
        top = f"enc_{i}_out"
        arch.extend(block_TransformerEncoderLayer(
            f"enc_{i}", prev, top, d_model=d_model, num_heads=n_heads,
            d_ff=d_ff, offset="(1.5,0,0)", size=size,
        ))
        if first_enc is None:
            first_enc = f"enc_{i}_attn"
        last_enc = top
        prev = top

    if n_layers > 1:
        arch.append(to_repeat_bracket(first_enc, last_enc, rf"\times {n_layers}"))

    # MLP classification head
    arch.append(to_Dense("mlp_head", d_model, offset="(2,0,0)", to=f"({last_enc}-east)",
                         width=2, height=3, depth=25, caption="MLP Head"))
    arch.append(to_connection(last_enc, "mlp_head"))
    arch.append(to_end())
    return arch


# ---------------------------------------------------------------------------
# Physics-informed / operator presets (parameterizable)
# ---------------------------------------------------------------------------

def pinn(
    n_hidden: int = 4,
    hidden_size: int = 64,
) -> list[str]:
    """Physics-Informed Neural Network (PINN)."""
    arch = [to_head("."), to_cor(), to_begin()]

    # Input
    arch.append(to_Dense("input", hidden_size, offset="(0,0,0)", to="(0,0,0)",
                         width=2, height=6, depth=20, caption="Input (x,t)"))

    # Hidden MLP stack
    arch.extend(block_MLPStack("hidden", "input", "hidden_out",
                               n_layers=n_hidden, hidden_size=hidden_size,
                               offset="(1,0,0)", size=(6, 20, 2)))

    # Branch into data loss and physics loss
    arch.append(to_Dense("data_branch", hidden_size, offset="(2,2,0)",
                         to="(hidden_out-east)",
                         width=2, height=6, depth=20, caption="u(x,t)"))
    arch.append(to_Dense("physics_branch", hidden_size, offset="(2,-2,0)",
                         to="(hidden_out-east)",
                         width=2, height=6, depth=20, caption="PDE residual"))

    arch.append(to_branch("hidden_out", ["data_branch", "physics_branch"], spread=2.0))

    # Loss balls
    arch.append(_render_ball("loss_data", r"\SumColor", "(1.5,0,0)", "(data_branch-east)",
                             radius=1.5, opacity=0.6, logo=r"$\mathcal{L}_d$"))
    arch.append(to_connection("data_branch", "loss_data"))

    arch.append(_render_ball("loss_physics", r"\PhysicsColor", "(1.5,0,0)",
                             "(physics_branch-east)",
                             radius=1.5, opacity=0.6, logo=r"$\mathcal{L}_p$"))
    arch.append(to_connection("physics_branch", "loss_physics"))

    # Total loss
    arch.append(to_Sum("total_loss", offset="(3,0,0)", to="(hidden_out-east)",
                       radius=2.0, opacity=0.6))
    arch.append(to_merge(["loss_data", "loss_physics"], "total_loss", spread=2.0))

    arch.append(to_end())
    return arch


def fno(
    n_layers: int = 4,
    modes: int = 16,
    width: int = 64,
) -> list[str]:
    """Fourier Neural Operator (FNO)."""
    arch = [to_head("."), to_cor(), to_begin()]
    size = (32, 32, 4)

    # Lifting layer
    arch.append(to_Lifting("lifting", offset="(0,0,0)", to="(0,0,0)",
                           width=2, height=40, depth=40, caption="Lifting P"))

    # Fourier layers
    prev = "lifting"
    first_fourier = last_fourier = None
    for i in range(n_layers):
        top = f"fourier_{i}_out"
        arch.extend(block_FourierLayer(
            f"fourier_{i}", prev, top, modes=modes,
            offset="(2,0,0)", size=size,
        ))
        if first_fourier is None:
            first_fourier = f"fourier_{i}_spectral"
        last_fourier = top
        prev = top

    if n_layers > 1:
        arch.append(to_repeat_bracket(first_fourier, last_fourier, rf"\times {n_layers}"))

    # Projection
    arch.append(to_Lifting("projection", offset="(2,0,0)", to=f"({last_fourier}-east)",
                           width=2, height=40, depth=40, caption="Projection Q"))
    arch.append(to_connection(last_fourier, "projection"))

    # Output
    arch.append(to_Dense("output", width, offset="(1,0,0)", to="(projection-east)",
                         width=2, height=3, depth=25, caption="Output"))
    arch.append(to_connection("projection", "output"))

    arch.append(to_end())
    return arch


# ---------------------------------------------------------------------------
# DSL-based presets (new layouts: horizontal, unet, side-by-side, etc.)
# ---------------------------------------------------------------------------

from .dsl import (
    Architecture, Embedding, PositionalEncoding, TransformerBlock, ConvBlock,
    PatchEmbedding, DenseLayer, ClassificationHead, FourierBlock, CustomBlock,
    SideBySide, BidirectionalFlow, ForkLoss, UNetLevel, Bottleneck,
    DetailPanel, SectionHeader, Separator, EncoderBlock, DecoderBlock,
    Generator, Discriminator, UNetBlock, NoiseHead, AdaptiveLayerNorm,
    LSTMBlock, GRUBlock, ResidualBlock, MambaBlock,
    EncoderDecoder, SPINNBlock,
)


def transformer_vaswani(
    n_enc: int = 6, n_dec: int = 6,
    d_model: int = 512, n_heads: int = 8, d_ff: int = 2048,
) -> str:
    """Encoder-Decoder Transformer (Vaswani et al. 2017) — with cross-attention."""
    arch = Architecture("Transformer (Vaswani et al. 2017)", theme="arxiv")
    arch.add(EncoderDecoder(
        encoder_input=[Embedding(d_model=d_model, label="Input Embedding"),
                       PositionalEncoding(encoding_type="sinusoidal", d_model=d_model)],
        decoder_input=[Embedding(d_model=d_model, label="Output Embedding"),
                       PositionalEncoding(encoding_type="sinusoidal", d_model=d_model)],
        encoder=[TransformerBlock(attention="self", norm="post_ln", d_model=d_model,
                                  heads=n_heads, d_ff=d_ff) for _ in range(n_enc)],
        decoder=[TransformerBlock(attention="cross", norm="post_ln", d_model=d_model,
                                  heads=n_heads, d_ff=d_ff,
                                  label="Masked-SA + Cross-SA + FFN") for _ in range(n_dec)],
        cross_attention="all",
        encoder_label="Encoder",
        decoder_label="Decoder",
    ))
    arch.add(ClassificationHead(label="Linear + Softmax"))
    return arch.render(show_n=2)


def vit_preset(
    n_layers: int = 12, d_model: int = 768,
    n_heads: int = 12, d_ff: int = 3072, patch_size: int = 16,
    layout: str = "horizontal",
) -> str:
    """Vision Transformer (Dosovitskiy et al. 2021) — horizontal or vertical."""
    arch = Architecture("ViT-B/16 (Dosovitskiy et al. 2021)", theme="arxiv", layout=layout)
    arch.add(PatchEmbedding(patch_size=patch_size, d_model=d_model))
    arch.add(PositionalEncoding(encoding_type="learned", d_model=d_model))
    for _ in range(n_layers):
        arch.add(TransformerBlock(attention="global", norm="pre_ln", ffn="gelu",
                                  heads=n_heads, d_model=d_model, d_ff=d_ff))
    arch.add(ClassificationHead(label="MLP Head"))
    return arch.render(show_n=3)


def vit_horizontal(**kw) -> str:
    """ViT horizontal (backward compat alias)."""
    return vit_preset(layout="horizontal", **kw)


def vit_vertical(**kw) -> str:
    """ViT vertical layout."""
    return vit_preset(layout="vertical", **kw)


def unet_canonical(
    levels: int = 4, base_filters: int = 64,
) -> str:
    """U-Net (Ronneberger et al. 2015) — canonical U-shaped layout."""
    arch = Architecture("U-Net (Ronneberger et al. 2015)", theme="arxiv", layout="unet")
    for i in range(levels):
        filters = base_filters * (2 ** i)
        res = 1.0 / (2 ** i)
        arch.add(UNetLevel(filters=filters, resolution=res))
    arch.add(Bottleneck(filters=base_filters * (2 ** levels)))
    return arch.render()


def ddpm_chain(n_steps: int = 5, layout: str = "horizontal") -> str:
    """DDPM (Ho et al. 2020) — bidirectional diffusion chain."""
    if layout == "vertical":
        # Vertical: stack diffusion steps top-down
        arch = Architecture("DDPM (Ho et al. 2020)", theme="modern")
        arch.add(CustomBlock(text=r"$x_0$ (clean)", color_role="embed"))
        for i in range(1, n_steps - 1):
            arch.add(CustomBlock(text=f"$x_{{{i}}}$", color_role="embed"))
        arch.add(CustomBlock(text=r"$x_T$ (noise)", color_role="physics"))
        return arch.render()
    # Horizontal (default)
    arch = Architecture("DDPM (Ho et al. 2020)", theme="modern")
    steps = [r"$x_0$"] + [f"$x_{{{i}}}$" for i in range(1, n_steps - 1)] + [r"$x_T$"]
    arch.add(BidirectionalFlow(
        steps=steps,
        forward_label=r"$q(x_t|x_{t-1})$",
        reverse_label=r"$p_\theta(x_{t-1}|x_t)$",
    ))
    return arch.render()


def pinn_branched(
    n_hidden: int = 4, hidden_size: int = 256,
) -> str:
    """PINN (Raissi et al. 2019) — MLP with forked data/physics loss."""
    arch = Architecture("PINN (Raissi et al. 2019)", theme="modern")
    arch.add(CustomBlock(text=r"Input $(x, t)$", color_role="embed"))
    for _ in range(n_hidden):
        arch.add(DenseLayer(units=hidden_size, label=f"Hidden ({hidden_size})"))
    arch.add(CustomBlock(text=r"$u(x,t)$", color_role="output"))
    arch.add(ForkLoss(
        branches=[
            ("Data Loss", "embed", r"$\mathcal{L}_{data}$"),
            ("PDE Residual", "physics", r"$\mathcal{L}_{PDE}$"),
        ],
        merge_label=r"$\mathcal{L}_{total}$",
    ))
    return arch.render(show_n=2)


def fno_spectral(
    n_layers: int = 4, modes: int = 16, width: int = 64,
) -> str:
    """FNO (Li et al. 2021) — Fourier layers with dual spectral/spatial paths."""
    arch = Architecture("FNO (Li et al. 2021)", theme="modern")
    arch.add(CustomBlock(text="Lifting Layer $P$", color_role="embed"))
    for _ in range(n_layers):
        arch.add(FourierBlock(modes=modes, width=width))
    arch.add(CustomBlock(text="Projection $Q$", color_role="output"))
    return arch.render(show_n=1)


def detr_pipeline(
    n_enc: int = 6, n_dec: int = 6,
    d_model: int = 256, n_heads: int = 8,
    layout: str = "horizontal",
) -> str:
    """DETR (Carion et al. 2020) — horizontal or vertical pipeline."""
    arch = Architecture("DETR (Carion et al. 2020)", theme="arxiv", layout=layout)
    arch.add(ConvBlock(filters=256, kernel_size=3, pool=None, label="CNN Backbone"))
    arch.add(CustomBlock(text="Flatten + PE", color_role="embed"))
    for _ in range(n_enc):
        arch.add(TransformerBlock(attention="self", norm="pre_ln", d_model=d_model,
                                  heads=n_heads))
    arch.add(Separator(label="Object Queries"))
    for _ in range(n_dec):
        arch.add(TransformerBlock(attention="cross", norm="pre_ln", d_model=d_model,
                                  heads=n_heads))
    arch.add(CustomBlock(text="FFN Heads", color_role="output"))
    return arch.render(show_n=2)


def stylegan_architecture(
    n_mapping: int = 8, n_synthesis: int = 9,
    d_latent: int = 512,
) -> str:
    """StyleGAN (Karras et al. 2019) — mapping + synthesis with AdaIN injection."""
    arch = Architecture("StyleGAN (Karras et al. 2019)", theme="vibrant")
    arch.add(EncoderDecoder(
        encoder_input=[CustomBlock(text=r"$z \sim \mathcal{N}(0,1)$", color_role="embed")],
        decoder_input=[CustomBlock(text="Const $4{\\times}4$", color_role="embed")],
        encoder=[DenseLayer(units=d_latent, label=f"FC {d_latent}") for _ in range(n_mapping)],
        decoder=[
            CustomBlock(
                text=(
                    f"Synth {4 * 2**i}$\\times${4 * 2**i} + AdaIN"
                    if i in (0, 1, n_synthesis - 1)
                    else f"Synth {4 * 2**i} + AdaIN"
                ),
                color_role="attention",
            )
            # Reorder so the first 3 visible blocks span the resolution range
            for i in (
                [0, 1, n_synthesis - 1] + [j for j in range(2, n_synthesis - 1)]
            )
        ],
        cross_attention="all",
        cross_attention_label="AdaIN ($w$)",
        encoder_label="Mapping Network ($z \\to w$)",
        decoder_label="Synthesis Network",
        encoder_input_label="$z$",
        decoder_input_label="",
    ))
    return arch.render(show_n=3)


def lstm_seq2seq(
    n_enc: int = 3, n_dec: int = 3,
    hidden_size: int = 512,
    style: str = "compact",
) -> str:
    """LSTM Seq2Seq (Sutskever et al. 2014) — encoder-decoder with LSTM.

    style: "compact" (single blocks) or "gates" (full gate detail) or "olah" (Chris Olah style).
    """
    arch = Architecture("LSTM Seq2Seq (Sutskever et al. 2014)", theme="paper")
    arch.add(EncoderDecoder(
        encoder_input=[Embedding(d_model=hidden_size, label="Input Embedding")],
        decoder_input=[Embedding(d_model=hidden_size, label="Output Embedding")],
        encoder=[LSTMBlock(hidden_size=hidden_size, style=style) for _ in range(n_enc)],
        decoder=[LSTMBlock(hidden_size=hidden_size, style=style) for _ in range(n_dec)],
        cross_attention="last",
        cross_attention_label="Final Hidden State $h_T$",
        encoder_label="Encoder",
        decoder_label="Decoder",
    ))
    arch.add(ClassificationHead(label="Output Projection"))
    return arch.render(show_n=2)


def resnet_geometric(
    blocks_per_stage: list[int] | None = None,
    filters_per_stage: list[int] | None = None,
) -> str:
    """ResNet (He et al. 2015) — with geometric dimension encoding."""
    blocks_per_stage = blocks_per_stage or [3, 4, 6, 3]
    filters_per_stage = filters_per_stage or [64, 128, 256, 512]
    arch = Architecture("ResNet-50 (He et al. 2015)", theme="arxiv")
    arch.add(ConvBlock(filters=64, kernel_size=7, pool="max", label="Conv 7x7 + MaxPool"))
    for stage_idx, (n_blocks, filters) in enumerate(zip(blocks_per_stage, filters_per_stage)):
        if stage_idx > 0:
            arch.add(Separator(label=f"Stage {stage_idx + 1}"))
        for _ in range(n_blocks):
            arch.add(ResidualBlock(filters=filters))
    arch.add(CustomBlock(text="Global AvgPool", color_role="ffn"))
    arch.add(ClassificationHead(label="FC 1000"))
    return arch.render(show_n=2)


def latent_diffusion(
    n_unet_blocks: int = 4, base_filters: int = 128,
) -> str:
    """Latent Diffusion (Rombach et al. 2022) — VAE + U-Net + text conditioning."""
    arch = Architecture("Latent Diffusion (Rombach et al. 2022)", theme="modern")
    arch.add(SectionHeader(title="Pixel Space"))
    arch.add(ConvBlock(filters=64, label="VAE Encoder", pool=None))
    arch.add(SectionHeader(title="Latent Space"))
    arch.add(CustomBlock(text=r"Text Encoder (CLIP) $\rightarrow$ Cross-Attn", color_role="spectral"))
    for i in range(n_unet_blocks):
        arch.add(UNetBlock(filters=base_filters * (2 ** min(i, 3)),
                           with_attention=(i >= 2)))
    arch.add(AdaptiveLayerNorm(label="AdaLN", condition="timestep $t$"))
    arch.add(NoiseHead(label=r"$\epsilon_\theta(z_t, t, c)$ Prediction"))
    arch.add(SectionHeader(title="Pixel Space"))
    arch.add(ConvBlock(filters=3, label="VAE Decoder", pool=None))
    return arch.render(show_n=2)


# ---------------------------------------------------------------------------
# New Tier-1 presets (Part F)
# ---------------------------------------------------------------------------

def t5_architecture(
    n_enc: int = 12, n_dec: int = 12,
    d_model: int = 768, n_heads: int = 12, d_ff: int = 3072,
) -> str:
    """T5-Base (Raffel et al. 2020) — encoder-decoder with relative position bias."""
    arch = Architecture("T5-Base (Raffel et al. 2020)", theme="arxiv")
    arch.add(EncoderDecoder(
        encoder_input=[Embedding(d_model=d_model, label="Shared Token Emb"),
                       CustomBlock(text="Relative Pos Bias", color_role="norm")],
        decoder_input=[Embedding(d_model=d_model, label="Shared Token Emb"),
                       CustomBlock(text="Relative Pos Bias", color_role="norm")],
        encoder=[TransformerBlock(attention="self", norm="pre_ln", ffn="gelu",
                                  d_model=d_model, heads=n_heads, d_ff=d_ff)
                 for _ in range(n_enc)],
        decoder=[TransformerBlock(attention="cross", norm="pre_ln", ffn="gelu",
                                  d_model=d_model, heads=n_heads, d_ff=d_ff,
                                  label="Masked-SA + Cross-SA + FFN")
                 for _ in range(n_dec)],
        cross_attention="all",
        encoder_label="Encoder (Pre-LN)",
        decoder_label="Decoder (Pre-LN + Cross-Attn)",
    ))
    arch.add(ClassificationHead(label="LM Head (tied weights)"))
    return arch.render(show_n=2)


def seq2seq_bahdanau(
    n_enc: int = 3, n_dec: int = 3,
    hidden_size: int = 512,
) -> str:
    """Seq2Seq + Bahdanau Attention (2015) — LSTM encoder-decoder with alignment."""
    arch = Architecture("Seq2Seq + Attention (Bahdanau et al. 2015)", theme="paper")
    arch.add(EncoderDecoder(
        encoder_input=[Embedding(d_model=hidden_size, label="Source Embedding")],
        decoder_input=[Embedding(d_model=hidden_size, label="Target Embedding")],
        encoder=[LSTMBlock(hidden_size=hidden_size, style="compact", bidirectional=True)
                 for _ in range(n_enc)],
        decoder=[LSTMBlock(hidden_size=hidden_size, style="compact") for _ in range(n_dec)],
        cross_attention="all",
        encoder_label="Bi-LSTM Encoder",
        decoder_label="LSTM Decoder + Attention",
    ))
    arch.add(ClassificationHead(label="Output Projection"))
    return arch.render(show_n=2)


def gpt_decoder_only(
    n_layers: int = 12, d_model: int = 768,
    n_heads: int = 12, d_ff: int = 3072,
    layout: str = "vertical",
) -> str:
    """GPT-2 (Radford et al. 2019) — decoder-only transformer."""
    arch = Architecture("GPT-2 (Radford et al. 2019)", theme="arxiv", layout=layout)
    arch.add(Embedding(d_model=d_model, label="Token + Position Embedding"))
    for _ in range(n_layers):
        arch.add(TransformerBlock(attention="masked", norm="pre_ln", ffn="gelu",
                                  heads=n_heads, d_model=d_model, d_ff=d_ff))
    arch.add(ClassificationHead(label="LM Head (Softmax)"))
    return arch.render(show_n=2)


def spinn_architecture(
    hidden_size: int = 300, tracking_size: int = 128,
) -> str:
    """SPINN (Bowman et al. 2016) — stack-augmented parser-interpreter."""
    arch = Architecture("SPINN (Bowman et al. 2016)", theme="modern")
    arch.add(Embedding(d_model=hidden_size, label="Word Embeddings"))
    arch.add(SPINNBlock(
        hidden_size=hidden_size,
        tracking_size=tracking_size,
        buffer_tokens=["The", "cat", "sat", "on"],
    ))
    arch.add(ClassificationHead(label="Entailment / Contradiction"))
    return arch.render()


def lstm_olah_cell(hidden_size: int = 256) -> str:
    """LSTM Cell — Chris Olah blog style (conveyor belt visualization)."""
    arch = Architecture("LSTM Cell (Olah style)", theme="modern")
    arch.add(LSTMBlock(hidden_size=hidden_size, style="olah"))
    return arch.render()


def deeponet_architecture(
    branch_layers: int = 4, trunk_layers: int = 4,
    hidden_size: int = 256,
) -> str:
    """DeepONet (Lu et al. 2021) — branch + trunk operator learning."""
    arch = Architecture("DeepONet (Lu et al. 2021)", theme="modern")
    arch.add(SideBySide(
        left=[DenseLayer(units=hidden_size, label=f"Branch FC ({hidden_size})")
              for _ in range(branch_layers)],
        right=[DenseLayer(units=hidden_size, label=f"Trunk FC ({hidden_size})")
               for _ in range(trunk_layers)],
        left_label="Branch Net (sensors)",
        right_label="Trunk Net (coordinates)",
        connections=[],  # no cross-connections — merge via dot product
    ))
    arch.add(CustomBlock(text=r"$\langle \cdot, \cdot \rangle$ (dot product)", color_role="residual"))
    arch.add(ClassificationHead(label="Output $G(u)(y)$"))
    return arch.render(show_n=2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PRESETS: dict[str, Callable[..., list[str] | str]] = {
    # --- Old 3D isometric presets ---
    "simple_cnn": simple_cnn,
    "vgg16": vgg16,
    "unet": unet,
    "resnet": resnet,
    "transformer": transformer,
    "bert": bert,
    "gpt": gpt,
    "vit": vit,
    "pinn": pinn,
    "fno": fno,
    # --- New DSL presets (Core A* style) ---
    "transformer_vaswani": transformer_vaswani,
    "vit_horizontal": vit_horizontal,
    "unet_canonical": unet_canonical,
    "ddpm_chain": ddpm_chain,
    "pinn_branched": pinn_branched,
    "fno_spectral": fno_spectral,
    "detr_pipeline": detr_pipeline,
    "stylegan": stylegan_architecture,
    "lstm_seq2seq": lstm_seq2seq,
    "resnet_geometric": resnet_geometric,
    "latent_diffusion": latent_diffusion,
    # --- Vertical variants ---
    "vit_vertical": vit_vertical,
    # --- New Tier-1 presets ---
    "t5": t5_architecture,
    "seq2seq_bahdanau": seq2seq_bahdanau,
    "gpt_decoder": gpt_decoder_only,
    "spinn": spinn_architecture,
    "lstm_olah": lstm_olah_cell,
    "deeponet": deeponet_architecture,
}
