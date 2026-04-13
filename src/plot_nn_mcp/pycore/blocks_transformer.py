"""
Composite block functions for Transformer, PINN, and FNO architectures.

Each block returns list[str] of TikZ code — same pattern as blocks.py.
"""

from __future__ import annotations

from .tikzeng import (
    to_connection, to_Conv, to_Dense, to_Embed, to_Lifting,
    to_MultiHeadAttn, to_Norm, to_skip, to_SpectralConv, to_Sum,
)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

def block_TransformerEncoderLayer(
    name: str,
    bottom: str,
    top: str,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    offset: str = "(2,0,0)",
    size: tuple[float, float, float] = (32, 32, 6),
) -> list[str]:
    """One transformer encoder layer: MHA → Add&Norm → FFN → Add&Norm."""
    h, d, w = size
    attn = f"{name}_attn"
    add1 = f"{name}_add1"
    norm1 = f"{name}_norm1"
    ff1 = f"{name}_ff1"
    ff2 = f"{name}_ff2"
    add2 = f"{name}_add2"
    norm2 = f"{name}_norm2"

    return [
        to_MultiHeadAttn(
            name=attn, num_heads=num_heads, d_model=d_model,
            offset=offset, to=f"({bottom}-east)",
            width=w, height=h, depth=d, caption="MHA",
        ),
        to_connection(bottom, attn),
        to_Sum(name=add1, offset="(0.8,0,0)", to=f"({attn}-east)", radius=1.5, opacity=0.6),
        to_connection(attn, add1),
        to_skip(bottom, add1, pos=1.25),
        to_Norm(
            name=norm1, offset="(0.5,0,0)", to=f"({add1}-east)",
            width=0.3, height=h, depth=d, caption="LN",
        ),
        to_connection(add1, norm1),
        to_Dense(
            name=ff1, n_units=d_ff, offset="(0.8,0,0)", to=f"({norm1}-east)",
            width=w / 2, height=h * 0.7, depth=d, caption="FFN",
        ),
        to_connection(norm1, ff1),
        to_Dense(
            name=ff2, n_units=d_model, offset="(0,0,0)", to=f"({ff1}-east)",
            width=w / 2, height=h * 0.5, depth=d, caption=" ",
        ),
        to_connection(ff1, ff2),
        to_Sum(name=add2, offset="(0.8,0,0)", to=f"({ff2}-east)", radius=1.5, opacity=0.6),
        to_connection(ff2, add2),
        to_skip(norm1, add2, pos=1.25),
        to_Norm(
            name=top, offset="(0.5,0,0)", to=f"({add2}-east)",
            width=0.3, height=h, depth=d, caption="LN",
        ),
        to_connection(add2, top),
    ]


def block_TransformerDecoderLayer(
    name: str,
    bottom: str,
    top: str,
    encoder_out: str,
    d_model: int = 512,
    num_heads: int = 8,
    d_ff: int = 2048,
    offset: str = "(2,0,0)",
    size: tuple[float, float, float] = (32, 32, 6),
) -> list[str]:
    """One transformer decoder layer: Masked MHA → Add&Norm → Cross-MHA → Add&Norm → FFN → Add&Norm."""
    h, d, w = size
    mattn = f"{name}_mattn"
    add1 = f"{name}_add1"
    norm1 = f"{name}_norm1"
    xattn = f"{name}_xattn"
    add2 = f"{name}_add2"
    norm2 = f"{name}_norm2"
    ff1 = f"{name}_ff1"
    ff2 = f"{name}_ff2"
    add3 = f"{name}_add3"
    norm3 = f"{name}_norm3"

    return [
        # Masked self-attention
        to_MultiHeadAttn(
            name=mattn, num_heads=num_heads, d_model=d_model,
            offset=offset, to=f"({bottom}-east)",
            width=w, height=h, depth=d, caption="Masked MHA",
        ),
        to_connection(bottom, mattn),
        to_Sum(name=add1, offset="(0.8,0,0)", to=f"({mattn}-east)", radius=1.5, opacity=0.6),
        to_connection(mattn, add1),
        to_skip(bottom, add1, pos=1.25),
        to_Norm(
            name=norm1, offset="(0.5,0,0)", to=f"({add1}-east)",
            width=0.3, height=h, depth=d, caption="LN",
        ),
        to_connection(add1, norm1),
        # Cross-attention (from encoder)
        to_MultiHeadAttn(
            name=xattn, num_heads=num_heads, d_model=d_model,
            offset="(1.5,0,0)", to=f"({norm1}-east)",
            width=w, height=h, depth=d, caption="Cross-MHA",
        ),
        to_connection(norm1, xattn),
        to_Sum(name=add2, offset="(0.8,0,0)", to=f"({xattn}-east)", radius=1.5, opacity=0.6),
        to_connection(xattn, add2),
        to_skip(norm1, add2, pos=1.25),
        to_Norm(
            name=norm2, offset="(0.5,0,0)", to=f"({add2}-east)",
            width=0.3, height=h, depth=d, caption="LN",
        ),
        to_connection(add2, norm2),
        # FFN
        to_Dense(
            name=ff1, n_units=d_ff, offset="(0.8,0,0)", to=f"({norm2}-east)",
            width=w / 2, height=h * 0.7, depth=d, caption="FFN",
        ),
        to_connection(norm2, ff1),
        to_Dense(
            name=ff2, n_units=d_model, offset="(0,0,0)", to=f"({ff1}-east)",
            width=w / 2, height=h * 0.5, depth=d, caption=" ",
        ),
        to_connection(ff1, ff2),
        to_Sum(name=add3, offset="(0.8,0,0)", to=f"({ff2}-east)", radius=1.5, opacity=0.6),
        to_connection(ff2, add3),
        to_skip(norm2, add3, pos=1.25),
        to_Norm(
            name=top, offset="(0.5,0,0)", to=f"({add3}-east)",
            width=0.3, height=h, depth=d, caption="LN",
        ),
        to_connection(add3, top),
    ]


def block_EmbeddingStack(
    name: str,
    bottom: str,
    top: str,
    d_model: int = 512,
    offset: str = "(1,0,0)",
    size: tuple[float, float, float] = (32, 32, 2),
    include_segment: bool = False,
) -> list[str]:
    """Token + position (+ optional segment) embeddings → sum."""
    h, d, w = size
    tok = f"{name}_tok"
    pos = f"{name}_pos"
    add = f"{name}_add"

    result = [
        to_Embed(
            name=tok, d_model=d_model, offset=offset, to=f"({bottom}-east)",
            width=w, height=h, depth=d, caption="Token Emb",
        ),
        to_connection(bottom, tok),
        to_Embed(
            name=pos, d_model=d_model, offset="(0,0,0)", to=f"({tok}-east)",
            width=w, height=h * 0.7, depth=d, caption="Pos Emb",
        ),
        to_connection(tok, pos),
    ]

    if include_segment:
        seg = f"{name}_seg"
        result.append(to_Embed(
            name=seg, d_model=d_model, offset="(0,0,0)", to=f"({pos}-east)",
            width=w, height=h * 0.5, depth=d, caption="Seg Emb",
        ))
        result.append(to_connection(pos, seg))
        last = seg
    else:
        last = pos

    result.extend([
        to_Sum(name=add, offset="(0.8,0,0)", to=f"({last}-east)", radius=1.5, opacity=0.6),
        to_connection(last, add),
        to_Norm(
            name=top, offset="(0.5,0,0)", to=f"({add}-east)",
            width=0.3, height=h, depth=d, caption="LN",
        ),
        to_connection(add, top),
    ])
    return result


# ---------------------------------------------------------------------------
# MLP / PINN blocks
# ---------------------------------------------------------------------------

def block_MLPStack(
    name: str,
    bottom: str,
    top: str,
    n_layers: int = 3,
    hidden_size: int = 64,
    offset: str = "(1,0,0)",
    size: tuple[float, float, float] = (8, 25, 2),
) -> list[str]:
    """Stack of N dense layers with connections."""
    h, d, w = size
    result: list[str] = []
    prev = bottom
    layer_names = [f"{name}_{i}" for i in range(n_layers - 1)] + [top]

    for i, lname in enumerate(layer_names):
        off = offset if i == 0 else "(0.5,0,0)"
        result.append(to_Dense(
            name=lname, n_units=hidden_size, offset=off, to=f"({prev}-east)",
            width=w, height=h, depth=d,
            caption=f"Dense" if i < n_layers - 1 else "Output",
        ))
        result.append(to_connection(prev, lname))
        prev = lname

    return result


# ---------------------------------------------------------------------------
# FNO blocks
# ---------------------------------------------------------------------------

def block_FourierLayer(
    name: str,
    bottom: str,
    top: str,
    modes: int = 16,
    offset: str = "(2,0,0)",
    size: tuple[float, float, float] = (32, 32, 4),
) -> list[str]:
    """One Fourier layer: spectral conv branch + linear skip branch → sum."""
    h, d, w = size
    spectral = f"{name}_spectral"
    linear = f"{name}_linear"
    add = f"{name}_add"

    return [
        to_SpectralConv(
            name=spectral, modes=modes, offset=offset, to=f"({bottom}-east)",
            width=w, height=h, depth=d, caption="Spectral Conv",
        ),
        to_connection(bottom, spectral),
        to_Conv(
            name=linear, s_filer=" ", n_filer=" ",
            offset="(0,-3,0)", to=f"({spectral}-south)",
            width=w * 0.5, height=h * 0.5, depth=d * 0.5, caption="Linear",
        ),
        to_Sum(name=add, offset="(1.5,0,0)", to=f"({spectral}-east)", radius=1.5, opacity=0.6),
        to_connection(spectral, add),
        to_Norm(
            name=top, offset="(0.5,0,0)", to=f"({add}-east)",
            width=0.3, height=h, depth=d, caption=" ",
        ),
        to_connection(add, top),
        to_skip(bottom, add, pos=1.25),
    ]
