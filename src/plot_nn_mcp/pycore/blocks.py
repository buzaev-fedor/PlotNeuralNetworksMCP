"""
Composite block functions for common neural network patterns.
Adapted from https://github.com/HarisIqbal88/PlotNeuralNet
"""

from __future__ import annotations

from .tikzeng import (
    to_connection,
    to_Conv,
    to_ConvConvRelu,
    to_ConvRes,
    to_Pool,
    to_skip,
    to_UnPool,
)


def block_2ConvPool(
    name: str,
    botton: str,
    top: str,
    s_filer: int = 256,
    n_filer: int = 64,
    offset: str = "(1,0,0)",
    size: tuple[float, float, float] = (32, 32, 3.5),
    opacity: float = 0.5,
) -> list[str]:
    h, d, w = size
    ccr_name = f"ccr_{name}"
    return [
        to_ConvConvRelu(
            name=ccr_name, s_filer=str(s_filer), n_filer=(n_filer, n_filer),
            offset=offset, to=f"({botton}-east)",
            width=(w, w), height=h, depth=d,
        ),
        to_Pool(
            name=top, offset="(0,0,0)", to=f"({ccr_name}-east)",
            width=1, height=h - int(h / 4), depth=d - int(h / 4),
            opacity=opacity,
        ),
        to_connection(botton, ccr_name),
    ]


def block_Unconv(
    name: str,
    botton: str,
    top: str,
    s_filer: int = 256,
    n_filer: int = 64,
    offset: str = "(1,0,0)",
    size: tuple[float, float, float] = (32, 32, 3.5),
    opacity: float = 0.5,
) -> list[str]:
    h, d, w = size
    sf, nf = str(s_filer), str(n_filer)
    unpool = f"unpool_{name}"
    ccr_res = f"ccr_res_{name}"
    ccr = f"ccr_{name}"
    ccr_res_c = f"ccr_res_c_{name}"
    return [
        to_UnPool(name=unpool, offset=offset, to=f"({botton}-east)",
                  width=1, height=h, depth=d, opacity=opacity),
        to_ConvRes(name=ccr_res, offset="(0,0,0)", to=f"({unpool}-east)",
                   s_filer=sf, n_filer=nf, width=w, height=h, depth=d, opacity=opacity),
        to_Conv(name=ccr, offset="(0,0,0)", to=f"({ccr_res}-east)",
                s_filer=sf, n_filer=nf, width=w, height=h, depth=d),
        to_ConvRes(name=ccr_res_c, offset="(0,0,0)", to=f"({ccr}-east)",
                   s_filer=sf, n_filer=nf, width=w, height=h, depth=d, opacity=opacity),
        to_Conv(name=top, offset="(0,0,0)", to=f"({ccr_res_c}-east)",
                s_filer=sf, n_filer=nf, width=w, height=h, depth=d),
        to_connection(botton, unpool),
    ]


def block_Res(
    num: int,
    name: str,
    botton: str,
    top: str,
    s_filer: int = 256,
    n_filer: int = 64,
    offset: str = "(0,0,0)",
    size: tuple[float, float, float] = (32, 32, 3.5),
    opacity: float = 0.5,
) -> list[str]:
    h, d, w = size
    sf, nf = str(s_filer), str(n_filer)
    layer_names = [f"{name}_{i}" for i in range(num - 1)] + [top]

    result: list[str] = []
    prev = botton
    for layer_name in layer_names:
        result.append(to_Conv(
            name=layer_name, offset=offset, to=f"({prev}-east)",
            s_filer=sf, n_filer=nf, width=w, height=h, depth=d,
        ))
        result.append(to_connection(prev, layer_name))
        prev = layer_name

    result.append(to_skip(of=layer_names[1], to=layer_names[-2], pos=1.25))
    return result
