"""
TikZ neural network layer generation functions.
Adapted from https://github.com/HarisIqbal88/PlotNeuralNet

All layer functions produce TikZ code strings. Four base renderers
(_render_box, _render_banded_box, _render_ball, _render_partitioned_box)
eliminate template duplication — each public to_* function is a thin
wrapper that fills in the right color/defaults.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------

def to_head(projectpath: str) -> str:
    pathlayers = os.path.join(projectpath, "layers/").replace("\\", "/")
    return (
        r"\documentclass[border=8pt, multi, tikz]{standalone}" "\n"
        r"\usepackage{import}" "\n"
        rf"\subimport{{{pathlayers}}}{{init}}" "\n"
        r"\usetikzlibrary{positioning}" "\n"
        r"\usetikzlibrary{3d}" "\n"
    )


def to_cor() -> str:
    colors = {
        # CNN colors (original)
        "ConvColor": "rgb:yellow,5;red,2.5;white,5",
        "ConvReluColor": "rgb:yellow,5;red,5;white,5",
        "PoolColor": "rgb:red,1;black,0.3",
        "UnpoolColor": "rgb:blue,2;green,1;black,0.3",
        "FcColor": "rgb:blue,5;red,2.5;white,5",
        "FcReluColor": "rgb:blue,5;red,5;white,4",
        "SoftmaxColor": "rgb:magenta,5;black,7",
        "SumColor": "rgb:blue,5;green,15",
        # Transformer / new architecture colors
        "NormColor": "rgb:yellow,1;white,5;black,0.2",
        "AttnColor": "rgb:blue,3;cyan,3;white,5",
        "EmbedColor": "rgb:green,3;yellow,2;white,5",
        "SpectralColor": "rgb:cyan,4;blue,3;white,5",
        "LiftColor": "rgb:green,2;blue,3;white,5",
        "PhysicsColor": "rgb:red,3;orange,2;white,5",
    }
    return "".join(rf"\def\{name}{{{value}}}" "\n" for name, value in colors.items())


def to_begin() -> str:
    return (
        r"\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,"
        r"draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}" "\n"
        "\n"
        r"\begin{document}" "\n"
        r"\begin{tikzpicture}" "\n"
        r"\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},"
        r"draw=\edgecolor,opacity=0.7]" "\n"
        r"\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},"
        r"draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]" "\n"
    )


def to_end() -> str:
    return (
        r"\end{tikzpicture}" "\n"
        r"\end{document}" "\n"
    )


# ---------------------------------------------------------------------------
# Base renderers  (DRY: all layers delegate here)
# ---------------------------------------------------------------------------

def _render_box(
    name: str,
    fill: str,
    offset: str,
    to: str,
    height: float,
    width: float,
    depth: float,
    caption: str = " ",
    opacity: float | None = None,
    xlabel: str | None = None,
    zlabel: str | None = None,
) -> str:
    lines = [
        rf"\pic[shift={{ {offset} }}] at {to}",
        r"    {Box={",
        f"        name={name},",
        f"        caption={caption},",
    ]
    if xlabel is not None:
        lines.append(f"        xlabel={xlabel},")
    if zlabel is not None:
        lines.append(f"        zlabel={zlabel},")
    lines.append(f"        fill={fill},")
    if opacity is not None:
        lines.append(f"        opacity={opacity},")
    lines.extend([
        f"        height={height},",
        f"        width={width},",
        f"        depth={depth}",
        r"        }",
        r"    };",
    ])
    return "\n".join(lines) + "\n"


def _render_banded_box(
    name: str,
    fill: str,
    bandfill: str,
    offset: str,
    to: str,
    height: float,
    width: str,
    depth: float,
    caption: str = " ",
    opacity: float | None = None,
    xlabel: str | None = None,
    zlabel: str | None = None,
) -> str:
    lines = [
        rf"\pic[shift={{ {offset} }}] at {to}",
        r"    {RightBandedBox={",
        f"        name={name},",
        f"        caption={caption},",
    ]
    if xlabel is not None:
        lines.append(f"        xlabel={xlabel},")
    if zlabel is not None:
        lines.append(f"        zlabel={zlabel},")
    lines.append(f"        fill={fill},")
    lines.append(f"        bandfill={bandfill},")
    if opacity is not None:
        lines.append(f"        opacity={opacity},")
    lines.extend([
        f"        height={height},",
        f"        width={width},",
        f"        depth={depth}",
        r"        }",
        r"    };",
    ])
    return "\n".join(lines) + "\n"


def _render_ball(
    name: str,
    fill: str,
    offset: str,
    to: str,
    radius: float,
    opacity: float,
    logo: str = "$+$",
) -> str:
    lines = [
        rf"\pic[shift={{{offset}}}] at {to}",
        r"    {Ball={",
        f"        name={name},",
        f"        fill={fill},",
        f"        opacity={opacity},",
        f"        radius={radius},",
        f"        logo={logo}",
        r"        }",
        r"    };",
    ]
    return "\n".join(lines) + "\n"


def _render_partitioned_box(
    name: str,
    fill: str,
    offset: str,
    to: str,
    height: float,
    width: float,
    depth: float,
    num_parts: int = 4,
    caption: str = " ",
    opacity: float = 0.7,
    zlabel: str | None = None,
) -> str:
    lines = [
        rf"\pic[shift={{ {offset} }}] at {to}",
        r"    {PartitionedBox={",
        f"        name={name},",
        f"        caption={caption},",
    ]
    if zlabel is not None:
        lines.append(f"        zlabel={zlabel},")
    lines.extend([
        f"        fill={fill},",
        f"        opacity={opacity},",
        f"        nparts={num_parts},",
        f"        height={height},",
        f"        width={width},",
        f"        depth={depth}",
        r"        }",
        r"    };",
    ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public layer functions — CNN (original)
# ---------------------------------------------------------------------------

def to_input(pathfile: str, to: str = "(-3,0,0)", width: float = 8,
             height: float = 8, name: str = "temp") -> str:
    return (
        rf"\node[canvas is zy plane at x=0] ({name}) at {to}"
        rf" {{\includegraphics[width={width}cm,height={height}cm]{{{pathfile}}}}};"
        "\n"
    )


def to_Conv(name: str, s_filer: int = 256, n_filer: int = 64,
            offset: str = "(0,0,0)", to: str = "(0,0,0)",
            width: float = 1, height: float = 40, depth: float = 40,
            caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\ConvColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        xlabel=f"{{{{{n_filer}, }}}}", zlabel=str(s_filer),
    )


def to_ConvConvRelu(name: str, s_filer: int = 256, n_filer: tuple[int, int] = (64, 64),
                    offset: str = "(0,0,0)", to: str = "(0,0,0)",
                    width: tuple[float, float] = (2, 2), height: float = 40,
                    depth: float = 40, caption: str = " ") -> str:
    return _render_banded_box(
        name=name, fill=r"\ConvColor", bandfill=r"\ConvReluColor",
        offset=offset, to=to, height=height, depth=depth, caption=caption,
        width=f"{{ {width[0]} , {width[1]} }}",
        xlabel=f"{{{{ {n_filer[0]}, {n_filer[1]} }}}}", zlabel=str(s_filer),
    )


def to_Pool(name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
            width: float = 1, height: float = 32, depth: float = 32,
            opacity: float = 0.5, caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\PoolColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        opacity=opacity,
    )


def to_UnPool(name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
              width: float = 1, height: float = 32, depth: float = 32,
              opacity: float = 0.5, caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\UnpoolColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        opacity=opacity,
    )


def to_ConvRes(name: str, s_filer: int = 256, n_filer: int = 64,
               offset: str = "(0,0,0)", to: str = "(0,0,0)",
               width: float = 6, height: float = 40, depth: float = 40,
               opacity: float = 0.2, caption: str = " ") -> str:
    return _render_banded_box(
        name=name, fill="{rgb:white,1;black,3}", bandfill="{rgb:white,1;black,2}",
        offset=offset, to=to, height=height, width=str(width), depth=depth,
        caption=caption, opacity=opacity,
        xlabel=f"{{{{ {n_filer}, }}}}", zlabel=str(s_filer),
    )


def to_ConvSoftMax(name: str, s_filer: int = 40, offset: str = "(0,0,0)",
                   to: str = "(0,0,0)", width: float = 1, height: float = 40,
                   depth: float = 40, caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\SoftmaxColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        zlabel=str(s_filer),
    )


def to_SoftMax(name: str, s_filer: int = 10, offset: str = "(0,0,0)",
               to: str = "(0,0,0)", width: float = 1.5, height: float = 3,
               depth: float = 25, opacity: float = 0.8, caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\SoftmaxColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        opacity=opacity, xlabel='{" ","dummy"}', zlabel=str(s_filer),
    )


def to_Sum(name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
           radius: float = 2.5, opacity: float = 0.6) -> str:
    return _render_ball(
        name=name, fill=r"\SumColor", offset=offset, to=to,
        radius=radius, opacity=opacity,
    )


# ---------------------------------------------------------------------------
# Public layer functions — new architectures (Transformer, PINN, FNO)
# ---------------------------------------------------------------------------

def to_Dense(name: str, n_units: int = 256, offset: str = "(0,0,0)",
             to: str = "(0,0,0)", width: float = 2, height: float = 3,
             depth: float = 25, caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\FcColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        xlabel=f"{{{{{n_units}, }}}}",
    )


def to_Norm(name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
            width: float = 0.3, height: float = 40, depth: float = 40,
            caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\NormColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        opacity=0.7,
    )


def to_Embed(name: str, d_model: int = 512, offset: str = "(0,0,0)",
             to: str = "(0,0,0)", width: float = 2, height: float = 40,
             depth: float = 10, caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\EmbedColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        xlabel=f"{{{{{d_model}, }}}}",
    )


def to_MultiHeadAttn(name: str, num_heads: int = 8, d_model: int = 512,
                     offset: str = "(0,0,0)", to: str = "(0,0,0)",
                     width: float = 6, height: float = 40, depth: float = 40,
                     opacity: float = 0.7, caption: str = " ") -> str:
    return _render_partitioned_box(
        name=name, fill=r"\AttnColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        num_parts=num_heads, opacity=opacity, zlabel=str(d_model),
    )


def to_Multiply(name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
                radius: float = 2.5, opacity: float = 0.6) -> str:
    return _render_ball(
        name=name, fill=r"\SumColor", offset=offset, to=to,
        radius=radius, opacity=opacity, logo=r"$\times$",
    )


def to_Concat(name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
              radius: float = 2.5, opacity: float = 0.6) -> str:
    return _render_ball(
        name=name, fill=r"\SumColor", offset=offset, to=to,
        radius=radius, opacity=opacity, logo=r"$\|$",
    )


def to_SpectralConv(name: str, modes: int = 16, offset: str = "(0,0,0)",
                    to: str = "(0,0,0)", width: float = 3, height: float = 40,
                    depth: float = 40, caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\SpectralColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
        xlabel=f"{{{{{modes}, }}}}", zlabel="FFT",
    )


def to_Lifting(name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
               width: float = 2, height: float = 40, depth: float = 40,
               caption: str = " ") -> str:
    return _render_box(
        name=name, fill=r"\LiftColor", offset=offset, to=to,
        height=height, width=width, depth=depth, caption=caption,
    )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def to_connection(of: str, to: str) -> str:
    return rf"\draw [connection]  ({of}-east)    -- node {{\midarrow}} ({to}-west);" "\n"


def to_skip(of: str, to: str, pos: float = 1.25) -> str:
    return (
        rf"\path ({of}-southeast) -- ({of}-northeast) coordinate[pos={pos}] ({of}-top) ;" "\n"
        rf"\path ({to}-south)  -- ({to}-north)  coordinate[pos={pos}] ({to}-top) ;" "\n"
        rf"\draw [copyconnection]  ({of}-northeast)" "\n"
        rf"-- node {{\copymidarrow}}({of}-top)" "\n"
        rf"-- node {{\copymidarrow}}({to}-top)" "\n"
        rf"-- node {{\copymidarrow}} ({to}-north);" "\n"
    )


def to_skip_bottom(of: str, to: str, pos: float = 1.25) -> str:
    return (
        rf"\path ({of}-southwest) -- ({of}-southeast) coordinate[pos=0.5] ({of}-bottom) ;" "\n"
        rf"\path ({of}-bottom) ++ (0,-{pos},0) coordinate ({of}-below) ;" "\n"
        rf"\path ({to}-southwest) -- ({to}-southeast) coordinate[pos=0.5] ({to}-bottom) ;" "\n"
        rf"\path ({to}-bottom) ++ (0,-{pos},0) coordinate ({to}-below) ;" "\n"
        rf"\draw [copyconnection]  ({of}-south)" "\n"
        rf"-- node {{\copymidarrow}}({of}-below)" "\n"
        rf"-- node {{\copymidarrow}}({to}-below)" "\n"
        rf"-- node {{\copymidarrow}} ({to}-south);" "\n"
    )


def to_branch(of: str, targets: list[str], spread: float = 1.0) -> str:
    lines = []
    n = len(targets)
    for i, target in enumerate(targets):
        y_off = spread * (i - (n - 1) / 2)
        lines.append(
            rf"\draw [connection]  ({of}-east) -- ++(1,{y_off:.2f},0) "
            rf"-- node {{\midarrow}} ({target}-west);"
            "\n"
        )
    return "".join(lines)


def to_merge(sources: list[str], to: str, spread: float = 1.0) -> str:
    lines = []
    n = len(sources)
    for i, source in enumerate(sources):
        y_off = spread * (i - (n - 1) / 2)
        lines.append(
            rf"\draw [connection]  ({source}-east) -- ++(1,{-y_off:.2f},0) "
            rf"-- node {{\midarrow}} ({to}-west);"
            "\n"
        )
    return "".join(lines)


def to_repeat_bracket(of_start: str, of_end: str, label: str = r"\times N",
                      xshift: float = 3.5) -> str:
    return (
        rf"\draw[decorate, decoration={{brace, amplitude=10pt, mirror}}, thick]"
        rf" ([xshift={xshift}cm]{of_start}-nearnortheast) -- "
        rf"([xshift={xshift}cm]{of_end}-nearsoutheast)"
        rf" node[midway, right=12pt] {{\large\bf ${label}$}};"
        "\n"
    )


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

def to_generate(arch: list[str], pathname: str = "file.tex") -> None:
    with open(pathname, "w") as f:
        for c in arch:
            f.write(c)
