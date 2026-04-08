"""
TikZ neural network layer generation functions.
Adapted from https://github.com/HarisIqbal88/PlotNeuralNet
"""

import os


def to_head(projectpath):
    pathlayers = os.path.join(projectpath, "layers/").replace("\\", "/")
    return (
        r"\documentclass[border=8pt, multi, tikz]{standalone}" "\n"
        r"\usepackage{import}" "\n"
        r"\subimport{" + pathlayers + r"}{init}" "\n"
        r"\usetikzlibrary{positioning}" "\n"
        r"\usetikzlibrary{3d}" "\n"
    )


def to_cor():
    return (
        r"\def\ConvColor{rgb:yellow,5;red,2.5;white,5}" "\n"
        r"\def\ConvReluColor{rgb:yellow,5;red,5;white,5}" "\n"
        r"\def\PoolColor{rgb:red,1;black,0.3}" "\n"
        r"\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}" "\n"
        r"\def\FcColor{rgb:blue,5;red,2.5;white,5}" "\n"
        r"\def\FcReluColor{rgb:blue,5;red,5;white,4}" "\n"
        r"\def\SoftmaxColor{rgb:magenta,5;black,7}" "\n"
        r"\def\SumColor{rgb:blue,5;green,15}" "\n"
    )


def to_begin():
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


_INPUT_TPL = (
    r"\node[canvas is zy plane at x=0] ({name}) at {to}"
    r" {{\includegraphics[width={width}cm,height={height}cm]{{{pathfile}}}}};"
    "\n"
)


def to_input(pathfile, to="(-3,0,0)", width=8, height=8, name="temp"):
    return _INPUT_TPL.format(name=name, to=to, width=width, height=height, pathfile=pathfile)


_CONV_TPL = (
    r"\pic[shift={{{offset}}}] at {to}" "\n"
    r"    {{Box={{" "\n"
    r"        name={name}," "\n"
    r"        caption={caption}," "\n"
    r"        xlabel={{{{{n_filer}, }}}}," "\n"
    r"        zlabel={s_filer}," "\n"
    r"        fill=\ConvColor," "\n"
    r"        height={height}," "\n"
    r"        width={width}," "\n"
    r"        depth={depth}" "\n"
    r"        }}" "\n"
    r"    }};" "\n"
)


def to_Conv(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)",
            width=1, height=40, depth=40, caption=" "):
    return _CONV_TPL.format(
        name=name, s_filer=s_filer, n_filer=n_filer,
        offset=offset, to=to, width=width, height=height, depth=depth,
        caption=caption,
    )


_CONVCONVRELU_TPL = (
    r"\pic[shift={{ {offset} }}] at {to}" "\n"
    r"    {{RightBandedBox={{" "\n"
    r"        name={name}," "\n"
    r"        caption={caption}," "\n"
    r"        xlabel={{{{ {n_filer_0}, {n_filer_1} }}}}," "\n"
    r"        zlabel={s_filer}," "\n"
    r"        fill=\ConvColor," "\n"
    r"        bandfill=\ConvReluColor," "\n"
    r"        height={height}," "\n"
    r"        width={{ {width_0} , {width_1} }}," "\n"
    r"        depth={depth}" "\n"
    r"        }}" "\n"
    r"    }};" "\n"
)


def to_ConvConvRelu(name, s_filer=256, n_filer=(64, 64), offset="(0,0,0)",
                    to="(0,0,0)", width=(2, 2), height=40, depth=40, caption=" "):
    return _CONVCONVRELU_TPL.format(
        name=name, s_filer=s_filer, n_filer_0=n_filer[0], n_filer_1=n_filer[1],
        offset=offset, to=to, width_0=width[0], width_1=width[1],
        height=height, depth=depth, caption=caption,
    )


_POOL_TPL = (
    r"\pic[shift={{ {offset} }}] at {to}" "\n"
    r"    {{Box={{" "\n"
    r"        name={name}," "\n"
    r"        caption={caption}," "\n"
    r"        fill=\PoolColor," "\n"
    r"        opacity={opacity}," "\n"
    r"        height={height}," "\n"
    r"        width={width}," "\n"
    r"        depth={depth}" "\n"
    r"        }}" "\n"
    r"    }};" "\n"
)


def to_Pool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32,
            depth=32, opacity=0.5, caption=" "):
    return _POOL_TPL.format(
        name=name, offset=offset, to=to, width=width,
        height=height, depth=depth, opacity=opacity, caption=caption,
    )


_UNPOOL_TPL = (
    r"\pic[shift={{ {offset} }}] at {to}" "\n"
    r"    {{Box={{" "\n"
    r"        name={name}," "\n"
    r"        caption={caption}," "\n"
    r"        fill=\UnpoolColor," "\n"
    r"        opacity={opacity}," "\n"
    r"        height={height}," "\n"
    r"        width={width}," "\n"
    r"        depth={depth}" "\n"
    r"        }}" "\n"
    r"    }};" "\n"
)


def to_UnPool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32,
              depth=32, opacity=0.5, caption=" "):
    return _UNPOOL_TPL.format(
        name=name, offset=offset, to=to, width=width,
        height=height, depth=depth, opacity=opacity, caption=caption,
    )


_CONVRES_TPL = (
    r"\pic[shift={{ {offset} }}] at {to}" "\n"
    r"    {{RightBandedBox={{" "\n"
    r"        name={name}," "\n"
    r"        caption={caption}," "\n"
    r"        xlabel={{{{ {n_filer}, }}}}," "\n"
    r"        zlabel={s_filer}," "\n"
    r"        fill={{rgb:white,1;black,3}}," "\n"
    r"        bandfill={{rgb:white,1;black,2}}," "\n"
    r"        opacity={opacity}," "\n"
    r"        height={height}," "\n"
    r"        width={width}," "\n"
    r"        depth={depth}" "\n"
    r"        }}" "\n"
    r"    }};" "\n"
)


def to_ConvRes(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)",
               width=6, height=40, depth=40, opacity=0.2, caption=" "):
    return _CONVRES_TPL.format(
        name=name, s_filer=s_filer, n_filer=n_filer,
        offset=offset, to=to, width=width, height=height, depth=depth,
        opacity=opacity, caption=caption,
    )


_CONVSOFTMAX_TPL = (
    r"\pic[shift={{{offset}}}] at {to}" "\n"
    r"    {{Box={{" "\n"
    r"        name={name}," "\n"
    r"        caption={caption}," "\n"
    r"        zlabel={s_filer}," "\n"
    r"        fill=\SoftmaxColor," "\n"
    r"        height={height}," "\n"
    r"        width={width}," "\n"
    r"        depth={depth}" "\n"
    r"        }}" "\n"
    r"    }};" "\n"
)


def to_ConvSoftMax(name, s_filer=40, offset="(0,0,0)", to="(0,0,0)",
                   width=1, height=40, depth=40, caption=" "):
    return _CONVSOFTMAX_TPL.format(
        name=name, s_filer=s_filer, offset=offset, to=to,
        width=width, height=height, depth=depth, caption=caption,
    )


_SOFTMAX_TPL = (
    r"\pic[shift={{{offset}}}] at {to}" "\n"
    r"    {{Box={{" "\n"
    r"        name={name}," "\n"
    r"        caption={caption}," "\n"
    r'        xlabel={{" ","dummy"}},' "\n"
    r"        zlabel={s_filer}," "\n"
    r"        fill=\SoftmaxColor," "\n"
    r"        opacity={opacity}," "\n"
    r"        height={height}," "\n"
    r"        width={width}," "\n"
    r"        depth={depth}" "\n"
    r"        }}" "\n"
    r"    }};" "\n"
)


def to_SoftMax(name, s_filer=10, offset="(0,0,0)", to="(0,0,0)",
               width=1.5, height=3, depth=25, opacity=0.8, caption=" "):
    return _SOFTMAX_TPL.format(
        name=name, s_filer=s_filer, offset=offset, to=to,
        width=width, height=height, depth=depth, opacity=opacity,
        caption=caption,
    )


_SUM_TPL = (
    r"\pic[shift={{{offset}}}] at {to}" "\n"
    r"    {{Ball={{" "\n"
    r"        name={name}," "\n"
    r"        fill=\SumColor," "\n"
    r"        opacity={opacity}," "\n"
    r"        radius={radius}," "\n"
    r"        logo=$+$" "\n"
    r"        }}" "\n"
    r"    }};" "\n"
)


def to_Sum(name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6):
    return _SUM_TPL.format(
        name=name, offset=offset, to=to, radius=radius, opacity=opacity,
    )


def to_connection(of, to):
    return (
        r"\draw [connection]  (" + of + r"-east)    -- node {\midarrow} (" + to + r"-west);"
        "\n"
    )


def to_skip(of, to, pos=1.25):
    return (
        r"\path (" + of + r"-southeast) -- (" + of + r"-northeast) coordinate[pos=" + str(pos) + r"] (" + of + r"-top) ;" "\n"
        r"\path (" + to + r"-south)  -- (" + to + r"-north)  coordinate[pos=" + str(pos) + r"] (" + to + r"-top) ;" "\n"
        r"\draw [copyconnection]  (" + of + r"-northeast)" "\n"
        r"-- node {\copymidarrow}(" + of + r"-top)" "\n"
        r"-- node {\copymidarrow}(" + to + r"-top)" "\n"
        r"-- node {\copymidarrow} (" + to + r"-north);" "\n"
    )


def to_end():
    return (
        r"\end{tikzpicture}" "\n"
        r"\end{document}" "\n"
    )


def to_generate(arch, pathname="file.tex"):
    with open(pathname, "w") as f:
        for c in arch:
            f.write(c)
