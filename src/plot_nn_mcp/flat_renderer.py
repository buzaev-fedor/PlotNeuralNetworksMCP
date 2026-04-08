"""
2D flat TikZ renderer for vertical architecture diagrams.

Instead of 3D isometric boxes (Box.sty), this renderer uses TikZ nodes
with rounded corners, centered text, and vertical (south→north) flow.
Produces clean, publication-quality diagrams in the style of the original
Transformer paper (Vaswani et al.).
"""

from __future__ import annotations

from .themes import Theme, get_theme, theme_to_tikz_colors


# ---------------------------------------------------------------------------
# Document structure (flat/vertical variant)
# ---------------------------------------------------------------------------

def flat_head() -> str:
    return (
        r"\documentclass[border=15pt, tikz]{standalone}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage{tikz}" "\n"
        r"\usetikzlibrary{positioning, arrows.meta, decorations.pathreplacing, calc, fit}" "\n"
    )


def flat_colors(theme: Theme) -> str:
    return theme_to_tikz_colors(theme)


def flat_begin() -> str:
    return (
        r"\begin{document}" "\n"
        r"\begin{tikzpicture}[" "\n"
        r"    >=Stealth," "\n"
        r"    every node/.style={font=\sffamily}," "\n"
        r"    block/.style={draw=clrborder, rounded corners=4pt, minimum width=3.8cm," "\n"
        r"                  minimum height=0.85cm, font=\sffamily\small\bfseries," "\n"
        r"                  text=clrtext, line width=0.6pt}," "\n"
        r"    smallblock/.style={block, minimum width=2.4cm, minimum height=0.65cm," "\n"
        r"                      font=\sffamily\scriptsize\bfseries}," "\n"
        r"    arrow/.style={->, thick, color=clrborder, line width=0.8pt}," "\n"
        r"    skiparrow/.style={->, thick, color=clrresidual, line width=0.8pt, rounded corners=3pt}," "\n"
        r"    label/.style={font=\sffamily\scriptsize, text=clrtext}," "\n"
        r"    subtitle/.style={font=\sffamily\footnotesize, text=clrtext}," "\n"
        r"]" "\n"
    )


def flat_end() -> str:
    return (
        r"\end{tikzpicture}" "\n"
        r"\end{document}" "\n"
    )


# ---------------------------------------------------------------------------
# Flat block primitives
# ---------------------------------------------------------------------------

def flat_block(
    name: str,
    text: str,
    fill: str,
    position: str = "(0,0)",
    width: float = 3.8,
    height: float = 0.85,
    opacity: float = 0.85,
    style: str = "block",
    anchor: str | None = None,
    below_of: str | None = None,
    above_of: str | None = None,
    node_distance: float = 0.4,
    text_color: str = "clrtext",
) -> str:
    """Render a flat 2D block (rounded rectangle with centered text)."""
    opts = [
        style,
        f"fill=clr{fill}!{int(opacity * 100)}" if "!" not in fill else f"fill={fill}",
        f"minimum width={width}cm",
        f"minimum height={height}cm",
        f"text={text_color}",
    ]
    if below_of:
        opts.append(f"below={node_distance}cm of {below_of}")
    elif above_of:
        opts.append(f"above={node_distance}cm of {above_of}")
    if anchor:
        opts.append(f"anchor={anchor}")

    opts_str = ", ".join(opts)
    pos = "" if (below_of or above_of) else f" at {position}"
    return rf"\node[{opts_str}] ({name}){pos} {{{text}}};" "\n"


def flat_arrow(from_name: str, to_name: str,
               from_anchor: str = "north", to_anchor: str = "south") -> str:
    """Vertical arrow between two blocks."""
    return rf"\draw[arrow] ({from_name}.{from_anchor}) -- ({to_name}.{to_anchor});" "\n"


def flat_skip_arrow(from_name: str, to_name: str, xshift: float = 0.3,
                    direction: str = "right") -> str:
    """Residual/skip connection arrow that curves to the right of blocks, inside the group frame."""
    sign = "" if direction == "right" else "-"
    return (
        rf"\draw[skiparrow] ({from_name}.east) -- ++({sign}{xshift},0) "
        rf"|- ({to_name}.east);" "\n"
    )


def flat_add_circle(name: str, below_of: str | None = None,
                    above_of: str | None = None,
                    node_distance: float = 0.3) -> str:
    """Small '+' circle for residual addition."""
    pos_opt = ""
    if below_of:
        pos_opt = f"below={node_distance}cm of {below_of}"
    elif above_of:
        pos_opt = f"above={node_distance}cm of {above_of}"
    return (
        rf"\node[circle, draw=clrborder, fill=clrresidual!30, "
        rf"inner sep=2pt, font=\sffamily\scriptsize\bfseries, {pos_opt}] "
        rf"({name}) {{+}};" "\n"
    )


# ---------------------------------------------------------------------------
# GroupFrame
# ---------------------------------------------------------------------------

def group_frame(
    name: str,
    fit_nodes: list[str],
    title: str = "",
    repeat: int | None = None,
    padding: float = 0.3,
) -> str:
    """Draw a rounded rectangle around a group of nodes with optional title and ×N."""
    fit_list = " ".join(f"({n})" for n in fit_nodes)
    lines = [
        rf"\node[draw=clrgroup_frame, rounded corners=6pt, dashed, line width=0.8pt,",
        rf"      fill=clrgroup_fill, fill opacity=0.4,",
        rf"      inner sep={padding}cm, fit={fit_list}] ({name}) {{}};",
    ]
    if title:
        lines.append(
            rf"\node[above=1pt of {name}.north west, anchor=south west, "
            rf"font=\sffamily\scriptsize\bfseries, text=clrgroup_frame] "
            rf"{{{title}}};"
        )
    if repeat and repeat > 1:
        lines.append(
            rf"\node[right=6pt of {name}.east, anchor=west, "
            rf"font=\sffamily\Large\bfseries, text=clrborder] "
            rf"{{$\times{repeat}$}};"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def flat_separator(above_of: str, name: str, width: float = 3.8) -> str:
    """A thin dashed line between encoder blocks for visual separation."""
    return (
        rf"\node[above=0.12cm of {above_of}, inner sep=0, minimum height=0] ({name}) {{}};" "\n"
        rf"\draw[densely dashed, color=clrgroup_frame!70, line width=0.6pt] "
        rf"([xshift=-{width/2}cm]{name}.center) -- ([xshift={width/2}cm]{name}.center);" "\n"
    )


def flat_io_arrow(node: str, direction: str = "below", label: str = "") -> str:
    """Draw an input or output arrow coming into or out of the diagram."""
    if direction == "below":
        label_node = rf" node[right, font=\sffamily\scriptsize, text=clrtext] {{{label}}}" if label else ""
        return rf"\draw[arrow] ({node}.south) ++ (0,-0.5) --{label_node} ({node}.south);" "\n"
    else:
        label_node = rf" node[right, font=\sffamily\scriptsize, text=clrtext] {{{label}}}" if label else ""
        return rf"\draw[arrow] ({node}.north) --{label_node} ++ (0,0.5);" "\n"


def flat_title(text: str, below_of: str = "current bounding box.south",
               distance: float = 0.6) -> str:
    """Add a title/subtitle below the diagram."""
    return (
        rf"\node[below={distance}cm of {below_of}, subtitle, "
        rf"text width=14cm, align=center] {{{text}}};" "\n"
    )


def flat_side_label(text: str, node: str, side: str = "right",
                    distance: float = 0.3) -> str:
    """Add a small label to the side of a block."""
    return (
        rf"\node[{side}={distance}cm of {node}, label, "
        rf"anchor={'west' if side == 'right' else 'east'}] {{{text}}};" "\n"
    )


def flat_dim_label(text: str, node: str, side: str = "left",
                   distance: float = 0.15) -> str:
    """Add a dimension annotation (e.g. '768') to a block."""
    anchor = "east" if side == "left" else "west"
    return (
        rf"\node[{side}={distance}cm of {node}, "
        rf"font=\sffamily\tiny, text=clrborder, anchor={anchor}] {{{text}}};" "\n"
    )
