"""
2D flat TikZ renderer for architecture diagrams.

Uses TikZ nodes with rounded corners, centered text, and configurable flow:
- Vertical (south→north): default, Transformer-paper style
- Horizontal (west→east): ViT/DETR pipeline style

Produces clean, publication-quality diagrams.
"""

from __future__ import annotations

from .themes import Theme, theme_to_tikz_colors

_LATEX_SPECIAL = str.maketrans({
    "#": r"\#",
    "%": r"\%",
    "&": r"\&",
    "_": r"\_",
})


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in user text.

    If the text already contains LaTeX commands (backslash or $),
    it is returned as-is to preserve intentional formatting.
    """
    if "\\" in text or "$" in text:
        return text
    return text.translate(_LATEX_SPECIAL)


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
# Dimension encoding utilities
# ---------------------------------------------------------------------------

def width_from_dim(dim: int, min_dim: int = 32, max_dim: int = 512,
                   min_width: float = 1.5, max_width: float = 7.0) -> float:
    """Scale block width proportionally to channel/feature dimension.

    Implements U-Net / ResNet best practice of encoding tensor
    dimensionality in block geometry.
    """
    ratio = min(max(dim - min_dim, 0) / max(max_dim - min_dim, 1), 1.0)
    return min_width + ratio * (max_width - min_width)



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
    left_of: str | None = None,
    right_of: str | None = None,
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
    elif left_of:
        opts.append(f"left={node_distance}cm of {left_of}")
    elif right_of:
        opts.append(f"right={node_distance}cm of {right_of}")
    if anchor:
        opts.append(f"anchor={anchor}")

    opts_str = ", ".join(opts)
    has_relative = below_of or above_of or left_of or right_of
    pos = "" if has_relative else f" at {position}"
    return rf"\node[{opts_str}] ({name}){pos} {{{_latex_escape(text)}}};" "\n"


def flat_arrow(from_name: str, to_name: str,
               from_anchor: str = "north", to_anchor: str = "south") -> str:
    """Arrow between two blocks (vertical: north→south, horizontal: east→west)."""
    return rf"\draw[arrow] ({from_name}.{from_anchor}) -- ({to_name}.{to_anchor});" "\n"


def flat_arrow_h(from_name: str, to_name: str) -> str:
    """Horizontal arrow (east→west) between two blocks."""
    return rf"\draw[arrow] ({from_name}.east) -- ({to_name}.west);" "\n"


def flat_skip_arrow(from_name: str, to_name: str, xshift: float = 2.2,
                    direction: str = "right") -> str:
    """Residual/skip connection arrow that curves around blocks.

    direction: "right" routes via east side, "left" routes via west side.
    xshift must be > half the block width to route outside blocks.
    """
    if direction == "left":
        return (
            rf"\draw[skiparrow] ({from_name}.west) -- ++(-{xshift},0) "
            rf"|- ({to_name}.west);" "\n"
        )
    return (
        rf"\draw[skiparrow] ({from_name}.east) -- ++({xshift},0) "
        rf"|- ({to_name}.east);" "\n"
    )



def flat_add_circle(name: str, below_of: str | None = None,
                    above_of: str | None = None,
                    left_of: str | None = None,
                    right_of: str | None = None,
                    node_distance: float = 0.3) -> str:
    """Small '+' circle for residual addition."""
    pos_opt = ""
    if below_of:
        pos_opt = f"below={node_distance}cm of {below_of}"
    elif above_of:
        pos_opt = f"above={node_distance}cm of {above_of}"
    elif left_of:
        pos_opt = f"left={node_distance}cm of {left_of}"
    elif right_of:
        pos_opt = f"right={node_distance}cm of {right_of}"
    return (
        rf"\node[circle, draw=clrborder, fill=clrresidual!30, "
        rf"inner sep=2pt, font=\sffamily\scriptsize\bfseries, {pos_opt}] "
        rf"({name}) {{+}};" "\n"
    )


def flat_op_circle(name: str, symbol: str = "+",
                   below_of: str | None = None,
                   above_of: str | None = None,
                   left_of: str | None = None,
                   right_of: str | None = None,
                   node_distance: float = 0.3,
                   fill: str = "clrresidual!30") -> str:
    """Small circle with an operation symbol (⊕, ⊗, +, etc.)."""
    pos_opt = ""
    if below_of:
        pos_opt = f"below={node_distance}cm of {below_of}"
    elif above_of:
        pos_opt = f"above={node_distance}cm of {above_of}"
    elif left_of:
        pos_opt = f"left={node_distance}cm of {left_of}"
    elif right_of:
        pos_opt = f"right={node_distance}cm of {right_of}"
    return (
        rf"\node[circle, draw=clrborder, fill={fill}, "
        rf"inner sep=2pt, font=\sffamily\scriptsize\bfseries, {pos_opt}] "
        rf"({name}) {{{symbol}}};" "\n"
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
    horizontal: bool = False,
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
        bracket_pos = "right=12pt of {n}.east, anchor=west" if not horizontal else "below=8pt of {n}.south, anchor=north"
        lines.append(
            rf"\node[{bracket_pos.format(n=name)}, "
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


def flat_separator_label(name: str, label: str, above_of: str,
                         width: float = 4.5, style: str = "thick") -> str:
    """A prominent labeled separator line (e.g. 'Generator' / 'Discriminator')."""
    line_style = {
        "line": "color=clrgroup_frame, line width=0.5pt",
        "thick": "color=clrborder, line width=1.2pt",
        "double": "color=clrborder, double, line width=0.6pt",
    }.get(style, "color=clrborder, line width=1.2pt")
    return (
        rf"\node[above=0.5cm of {above_of}, inner sep=0, minimum height=0] ({name}_rule) {{}};" "\n"
        rf"\draw[{line_style}] "
        rf"([xshift=-{width/2}cm]{name}_rule.center) -- ([xshift={width/2}cm]{name}_rule.center);" "\n"
        rf"\node[above=3pt of {name}_rule, font=\sffamily\small\bfseries, text=clrborder] "
        rf"({name}) {{{_latex_escape(label)}}};" "\n"
    )


def flat_section_header(name: str, title: str, above_of: str,
                        subtitle: str = "") -> str:
    """A section header with bold title and optional subtitle — no box, just text + thin rule."""
    lines = (
        rf"\node[above=0.6cm of {above_of}, inner sep=0, minimum height=0] ({name}_rule) {{}};" "\n"
        rf"\draw[color=clrgroup_frame!50, line width=0.4pt] "
        rf"([xshift=-2.2cm]{name}_rule.center) -- ([xshift=2.2cm]{name}_rule.center);" "\n"
        rf"\node[above=4pt of {name}_rule, font=\sffamily\normalsize\bfseries, text=clrborder] "
        rf"({name}) {{{_latex_escape(title)}}};" "\n"
    )
    if subtitle:
        lines += (
            rf"\node[above=2pt of {name}, font=\sffamily\scriptsize, text=clrgroup_frame] "
            rf"{{{subtitle}}};" "\n"
        )
    return lines


def flat_io_arrow(node: str, direction: str = "below", label: str = "") -> str:
    """Draw an input or output arrow coming into or out of the diagram.

    Directions: below/above for vertical layout, left/right for horizontal.
    """
    if direction == "below":
        label_node = rf" node[right, yshift=-2pt, font=\sffamily\scriptsize, text=clrtext] {{{label}}}" if label else ""
        return rf"\draw[arrow] ({node}.south) ++ (0,-0.5) --{label_node} ({node}.south);" "\n"
    elif direction == "above":
        label_node = rf" node[right, yshift=-2pt, font=\sffamily\scriptsize, text=clrtext] {{{label}}}" if label else ""
        return rf"\draw[arrow] ({node}.north) --{label_node} ++ (0,0.5);" "\n"
    elif direction == "left":
        label_node = rf" node[above, yshift=2pt, font=\sffamily\scriptsize, text=clrtext] {{{label}}}" if label else ""
        return rf"\draw[arrow] ({node}.west) ++ (-0.5,0) --{label_node} ({node}.west);" "\n"
    else:  # right
        label_node = rf" node[above, xshift=6pt, yshift=2pt, font=\sffamily\scriptsize, text=clrtext] {{{label}}}" if label else ""
        return rf"\draw[arrow] ({node}.east) --{label_node} ++ (1.0,0);" "\n"


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
                   distance: float = 0.28) -> str:
    """Add a dimension annotation (e.g. '768') to a block."""
    anchor = "east" if side == "left" else "west"
    return (
        rf"\node[{side}={distance}cm of {node}, "
        rf"font=\sffamily\scriptsize, text=clrborder, anchor={anchor}] {{{text}}};" "\n"
    )


# ---------------------------------------------------------------------------
# LSTM Olah-style primitives (horizontal conveyor belt)
# ---------------------------------------------------------------------------

def olah_gate_node(name: str, symbol: str, fill: str,
                   x: float = 0, y: float = 0,
                   relative_to: str | None = None,
                   xshift: float = 0, yshift: float = 0) -> str:
    """Small colored circle for LSTM gate operations (σ, tanh, ×, +)."""
    if relative_to:
        pos = rf"at ([xshift={xshift}cm, yshift={yshift}cm]{relative_to})"
    else:
        pos = rf"at ({x},{y})"
    return (
        rf"\node[circle, draw=clrborder!60, fill={fill}, "
        rf"inner sep=3pt, minimum size=0.5cm, "
        rf"font=\sffamily\scriptsize\bfseries] ({name}) {pos} {{{symbol}}};" "\n"
    )


def olah_cell_highway(from_name: str, to_name: str) -> str:
    """Thick horizontal cell state line (the 'conveyor belt')."""
    return (
        rf"\draw[line width=2.5pt, color=clrresidual, ->, >=Stealth] "
        rf"({from_name}.east) -- ({to_name}.west);" "\n"
    )


def olah_thin_arrow(from_name: str, to_name: str,
                    from_anchor: str = "north", to_anchor: str = "south",
                    color: str = "clrborder!70") -> str:
    """Thin arrow for data flow inside LSTM cell."""
    return (
        rf"\draw[->, color={color}, line width=0.5pt] "
        rf"({from_name}.{from_anchor}) -- ({to_name}.{to_anchor});" "\n"
    )


def olah_curved_arrow(from_name: str, to_name: str,
                      bend: str = "left", color: str = "clrborder!70",
                      angle: int = 30) -> str:
    """Curved arrow for routing inside LSTM cell."""
    return (
        rf"\draw[->, color={color}, line width=0.5pt, bend {bend}={angle}] "
        rf"({from_name}) to ({to_name});" "\n"
    )


def olah_label(text: str, x: float = 0, y: float = 0,
               anchor: str = "center",
               relative_to: str | None = None,
               xshift: float = 0, yshift: float = 0) -> str:
    """Small text label for LSTM diagrams."""
    if relative_to:
        pos = rf"at ([xshift={xshift}cm, yshift={yshift}cm]{relative_to})"
    else:
        pos = rf"at ({x},{y})"
    return (
        rf"\node[font=\sffamily\scriptsize, text=clrborder, anchor={anchor}] "
        rf"{pos} {{{text}}};" "\n"
    )


# LSTM Olah gate color constants (Chris Olah blog palette)
OLAH_SIGMA = "clrnorm!80"       # yellow/amber for sigmoid
OLAH_TANH = "clrffn!50"         # pink/salmon for tanh
OLAH_MULTIPLY = "clrembed!70"   # green for pointwise multiply
OLAH_ADD = "clrresidual!50"     # blue for pointwise add


def olah_copy_dot(name: str, relative_to: str,
                  xshift: float = 0, yshift: float = 0) -> str:
    """Small filled black dot for branch/copy points (Chris Olah convention)."""
    return (
        rf"\node[circle, fill=clrborder, minimum size=3pt, inner sep=0pt] "
        rf"({name}) at ([xshift={xshift}cm, yshift={yshift}cm]{relative_to}) {{}};" "\n"
    )


def olah_bus_line(from_name: str, to_x: float, color: str = "clrborder!60") -> str:
    """Horizontal fan-out bus line from concat point to rightmost gate tap."""
    return (
        rf"\draw[-, color={color}, line width=0.8pt] "
        rf"({from_name}.east) -- ++({to_x},0);" "\n"
    )


def olah_branch_tap(tap_name: str, gate_name: str,
                    color: str = "clrborder!60") -> str:
    """Vertical branch tap arrow from bus line to gate node."""
    return (
        rf"\draw[->, color={color}, line width=0.5pt] "
        rf"({tap_name}) -- ({gate_name}.south);" "\n"
    )


# ---------------------------------------------------------------------------
# Cross-attention arrow (for EncoderDecoder)
# ---------------------------------------------------------------------------

def flat_cross_attention_arrow(from_name: str, to_name: str,
                               label: str = "",
                               style: str = "all") -> str:
    """Horizontal cross-attention arrow from encoder to decoder side."""
    label_part = ""
    if label:
        label_part = rf" node[midway, yshift=22pt, font=\sffamily\footnotesize\bfseries, text=clrborder, fill=white, inner sep=2pt] {{{label}}}"
    return (
        rf"\draw[->, thick, color=clrresidual, densely dashed, line width=1pt] "
        rf"({from_name}.east) --{label_part} ({to_name}.west);" "\n"
    )
