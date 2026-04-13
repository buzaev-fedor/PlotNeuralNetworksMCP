"""
Color theme system for neural network architecture diagrams.

Each theme maps semantic roles to TikZ-compatible color definitions.
Themes produce ``\\definecolor{...}{HTML}{...}`` blocks for use with TikZ nodes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """A harmonious color palette for architecture diagrams."""
    name: str
    # Semantic color roles → hex RGB (without #)
    attention: str
    attention_alt: str   # for local/masked/cross distinction
    ffn: str
    norm: str
    embed: str
    residual: str
    output: str
    dense: str
    spectral: str        # FNO
    physics: str         # PINN
    background: str
    border: str
    text: str
    group_frame: str     # GroupFrame border
    group_fill: str      # GroupFrame background


THEMES: dict[str, Theme] = {
    "modern": Theme(
        name="modern",
        attention="45B7D1",    # bright teal
        attention_alt="96E6F0", # light teal (local/masked)
        ffn="F97068",          # coral red
        norm="FFD93D",         # warm amber
        embed="6BCB77",        # fresh green
        residual="4D96FF",     # clean blue
        output="C084FC",       # soft purple
        dense="4D96FF",        # blue
        spectral="22D3EE",     # cyan
        physics="FB923C",      # orange
        background="FAFAFA",
        border="334155",       # slate-700
        text="1E293B",         # slate-800
        group_frame="94A3B8",  # slate-400
        group_fill="F1F5F9",   # slate-100
    ),
    "paper": Theme(
        name="paper",
        attention="6B9BD2",    # steel blue
        attention_alt="A8C8E8", # light steel
        ffn="D4956A",          # muted coral
        norm="E8D78E",         # pale gold
        embed="7DB892",        # sage green
        residual="8EAEC8",     # powder blue
        output="B08EAF",       # dusty purple
        dense="8EAEC8",
        spectral="7BC8C8",
        physics="D4956A",
        background="FFFFFF",
        border="4A4A4A",
        text="333333",
        group_frame="AAAAAA",
        group_fill="F5F5F5",
    ),
    "vibrant": Theme(
        name="vibrant",
        attention="00B4D8",    # vivid cyan
        attention_alt="90E0EF", # light cyan
        ffn="E63946",          # vivid red
        norm="FFBE0B",         # vivid yellow
        embed="06D6A0",        # vivid green
        residual="118AB2",     # vivid blue
        output="7B2CBF",       # vivid purple
        dense="118AB2",
        spectral="00F5D4",
        physics="F77F00",
        background="FFFFFF",
        border="073B4C",       # dark teal
        text="073B4C",
        group_frame="073B4C",
        group_fill="EDF6F9",
    ),
    "monochrome": Theme(
        name="monochrome",
        attention="666666",
        attention_alt="999999",
        ffn="888888",
        norm="CCCCCC",
        embed="AAAAAA",
        residual="777777",
        output="555555",
        dense="777777",
        spectral="888888",
        physics="999999",
        background="FFFFFF",
        border="333333",
        text="222222",
        group_frame="999999",
        group_fill="F0F0F0",
    ),
    "arxiv": Theme(
        name="arxiv",
        attention="B8D4E3",    # light cyan (Vaswani-style)
        attention_alt="D4E8F0", # lighter cyan
        ffn="F5D0A9",          # light orange
        norm="FFF3C4",         # light yellow
        embed="C8E6C9",        # light green
        residual="90CAF9",     # soft blue
        output="E1BEE7",       # soft lavender
        dense="90CAF9",
        spectral="B2EBF2",    # pale cyan
        physics="FFCCBC",      # pale orange
        background="FFFFFF",
        border="424242",       # gray-800
        text="212121",         # gray-900
        group_frame="9E9E9E",  # gray-500
        group_fill="FAFAFA",   # gray-50
    ),
    "nature": Theme(
        name="nature",
        attention="7B9EB2",    # steel blue (restrained)
        attention_alt="A3C1D0",
        ffn="C4956A",          # warm brown
        norm="D4C5A0",         # khaki
        embed="8BAA7F",        # olive green
        residual="7B9EB2",     # steel blue
        output="9B8EAD",       # muted purple
        dense="7B9EB2",
        spectral="7FB5B0",     # sage teal
        physics="C4956A",
        background="FFFFFF",
        border="3C3C3C",
        text="2B2B2B",
        group_frame="8C8C8C",
        group_fill="F7F7F7",
    ),
}


def get_theme(name: str) -> Theme:
    if name not in THEMES:
        raise ValueError(f"Unknown theme: {name!r}. Available: {list(THEMES.keys())}")
    return THEMES[name]


def theme_to_tikz_colors(theme: Theme) -> str:
    """Generate TikZ \\definecolor commands for all semantic roles."""
    lines = []
    for f in dataclasses.fields(theme):
        if f.name == "name":
            continue
        hex_color = getattr(theme, f.name)
        lines.append(rf"\definecolor{{clr{f.name}}}{{HTML}}{{{hex_color}}}")
    return "\n".join(lines) + "\n"
