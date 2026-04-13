"""
ModernBERT architecture diagram generator.

Key architectural features visualized:
- Token embedding + RoPE (no learned positional embedding)
- Pre-LN (LayerNorm BEFORE attention and FFN)
- Alternating Local (128-window) / Global attention every 3rd layer
- GeGLU activation in FFN
- 22 layers (base), d_model=768, 12 heads
- Unpadding + Flash Attention annotations
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from plot_nn_mcp.pycore.tikzeng import (
    _render_box, _render_partitioned_box,
    to_begin, to_connection, to_cor, to_Dense,
    to_end, to_generate, to_head, to_Norm,
    to_repeat_bracket, to_skip, to_SoftMax, to_Sum,
)


def block_ModernBERTLayer(
    name: str,
    bottom: str,
    top: str,
    d_model: int = 768,
    num_heads: int = 12,
    d_ff: int = 2048,
    is_global: bool = False,
    offset: str = "(3,0,0)",
    size: tuple[float, float, float] = (25, 25, 5),
) -> list[str]:
    """One ModernBERT encoder layer: Pre-LN → MHA → Add → Pre-LN → GeGLU → Add."""
    h, d, w = size
    attn_label = "Global" if is_global else "Local"
    attn_color = r"\AttnColor" if is_global else "{rgb:blue,2;cyan,2;white,7}"
    attn_opacity = 0.85 if is_global else 0.5

    n1 = f"{name}_n1"
    at = f"{name}_at"
    r1 = f"{name}_r1"
    n2 = f"{name}_n2"
    g1 = f"{name}_g1"
    g2 = f"{name}_g2"
    r2 = f"{name}_r2"

    return [
        # ── Pre-LN → Attention ──
        to_Norm(name=n1, offset=offset, to=f"({bottom}-east)",
                width=0.3, height=h, depth=d, caption=" "),
        to_connection(bottom, n1),

        _render_partitioned_box(
            name=at, fill=attn_color, offset="(1.2,0,0)", to=f"({n1}-east)",
            height=h, width=w, depth=d, num_parts=num_heads,
            opacity=attn_opacity, caption=attn_label, zlabel=str(d_model),
        ),
        to_connection(n1, at),

        # ── Residual Add ──
        to_Sum(name=r1, offset="(1.2,0,0)", to=f"({at}-east)", radius=1.5, opacity=0.6),
        to_connection(at, r1),
        to_skip(bottom, r1, pos=1.25),

        # ── Pre-LN → GeGLU FFN ──
        to_Norm(name=n2, offset="(0.8,0,0)", to=f"({r1}-east)",
                width=0.3, height=h, depth=d, caption=" "),
        to_connection(r1, n2),

        _render_box(
            name=g1, fill=r"\FcColor", offset="(1.2,0,0)", to=f"({n2}-east)",
            height=h * 0.75, width=w * 0.45, depth=d,
            caption="GeGLU", xlabel=f"{{{{{d_ff}, }}}}",
        ),
        to_connection(n2, g1),

        _render_box(
            name=g2, fill=r"\FcReluColor", offset="(0,0,0)", to=f"({g1}-east)",
            height=h * 0.55, width=w * 0.45, depth=d,
            xlabel=f"{{{{{d_model}, }}}}",
        ),
        to_connection(g1, g2),

        # ── Residual Add ──
        to_Sum(name=r2, offset="(1.2,0,0)", to=f"({g2}-east)", radius=1.5, opacity=0.6),
        to_connection(g2, r2),
        to_skip(n2, r2, pos=1.25),

        # Invisible anchor for chaining
        _render_box(
            name=top, fill=r"\NormColor", offset="(0.2,0,0)", to=f"({r2}-east)",
            height=1, width=0.01, depth=1, opacity=0.0,
        ),
        to_connection(r2, top),
    ]


def modernbert(
    n_layers: int = 22,
    d_model: int = 768,
    n_heads: int = 12,
    d_ff: int = 2048,
    global_every: int = 3,
    show_layers: int = 4,
) -> list[str]:
    """Generate ModernBERT-base architecture."""
    arch = [to_head("."), to_cor(), to_begin()]
    size = (25, 25, 5)

    # ── Token Embedding ──
    arch.append(_render_box(
        name="inp", fill=r"\EmbedColor", offset="(0,0,0)", to="(0,0,0)",
        height=30, width=2, depth=12, caption="Token Emb",
    ))
    arch.append(
        rf"\node[above=6pt, font=\scriptsize\bfseries, text=black!60] "
        rf"at (inp-north) {{+\,RoPE}};" "\n"
    )

    # Post-embedding LN
    arch.append(to_Norm("en", offset="(2,0,0)", to="(inp-east)",
                        width=0.3, height=25, depth=25, caption=" "))
    arch.append(to_connection("inp", "en"))

    # ── Encoder layers ──
    prev = "en"
    first_attn = last_out = None

    for i in range(show_layers):
        is_global = ((i + 1) % global_every == 0)
        top = f"L{i}_o"
        arch.extend(block_ModernBERTLayer(
            f"L{i}", prev, top,
            d_model=d_model, num_heads=n_heads, d_ff=d_ff,
            is_global=is_global, offset="(2,0,0)", size=size,
        ))
        if first_attn is None:
            first_attn = f"L{i}_at"
        last_out = top
        prev = top

    # ── ×N bracket ──
    arch.append(to_repeat_bracket(
        first_attn, last_out, rf"\times {n_layers}", xshift=4.0,
    ))

    # ── Output head ──
    arch.append(to_Dense("cls", d_model, offset="(3,0,0)", to=f"({last_out}-east)",
                         width=2, height=3, depth=22, caption="[CLS] Pool"))
    arch.append(to_connection(last_out, "cls"))

    arch.append(to_SoftMax("out", d_model, offset="(1.5,0,0)", to="(cls-east)",
                           caption="Output"))
    arch.append(to_connection("cls", "out"))

    # ── Subtitle ──
    arch.append(
        r"\node[below=25pt, font=\small, text width=18cm, align=center] "
        r"at (current bounding box.south) "
        r"{\textbf{ModernBERT\,--\,base}"
        r" \quad 22 layers \,$\cdot$\, 768\,dim \,$\cdot$\, 12 heads"
        r" \quad Alternating local\,(w\!=\!128)\,/\,global attention"
        r" \quad Pre-LN \,$\cdot$\, GeGLU \,$\cdot$\, RoPE"
        r" \,$\cdot$\, Flash Attention \,$\cdot$\, Unpadding};" "\n"
    )

    arch.append(to_end())
    return arch


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    arch = modernbert(show_layers=4)
    tex_path = os.path.join(output_dir, "modernbert.tex")
    to_generate(arch, tex_path)
    print(f"Generated: {tex_path}")

    # Compile
    import shutil
    import subprocess
    layers_src = os.path.join(os.path.dirname(__file__), "..", "src", "plot_nn_mcp", "layers")
    layers_dst = os.path.join(output_dir, "layers")
    if not os.path.exists(layers_dst):
        shutil.copytree(layers_src, layers_dst)

    for compiler in ("tectonic", "pdflatex"):
        if shutil.which(compiler):
            env = os.environ.copy()
            env["TEXINPUTS"] = layers_dst + ":" + output_dir + ":"
            cmd = ([compiler, tex_path] if compiler == "tectonic"
                   else [compiler, "-interaction=nonstopmode", "-output-directory", output_dir, tex_path])
            subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           env=env, cwd=output_dir)
            pdf = tex_path.replace(".tex", ".pdf")
            if os.path.exists(pdf):
                print(f"Compiled with {compiler}: {pdf}")
                break
    else:
        print("No LaTeX compiler — .tex generated")
