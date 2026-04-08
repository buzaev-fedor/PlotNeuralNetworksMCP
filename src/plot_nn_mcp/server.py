"""
MCP server for generating neural network architecture diagrams.
Uses PlotNeuralNet engine to produce LaTeX/TikZ and PDF output.
"""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from .compiler import compile_tex, copy_layers_to, prepare_work_dir, write_and_compile
from .presets import PRESETS
from .pycore.tikzeng import (
    to_begin, to_branch, to_connection, to_cor, to_end, to_head,
    to_merge, to_repeat_bracket, to_skip, to_skip_bottom,
)
from .registry import LAYER_REGISTRY, coerce_params, get_layer_metadata

mcp = FastMCP(
    "PlotNeuralNetwork",
    instructions="Generate publication-quality neural network architecture diagrams using LaTeX/TikZ",
)


# ---------------------------------------------------------------------------
# Architecture builder
# ---------------------------------------------------------------------------

def _build_arch(
    layers: list[dict],
    connections: list[dict] | None = None,
    skip_connections: list[dict] | None = None,
) -> list[str]:
    """Build architecture list from layer/connection specs."""
    arch = [to_head("."), to_cor(), to_begin()]

    for layer in layers:
        layer_type = layer.pop("type")
        spec = LAYER_REGISTRY.get(layer_type)
        if spec is None:
            raise ValueError(
                f"Unknown layer type: {layer_type}. "
                f"Available: {list(LAYER_REGISTRY.keys())}"
            )
        params = coerce_params(layer_type, layer)
        arch.append(spec.builder(**params))

    for conn in connections or []:
        arch.append(to_connection(conn["from"], conn["to"]))

    for skip in skip_connections or []:
        arch.append(to_skip(skip["from"], skip["to"], pos=skip.get("pos", 1.25)))

    arch.append(to_end())
    return arch


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_layer_types() -> str:
    """List all available neural network layer types and their parameters.

    Returns a JSON description of each layer type with its parameters,
    defaults, and usage examples.
    """
    layer_info = get_layer_metadata()
    blocks_info = {
        "block_2ConvPool": {
            "description": "Two convolutions + ReLU + pooling block",
            "params": "name, botton, top, s_filer=256, n_filer=64, offset='(1,0,0)', size=(32,32,3.5), opacity=0.5",
        },
        "block_Unconv": {
            "description": "Unpooling + residual convolutions decoder block",
            "params": "name, botton, top, s_filer=256, n_filer=64, offset='(1,0,0)', size=(32,32,3.5), opacity=0.5",
        },
        "block_Res": {
            "description": "Residual block with skip connection",
            "params": "num, name, botton, top, s_filer=256, n_filer=64, offset='(0,0,0)', size=(32,32,3.5), opacity=0.5",
        },
        "block_TransformerEncoderLayer": {
            "description": "Transformer encoder layer: MHA → Add&Norm → FFN → Add&Norm",
            "params": "name, botton, top, d_model=512, num_heads=8, d_ff=2048, offset='(2,0,0)', size=(32,32,6)",
        },
        "block_TransformerDecoderLayer": {
            "description": "Transformer decoder layer: Masked MHA → Cross-MHA → FFN with Add&Norm",
            "params": "name, botton, top, encoder_out, d_model=512, num_heads=8, d_ff=2048, offset='(2,0,0)', size=(32,32,6)",
        },
        "block_EmbeddingStack": {
            "description": "Token + position (+ optional segment) embeddings → sum → norm",
            "params": "name, botton, top, d_model=512, offset='(1,0,0)', size=(32,32,2), include_segment=False",
        },
        "block_MLPStack": {
            "description": "Stack of N dense layers with connections",
            "params": "name, botton, top, n_layers=3, hidden_size=64, offset='(1,0,0)', size=(8,25,2)",
        },
        "block_FourierLayer": {
            "description": "Fourier layer: spectral conv + linear skip → sum",
            "params": "name, botton, top, modes=16, offset='(2,0,0)', size=(32,32,4)",
        },
    }
    return json.dumps({"layers": layer_info, "blocks": blocks_info}, indent=2)


@mcp.tool()
def generate_diagram(
    layers: list[dict],
    connections: list[dict] | None = None,
    skip_connections: list[dict] | None = None,
    output_dir: str | None = None,
    filename: str = "nn_architecture",
    compile_pdf: bool = True,
) -> str:
    """Generate a neural network architecture diagram from layer specifications.

    Args:
        layers: List of layer dicts. Each must have a "type" key (e.g. "Conv", "Pool")
                plus the layer's parameters. Example:
                [
                    {"type": "Conv", "name": "conv1", "s_filer": 512, "n_filer": 64,
                     "offset": "(0,0,0)", "to": "(0,0,0)", "height": 64, "depth": 64, "width": 2},
                    {"type": "Pool", "name": "pool1", "offset": "(0,0,0)", "to": "(conv1-east)",
                     "height": 32, "depth": 32}
                ]
        connections: List of connection dicts with "from" and "to" layer names.
                     Example: [{"from": "pool1", "to": "conv2"}]
        skip_connections: List of skip connection dicts with "from", "to", and optional "pos".
                          Example: [{"from": "conv1", "to": "conv5", "pos": 1.25}]
        output_dir: Directory to save output files. Uses temp dir if not specified.
        filename: Base filename (without extension). Default: "nn_architecture".
        compile_pdf: Whether to compile LaTeX to PDF (requires pdflatex). Default: true.

    Returns:
        JSON with paths to generated files and the LaTeX source.
    """
    work_dir, _ = prepare_work_dir(output_dir)
    layers_copy = [dict(l) for l in layers]
    arch = _build_arch(layers_copy, connections, skip_connections)
    result = write_and_compile(arch, work_dir, filename, compile_pdf)
    return json.dumps(result, indent=2)


@mcp.tool()
def generate_preset(
    preset: str,
    params: dict | None = None,
    output_dir: str | None = None,
    filename: str | None = None,
    compile_pdf: bool = True,
) -> str:
    """Generate a diagram from a preset neural network architecture.

    Args:
        preset: Architecture preset name. One of:
                - "simple_cnn" - Simple CNN with 3 conv+pool layers and softmax
                - "vgg16" - VGG-16 style architecture
                - "unet" - U-Net encoder-decoder architecture
                - "resnet" - ResNet-style with residual connections
                - "transformer" - Full encoder-decoder Transformer (params: n_enc, n_dec, d_model, n_heads, d_ff)
                - "bert" - BERT encoder-only (params: n_layers, d_model, n_heads, d_ff)
                - "gpt" - GPT decoder-only (params: n_layers, d_model, n_heads, d_ff)
                - "vit" - Vision Transformer (params: n_layers, d_model, n_heads, d_ff, patch_size)
                - "pinn" - Physics-Informed Neural Network (params: n_hidden, hidden_size)
                - "fno" - Fourier Neural Operator (params: n_layers, modes, width)
        params: Optional dict of architecture parameters (e.g. {"n_layers": 6, "d_model": 512}).
        output_dir: Directory to save output files. Uses temp dir if not specified.
        filename: Base filename. Defaults to the preset name.
        compile_pdf: Whether to compile to PDF. Default: true.

    Returns:
        JSON with paths to generated files and the LaTeX source.
    """
    if preset not in PRESETS:
        return json.dumps({
            "error": f"Unknown preset: {preset}",
            "available_presets": list(PRESETS.keys()),
        })

    work_dir, _ = prepare_work_dir(output_dir)
    arch = PRESETS[preset](**(params or {}))
    result = write_and_compile(arch, work_dir, filename or preset, compile_pdf)
    result["preset"] = preset
    return json.dumps(result, indent=2)


@mcp.tool()
def compile_tex_to_pdf(tex_path: str) -> str:
    """Compile an existing .tex file to PDF using pdflatex.

    Args:
        tex_path: Absolute path to the .tex file.

    Returns:
        JSON with compilation result and PDF path.
    """
    if not os.path.exists(tex_path):
        return json.dumps({"error": f"File not found: {tex_path}"})

    work_dir = os.path.dirname(tex_path)
    copy_layers_to(work_dir)

    pdf_path = compile_tex(tex_path, work_dir)
    if pdf_path:
        return json.dumps({"status": "success", "pdf_path": pdf_path})
    return json.dumps({
        "status": "error",
        "message": "Compilation failed. Ensure pdflatex and texlive-latex-extra are installed.",
    })


@mcp.tool()
def generate_latex_snippet(
    layers: list[dict],
    connections: list[dict] | None = None,
    skip_connections: list[dict] | None = None,
) -> str:
    """Generate only the LaTeX/TikZ source code without writing files.

    Useful for embedding in existing LaTeX documents or previewing.

    Args:
        layers: List of layer dicts (same format as generate_diagram).
        connections: List of connection dicts.
        skip_connections: List of skip connection dicts.

    Returns:
        The complete LaTeX source as a string.
    """
    layers_copy = [dict(l) for l in layers]
    arch = _build_arch(layers_copy, connections, skip_connections)
    return "".join(arch)


@mcp.tool()
def generate_architecture(
    name: str,
    layers: list[dict],
    theme: str = "modern",
    show_n: int = 4,
    output_dir: str | None = None,
    filename: str = "architecture",
    compile_pdf: bool = True,
) -> str:
    """Generate a architecture diagram using the semantic DSL with 2D vertical layout.

    Produces clean, publication-quality diagrams with themed colors,
    auto-grouping of repeated blocks, and vertical stacking.

    Args:
        name: Architecture title (e.g. "ModernBERT-base").
        layers: List of layer dicts. Each must have a "layer" key specifying the type.
                Supported types and their params:
                - {"layer": "Embedding", "d_model": 768, "rope": true, "label": "..."}
                - {"layer": "TransformerBlock", "attention": "local"|"global"|"self"|"masked",
                   "norm": "pre_ln"|"post_ln", "ffn": "geglu"|"gelu"|"swiglu",
                   "d_ff": 2048, "heads": 12, "d_model": 768}
                - {"layer": "ConvBlock", "filters": 64, "kernel_size": 3, "pool": "max"}
                - {"layer": "DenseLayer", "units": 256}
                - {"layer": "ClassificationHead", "label": "Output"}
                - {"layer": "FourierBlock", "modes": 16}
                - {"layer": "CustomBlock", "text": "My Layer", "color_role": "attention"}
        theme: Color theme. One of: "modern", "paper", "vibrant", "monochrome".
        show_n: Number of repeated blocks to show explicitly (rest collapsed with xN).
        output_dir: Directory for output files. Uses temp dir if not specified.
        filename: Base filename (without extension).
        compile_pdf: Whether to compile to PDF.

    Returns:
        JSON with tex_path, tex_source, status, and optional pdf_path.
    """
    from .dsl import (
        Architecture, Embedding, TransformerBlock, ConvBlock,
        DenseLayer, ClassificationHead, FourierBlock, CustomBlock,
    )

    LAYER_MAP = {
        "Embedding": Embedding,
        "TransformerBlock": TransformerBlock,
        "ConvBlock": ConvBlock,
        "DenseLayer": DenseLayer,
        "ClassificationHead": ClassificationHead,
        "FourierBlock": FourierBlock,
        "CustomBlock": CustomBlock,
    }

    arch = Architecture(name, theme=theme)
    for spec in layers:
        spec = dict(spec)
        layer_type = spec.pop("layer")
        cls = LAYER_MAP.get(layer_type)
        if cls is None:
            return json.dumps({
                "error": f"Unknown DSL layer type: {layer_type}",
                "available": list(LAYER_MAP.keys()),
            })
        arch.add(cls(**spec))

    work_dir, _ = prepare_work_dir(output_dir)
    tex_source = arch.render(show_n=show_n)
    tex_path = os.path.join(work_dir, f"{filename}.tex")
    with open(tex_path, "w") as f:
        f.write(tex_source)

    result = {
        "tex_path": tex_path,
        "work_dir": work_dir,
        "tex_source": tex_source,
    }

    if compile_pdf:
        from .compiler import compile_tex
        pdf_path = compile_tex(tex_path, work_dir)
        if pdf_path:
            result["pdf_path"] = pdf_path
            result["status"] = "success"
        else:
            result["status"] = "tex_generated"
            result["note"] = "pdflatex not found or compilation failed."
    else:
        result["status"] = "tex_generated"

    return json.dumps(result, indent=2)


@mcp.tool()
def list_themes() -> str:
    """List available color themes for architecture diagrams.

    Returns JSON with theme names and their color palettes.
    """
    from .themes import THEMES
    result = {}
    for name, theme in THEMES.items():
        result[name] = {
            "attention": f"#{theme.attention}",
            "ffn": f"#{theme.ffn}",
            "norm": f"#{theme.norm}",
            "embed": f"#{theme.embed}",
            "output": f"#{theme.output}",
        }
    return json.dumps(result, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
