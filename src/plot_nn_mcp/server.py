"""
MCP server for generating neural network architecture diagrams.
Uses PlotNeuralNet engine to produce LaTeX/TikZ and PDF output.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .pycore.blocks import block_2ConvPool, block_Res, block_Unconv
from .pycore.tikzeng import (
    to_begin,
    to_connection,
    to_Conv,
    to_ConvConvRelu,
    to_ConvRes,
    to_ConvSoftMax,
    to_cor,
    to_end,
    to_generate,
    to_head,
    to_input,
    to_Pool,
    to_skip,
    to_SoftMax,
    to_Sum,
    to_UnPool,
)

mcp = FastMCP(
    "PlotNeuralNetwork",
    instructions="Generate publication-quality neural network architecture diagrams using LaTeX/TikZ",
)

LAYERS_DIR = str(Path(__file__).parent / "layers")

# --- Layer dispatch ---

LAYER_BUILDERS = {
    "Conv": to_Conv,
    "ConvConvRelu": to_ConvConvRelu,
    "Pool": to_Pool,
    "UnPool": to_UnPool,
    "ConvRes": to_ConvRes,
    "ConvSoftMax": to_ConvSoftMax,
    "SoftMax": to_SoftMax,
    "Sum": to_Sum,
    "Input": to_input,
}


def _coerce_params(layer_type: str, params: dict) -> dict:
    """Convert JSON-friendly params to the types expected by builder functions."""
    p = dict(params)
    # tuples for ConvConvRelu
    if layer_type == "ConvConvRelu":
        if "n_filer" in p and isinstance(p["n_filer"], list):
            p["n_filer"] = tuple(p["n_filer"])
        if "width" in p and isinstance(p["width"], list):
            p["width"] = tuple(p["width"])
    # numeric conversions
    for key in ("height", "depth", "width", "s_filer", "n_filer", "radius", "opacity"):
        if key in p and isinstance(p[key], str) and key not in ("offset", "to"):
            try:
                p[key] = float(p[key]) if "." in p[key] else int(p[key])
            except ValueError:
                pass
    return p


def _build_arch(layers: list[dict], connections: list[dict] | None = None,
                skip_connections: list[dict] | None = None) -> list[str]:
    """Build architecture list from layer/connection specs."""
    # Use a relative path that points to the layers dir
    arch = [to_head("."), to_cor(), to_begin()]

    for layer in layers:
        layer_type = layer.pop("type")
        builder = LAYER_BUILDERS.get(layer_type)
        if builder is None:
            raise ValueError(f"Unknown layer type: {layer_type}. "
                             f"Available: {list(LAYER_BUILDERS.keys())}")
        params = _coerce_params(layer_type, layer)
        arch.append(builder(**params))

    for conn in (connections or []):
        arch.append(to_connection(conn["from"], conn["to"]))

    for skip in (skip_connections or []):
        pos = skip.get("pos", 1.25)
        arch.append(to_skip(skip["from"], skip["to"], pos=pos))

    arch.append(to_end())
    return arch


def _compile_tex(tex_path: str, work_dir: str) -> str | None:
    """Compile .tex to .pdf using pdflatex. Returns pdf path or None."""
    if not shutil.which("pdflatex"):
        return None
    env = os.environ.copy()
    env["TEXINPUTS"] = LAYERS_DIR + ":" + work_dir + ":"
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-output-directory", work_dir,
         tex_path],
        capture_output=True, text=True, timeout=60, env=env,
    )
    pdf_path = tex_path.replace(".tex", ".pdf")
    if os.path.exists(pdf_path):
        return pdf_path
    return None


def _prepare_work_dir(output_dir: str | None) -> tuple[str, bool]:
    """Return (work_dir, is_temp). Creates output_dir if needed."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir, False
    return tempfile.mkdtemp(prefix="plotnn_"), True


def _copy_layers_to(work_dir: str):
    """Copy LaTeX layer definitions into work directory for pdflatex."""
    layers_dest = os.path.join(work_dir, "layers")
    if not os.path.exists(layers_dest):
        shutil.copytree(LAYERS_DIR, layers_dest)


# --- MCP Tools ---

@mcp.tool()
def list_layer_types() -> str:
    """List all available neural network layer types and their parameters.

    Returns a JSON description of each layer type with its parameters,
    defaults, and usage examples.
    """
    layer_info = {
        "Conv": {
            "description": "Convolution layer (3D box)",
            "params": {
                "name": "str (required) - unique layer identifier",
                "s_filer": "int (default 256) - spatial filter size label",
                "n_filer": "int (default 64) - number of filters label",
                "offset": "str (default '(0,0,0)') - position offset from reference",
                "to": "str (default '(0,0,0)') - reference position or layer anchor",
                "width": "float (default 1) - box width in TikZ units",
                "height": "float (default 40) - box height",
                "depth": "float (default 40) - box depth",
                "caption": "str - label below the layer",
            },
        },
        "ConvConvRelu": {
            "description": "Two convolutions with ReLU (banded box)",
            "params": {
                "name": "str (required)",
                "s_filer": "int (default 256)",
                "n_filer": "list[int,int] (default [64,64]) - filter counts for each conv",
                "offset": "str (default '(0,0,0)')",
                "to": "str (default '(0,0,0)')",
                "width": "list[float,float] (default [2,2]) - widths for each conv",
                "height": "float (default 40)",
                "depth": "float (default 40)",
                "caption": "str",
            },
        },
        "Pool": {
            "description": "Pooling layer",
            "params": {
                "name": "str (required)",
                "offset": "str", "to": "str",
                "width": "float (default 1)",
                "height": "float (default 32)",
                "depth": "float (default 32)",
                "opacity": "float (default 0.5)",
                "caption": "str",
            },
        },
        "UnPool": {
            "description": "Unpooling / deconvolution layer",
            "params": {
                "name": "str (required)",
                "offset": "str", "to": "str",
                "width": "float (default 1)",
                "height": "float (default 32)",
                "depth": "float (default 32)",
                "opacity": "float (default 0.5)",
                "caption": "str",
            },
        },
        "ConvRes": {
            "description": "Residual convolution layer (banded, semi-transparent)",
            "params": {
                "name": "str (required)",
                "s_filer": "int (default 256)",
                "n_filer": "int (default 64)",
                "offset": "str", "to": "str",
                "width": "float (default 6)",
                "height": "float (default 40)",
                "depth": "float (default 40)",
                "opacity": "float (default 0.2)",
                "caption": "str",
            },
        },
        "ConvSoftMax": {
            "description": "Convolution followed by softmax",
            "params": {
                "name": "str (required)",
                "s_filer": "int (default 40)",
                "offset": "str", "to": "str",
                "width": "float (default 1)",
                "height": "float (default 40)",
                "depth": "float (default 40)",
                "caption": "str",
            },
        },
        "SoftMax": {
            "description": "Standalone softmax layer",
            "params": {
                "name": "str (required)",
                "s_filer": "int (default 10)",
                "offset": "str", "to": "str",
                "width": "float (default 1.5)",
                "height": "float (default 3)",
                "depth": "float (default 25)",
                "opacity": "float (default 0.8)",
                "caption": "str",
            },
        },
        "Sum": {
            "description": "Element-wise sum operation (ball with + symbol)",
            "params": {
                "name": "str (required)",
                "offset": "str", "to": "str",
                "radius": "float (default 2.5)",
                "opacity": "float (default 0.6)",
            },
        },
        "Input": {
            "description": "Input image embedding",
            "params": {
                "pathfile": "str (required) - path to image file",
                "to": "str (default '(-3,0,0)')",
                "width": "float (default 8) - image width in cm",
                "height": "float (default 8) - image height in cm",
                "name": "str (default 'temp')",
            },
        },
    }

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
    work_dir, is_temp = _prepare_work_dir(output_dir)
    _copy_layers_to(work_dir)

    # deep copy layers to avoid mutating input
    layers_copy = [dict(l) for l in layers]
    arch = _build_arch(layers_copy, connections, skip_connections)

    tex_path = os.path.join(work_dir, f"{filename}.tex")
    to_generate(arch, tex_path)

    result = {
        "tex_path": tex_path,
        "work_dir": work_dir,
        "tex_source": open(tex_path).read(),
    }

    if compile_pdf:
        pdf_path = _compile_tex(tex_path, work_dir)
        if pdf_path:
            result["pdf_path"] = pdf_path
            result["status"] = "success"
        else:
            result["status"] = "tex_generated"
            result["note"] = "pdflatex not found or compilation failed. LaTeX source is available."
    else:
        result["status"] = "tex_generated"

    return json.dumps(result, indent=2)


@mcp.tool()
def generate_preset(
    preset: str,
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
        output_dir: Directory to save output files. Uses temp dir if not specified.
        filename: Base filename. Defaults to the preset name.
        compile_pdf: Whether to compile to PDF. Default: true.

    Returns:
        JSON with paths to generated files and the LaTeX source.
    """
    presets = {
        "simple_cnn": _preset_simple_cnn,
        "vgg16": _preset_vgg16,
        "unet": _preset_unet,
        "resnet": _preset_resnet,
    }

    if preset not in presets:
        return json.dumps({
            "error": f"Unknown preset: {preset}",
            "available_presets": list(presets.keys()),
        })

    work_dir, is_temp = _prepare_work_dir(output_dir)
    _copy_layers_to(work_dir)

    fname = filename or preset
    arch = presets[preset]()
    tex_path = os.path.join(work_dir, f"{fname}.tex")
    to_generate(arch, tex_path)

    result = {
        "tex_path": tex_path,
        "work_dir": work_dir,
        "preset": preset,
        "tex_source": open(tex_path).read(),
    }

    if compile_pdf:
        pdf_path = _compile_tex(tex_path, work_dir)
        if pdf_path:
            result["pdf_path"] = pdf_path
            result["status"] = "success"
        else:
            result["status"] = "tex_generated"
            result["note"] = "pdflatex not found or compilation failed. LaTeX source is available."
    else:
        result["status"] = "tex_generated"

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
    _copy_layers_to(work_dir)

    pdf_path = _compile_tex(tex_path, work_dir)
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


# --- Preset architectures ---

def _preset_simple_cnn() -> list[str]:
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


def _preset_vgg16() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]

    # Block 1
    arch.append(to_ConvConvRelu("b1", 224, (64, 64), offset="(0,0,0)", to="(0,0,0)",
                                width=(2, 2), height=64, depth=64, caption="Block1"))
    arch.append(to_Pool("pool1", offset="(0,0,0)", to="(b1-east)",
                        height=48, depth=48))
    # Block 2
    arch.append(to_ConvConvRelu("b2", 112, (128, 128), offset="(1,0,0)", to="(pool1-east)",
                                width=(3, 3), height=48, depth=48, caption="Block2"))
    arch.append(to_Pool("pool2", offset="(0,0,0)", to="(b2-east)",
                        height=36, depth=36))
    arch.append(to_connection("pool1", "b2"))
    # Block 3
    arch.append(to_ConvConvRelu("b3", 56, (256, 256), offset="(1,0,0)", to="(pool2-east)",
                                width=(4, 4), height=36, depth=36, caption="Block3"))
    arch.append(to_Pool("pool3", offset="(0,0,0)", to="(b3-east)",
                        height=24, depth=24))
    arch.append(to_connection("pool2", "b3"))
    # Block 4
    arch.append(to_ConvConvRelu("b4", 28, (512, 512), offset="(1,0,0)", to="(pool3-east)",
                                width=(5, 5), height=24, depth=24, caption="Block4"))
    arch.append(to_Pool("pool4", offset="(0,0,0)", to="(b4-east)",
                        height=16, depth=16))
    arch.append(to_connection("pool3", "b4"))
    # Block 5
    arch.append(to_ConvConvRelu("b5", 14, (512, 512), offset="(1,0,0)", to="(pool4-east)",
                                width=(5, 5), height=16, depth=16, caption="Block5"))
    arch.append(to_Pool("pool5", offset="(0,0,0)", to="(b5-east)",
                        height=8, depth=8))
    arch.append(to_connection("pool4", "b5"))
    # FC + Softmax
    arch.append(to_SoftMax("soft1", 1000, offset="(2,0,0)", to="(pool5-east)",
                           width=1.5, height=3, depth=25, caption="Softmax"))
    arch.append(to_connection("pool5", "soft1"))

    arch.append(to_end())
    return arch


def _preset_unet() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]

    # Encoder
    arch.extend(block_2ConvPool("b1", "start", "pool_b1", s_filer=512, n_filer=64,
                                offset="(0,0,0)", size=(64, 64, 3.5)))
    arch.extend(block_2ConvPool("b2", "pool_b1", "pool_b2", s_filer=256, n_filer=128,
                                offset="(1,0,0)", size=(48, 48, 4.5)))
    arch.extend(block_2ConvPool("b3", "pool_b2", "pool_b3", s_filer=128, n_filer=256,
                                offset="(1,0,0)", size=(32, 32, 6)))
    arch.extend(block_2ConvPool("b4", "pool_b3", "pool_b4", s_filer=64, n_filer=512,
                                offset="(1,0,0)", size=(16, 16, 8)))

    # Bottleneck
    arch.append(to_ConvConvRelu("bneck", 32, (1024, 1024), offset="(2,0,0)",
                                to="(pool_b4-east)", width=(10, 10), height=8,
                                depth=8, caption="Bottleneck"))
    arch.append(to_connection("pool_b4", "bneck"))

    # Decoder
    arch.extend(block_Unconv("b5", "bneck", "end_b5", s_filer=64, n_filer=512,
                             offset="(2,0,0)", size=(16, 16, 8)))
    arch.extend(block_Unconv("b6", "end_b5", "end_b6", s_filer=128, n_filer=256,
                             offset="(2,0,0)", size=(32, 32, 6)))
    arch.extend(block_Unconv("b7", "end_b6", "end_b7", s_filer=256, n_filer=128,
                             offset="(2,0,0)", size=(48, 48, 4.5)))
    arch.extend(block_Unconv("b8", "end_b7", "end_b8", s_filer=512, n_filer=64,
                             offset="(2,0,0)", size=(64, 64, 3.5)))

    # Output
    arch.append(to_ConvSoftMax("soft1", 512, offset="(1,0,0)", to="(end_b8-east)",
                               width=1, height=64, depth=64, caption="Output"))
    arch.append(to_connection("end_b8", "soft1"))

    # Skip connections
    arch.append(to_skip("ccr_b4", "unpool_b5", pos=1.25))
    arch.append(to_skip("ccr_b3", "unpool_b6", pos=1.25))
    arch.append(to_skip("ccr_b2", "unpool_b7", pos=1.25))
    arch.append(to_skip("ccr_b1", "unpool_b8", pos=1.25))

    arch.append(to_end())
    return arch


def _preset_resnet() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]

    # Initial conv
    arch.append(to_Conv("conv1", 224, 64, offset="(0,0,0)", to="(0,0,0)",
                        height=64, depth=64, width=3, caption="Conv1"))
    arch.append(to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
                        height=48, depth=48))

    # Residual blocks
    arch.extend(block_Res(3, "res_a", "pool1", "res_a_top", s_filer=56,
                          n_filer=64, offset="(1,0,0)", size=(48, 48, 3.5)))

    arch.extend(block_Res(4, "res_b", "res_a_top", "res_b_top", s_filer=28,
                          n_filer=128, offset="(1,0,0)", size=(32, 32, 4.5)))

    arch.extend(block_Res(3, "res_c", "res_b_top", "res_c_top", s_filer=14,
                          n_filer=256, offset="(1,0,0)", size=(16, 16, 6)))

    # FC + Softmax
    arch.append(to_SoftMax("soft1", 1000, offset="(2,0,0)", to="(res_c_top-east)",
                           width=1.5, height=3, depth=25, caption="Softmax"))
    arch.append(to_connection("res_c_top", "soft1"))

    arch.append(to_end())
    return arch


def main():
    mcp.run()


if __name__ == "__main__":
    main()
