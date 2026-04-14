# PlotNeuralNetworksMCP

MCP server for generating publication-quality neural network architecture diagrams using [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) engine.

## Features

- Generate neural network diagrams via MCP tools
- Preset architectures: Simple CNN, VGG-16, U-Net, ResNet
- Custom architectures from layer specifications
- LaTeX/TikZ source generation
- PDF compilation (requires `pdflatex`)

## Installation

```bash
conda activate plot_NN
pip install -e .
```

### LaTeX dependency (for PDF output)

```bash
# macOS
brew install --cask mactex-no-gui

# Ubuntu/Debian
sudo apt-get install texlive-latex-extra texlive-fonts-recommended
```

## Usage

### As MCP server

Add to your Claude Desktop or Claude Code config:

```json
{
  "mcpServers": {
    "plot-neural-network": {
      "command": "plot-nn-mcp"
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `list_layer_types` | List all available layer types and parameters |
| `generate_diagram` | Generate diagram from layer specifications |
| `generate_preset` | Generate from preset architecture (simple_cnn, vgg16, unet, resnet) |
| `generate_latex_snippet` | Get LaTeX source without writing files |
| `compile_tex_to_pdf` | Compile existing .tex to PDF |

### Example: Generate a simple CNN

```json
{
  "layers": [
    {"type": "Conv", "name": "conv1", "s_filer": 512, "n_filer": 64, "offset": "(0,0,0)", "to": "(0,0,0)", "height": 64, "depth": 64, "width": 2},
    {"type": "Pool", "name": "pool1", "offset": "(0,0,0)", "to": "(conv1-east)", "height": 48, "depth": 48},
    {"type": "Conv", "name": "conv2", "s_filer": 256, "n_filer": 128, "offset": "(1,0,0)", "to": "(pool1-east)", "height": 48, "depth": 48, "width": 3},
    {"type": "SoftMax", "name": "soft1", "s_filer": 10, "offset": "(2,0,0)", "to": "(conv2-east)"}
  ],
  "connections": [
    {"from": "pool1", "to": "conv2"},
    {"from": "conv2", "to": "soft1"}
  ]
}
```

## Architecture

The render pipeline is split into three layers, each independently testable:

```text
DSL (dataclass)  →  IR (DAG)  →  emit_tikz (layout + render)
```

### DSL — [src/plot_nn_mcp/dsl.py](src/plot_nn_mcp/dsl.py)

Declarative layer types (`Embedding`, `TransformerBlock`, `ConvBlock`, ...) and the
top-level `Architecture` container.

```python
arch = Architecture("BERT", theme="modern")
arch.add(Embedding(d_model=768, label="Token Emb"))
arch.add(TransformerBlock(attention="self", norm="post_ln"))
arch.add(ClassificationHead(label="[CLS]"))
tex = arch.render(show_n=3)  # use_ir=True is default
```

### IR — [src/plot_nn_mcp/ir.py](src/plot_nn_mcp/ir.py)

Typed DAG with composable operators:

- `IRBlockOp(role, label, size_hint)` — leaf block.
- `IRSequenceOp(ops)` — chain of `data` edges.
- `IRResidualOp(body)` — `entry → body → add ← skip(entry)`. Reused for
  Transformer attention, FFN, MoE, ResNet bottleneck — structurally
  impossible to forget the residual.
- `IRParallelOp(branches, merge)` — split / parallel / optional merge.
- `IRCustomOp` — escape hatch for pre-built sub-graphs.

### Render — [src/plot_nn_mcp/render.py](src/plot_nn_mcp/render.py)

`emit_tikz(graph, theme) → str`. Applies width normalization,
rounds cm to 2 decimals (no more `1.8666666cm`), computes safe
skip-arrow `xshift` from block bounding boxes, and draws group
frames with `×N` badges.

### Adding a new block type

```python
from dataclasses import dataclass
from plot_nn_mcp.lowering import register
from plot_nn_mcp.ir import IRBlockOp, IRResidualOp
from plot_nn_mcp.themes import Role

@dataclass
class MyBlock:
    units: int = 64
    label: str = "MyBlock"

def my_block_to_ir(block: MyBlock):
    return IRResidualOp(body=[
        IRBlockOp(role=Role.NORM, label="LayerNorm"),
        IRBlockOp(role=Role.ATTENTION, label=block.label, dim=block.units),
    ])

register(MyBlock, my_block_to_ir)
```

The `Role` enum drives color selection via the active theme — use the
semantic role (`ATTENTION`, `FFN`, `NORM`, ...) rather than a raw color name.

### Legacy path

Before the refactor, rendering went through a 1500+ line monolithic
function in `dsl.py` with 50+ `isinstance` branches. That path is still
accessible via `arch.render(use_ir=False)` but emits a `DeprecationWarning`
and lacks several structural fixes:

- Mixtral MoE now has the add2 residual (was missing).
- DeBERTa no longer emits two `{Output}` labels.
- ResNet/YOLOv8 widths are rounded (no more `1.8666...cm`).
- Input arrow attaches to the first semantic node, never to a
  SectionHeader rule.

### Testing

600+ tests enforce the architecture. Key invariants:

- Every baseline `.tex` has zero fractional cm and ≤1 Input/Output label.
- Every `IRResidualOp` lowers to exactly one skip edge.
- Every layer type is either registered in the dispatch table or
  explicitly listed in `LEGACY_ONLY` (currently empty — all 47 types
  migrated).

Regenerate snapshots after intentional output changes:

```bash
UPDATE_GOLDEN=1 pytest tests/test_golden.py::test_snapshot_matches_golden
```

See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) and [REFACTOR_HANDOFF.md](REFACTOR_HANDOFF.md)
for the full rewrite history and remaining tasks.

## Credits

Based on [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) by Haris Iqbal.
