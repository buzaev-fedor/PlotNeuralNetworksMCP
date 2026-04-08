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

## Credits

Based on [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) by Haris Iqbal.
