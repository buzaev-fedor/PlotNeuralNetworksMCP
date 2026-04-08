"""Tests for the PlotNeuralNetwork MCP server."""

import json
import os
import tempfile

import pytest

from plot_nn_mcp.pycore.tikzeng import (
    to_begin, to_Conv, to_cor, to_connection, to_end, to_generate,
    to_head, to_Pool, to_SoftMax, to_Sum, to_skip,
)
from plot_nn_mcp.pycore.blocks import block_2ConvPool, block_Unconv, block_Res
from plot_nn_mcp.server import (
    _build_arch, _coerce_params, _copy_layers_to,
    list_layer_types, generate_diagram, generate_preset,
    generate_latex_snippet,
)


class TestTikzeng:
    def test_to_head(self):
        result = to_head(".")
        assert r"\documentclass" in result
        assert r"\subimport" in result

    def test_to_cor(self):
        result = to_cor()
        assert r"\ConvColor" in result
        assert r"\PoolColor" in result

    def test_to_begin(self):
        result = to_begin()
        assert r"\begin{tikzpicture}" in result

    def test_to_end(self):
        result = to_end()
        assert r"\end{tikzpicture}" in result
        assert r"\end{document}" in result

    def test_to_conv(self):
        result = to_Conv("conv1", 256, 64, height=40, depth=40, width=2)
        assert "conv1" in result
        assert "Box" in result
        assert "64" in result

    def test_to_pool(self):
        result = to_Pool("pool1", height=32, depth=32)
        assert "pool1" in result
        assert "PoolColor" in result

    def test_to_softmax(self):
        result = to_SoftMax("soft1", 10)
        assert "soft1" in result
        assert "SoftmaxColor" in result

    def test_to_sum(self):
        result = to_Sum("sum1", radius=2.5)
        assert "sum1" in result
        assert "Ball" in result

    def test_to_connection(self):
        result = to_connection("conv1", "pool1")
        assert "conv1-east" in result
        assert "pool1-west" in result

    def test_to_skip(self):
        result = to_skip("conv1", "conv5", pos=1.25)
        assert "conv1" in result
        assert "conv5" in result
        assert "copyconnection" in result

    def test_to_generate(self):
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as f:
            path = f.name
        try:
            arch = [to_head("."), to_cor(), to_begin(),
                    to_Conv("c1", 256, 64), to_end()]
            to_generate(arch, path)
            content = open(path).read()
            assert r"\documentclass" in content
            assert "c1" in content
            assert r"\end{document}" in content
        finally:
            os.unlink(path)


class TestBlocks:
    def test_block_2ConvPool(self):
        result = block_2ConvPool("b1", "start", "pool_b1", s_filer=512, n_filer=64)
        assert isinstance(result, list)
        assert len(result) == 3  # ConvConvRelu + Pool + connection
        combined = "".join(result)
        assert "ccr_b1" in combined
        assert "pool_b1" in combined

    def test_block_Unconv(self):
        result = block_Unconv("b5", "bneck", "end_b5", s_filer=64, n_filer=512)
        assert isinstance(result, list)
        assert len(result) == 6
        combined = "".join(result)
        assert "unpool_b5" in combined

    def test_block_Res(self):
        result = block_Res(3, "res_a", "pool1", "res_a_top", s_filer=56, n_filer=64)
        assert isinstance(result, list)
        combined = "".join(result)
        assert "res_a_top" in combined
        assert "copyconnection" in combined


class TestServer:
    def test_coerce_params_conv_conv_relu(self):
        params = {"n_filer": [64, 64], "width": [2, 2], "height": 40}
        result = _coerce_params("ConvConvRelu", params)
        assert result["n_filer"] == (64, 64)
        assert result["width"] == (2, 2)

    def test_build_arch(self):
        layers = [
            {"type": "Conv", "name": "c1", "s_filer": 256, "n_filer": 64,
             "offset": "(0,0,0)", "to": "(0,0,0)", "height": 40, "depth": 40, "width": 2},
            {"type": "Pool", "name": "p1", "offset": "(0,0,0)", "to": "(c1-east)",
             "height": 32, "depth": 32},
        ]
        connections = [{"from": "c1", "to": "p1"}]
        arch = _build_arch(layers, connections)
        tex = "".join(arch)
        assert r"\documentclass" in tex
        assert "c1" in tex
        assert "p1" in tex
        assert r"\end{document}" in tex

    def test_build_arch_unknown_layer(self):
        with pytest.raises(ValueError, match="Unknown layer type"):
            _build_arch([{"type": "FakeLayer", "name": "x"}])

    def test_list_layer_types(self):
        result = json.loads(list_layer_types())
        assert "layers" in result
        assert "blocks" in result
        assert "Conv" in result["layers"]
        assert "Pool" in result["layers"]
        assert "block_2ConvPool" in result["blocks"]

    def test_generate_diagram(self):
        layers = [
            {"type": "Conv", "name": "c1", "s_filer": 256, "n_filer": 64,
             "offset": "(0,0,0)", "to": "(0,0,0)", "height": 40, "depth": 40, "width": 2},
        ]
        result = json.loads(generate_diagram(layers, compile_pdf=False))
        assert result["status"] == "tex_generated"
        assert "tex_path" in result
        assert "tex_source" in result
        assert r"\documentclass" in result["tex_source"]
        # cleanup
        if os.path.isdir(result["work_dir"]):
            import shutil
            shutil.rmtree(result["work_dir"])

    def test_generate_preset_simple_cnn(self):
        result = json.loads(generate_preset("simple_cnn", compile_pdf=False))
        assert "tex_source" in result
        assert "conv1" in result["tex_source"]
        assert result["preset"] == "simple_cnn"
        if os.path.isdir(result["work_dir"]):
            import shutil
            shutil.rmtree(result["work_dir"])

    def test_generate_preset_unknown(self):
        result = json.loads(generate_preset("unknown_arch", compile_pdf=False))
        assert "error" in result

    def test_generate_preset_all(self):
        for preset in ("simple_cnn", "vgg16", "unet", "resnet"):
            result = json.loads(generate_preset(preset, compile_pdf=False))
            assert "tex_source" in result, f"Failed for preset: {preset}"
            assert r"\documentclass" in result["tex_source"]
            if os.path.isdir(result["work_dir"]):
                import shutil
                shutil.rmtree(result["work_dir"])

    def test_generate_latex_snippet(self):
        layers = [
            {"type": "Conv", "name": "c1", "s_filer": 256, "n_filer": 64,
             "offset": "(0,0,0)", "to": "(0,0,0)", "height": 40, "depth": 40, "width": 2},
        ]
        result = generate_latex_snippet(layers)
        assert r"\documentclass" in result
        assert "c1" in result

    def test_generate_diagram_with_connections(self):
        layers = [
            {"type": "Conv", "name": "c1", "s_filer": 256, "n_filer": 64,
             "offset": "(0,0,0)", "to": "(0,0,0)", "height": 40, "depth": 40, "width": 2},
            {"type": "Conv", "name": "c2", "s_filer": 128, "n_filer": 128,
             "offset": "(1,0,0)", "to": "(c1-east)", "height": 32, "depth": 32, "width": 3},
        ]
        connections = [{"from": "c1", "to": "c2"}]
        skip_conns = [{"from": "c1", "to": "c2", "pos": 1.25}]
        result = json.loads(generate_diagram(layers, connections, skip_conns, compile_pdf=False))
        assert "c1-east" in result["tex_source"]
        assert "copyconnection" in result["tex_source"]
        if os.path.isdir(result["work_dir"]):
            import shutil
            shutil.rmtree(result["work_dir"])

    def test_copy_layers_to(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _copy_layers_to(tmpdir)
            layers_dir = os.path.join(tmpdir, "layers")
            assert os.path.isdir(layers_dir)
            assert os.path.exists(os.path.join(layers_dir, "Box.sty"))
            assert os.path.exists(os.path.join(layers_dir, "Ball.sty"))
            assert os.path.exists(os.path.join(layers_dir, "RightBandedBox.sty"))
            assert os.path.exists(os.path.join(layers_dir, "init.tex"))
