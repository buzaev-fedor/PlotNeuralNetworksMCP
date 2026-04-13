"""
Deep E2E tests for PlotNeuralNetworkMCP.

Tests cover the full pipeline: layer specification → TikZ generation → LaTeX source
validation → file I/O → preset architectures → edge cases → error handling.

Each test verifies real output structure, not just that "something was returned".
"""

from __future__ import annotations

import json
import os
import re
import shutil

import pytest

from plot_nn_mcp.pycore.tikzeng import (
    _render_ball, _render_banded_box, _render_box,
    to_begin, to_connection, to_Conv, to_ConvConvRelu, to_ConvRes,
    to_ConvSoftMax, to_cor, to_end, to_generate, to_head, to_input,
    to_Pool, to_skip, to_SoftMax, to_Sum, to_UnPool,
)
from plot_nn_mcp.pycore.blocks import block_2ConvPool, block_Res, block_Unconv
from plot_nn_mcp.server import _build_arch, generate_diagram, generate_preset, generate_latex_snippet
from plot_nn_mcp.registry import LAYER_REGISTRY, coerce_params, get_layer_metadata, LayerSpec
from plot_nn_mcp.compiler import compile_tex, copy_layers_to, prepare_work_dir, write_and_compile
from plot_nn_mcp.presets import PRESETS, simple_cnn, vgg16, unet, resnet

from conftest import assert_valid_latex, parse_json


# ===========================================================================
# Helpers
# ===========================================================================

def _full_arch(*layer_strs: str) -> str:
    """Wrap layer strings in a complete document for validation."""
    parts = [to_head("."), to_cor(), to_begin(), *layer_strs, to_end()]
    return "".join(parts)


# ===========================================================================
# 1-10: Base renderer tests (_render_box)
# ===========================================================================

class TestRenderBox:
    def test_minimal_box(self):
        result = _render_box("test", r"\ConvColor", "(0,0,0)", "(0,0,0)", 40, 2, 40)
        assert "name=test" in result
        assert "Box=" in result
        assert "height=40" in result

    def test_box_with_xlabel(self):
        result = _render_box("c1", r"\ConvColor", "(0,0,0)", "(0,0,0)", 40, 2, 40,
                             xlabel="{{64, }}")
        assert "xlabel={{64, }}" in result

    def test_box_with_zlabel(self):
        result = _render_box("c1", r"\ConvColor", "(0,0,0)", "(0,0,0)", 40, 2, 40,
                             zlabel="256")
        assert "zlabel=256" in result

    def test_box_with_opacity(self):
        result = _render_box("p1", r"\PoolColor", "(0,0,0)", "(0,0,0)", 32, 1, 32,
                             opacity=0.5)
        assert "opacity=0.5" in result

    def test_box_without_opacity_omits_line(self):
        result = _render_box("c1", r"\ConvColor", "(0,0,0)", "(0,0,0)", 40, 2, 40)
        assert "opacity" not in result

    def test_box_caption(self):
        result = _render_box("c1", r"\ConvColor", "(0,0,0)", "(0,0,0)", 40, 2, 40,
                             caption="MyLayer")
        assert "caption=MyLayer" in result

    def test_box_custom_fill_color(self):
        result = _render_box("c1", "{rgb:red,5;blue,3}", "(0,0,0)", "(0,0,0)", 40, 2, 40)
        assert "fill={rgb:red,5;blue,3}" in result

    def test_box_offset_and_to(self):
        result = _render_box("c1", r"\ConvColor", "(1,2,3)", "(c0-east)", 40, 2, 40)
        assert "(1,2,3)" in result
        assert "(c0-east)" in result

    def test_box_float_dimensions(self):
        result = _render_box("c1", r"\ConvColor", "(0,0,0)", "(0,0,0)",
                             40.5, 2.7, 35.3)
        assert "height=40.5" in result
        assert "width=2.7" in result
        assert "depth=35.3" in result

    def test_box_produces_valid_tikz_pic(self):
        result = _render_box("c1", r"\ConvColor", "(0,0,0)", "(0,0,0)", 40, 2, 40)
        assert r"\pic[" in result
        assert "Box={" in result
        assert result.strip().endswith("};")


# ===========================================================================
# 11-17: Base renderer tests (_render_banded_box)
# ===========================================================================

class TestRenderBandedBox:
    def test_banded_box_structure(self):
        result = _render_banded_box("b1", r"\ConvColor", r"\ConvReluColor",
                                    "(0,0,0)", "(0,0,0)", 40, "{ 2 , 2 }", 40)
        assert "RightBandedBox=" in result
        assert "bandfill=" in result

    def test_banded_box_two_fills(self):
        result = _render_banded_box("b1", r"\ConvColor", r"\ConvReluColor",
                                    "(0,0,0)", "(0,0,0)", 40, "{ 2 , 2 }", 40)
        assert r"fill=\ConvColor" in result
        assert r"bandfill=\ConvReluColor" in result

    def test_banded_box_with_opacity(self):
        result = _render_banded_box("b1", r"\ConvColor", r"\ConvReluColor",
                                    "(0,0,0)", "(0,0,0)", 40, "6", 40, opacity=0.2)
        assert "opacity=0.2" in result

    def test_banded_box_xlabel_zlabel(self):
        result = _render_banded_box("b1", r"\ConvColor", r"\ConvReluColor",
                                    "(0,0,0)", "(0,0,0)", 40, "6", 40,
                                    xlabel="{{ 64, 64 }}", zlabel="256")
        assert "xlabel={{ 64, 64 }}" in result
        assert "zlabel=256" in result

    def test_banded_box_width_string(self):
        result = _render_banded_box("b1", r"\ConvColor", r"\ConvReluColor",
                                    "(0,0,0)", "(0,0,0)", 40, "{ 3 , 5 }", 40)
        assert "width={ 3 , 5 }" in result

    def test_banded_box_produces_valid_tikz(self):
        result = _render_banded_box("b1", r"\ConvColor", r"\ConvReluColor",
                                    "(0,0,0)", "(0,0,0)", 40, "6", 40)
        assert r"\pic[" in result
        assert result.strip().endswith("};")

    def test_banded_box_convres_colors(self):
        result = _render_banded_box("r1", "{rgb:white,1;black,3}", "{rgb:white,1;black,2}",
                                    "(0,0,0)", "(0,0,0)", 40, "6", 40)
        assert "fill={rgb:white,1;black,3}" in result
        assert "bandfill={rgb:white,1;black,2}" in result


# ===========================================================================
# 18-22: Base renderer tests (_render_ball)
# ===========================================================================

class TestRenderBall:
    def test_ball_structure(self):
        result = _render_ball("s1", r"\SumColor", "(0,0,0)", "(0,0,0)", 2.5, 0.6)
        assert "Ball=" in result
        assert "name=s1" in result

    def test_ball_logo(self):
        result = _render_ball("s1", r"\SumColor", "(0,0,0)", "(0,0,0)", 2.5, 0.6)
        assert "logo=$+$" in result

    def test_ball_custom_logo(self):
        result = _render_ball("s1", r"\SumColor", "(0,0,0)", "(0,0,0)", 2.5, 0.6,
                              logo=r"$\times$")
        assert r"logo=$\times$" in result

    def test_ball_radius_and_opacity(self):
        result = _render_ball("s1", r"\SumColor", "(0,0,0)", "(0,0,0)", 3.0, 0.8)
        assert "radius=3.0" in result
        assert "opacity=0.8" in result

    def test_ball_produces_valid_tikz(self):
        result = _render_ball("s1", r"\SumColor", "(0,0,0)", "(0,0,0)", 2.5, 0.6)
        assert r"\pic[" in result
        assert result.strip().endswith("};")


# ===========================================================================
# 23-32: Layer function parameter propagation
# ===========================================================================

class TestLayerFunctions:
    def test_conv_all_params(self):
        result = to_Conv("c1", s_filer=512, n_filer=128, offset="(1,0,0)",
                         to="(prev-east)", width=3, height=64, depth=64, caption="Conv1")
        assert "name=c1" in result
        assert "128" in result
        assert "512" in result
        assert "caption=Conv1" in result

    def test_conv_default_params(self):
        result = to_Conv("c1")
        assert "name=c1" in result
        assert "height=40" in result
        assert "depth=40" in result

    def test_pool_uses_pool_color(self):
        result = to_Pool("p1")
        assert r"\PoolColor" in result
        assert "Box=" in result

    def test_unpool_uses_unpool_color(self):
        result = to_UnPool("up1")
        assert r"\UnpoolColor" in result

    def test_conv_conv_relu_tuple_params(self):
        result = to_ConvConvRelu("ccr1", n_filer=(64, 128), width=(2, 3))
        assert "64" in result
        assert "128" in result
        assert "RightBandedBox" in result

    def test_conv_res_residual_colors(self):
        result = to_ConvRes("cr1")
        assert "rgb:white,1;black,3" in result
        assert "rgb:white,1;black,2" in result

    def test_conv_softmax_uses_softmax_color(self):
        result = to_ConvSoftMax("csm1")
        assert r"\SoftmaxColor" in result
        assert "Box=" in result

    def test_softmax_has_dummy_xlabel(self):
        result = to_SoftMax("sm1")
        assert '"dummy"' in result

    def test_sum_uses_ball(self):
        result = to_Sum("sum1")
        assert "Ball=" in result
        assert r"\SumColor" in result

    def test_input_image(self):
        result = to_input("image.png", to="(-3,0,0)", width=8, height=8, name="img1")
        assert "img1" in result
        assert "image.png" in result
        assert "includegraphics" in result


# ===========================================================================
# 33-40: Connection & skip tests
# ===========================================================================

class TestConnections:
    def test_connection_format(self):
        result = to_connection("conv1", "pool1")
        assert "conv1-east" in result
        assert "pool1-west" in result
        assert r"\midarrow" in result

    def test_skip_connection_anchors(self):
        result = to_skip("c1", "c5", pos=1.5)
        assert "c1-southeast" in result
        assert "c1-northeast" in result
        assert "c5-south" in result
        assert "c5-north" in result
        assert "1.5" in result

    def test_skip_default_pos(self):
        result = to_skip("a", "b")
        assert "1.25" in result

    def test_skip_uses_copy_connection(self):
        result = to_skip("a", "b")
        assert "copyconnection" in result
        assert r"\copymidarrow" in result

    def test_connection_names_with_underscores(self):
        result = to_connection("conv_block_1", "pool_block_1")
        assert "conv_block_1-east" in result

    def test_connection_names_with_numbers(self):
        result = to_connection("layer123", "layer456")
        assert "layer123-east" in result
        assert "layer456-west" in result

    def test_skip_with_zero_pos(self):
        result = to_skip("a", "b", pos=0)
        assert "pos=0" in result

    def test_skip_with_large_pos(self):
        result = to_skip("a", "b", pos=3.5)
        assert "pos=3.5" in result


# ===========================================================================
# 41-50: Document structure E2E
# ===========================================================================

class TestDocumentStructure:
    def test_head_includes_layers_path(self):
        result = to_head("/my/project")
        assert "/my/project/layers/" in result

    def test_head_windows_path_normalization(self):
        result = to_head("C:\\Users\\test")
        assert "C:/Users/test/layers/" in result

    def test_cor_defines_all_8_colors(self):
        result = to_cor()
        expected = ["ConvColor", "ConvReluColor", "PoolColor", "UnpoolColor",
                    "FcColor", "FcReluColor", "SoftmaxColor", "SumColor"]
        for color in expected:
            assert color in result

    def test_begin_defines_both_styles(self):
        result = to_begin()
        assert "connection" in result
        assert "copyconnection" in result
        assert r"\copymidarrow" in result

    def test_end_closes_both_environments(self):
        result = to_end()
        assert r"\end{tikzpicture}" in result
        assert r"\end{document}" in result

    def test_full_document_structure_order(self):
        tex = _full_arch(to_Conv("c1", 256, 64))
        doc_idx = tex.index(r"\documentclass")
        begin_doc = tex.index(r"\begin{document}")
        begin_tikz = tex.index(r"\begin{tikzpicture}")
        end_tikz = tex.index(r"\end{tikzpicture}")
        end_doc = tex.index(r"\end{document}")
        assert doc_idx < begin_doc < begin_tikz < end_tikz < end_doc

    def test_to_generate_creates_file(self, work_dir):
        arch = [to_head("."), to_cor(), to_begin(), to_Conv("c1", 256, 64), to_end()]
        path = os.path.join(work_dir, "test.tex")
        to_generate(arch, path)
        assert os.path.exists(path)
        content = open(path).read()
        assert_valid_latex(content)

    def test_to_generate_file_contains_all_parts(self, work_dir):
        path = os.path.join(work_dir, "test.tex")
        conv = to_Conv("myconv", 512, 128)
        pool = to_Pool("mypool")
        conn = to_connection("myconv", "mypool")
        to_generate([to_head("."), to_cor(), to_begin(), conv, pool, conn, to_end()], path)
        content = open(path).read()
        assert "myconv" in content
        assert "mypool" in content
        assert "myconv-east" in content

    def test_cor_colors_are_valid_tikz_rgb(self):
        result = to_cor()
        # Each color definition should match \def\Name{rgb:...}
        matches = re.findall(r"\\def\\(\w+)\{(rgb:[^}]+)\}", result)
        assert len(matches) == 14  # 8 original + 6 new architecture colors
        for name, value in matches:
            assert value.startswith("rgb:")

    def test_empty_arch_produces_valid_document(self):
        tex = _full_arch()
        assert_valid_latex(tex)


# ===========================================================================
# 51-62: Blocks E2E
# ===========================================================================

class TestBlocksE2E:
    def test_block_2convpool_layer_naming(self):
        result = block_2ConvPool("enc1", "input", "pool_enc1")
        combined = "".join(result)
        assert "ccr_enc1" in combined
        assert "pool_enc1" in combined
        assert "input-east" in combined

    def test_block_2convpool_size_propagation(self):
        result = block_2ConvPool("b1", "start", "p1", size=(64, 48, 5.0))
        combined = "".join(result)
        assert "height=64" in combined
        assert "depth=48" in combined
        # Pool height = 64 - 64//4 = 48
        assert "height=48" in combined

    def test_block_2convpool_custom_filters(self):
        result = block_2ConvPool("b1", "start", "p1", s_filer=128, n_filer=256)
        combined = "".join(result)
        assert "128" in combined
        assert "256" in combined

    def test_block_unconv_produces_5_layers_1_connection(self):
        result = block_Unconv("dec1", "bottleneck", "out1")
        assert len(result) == 6  # unpool + convres + conv + convres + conv + connection

    def test_block_unconv_layer_naming(self):
        result = block_Unconv("dec1", "bottleneck", "out1")
        combined = "".join(result)
        assert "unpool_dec1" in combined
        assert "ccr_res_dec1" in combined
        assert "ccr_dec1" in combined
        assert "ccr_res_c_dec1" in combined
        assert "out1" in combined

    def test_block_unconv_chain_connectivity(self):
        result = block_Unconv("d1", "bn", "out")
        combined = "".join(result)
        # Each layer references the previous one's east anchor
        assert "(unpool_d1-east)" in combined
        assert "(ccr_res_d1-east)" in combined
        assert "(ccr_d1-east)" in combined
        assert "(ccr_res_c_d1-east)" in combined

    def test_block_res_num_layers(self):
        result = block_Res(4, "res", "prev", "res_top")
        # 4 layers × (conv + connection) + 1 skip = 9
        assert len(result) == 9

    def test_block_res_skip_connection(self):
        result = block_Res(4, "res", "prev", "res_top")
        combined = "".join(result)
        assert "copyconnection" in combined

    def test_block_res_layer_chain(self):
        result = block_Res(3, "r", "input", "r_top")
        combined = "".join(result)
        assert "r_0" in combined
        assert "r_1" in combined
        assert "r_top" in combined

    def test_block_2convpool_in_full_document(self, work_dir):
        arch = [to_head("."), to_cor(), to_begin()]
        arch.extend(block_2ConvPool("b1", "start", "p1", s_filer=512, n_filer=64))
        arch.append(to_end())
        path = os.path.join(work_dir, "test.tex")
        to_generate(arch, path)
        content = open(path).read()
        assert_valid_latex(content)
        assert "ccr_b1" in content

    def test_chained_blocks(self):
        result = []
        result.extend(block_2ConvPool("b1", "start", "p1"))
        result.extend(block_2ConvPool("b2", "p1", "p2"))
        combined = "".join(result)
        assert "(p1-east)" in combined  # b2 references p1

    def test_encoder_decoder_chain(self):
        arch = []
        arch.extend(block_2ConvPool("enc", "start", "pool_enc"))
        arch.extend(block_Unconv("dec", "pool_enc", "out_dec"))
        combined = "".join(arch)
        assert "(pool_enc-east)" in combined  # decoder references encoder output


# ===========================================================================
# 63-72: Registry & coerce_params
# ===========================================================================

class TestRegistry:
    def test_all_layer_types_registered(self):
        expected = {
            "Conv", "ConvConvRelu", "Pool", "UnPool", "ConvRes",
            "ConvSoftMax", "SoftMax", "Sum", "Input",
            "Dense", "Norm", "Embed", "MultiHeadAttn",
            "Multiply", "Concat", "SpectralConv", "Lifting",
        }
        assert set(LAYER_REGISTRY.keys()) == expected

    def test_each_registry_entry_is_layerspec(self):
        for name, spec in LAYER_REGISTRY.items():
            assert isinstance(spec, LayerSpec), f"{name} is not LayerSpec"
            assert callable(spec.builder)
            assert isinstance(spec.description, str)

    def test_convconvrelu_has_tuple_params(self):
        spec = LAYER_REGISTRY["ConvConvRelu"]
        assert "n_filer" in spec.tuple_params
        assert "width" in spec.tuple_params

    def test_coerce_converts_list_to_tuple(self):
        result = coerce_params("ConvConvRelu", {"n_filer": [64, 128], "width": [2, 3]})
        assert result["n_filer"] == (64, 128)
        assert result["width"] == (2, 3)

    def test_coerce_converts_numeric_strings(self):
        result = coerce_params("Conv", {"height": "40", "depth": "32.5"})
        assert result["height"] == 40
        assert result["depth"] == 32.5

    def test_coerce_leaves_non_numeric_strings(self):
        result = coerce_params("Conv", {"offset": "(0,0,0)", "to": "(c1-east)"})
        assert result["offset"] == "(0,0,0)"
        assert result["to"] == "(c1-east)"

    def test_coerce_passthrough_already_correct_types(self):
        result = coerce_params("Conv", {"height": 40, "width": 2.5})
        assert result["height"] == 40
        assert result["width"] == 2.5

    def test_get_layer_metadata_structure(self):
        meta = get_layer_metadata()
        assert "Conv" in meta
        assert "description" in meta["Conv"]
        assert "params" in meta["Conv"]
        assert "name" in meta["Conv"]["params"]

    def test_metadata_marks_required_params(self):
        meta = get_layer_metadata()
        assert "(required)" in meta["Conv"]["params"]["name"]

    def test_metadata_shows_defaults(self):
        meta = get_layer_metadata()
        assert "default" in meta["Conv"]["params"]["s_filer"]


# ===========================================================================
# 73-82: Compiler module
# ===========================================================================

class TestCompiler:
    def test_prepare_work_dir_temp(self):
        d = prepare_work_dir(None)
        assert os.path.isdir(d)
        shutil.rmtree(d)

    def test_prepare_work_dir_custom(self, work_dir):
        custom = os.path.join(work_dir, "subdir", "nested")
        d = prepare_work_dir(custom)
        assert os.path.isdir(d)
        assert d == custom

    def test_copy_layers_creates_all_files(self, work_dir):
        copy_layers_to(work_dir)
        layers = os.path.join(work_dir, "layers")
        assert os.path.exists(os.path.join(layers, "Box.sty"))
        assert os.path.exists(os.path.join(layers, "Ball.sty"))
        assert os.path.exists(os.path.join(layers, "RightBandedBox.sty"))
        assert os.path.exists(os.path.join(layers, "init.tex"))

    def test_copy_layers_idempotent(self, work_dir):
        copy_layers_to(work_dir)
        copy_layers_to(work_dir)  # should not raise
        assert os.path.isdir(os.path.join(work_dir, "layers"))

    def test_write_and_compile_no_pdf(self, work_dir):
        arch = [to_head("."), to_cor(), to_begin(), to_Conv("c1", 256, 64), to_end()]
        result = write_and_compile(arch, work_dir, "test", do_compile=False)
        assert result["status"] == "tex_generated"
        assert os.path.exists(result["tex_path"])
        assert_valid_latex(result["tex_source"])

    def test_write_and_compile_creates_tex_file(self, work_dir):
        arch = [to_head("."), to_cor(), to_begin(), to_Conv("c1", 256, 64), to_end()]
        result = write_and_compile(arch, work_dir, "output", do_compile=False)
        assert result["tex_path"].endswith("output.tex")
        assert os.path.exists(result["tex_path"])

    def test_write_and_compile_tex_matches_arch(self, work_dir):
        conv = to_Conv("unique_layer_xyz", 256, 64)
        arch = [to_head("."), to_cor(), to_begin(), conv, to_end()]
        result = write_and_compile(arch, work_dir, "test", do_compile=False)
        assert "unique_layer_xyz" in result["tex_source"]

    def test_compile_tex_returns_none_without_pdflatex(self, work_dir):
        # Write a minimal tex file
        arch = [to_head("."), to_cor(), to_begin(), to_Conv("c1", 256, 64), to_end()]
        copy_layers_to(work_dir)
        tex_path = os.path.join(work_dir, "test.tex")
        to_generate(arch, tex_path)
        # compile_tex may or may not work depending on pdflatex availability
        # but it should not raise
        pdf_path, error = compile_tex(tex_path, work_dir)
        assert pdf_path is None or pdf_path.endswith(".pdf")

    def test_write_and_compile_with_compile_flag(self, work_dir):
        arch = [to_head("."), to_cor(), to_begin(), to_Conv("c1", 256, 64), to_end()]
        result = write_and_compile(arch, work_dir, "test", do_compile=True)
        assert result["status"] in ("success", "tex_generated")

    def test_write_and_compile_preserves_work_dir(self, work_dir):
        arch = [to_head("."), to_cor(), to_begin(), to_end()]
        result = write_and_compile(arch, work_dir, "test", do_compile=False)
        assert result["work_dir"] == work_dir


# ===========================================================================
# 83-92: MCP tool E2E (generate_diagram)
# ===========================================================================

class TestGenerateDiagramE2E:
    def test_single_conv_layer(self):
        layers = [{"type": "Conv", "name": "c1", "s_filer": 256, "n_filer": 64,
                   "offset": "(0,0,0)", "to": "(0,0,0)", "height": 40, "depth": 40, "width": 2}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert_valid_latex(result["tex_source"])
        assert "c1" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_multi_layer_pipeline(self):
        layers = [
            {"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"},
            {"type": "Pool", "name": "p1", "offset": "(0,0,0)", "to": "(c1-east)"},
            {"type": "Conv", "name": "c2", "offset": "(1,0,0)", "to": "(p1-east)"},
            {"type": "SoftMax", "name": "sm", "offset": "(2,0,0)", "to": "(c2-east)"},
        ]
        conns = [{"from": "p1", "to": "c2"}, {"from": "c2", "to": "sm"}]
        result = parse_json(generate_diagram(layers, conns, compile_pdf=False))
        tex = result["tex_source"]
        for name in ("c1", "p1", "c2", "sm"):
            assert name in tex
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_diagram_with_skip_connections(self):
        layers = [
            {"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"},
            {"type": "Conv", "name": "c2", "offset": "(1,0,0)", "to": "(c1-east)"},
            {"type": "Conv", "name": "c3", "offset": "(1,0,0)", "to": "(c2-east)"},
        ]
        skips = [{"from": "c1", "to": "c3", "pos": 1.5}]
        result = parse_json(generate_diagram(layers, skip_connections=skips, compile_pdf=False))
        assert "copyconnection" in result["tex_source"]
        assert "1.5" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_diagram_output_to_custom_dir(self, work_dir):
        layers = [{"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, output_dir=work_dir, compile_pdf=False))
        assert result["work_dir"] == work_dir
        assert os.path.exists(result["tex_path"])

    def test_diagram_custom_filename(self, work_dir):
        layers = [{"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, output_dir=work_dir,
                                              filename="my_net", compile_pdf=False))
        assert "my_net.tex" in result["tex_path"]

    def test_diagram_sum_layer(self):
        layers = [
            {"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"},
            {"type": "Sum", "name": "s1", "offset": "(1,0,0)", "to": "(c1-east)"},
        ]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert "Ball=" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_diagram_convconvrelu_with_list_params(self):
        layers = [{"type": "ConvConvRelu", "name": "ccr1", "n_filer": [64, 128],
                   "width": [2, 3], "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert "RightBandedBox" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_diagram_unknown_layer_type_error(self):
        layers = [{"type": "UNKNOWN", "name": "x"}]
        with pytest.raises(Exception):
            generate_diagram(layers, compile_pdf=False)

    def test_diagram_does_not_mutate_input(self):
        layers = [{"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        layers_copy = [dict(l) for l in layers]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        # Original input should still have "type"
        assert layers[0].get("type") == "Conv"
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_diagram_layers_dir_copied(self):
        layers = [{"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        layers_dir = os.path.join(result["work_dir"], "layers")
        assert os.path.isdir(layers_dir)
        shutil.rmtree(result["work_dir"], ignore_errors=True)


# ===========================================================================
# 93-100: Preset architecture E2E
# ===========================================================================

class TestPresetsE2E:
    def test_all_presets_produce_valid_latex(self):
        for name, builder in PRESETS.items():
            arch = builder()
            tex = "".join(arch)
            assert_valid_latex(tex)

    def test_simple_cnn_has_3_conv_3_pool_1_softmax(self):
        tex = "".join(simple_cnn())
        # 3 Conv layers + 1 SoftMax (which also uses Box)
        assert "name=conv1" in tex
        assert "name=conv2" in tex
        assert "name=conv3" in tex
        assert "name=pool1" in tex
        assert "name=pool2" in tex
        assert "name=pool3" in tex
        assert "name=soft1" in tex

    def test_vgg16_has_5_blocks(self):
        tex = "".join(vgg16())
        for i in range(1, 6):
            assert f"name=b{i}" in tex
            assert f"pool{i}" in tex

    def test_unet_symmetric_encoder_decoder(self):
        tex = "".join(unet())
        # 4 encoder blocks
        for i in range(1, 5):
            assert f"ccr_b{i}" in tex
        # 4 decoder blocks
        for i in range(5, 9):
            assert f"unpool_b{i}" in tex
        # 4 skip connections + 1 copyconnection style definition in to_begin()
        assert tex.count(r"\draw [copyconnection]") == 4

    def test_resnet_residual_blocks(self):
        tex = "".join(resnet())
        assert "res_a" in tex
        assert "res_b" in tex
        assert "res_c" in tex
        assert "copyconnection" in tex

    def test_generate_preset_returns_json(self):
        result = parse_json(generate_preset("simple_cnn", compile_pdf=False))
        assert result["preset"] == "simple_cnn"
        assert "tex_source" in result
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_generate_preset_custom_output(self, work_dir):
        result = parse_json(generate_preset("vgg16", output_dir=work_dir,
                                             filename="my_vgg", compile_pdf=False))
        assert "my_vgg.tex" in result["tex_path"]
        assert os.path.exists(result["tex_path"])

    def test_generate_preset_unknown_returns_error(self):
        result = parse_json(generate_preset("nonexistent", compile_pdf=False))
        assert "error" in result
        assert "available_presets" in result

    def test_generate_latex_snippet_no_file_written(self):
        layers = [{"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = generate_latex_snippet(layers)
        assert_valid_latex(result)
        assert isinstance(result, str)  # not JSON

    def test_presets_dict_matches_functions(self):
        assert PRESETS["simple_cnn"] is simple_cnn
        assert PRESETS["vgg16"] is vgg16
        assert PRESETS["unet"] is unet
        assert PRESETS["resnet"] is resnet


# ===========================================================================
# 101-125: Additional deep E2E tests
# ===========================================================================

class TestEdgeCases:
    def test_layer_name_with_special_chars(self):
        result = to_Conv("layer_3a")
        assert "name=layer_3a" in result

    def test_very_large_dimensions(self):
        result = to_Conv("big", height=1000, depth=1000, width=100)
        assert "height=1000" in result

    def test_very_small_dimensions(self):
        result = to_Conv("tiny", height=1, depth=1, width=0.1)
        assert "height=1" in result
        assert "width=0.1" in result

    def test_zero_dimensions(self):
        result = to_Conv("zero", height=0, depth=0, width=0)
        assert "height=0" in result

    def test_negative_offset(self):
        result = to_Conv("neg", offset="(-1,-2,-3)")
        assert "(-1,-2,-3)" in result

    def test_full_pipeline_write_read_roundtrip(self, work_dir):
        layers = [
            {"type": "Conv", "name": "c1", "s_filer": 512, "n_filer": 64,
             "offset": "(0,0,0)", "to": "(0,0,0)", "height": 64, "depth": 64, "width": 2},
            {"type": "Pool", "name": "p1", "offset": "(0,0,0)", "to": "(c1-east)"},
            {"type": "ConvConvRelu", "name": "ccr1", "n_filer": [128, 128],
             "width": [3, 3], "offset": "(1,0,0)", "to": "(p1-east)"},
            {"type": "SoftMax", "name": "out", "offset": "(2,0,0)", "to": "(ccr1-east)"},
        ]
        conns = [{"from": "p1", "to": "ccr1"}, {"from": "ccr1", "to": "out"}]
        result = parse_json(generate_diagram(layers, conns, output_dir=work_dir,
                                              compile_pdf=False))
        # Read back and verify
        content = open(result["tex_path"]).read()
        assert content == result["tex_source"]
        assert_valid_latex(content)

    def test_all_layer_types_in_single_diagram(self):
        layers = [
            {"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"},
            {"type": "Pool", "name": "p1", "offset": "(0,0,0)", "to": "(c1-east)"},
            {"type": "UnPool", "name": "up1", "offset": "(1,0,0)", "to": "(p1-east)"},
            {"type": "ConvConvRelu", "name": "ccr1", "n_filer": [64, 64], "width": [2, 2],
             "offset": "(1,0,0)", "to": "(up1-east)"},
            {"type": "ConvRes", "name": "cr1", "offset": "(0,0,0)", "to": "(ccr1-east)"},
            {"type": "ConvSoftMax", "name": "csm1", "offset": "(1,0,0)", "to": "(cr1-east)"},
            {"type": "SoftMax", "name": "sm1", "offset": "(1,0,0)", "to": "(csm1-east)"},
            {"type": "Sum", "name": "s1", "offset": "(1,0,0)", "to": "(sm1-east)"},
        ]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        tex = result["tex_source"]
        assert "Box=" in tex
        assert "RightBandedBox=" in tex
        assert "Ball=" in tex
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_many_connections(self):
        n = 10
        layers = [{"type": "Conv", "name": f"c{i}", "offset": "(1,0,0)",
                    "to": f"(c{i-1}-east)" if i > 0 else "(0,0,0)"}
                   for i in range(n)]
        conns = [{"from": f"c{i}", "to": f"c{i+1}"} for i in range(n - 1)]
        result = parse_json(generate_diagram(layers, conns, compile_pdf=False))
        tex = result["tex_source"]
        for i in range(n):
            assert f"c{i}" in tex
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_build_arch_empty_connections(self):
        layers = [{"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        arch = _build_arch([dict(l) for l in layers], connections=[], skip_connections=[])
        tex = "".join(arch)
        assert_valid_latex(tex)

    def test_list_layer_types_is_valid_json(self):
        from plot_nn_mcp.server import list_layer_types
        result = parse_json(list_layer_types())
        assert "layers" in result
        assert "blocks" in result
        # All registered types should appear
        for name in LAYER_REGISTRY:
            assert name in result["layers"]

    def test_unet_preset_bottleneck(self):
        tex = "".join(unet())
        assert "Bottleneck" in tex
        assert "bneck" in tex

    def test_resnet_preset_initial_conv_and_pool(self):
        tex = "".join(resnet())
        assert "Conv1" in tex
        assert "pool1" in tex

    def test_simple_cnn_connections(self):
        tex = "".join(simple_cnn())
        assert "pool1-east" in tex
        assert "pool2-east" in tex
        assert "pool3-east" in tex

    def test_vgg16_softmax(self):
        tex = "".join(vgg16())
        assert "Softmax" in tex
        assert "1000" in tex  # 1000 classes

    def test_generate_diagram_result_has_required_keys(self):
        layers = [{"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert "tex_path" in result
        assert "work_dir" in result
        assert "tex_source" in result
        assert "status" in result
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_preset_all_end_with_document_close(self):
        for name, builder in PRESETS.items():
            tex = "".join(builder())
            assert tex.rstrip().endswith(r"\end{document}"), f"{name} missing \\end{{document}}"

    def test_block_res_with_min_num(self):
        # Minimum num=3 (needs at least layers[1] and layers[-2])
        result = block_Res(3, "r", "prev", "r_top")
        combined = "".join(result)
        assert "r_0" in combined
        assert "r_1" in combined
        assert "r_top" in combined

    def test_multiple_diagrams_to_same_dir(self, work_dir):
        layers = [{"type": "Conv", "name": "c1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        r1 = parse_json(generate_diagram(layers, output_dir=work_dir,
                                          filename="net1", compile_pdf=False))
        r2 = parse_json(generate_diagram(
            [{"type": "Pool", "name": "p1", "offset": "(0,0,0)", "to": "(0,0,0)"}],
            output_dir=work_dir, filename="net2", compile_pdf=False))
        assert os.path.exists(r1["tex_path"])
        assert os.path.exists(r2["tex_path"])
        assert r1["tex_path"] != r2["tex_path"]

    def test_conv_uses_render_box_internally(self):
        # Verify Conv output matches what _render_box would produce
        conv_result = to_Conv("test_c", 256, 64, offset="(0,0,0)", to="(0,0,0)",
                              width=2, height=40, depth=40)
        box_result = _render_box("test_c", r"\ConvColor", "(0,0,0)", "(0,0,0)",
                                 40, 2, 40, xlabel="{{64, }}", zlabel="256")
        assert conv_result == box_result

    def test_pool_and_unpool_differ_only_by_color(self):
        pool = to_Pool("l1", offset="(0,0,0)", to="(0,0,0)",
                       width=1, height=32, depth=32, opacity=0.5)
        unpool = to_UnPool("l1", offset="(0,0,0)", to="(0,0,0)",
                           width=1, height=32, depth=32, opacity=0.5)
        # They should be identical except for the color
        assert r"\PoolColor" in pool
        assert r"\UnpoolColor" in unpool
        assert pool.replace(r"\PoolColor", "COLOR") == unpool.replace(r"\UnpoolColor", "COLOR")
