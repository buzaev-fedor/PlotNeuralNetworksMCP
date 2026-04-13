"""Tests for DSL, themes, flat renderer, and generate_architecture MCP tool."""

from __future__ import annotations

import json
import os
import shutil

import pytest

from plot_nn_mcp.themes import get_theme, THEMES, theme_to_tikz_colors, Theme
from plot_nn_mcp.flat_renderer import (
    flat_head, flat_colors, flat_begin, flat_end,
    flat_block, flat_arrow, flat_skip_arrow, flat_add_circle,
    group_frame, flat_title, flat_side_label, flat_dim_label,
)
from plot_nn_mcp.dsl import (
    Architecture, Embedding, TransformerBlock, ConvBlock,
    DenseLayer, ClassificationHead, FourierBlock, CustomBlock,
    _detect_groups,
)
from plot_nn_mcp.server import generate_architecture, list_themes

from conftest import assert_valid_latex


# ===========================================================================
# Themes
# ===========================================================================

class TestThemes:
    def test_get_known_theme(self):
        theme = get_theme("modern")
        assert isinstance(theme, Theme)
        assert theme.name == "modern"

    def test_get_unknown_theme_raises(self):
        with pytest.raises(ValueError, match="Unknown theme"):
            get_theme("nonexistent")

    def test_all_themes_have_required_fields(self):
        for name, theme in THEMES.items():
            assert theme.attention, f"{name} missing attention"
            assert theme.ffn, f"{name} missing ffn"
            assert theme.norm, f"{name} missing norm"
            assert theme.embed, f"{name} missing embed"
            assert theme.output, f"{name} missing output"

    def test_theme_to_tikz_colors(self):
        theme = get_theme("modern")
        tikz = theme_to_tikz_colors(theme)
        assert r"\definecolor{clrattention}" in tikz
        assert r"\definecolor{clrffn}" in tikz
        assert r"\definecolor{clrnorm}" in tikz
        assert "{HTML}" in tikz

    def test_all_themes_exist(self):
        assert set(THEMES.keys()) == {"modern", "paper", "vibrant", "monochrome", "arxiv", "nature"}

    def test_colors_are_6_char_hex(self):
        for name, theme in THEMES.items():
            for role in ("attention", "ffn", "norm", "embed", "output"):
                color = getattr(theme, role)
                assert len(color) == 6, f"{name}.{role} = {color!r} is not 6 chars"
                int(color, 16)  # must be valid hex


# ===========================================================================
# Flat renderer primitives
# ===========================================================================

class TestFlatRenderer:
    def test_flat_head(self):
        result = flat_head()
        assert r"\documentclass" in result
        assert "tikz" in result
        assert "positioning" in result

    def test_flat_begin(self):
        result = flat_begin()
        assert r"\begin{tikzpicture}" in result
        assert "block/.style" in result

    def test_flat_end(self):
        result = flat_end()
        assert r"\end{tikzpicture}" in result
        assert r"\end{document}" in result

    def test_flat_block(self):
        result = flat_block("myblock", "Attention", "attention")
        assert "myblock" in result
        assert "Attention" in result
        assert r"\node[" in result

    def test_flat_block_above_of(self):
        result = flat_block("b2", "FFN", "ffn", above_of="b1", node_distance=0.5)
        assert "above=0.5cm of b1" in result

    def test_flat_arrow(self):
        result = flat_arrow("a", "b")
        assert r"\draw[arrow]" in result
        assert "a.north" in result
        assert "b.south" in result

    def test_flat_skip_arrow(self):
        result = flat_skip_arrow("a", "b")
        assert r"\draw[skiparrow]" in result
        assert "a.east" in result
        assert "b.east" in result

    def test_flat_add_circle(self):
        result = flat_add_circle("add1", above_of="attn")
        assert "circle" in result
        assert "add1" in result
        assert "+" in result

    def test_group_frame(self):
        result = group_frame("grp1", ["a", "b", "c"], title="Encoder", repeat=6)
        assert "fit" in result
        assert "(a)" in result
        assert "(b)" in result
        assert "Encoder" in result
        assert r"\times6" in result

    def test_group_frame_no_repeat(self):
        result = group_frame("grp1", ["a"], title="Block")
        assert r"\times" not in result

    def test_flat_title(self):
        result = flat_title("My Architecture")
        assert "My Architecture" in result

    def test_flat_dim_label(self):
        result = flat_dim_label("768", "layer1")
        assert "768" in result

    def test_flat_side_label(self):
        result = flat_side_label("12 heads", "attn1")
        assert "12 heads" in result


# ===========================================================================
# DSL layer dataclasses
# ===========================================================================

class TestDSLLayers:
    def test_embedding_defaults(self):
        e = Embedding()
        assert e.d_model == 512
        assert e.rope is False

    def test_transformer_block_defaults(self):
        t = TransformerBlock()
        assert t.attention == "self"
        assert t.norm == "pre_ln"
        assert t.ffn == "gelu"

    def test_conv_block(self):
        c = ConvBlock(filters=128, kernel_size=5)
        assert c.filters == 128
        assert c.kernel_size == 5

    def test_classification_head(self):
        h = ClassificationHead(n_classes=1000, label="Softmax")
        assert h.label == "Softmax"

    def test_custom_block(self):
        b = CustomBlock(text="My Op", color_role="spectral")
        assert b.text == "My Op"


# ===========================================================================
# Auto-grouping
# ===========================================================================

class TestAutoGrouping:
    def test_detect_consecutive_same_type(self):
        layers = [Embedding(), TransformerBlock(), TransformerBlock(), TransformerBlock(),
                  ClassificationHead()]
        groups = _detect_groups(layers)
        assert len(groups) == 1
        assert groups[0].start == 1
        assert groups[0].end == 4
        assert groups[0].count == 3

    def test_no_groups_all_different(self):
        layers = [Embedding(), TransformerBlock(), DenseLayer(), ClassificationHead()]
        groups = _detect_groups(layers)
        assert len(groups) == 0

    def test_multiple_groups(self):
        layers = [DenseLayer(), DenseLayer(), TransformerBlock(), TransformerBlock()]
        groups = _detect_groups(layers)
        assert len(groups) == 2

    def test_single_layer_not_grouped(self):
        layers = [Embedding(), TransformerBlock(), ClassificationHead()]
        groups = _detect_groups(layers)
        assert len(groups) == 0


# ===========================================================================
# Architecture.render()
# ===========================================================================

class TestArchitectureRender:
    def test_minimal_arch(self):
        arch = Architecture("Test", theme="modern")
        arch.add(Embedding(768))
        arch.add(ClassificationHead())
        tex = arch.render()
        assert_valid_latex(tex)
        assert "Embedding" in tex
        assert "Output" in tex

    def test_transformer_arch(self):
        arch = Architecture("BERT", theme="modern")
        arch.add(Embedding(768))
        for _ in range(3):
            arch.add(TransformerBlock(attention="self", ffn="gelu", heads=12))
        arch.add(ClassificationHead())
        tex = arch.render(show_n=2)
        assert_valid_latex(tex)
        assert "Self-Attention" in tex
        assert "FFN (GeLU)" in tex
        assert "LayerNorm" in tex
        assert r"\times3" in tex

    def test_cnn_arch(self):
        arch = Architecture("SimpleCNN", theme="paper")
        arch.add(ConvBlock(filters=32))
        arch.add(ConvBlock(filters=64))
        arch.add(ClassificationHead())
        tex = arch.render()
        assert_valid_latex(tex)
        assert "Conv3" in tex

    def test_modernbert_pattern(self):
        arch = Architecture("ModernBERT", theme="modern")
        arch.add(Embedding(768, rope=True))
        for i in range(6):
            attn = "global" if (i + 1) % 3 == 0 else "local"
            arch.add(TransformerBlock(attention=attn, ffn="geglu", heads=12))
        arch.add(ClassificationHead())
        tex = arch.render(show_n=3)
        assert_valid_latex(tex)
        assert "Local Attention" in tex
        assert "Global Attention" in tex
        assert "GeGLU" in tex
        assert "RoPE" in tex
        assert r"\times6" in tex

    def test_theme_applied(self):
        arch = Architecture("Test", theme="vibrant")
        arch.add(Embedding(512))
        arch.add(ClassificationHead())
        tex = arch.render()
        assert r"\definecolor{clrattention}{HTML}" in tex
        vibrant = get_theme("vibrant")
        assert vibrant.attention in tex

    def test_all_themes_render(self):
        for theme_name in THEMES:
            arch = Architecture("Test", theme=theme_name)
            arch.add(Embedding(512))
            arch.add(TransformerBlock())
            arch.add(ClassificationHead())
            tex = arch.render()
            assert_valid_latex(tex)

    def test_render_to_file(self, work_dir):
        arch = Architecture("Test", theme="modern")
        arch.add(Embedding(512))
        arch.add(ClassificationHead())
        path = os.path.join(work_dir, "test.tex")
        arch.render_to_file(path)
        assert os.path.exists(path)
        content = open(path).read()
        assert_valid_latex(content)

    def test_fourier_block(self):
        arch = Architecture("FNO", theme="modern")
        arch.add(Embedding(64))
        arch.add(FourierBlock(modes=16))
        arch.add(FourierBlock(modes=16))
        arch.add(ClassificationHead())
        tex = arch.render()
        assert_valid_latex(tex)
        assert "Spectral Conv" in tex

    def test_custom_block(self):
        arch = Architecture("Custom", theme="modern")
        arch.add(CustomBlock(text="My Custom Op", color_role="spectral"))
        tex = arch.render()
        assert "My Custom Op" in tex

    def test_dense_layers(self):
        arch = Architecture("MLP", theme="paper")
        arch.add(DenseLayer(256, label="Hidden 1"))
        arch.add(DenseLayer(128, label="Hidden 2"))
        arch.add(ClassificationHead())
        tex = arch.render()
        assert "Hidden 1" in tex
        assert "Hidden 2" in tex

    def test_show_n_controls_visible_blocks(self):
        arch = Architecture("Test", theme="modern")
        arch.add(Embedding(512))
        for _ in range(10):
            arch.add(TransformerBlock())
        arch.add(ClassificationHead())
        tex_show2 = arch.render(show_n=2)
        tex_show5 = arch.render(show_n=5)
        # show_n=2 should have fewer attention blocks rendered
        assert tex_show2.count("Self-Attention") == 2
        assert tex_show5.count("Self-Attention") == 5

    def test_pre_ln_structure(self):
        arch = Architecture("PreLN", theme="modern")
        arch.add(Embedding(512))
        arch.add(TransformerBlock(norm="pre_ln"))
        arch.add(ClassificationHead())
        tex = arch.render()
        # In Pre-LN, LayerNorm comes before Attention
        ln_pos = tex.index("LayerNorm")
        attn_pos = tex.index("Self-Attention")
        assert ln_pos < attn_pos

    def test_post_ln_structure(self):
        arch = Architecture("PostLN", theme="modern")
        arch.add(Embedding(512))
        arch.add(TransformerBlock(norm="post_ln"))
        arch.add(ClassificationHead())
        tex = arch.render()
        # In Post-LN, Attention comes before first LayerNorm
        attn_pos = tex.index("Self-Attention")
        # Find LayerNorm that comes after attention
        ln_after = tex.index("LayerNorm", attn_pos)
        assert ln_after > attn_pos


# ===========================================================================
# MCP tool: generate_architecture
# ===========================================================================

class TestGenerateArchitectureTool:
    def test_basic_call(self):
        layers = [
            {"layer": "Embedding", "d_model": 768},
            {"layer": "TransformerBlock", "attention": "self", "heads": 12},
            {"layer": "ClassificationHead", "label": "Output"},
        ]
        result = json.loads(generate_architecture("Test", layers, compile_pdf=False))
        assert "tex_source" in result
        assert_valid_latex(result["tex_source"])
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_with_theme(self):
        layers = [
            {"layer": "Embedding", "d_model": 512},
            {"layer": "ClassificationHead"},
        ]
        result = json.loads(generate_architecture("Test", layers, theme="vibrant",
                                                   compile_pdf=False))
        vibrant = get_theme("vibrant")
        assert vibrant.attention in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_unknown_layer_type(self):
        layers = [{"layer": "UnknownType"}]
        result = json.loads(generate_architecture("Test", layers, compile_pdf=False))
        assert "error" in result

    def test_output_dir(self, work_dir):
        layers = [
            {"layer": "Embedding", "d_model": 512},
            {"layer": "ClassificationHead"},
        ]
        result = json.loads(generate_architecture("Test", layers, output_dir=work_dir,
                                                   compile_pdf=False))
        assert os.path.exists(result["tex_path"])

    def test_custom_filename(self, work_dir):
        layers = [{"layer": "Embedding"}, {"layer": "ClassificationHead"}]
        result = json.loads(generate_architecture("T", layers, output_dir=work_dir,
                                                   filename="myarch", compile_pdf=False))
        assert "myarch.tex" in result["tex_path"]


# ===========================================================================
# MCP tool: list_themes
# ===========================================================================

class TestListThemesTool:
    def test_returns_valid_json(self):
        result = json.loads(list_themes())
        assert "modern" in result
        assert "paper" in result
        assert "vibrant" in result
        assert "monochrome" in result

    def test_theme_has_colors(self):
        result = json.loads(list_themes())
        for name, colors in result.items():
            assert "attention" in colors
            assert "ffn" in colors
            assert colors["attention"].startswith("#")


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_architecture(self):
        arch = Architecture("Empty", theme="modern")
        tex = arch.render()
        assert_valid_latex(tex)

    def test_single_embedding(self):
        arch = Architecture("Single", theme="modern")
        arch.add(Embedding(256))
        tex = arch.render()
        assert_valid_latex(tex)
        assert "Embedding" in tex

    def test_many_layers_performance(self):
        arch = Architecture("Deep", theme="modern")
        arch.add(Embedding(512))
        for _ in range(100):
            arch.add(TransformerBlock())
        arch.add(ClassificationHead())
        # Should render without error, showing only show_n blocks
        tex = arch.render(show_n=3)
        assert_valid_latex(tex)
        assert r"\times100" in tex
        assert tex.count("Self-Attention") == 3

    def test_mixed_architecture(self):
        arch = Architecture("Mixed", theme="modern")
        arch.add(ConvBlock(64))
        arch.add(ConvBlock(128))
        arch.add(DenseLayer(256))
        arch.add(TransformerBlock(attention="self"))
        arch.add(ClassificationHead())
        tex = arch.render()
        assert_valid_latex(tex)
