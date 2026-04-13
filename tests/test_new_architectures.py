"""
Tests for new architecture support: Transformer, BERT, GPT, ViT, PINN, FNO.
Covers new layer types, connection functions, composite blocks, and presets.
"""

from __future__ import annotations

import json
import os
import re
import shutil

import pytest

from plot_nn_mcp.pycore.tikzeng import (
    _render_partitioned_box, to_branch, to_Concat, to_connection,
    to_cor, to_Dense, to_Embed, to_head, to_begin, to_end,
    to_Lifting, to_merge, to_MultiHeadAttn, to_Multiply, to_Norm,
    to_repeat_bracket, to_skip_bottom, to_SpectralConv, to_Sum,
    to_generate,
)
from plot_nn_mcp.pycore.blocks_transformer import (
    block_EmbeddingStack, block_FourierLayer, block_MLPStack,
    block_TransformerDecoderLayer, block_TransformerEncoderLayer,
)
from plot_nn_mcp.presets import (
    PRESETS, bert, fno, gpt, pinn, transformer, vit,
)
from plot_nn_mcp.registry import LAYER_REGISTRY, coerce_params, get_layer_metadata
from plot_nn_mcp.server import generate_diagram, generate_preset
from plot_nn_mcp.compiler import write_and_compile

from conftest import assert_valid_latex, parse_json


# ===========================================================================
# New colors
# ===========================================================================

class TestNewColors:
    def test_cor_has_norm_color(self):
        assert "NormColor" in to_cor()

    def test_cor_has_attn_color(self):
        assert "AttnColor" in to_cor()

    def test_cor_has_embed_color(self):
        assert "EmbedColor" in to_cor()

    def test_cor_has_spectral_color(self):
        assert "SpectralColor" in to_cor()

    def test_cor_has_lift_color(self):
        assert "LiftColor" in to_cor()

    def test_cor_has_physics_color(self):
        assert "PhysicsColor" in to_cor()

    def test_all_new_colors_are_rgb(self):
        cor = to_cor()
        for name in ("NormColor", "AttnColor", "EmbedColor",
                     "SpectralColor", "LiftColor", "PhysicsColor"):
            pattern = rf"\\def\\{name}\{{(rgb:[^}}]+)\}}"
            match = re.search(pattern, cor)
            assert match, f"{name} not found or not rgb format"


# ===========================================================================
# New base renderer: _render_partitioned_box
# ===========================================================================

class TestPartitionedBox:
    def test_basic_structure(self):
        result = _render_partitioned_box("pb1", r"\AttnColor", "(0,0,0)", "(0,0,0)",
                                         40, 6, 40, num_parts=4)
        assert "PartitionedBox=" in result
        assert "nparts=4" in result
        assert "name=pb1" in result

    def test_custom_num_parts(self):
        result = _render_partitioned_box("pb1", r"\AttnColor", "(0,0,0)", "(0,0,0)",
                                         40, 6, 40, num_parts=8)
        assert "nparts=8" in result

    def test_zlabel(self):
        result = _render_partitioned_box("pb1", r"\AttnColor", "(0,0,0)", "(0,0,0)",
                                         40, 6, 40, zlabel="512")
        assert "zlabel=512" in result

    def test_opacity(self):
        result = _render_partitioned_box("pb1", r"\AttnColor", "(0,0,0)", "(0,0,0)",
                                         40, 6, 40, opacity=0.9)
        assert "opacity=0.9" in result

    def test_ends_with_semicolon(self):
        result = _render_partitioned_box("pb1", r"\AttnColor", "(0,0,0)", "(0,0,0)",
                                         40, 6, 40)
        assert result.strip().endswith("};")


# ===========================================================================
# New layer functions
# ===========================================================================

class TestNewLayers:
    def test_dense_uses_fc_color(self):
        result = to_Dense("d1", n_units=256)
        assert r"\FcColor" in result
        assert "Box=" in result
        assert "256" in result

    def test_dense_default_dims(self):
        result = to_Dense("d1")
        assert "height=3" in result
        assert "depth=25" in result

    def test_norm_thin_width(self):
        result = to_Norm("n1")
        assert r"\NormColor" in result
        assert "width=0.3" in result

    def test_norm_custom_height(self):
        result = to_Norm("n1", height=64, depth=64)
        assert "height=64" in result
        assert "depth=64" in result

    def test_embed_uses_embed_color(self):
        result = to_Embed("e1", d_model=768)
        assert r"\EmbedColor" in result
        assert "768" in result

    def test_multi_head_attn_uses_partitioned_box(self):
        result = to_MultiHeadAttn("mha1", num_heads=8, d_model=512)
        assert "PartitionedBox=" in result
        assert "nparts=8" in result
        assert "512" in result

    def test_multi_head_attn_caption(self):
        result = to_MultiHeadAttn("mha1", caption="Self-Attention")
        assert "caption=Self-Attention" in result

    def test_multiply_logo(self):
        result = to_Multiply("m1")
        assert r"$\times$" in result
        assert "Ball=" in result

    def test_concat_logo(self):
        result = to_Concat("c1")
        assert r"$\|$" in result
        assert "Ball=" in result

    def test_spectral_conv_uses_spectral_color(self):
        result = to_SpectralConv("sc1", modes=16)
        assert r"\SpectralColor" in result
        assert "16" in result

    def test_spectral_conv_fft_label(self):
        result = to_SpectralConv("sc1")
        assert "FFT" in result

    def test_lifting_uses_lift_color(self):
        result = to_Lifting("l1")
        assert r"\LiftColor" in result
        assert "Box=" in result


# ===========================================================================
# New connection functions
# ===========================================================================

class TestNewConnections:
    def test_skip_bottom_structure(self):
        result = to_skip_bottom("a", "b", pos=1.5)
        assert "copyconnection" in result
        assert "a-south" in result
        assert "b-south" in result

    def test_skip_bottom_default_pos(self):
        result = to_skip_bottom("a", "b")
        assert "1.25" in result

    def test_branch_one_to_many(self):
        result = to_branch("source", ["t1", "t2", "t3"])
        assert "source-east" in result
        assert "t1-west" in result
        assert "t2-west" in result
        assert "t3-west" in result
        assert result.count("connection") == 3

    def test_branch_spread(self):
        result = to_branch("s", ["t1", "t2"], spread=2.0)
        assert "2.00" in result or "-1.00" in result or "1.00" in result

    def test_merge_many_to_one(self):
        result = to_merge(["s1", "s2"], "target")
        assert "s1-east" in result
        assert "s2-east" in result
        assert "target-west" in result

    def test_repeat_bracket_label(self):
        result = to_repeat_bracket("start", "end", r"\times 6")
        assert "brace" in result
        assert r"\times 6" in result

    def test_repeat_bracket_xshift(self):
        result = to_repeat_bracket("a", "b", xshift=5.0)
        assert "5.0cm" in result

    def test_repeat_bracket_anchors(self):
        result = to_repeat_bracket("layer1", "layer5")
        assert "layer1-nearnortheast" in result
        assert "layer5-nearsoutheast" in result


# ===========================================================================
# Composite blocks (blocks_transformer.py)
# ===========================================================================

class TestTransformerBlocks:
    def test_encoder_layer_returns_list(self):
        result = block_TransformerEncoderLayer("enc0", "prev", "enc0_out")
        assert isinstance(result, list)
        assert len(result) > 0
        combined = "".join(result)
        assert "MHA" in combined

    def test_encoder_layer_contains_mha(self):
        combined = "".join(block_TransformerEncoderLayer("e0", "p", "e0_out"))
        assert "PartitionedBox" in combined

    def test_encoder_layer_contains_ffn(self):
        combined = "".join(block_TransformerEncoderLayer("e0", "p", "e0_out"))
        assert "FFN" in combined
        assert r"\FcColor" in combined

    def test_encoder_layer_contains_norm(self):
        combined = "".join(block_TransformerEncoderLayer("e0", "p", "e0_out"))
        assert "LN" in combined
        assert r"\NormColor" in combined

    def test_encoder_layer_contains_residual(self):
        combined = "".join(block_TransformerEncoderLayer("e0", "p", "e0_out"))
        assert "copyconnection" in combined  # skip connections for residuals

    def test_encoder_layer_custom_heads(self):
        combined = "".join(block_TransformerEncoderLayer(
            "e0", "p", "e0_out", num_heads=16))
        assert "nparts=16" in combined

    def test_decoder_layer_has_masked_and_cross_mha(self):
        combined = "".join(block_TransformerDecoderLayer(
            "d0", "p", "d0_out", "enc_out"))
        assert "Masked MHA" in combined
        assert "Cross-MHA" in combined

    def test_decoder_layer_three_add_norms(self):
        combined = "".join(block_TransformerDecoderLayer(
            "d0", "p", "d0_out", "enc_out"))
        # 3 Sum balls for residual add
        assert combined.count("Ball=") == 3

    def test_embedding_stack_basic(self):
        result = block_EmbeddingStack("emb", "start", "emb_out")
        combined = "".join(result)
        assert "Token Emb" in combined
        assert "Pos Emb" in combined
        assert r"\EmbedColor" in combined

    def test_embedding_stack_with_segment(self):
        result = block_EmbeddingStack("emb", "start", "emb_out", include_segment=True)
        combined = "".join(result)
        assert "Seg Emb" in combined

    def test_embedding_stack_without_segment(self):
        result = block_EmbeddingStack("emb", "start", "emb_out", include_segment=False)
        combined = "".join(result)
        assert "Seg Emb" not in combined

    def test_mlp_stack_produces_n_layers(self):
        result = block_MLPStack("mlp", "input", "output", n_layers=5)
        combined = "".join(result)
        # 5 Dense layers + 5 connections
        assert combined.count(r"\FcColor") == 5

    def test_mlp_stack_chain_connectivity(self):
        result = block_MLPStack("mlp", "input", "mlp_out", n_layers=3)
        combined = "".join(result)
        assert "mlp_0" in combined
        assert "mlp_1" in combined
        assert "mlp_out" in combined

    def test_fourier_layer_has_spectral_and_linear(self):
        result = block_FourierLayer("f0", "prev", "f0_out")
        combined = "".join(result)
        assert "Spectral Conv" in combined
        assert "Linear" in combined
        assert r"\SpectralColor" in combined

    def test_fourier_layer_has_sum(self):
        combined = "".join(block_FourierLayer("f0", "prev", "f0_out"))
        assert "Ball=" in combined


# ===========================================================================
# Preset: Transformer
# ===========================================================================

class TestTransformerPreset:
    def test_valid_latex(self):
        tex = "".join(transformer(n_enc=1, n_dec=1))
        assert_valid_latex(tex)

    def test_has_encoder_and_decoder(self):
        tex = "".join(transformer(n_enc=2, n_dec=2))
        assert "enc_0_attn" in tex
        assert "enc_1_attn" in tex
        assert "dec_0_mattn" in tex
        assert "dec_1_mattn" in tex

    def test_has_embeddings(self):
        tex = "".join(transformer(n_enc=1, n_dec=1))
        assert "Token Emb" in tex
        assert "Pos Emb" in tex

    def test_has_softmax_output(self):
        tex = "".join(transformer(n_enc=1, n_dec=1))
        assert "Softmax" in tex

    def test_custom_params(self):
        tex = "".join(transformer(n_enc=3, n_dec=2, d_model=256, n_heads=4))
        assert "nparts=4" in tex
        assert "256" in tex

    def test_repeat_bracket_enc(self):
        tex = "".join(transformer(n_enc=6, n_dec=1))
        assert r"\times 6" in tex


# ===========================================================================
# Preset: BERT
# ===========================================================================

class TestBertPreset:
    def test_valid_latex(self):
        tex = "".join(bert(n_layers=1))
        assert_valid_latex(tex)

    def test_has_segment_embedding(self):
        tex = "".join(bert(n_layers=1))
        assert "Seg Emb" in tex

    def test_has_cls_head(self):
        tex = "".join(bert(n_layers=1))
        assert "CLS Head" in tex

    def test_custom_layers(self):
        tex = "".join(bert(n_layers=6))
        assert "enc_5_attn" in tex

    def test_repeat_bracket(self):
        tex = "".join(bert(n_layers=12))
        assert r"\times 12" in tex


# ===========================================================================
# Preset: GPT
# ===========================================================================

class TestGptPreset:
    def test_valid_latex(self):
        tex = "".join(gpt(n_layers=1))
        assert_valid_latex(tex)

    def test_has_masked_mha(self):
        tex = "".join(gpt(n_layers=1))
        assert "Masked MHA" in tex

    def test_has_lm_head(self):
        tex = "".join(gpt(n_layers=1))
        assert "LM Head" in tex

    def test_no_cross_attention(self):
        tex = "".join(gpt(n_layers=1))
        assert "Cross-MHA" not in tex

    def test_repeat_bracket(self):
        tex = "".join(gpt(n_layers=6))
        assert r"\times 6" in tex


# ===========================================================================
# Preset: ViT
# ===========================================================================

class TestVitPreset:
    def test_valid_latex(self):
        tex = "".join(vit(n_layers=1))
        assert_valid_latex(tex)

    def test_has_patch_embedding(self):
        tex = "".join(vit(n_layers=1, patch_size=16))
        assert "Patch 16x16" in tex

    def test_has_cls_token(self):
        tex = "".join(vit(n_layers=1))
        assert "CLS Token" in tex

    def test_has_mlp_head(self):
        tex = "".join(vit(n_layers=1))
        assert "MLP Head" in tex

    def test_custom_params(self):
        tex = "".join(vit(n_layers=6, d_model=512, n_heads=8, patch_size=32))
        assert "nparts=8" in tex
        assert "Patch 32x32" in tex


# ===========================================================================
# Preset: PINN
# ===========================================================================

class TestPinnPreset:
    def test_valid_latex(self):
        tex = "".join(pinn())
        assert_valid_latex(tex)

    def test_has_input(self):
        tex = "".join(pinn())
        assert "Input (x,t)" in tex

    def test_has_two_branches(self):
        tex = "".join(pinn())
        assert "u(x,t)" in tex
        assert "PDE residual" in tex

    def test_has_loss_balls(self):
        tex = "".join(pinn())
        assert r"$\mathcal{L}_d$" in tex
        assert r"$\mathcal{L}_p$" in tex

    def test_has_total_loss(self):
        tex = "".join(pinn())
        assert "total_loss" in tex

    def test_custom_hidden_layers(self):
        tex = "".join(pinn(n_hidden=6, hidden_size=128))
        assert "hidden_5" in tex or "hidden_out" in tex


# ===========================================================================
# Preset: FNO
# ===========================================================================

class TestFnoPreset:
    def test_valid_latex(self):
        tex = "".join(fno())
        assert_valid_latex(tex)

    def test_has_lifting_and_projection(self):
        tex = "".join(fno())
        assert "Lifting P" in tex
        assert "Projection Q" in tex

    def test_has_fourier_layers(self):
        tex = "".join(fno(n_layers=4))
        assert "Spectral Conv" in tex

    def test_has_output(self):
        tex = "".join(fno())
        assert "Output" in tex

    def test_custom_modes(self):
        tex = "".join(fno(modes=32))
        assert "32" in tex

    def test_repeat_bracket(self):
        tex = "".join(fno(n_layers=4))
        assert r"\times 4" in tex


# ===========================================================================
# Preset registry + generate_preset integration
# ===========================================================================

class TestPresetIntegration:
    def test_all_new_presets_in_registry(self):
        for name in ("transformer", "bert", "gpt", "vit", "pinn", "fno"):
            assert name in PRESETS

    def test_generate_preset_transformer(self):
        result = parse_json(generate_preset("transformer",
                             params={"n_enc": 1, "n_dec": 1},
                             compile_pdf=False))
        assert "tex_source" in result
        assert_valid_latex(result["tex_source"])
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_generate_preset_bert_with_params(self):
        result = parse_json(generate_preset("bert",
                             params={"n_layers": 2, "d_model": 256},
                             compile_pdf=False))
        assert "256" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_generate_preset_gpt(self):
        result = parse_json(generate_preset("gpt",
                             params={"n_layers": 1},
                             compile_pdf=False))
        assert "Masked MHA" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_generate_preset_vit(self):
        result = parse_json(generate_preset("vit",
                             params={"n_layers": 1, "patch_size": 32},
                             compile_pdf=False))
        assert "Patch 32x32" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_generate_preset_pinn(self):
        result = parse_json(generate_preset("pinn", compile_pdf=False))
        assert r"$\mathcal{L}_d$" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_generate_preset_fno(self):
        result = parse_json(generate_preset("fno",
                             params={"n_layers": 2},
                             compile_pdf=False))
        assert "Spectral Conv" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_generate_preset_old_presets_still_work(self):
        for name in ("simple_cnn", "vgg16", "unet", "resnet"):
            result = parse_json(generate_preset(name, compile_pdf=False))
            assert_valid_latex(result["tex_source"])
            shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_all_new_presets_valid_latex(self):
        configs = {
            "transformer": {"n_enc": 1, "n_dec": 1},
            "bert": {"n_layers": 1},
            "gpt": {"n_layers": 1},
            "vit": {"n_layers": 1},
            "pinn": {},
            "fno": {"n_layers": 1},
        }
        for name, params in configs.items():
            result = parse_json(generate_preset(name, params=params, compile_pdf=False))
            assert_valid_latex(result["tex_source"])
            shutil.rmtree(result["work_dir"], ignore_errors=True)


# ===========================================================================
# New layer types via generate_diagram
# ===========================================================================

class TestNewLayersViaDiagram:
    def test_dense_layer(self):
        layers = [{"type": "Dense", "name": "d1", "n_units": 128,
                   "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert r"\FcColor" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_norm_layer(self):
        layers = [{"type": "Norm", "name": "n1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert r"\NormColor" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_embed_layer(self):
        layers = [{"type": "Embed", "name": "e1", "d_model": 768,
                   "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert r"\EmbedColor" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_multi_head_attn_layer(self):
        layers = [{"type": "MultiHeadAttn", "name": "mha1", "num_heads": 12,
                   "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert "PartitionedBox" in result["tex_source"]
        assert "nparts=12" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_multiply_layer(self):
        layers = [{"type": "Multiply", "name": "m1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert r"$\times$" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_spectral_conv_layer(self):
        layers = [{"type": "SpectralConv", "name": "sc1", "modes": 32,
                   "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert r"\SpectralColor" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)

    def test_lifting_layer(self):
        layers = [{"type": "Lifting", "name": "l1", "offset": "(0,0,0)", "to": "(0,0,0)"}]
        result = parse_json(generate_diagram(layers, compile_pdf=False))
        assert r"\LiftColor" in result["tex_source"]
        shutil.rmtree(result["work_dir"], ignore_errors=True)


# ===========================================================================
# Registry metadata for new types
# ===========================================================================

class TestNewRegistryMetadata:
    def test_metadata_includes_new_types(self):
        meta = get_layer_metadata()
        for name in ("Dense", "Norm", "Embed", "MultiHeadAttn",
                     "Multiply", "Concat", "SpectralConv", "Lifting"):
            assert name in meta, f"{name} not in metadata"

    def test_dense_metadata_has_n_units(self):
        meta = get_layer_metadata()
        assert "n_units" in meta["Dense"]["params"]

    def test_multi_head_attn_metadata_has_num_heads(self):
        meta = get_layer_metadata()
        assert "num_heads" in meta["MultiHeadAttn"]["params"]

    def test_spectral_conv_metadata_has_modes(self):
        meta = get_layer_metadata()
        assert "modes" in meta["SpectralConv"]["params"]

    def test_embed_metadata_has_d_model(self):
        meta = get_layer_metadata()
        assert "d_model" in meta["Embed"]["params"]
