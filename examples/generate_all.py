"""Generate diagrams for 24 popular neural network architectures."""

import os
import sys
import shutil
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from plot_nn_mcp.dsl import *

OUTPUT = os.path.join(os.path.dirname(__file__), "output", "architectures")

# Registry of (name, Architecture, show_n) in definition order — used by tests and CLI.
ARCHITECTURES: list[tuple[str, "Architecture", int]] = []
# Tests import this module to populate ARCHITECTURES without touching disk.
WRITE_OUTPUTS = os.environ.get("PLOTNN_SKIP_WRITE") != "1"
SHOW_N_DEFAULT = 3
_announced_no_tex = False


def save(arch: "Architecture", name: str, show_n: int = SHOW_N_DEFAULT):
    global _announced_no_tex
    ARCHITECTURES.append((name, arch, show_n))
    if not WRITE_OUTPUTS:
        return
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, f"{name}.tex")
    arch.render_to_file(path, show_n=show_n)
    engine = next((c for c in ("tectonic", "pdflatex") if shutil.which(c)), None)
    if engine is None:
        if not _announced_no_tex:
            print("  (no LaTeX engine found — skipping PDF compilation)")
            _announced_no_tex = True
    else:
        subprocess.run([engine, path], capture_output=True, timeout=120, cwd=OUTPUT)
    print(f"  {name}")


# ═══════════════════════════════════════════════════════════════════════
# 1. LSTM (Bidirectional NER)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"Bidirectional LSTM -- NER", theme="modern")
a.add(Embedding(128, label="Word Embedding"))
a.add(PositionalEncoding("learned", 128, label="Pos. Embedding"))
a.add(LSTMBlock(hidden_size=128, bidirectional=True))
a.add(Dropout(0.3))
a.add(LSTMBlock(hidden_size=128, bidirectional=True))
a.add(Dropout(0.3))
a.add(DenseLayer(256, label="Dense (256)"))
a.add(Activation("relu"))
a.add(ClassificationHead(label="CRF / Softmax"))
save(a, "01_lstm_ner")

# ═══════════════════════════════════════════════════════════════════════
# 2. GRU (Machine Translation)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"GRU Seq2Seq -- Translation", theme="modern")
a.add(SectionHeader("Encoder"))
a.add(Embedding(200, label="Source Embedding"))
a.add(GRUBlock(hidden_size=200))
a.add(GRUBlock(hidden_size=200))
a.add(SectionHeader("Decoder"))
a.add(CustomBlock("Context Vector", "residual"))
a.add(Embedding(200, label="Target Embedding"))
a.add(GRUBlock(hidden_size=200))
a.add(GRUBlock(hidden_size=200))
a.add(DenseLayer(200, label="Linear"))
a.add(ClassificationHead(label="Output Tokens"))
save(a, "02_gru_translation")

# ═══════════════════════════════════════════════════════════════════════
# 3. Seq2Seq + Bahdanau Attention
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"Seq2Seq + Bahdanau Attention", theme="modern")
a.add(SectionHeader("Encoder"))
a.add(Embedding(256, label="Source Embedding"))
a.add(LSTMBlock(hidden_size=256, bidirectional=True))
a.add(LSTMBlock(hidden_size=256, bidirectional=True))
a.add(SectionHeader("Attention"))
a.add(CustomBlock("Attention Weights (Bahdanau)", "norm"))
a.add(CustomBlock("Context Vector", "residual"))
a.add(SectionHeader("Decoder"))
a.add(Embedding(256, label="Target Embedding"))
a.add(LSTMBlock(hidden_size=256))
a.add(DenseLayer(256, label="Linear"))
a.add(ClassificationHead(label="Output Tokens"))
save(a, "03_seq2seq_attention")

# ═══════════════════════════════════════════════════════════════════════
# 4. BERT-base
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"BERT-base -- 12L -- 768d -- 12h", theme="modern")
a.add(Embedding(768, label="Token + Segment Emb"))
a.add(PositionalEncoding("learned", 768))
for _ in range(12):
    a.add(TransformerBlock("self", "post_ln", "gelu", 3072, 12, 768))
a.add(ClassificationHead(label="[CLS] Output"))
save(a, "04_bert_base")

# ═══════════════════════════════════════════════════════════════════════
# 5. RoBERTa-base
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"RoBERTa-base -- Dynamic Masking", theme="paper")
a.add(Embedding(768, label="BPE Embedding (50k)"))
a.add(PositionalEncoding("learned", 768))
for _ in range(12):
    a.add(TransformerBlock("self", "post_ln", "gelu", 3072, 12, 768))
a.add(ClassificationHead(label="[CLS] Output"))
save(a, "05_roberta_base")

# ═══════════════════════════════════════════════════════════════════════
# 6. DeBERTa-v3-base
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"DeBERTa-v3 -- Disentangled Attention", theme="vibrant")
a.add(Embedding(768, label="Token Embedding (128k)"))
a.add(PositionalEncoding("learned", 768, label="Relative Pos. Bias"))
for _ in range(12):
    a.add(TransformerBlock("self", "post_ln", "gelu", 3072, 12, 768,
                           label="Disentangled Attn"))
a.add(CustomBlock("Enhanced Mask Decoder", "output"))
a.add(ClassificationHead(label="Output"))
save(a, "06_deberta_v3")

# ═══════════════════════════════════════════════════════════════════════
# 7. ModernBERT-base
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"ModernBERT-base -- Pre-LN -- GeGLU -- RoPE", theme="modern")
a.add(Embedding(768))
a.add(PositionalEncoding("rope", 768))
for i in range(22):
    attn = "global" if (i + 1) % 3 == 0 else "local"
    a.add(TransformerBlock(attn, "pre_ln", "geglu", 2048, 12, 768))
a.add(ClassificationHead(label="[CLS] Output"))
save(a, "07_modernbert")

# ═══════════════════════════════════════════════════════════════════════
# 8. GPT-2 (117M)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"GPT-2 (117M) -- 12L -- 768d", theme="modern")
a.add(Embedding(768, label="Token Embedding (50k)"))
a.add(PositionalEncoding("learned", 768))
for _ in range(12):
    a.add(TransformerBlock("masked", "pre_ln", "gelu", 3072, 12, 768))
a.add(DenseLayer(768, label="LM Head"))
a.add(ClassificationHead(label="Next Token"))
save(a, "08_gpt2")

# ═══════════════════════════════════════════════════════════════════════
# 9. LLaMA-2-7B
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"LLaMA-2-7B -- 32L -- SwiGLU -- RoPE", theme="modern")
a.add(Embedding(4096, label="Token Embedding (32k)"))
a.add(PositionalEncoding("rope", 4096))
for _ in range(32):
    a.add(TransformerBlock("self", "pre_ln", "swiglu", 11008, 32, 4096))
a.add(RMSNorm())
a.add(DenseLayer(4096, label="LM Head"))
a.add(ClassificationHead(label="Next Token"))
save(a, "09_llama2_7b", show_n=2)

# ═══════════════════════════════════════════════════════════════════════
# 10. Mistral-7B
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"Mistral-7B -- GQA -- Sliding Window 4096", theme="vibrant")
a.add(Embedding(4096, label="Token Embedding (32k)"))
a.add(PositionalEncoding("rope", 4096))
for _ in range(32):
    a.add(TransformerBlock(attention="gqa", norm="pre_ln", ffn="swiglu",
                           d_ff=14336, heads=32, d_model=4096, kv_heads=8))
a.add(RMSNorm())
a.add(DenseLayer(4096, label="LM Head"))
a.add(ClassificationHead(label="Next Token"))
save(a, "10_mistral_7b", show_n=2)

# ═══════════════════════════════════════════════════════════════════════
# 11. Mixtral-8x7B
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"Mixtral-8x7B -- MoE -- GQA", theme="vibrant")
a.add(Embedding(4096, label="Token Embedding (32k)"))
a.add(PositionalEncoding("rope", 4096))
for _ in range(32):
    a.add(TransformerBlock(attention="gqa", norm="pre_ln", ffn="swiglu",
                           d_ff=14336, heads=32, d_model=4096, kv_heads=8,
                           skip_ffn=True))
    a.add(MoELayer(num_experts=8, top_k=2, d_ff=14336))
a.add(RMSNorm())
a.add(DenseLayer(4096, label="LM Head"))
a.add(ClassificationHead(label="Next Token"))
save(a, "11_mixtral_8x7b", show_n=2)

# ═══════════════════════════════════════════════════════════════════════
# 12. DCGAN
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"DCGAN", theme="modern")
a.add(SectionHeader("Generator"))
a.add(DenseLayer(100, label="Noise z (100d)"))
a.add(Generator(512, label="TransConv 4x4 (512) + BN + ReLU"))
a.add(Generator(256, label="TransConv 4x4 (256) + BN + ReLU"))
a.add(Generator(128, label="TransConv 4x4 (128) + BN + ReLU"))
a.add(Generator(3, label="TransConv 4x4 (3) + Tanh"))
a.add(CustomBlock("Generated Image 64x64", "embed"))
a.add(Separator("Discriminator"))
a.add(Discriminator(64, label="Conv 4x4 (64) + LeakyReLU"))
a.add(Discriminator(128, label="Conv 4x4 (128) + BN + LeakyReLU"))
a.add(Discriminator(256, label="Conv 4x4 (256) + BN + LeakyReLU"))
a.add(ClassificationHead(label="Real / Fake"))
save(a, "12_dcgan")

# ═══════════════════════════════════════════════════════════════════════
# 13. StyleGAN2
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"StyleGAN2", theme="paper")
a.add(SectionHeader("Mapping Network"))
a.add(DenseLayer(512, label="Latent z (512)"))
a.add(DenseLayer(512, label="FC + LeakyReLU x8"))
a.add(CustomBlock("Style w (512)", "norm"))
a.add(SectionHeader("Synthesis Network"))
a.add(CustomBlock("Const 4x4 + Style Inject", "embed"))
a.add(CustomBlock("Synthesis 8x8 + Noise", "attention"))
a.add(CustomBlock("Synthesis 16x16 + Noise", "attention"))
a.add(CustomBlock("Synthesis 32x32 + Noise", "attention"))
a.add(CustomBlock("Synthesis 64x64 + Noise", "attention"))
a.add(CustomBlock("... up to 1024x1024", "attention_alt"))
a.add(ClassificationHead(label="Output Image"))
save(a, "13_stylegan2")

# ═══════════════════════════════════════════════════════════════════════
# 14. DDPM (U-Net)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"DDPM -- U-Net -- 1000 steps", theme="modern")
a.add(CustomBlock("Noisy Image + t", "embed"))
a.add(CustomBlock("Sinusoidal Time Emb", "residual"))
a.add(SectionHeader(r"Encoder $\downarrow$"))
a.add(UNetBlock(64, label="ResBlock 32x32"))
a.add(UNetBlock(64, label="ResBlock 32x32"))
a.add(UNetBlock(128, with_attention=False, label="ResBlock 16x16"))
a.add(UNetBlock(128, with_attention=True, label="ResBlock 16x16 + Attn"))
a.add(SectionHeader("Bottleneck"))
a.add(UNetBlock(256, with_attention=True, label="Bottleneck 8x8 + Attn"))
a.add(SectionHeader(r"Decoder $\uparrow$ + Skip Connections"))
a.add(UNetBlock(128, with_attention=True, label="ResBlock 16x16 + Attn"))
a.add(UNetBlock(128, with_attention=False, label="ResBlock 16x16"))
a.add(UNetBlock(64, label="ResBlock 32x32"))
a.add(UNetBlock(64, label="ResBlock 32x32"))
a.add(NoiseHead())
save(a, "14_ddpm")

# ═══════════════════════════════════════════════════════════════════════
# 15. DiT (Diffusion Transformer)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"DiT-B/4 -- Diffusion Transformer", theme="vibrant")
a.add(CustomBlock("Noisy Image", "embed"))
a.add(ConvBlock(768, kernel_size=4, pool=None, label="Patchify 4x4"))
a.add(PositionalEncoding("sinusoidal", 768, label="Pos. Embedding"))
a.add(CustomBlock("Timestep + Class Emb", "residual"))
for _ in range(12):
    a.add(AdaptiveLayerNorm(condition="t + class"))
    a.add(TransformerBlock("self", "pre_ln", "gelu", 3072, 12, 768))
a.add(DenseLayer(768, label="Linear Decoder"))
a.add(CustomBlock("De-patchify", "attention_alt"))
a.add(NoiseHead())
save(a, "15_dit", show_n=2)

# ═══════════════════════════════════════════════════════════════════════
# 16. ResNet-50 (4 stages)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"ResNet-50", theme="modern")
a.add(ConvBlock(64, kernel_size=7, pool="max", label="Conv 7x7, stride 2"))
a.add(SectionHeader("Stage 2 (56x56)"))
for _ in range(3):
    a.add(BottleneckBlock(64, expansion=4))
a.add(SectionHeader("Stage 3 (28x28)"))
for _ in range(4):
    a.add(BottleneckBlock(128, expansion=4))
a.add(SectionHeader("Stage 4 (14x14)"))
for _ in range(6):
    a.add(BottleneckBlock(256, expansion=4))
a.add(SectionHeader("Stage 5 (7x7)"))
for _ in range(3):
    a.add(BottleneckBlock(512, expansion=4))
a.add(CustomBlock("Global AvgPool", "norm"))
a.add(ClassificationHead(label="FC (1000) + Softmax"))
save(a, "16_resnet50", show_n=2)

# ═══════════════════════════════════════════════════════════════════════
# 17. EfficientNet-B0
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"EfficientNet-B0 -- Compound Scaling", theme="paper")
a.add(ConvBlock(32, kernel_size=3, pool=None, label="Stem Conv 3x3"))
a.add(MBConvBlock(filters=16, expansion=1, kernel_size=3, se=True))
a.add(MBConvBlock(filters=24, expansion=6, kernel_size=3, se=True))
a.add(MBConvBlock(filters=40, expansion=6, kernel_size=5, se=True))
a.add(MBConvBlock(filters=80, expansion=6, kernel_size=3, se=True))
a.add(MBConvBlock(filters=112, expansion=6, kernel_size=5, se=True))
a.add(MBConvBlock(filters=192, expansion=6, kernel_size=5, se=True))
a.add(MBConvBlock(filters=320, expansion=6, kernel_size=3, se=True))
a.add(ConvBlock(1280, kernel_size=1, pool=None, label="Conv 1x1 (1280)"))
a.add(CustomBlock("Global AvgPool", "norm"))
a.add(Dropout(0.2))
a.add(ClassificationHead(label="FC (1000) + Softmax"))
save(a, "17_efficientnet_b0", show_n=2)

# ═══════════════════════════════════════════════════════════════════════
# 18. ViT-B/16
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"ViT-B/16 -- 12L -- 768d -- 12h", theme="modern")
a.add(PatchEmbedding(patch_size=16, d_model=768))
a.add(CustomBlock("[CLS] Token + Patch Tokens", "embed"))
a.add(PositionalEncoding("learned", 768))
for _ in range(12):
    a.add(TransformerBlock("self", "pre_ln", "gelu", 3072, 12, d_model=768))
a.add(DenseLayer(768, label="MLP Head"))
a.add(ClassificationHead(label="Classification"))
save(a, "18_vit_b16")

# ═══════════════════════════════════════════════════════════════════════
# 19. YOLOv8
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"YOLOv8 -- Detection", theme="vibrant")
a.add(SectionHeader("Backbone (CSPDarknet)"))
a.add(ConvBlock(64, kernel_size=3, pool=None, label="Stem Conv"))
a.add(CustomBlock("CSPDarknet Stage 1 (128)", "attention"))
a.add(CustomBlock("CSPDarknet Stage 2 (256)", "attention"))
a.add(CustomBlock("CSPDarknet Stage 3 (512)", "attention"))
a.add(CustomBlock("SPPF", "norm"))
a.add(SectionHeader("Neck (PANet)"))
a.add(CustomBlock("FPN + PAN (multi-scale)", "attention_alt"))
a.add(SectionHeader("Detection Head"))
a.add(CustomBlock("Detect P3 (stride 8) -- small", "ffn"))
a.add(CustomBlock("Detect P4 (stride 16) -- medium", "ffn"))
a.add(CustomBlock("Detect P5 (stride 32) -- large", "ffn"))
a.add(ClassificationHead(label="BBox + Class"))
save(a, "19_yolov8")

# ═══════════════════════════════════════════════════════════════════════
# 20. Standard PINN (Raissi)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"PINN (Raissi) -- Navier-Stokes", theme="modern")
a.add(SectionHeader("Forward Pass"))
a.add(CustomBlock("Input (x, y, t)", "embed"))
a.add(DenseLayer(200, label="Dense (200) + tanh"))
a.add(DenseLayer(200, label="Dense (200) + tanh"))
a.add(DenseLayer(200, label="Dense (200) + tanh"))
a.add(DenseLayer(200, label="Dense (200) + tanh"))
a.add(CustomBlock(r"Output: $u, v, p$", "output"))
a.add(SectionHeader("Loss"))
a.add(CustomBlock(r"$\mathcal{L}_{data}$ = MSE(pred, data)", "residual"))
a.add(CustomBlock(r"$\mathcal{L}_{PDE}$ = NS residual via autograd", "physics"))
a.add(CustomBlock(r"$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda\mathcal{L}_{PDE}$", "ffn"))
save(a, "20_pinn_raissi")

# ═══════════════════════════════════════════════════════════════════════
# 21. DeepONet
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"DeepONet -- Operator Learning", theme="paper")
a.add(SectionHeader("Branch Network"))
a.add(CustomBlock("Input u(x) at sensors", "embed"))
a.add(DenseLayer(128, label="FC (128) + tanh"))
a.add(DenseLayer(128, label="FC (128) + tanh"))
a.add(DenseLayer(128, label="Branch Output (p)"))
a.add(SectionHeader("Trunk Network"))
a.add(CustomBlock("Eval point y", "attention_alt"))
a.add(DenseLayer(128, label="FC (128) + tanh"))
a.add(DenseLayer(128, label="FC (128) + tanh"))
a.add(DenseLayer(128, label="Trunk Output (p)"))
a.add(SectionHeader("Combination"))
a.add(CustomBlock(r"$\sum b_k \cdot t_k + b_0$", "residual"))
a.add(ClassificationHead(label=r"$G(u)(y)$"))
save(a, "21_deeponet")

# ═══════════════════════════════════════════════════════════════════════
# 22. Physics-Constrained PINN
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"Hard-Constrained PINN", theme="vibrant")
a.add(CustomBlock("Input (x, t)", "embed"))
a.add(DenseLayer(100, label="Dense (100) + tanh"))
a.add(DenseLayer(100, label="Dense (100) + tanh"))
a.add(DenseLayer(100, label="Dense (100) + tanh"))
a.add(CustomBlock(r"Raw Output $\hat{u}$", "dense"))
a.add(Separator("Hard Constraint Projection"))
a.add(CustomBlock(r"$u = g(x) + D(x)\hat{u}$", "physics"))
a.add(CustomBlock(r"$\mathcal{L}_{PDE}$ only (no BC loss)", "ffn"))
a.add(ClassificationHead(label=r"$u(x,t)$ exact BCs"))
save(a, "22_pinn_hard")

# ═══════════════════════════════════════════════════════════════════════
# 23. FNO-2D
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"FNO-2D -- Fourier Neural Operator", theme="modern")
a.add(CustomBlock("Input a(x) on grid", "embed"))
a.add(CustomBlock("Lifting P (pointwise)", "dense"))
for _ in range(4):
    a.add(FourierBlock(modes=16, width=64))
a.add(CustomBlock("Projection Q (pointwise)", "dense"))
a.add(ClassificationHead(label=r"Output $u(x)$"))
save(a, "23_fno_2d")

# ═══════════════════════════════════════════════════════════════════════
# 24. U-NO (U-shaped Neural Operator)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"U-NO -- U-shaped Neural Operator", theme="paper")
a.add(CustomBlock("Input a(x) on grid", "embed"))
a.add(CustomBlock("Lifting P", "dense"))
a.add(SectionHeader(r"Encoder $\downarrow$"))
a.add(FourierBlock(modes=32, width=64, label="Fourier Enc 1"))
a.add(FourierBlock(modes=16, width=128, label="Fourier Enc 2"))
a.add(SectionHeader("Bottleneck"))
a.add(FourierBlock(modes=8, width=256, label="Bottleneck"))
a.add(SectionHeader(r"Decoder $\uparrow$ + Skip"))
a.add(FourierBlock(modes=16, width=128, label="Fourier Dec 2"))
a.add(FourierBlock(modes=32, width=64, label="Fourier Dec 1"))
a.add(CustomBlock("Projection Q", "dense"))
a.add(ClassificationHead(label=r"Output $u(x)$"))
save(a, "24_uno")


# ═══════════════════════════════════════════════════════════════════════
# 25. Swin Transformer (Swin-T)
# ═══════════════════════════════════════════════════════════════════════
a = Architecture(r"Swin-T -- Hierarchical ViT", theme="modern")
a.add(PatchEmbedding(patch_size=4, d_model=96, label="Patch Partition + Linear Embed"))
a.add(SectionHeader("Stage 1"))
a.add(SwinBlock(window_type="regular", heads=3, d_model=96))
a.add(SwinBlock(window_type="shifted", heads=3, d_model=96))
a.add(PatchMerging(d_model=192))
a.add(SectionHeader("Stage 2"))
a.add(SwinBlock(window_type="regular", heads=6, d_model=192))
a.add(SwinBlock(window_type="shifted", heads=6, d_model=192))
a.add(PatchMerging(d_model=384))
a.add(SectionHeader("Stage 3"))
for i in range(6):
    wt = "regular" if i % 2 == 0 else "shifted"
    a.add(SwinBlock(window_type=wt, heads=12, d_model=384))
a.add(PatchMerging(d_model=768))
a.add(SectionHeader("Stage 4"))
a.add(SwinBlock(window_type="regular", heads=24, d_model=768))
a.add(SwinBlock(window_type="shifted", heads=24, d_model=768))
a.add(CustomBlock("Global AvgPool", "norm"))
a.add(ClassificationHead(label="FC (1000)"))
save(a, "25_swin_t", show_n=2)


if WRITE_OUTPUTS:
    print(f"\nAll 25 architectures generated in {OUTPUT}/")
