"""ModernBERT via the semantic DSL — 10 lines of architecture definition."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from plot_nn_mcp.dsl import Architecture, Embedding, TransformerBlock, ClassificationHead

# ── Define ModernBERT in 10 lines ──
arch = Architecture(
    r"ModernBERT-base \enspace$\cdot$ 22L $\cdot$ 768d $\cdot$ 12h $\cdot$ Pre-LN $\cdot$ GeGLU $\cdot$ RoPE",
    theme="modern",
)
arch.add(Embedding(d_model=768, rope=True))
for i in range(22):
    attn = "global" if (i + 1) % 3 == 0 else "local"
    arch.add(TransformerBlock(
        attention=attn, norm="pre_ln", ffn="geglu",
        d_ff=2048, heads=12, d_model=768,
    ))
arch.add(ClassificationHead(label="[CLS] → Output"))

# ── Render ──
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
tex_path = os.path.join(output_dir, "modernbert_dsl.tex")
arch.render_to_file(tex_path, show_n=3)
print(f"Generated: {tex_path}")

# ── Compile ──
import shutil
import subprocess
for compiler in ("tectonic", "pdflatex"):
    if shutil.which(compiler):
        subprocess.run([compiler, tex_path], capture_output=True, timeout=120,
                       cwd=output_dir)
        pdf = tex_path.replace(".tex", ".pdf")
        if os.path.exists(pdf):
            print(f"Compiled:  {pdf}")
            break
