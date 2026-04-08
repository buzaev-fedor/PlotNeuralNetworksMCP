"""LaTeX compilation and file management."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

LAYERS_DIR = str(Path(__file__).parent / "layers")


def prepare_work_dir(output_dir: str | None) -> tuple[str, bool]:
    """Return (work_dir, is_temp). Creates output_dir if needed."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir, False
    return tempfile.mkdtemp(prefix="plotnn_"), True


def copy_layers_to(work_dir: str) -> None:
    """Copy LaTeX layer definitions into work directory for pdflatex."""
    layers_dest = os.path.join(work_dir, "layers")
    if not os.path.exists(layers_dest):
        shutil.copytree(LAYERS_DIR, layers_dest)


def compile_tex(tex_path: str, work_dir: str) -> str | None:
    """Compile .tex to .pdf using pdflatex. Returns pdf path or None."""
    if not shutil.which("pdflatex"):
        return None
    env = os.environ.copy()
    env["TEXINPUTS"] = LAYERS_DIR + ":" + work_dir + ":"
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", work_dir,
             tex_path],
            capture_output=True, text=True, timeout=60, env=env,
        )
    except subprocess.TimeoutExpired:
        return None
    pdf_path = tex_path.replace(".tex", ".pdf")
    if os.path.exists(pdf_path):
        return pdf_path
    return None


def write_and_compile(
    arch: list[str],
    work_dir: str,
    filename: str,
    do_compile: bool,
) -> dict:
    """Write architecture to .tex and optionally compile to PDF.

    Returns a result dict with tex_path, tex_source, status, and optional pdf_path.
    """
    from .pycore.tikzeng import to_generate

    copy_layers_to(work_dir)
    tex_path = os.path.join(work_dir, f"{filename}.tex")
    to_generate(arch, tex_path)

    result = {
        "tex_path": tex_path,
        "work_dir": work_dir,
        "tex_source": Path(tex_path).read_text(),
    }

    if do_compile:
        pdf_path = compile_tex(tex_path, work_dir)
        if pdf_path:
            result["pdf_path"] = pdf_path
            result["status"] = "success"
        else:
            result["status"] = "tex_generated"
            result["note"] = "pdflatex not found or compilation failed. LaTeX source is available."
    else:
        result["status"] = "tex_generated"

    return result
