"""LaTeX compilation and file management."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .pycore.tikzeng import to_generate

LAYERS_DIR = str(Path(__file__).parent / "layers")


def prepare_work_dir(output_dir: str | None) -> str:
    """Return work_dir path. Creates output_dir if needed, or uses a temp dir."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return tempfile.mkdtemp(prefix="plotnn_")


def copy_layers_to(work_dir: str) -> None:
    """Copy LaTeX layer definitions into work directory for pdflatex."""
    layers_dest = os.path.join(work_dir, "layers")
    if not os.path.exists(layers_dest):
        shutil.copytree(LAYERS_DIR, layers_dest)


def compile_tex(tex_path: str, work_dir: str) -> tuple[str | None, str | None]:
    """Compile .tex to .pdf using pdflatex.

    Returns (pdf_path, error_message). pdf_path is None on failure.
    """
    if not shutil.which("pdflatex"):
        return None, "pdflatex not found on PATH"
    env = os.environ.copy()
    env["TEXINPUTS"] = LAYERS_DIR + ":" + work_dir + ":"
    try:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", work_dir,
             tex_path],
            capture_output=True, text=True, timeout=60, env=env,
        )
    except subprocess.TimeoutExpired:
        return None, "pdflatex timed out after 60 seconds"
    pdf_path = str(Path(tex_path).with_suffix(".pdf"))
    if os.path.exists(pdf_path):
        return pdf_path, None
    # Compilation failed — extract last 20 lines of log for diagnostics
    error_tail = (proc.stdout or "")[-1500:]
    return None, f"pdflatex failed (exit code {proc.returncode}):\n{error_tail}"


def write_and_compile(
    arch: list[str] | str,
    work_dir: str,
    filename: str,
    do_compile: bool,
) -> dict:
    """Write architecture to .tex and optionally compile to PDF.

    Returns a result dict with tex_path, tex_source, status, and optional pdf_path.
    """
    copy_layers_to(work_dir)
    tex_path = os.path.join(work_dir, f"{filename}.tex")
    if isinstance(arch, str):
        with open(tex_path, "w") as f:
            f.write(arch)
    else:
        to_generate(arch, tex_path)

    result = {
        "tex_path": tex_path,
        "work_dir": work_dir,
        "tex_source": Path(tex_path).read_text(),
    }

    if do_compile:
        pdf_path, compile_error = compile_tex(tex_path, work_dir)
        if pdf_path:
            result["pdf_path"] = pdf_path
            result["status"] = "success"
        else:
            result["status"] = "tex_generated"
            result["note"] = compile_error or "Compilation failed."
    else:
        result["status"] = "tex_generated"

    return result
