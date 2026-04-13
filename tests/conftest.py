"""Shared test fixtures and helpers."""

from __future__ import annotations

import json
import shutil
import tempfile

import pytest


@pytest.fixture
def work_dir():
    d = tempfile.mkdtemp(prefix="test_plotnn_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def assert_valid_latex(tex: str):
    """Basic structural checks on generated LaTeX."""
    assert r"\documentclass" in tex
    assert r"\begin{document}" in tex
    assert r"\end{document}" in tex


def parse_json(result_str: str) -> dict:
    data = json.loads(result_str)
    assert isinstance(data, dict)
    return data
