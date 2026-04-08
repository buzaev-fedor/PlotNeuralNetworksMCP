"""Layer type registry with introspectable metadata.

Adding a new layer type requires only one change: register it here.
The MCP list_layer_types tool auto-derives parameter info from this registry.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

from .pycore.tikzeng import (
    to_Conv, to_ConvConvRelu, to_ConvRes, to_ConvSoftMax,
    to_input, to_Pool, to_SoftMax, to_Sum, to_UnPool,
)


@dataclass(frozen=True)
class LayerSpec:
    builder: Callable[..., str]
    description: str
    tuple_params: tuple[str, ...] = field(default_factory=tuple)


LAYER_REGISTRY: dict[str, LayerSpec] = {
    "Conv": LayerSpec(to_Conv, "Convolution layer (3D box)"),
    "ConvConvRelu": LayerSpec(
        to_ConvConvRelu, "Two convolutions with ReLU (banded box)",
        tuple_params=("n_filer", "width"),
    ),
    "Pool": LayerSpec(to_Pool, "Pooling layer"),
    "UnPool": LayerSpec(to_UnPool, "Unpooling / deconvolution layer"),
    "ConvRes": LayerSpec(to_ConvRes, "Residual convolution layer (banded, semi-transparent)"),
    "ConvSoftMax": LayerSpec(to_ConvSoftMax, "Convolution followed by softmax"),
    "SoftMax": LayerSpec(to_SoftMax, "Standalone softmax layer"),
    "Sum": LayerSpec(to_Sum, "Element-wise sum operation (ball with + symbol)"),
    "Input": LayerSpec(to_input, "Input image embedding"),
}


def coerce_params(layer_type: str, params: dict) -> dict:
    """Convert JSON-friendly params to the types expected by builder functions."""
    p = dict(params)
    spec = LAYER_REGISTRY.get(layer_type)
    if spec:
        for key in spec.tuple_params:
            if key in p and isinstance(p[key], list):
                p[key] = tuple(p[key])
    for key in ("height", "depth", "width", "s_filer", "n_filer", "radius", "opacity"):
        if key in p and isinstance(p[key], str):
            try:
                p[key] = float(p[key]) if "." in p[key] else int(p[key])
            except ValueError:
                pass
    return p


def get_layer_metadata() -> dict:
    """Auto-derive parameter metadata from function signatures."""
    result = {}
    for name, spec in LAYER_REGISTRY.items():
        sig = inspect.signature(spec.builder)
        params = {}
        for pname, param in sig.parameters.items():
            ann = param.annotation
            if isinstance(ann, type):
                type_name = ann.__name__
            elif isinstance(ann, str):
                type_name = ann
            elif ann is inspect.Parameter.empty:
                type_name = "str"
            else:
                type_name = str(ann)

            if param.default is inspect.Parameter.empty:
                params[pname] = f"{type_name} (required)"
            else:
                default = param.default
                params[pname] = f"{type_name} (default {default!r})"
        result[name] = {"description": spec.description, "params": params}
    return result
