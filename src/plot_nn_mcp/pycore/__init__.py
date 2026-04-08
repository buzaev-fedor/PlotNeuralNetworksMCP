"""Public API for the pycore TikZ generation engine."""

from .tikzeng import (
    to_begin, to_branch, to_Concat, to_connection, to_Conv, to_ConvConvRelu,
    to_ConvRes, to_ConvSoftMax, to_cor, to_Dense, to_Embed, to_end,
    to_generate, to_head, to_input, to_Lifting, to_merge, to_MultiHeadAttn,
    to_Multiply, to_Norm, to_Pool, to_repeat_bracket, to_skip, to_skip_bottom,
    to_SoftMax, to_SpectralConv, to_Sum, to_UnPool,
)
from .blocks import block_2ConvPool, block_Res, block_Unconv

__all__ = [
    # Document structure
    "to_begin", "to_cor", "to_end", "to_generate", "to_head",
    # CNN layers (original)
    "to_Conv", "to_ConvConvRelu", "to_ConvRes", "to_ConvSoftMax",
    "to_input", "to_Pool", "to_SoftMax", "to_Sum", "to_UnPool",
    # New architecture layers
    "to_Concat", "to_Dense", "to_Embed", "to_Lifting", "to_MultiHeadAttn",
    "to_Multiply", "to_Norm", "to_SpectralConv",
    # Connections
    "to_branch", "to_connection", "to_merge", "to_repeat_bracket",
    "to_skip", "to_skip_bottom",
    # CNN blocks
    "block_2ConvPool", "block_Res", "block_Unconv",
]
