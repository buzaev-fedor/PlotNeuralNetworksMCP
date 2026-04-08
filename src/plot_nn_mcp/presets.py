"""Preset neural network architecture definitions."""

from __future__ import annotations

from .pycore.tikzeng import (
    to_begin, to_connection, to_Conv, to_ConvConvRelu, to_ConvSoftMax,
    to_cor, to_end, to_head, to_Pool, to_SoftMax, to_skip,
)
from .pycore.blocks import block_2ConvPool, block_Res, block_Unconv


def simple_cnn() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]
    arch.append(to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)",
                        height=64, depth=64, width=2, caption="Conv1"))
    arch.append(to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
                        height=48, depth=48, width=1, caption=" "))
    arch.append(to_Conv("conv2", 256, 128, offset="(1,0,0)", to="(pool1-east)",
                        height=48, depth=48, width=3, caption="Conv2"))
    arch.append(to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)",
                        height=32, depth=32, width=1, caption=" "))
    arch.append(to_Conv("conv3", 128, 256, offset="(1,0,0)", to="(pool2-east)",
                        height=32, depth=32, width=4, caption="Conv3"))
    arch.append(to_Pool("pool3", offset="(0,0,0)", to="(conv3-east)",
                        height=16, depth=16, width=1, caption=" "))
    arch.append(to_SoftMax("soft1", 10, offset="(2,0,0)", to="(pool3-east)",
                           caption="Softmax"))
    arch.append(to_connection("pool1", "conv2"))
    arch.append(to_connection("pool2", "conv3"))
    arch.append(to_connection("pool3", "soft1"))
    arch.append(to_end())
    return arch


def vgg16() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]
    blocks = [
        ("b1", 224, 64, "(0,0,0)", "(0,0,0)", 2, 64, "Block1", None),
        ("b2", 112, 128, "(1,0,0)", "(pool1-east)", 3, 48, "Block2", "pool1"),
        ("b3", 56, 256, "(1,0,0)", "(pool2-east)", 4, 36, "Block3", "pool2"),
        ("b4", 28, 512, "(1,0,0)", "(pool3-east)", 5, 24, "Block4", "pool3"),
        ("b5", 14, 512, "(1,0,0)", "(pool4-east)", 5, 16, "Block5", "pool4"),
    ]
    for i, (name, sf, nf, off, to, w, size, cap, conn_from) in enumerate(blocks, 1):
        arch.append(to_ConvConvRelu(name, sf, (nf, nf), offset=off, to=to,
                                    width=(w, w), height=size, depth=size, caption=cap))
        pool_h = size - int(size / 4) if i < 5 else size // 2
        arch.append(to_Pool(f"pool{i}", offset="(0,0,0)", to=f"({name}-east)",
                            height=pool_h, depth=pool_h))
        if conn_from:
            arch.append(to_connection(conn_from, name))

    arch.append(to_SoftMax("soft1", 1000, offset="(2,0,0)", to="(pool5-east)",
                           width=1.5, height=3, depth=25, caption="Softmax"))
    arch.append(to_connection("pool5", "soft1"))
    arch.append(to_end())
    return arch


def unet() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]

    encoder_cfg = [
        ("b1", "start", "pool_b1", 512, 64, "(0,0,0)", (64, 64, 3.5)),
        ("b2", "pool_b1", "pool_b2", 256, 128, "(1,0,0)", (48, 48, 4.5)),
        ("b3", "pool_b2", "pool_b3", 128, 256, "(1,0,0)", (32, 32, 6)),
        ("b4", "pool_b3", "pool_b4", 64, 512, "(1,0,0)", (16, 16, 8)),
    ]
    for name, bot, top, sf, nf, off, sz in encoder_cfg:
        arch.extend(block_2ConvPool(name, bot, top, s_filer=sf, n_filer=nf,
                                    offset=off, size=sz))

    arch.append(to_ConvConvRelu("bneck", 32, (1024, 1024), offset="(2,0,0)",
                                to="(pool_b4-east)", width=(10, 10), height=8,
                                depth=8, caption="Bottleneck"))
    arch.append(to_connection("pool_b4", "bneck"))

    decoder_cfg = [
        ("b5", "bneck", "end_b5", 64, 512, (16, 16, 8)),
        ("b6", "end_b5", "end_b6", 128, 256, (32, 32, 6)),
        ("b7", "end_b6", "end_b7", 256, 128, (48, 48, 4.5)),
        ("b8", "end_b7", "end_b8", 512, 64, (64, 64, 3.5)),
    ]
    for name, bot, top, sf, nf, sz in decoder_cfg:
        arch.extend(block_Unconv(name, bot, top, s_filer=sf, n_filer=nf,
                                 offset="(2,0,0)", size=sz))

    arch.append(to_ConvSoftMax("soft1", 512, offset="(1,0,0)", to="(end_b8-east)",
                               width=1, height=64, depth=64, caption="Output"))
    arch.append(to_connection("end_b8", "soft1"))

    skip_pairs = [("ccr_b4", "unpool_b5"), ("ccr_b3", "unpool_b6"),
                  ("ccr_b2", "unpool_b7"), ("ccr_b1", "unpool_b8")]
    for src, dst in skip_pairs:
        arch.append(to_skip(src, dst, pos=1.25))

    arch.append(to_end())
    return arch


def resnet() -> list[str]:
    arch = [to_head("."), to_cor(), to_begin()]

    arch.append(to_Conv("conv1", 224, 64, offset="(0,0,0)", to="(0,0,0)",
                        height=64, depth=64, width=3, caption="Conv1"))
    arch.append(to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)",
                        height=48, depth=48))

    res_blocks = [
        (3, "res_a", "pool1", "res_a_top", 56, 64, (48, 48, 3.5)),
        (4, "res_b", "res_a_top", "res_b_top", 28, 128, (32, 32, 4.5)),
        (3, "res_c", "res_b_top", "res_c_top", 14, 256, (16, 16, 6)),
    ]
    for num, name, bot, top, sf, nf, sz in res_blocks:
        arch.extend(block_Res(num, name, bot, top, s_filer=sf, n_filer=nf,
                              offset="(1,0,0)", size=sz))

    arch.append(to_SoftMax("soft1", 1000, offset="(2,0,0)", to="(res_c_top-east)",
                           width=1.5, height=3, depth=25, caption="Softmax"))
    arch.append(to_connection("res_c_top", "soft1"))
    arch.append(to_end())
    return arch


PRESETS: dict[str, callable] = {
    "simple_cnn": simple_cnn,
    "vgg16": vgg16,
    "unet": unet,
    "resnet": resnet,
}
