# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tensor-parallel projections for the checkpoint's H3 video decoder."""

from __future__ import annotations

import torch
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


def _parallel_linear(source: nn.Linear, *, row: bool = False, gated: bool = False) -> nn.Module:
    kwargs = dict(
        input_size=source.in_features,
        bias=source.bias is not None,
        params_dtype=source.weight.dtype,
        return_bias=False,
    )
    with torch.device(source.weight.device):
        if row:
            target = RowParallelLinear(output_size=source.out_features, **kwargs)
        elif gated:
            target = MergedColumnParallelLinear(output_sizes=[source.out_features // 2] * 2, **kwargs)
        else:
            target = ColumnParallelLinear(output_size=source.out_features, **kwargs)
    for name in ("weight", "bias"):
        original = getattr(source, name)
        if original is not None:
            parameter = getattr(target, name)
            parameter.weight_loader(parameter, original.detach())
    return target


def shard_video_decoder(decoder: nn.Module, parallel_size: int) -> None:
    """Shard loaded CPU weights before their first device placement.

    QKV rows are interleaved per head in this checkpoint, so a contiguous
    column shard preserves whole heads. Gated FFN rows use separate gate/up
    halves and therefore require the merged-column loader.
    """
    if parallel_size != get_tensor_model_parallel_world_size():
        raise ValueError("H3 VAE tensor parallelism must use the full DiT tensor-parallel group")
    for block in decoder.transformer_blocks:
        attention = block.attn
        if attention.heads % parallel_size:
            raise ValueError(f"VAE attention heads ({attention.heads}) must divide by TP ({parallel_size})")
        attention.to_qkv = _parallel_linear(attention.to_qkv)
        attention.to_out = _parallel_linear(attention.to_out, row=True)
        attention.heads //= parallel_size
        attention.attn_inner_dim //= parallel_size
        block.ff.w1 = _parallel_linear(block.ff.w1, gated=block.ff.use_gated)
        block.ff.w2 = _parallel_linear(block.ff.w2, row=True)
