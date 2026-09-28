# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _decode_rms_norm(
    X,
    W,
    Y,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    EPS: tl.constexpr,
    REDUCE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = row // GROUP * STRIDE_0 + row % GROUP * STRIDE_1
    lane = tl.arange(0, REDUCE_WIDTH)
    a0 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    a1 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    a2 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    a3 = tl.full((REDUCE_WIDTH,), 0, tl.float32)
    for step in range(triton.cdiv(N, 4 * REDUCE_WIDTH)):
        j = 4 * (lane + step * REDUCE_WIDTH)
        x0 = tl.load(X + base + j, j < N, other=0).to(tl.float32)
        x1 = tl.load(X + base + j + 1, j + 1 < N, other=0).to(tl.float32)
        x2 = tl.load(X + base + j + 2, j + 2 < N, other=0).to(tl.float32)
        x3 = tl.load(X + base + j + 3, j + 3 < N, other=0).to(tl.float32)
        a0 = a0 + x0 * x0
        a1 = a1 + x1 * x1
        a2 = a2 + x2 * x2
        a3 = a3 + x3 * x3
    # Match CUDA mean's vector accumulation and cross-warp-first reduction.
    partial = (((a0 + a1) + a2) + a3).reshape((REDUCE_WIDTH // 32, 32))
    total = tl.sum(tl.sum(partial, axis=0), axis=0)
    inverse = tl.rsqrt(total * (1.0 / N) + EPS)
    j = tl.arange(0, BLOCK)
    x = tl.load(X + base + j, j < N, other=0).to(tl.float32)
    weight = tl.load(W + j, j < N, other=0).to(tl.float32)
    tl.store(Y + row * N + j, (x * inverse) * weight, j < N)


def decode_rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor | None:
    if (
        not hidden.is_cuda
        or torch.version.hip is not None
        or hidden.ndim not in (2, 3)
        or hidden.shape[-1] not in (128, 3072)
        or hidden.stride(-1) != 1
        or hidden.dtype not in (torch.bfloat16, torch.float32)
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or weight.device != hidden.device
        or weight.numel() != hidden.shape[-1]
        or weight.ndim != 1
        or weight.stride(0) != 1
    ):
        return None
    size = hidden.shape[-1]
    rows = hidden.numel() // size
    if not 0 < rows <= 512:
        return None
    dim0 = min(512, 1 << ((size // 4).bit_length() - 1))
    dim1 = min(512, 1 << (rows.bit_length() - 1))
    width = min(dim0, 32)
    height = min(dim1, 512 // width)
    width = min(dim0, 512 // height)
    group = hidden.shape[-2] if hidden.ndim == 3 else 1
    stride_0 = hidden.stride(-3) if hidden.ndim == 3 else hidden.stride(0)
    result = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    _decode_rms_norm[(rows,)](
        hidden,
        weight,
        result,
        size,
        group,
        stride_0,
        hidden.stride(-2),
        eps,
        width,
        triton.next_power_of_2(size),
        num_warps=max(1, width // 32),
        enable_fp_fusion=False,
    )
    return result
