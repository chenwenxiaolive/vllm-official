# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch BF16 paged attention with a learned zero-valued sink."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _bf16_attention_partial(
    Q,
    Cache,
    Table,
    Length,
    QueryStart,
    Partial,
    Lse,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
):
    row = tl.program_id(0) * BH + tl.arange(0, BH)
    split = tl.program_id(1)
    token, head = row // H, row % H
    d, n = tl.arange(0, D), tl.arange(0, BN)
    length = tl.load(Length)
    query_len = tl.load(QueryStart + 1) - tl.load(QueryStart)
    query_end = length - query_len + token + 1
    begin = tl.maximum(0, length - query_len + 1 - WINDOW) if WINDOW > 0 else 0
    width = tl.cdiv(length - begin, SPLITS * BN) * BN
    start = begin + split * width
    stop = tl.minimum(start + width, length)
    q = tl.load(
        Q + token[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :],
        (row[:, None] < M * H) & (token[:, None] < query_len),
        0,
    )
    maximum = tl.full((BH,), float("-inf"), tl.float32)
    denom = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, D), tl.float32)
    for block in tl.range(start, stop, BN):
        pos = block + n
        page = tl.load(Table + pos // PAGE, pos < stop, 0).to(tl.int64)
        offset = page * PAGE * (2 * D) + (pos % PAGE) * (2 * D)
        k = tl.load(Cache + offset[None, :] + d[:, None], pos[None, :] < stop, 0)
        v = tl.load(Cache + offset[:, None] + D + d[None, :], pos[:, None] < stop, 0)
        scores = tl.dot(q, k) * SCALE
        visible = (pos[None, :] < stop) & (pos[None, :] < query_end[:, None])
        visible &= token[:, None] < query_len
        if WINDOW > 0:
            visible &= pos[None, :] >= query_end[:, None] - WINDOW
        scores = tl.where(visible, scores, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.exp(scores - safe_max[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        denom = denom * alpha + tl.sum(p, axis=1)
        maximum = next_max
    valid = denom > 0
    normalized = acc / tl.where(valid, denom, 1.0)[:, None]
    logsum = tl.where(valid, maximum + tl.log(denom), float("-inf"))
    tl.store(
        Partial + (row[:, None] * SPLITS + split) * D + d[None, :],
        normalized,
        row[:, None] < M * H,
    )
    tl.store(Lse + row * SPLITS + split, logsum, row < M * H)


@triton.jit
def _bf16_attention_combine_sink(
    Partial,
    Lse,
    Q,
    Sink,
    QueryStart,
    Output,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    output_stride_t: tl.constexpr,
    output_stride_h: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    token, head = row // H, row % H
    active = token < tl.load(QueryStart + 1) - tl.load(QueryStart)
    s, d = tl.arange(0, SPLITS), tl.arange(0, D)
    split_lse = tl.load(Lse + row * SPLITS + s)
    maximum = tl.max(split_lse, axis=0)
    safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
    p = tl.exp(split_lse - safe_max)
    denom = tl.sum(p, axis=0)
    values = tl.load(Partial + (row * SPLITS + s[:, None]) * D + d[None, :])
    normal = tl.sum(values * p[:, None], axis=0) / tl.where(denom > 0, denom, 1.0)
    # FA3 rounds the normal attention output before applying the learned sink.
    normal = normal.to(tl.bfloat16).to(tl.float32)
    q = tl.load(Q + token * stride_qt + head * stride_qh + d, active, 0).to(tl.float32)
    sink = tl.load(Sink + d).to(tl.float32)
    sink_score = tl.sum(q * sink, axis=0) * SCALE
    normal_lse = tl.where(denom > 0, maximum + tl.log(denom), float("-inf"))
    result = normal * tl.sigmoid(normal_lse - sink_score)
    tl.store(
        Output + token * output_stride_t + head * output_stride_h + d, result, active
    )


def bf16_paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    sink_key: torch.Tensor,
    output: torch.Tensor,
    window: int,
    scale: float,
) -> None:
    """Attend to one sequence with six query heads and one BF16 KV head."""
    num_tokens, num_heads, head_size = query.shape
    num_splits = 16 if window == 512 else 128
    block_n = 64 if window > 0 else 128
    partial_output = torch.empty(
        (num_tokens, num_heads, num_splits, head_size),
        device=query.device,
        dtype=torch.float32,
    )
    partial_lse = torch.empty(
        (num_tokens, num_heads, num_splits),
        device=query.device,
        dtype=torch.float32,
    )
    _bf16_attention_partial[(triton.cdiv(num_tokens * num_heads, 16), num_splits)](
        query,
        kv_cache,
        block_table,
        seq_lens,
        query_start_loc,
        partial_output,
        partial_lse,
        query.stride(0),
        query.stride(1),
        num_tokens,
        num_heads,
        head_size,
        kv_cache.shape[2],
        window,
        scale,
        num_splits,
        block_n,
        16,
        num_warps=4,
        num_stages=2,
    )
    _bf16_attention_combine_sink[(num_tokens * num_heads,)](
        partial_output,
        partial_lse,
        query,
        sink_key,
        query_start_loc,
        output,
        query.stride(0),
        query.stride(1),
        output.stride(0),
        output.stride(1),
        num_heads,
        head_size,
        scale,
        num_splits,
        num_warps=4,
    )
