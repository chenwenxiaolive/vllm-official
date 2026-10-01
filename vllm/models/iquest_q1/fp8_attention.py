# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch paged attention with FP8 KV and FP32 accumulation."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fp8_attention_partial(
    Q,
    Cache,
    Table,
    Length,
    QueryStart,
    QS,
    KS,
    VS,
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
    tile = tl.program_id(0)
    split = tl.program_id(1)
    row = tile * BH + tl.arange(0, BH)
    token = row // H
    h = row % H
    d = tl.arange(0, D)
    n = tl.arange(0, BN)
    length = tl.load(Length)
    end = length
    query_len = tl.load(QueryStart + 1) - tl.load(QueryStart)
    query_end = length - query_len + token + 1
    begin = tl.maximum(0, end - query_len + 1 - WINDOW) if WINDOW > 0 else 0
    width = tl.cdiv(end - begin, SPLITS * BN) * BN
    start = begin + split * width
    stop = tl.minimum(start + width, end)
    qs = tl.load(QS)
    ks = tl.load(KS)
    vs = tl.load(VS)
    q = tl.load(
        Q + token[:, None] * stride_qt + h[:, None] * stride_qh + d[None, :],
        (row[:, None] < M * H) & (token[:, None] < query_len),
        0,
    ).to(tl.float32)
    q = (
        tl.minimum(tl.maximum(tl.div_rn(q, qs), -448.0), 448.0)
        .to(tl.float8e4nv)
        .to(tl.float16)
    )
    maximum = tl.full((BH,), float("-inf"), tl.float32)
    denom = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, D), tl.float32)
    for block in tl.range(start, stop, BN):
        pos = block + n
        pages = tl.load(Table + pos // PAGE, pos < stop, 0).to(tl.int64)
        base = pages * PAGE * (2 * D) + (pos % PAGE) * (2 * D)
        k = tl.load(Cache + base[None, :] + d[:, None], pos[None, :] < stop, 0.0).to(
            tl.float16
        )
        v = tl.load(
            Cache + base[:, None] + D + d[None, :], pos[:, None] < stop, 0.0
        ).to(tl.float16)
        score = tl.dot(q, k) * (SCALE * qs * ks)
        visible = (
            (pos[None, :] < stop)
            & (pos[None, :] < query_end[:, None])
            & (token[:, None] < query_len)
        )
        if WINDOW > 0:
            visible &= pos[None, :] >= query_end[:, None] - WINDOW
        score = tl.where(visible, score, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, axis=1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        p = tl.exp(score - safe_max[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        denom = denom * alpha + tl.sum(p, axis=1)
        maximum = next_max
    valid = denom > 0
    normalized = tl.where(
        valid[:, None], acc / tl.where(valid, denom, 1.0)[:, None] * vs, 0.0
    )
    logsum = tl.where(valid, maximum + tl.log(denom), float("-inf"))
    tl.store(
        Partial + (row[:, None] * SPLITS + split) * D + d[None, :],
        normalized,
        row[:, None] < M * H,
    )
    tl.store(Lse + row * SPLITS + split, logsum, row < M * H)


@triton.jit
def _fp8_attention_combine(
    Partial,
    Logsum,
    Output,
    Lse,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    split_lse = tl.load(Logsum + row * SPLITS + s, s < SPLITS, float("-inf"))
    maximum = tl.max(split_lse, axis=0)
    safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
    p = tl.exp(split_lse - safe_max)
    den = tl.sum(p, axis=0)
    values = tl.load(
        Partial + (row * SPLITS + s[:, None]) * D + d[None, :], s[:, None] < SPLITS, 0
    )
    result = tl.sum(values * p[:, None], axis=0) / tl.where(den > 0, den, 1.0)
    tl.store(Output + (row // H) * stride_ot + (row % H) * stride_oh + d, result)
    tl.store(
        Lse + (row % H) * M + row // H, tl.where(den > 0, maximum + tl.log(den), 0.0)
    )


def fp8_paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Return natural-log LSE for one sequence with up to eight query tokens."""
    num_tokens, num_heads, head_size = query.shape
    num_splits = 32 if window > 0 else (256 if num_tokens == 1 else 128)
    block_rows = 16 if window > 0 or num_tokens == 1 else 32
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
    lse = torch.empty((num_heads, num_tokens), device=query.device, dtype=torch.float32)
    _fp8_attention_partial[
        (triton.cdiv(num_tokens * num_heads, block_rows), num_splits)
    ](
        query,
        kv_cache.view(torch.float8_e4m3fn),
        block_table,
        seq_lens,
        query_start_loc,
        q_scale,
        k_scale,
        v_scale,
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
        128,
        block_rows,
        num_warps=4,
        num_stages=2,
    )
    _fp8_attention_combine[(num_tokens * num_heads,)](
        partial_output,
        partial_lse,
        output,
        lse,
        output.stride(0),
        output.stride(1),
        num_tokens,
        num_heads,
        head_size,
        num_splits,
        num_splits,
        num_warps=4,
    )
    return lse
