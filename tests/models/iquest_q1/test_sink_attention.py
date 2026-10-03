# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical check for the IQuestQ1 learned attention sink.

The sink contributes a key but no value, so it only grows the softmax
denominator. The paged kernel must match a dense reference that appends the
sink key explicitly.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cuda():
    pytest.skip("IQuest sink attention requires CUDA", allow_module_level=True)

try:
    from vllm.vllm_flash_attn import is_fa_version_supported
except ImportError:
    pytest.skip("vllm_flash_attn is unavailable", allow_module_level=True)


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="Hopper small-batch decode"
)
@pytest.mark.parametrize("num_tokens", [1, 6, 8])
@pytest.mark.parametrize("seq_len", [17, 4097])
@pytest.mark.parametrize("window", [None, 17, 4096])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize(
    "kv_cache_dtype,q_scale", [("auto", 1.0), ("fp8_e4m3", 1.0), ("fp8_e4m3", 0.3)]
)
@torch.inference_mode()
def test_small_batch_sink_matches_dense(
    num_tokens, seq_len, window, page_size, kv_cache_dtype, q_scale
):
    """Decode preserves masks and learned sinks when graph inputs change."""
    from types import SimpleNamespace

    from vllm.models.iquest_q1.attention import IQuestFlashAttentionImpl
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    set_random_seed(93010)
    device = "cuda"
    heads, dim = 6, 128
    quantized = kv_cache_dtype.startswith("fp8")
    pages = (seq_len + page_size - 1) // page_size
    query = torch.randn(
        num_tokens + 2, heads + 1, dim, device=device, dtype=torch.bfloat16
    )[1:, :heads]
    cache = torch.randn(
        pages, 1, page_size, 2 * dim, device=device, dtype=torch.bfloat16
    )
    if quantized:
        cache = cache.to(torch.float8_e4m3fn).view(torch.uint8)
    table = torch.randperm(pages, device=device).int().view(1, -1)
    lengths = torch.tensor([seq_len], device=device, dtype=torch.int32)
    query_start = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    layer = SimpleNamespace(
        sink_key=4 * torch.randn(1, dim, device=device, dtype=torch.bfloat16),
        _q_scale=torch.tensor([q_scale], device=device),
        _k_scale=torch.tensor([0.7], device=device),
        _v_scale=torch.tensor([1.3], device=device),
    )
    metadata = FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        query_start_loc=query_start,
        max_seq_len=seq_len,
        seq_lens=lengths,
        block_table=table,
        slot_mapping=torch.empty(0, device=device, dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    impl = IQuestFlashAttentionImpl(
        heads, dim, dim**-0.5, 1, None, window, kv_cache_dtype
    )
    impl.vllm_flash_attn_version = 3
    output = torch.full(
        (num_tokens + 2, heads + 1, dim), -73, device=device, dtype=torch.bfloat16
    )[1:, :heads]

    def run():
        impl.forward(layer, query, None, None, cache, metadata, output)

    def dense(active_tokens, active_seq_len):
        packed = cache.view(torch.float8_e4m3fn) if quantized else cache
        packed = packed.float()[table[0].long()]
        key, value = packed.reshape(-1, 2 * dim)[:active_seq_len].chunk(2, -1)
        q = query[:active_tokens].float()
        qdq = q
        if quantized:
            key = key * layer._k_scale
            value = value * layer._v_scale
            qdq = (q / layer._q_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            qdq = qdq.float() * layer._q_scale
        scores = torch.einsum("mhd,nd->mhn", qdq, key) * dim**-0.5
        query_positions = torch.arange(
            active_seq_len - active_tokens, active_seq_len, device=device
        )
        key_positions = torch.arange(active_seq_len, device=device)
        visible = key_positions[None, :] <= query_positions[:, None]
        if window is not None:
            visible &= key_positions[None, :] > query_positions[:, None] - window
        scores.masked_fill_(~visible[:, None, :], -torch.inf)
        sink = torch.einsum("mhd,kd->mh", q, layer.sink_key.float()) * dim**-0.5
        probabilities = torch.cat([scores, sink[:, :, None]], -1).softmax(-1)
        return torch.einsum("mhn,nd->mhd", probabilities[:, :, :-1], value)

    run()
    torch.testing.assert_close(
        output[:num_tokens].float(), dense(num_tokens, seq_len), atol=1e-2, rtol=1e-2
    )
    assert torch.all(output[num_tokens:] == -73)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.testing.assert_close(
        output[:num_tokens].float(), dense(num_tokens, seq_len), atol=1e-2, rtol=1e-2
    )
    # Replay uses new device lengths without recapturing the padded query shape.
    query_start[1] = 1
    lengths[0] = seq_len - 1
    query.mul_(0.5)
    table.copy_(table.flip(-1).clone())
    layer.sink_key.mul_(2)
    graph.replay()
    torch.testing.assert_close(
        output[:1].float(), dense(1, seq_len - 1), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 1024])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_sink_preserves_fp32_eager_rounding(num_tokens, output_dtype, strided):
    """The sink correction must preserve rounding and leave padded rows untouched."""
    from vllm.models.iquest_q1.attention import apply_sink_key

    set_random_seed(42)
    num_heads, num_kv_heads, head_size = 6, 2, 128
    storage_heads = num_heads + int(strided)
    query = torch.randn(
        num_tokens + 3, storage_heads, head_size, device="cuda", dtype=torch.bfloat16
    )[1:, :num_heads]
    sink = torch.randn(num_kv_heads, head_size, device="cuda", dtype=torch.bfloat16)
    output = torch.randn(
        num_tokens + 3, storage_heads, head_size, device="cuda", dtype=output_dtype
    )[1:, :num_heads]
    before = output.clone()
    repeated_sink = sink.repeat_interleave(num_heads // num_kv_heads, dim=0)
    scale = head_size**-0.5
    dot = (query[:num_tokens].float() * repeated_sink.float()).sum(-1) * scale
    lse = (dot + torch.randn_like(dot)).T
    if not strided:
        lse = lse.contiguous()
    expected = (
        before[:num_tokens].float() * torch.sigmoid(lse.T - dot).unsqueeze(-1)
    ).to(output_dtype)

    apply_sink_key(query, sink, output, lse, scale, num_tokens)

    torch.testing.assert_close(output[:num_tokens], expected, rtol=0, atol=0)
    torch.testing.assert_close(output[num_tokens:], before[num_tokens:], rtol=0, atol=0)


@pytest.mark.parametrize("fa_version", [2, 3, 4])
@pytest.mark.parametrize("num_heads", [(6, 1), (6, 2)])
@pytest.mark.parametrize("head_size", [72, 128])
@pytest.mark.parametrize("sliding_window", [None, 17])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_lens", [[1, 6, 19], [1, 1, 1]])
@pytest.mark.parametrize(
    "kv_cache_dtype,per_head_scales",
    [("auto", False), ("fp8", False), ("fp8_e4m3", True)],
)
@torch.inference_mode()
def test_iquest_paged_learned_sink_matches_dense_attention(
    fa_version: int,
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    query_lens: list[int],
    kv_cache_dtype: str,
    per_head_scales: bool,
) -> None:
    """A learned key remains visible outside SWA and respects KV-head sharing."""
    if not is_fa_version_supported(fa_version):
        pytest.skip(f"FlashAttention {fa_version} is unavailable")
    quantized = kv_cache_dtype.startswith("fp8")
    if quantized and (
        dtype != torch.bfloat16
        or head_size % 16 != 0
        or not (
            fa_version == 3
            and current_platform.is_device_capability_family(90)
            or fa_version == 4
            and current_platform.is_device_capability_family(100)
        )
    ):
        pytest.skip("FP8 requires supported FA/device, BF16 output and aligned heads")
    if (
        fa_version == 4
        and current_platform.is_device_capability(90)
        and head_size % 16 != 0
    ):
        pytest.skip("FA4 paged KV on SM90 requires a head size divisible by 16")

    from types import SimpleNamespace

    from vllm.models.iquest_q1.attention import IQuestFlashAttentionImpl
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    set_random_seed(0)
    device = "cuda"
    kv_lens = [47, 31, 19]
    num_tokens = sum(query_lens)
    num_q_heads, num_kv_heads = num_heads
    block_size = 16
    scale = head_size**-0.5
    query = torch.randn(
        num_tokens + 5, num_q_heads + 1, head_size, dtype=dtype, device=device
    )[:, :num_q_heads]
    sink_key = 4 * torch.randn(num_kv_heads, head_size, dtype=dtype, device=device)
    kv_cache = torch.randn(
        9, num_kv_heads, block_size, 2 * head_size, dtype=dtype, device=device
    )
    block_table = torch.randperm(9, device=device).to(torch.int32).reshape(3, 3)
    query_start_loc = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(0, dtype=torch.int32)
    metadata = FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        query_start_loc=query_start_loc,
        max_seq_len=max(kv_lens),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        block_table=block_table,
        slot_mapping=torch.empty(num_tokens, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    impl = IQuestFlashAttentionImpl(
        num_q_heads,
        head_size,
        scale,
        num_kv_heads,
        None,
        sliding_window,
        kv_cache_dtype,
    )
    impl.vllm_flash_attn_version = fa_version
    impl.fa4_hd256 = False
    layer = SimpleNamespace(sink_key=sink_key)
    reference_query = query[:num_tokens].float()
    reference_cache = kv_cache.float()
    if quantized:
        scale_heads = num_kv_heads if per_head_scales else 1
        layer._q_scale = torch.linspace(0.25, 0.5, scale_heads, device=device)
        layer._k_scale = torch.linspace(0.125, 0.25, scale_heads, device=device)
        layer._v_scale = torch.linspace(0.5, 1.0, scale_heads, device=device)
        keys, values = (
            kv_cache.transpose(1, 2)
            .reshape(-1, num_kv_heads, 2 * head_size)
            .chunk(2, dim=-1)
        )
        kv_cache = torch.empty_like(kv_cache, dtype=torch.uint8)
        slots = torch.arange(keys.shape[0], device=device, dtype=torch.int64)
        impl.do_kv_cache_update(
            layer, keys.contiguous(), values.contiguous(), kv_cache, slots
        )
        fp8_dtype = current_platform.fp8_dtype()
        fp8_max = torch.finfo(fp8_dtype).max
        k_scale = layer._k_scale.view(1, -1, 1, 1)
        v_scale = layer._v_scale.view(1, -1, 1, 1)
        k, v = reference_cache.chunk(2, dim=-1)
        packed_reference = torch.cat(
            [
                (k / k_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype),
                (v / v_scale).clamp(-fp8_max, fp8_max).to(fp8_dtype),
            ],
            dim=-1,
        )
        torch.testing.assert_close(kv_cache, packed_reference.view(torch.uint8))
        k, v = packed_reference.float().chunk(2, dim=-1)
        reference_cache = torch.cat([k * k_scale, v * v_scale], dim=-1)
        q_scale = (
            layer._q_scale.expand(num_kv_heads)
            .repeat_interleave(num_q_heads // num_kv_heads)
            .view(1, num_q_heads, 1)
        )
        reference_query = (reference_query / q_scale).clamp(-fp8_max, fp8_max).to(
            fp8_dtype
        ).float() * q_scale
    output = torch.full(
        (num_tokens + 5, num_q_heads + 1, head_size),
        -73,
        dtype=dtype,
        device=device,
    )[:, :num_q_heads]
    impl.forward(
        layer,
        query,
        None,
        None,
        kv_cache,
        metadata,
        output,
    )
    if quantized:
        eager_output = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            impl.forward(layer, query, None, None, kv_cache, metadata, output)
        graph.replay()
        torch.testing.assert_close(output, eager_output, rtol=0, atol=0)
        from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func

        k, v = packed_reference.transpose(1, 2).chunk(2, dim=-1)
        normal_output, normal_lse = flash_attn_varlen_func(
            q=(reference_query / q_scale).to(fp8_dtype),
            k=k,
            v=v,
            cu_seqlens_q=query_start_loc,
            seqused_k=metadata.seq_lens,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max(kv_lens),
            softmax_scale=scale,
            causal=True,
            window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
            block_table=block_table,
            fa_version=fa_version,
            q_descale=layer._q_scale.expand(len(kv_lens), num_kv_heads),
            k_descale=layer._k_scale.expand(len(kv_lens), num_kv_heads),
            v_descale=layer._v_scale.expand(len(kv_lens), num_kv_heads),
            return_softmax_lse=True,
        )

    repeat = num_q_heads // num_kv_heads
    sink = sink_key.float().repeat_interleave(repeat, dim=0)
    start = 0
    for seq, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        cache = (
            reference_cache[block_table[seq]]
            .transpose(1, 2)
            .reshape(-1, num_kv_heads, 2 * head_size)[:kv_len]
        )
        key, value = cache.float().repeat_interleave(repeat, dim=1).chunk(2, dim=-1)
        q = reference_query[start : start + query_len]
        logits = torch.einsum("qhd,khd->hqk", q, key) * scale
        query_pos = torch.arange(kv_len - query_len, kv_len, device=device)
        key_pos = torch.arange(kv_len, device=device)
        visible = key_pos[None, :] <= query_pos[:, None]
        if sliding_window is not None:
            visible &= key_pos[None, :] > query_pos[:, None] - sliding_window
        logits.masked_fill_(~visible[None, :, :], -torch.inf)
        sink_logits = (
            torch.einsum("qhd,hd->hq", query[start : start + query_len].float(), sink)
            * scale
        )
        if quantized:
            dense_lse = logits.logsumexp(-1)
            torch.testing.assert_close(
                normal_lse[:, start : start + query_len],
                dense_lse,
                atol=1e-3,
                rtol=1e-3,
            )
            corrected = normal_output[start : start + query_len].float() * (
                dense_lse - sink_logits
            ).sigmoid().T.unsqueeze(-1)
            torch.testing.assert_close(
                output[start : start + query_len].float(),
                corrected,
                atol=1e-2,
                rtol=1e-2,
            )
        probabilities = torch.cat([logits, sink_logits[..., None]], dim=-1).softmax(-1)
        expected = torch.einsum("hqk,khd->qhd", probabilities[..., :-1], value)
        # FP8 FlashAttention also rounds its internal attention probabilities.
        tolerance = 5e-2 if quantized else (1e-2 if dtype == torch.bfloat16 else 1e-3)
        torch.testing.assert_close(
            output[start : start + query_len].float(),
            expected,
            atol=tolerance,
            rtol=tolerance,
        )
        start += query_len
    assert torch.all(output[num_tokens:] == -73)
