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

try:
    from vllm.vllm_flash_attn import is_fa_version_supported
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "vllm_flash_attn is not supported for vLLM on ROCm.",
            allow_module_level=True,
        )


@pytest.mark.parametrize("fa_version", [2, 3, 4])
@pytest.mark.parametrize("num_heads", [(6, 1), (6, 2)])
@pytest.mark.parametrize("head_size", [72, 128])
@pytest.mark.parametrize("sliding_window", [None, 17])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_lens", [[1, 3, 19], [1, 1, 1]])
@torch.inference_mode()
def test_iquest_paged_learned_sink_matches_dense_attention(
    fa_version: int,
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    query_lens: list[int],
) -> None:
    """A learned key remains visible outside SWA and respects KV-head sharing."""
    if not is_fa_version_supported(fa_version):
        pytest.skip(f"FlashAttention {fa_version} is unavailable")
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
        num_q_heads, head_size, scale, num_kv_heads, None, sliding_window, "auto"
    )
    impl.vllm_flash_attn_version = fa_version
    impl.fa4_hd256 = False
    output = torch.full(
        (num_tokens + 5, num_q_heads + 1, head_size),
        -73,
        dtype=dtype,
        device=device,
    )[:, :num_q_heads]
    impl.forward(
        SimpleNamespace(sink_key=sink_key),
        query,
        None,
        None,
        kv_cache,
        metadata,
        output,
    )

    repeat = num_q_heads // num_kv_heads
    sink = sink_key.float().repeat_interleave(repeat, dim=0)
    start = 0
    for seq, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        cache = (
            kv_cache[block_table[seq]]
            .transpose(1, 2)
            .reshape(-1, num_kv_heads, 2 * head_size)[:kv_len]
        )
        key, value = cache.float().repeat_interleave(repeat, dim=1).chunk(2, dim=-1)
        q = query[start : start + query_len].float()
        logits = torch.einsum("qhd,khd->hqk", q, key) * scale
        query_pos = torch.arange(kv_len - query_len, kv_len, device=device)
        key_pos = torch.arange(kv_len, device=device)
        visible = key_pos[None, :] <= query_pos[:, None]
        if sliding_window is not None:
            visible &= key_pos[None, :] > query_pos[:, None] - sliding_window
        logits.masked_fill_(~visible[None, :, :], -torch.inf)
        sink_logits = torch.einsum("qhd,hd->hq", q, sink) * scale
        probabilities = torch.cat([logits, sink_logits[..., None]], dim=-1).softmax(-1)
        expected = torch.einsum("hqk,khd->qhd", probabilities[..., :-1], value)
        torch.testing.assert_close(
            output[start : start + query_len].float(),
            expected,
            atol=1e-2 if dtype == torch.bfloat16 else 1e-3,
            rtol=1e-2 if dtype == torch.bfloat16 else 1e-3,
        )
        start += query_len
    assert torch.all(output[num_tokens:] == -73)
