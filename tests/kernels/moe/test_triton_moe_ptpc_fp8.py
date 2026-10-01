# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/blob/main/test/srt/test_triton_moe_channel_fp8_kernel.py
import importlib
import itertools

import pytest
import torch

from tests.kernels.moe.utils import fused_moe
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_per_token_quant_fp8,
)
from vllm.platforms import current_platform

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()


@pytest.mark.skipif(
    current_platform.get_device_name() != "NVIDIA H200",
    reason="Measured H200 small-batch dispatch",
)
@pytest.mark.parametrize("routing", ["random", "two_groups", "shared"])
@pytest.mark.parametrize("implementation", ["modular", "functional"])
@torch.inference_mode()
def test_fp8_six_token_grouping_preserves_expert_outputs(
    monkeypatch, routing, implementation
):
    """Grouping and the 192-wide down GEMM preserve the original FP8 output."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    functional = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.fused_moe"
    )
    modular = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.experts.triton_moe"
    )
    torch.manual_seed(93007)
    init_workspace_manager(torch.device("cuda:0"))

    def quantize_weight(shape):
        source = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
        weight, scale = ops.scaled_fp8_quant(
            source.flatten(0, 1), use_per_token_if_dynamic=True
        )
        return weight.reshape(shape), scale.reshape(*shape[:-1], 1)

    w1, s1 = quantize_weight((256, 384, 3072))
    w2, s2 = quantize_weight((256, 3072, 192))
    quant = fp8_w8a8_moe_quant_config(
        s1, s2, per_act_token_quant=True, per_out_ch_quant=True
    )
    config = make_dummy_moe_config(
        num_experts=256,
        experts_per_token=8,
        hidden_dim=3072,
        intermediate_size=192,
        max_num_tokens=6,
    )
    kernel = FusedMoEKernel(
        prepare_finalize=MoEPrepareAndFinalizeNoDPEPModular(),
        fused_experts=TritonExperts(config, quant),
    )
    x = torch.randn(6, 3072, device="cuda", dtype=torch.bfloat16)
    values, ids = torch.randn(6, 256, device="cuda").topk(8)
    ids = ids.int()
    if routing != "random":
        ids[:] = torch.arange(8, device="cuda", dtype=torch.int32)
        if routing == "two_groups":
            ids[3:] += 8
    weights = values.softmax(-1)

    def run():
        if implementation == "functional":
            return functional.fused_experts(x, w1, w2, weights, ids, quant_config=quant)
        return kernel.apply(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=weights,
            topk_ids=ids,
            global_num_experts=256,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

    original = functional._prepare_expert_assignment
    grouped = []

    def assignment(*args, **kwargs):
        result = original(*args, **kwargs)
        grouped.append(result[0] is not None)
        return result

    def ungrouped(*args, **kwargs):
        kwargs["use_small_batch_grouping"] = False
        return original(*args, **kwargs)

    with set_current_vllm_config(vllm_config):
        with monkeypatch.context() as patch:
            patch.setattr(
                functional, "_get_fp8_moe_down_config", lambda a, b, config: config
            )
            for module in [functional, modular]:
                patch.setattr(module, "_prepare_expert_assignment", ungrouped)
            expected = run().clone()
        with monkeypatch.context() as patch:
            for module in [functional, modular]:
                patch.setattr(module, "_prepare_expert_assignment", assignment)
            actual = run().clone()
        assert grouped and all(grouped)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if current_platform.is_fp8_fnuz():
    pytest.skip(
        "Tests in this file require float8_e4m3fn and platform does not support",
        allow_module_level=True,
    )


def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
    """Matrix multiplication function that supports per-token input
    quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    return C.reshape(origin_C_shape).to(output_dtype)


def fp8_mask(a, mask):
    dtype = a.dtype
    return a.view(torch.int8)[mask].view(dtype)


def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk):
    """This function performs fused moe with per-column int8
    quantization using native torch."""
    B, D = a.shape
    # Perform per-token quantization
    a_q, a_s = ops.scaled_fp8_quant(a, use_per_token_if_dynamic=True)
    # Repeat tokens to match topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    # Also repeat the scale
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]

    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    # Calculate routing
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # Process each expert
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # First MLP layer: note that a_s is now per-token
            inter_out = native_w8a8_per_token_matmul(
                fp8_mask(a_q, mask),
                w1[i],
                fp8_mask(a_s, mask),
                w1_s[i],
                output_dtype=a.dtype,
            )
            # Activation function
            act_out = SiluAndMul().forward_native(inter_out)
            # Quantize activation output with per-token
            act_out_q, act_out_s = ops.scaled_fp8_quant(
                act_out, use_per_token_if_dynamic=True
            )

            # Second MLP layer
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=a.dtype
            )
    # Apply routing weights and sum
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


@pytest.fixture(autouse=True)
def setup_cuda():
    """Sets the default CUDA device before each test in this module."""
    torch.set_default_device("cuda")


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA FP8 fusion")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [0, 1, 8, 48, 1024])
@pytest.mark.parametrize("hidden_size", [64, 192, 1024])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_fused_silu_preserves_per_token_fp8_values(
    dtype, num_tokens, hidden_size, strided
):
    """Preserve CUDA activation rounding and FP8 scale boundaries, including graphs."""
    torch.manual_seed(13)
    width = 2 * hidden_size
    storage = torch.randn(num_tokens + 1, width + 8, dtype=dtype)
    x = storage[1:, 1 : width + 1] if strided else storage[:num_tokens, :width].clone()
    for amplitude in [0.0001, 1.0, 16.0]:
        x.copy_(torch.randn_like(x) * amplitude)
        if num_tokens:
            x[0].zero_()
            x[0, 0], x[0, hidden_size] = 24, 28
            x[0, 1], x[0, hidden_size + 1] = 14.125, -11.5
        quantized, scales = silu_mul_per_token_quant_fp8(x)
        assert quantized.shape == (num_tokens, hidden_size)
        assert scales.shape == (num_tokens, 1)
        if not num_tokens:
            continue
        intermediate = torch.empty_like(quantized, dtype=dtype)
        # A one-row contiguous view can still have an unaligned storage offset.
        aligned = x.contiguous().clone()
        torch.ops._C.silu_and_mul(intermediate, aligned)
        expected, expected_scales = ops.scaled_fp8_quant(
            intermediate, use_per_token_if_dynamic=True
        )
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(quantized.float(), expected.float(), rtol=0, atol=0)
        assert scales[0, 0] == 1.5
        assert quantized[0, 1].float() == -112
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output, graph_scales = silu_mul_per_token_quant_fp8(x)
        graph.replay()
        torch.testing.assert_close(graph_scales, expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(
            graph_output.float(), expected.float(), rtol=0, atol=0
        )


DTYPES = [torch.half, torch.bfloat16]
M = [1, 33]
N = [128, 1024]
K = [256, 4096]
E = [8]
TOP_KS = [2, 6]
SEEDS = [0]


@pytest.mark.parametrize(
    "M, N, K, E, topk, dtype, seed",
    itertools.product(M, N, K, E, TOP_KS, DTYPES, SEEDS),
)
@torch.inference_mode()
def test_w8a8_fp8_fused_moe(M, N, K, E, topk, dtype, seed):
    torch.manual_seed(seed)
    # Initialize int8 quantization parameters
    factor_for_scale = 1e-2
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = finfo.min

    # Input tensor
    # M * K
    a = torch.randn((M, K), dtype=dtype) / 10

    # Generate int8 weights
    w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
    w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
    w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    # Generate scale for each column (per-column quantization)
    w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * factor_for_scale
    w2_s = torch.rand(E, K, device=w2_fp32.device) * factor_for_scale
    score = torch.randn((M, E), dtype=dtype)

    with set_current_vllm_config(vllm_config):
        ref_out = torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk)
        out = fused_moe(
            a,
            w1,
            w2,
            score,
            topk,
            renormalize=False,
            quant_config=fp8_w8a8_moe_quant_config(
                per_act_token_quant=True,
                w1_scale=w1_s,
                w2_scale=w2_s,
                block_shape=None,  # Not using block quantization
            ),
        )

    # Check results
    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.05
