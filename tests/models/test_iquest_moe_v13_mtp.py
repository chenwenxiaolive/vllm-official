# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the M1 draft head's checkpoint and residual contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.model_executor.models import iquest_moe_v13_mtp as mtp


class _Attention(nn.Module):
    def forward(self, positions, hidden_states):
        return (hidden_states.roll(1, dims=-1) + positions[:, None] * 0.1).to(
            hidden_states.dtype
        )


@pytest.fixture
def tiny_config(monkeypatch):
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=3,
        num_experts=2,
        num_experts_per_tok=1,
        rms_norm_eps=1e-5,
        first_layer_attn_out_scale=0.7,
        first_layer_ffn_out_scale=0.4,
        attn_out_scale=1.3,
        ffn_out_scale=1.7,
        num_hidden_layers=42,
        num_mtp_layers=2,
        vocab_size=8,
    )
    monkeypatch.setattr(mtp, "IquestMoeAttention", lambda **kwargs: _Attention())
    monkeypatch.setattr(mtp, "IquestMoEBlock", lambda **kwargs: nn.SiLU())
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config), quant_config=None
    )


def _rms_norm(x, module):
    return F.rms_norm(
        x.float(), (x.shape[-1],), module.weight.float(), module.variance_epsilon
    ).to(x.dtype)


@pytest.mark.parametrize("first_layer", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mtp_residual_and_scaling_match_training(tiny_config, first_layer, dtype):
    """Only the first MTP block retains the unnormalized attention residual."""
    layer = mtp.IquestMoeV13MTPInnerLayer(
        vllm_config=tiny_config, use_sandwich_norm=first_layer
    ).to(dtype=dtype)
    with torch.no_grad():
        for module in layer.modules():
            if isinstance(module, mtp.IquestMoeRMSNorm):
                module.weight.copy_(torch.tensor([0.8, 1.1, 0.7, 1.3]))
    hidden = torch.tensor([[2.0, -1.0, 4.0, 0.5], [-3.0, 1.0, 0.5, 2.0]]).to(dtype)
    positions = torch.tensor([2, 7])
    normalized = _rms_norm(hidden, layer.attention_norm)
    attn_output = layer.self_attn(positions, normalized)
    residual = hidden if first_layer else normalized
    residual = residual + _rms_norm(attn_output, layer.attn_out_norm) * (
        0.7 if first_layer else 1.3
    )
    ffn_output = F.silu(_rms_norm(residual, layer.feed_forward_norm))
    if first_layer:
        ffn_output = _rms_norm(ffn_output, layer.ffn_out_norm)
    expected = residual + ffn_output * (0.4 if first_layer else 1.7)

    torch.testing.assert_close(layer(positions, hidden), expected)


def test_mtp_masks_only_zero_position_embeddings(tiny_config):
    """The shifted BOS embedding is zeroed, while hidden states still contribute."""
    layer = mtp.IquestMoeV13MTPLayer(vllm_config=tiny_config)
    with torch.no_grad():
        layer.eh_proj.weight.copy_(torch.cat([torch.eye(4), torch.eye(4)], dim=1))
    layer.mtp_model_layer = _Attention()
    positions = torch.tensor([0, 1, 0, 6])
    hidden = torch.tensor([[1.0, -2.0, 3.0, -4.0]]).expand(4, -1)
    embeds = torch.tensor([[4.0, 3.0, 2.0, 1.0]]).expand(4, -1).clone()
    original_embeds = embeds.clone()
    masked_embeds = embeds.clone()
    masked_embeds[[0, 2]] = 0
    projected = _rms_norm(masked_embeds, layer.enorm) + _rms_norm(hidden, layer.hnorm)
    expected = _rms_norm(
        projected.roll(1, dims=-1) + positions[:, None] * 0.1,
        layer.final_layernorm,
    )

    torch.testing.assert_close(layer(positions, hidden, embeds), expected)
    torch.testing.assert_close(embeds, original_embeds)
    assert torch.count_nonzero(expected[0]) == 4


def test_mtp_steps_use_distinct_trained_layers(tiny_config, monkeypatch):
    """A later draft consumes the preceding head's states and its own parameters."""
    monkeypatch.setattr(
        mtp,
        "VocabParallelEmbedding",
        lambda vocab_size, hidden_size, **kwargs: nn.Embedding(vocab_size, hidden_size),
    )
    monkeypatch.setattr(
        mtp,
        "IquestMoeV13MTPFirstLayer",
        lambda **kwargs: mtp.IquestMoeV13MTPLayer(**kwargs, use_sandwich_norm=True),
    )
    monkeypatch.setattr(
        mtp,
        "IquestMoeV13MTPNextLayer",
        lambda **kwargs: mtp.IquestMoeV13MTPLayer(**kwargs),
    )
    torch.manual_seed(0)
    model = mtp.IquestMoeV13MultiTokenPredictor(vllm_config=tiny_config)
    ids = torch.tensor([1, 3, 2])
    positions = torch.tensor([0, 1, 2])
    hidden = torch.randn(3, 4)
    embeds = model.embed_input_ids(ids)
    first = model(ids, positions, hidden, spec_step_idx=0)
    second = model(None, positions, first, embeds, spec_step_idx=1)

    torch.testing.assert_close(first, model.layers["42"](positions, hidden, embeds))
    torch.testing.assert_close(second, model.layers["43"](positions, first, embeds))
    assert not torch.allclose(second, model.layers["42"](positions, first, embeds))
    assert not torch.allclose(second, model.layers["43"](positions, hidden, embeds))


@pytest.mark.parametrize("fused_experts", [False, True])
def test_mtp_loads_offset_layers_and_fused_weights(fused_experts):
    """MTP checkpoint layer IDs are relative; expert and QKV shards must not mix."""
    params = {}
    checkpoint = []
    expected = {}

    def add_parameter(name, value, loader=None):
        param = nn.Parameter(torch.zeros_like(value), requires_grad=False)
        if loader is not None:
            param.weight_loader = loader
        params[name] = param
        expected[name] = value

    def qkv_loader(param, value, shard_id):
        param[{"q": 0, "k": 1, "v": 2}[shard_id]].copy_(value)

    def expert_loader(param, value, name, shard_id, expert_id):
        if shard_id == "w2":
            param[expert_id].copy_(value)
        else:
            param[expert_id, {"w1": 0, "w3": 1}[shard_id]].copy_(value)

    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        value = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        add_parameter(name, value)
        checkpoint.append((name, value))
    for idx in range(2):
        src = f"mtp_layers.{idx}"
        dst = f"model.layers.{42 + idx}"
        qkv = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) + idx * 100
        add_parameter(
            f"{dst}.mtp_model_layer.self_attn.qkv_proj.weight", qkv, qkv_loader
        )
        for shard, value in zip(("q", "k", "v"), qkv):
            checkpoint.append(
                (f"{src}.mtp_model_layer.self_attn.{shard}_proj.weight", value)
            )

        w13 = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4) + idx * 100
        w2 = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) + idx * 100
        for name, value in (("w13_weight", w13), ("w2_weight", w2)):
            add_parameter(
                f"{dst}.mtp_model_layer.mlp.experts.routed_experts.{name}",
                value,
                expert_loader,
            )
        if fused_experts:
            checkpoint.extend(
                [
                    (f"{src}.mtp_model_layer.mlp.experts.fc", w13.reshape(2, 6, 4)),
                    (f"{src}.mtp_model_layer.mlp.experts.proj", w2),
                ]
            )
        else:
            for expert_id in range(2):
                for projection, value in (
                    ("gate_proj", w13[expert_id, 0]),
                    ("up_proj", w13[expert_id, 1]),
                    ("down_proj", w2[expert_id]),
                ):
                    checkpoint.append(
                        (
                            f"{src}.mtp_model_layer.mlp.experts.{expert_id}."
                            f"{projection}.weight",
                            value,
                        )
                    )
        router = torch.full((2, 4), 0.25 + idx)
        add_parameter(f"{dst}.mtp_model_layer.mlp.gate.weight", router)
        checkpoint.append((f"{src}.mtp_model_layer.mlp.router.weight", router))
        norm = torch.full((4,), 0.5 + idx)
        add_parameter(f"{dst}.final_layernorm.weight", norm)
        checkpoint.append((f"{src}.final_layernorm.weight", norm))

    checkpoint.append(("model.layers.0.self_attn.q_proj.weight", torch.ones(4, 4)))
    draft = nn.Module()
    draft.config = SimpleNamespace(num_experts=2, intermediate_size=3)
    draft.mtp_start_layer_idx = 42
    draft.named_parameters = lambda: iter(params.items())
    loaded = mtp.IquestMoeV13MTP.load_weights(draft, checkpoint)

    assert loaded == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(params[name], value, msg=name)


def test_mtp_proposer_chains_full_query_hidden_states(monkeypatch):
    """Every MTP layer needs all preceding hidden states to populate its own KV."""
    from vllm.config import CUDAGraphMode
    from vllm.v1.spec_decode import llm_base_proposer

    monkeypatch.setattr(
        llm_base_proposer, "set_forward_context", lambda *args, **kwargs: nullcontext()
    )
    calls = []
    initial_hidden = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    sample_indices = torch.tensor([2, 4])
    positions = torch.tensor([0, 1, 2, 0, 1, 0])

    def forward_mtp_layer(**kwargs):
        calls.append(
            {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
        )
        return kwargs["hidden_states"].roll(1, dims=-1) + kwargs["inputs_embeds"]

    def sample(hidden, metadata):
        return hidden.argmax(-1), hidden.softmax(-1)

    proposer = SimpleNamespace(
        num_speculative_tokens=3,
        input_ids=torch.tensor([1, 2, 3, 4, 5, 0]),
        hidden_states=torch.full((6, 8), -99.0),
        inputs_embeds=torch.zeros(6, 8),
        model=SimpleNamespace(
            embed_input_ids=lambda ids: F.one_hot(ids, 8).float() * 0.1,
            forward_mtp_layer=forward_mtp_layer,
        ),
        _sample_draft_tokens=sample,
        _get_positions=lambda num_tokens: positions[:num_tokens],
        _get_slot_mapping=lambda num_tokens, slots: slots,
        vllm_config=None,
    )
    result = llm_base_proposer.SpecDecodeBaseProposer._propose_iquest_mtp_chained(
        proposer,
        initial_hidden,
        initial_hidden[sample_indices],
        sample_indices,
        {},
        SimpleNamespace(slot_mapping=torch.arange(6)),
        None,
        num_tokens=5,
        num_input_tokens=6,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        num_tokens_across_dp=None,
    )

    assert [call["spec_step_idx"] for call in calls] == [1, 2]
    torch.testing.assert_close(calls[0]["hidden_states"][:5], initial_hidden[:5])
    first_layer_output = initial_hidden.roll(1, dims=-1)
    first_layer_output += F.one_hot(torch.tensor([2, 3, 7, 5, 7, 0]), 8) * 0.1
    torch.testing.assert_close(calls[1]["hidden_states"][:5], first_layer_output[:5])
    for call in calls:
        assert call["hidden_states"].shape == (6, 8)
        torch.testing.assert_close(call["positions"], positions)
    torch.testing.assert_close(result, torch.tensor([[7, 0, 1], [7, 0, 1]]))
    assert proposer._last_draft_probs.shape == (2, 3, 8)


@pytest.mark.parametrize("piecewise", [False, True])
def test_mtp_v2_preserves_embedding_addresses_between_layers(monkeypatch, piecewise):
    """Per-layer CUDA graphs must read updated embeddings at the captured address."""
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.multi_module_mtp import speculator

    monkeypatch.setattr(
        speculator, "set_forward_context", lambda *args, **kwargs: nullcontext()
    )
    calls = []

    def forward_mtp_layer(**kwargs):
        calls.append(
            (
                kwargs["spec_step_idx"],
                kwargs["inputs_embeds"].data_ptr(),
                kwargs["inputs_embeds"].clone(),
            )
        )
        assert kwargs["input_ids"] is None
        return kwargs["hidden_states"] + kwargs["inputs_embeds"]

    runner = SimpleNamespace(
        vllm_config=None,
        inputs_embeds=None,
        mtp_inputs_embeds=torch.zeros(4, 8),
        hidden_states=torch.ones(4, 8),
        input_buffers=SimpleNamespace(
            input_ids=torch.tensor([1, 2, 3, 0]), positions=torch.arange(4)
        ),
        model=SimpleNamespace(
            embed_input_ids=lambda ids: F.one_hot(ids, 8).float(),
            forward_mtp_layer=forward_mtp_layer,
        ),
    )
    for step in range(2):
        runner.input_buffers.input_ids[:3] = torch.tensor([1, 2, 3]) + step
        logits_hidden, feedback_hidden = speculator.MultiModuleMTPSpeculator._run_model(
            runner,
            3,
            {},
            {},
            None,
            spec_module_idx=step,
            cudagraph_runtime_mode=(
                CUDAGraphMode.PIECEWISE if piecewise else CUDAGraphMode.NONE
            ),
        )
        expected = 1 + F.one_hot(torch.tensor([1, 2, 3]) + step, 8).float()
        torch.testing.assert_close(logits_hidden, expected)
        torch.testing.assert_close(feedback_hidden, expected)

    assert [call[0] for call in calls] == [0, 1]
    assert calls[0][1] == calls[1][1] == runner.mtp_inputs_embeds.data_ptr()
    assert not torch.equal(calls[0][2], calls[1][2])
