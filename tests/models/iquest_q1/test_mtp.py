# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the IQuestQ1 draft head's checkpoint and residual contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.models.iquest_q1 import mtp, mtp_recursive


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
        sliding_window=512,
        swa_rope_theta=10000.0,
        fp32_residual_connection=True,
    )
    monkeypatch.setattr(mtp, "IQuestQ1Attention", lambda **kwargs: _Attention())
    monkeypatch.setattr(mtp, "IQuestQ1MoEBlock", lambda **kwargs: nn.SiLU())
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config), quant_config=None
    )


def _rms_norm(x, module):
    return F.rms_norm(
        x.float(), (x.shape[-1],), module.weight.float(), module.variance_epsilon
    ).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mtp_residual_and_scaling_match_training(tiny_config, dtype):
    """The served head retains the unnormalized residual and both output norms."""
    layer = mtp.IQuestQ1MTPInnerLayer(vllm_config=tiny_config).to(dtype=dtype)
    with torch.no_grad():
        for module in layer.modules():
            if isinstance(module, mtp.IQuestQ1RMSNorm):
                module.weight.copy_(torch.tensor([0.8, 1.1, 0.7, 1.3]))
    hidden = torch.tensor([[2.0, -1.0, 4.0, 0.5], [-3.0, 1.0, 0.5, 2.0]]).to(dtype)
    positions = torch.tensor([2, 7])
    normalized = _rms_norm(hidden, layer.attention_norm)
    attn_output = layer.self_attn(positions, normalized)
    residual = hidden + _rms_norm(attn_output, layer.attn_out_norm) * 0.7
    ffn_output = _rms_norm(
        F.silu(_rms_norm(residual, layer.feed_forward_norm)), layer.ffn_out_norm
    )
    expected = residual + ffn_output * 0.4

    torch.testing.assert_close(layer(positions, hidden), expected)


def test_mtp_masks_only_zero_position_embeddings(tiny_config):
    """The shifted BOS embedding is zeroed, while hidden states still contribute."""
    layer = mtp.IQuestQ1MTPLayer(vllm_config=tiny_config)
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


def test_mtp_steps_reuse_first_trained_layer(tiny_config, monkeypatch):
    """A two-head checkpoint still uses only its first head for every draft step."""
    monkeypatch.setattr(
        mtp,
        "VocabParallelEmbedding",
        lambda vocab_size, hidden_size, **kwargs: nn.Embedding(vocab_size, hidden_size),
    )
    torch.manual_seed(0)
    model = mtp.IQuestQ1MultiTokenPredictor(vllm_config=tiny_config)
    ids = torch.tensor([1, 3, 2])
    positions = torch.tensor([0, 1, 2])
    hidden = torch.randn(3, 4)
    embeds = model.embed_input_ids(ids)
    first = model(ids, positions, hidden, spec_step_idx=0)
    second = model(None, positions, first, embeds, spec_step_idx=1)

    assert list(model.layers) == ["42"]
    torch.testing.assert_close(first, model.layers["42"](positions, hidden, embeds))
    torch.testing.assert_close(second, model.layers["42"](positions, first, embeds))
    assert not torch.allclose(second, first)


@pytest.mark.parametrize("fused_experts", [False, True])
@pytest.mark.parametrize("recursive", [False, True])
def test_mtp_loads_offset_layers_and_fused_weights(fused_experts, recursive):
    """Only the first head loads, with correct offsets and expert/QKV shards."""
    params = {}
    checkpoint = []
    expected = {}

    def add_served_parameter(name, value, loader=None):
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
        add_served_parameter(name, value)
        source_name = name.removeprefix("model.") if recursive else name
        checkpoint.append((source_name, value))
    for idx in range(1 if recursive else 2):
        add_parameter = add_served_parameter if idx == 0 else lambda *a, **k: None
        src = "mtp" if recursive else f"mtp_layers.{idx}"
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

    if recursive:
        checkpoint.append(("target_final_norm.weight", torch.ones(4)))
    else:
        checkpoint.append(("model.layers.0.self_attn.q_proj.weight", torch.ones(4, 4)))
    draft = nn.Module()
    draft.config = SimpleNamespace(num_experts=2, intermediate_size=3)
    draft.mtp_start_layer_idx = 42
    draft.named_parameters = lambda: iter(params.items())
    model_cls = mtp_recursive.IQuestQ1MTPRecursive if recursive else mtp.IQuestQ1MTP
    loaded = model_cls.load_weights(draft, checkpoint)

    assert loaded == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(params[name], value, msg=name)

    if recursive:
        with pytest.raises(ValueError, match="missing weights"):
            model_cls.load_weights(draft, checkpoint[1:])
        with pytest.raises(ValueError, match="missing QKV"):
            model_cls.load_weights(
                draft,
                [
                    (name, weight)
                    for name, weight in checkpoint
                    if ".q_proj." not in name
                ],
            )


@pytest.mark.parametrize("fp32_residual", [False, True])
def test_recursive_mtp_keeps_bos_embedding_and_residual_precision(
    tiny_config, fp32_residual
):
    """A recursive step consumes real shifted tokens, including at position zero."""
    tiny_config.model_config.hf_config.fp32_residual_connection = fp32_residual
    torch.manual_seed(7)
    layer = mtp_recursive.IQuestQ1RecursiveLayer(vllm_config=tiny_config).bfloat16()
    positions = torch.tensor([0, 511, 512])
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    embeds = torch.randn(3, 4, dtype=torch.bfloat16)
    projected = F.linear(
        torch.cat([_rms_norm(embeds, layer.enorm), _rms_norm(hidden, layer.hnorm)], -1),
        layer.eh_proj.weight,
    )
    inner = layer.mtp_model_layer
    attention = inner.self_attn(positions, _rms_norm(projected, inner.attention_norm))
    residual = projected.float() if fp32_residual else projected
    residual = residual + _rms_norm(attention, inner.attn_out_norm) * 0.7
    ffn = F.silu(_rms_norm(residual, inner.feed_forward_norm).bfloat16())
    expected = _rms_norm(
        residual + _rms_norm(ffn, inner.ffn_out_norm) * 0.4,
        layer.final_layernorm,
    ).bfloat16()
    actual = layer(positions, hidden, embeds)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    zeroed = embeds.clone()
    zeroed[0] = 0
    assert not torch.equal(actual[0], layer(positions, hidden, zeroed)[0])
