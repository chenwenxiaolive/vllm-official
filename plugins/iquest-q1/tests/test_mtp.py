# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the IQuestQ1 draft head's checkpoint and residual contracts."""

from copy import deepcopy
from functools import wraps
from inspect import signature
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vllm_iquest_q1 import mtp_recursive
from vllm_iquest_q1.configs import IQuestQ1Config, IQuestQ1MTPRecursiveConfig

from vllm.transformers_utils.configs.eagle import EAGLEConfig


def _speculative_config():
    target = IQuestQ1Config()
    draft = IQuestQ1MTPRecursiveConfig(
        target_config=target.to_dict(), architectures=["IQuestQ1MtpRecursive"]
    )
    return SimpleNamespace(
        method="eagle",
        parallel_drafting=False,
        num_speculative_tokens=7,
        target_model_config=SimpleNamespace(hf_config=target),
        draft_model_config=SimpleNamespace(
            hf_config=EAGLEConfig(draft, method="eagle", model_type="eagle")
        ),
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("method", "eagle3", "requires method='eagle'"),
        ("parallel_drafting", True, "serial drafting"),
        ("hidden_size", 4096, "must match"),
        ("vocab_size", 32000, "must match"),
        ("model_type", "llama", "IQuestQ1 target"),
    ],
)
def test_recursive_plugin_rejects_incompatible_target_or_proposer(field, value, error):
    config = _speculative_config()
    mtp_recursive.validate_recursive_draft(config)
    obj = config if hasattr(config, field) else config.target_model_config.hf_config
    setattr(obj, field, value)
    with pytest.raises(ValueError, match=error):
        mtp_recursive.validate_recursive_draft(config)


def test_recursive_eagle_returns_same_normalized_state_for_logits_and_feedback():
    """The EAGLE tuple must preserve the previous single-tensor feedback rule."""
    expected = torch.randn(3, 4)
    draft = SimpleNamespace(model=lambda *args: expected)
    logits_state, feedback = mtp_recursive.IQuestQ1MTPRecursive.forward(
        draft, torch.tensor([1, 2, 3]), torch.arange(3), torch.randn(3, 4)
    )
    assert logits_state is expected
    assert feedback is expected


class _Attention(nn.Module):
    def forward(self, positions, hidden_states, spec_step_idx=0):
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
        moe_router_dtype="float32",
        rms_norm_eps=1e-5,
        first_layer_attn_out_scale=0.7,
        first_layer_ffn_out_scale=0.4,
        attn_out_scale=1.3,
        ffn_out_scale=1.7,
        num_hidden_layers=42,
        vocab_size=8,
        sliding_window=512,
        swa_rope_theta=10000.0,
        fp32_residual_connection=True,
    )
    monkeypatch.setattr(
        mtp_recursive, "IQuestQ1Attention", lambda **kwargs: _Attention()
    )
    monkeypatch.setattr(mtp_recursive, "IQuestQ1MoEBlock", lambda **kwargs: nn.SiLU())
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7,
            target_model_config=SimpleNamespace(hf_config=config),
        ),
    )


def _rms_norm(x, module):
    return F.rms_norm(
        x.float(), (x.shape[-1],), module.weight.float(), module.variance_epsilon
    ).to(x.dtype)


def test_recursive_steps_reuse_one_draft_layer(tiny_config, monkeypatch):
    """Recursive steps reuse the draft layer and feed back normalized states."""
    monkeypatch.setattr(
        mtp_recursive,
        "VocabParallelEmbedding",
        lambda vocab_size, hidden_size, **kwargs: nn.Embedding(vocab_size, hidden_size),
    )
    torch.manual_seed(0)
    model = mtp_recursive.IQuestQ1RecursivePredictor(vllm_config=tiny_config)
    ids = torch.tensor([1, 3, 2])
    positions = torch.tensor([0, 1, 2])
    hidden = torch.randn(3, 4)
    embeds = model.embed_tokens(ids)
    first = model(ids, positions, hidden)
    second = model(None, positions, first, embeds)

    assert list(model.layers) == ["42"]
    torch.testing.assert_close(first, model.layers["42"](positions, hidden, embeds))
    torch.testing.assert_close(second, model.layers["42"](positions, first, embeds))
    assert not torch.allclose(second, first)


@pytest.mark.parametrize("fused_experts", [False, True])
def test_recursive_loads_offset_layers_and_fused_weights(fused_experts):
    """Standalone weights load with correct layer offsets and expert/QKV shards."""
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
        source_name = name.removeprefix("model.")
        checkpoint.append((source_name, value))
    src = "mtp"
    dst = "model.layers.42"
    qkv = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
    add_served_parameter(
        f"{dst}.mtp_model_layer.self_attn.qkv_proj.weight", qkv, qkv_loader
    )
    for shard, value in zip(("q", "k", "v"), qkv):
        checkpoint.append(
            (f"{src}.mtp_model_layer.self_attn.{shard}_proj.weight", value)
        )

    w13 = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)
    w2 = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    for name, value in (("w13_weight", w13), ("w2_weight", w2)):
        add_served_parameter(
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
                        (
                            f"{src}.mtp_model_layer.mlp.experts.{expert_id}."
                            f"{projection}.weight"
                        ),
                        value,
                    )
                )
    router = torch.full((2, 4), 0.25)
    add_served_parameter(f"{dst}.mtp_model_layer.mlp.gate.weight", router)
    checkpoint.append((f"{src}.mtp_model_layer.mlp.router.weight", router))
    norm = torch.full((4,), 0.5)
    add_served_parameter(f"{dst}.final_layernorm.weight", norm)
    checkpoint.append((f"{src}.final_layernorm.weight", norm))

    checkpoint.append(("target_final_norm.weight", torch.ones(4)))
    draft = nn.Module()
    draft.config = SimpleNamespace(num_experts=2, intermediate_size=3)
    draft.mtp_start_layer_idx = 42
    draft.named_parameters = lambda: iter(params.items())
    model_cls = mtp_recursive.IQuestQ1MTPRecursive
    loaded = model_cls.load_weights(draft, checkpoint)

    assert loaded == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(params[name], value, msg=name)

    with pytest.raises(ValueError, match="missing weights"):
        model_cls.load_weights(draft, checkpoint[1:])
    with pytest.raises(ValueError, match="missing QKV"):
        model_cls.load_weights(
            draft,
            [(name, weight) for name, weight in checkpoint if ".q_proj." not in name],
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


class _SharedRecursiveReference:
    """FP64 oracle that recomputes the committed prefix before each proposal."""

    def __init__(self):
        generator = torch.Generator().manual_seed(31)
        self.embedding = torch.randn(19, 4, generator=generator, dtype=torch.float64)
        self.head = torch.randn(4, 19, generator=generator, dtype=torch.float64)
        self.cache = [torch.empty(0, 4, dtype=torch.float64)]

    def layer(self, ids, hidden, start=0, depth=1):
        assert depth == 1
        query = (self.embedding[ids] + hidden).tanh()
        keys = torch.cat((self.cache[0][:start], query))
        self.cache[0] = keys
        mask = (
            torch.arange(len(keys))[None, :]
            <= (torch.arange(len(query)) + start)[:, None]
        )
        scores = (query @ keys.T / 2).masked_fill(~mask, -torch.inf)
        return (query + scores.softmax(-1) @ keys).tanh()

    def recompute(self, ids, target_hidden, branch):
        oracle = deepcopy(self)
        shifted = torch.cat((ids[1:], branch[:1]))
        hidden = oracle.layer(shifted, target_hidden)
        logits = [hidden[-1] @ oracle.head]
        for depth in range(1, len(branch) - 1):
            hidden = oracle.layer(
                branch[depth : depth + 1], hidden[-1:], len(ids) + depth - 1
            )
            logits.append(hidden[-1] @ oracle.head)
        return torch.stack(logits)


@pytest.mark.parametrize("accepted_counts", [(4, 1), (8, 8), (1, 1), (2, 7)])
def test_recursive_proposer_tracks_requests_across_reordering_and_padding(
    accepted_counts,
):
    """Batch reordering and rollback must preserve each request's verified KV."""
    from vllm_iquest_q1.recursive_proposer import RecursiveProposer

    refs = {key: _SharedRecursiveReference() for key in ["a", "b"]}
    runner = SimpleNamespace(
        requests={"a": object(), "b": object()},
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
    )
    proposer = SimpleNamespace(
        num_speculative_tokens=7,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)),
        dtype=torch.float64,
        runner=runner,
        _sample_draft_tokens=lambda hidden, _: (
            (hidden @ refs["a"].head).argmax(-1),
            None,
        ),
    )
    runtime = RecursiveProposer(None, proposer)
    generator = torch.Generator().manual_seed(25)
    histories = {}
    for key in runner.requests:
        histories[key] = (
            torch.randint(0, 19, (15,), generator=generator),
            torch.randn(15, 4, generator=generator, dtype=torch.float64),
        )

    def forward(depth, jobs, common):
        result = {}
        for i, job in enumerate(jobs):
            if job is not None:
                start, ids, hidden = job
                result[i] = refs[runner.input_batch.req_ids[i]].layer(
                    ids, hidden, start, depth
                )
        return result

    runtime._forward = forward

    def propose(chunks, starts, seeds, padding):
        order = runner.input_batch.req_ids
        lengths = [len(chunks[key][0]) for key in order]
        offsets = [0]
        for count in lengths:
            offsets.append(offsets[-1] + count)
        return runtime.propose(
            num_speculative_tokens=7,
            target_token_ids=torch.cat([chunks[key][0] for key in order]),
            target_hidden_states=torch.cat([chunks[key][1] for key in order]),
            target_positions=torch.cat(
                [
                    torch.arange(starts[key], starts[key] + lengths[i])
                    for i, key in enumerate(order)
                ]
            ),
            next_token_ids=torch.tensor(seeds),
            sampling_metadata=None,
            num_rejected_tokens_gpu=torch.tensor(padding),
            common_attn_metadata=SimpleNamespace(
                query_start_loc_cpu=torch.tensor(offsets)
            ),
        )

    first = propose(histories, {"a": 0, "b": 0}, [3, 8], [0, 0])
    branches = {
        "a": torch.cat((torch.tensor([3]), first[0])),
        "b": torch.cat((torch.tensor([8]), first[1])),
    }
    runner.input_batch.req_ids = ["b", "a"]
    chunks = {}
    for key, accepted in zip(["b", "a"], accepted_counts):
        extra = torch.randn(8, 4, generator=generator, dtype=torch.float64)
        chunks[key] = branches[key], extra
        ids, hidden = histories[key]
        histories[key] = (
            torch.cat((ids, branches[key][:accepted])),
            torch.cat((hidden, extra[:accepted])),
        )
    second = propose(
        chunks, {"a": 15, "b": 15}, [4, 9], [8 - n for n in accepted_counts]
    )
    for i, key in enumerate(runner.input_batch.req_ids):
        branch = torch.cat((torch.tensor([[4, 9][i]]), second[i]))
        expected = refs[key].recompute(*histories[key], branch).argmax(-1)
        assert second[i].tolist() == expected.tolist()
    del runner.requests["a"]
    runner.input_batch.req_ids = ["b"]
    ids = torch.tensor([4])
    hidden = torch.randn(1, 4, generator=generator, dtype=torch.float64)
    propose({"b": (ids, hidden)}, {"b": len(histories["b"][0])}, [2], [0])

    # A preempted request may resume from a previously finalized prefix block.
    original_ids, original_hidden = histories["b"]
    all_ids = torch.cat((original_ids, ids))
    all_hidden = torch.cat((original_hidden, hidden))
    refs["b"].cache = [value[:8].clone() for value in refs["b"].cache]
    cached_prefix = refs["b"].cache[0].clone()
    start = 8
    for end in [9, 12, len(all_ids)]:
        seed = int(all_ids[end]) if end < len(all_ids) else 2
        resumed = propose(
            {"b": (all_ids[start:end], all_hidden[start:end])},
            {"b": start},
            [seed],
            [0],
        )
        branch = torch.cat((torch.tensor([seed]), resumed[0]))
        expected = (
            refs["b"].recompute(all_ids[:end], all_hidden[:end], branch).argmax(-1)
        )
        assert resumed[0].tolist() == expected.tolist()
        torch.testing.assert_close(
            refs["b"].cache[0][:8], cached_prefix, rtol=0, atol=0
        )
        start = end


def test_recursive_v2_dispatch_accepts_runner_positional_arguments():
    """The serving runner and warmup both pass sampler inputs positionally."""
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

    calls = []
    speculator = object.__new__(EagleSpeculator)
    speculator.model = SimpleNamespace(
        _iquest_recursive_proposal=True,
        propose_draft=lambda proposer, **kwargs: calls.append(kwargs) or "draft",
    )
    inputs = [object() for _ in range(11)]
    assert speculator.propose(*inputs) == "draft"
    assert calls[0]["input_batch"] is inputs[0]
    assert calls[0]["last_hidden_states"] is inputs[3]
    assert calls[0]["num_rejected"] is inputs[6]
    assert calls[0]["seeds"] is inputs[10]


@pytest.mark.parametrize("version", [1, 2])
def test_recursive_hooks_preserve_other_models_and_warmup(version):
    """Only marked drafts use the plugin, with runner arguments preserved."""
    from vllm_iquest_q1.runtime_hooks import _install_proposer_hook

    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

    real_class = EagleProposer if version == 1 else EagleSpeculator
    original = real_class.propose.__wrapped__
    calls = []
    fallback = []

    class Proposer:
        def __init__(self, vllm_config=None, device=None, runner=None):
            pass

        @wraps(original)
        def propose(self, *args, **kwargs):
            fallback.append((args, kwargs))
            return "original"

    _install_proposer_hook(Proposer, keep_runner=version == 1)
    installed = Proposer.propose, Proposer.__init__
    _install_proposer_hook(Proposer, keep_runner=version == 1)
    assert installed == (Proposer.propose, Proposer.__init__)
    runner = object()
    proposer = Proposer(None, None, runner)
    if version == 1:
        assert proposer.runner is runner
        assert Proposer(runner=runner).runner is runner
    proposer.model = SimpleNamespace(
        _iquest_recursive_proposal=True,
        propose_draft=lambda proposer, **kwargs: calls.append(kwargs) or "recursive",
    )
    inputs = tuple(object() for _ in range(8 if version == 1 else 11))
    bound = signature(original).bind(proposer, *inputs).arguments
    bound.pop("self")
    assert proposer.propose(*inputs) == "recursive"
    assert proposer.propose(**bound) == "recursive"
    assert calls == [bound, bound]
    if version == 2:
        assert proposer.propose(*inputs, dummy_run=True) == "original"
        assert proposer.propose(*inputs, None, True) == "original"
        assert len(calls) == 2
    del proposer.model._iquest_recursive_proposal
    assert proposer.propose(*inputs) == "original"
    assert fallback[-1] == (inputs, {})
    assert proposer.propose(**bound) == "original"
    assert fallback[-1] == ((), bound)
    assert len(calls) == 2


def test_recursive_hooks_reject_incompatible_proposer_before_modifying_it():
    from vllm_iquest_q1.runtime_hooks import _install_proposer_hook

    class Proposer:
        def propose(self, new_api):
            pass

    original = Proposer.propose, Proposer.__init__
    with pytest.raises(RuntimeError, match="incompatible"):
        _install_proposer_hook(Proposer, keep_runner=True)
    assert original == (Proposer.propose, Proposer.__init__)


def test_recursive_v2_uses_one_seed_per_request_with_mixed_prefill(monkeypatch):
    """Column-shaped sampled tokens must not broadcast across request rows."""
    import numpy as np
    from vllm_iquest_q1.recursive_proposer import RecursiveProposer, RecursiveProposerV2

    speculator = SimpleNamespace(
        vllm_config=None,
        dtype=torch.float64,
        num_speculative_steps=7,
        max_num_reqs=4,
        attn_groups=[[SimpleNamespace(kv_cache_group_id=0)]],
        block_tables=SimpleNamespace(
            kernel_block_sizes=[128], input_block_tables=[torch.zeros(4, 8)]
        ),
        _copy_request_inputs=lambda *args: None,
    )
    runtime = RecursiveProposerV2(None, speculator)
    batch = SimpleNamespace(
        req_ids=["decode", "prefill"],
        idx_mapping=torch.tensor([3, 1]),
        idx_mapping_np=np.array([3, 1]),
        num_reqs=2,
        query_start_loc_np=np.array([0, 1, 3], dtype=np.int32),
        seq_lens=torch.tensor([22, 10]),
        input_ids=torch.tensor([2, 3, 4]),
        positions=torch.tensor([21, 8, 9]),
    )
    calls = []
    monkeypatch.setattr(
        RecursiveProposer, "propose", lambda self, **kwargs: calls.append(kwargs)
    )
    runtime.propose(
        input_batch=batch,
        last_hidden_states=torch.zeros(3, 4),
        num_sampled=torch.tensor([1, 0]),
        num_rejected=torch.tensor([0, 0]),
        last_sampled=torch.tensor([[10], [11], [12], [13]]),
        next_prefill_tokens=torch.tensor([[20, 21, 22, 23]]),
        temperature=torch.zeros(4),
        seeds=torch.arange(4),
    )
    assert calls[0]["next_token_ids"].tolist() == [13, 21]
    assert calls[0]["common_attn_metadata"].query_start_loc_cpu.tolist() == [0, 1, 3]
    assert runtime.proposer.sample_positions.tolist() == [22, 10]


def test_recursive_lookahead_hook_preserves_other_drafters():
    """Only recursive MTP needs a full speculation window at prefill boundaries."""
    from vllm_iquest_q1.runtime_hooks import _install_prefill_lookahead_hook

    class Config:
        speculative_config = None

        @property
        def num_prefill_lookahead_tokens(self):
            return 1 if self.speculative_config else 0

    config = Config()
    _install_prefill_lookahead_hook(Config)
    installed = Config.num_prefill_lookahead_tokens
    _install_prefill_lookahead_hook(Config)
    assert Config.num_prefill_lookahead_tokens is installed
    assert config.num_prefill_lookahead_tokens == 0
    config.speculative_config = _speculative_config()
    assert config.num_prefill_lookahead_tokens == 7
    config.speculative_config.draft_model_config.hf_config = SimpleNamespace(
        model_type="eagle"
    )
    assert config.num_prefill_lookahead_tokens == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("num_splits", [1, 32])
@torch.inference_mode()
def test_recursive_graph_replay_refreshes_inputs_and_attention_metadata(
    monkeypatch, num_splits
):
    """Reordered requests and reused graph pools must not retain old tensor data."""
    from contextlib import contextmanager, nullcontext

    from vllm_iquest_q1.recursive_graph import RecursiveGraphCache

    import vllm.distributed
    import vllm.forward_context
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    @contextmanager
    def capture(device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            yield SimpleNamespace(stream=stream)

    monkeypatch.setattr(vllm.distributed, "graph_capture", capture)
    monkeypatch.setattr(
        vllm.forward_context, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            enforce_eager=False, num_speculative_tokens=7
        ),
        model_config=SimpleNamespace(max_model_len=100, enforce_eager=False),
    )
    cache = RecursiveGraphCache(config)

    def forward(ids, positions, hidden, per_layer):
        metadata = per_layer["layer"]
        return torch.sin(
            hidden
            + ids[:, None] * 0.01
            + positions[:, None] * 0.02
            + metadata.slot_mapping[:, None] * 0.03
            + metadata.seq_lens.sum() * 0.04
            + metadata.query_start_loc.sum() * 0.06
            + metadata.block_table.sum() * 0.05
            + metadata.max_num_splits * 0.07
        )

    def run(depth, shift, lengths=None):
        lengths = lengths or [depth, depth]
        count = sum(lengths)
        ids = torch.arange(count, device="cuda") + shift
        positions = ids + 10
        hidden = torch.arange(count * 4, device="cuda").reshape(-1, 4).float()
        metadata = FlashAttentionMetadata(
            num_actual_tokens=count,
            max_query_len=max(lengths),
            query_start_loc=torch.tensor([0, lengths[0], count], device="cuda"),
            max_seq_len=30,
            seq_lens=torch.tensor([20, 30], device="cuda") + shift,
            block_table=torch.tensor([[0, 1], [2, 3]], device="cuda") + shift,
            slot_mapping=ids + shift,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            max_num_splits=num_splits,
        )
        per_layer = {"layer": metadata}
        result = cache.run(None, depth, ids, positions, hidden, per_layer, forward)
        torch.testing.assert_close(
            result, forward(ids, positions, hidden, per_layer), rtol=0, atol=0
        )

    run(2, 0)
    run(3, 1)
    run(2, 7)
    assert cache.captures == 2 and cache.replays == 3
    run(2, 4, [1, 3])
    run(2, 6, [4, 1])
    run(2, 2, [1, 1])
    assert cache.captures == 5 and cache.replays == 6
    num_splits = 4
    run(2, 0)
    assert cache.captures == 6 and cache.replays == 7
    num_splits = 0
    run(2, 0)
    assert cache.captures == 6 and cache.replays == 7
    num_splits = 4
    config.speculative_config.enforce_eager = True
    run(2, 11)
    assert cache.captures == 6 and cache.replays == 7

    config.speculative_config.enforce_eager = None
    config.model_config.enforce_eager = True
    run(2, 12)
    assert cache.captures == 6 and cache.replays == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("table_width", [8, 2048])
@torch.inference_mode()
def test_shared_graph_refreshes_cache_and_sampler_inputs(monkeypatch, table_width):
    """Replayed proposals must follow changed requests, positions and seeds."""
    from contextlib import contextmanager

    from vllm_iquest_q1.shared_cache import SharedRecursiveGraphCache

    import vllm.distributed
    import vllm.forward_context
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    active = {}

    @contextmanager
    def context(metadata, *args, **kwargs):
        active["metadata"] = metadata["layer"]
        yield

    @contextmanager
    def capture(device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            yield SimpleNamespace(stream=stream)

    monkeypatch.setattr(vllm.distributed, "graph_capture", capture)
    monkeypatch.setattr(vllm.forward_context, "set_forward_context", context)
    depth_count = 7
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False, max_model_len=128),
        speculative_config=SimpleNamespace(enforce_eager=False),
    )

    def metadata(common, draft_index=0):
        return [], {
            "layer": FlashAttentionMetadata(
                num_actual_tokens=common.num_actual_tokens,
                max_query_len=common.max_query_len,
                query_start_loc=common.query_start_loc,
                max_seq_len=common.max_seq_len,
                seq_lens=common.seq_lens,
                block_table=common.block_table_tensor,
                slot_mapping=common.slot_mapping,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
                max_num_splits=32,
            )
        }

    def model(ids, positions, hidden, spec_step_idx=0):
        m = active["metadata"]
        return torch.sin(
            hidden
            + ids[:, None] * 0.01
            + positions[:, None] * 0.02
            + m.slot_mapping[:, None] * 0.001
            + m.seq_lens.sum() * 0.003
            + m.block_table.sum() * 0.0001
            + m.query_start_loc.sum() * 0.004
        )

    speculator = SimpleNamespace(
        current_draft_step=torch.zeros((), device="cuda", dtype=torch.int32),
        temperature=torch.zeros(8, device="cuda"),
        seeds=torch.arange(8, device="cuda"),
        draft_logits=torch.zeros((8, 7, 4), device="cuda"),
        draft_watermarker=None,
        acceptance_estimator=None,
    )

    position_weights = torch.tensor([0.001, -0.002, 0.003, -0.004], device="cuda")
    seed_weights = torch.tensor([0.004, 0.003, -0.002, -0.001], device="cuda")

    def sample(hidden, positions, indices, temperature, seeds, step, logits):
        value = hidden + positions[:, None] * position_weights
        value = value + seeds[indices][:, None] * seed_weights
        logits[indices, step.expand(len(hidden))] = value
        return value.argmax(-1)

    speculator.sample_draft = sample
    adapter = SimpleNamespace(
        speculator=speculator,
        vllm_config=config,
        block_size=16,
        build_per_group_and_layer_attn_metadata=metadata,
        runner=SimpleNamespace(input_batch=None),
    )
    runtime = SimpleNamespace(
        proposer=adapter, model=SimpleNamespace(model=model), depth=depth_count
    )
    cache = SharedRecursiveGraphCache()

    for turn, (identity, width) in enumerate([(0, 1), (1, 8), (2, 3), (0, 8), (1, 1)]):
        length = 23 + 3 * turn
        ids = torch.arange(width + 2, device="cuda", dtype=torch.int64)[2:] + turn
        hidden = torch.arange((width + 1) * 8, device="cuda").float() * 0.01
        hidden = hidden.reshape(width + 1, 8)[1:, :4]
        idx = torch.tensor([identity], device="cuda")
        adapter.runner.input_batch = SimpleNamespace(idx_mapping=idx)
        adapter.sample_positions = torch.tensor(
            [length], device="cuda", dtype=torch.int32
        )
        block = torch.arange(table_width, device="cuda").reshape(1, -1).int() + turn
        common = SimpleNamespace(block_table_tensor=block)
        initial = torch.tensor([identity + turn], device="cuda")
        speculator.seeds.add_(turn)
        actual = cache.run_shared(runtime, ids, hidden, initial, common)
        assert actual is not None
        actual_branch = actual.clone()
        actual_logits = speculator.draft_logits[idx].clone()
        branch = initial[:, None]
        next_ids = torch.cat((ids[1:], initial))
        for depth in range(7):
            n = width if depth == 0 else 1
            end = length + depth
            positions = torch.arange(end - n, end, device="cuda")
            slots = block[0, positions // 16].long() * 16 + positions % 16
            expected = torch.sin(
                hidden
                + next_ids[:, None] * 0.01
                + positions[:, None] * 0.02
                + slots[:, None] * 0.001
                + torch.tensor([end], device="cuda", dtype=torch.int32).sum() * 0.003
                + block.sum() * 0.0001
                + torch.tensor([0, n], device="cuda", dtype=torch.int32).sum() * 0.004
            )[-1:]
            token = sample(
                expected,
                adapter.sample_positions + depth,
                idx,
                speculator.temperature,
                speculator.seeds,
                torch.tensor(depth, device="cuda"),
                speculator.draft_logits,
            )
            branch = torch.cat((branch, token[:, None]), dim=1)
            hidden, next_ids = expected, token
        assert torch.equal(actual_branch, branch)
        assert torch.equal(actual_logits, speculator.draft_logits[idx])
    assert cache.captures == 3 and cache.replays == 5
    config.speculative_config.enforce_eager = True
    assert cache.run_shared(runtime, ids, hidden, initial, common) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("length,width", [(22, 1), (25, 8), (126, 3), (128, 8)])
@torch.inference_mode()
def test_shared_graph_metadata_refreshes_and_clamps_context(length, width):
    """Replays must use new block IDs and never address past the context limit."""
    from vllm_iquest_q1.shared_cache import _shared_metadata

    seq = torch.tensor([length], dtype=torch.int32, device="cuda")
    table = torch.arange(8, dtype=torch.int32, device="cuda")
    positions = torch.empty((7, 8), dtype=torch.int64, device="cuda")
    slots = torch.empty_like(positions)
    ends = torch.empty(7, dtype=torch.int32, device="cuda")

    def launch():
        _shared_metadata[(7,)](
            seq, table, positions, slots, ends, width, 8, 16, 128, 8, num_warps=1
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    for advance in [0, 1, 4]:
        current = min(length + advance, 128)
        seq.fill_(current)
        table.add_(3)
        graph.replay()
        expected_ends = (current + torch.arange(7, device="cuda")).clamp(max=128)
        torch.testing.assert_close(ends, expected_ends.int(), rtol=0, atol=0)
        for depth in range(7):
            count = width if depth == 0 else 1
            expected = expected_ends[depth] - count + torch.arange(count, device="cuda")
            torch.testing.assert_close(
                positions[depth, :count], expected, rtol=0, atol=0
            )
            expected_slots = table[expected // 16].long() * 16 + expected % 16
            torch.testing.assert_close(
                slots[depth, :count], expected_slots, rtol=0, atol=0
            )
