# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-branch recursive drafting, following OpenRT's cache scheduling."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _stage_shared_inputs(
    IDS,
    HIDDEN,
    NEXT,
    SEQ,
    INDEX,
    TABLE,
    OUT_IDS,
    OUT_HIDDEN,
    OUT_NEXT,
    OUT_SEQ,
    OUT_INDEX,
    OUT_TABLE,
    SAMPLE_POSITIONS,
    CURRENT_STEP,
    N: tl.constexpr,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hidden = tl.load(HIDDEN + i // H * STRIDE + i % H, i < N * H, other=0)
    tl.store(OUT_HIDDEN + i, hidden, i < N * H)
    ids = tl.load(IDS + i + 1, i < N - 1, other=0)
    next_token = tl.load(NEXT)
    tl.store(OUT_IDS + i, tl.where(i == N - 1, next_token, ids), i < N)
    blocks = tl.load(TABLE + i, i < TABLE_WIDTH, other=0)
    tl.store(OUT_TABLE + i, blocks, i < TABLE_WIDTH)
    seq = tl.load(SEQ)
    tl.store(SAMPLE_POSITIONS + i, seq + i, i < K)
    tl.store(OUT_SEQ + i, seq, i == 0)
    tl.store(OUT_INDEX + i, tl.load(INDEX), i == 0)
    tl.store(OUT_NEXT + i, next_token, i == 0)
    tl.store(CURRENT_STEP + i, K - 1, i == 0)


@triton.jit
def _shared_metadata(
    SEQ,
    TABLE,
    POS,
    SLOTS,
    ENDS,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_MODEL_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    d = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    n = tl.where(d == 0, N, 1)
    end = tl.minimum(tl.load(SEQ).to(tl.int64) + d, MAX_MODEL_LEN)
    p = end - n + i
    b = tl.load(TABLE + p // BLOCK_SIZE, i < n, other=0).to(tl.int64)
    tl.store(POS + d * WIDTH + i, p, i < n)
    tl.store(SLOTS + d * WIDTH + i, b * BLOCK_SIZE + p % BLOCK_SIZE, i < n)
    tl.store(ENDS + d, end)


class SharedRecursiveGraphCache:
    """Replay one committed-prefix update and serial single-token proposals."""

    def __init__(self):
        self.entries = {}
        self.pool = None
        self.captures = 0
        self.replays = 0

    def run_shared(self, runtime, ids, hidden, next_tokens, common):
        adapter = runtime.proposer
        if ids.device.type != "cuda" or not hasattr(adapter, "speculator"):
            return None
        speculator = adapter.speculator
        eager = adapter.vllm_config.speculative_config.enforce_eager
        if eager is None:
            eager = adapter.vllm_config.model_config.enforce_eager
        if eager or ids.numel() > runtime.depth + 1 or len(next_tokens) != 1:
            return None
        if (
            getattr(speculator, "draft_watermarker", None) is not None
            or getattr(speculator, "acceptance_estimator", None) is not None
        ):
            return None
        n = ids.numel()
        key = (n, common.block_table_tensor.shape[1], ids.dtype, hidden.dtype)
        if key not in self.entries:
            if len(self.entries) >= 16:
                return None
            self.entries[key] = self.create(runtime, ids, hidden, next_tokens, common)
        entry = self.entries[key]
        if entry is None:
            return None
        table_width = common.block_table_tensor.shape[1]
        _stage_shared_inputs[(triton.cdiv(max(hidden.numel(), table_width), 1024),)](
            ids,
            hidden,
            next_tokens,
            adapter.sample_positions,
            adapter.runner.input_batch.idx_mapping,
            common.block_table_tensor,
            entry["ids"],
            entry["hidden"],
            entry["next"],
            entry["seq_lens"],
            entry["idx_mapping"],
            entry["block_table"],
            entry["sample_positions"],
            speculator.current_draft_step,
            n,
            hidden.shape[-1],
            hidden.stride(0),
            table_width,
            runtime.depth,
            1024,
        )
        if "graph" not in entry:
            self._capture(entry, runtime)
        entry["graph"].replay()
        self.replays += 1
        adapter.step = runtime.depth
        return entry["branch"]

    def create(self, runtime, ids, hidden, next_tokens, common):
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        adapter = runtime.proposer
        k = runtime.depth
        n = ids.numel()
        device = ids.device
        entry = {
            "ids": torch.empty_like(ids),
            "hidden": torch.empty_like(hidden),
            "next": torch.empty_like(next_tokens.reshape(-1)),
            "branch": torch.empty((1, k + 1), dtype=next_tokens.dtype, device=device),
            "seq_lens": torch.empty_like(adapter.sample_positions),
            "sample_positions": torch.empty(
                (k,), dtype=adapter.sample_positions.dtype, device=device
            ),
            "steps": torch.arange(
                k, dtype=adapter.speculator.current_draft_step.dtype, device=device
            ),
            "idx_mapping": torch.empty_like(adapter.runner.input_batch.idx_mapping),
            "block_table": torch.empty_like(common.block_table_tensor[:1]),
            "positions": torch.empty((k, k + 1), dtype=torch.int64, device=device),
            "slots": torch.empty((k, k + 1), dtype=torch.int64, device=device),
            "ends": torch.empty((k,), dtype=torch.int32, device=device),
            "metadata": [],
        }
        for d in range(k):
            width = n if d == 0 else 1
            qcpu = torch.tensor([0, width], dtype=torch.int32)
            common_d = CommonAttentionMetadata(
                query_start_loc=qcpu.to(device),
                query_start_loc_cpu=qcpu,
                seq_lens=entry["ends"][d : d + 1],
                num_reqs=1,
                num_actual_tokens=width,
                max_query_len=width,
                max_seq_len=adapter.vllm_config.model_config.max_model_len,
                block_table_tensor=entry["block_table"],
                slot_mapping=entry["slots"][d, :width],
                positions=entry["positions"][d, :width],
                seq_lens_cpu_upper_bound=torch.tensor(
                    [adapter.vllm_config.model_config.max_model_len], dtype=torch.int32
                ),
            )
            _, layers = adapter.build_per_group_and_layer_attn_metadata(
                common_d, draft_index=0
            )
            if not layers or any(
                not isinstance(m, FlashAttentionMetadata)
                or m.use_cascade
                or m.scheduler_metadata is not None
                or m.max_num_splits < 1
                or m.dcp_context_kv_lens is not None
                or m.mm_prefix_query_range_tensor is not None
                or m.rswa_prefix_lens is not None
                for m in layers.values()
            ):
                return None
            entry["metadata"].append(layers)
        return entry

    def _execute(self, entry, runtime):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        adapter = runtime.proposer
        speculator = adapter.speculator
        k = runtime.depth
        n = entry["ids"].numel()
        _shared_metadata[(k,)](
            entry["seq_lens"],
            entry["block_table"],
            entry["positions"],
            entry["slots"],
            entry["ends"],
            n,
            k + 1,
            adapter.block_size,
            adapter.vllm_config.model_config.max_model_len,
            triton.next_power_of_2(k + 1),
            num_warps=1,
        )
        ids = entry["ids"]
        tokens_by_depth = [entry["next"]]
        h = entry["hidden"]
        for d in range(k):
            width = n if d == 0 else 1
            layers = entry["metadata"][d]
            with set_forward_context(
                layers,
                adapter.vllm_config,
                num_tokens=width,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping={name: m.slot_mapping for name, m in layers.items()},
            ):
                h = runtime.model.model(
                    ids, entry["positions"][d, :width], h, spec_step_idx=0
                )[-1:]
            tokens = speculator.sample_draft(
                h,
                entry["sample_positions"][d : d + 1],
                entry["idx_mapping"],
                speculator.temperature,
                speculator.seeds,
                entry["steps"][d],
                speculator.draft_logits,
            )
            tokens_by_depth.append(tokens)
            ids = tokens
        torch.cat(tokens_by_depth, out=entry["branch"].view(-1))
        return h

    def _capture(self, entry, runtime):
        from vllm.distributed import graph_capture
        from vllm.logger import init_logger

        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        entry["graph"] = torch.cuda.CUDAGraph()
        with graph_capture(entry["branch"].device) as context:
            self._execute(entry, runtime)
            with torch.cuda.graph(entry["graph"], self.pool, stream=context.stream):
                entry["outputs"] = self._execute(entry, runtime)
        torch.cuda.current_stream(entry["branch"].device).wait_stream(context.stream)
        self.captures += 1
        init_logger("vllm.plugins.iquest_q1.shared_cache").info(
            "Captured shared recursive graph: requests=%d depth=%d",
            entry["branch"].shape[0],
            runtime.depth,
        )


def propose_shared(
    runtime,
    *,
    num_speculative_tokens,
    target_token_ids,
    target_positions,
    target_hidden_states,
    next_token_ids,
    sampling_metadata,
    common_attn_metadata,
    num_rejected_tokens_gpu=None,
    **kwargs,
):
    if num_speculative_tokens != runtime.depth:
        raise ValueError("Recursive MTP requires a fixed speculation depth")
    # Depth zero rewrites accepted rows from target states; branch KV is temporary.
    adapter = runtime.proposer
    adapter._last_draft_probs = None
    hidden = target_hidden_states.to(adapter.dtype)
    req_ids = adapter.runner.input_batch.req_ids
    offsets = common_attn_metadata.query_start_loc_cpu.tolist()
    rejected = (
        [0] * len(req_ids)
        if num_rejected_tokens_gpu is None
        else num_rejected_tokens_gpu.tolist()
    )
    if not hasattr(runtime, "shared_graphs"):
        runtime.shared_graphs = SharedRecursiveGraphCache()
    if len(req_ids) == 1:
        start, end = offsets[0], offsets[1] - rejected[0]
        if end <= start:
            raise RuntimeError("Recursive drafting requires a verified input token")
        branch = runtime.shared_graphs.run_shared(
            runtime,
            target_token_ids[start:end],
            hidden[start:end],
            next_token_ids,
            common_attn_metadata,
        )
        if branch is not None:
            return branch[:, 1:].contiguous()
    maximum = adapter.vllm_config.model_config.max_model_len
    positions = target_positions[: offsets[-1]].tolist()
    jobs = []
    ends = []
    for i in range(len(req_ids)):
        start, end = offsets[i], offsets[i + 1] - rejected[i]
        if end <= start:
            raise RuntimeError("Recursive drafting requires a verified input token")
        jobs.append(
            (
                positions[start],
                torch.cat(
                    (target_token_ids[start + 1 : end], next_token_ids[i : i + 1])
                ),
                hidden[start:end],
            )
        )
        ends.append(positions[start] + end - start)
    branch = next_token_ids.reshape(len(req_ids), 1)
    probabilities = []
    for d in range(runtime.depth):
        outputs = runtime._forward(1, jobs, common_attn_metadata)
        last = torch.stack([outputs[i][-1] for i in range(len(req_ids))])
        tokens, probs = adapter._sample_draft_tokens(last, sampling_metadata)
        branch = torch.cat((branch, tokens[:, None]), dim=1)
        if probs is not None:
            probabilities.append(probs)
        jobs = [
            (min(ends[i] + d, maximum - 1), tokens[i : i + 1], last[i : i + 1])
            for i in range(len(req_ids))
        ]
    if probabilities:
        adapter._last_draft_probs = torch.stack(probabilities, dim=1).contiguous()
    return branch[:, 1:].contiguous()
