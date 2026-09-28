# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recursive drafting with one shared draft KV cache per request."""

import torch


class RecursiveProposer:
    def __init__(self, model, proposer):
        from .recursive_graph import RecursiveGraphCache

        self.model = model
        self.proposer = proposer
        self.depth = proposer.num_speculative_tokens
        self.graphs = RecursiveGraphCache(getattr(proposer, "vllm_config", None))

    def _forward(self, depth, jobs, common):
        from itertools import accumulate

        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
        from vllm.v1.attention.backend import CommonAttentionMetadata

        active = [(i, job) for i, job in enumerate(jobs) if job is not None]
        if not active:
            return {}
        device = common.seq_lens.device
        lengths = [len(job[1]) for _, job in active]
        offsets = list(accumulate(lengths, initial=0))
        index_data = async_tensor_h2d(
            [p for _, job in active for p in range(job[0], job[0] + len(job[1]))]
            + [i for i, length in enumerate(lengths) for _ in range(length)]
            + [i for i, _ in active],
            device=device,
            dtype=torch.int64,
        )
        positions, rows, indices = index_data.split(
            [offsets[-1], offsets[-1], len(active)]
        )
        block_table = common.block_table_tensor
        if len(active) != len(jobs):
            block_table = block_table.index_select(0, indices)
        else:
            block_table = block_table[: len(active)]
        block_size = self.proposer.block_size
        slots = (
            block_table[rows, positions // block_size].long() * block_size
            + positions % block_size
        )
        seq_lengths = [job[0] + len(job[1]) for _, job in active]
        sequence_data_cpu = torch.tensor(
            offsets + seq_lengths, dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        query_start_cpu, seq_lens_cpu = sequence_data_cpu.split(
            [len(offsets), len(seq_lengths)]
        )
        sequence_data = async_tensor_h2d(sequence_data_cpu, device=device)
        query_start, seq_lens = sequence_data.split([len(offsets), len(seq_lengths)])
        metadata = CommonAttentionMetadata(
            query_start_loc=query_start,
            query_start_loc_cpu=query_start_cpu,
            seq_lens=seq_lens,
            num_reqs=len(active),
            num_actual_tokens=offsets[-1],
            max_query_len=max(lengths),
            max_seq_len=max(seq_lengths),
            block_table_tensor=block_table,
            slot_mapping=slots,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            positions=positions,
        )
        _, per_layer = self.proposer.build_per_group_and_layer_attn_metadata(
            metadata, draft_index=depth - 1
        )

        def forward(ids, positions, hidden, metadata):
            with set_forward_context(
                metadata,
                self.proposer.vllm_config,
                num_tokens=ids.numel(),
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping={
                    name: value.slot_mapping for name, value in metadata.items()
                },
            ):
                return self.model.model(ids, positions, hidden, spec_step_idx=depth - 1)

        output = self.graphs.run(
            self.model,
            depth,
            torch.cat([job[1] for _, job in active]),
            positions,
            torch.cat([job[2] for _, job in active]),
            per_layer,
            forward,
        )
        return {
            i: output[offsets[j] : offsets[j + 1]] for j, (i, _) in enumerate(active)
        }

    def propose(
        self,
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
        from .shared_cache import propose_shared

        return propose_shared(
            self,
            num_speculative_tokens=num_speculative_tokens,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            sampling_metadata=sampling_metadata,
            common_attn_metadata=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            **kwargs,
        )


class _V2Adapter:
    def __init__(self, speculator):
        from types import SimpleNamespace

        self.speculator = speculator
        self.vllm_config = speculator.vllm_config
        self.dtype = speculator.dtype
        self.num_speculative_tokens = speculator.num_speculative_steps
        self.runner = SimpleNamespace(input_batch=None)
        self.groups = [group for groups in speculator.attn_groups for group in groups]
        group_ids = {group.kv_cache_group_id for group in self.groups}
        if len(group_ids) != 1:
            raise ValueError("Recursive MTP requires one draft KV cache group")
        self.group_id = group_ids.pop()
        self.block_size = speculator.block_tables.kernel_block_sizes[self.group_id]
        self.step = 0

    def build_per_group_and_layer_attn_metadata(self, common, draft_index=0):
        per_layer = {}
        for group in self.groups:
            metadata = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common, draft_index=draft_index
            )
            per_layer.update({name: metadata for name in group.layer_names})
        return [], per_layer

    def _sample_draft_tokens(self, hidden, _):
        speculator = self.speculator
        speculator.current_draft_step.fill_(self.step)
        tokens = speculator.sample_draft(
            hidden,
            self.sample_positions + self.step,
            self.runner.input_batch.idx_mapping,
            speculator.temperature,
            speculator.seeds,
            speculator.current_draft_step,
            speculator.draft_logits,
        )
        self.step += 1
        return tokens, None


class RecursiveProposerV2(RecursiveProposer):
    def __init__(self, model, speculator):
        super().__init__(model, _V2Adapter(speculator))

    def propose(
        self,
        *,
        input_batch,
        last_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        **kwargs,
    ):
        from types import SimpleNamespace

        adapter = self.proposer
        speculator = adapter.speculator
        adapter.runner.input_batch = input_batch
        speculator._copy_request_inputs(
            input_batch.num_reqs, input_batch.idx_mapping, temperature, seeds
        )
        indices = input_batch.idx_mapping.long()
        prefill = next_prefill_tokens.reshape(-1, speculator.max_num_reqs)[0]
        next_tokens = torch.where(
            num_sampled.reshape(-1) > 0,
            last_sampled.reshape(-1)[indices],
            prefill[indices],
        )
        adapter.step = 0
        adapter.sample_positions = input_batch.seq_lens - num_rejected
        common = SimpleNamespace(
            seq_lens=input_batch.seq_lens,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            block_table_tensor=speculator.block_tables.input_block_tables[
                adapter.group_id
            ][: input_batch.num_reqs],
        )
        return super().propose(
            num_speculative_tokens=self.depth,
            target_token_ids=input_batch.input_ids,
            target_positions=input_batch.positions,
            target_hidden_states=last_hidden_states,
            next_token_ids=next_tokens,
            sampling_metadata=None,
            common_attn_metadata=common,
            num_rejected_tokens_gpu=num_rejected,
        )
