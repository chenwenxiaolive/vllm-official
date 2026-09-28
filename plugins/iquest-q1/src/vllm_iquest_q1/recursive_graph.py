# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import fields, replace

import torch


class RecursiveGraphCache:
    """Capture bounded recursive queries, including mixed rejection lengths."""

    def __init__(self, vllm_config):
        self.config = vllm_config
        self.entries = {}
        self.pool = None
        self.captures = 0
        self.replays = 0
        self.logged_metadata = False

    def run(self, model, depth, ids, positions, hidden, per_layer, forward):
        from vllm.config import CUDAGraphMode
        from vllm.distributed import graph_capture
        from vllm.forward_context import set_forward_context
        from vllm.logger import init_logger
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        if ids.device.type != "cuda" or not per_layer:
            return forward(ids, positions, hidden, per_layer)
        spec = self.config.speculative_config
        enforce_eager = (
            spec.enforce_eager
            if spec.enforce_eager is not None
            else self.config.model_config.enforce_eager
        )
        if not self.logged_metadata:
            self.logged_metadata = True
            value = next(iter(per_layer.values()))
            init_logger("vllm.plugins.iquest_q1.recursive_graph").info(
                "Recursive graph configuration: eager=%s metadata=%s "
                "query=%s tokens=%s requests=%s splits=%s scheduler=%s",
                enforce_eager,
                type(value).__name__,
                getattr(value, "max_query_len", None),
                ids.numel(),
                getattr(value, "seq_lens", ids).numel(),
                getattr(value, "max_num_splits", None),
                getattr(value, "scheduler_metadata", None) is not None,
            )
        if enforce_eager:
            return forward(ids, positions, hidden, per_layer)
        metadata = next(iter(per_layer.values()))
        if not isinstance(metadata, FlashAttentionMetadata):
            return forward(ids, positions, hidden, per_layer)
        num_reqs = metadata.seq_lens.numel()
        capacity = spec.num_speculative_tokens + 1
        if (
            metadata.max_query_len > capacity
            or ids.numel() > num_reqs * capacity
            or metadata.use_cascade
            or metadata.scheduler_metadata is not None
            or metadata.max_num_splits < 1
        ):
            return forward(ids, positions, hidden, per_layer)
        key = (
            depth,
            num_reqs,
            ids.numel(),
            metadata.max_query_len,
            metadata.max_num_splits,
            metadata.block_table.shape[1],
        )
        entry = self.entries.get(key)
        if entry is None and len(self.entries) >= 256:
            return forward(ids, positions, hidden, per_layer)
        if entry is None:
            static_metadata = {}
            copies = {}
            for name, value in per_layer.items():
                if id(value) not in copies:
                    copies[id(value)] = replace(
                        value,
                        **{
                            field.name: getattr(value, field.name).clone()
                            for field in fields(value)
                            if isinstance(getattr(value, field.name), torch.Tensor)
                        },
                    )
                    copies[
                        id(value)
                    ].max_seq_len = self.config.model_config.max_model_len
                static_metadata[name] = copies[id(value)]
            entry = {
                "ids": ids.clone(),
                "positions": positions.clone(),
                "hidden": hidden.clone(),
                "metadata": static_metadata,
                "graph": torch.cuda.CUDAGraph(),
            }
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            with graph_capture(ids.device) as context:
                forward(
                    entry["ids"],
                    entry["positions"],
                    entry["hidden"],
                    static_metadata,
                )
                with torch.cuda.graph(entry["graph"], self.pool, stream=context.stream):
                    entry["output"] = forward(
                        entry["ids"],
                        entry["positions"],
                        entry["hidden"],
                        static_metadata,
                    )
            torch.cuda.current_stream(ids.device).wait_stream(context.stream)
            self.entries[key] = entry
            self.captures += 1
            init_logger("vllm.plugins.iquest_q1.recursive_graph").info(
                "Captured recursive draft CUDA Graph: depth=%d requests=%d",
                depth,
                num_reqs,
            )
        else:
            entry["ids"].copy_(ids)
            entry["positions"].copy_(positions)
            entry["hidden"].copy_(hidden)
            copied = set()
            for name, value in per_layer.items():
                static = entry["metadata"][name]
                if id(static) in copied:
                    continue
                copied.add(id(static))
                for field in fields(value):
                    tensor = getattr(value, field.name)
                    if isinstance(tensor, torch.Tensor):
                        getattr(static, field.name).copy_(tensor)
        with set_forward_context(
            entry["metadata"],
            self.config,
            num_tokens=ids.numel(),
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping={
                name: value.slot_mapping for name, value in entry["metadata"].items()
            },
        ):
            entry["graph"].replay()
        self.replays += 1
        return entry["output"]
