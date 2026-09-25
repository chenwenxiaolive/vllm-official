# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from copy import copy

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .model import IQuestQ1RMSNorm
from .mtp import IQuestQ1MTP, IQuestQ1MTPInnerLayer


class IQuestQ1RecursiveInnerLayer(IQuestQ1MTPInnerLayer):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            draft_sliding_window=config.sliding_window,
            draft_rope_theta=(config.swa_rope_theta if config.sliding_window else None),
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        dtype = hidden_states.dtype
        attention = self.self_attn(positions, self.attention_norm(hidden_states))
        residual = (
            hidden_states.float() if self.fp32_residual_connection else hidden_states
        )
        hidden_states = residual + self.attn_out_norm(attention) * self.attn_out_scale
        normalized = self.feed_forward_norm(hidden_states).to(dtype)
        output = self.mlp(normalized)
        return hidden_states + self.ffn_out_norm(output) * self.ffn_out_scale


class IQuestQ1RecursiveLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.enorm = IQuestQ1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = IQuestQ1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_model_layer = IQuestQ1RecursiveInnerLayer(
            vllm_config=vllm_config, prefix=f"{prefix}.mtp_model_layer"
        )
        self.final_layernorm = IQuestQ1RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, positions, hidden_states, inputs_embeds):
        dtype = self.eh_proj.weight.dtype
        hidden_states = self.eh_proj(
            torch.cat(
                [
                    self.enorm(inputs_embeds).to(dtype),
                    self.hnorm(hidden_states).to(dtype),
                ],
                dim=-1,
            )
        )
        hidden_states = self.mtp_model_layer(positions, hidden_states)
        return self.final_layernorm(hidden_states).to(dtype)


class IQuestQ1RecursivePredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert vllm_config.speculative_config is not None
        target_config = vllm_config.speculative_config.target_model_config.hf_config
        self.layer_idx = target_config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.layers = nn.ModuleDict(
            {
                str(self.layer_idx): IQuestQ1RecursiveLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{self.layer_idx}",
                )
            }
        )

    def forward(self, input_ids, positions, hidden_states, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return self.layers[str(self.layer_idx)](positions, hidden_states, inputs_embeds)


@support_torch_compile
class IQuestQ1MTPRecursive(nn.Module, SupportsPP):
    """One draft layer recursively feeding its normalized hidden state back."""

    has_own_embed_tokens = True
    has_own_lm_head = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        vllm_config = copy(vllm_config)
        assert vllm_config.speculative_config is not None
        vllm_config.model_config = vllm_config.speculative_config.draft_model_config
        vllm_config.quant_config = get_draft_quant_config(vllm_config)
        self.config = vllm_config.model_config.hf_config
        self.model = IQuestQ1RecursivePredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.mtp_start_layer_idx = self.model.layer_idx
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        assert hidden_states is not None
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        qkv_shards: set[str] = set()

        def mapped_weights():
            for name, weight in weights:
                if name == "target_final_norm.weight":
                    # The proposer already supplies post-norm target hidden states.
                    continue
                if name == "embed_tokens.weight":
                    yield "model.embed_tokens.weight", weight
                elif name.startswith("mtp."):
                    for shard in ("q", "k", "v"):
                        if name.endswith(f".self_attn.{shard}_proj.weight"):
                            qkv_shards.add(shard)
                    yield "mtp_layers.0." + name.removeprefix("mtp."), weight
                elif name == "lm_head.weight":
                    yield name, weight
                else:
                    raise ValueError(f"Unexpected recursive MTP weight: {name}")

        loaded = IQuestQ1MTP.load_weights(self, mapped_weights())
        if qkv_shards != {"q", "k", "v"}:
            raise ValueError("Recursive MTP checkpoint is missing QKV projections")
        missing = set(dict(self.named_parameters())) - loaded
        if missing:
            raise ValueError(
                f"Recursive MTP checkpoint is missing weights: {sorted(missing)}"
            )
        return loaded
