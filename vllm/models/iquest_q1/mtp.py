# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP
from .model import (
    IQuestQ1Attention,
    IQuestQ1MoEBlock,
    IQuestQ1RMSNorm,
)
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)

logger = init_logger(__name__)


def get_spec_layer_idx_from_name(weight_name: str) -> int:
    spec_layer_idx = weight_name.split(".")[1]
    return int(spec_layer_idx)


class IQuestQ1MTPInnerLayer(nn.Module):
    """Decoder block of the served MTP head, which uses sandwich norms."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size

        self.self_attn = IQuestQ1Attention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
            is_mtp_layer=True,
        )
        self.mlp = IQuestQ1MoEBlock(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.attention_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_out_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.ffn_out_norm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_out_scale = getattr(config, "first_layer_attn_out_scale", 1.0)
        self.ffn_out_scale = getattr(config, "first_layer_ffn_out_scale", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        norm_hidden_states = self.attention_norm(hidden_states)
        attn_output = self.self_attn(
            positions=positions, hidden_states=norm_hidden_states
        )
        h = hidden_states + self.attn_out_norm(attn_output) * self.attn_out_scale

        ffn_out = self.mlp(self.feed_forward_norm(h))
        return h + self.ffn_out_norm(ffn_out) * self.ffn_out_scale


class IQuestQ1MTPLayer(nn.Module):
    """One MTP module: enorm/hnorm + eh_proj + inner decoder block + final LN."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.enorm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_model_layer = IQuestQ1MTPInnerLayer(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mtp_model_layer",
        )
        self.final_layernorm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states = self.mtp_model_layer(positions, hidden_states)
        return self.final_layernorm(hidden_states)


@support_torch_compile
class IQuestQ1MTPCompiledLayer(IQuestQ1MTPLayer):
    """Compiled entry point for the served MTP layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(positions, previous_hidden_states, inputs_embeds)


class IQuestQ1MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        # A checkpoint may ship several trained MTP heads; only the first one
        # is served. The ModuleDict keeps the layer-indexed parameter names the
        # checkpoint uses.
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx): IQuestQ1MTPCompiledLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx}",
                )
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None, (
                "IQuestQ1 MTP requires input_ids when inputs_embeds is None"
            )
            inputs_embeds = self.embed_tokens(input_ids)
        return self.layers[str(self.mtp_start_layer_idx)](
            positions, previous_hidden_states, inputs_embeds
        )


@support_torch_compile
class IQuestQ1MTP(nn.Module, SupportsPP):
    """Draft head for IQuestQ1.

    The whole draft head resides on the last pipeline stage.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.model = IQuestQ1MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.mtp_start_layer_idx = self.config.num_hidden_layers
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

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
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mtp_prefix = "mtp_layers."
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        enable_sink_attention = getattr(self.config, "enable_sink_attention", False)

        def _load_into(param_name: str, weight: torch.Tensor) -> None:
            param = params_dict[param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded_params.add(param_name)

        for name, loaded_weight in weights:
            # Shared weights loaded into the draft head.
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                _load_into(name, loaded_weight)
                continue

            if not name.startswith(mtp_prefix):
                continue

            # Only the first trained MTP head is served.
            spec_layer_idx = get_spec_layer_idx_from_name(name)
            if spec_layer_idx != 0:
                continue

            # Rewrite mtp_layers.0.<rest> -> model.layers.<mtp_start>.<rest>
            name = name.replace(
                f"mtp_layers.{spec_layer_idx}",
                f"model.layers.{spec_layer_idx + self.mtp_start_layer_idx}",
            )

            # Fused QKV: match on the exact ".<component>.weight" suffix.
            for param_name, weight_name, shard_id in stacked_params_mapping:
                suffix = f".{weight_name}.weight"
                if not name.endswith(suffix):
                    continue
                mapped = name[: -len(suffix)] + f".{param_name}.weight"
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                expert_weight_matched = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    expert_weight_matched = True
                    mapped = name.replace(weight_name, param_name)
                    if not is_pp_missing_parameter(mapped, self):
                        param = params_dict[mapped]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded_params.add(mapped)
                    break

                if expert_weight_matched:
                    continue

                # Sonic-MoE fused expert tensors: match exact ".experts.fc" /
                # ".experts.proj" component suffixes.
                if name.endswith(".mlp.experts.fc"):
                    mapped = name[: -len(".fc")] + ".routed_experts.w13_weight"
                    param = params_dict[mapped]
                    weight_loader = param.weight_loader
                    for expert_id in range(self.config.num_experts):
                        weight_loader(
                            param,
                            loaded_weight[expert_id][: self.config.intermediate_size],
                            mapped,
                            shard_id="w1",
                            expert_id=expert_id,
                        )
                        weight_loader(
                            param,
                            loaded_weight[expert_id][self.config.intermediate_size :],
                            mapped,
                            shard_id="w3",
                            expert_id=expert_id,
                        )
                    loaded_params.add(mapped)
                    continue
                if name.endswith(".mlp.experts.proj"):
                    mapped = name[: -len(".proj")] + ".routed_experts.w2_weight"
                    param = params_dict[mapped]
                    weight_loader = param.weight_loader
                    for expert_id in range(self.config.num_experts):
                        weight_loader(
                            param,
                            loaded_weight[expert_id],
                            mapped,
                            shard_id="w2",
                            expert_id=expert_id,
                        )
                    loaded_params.add(mapped)
                    continue

                if not enable_sink_attention and name.endswith(".sink_k"):
                    logger.warning_once("sink attention feature is disabled")
                    continue

                # Sonic-MoE router naming: ".mlp.router.weight" -> ".mlp.gate.weight".
                if name.endswith(".mlp.router.weight"):
                    name = name[: -len(".router.weight")] + ".gate.weight"

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning_once("Unexpected MTP weight skipped: %s", name)
                    continue
                _load_into(name, loaded_weight)

        return loaded_params
