# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import wraps
from inspect import signature


def _install_proposer_hook(proposer_class: type, *, keep_runner: bool) -> None:
    original_propose = proposer_class.propose
    if getattr(original_propose, "_iquest_recursive_hook", False):
        return

    propose_signature = signature(original_propose)
    required = (
        {
            "self",
            "num_speculative_tokens",
            "target_token_ids",
            "target_positions",
            "target_hidden_states",
            "next_token_ids",
            "sampling_metadata",
            "common_attn_metadata",
            "num_rejected_tokens_gpu",
        }
        if keep_runner
        else {
            "self",
            "input_batch",
            "last_hidden_states",
            "num_sampled",
            "num_rejected",
            "last_sampled",
            "next_prefill_tokens",
            "temperature",
            "seeds",
            "dummy_run",
        }
    )
    if not required.issubset(propose_signature.parameters):
        raise RuntimeError(
            f"IQuest-Q1 recursive MTP is incompatible with "
            f"{proposer_class.__module__}.{proposer_class.__name__}.propose"
            f"{propose_signature}; use the vLLM revision documented by the plugin"
        )

    if keep_runner:
        original_init = proposer_class.__init__
        init_signature = signature(original_init)
        if "runner" not in init_signature.parameters:
            raise RuntimeError(
                "IQuest-Q1 recursive MTP requires an EagleProposer runner argument"
            )

        @wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            arguments = init_signature.bind(self, *args, **kwargs).arguments
            self.runner = arguments.get("runner")

        proposer_class.__init__ = init

    @wraps(original_propose)
    def propose(self, *args, **kwargs):
        model = self.model
        if not getattr(model, "_iquest_recursive_proposal", False):
            return original_propose(self, *args, **kwargs)
        arguments = propose_signature.bind(self, *args, **kwargs).arguments
        if arguments.get("dummy_run", False):
            return original_propose(self, *args, **kwargs)
        arguments.pop("self")
        return model.propose_draft(self, **arguments)

    propose._iquest_recursive_hook = True
    proposer_class.propose = propose


def _install_prefill_lookahead_hook(config_class: type) -> None:
    original = config_class.num_prefill_lookahead_tokens.fget
    if getattr(original, "_iquest_recursive_hook", False):
        return

    @wraps(original)
    def num_prefill_lookahead_tokens(self):
        spec = self.speculative_config
        draft = getattr(spec, "draft_model_config", None)
        config = getattr(draft, "hf_config", None)
        config = getattr(config, "model", config)
        if getattr(config, "model_type", None) == "iquest_q1_mtp_recursive":
            return spec.num_speculative_tokens
        return original(self)

    num_prefill_lookahead_tokens._iquest_recursive_hook = True
    config_class.num_prefill_lookahead_tokens = property(num_prefill_lookahead_tokens)


def install_recursive_proposer_hooks() -> None:
    """Route only IQuest recursive drafts through their shared-cache proposer."""
    from vllm.config import VllmConfig
    from vllm.v1.spec_decode.eagle import EagleProposer
    from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

    _install_prefill_lookahead_hook(VllmConfig)
    _install_proposer_hook(EagleProposer, keep_runner=True)
    _install_proposer_hook(EagleSpeculator, keep_runner=False)
    original_init = EagleSpeculator.init_cudagraph_manager
    if not getattr(original_init, "_iquest_recursive_hook", False):

        @wraps(original_init)
        def init_cudagraph_manager(self, cudagraph_mode):
            from vllm.config import CUDAGraphMode

            if getattr(self.model, "_iquest_recursive_proposal", False):
                cudagraph_mode = CUDAGraphMode.NONE
            return original_init(self, cudagraph_mode)

        init_cudagraph_manager._iquest_recursive_hook = True
        EagleSpeculator.init_cudagraph_manager = init_cudagraph_manager
