# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## Offline Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## IQuestQ1 Recursive Drafts

IQuestQ1 supports standalone `mtp_strict_draft` checkpoints trained with
`draft_type: eagle3`. Select `mtp_recursive` and supply the draft checkpoint:

```bash
vllm serve /path/to/iquest-q1 \
    --tensor-parallel-size 8 \
    --speculative-config '{"method":"mtp_recursive","model":"/path/to/draft","num_speculative_tokens":7}'
```

The native `method="mtp"` path is not supported for IQuestQ1. Embedded
`mtp_layers.*` weights in target checkpoints are ignored.

The EAGLE proposer runs one physical draft layer repeatedly. Each step consumes
the previous step's token and normalized hidden state and appends its own KV.
The initial hidden state is the target's final normalized output. The draft's
`target_config` supplies architecture dimensions; its own `sliding_window` and
`swa_rope_theta` control draft attention. FP32 residual connections are honored
when requested by the checkpoint.

When omitted, `num_speculative_tokens` defaults to `num_draft_slots`. Depths
beyond the training depth are allowed with a warning. Parallel drafting,
diffusion drafts, dense FFN variants, and shared-KV variants are unsupported.
The draft and target must have matching hidden sizes and vocabularies.

## Notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
