# vllm-iquest-q1

A vLLM plugin for IQuest-Q1, with recursive MTP, reasoning parsing, and tool calling.

## Installation

Tested with vLLM commit `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`. Other
versions are not yet verified. This package is not yet published on PyPI.

With a compatible vLLM environment activated, install from this repository:

```bash
uv pip install --no-deps .
```

vLLM loads the plugin automatically. Install it in every worker environment.
If `VLLM_PLUGINS` is set, include `iquest_q1` in the list.

## Serving

```bash
vllm serve IQuestLab/IQuest-Q1 \
  --served-model-name iquest-q1 \
  --tensor-parallel-size 8 \
  --reasoning-parser iquest_q1 \
  --enable-auto-tool-choice \
  --tool-call-parser iquest_q1
```

This downloads the target model from the Hugging Face repository
`IQuestLab/IQuest-Q1`, reusing the local Hugging Face cache when available.
Use a local model directory instead to load existing weights. For online
downloads, do not set `HF_HUB_OFFLINE=1` or `VLLM_USE_MODELSCOPE=1`.
If `HF_ENDPOINT` is set, use `https://huggingface.co` to download from the
official Hub. Authenticate with `hf auth login` for a private or gated repo.

The plugin installs runtime hooks for vLLM's V1 `EagleProposer` and V2
`EagleSpeculator`; no vLLM source changes are required. Only IQuest recursive
drafts use the custom proposer. These hooks depend on internal vLLM APIs, so
use the tested revision above when installing the plugin.

To enable recursive MTP, append:

```bash
--enable-prefix-caching \
--speculative-config '{
  "method": "eagle",
  "model": "/path/to/draft",
  "num_speculative_tokens": 5,
  "draft_sample_method": "probabilistic",
  "rejection_sample_method": "standard",
  "enforce_eager": false
}'
```

The draft's `model` is configured separately. Use its local directory or its
own Hugging Face repository ID; the target repository does not select a draft
automatically.

Only serial drafting is supported; target and draft hidden sizes and
vocabularies must match. One draft layer recursively reuses its weights and a
single KV cache per request. The first step refreshes newly verified rows from
target hidden states; later steps append one row each. The next round rewrites
accepted rows, and causal sequence lengths exclude rejected rows.

The recursive path supports TP with DP=1 and PP=1, including prefix caching.
External KV transfer is not supported. No cache-mode environment variable is
needed.

With FlashAttention, the V2 runner captures all proposal steps and sampling in
one CUDA graph for single-request decode. Larger batches use per-forward graphs
for bounded queries; long prefill and unsupported attention metadata stay eager.
Set the draft's `enforce_eager` to `true` to disable its graphs. Batch-Invariant
mode is not required. Draft RMSNorm is fused while preserving its FP32 reduction
order; the target keeps its own graph configuration.
GPU validation uses the V2 model runner.

## Development

```bash
.venv/bin/python -m pytest tests -q
uv build --wheel --out-dir dist
```
