# vllm-iquest-q1

An out-of-tree vLLM plugin for IQuest-Q1, including its recursive MTP draft,
reasoning parser, and tool parser. No patches to vLLM core are required.

## Compatibility and installation

This is a development package, not a published PyPI release. The validated
vLLM baseline is commit
`81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`. Compatibility with other vLLM
releases is not yet established; use the matching vLLM environment.

From the standalone plugin repository root, with a virtual environment
containing the compatible vLLM installation activated:

```bash
uv pip install --no-deps -e .
```

If working inside the original vLLM checkout instead:

```bash
uv pip install --python .venv/bin/python --no-deps -e plugins/iquest-q1
```

The package registers through `vllm.general_plugins` in API and worker
processes. Install the same plugin in every worker environment. By default
vLLM discovers all general plugins. If `VLLM_PLUGINS` is set, include
`iquest_q1` in its comma-separated list.

## Serving

```bash
VLLM_PLUGINS=iquest_q1 vllm serve /path/to/target --served-model-name iquest-q1 --tensor-parallel-size 8 --reasoning-parser iquest_q1 --enable-auto-tool-choice --tool-call-parser iquest_q1
```

The target config uses `model_type: iquest_q1` and architecture
`IQuestQ1ForCausalLM`. Config and model loading do not require
`--trust-remote-code`. A tokenizer requiring custom code is a separate concern.
OpenAI Chat Completions, Responses, and Anthropic endpoints remain provided by
vLLM; the two parsers are shipped with this plugin.

To enable recursive MTP, add:

```bash
--speculative-config '{"method":"eagle","model":"/path/to/draft","num_speculative_tokens":7}'
```

The draft config uses `model_type: iquest_q1_mtp_recursive` with architecture
`IQuestQ1MtpRecursive` (or `IQuestQ1MTPRecursive`). No checkpoint edits are
needed. The plugin adapts the existing recursive draft to vLLM's standard
EAGLE proposer, returning the same normalized hidden state for logits and
recursive feedback. It does not use EAGLE3 auxiliary target states.

The old fork-only `method: mtp_recursive` is replaced by `method: eagle`.
Set `num_speculative_tokens` explicitly, normally to the draft's
`num_draft_slots`. Only serial drafting is supported. Target and draft hidden
sizes and vocabularies must match.

`moe_router_dtype` and `enable_lm_head_fp32` retain their existing behavior.
The draft's LM head defaults to BF16 when serving in BF16; only its own
top-level `enable_lm_head_fp32` enables FP32, not the nested target flag.

## Development

From the standalone plugin repository, using its configured environment:

```bash
.venv/bin/python -m pytest tests -q
uv build --wheel --out-dir dist
```

From the original vLLM checkout:

```bash
.venv/bin/python -m pytest plugins/iquest-q1/tests -q
uv build --wheel plugins/iquest-q1 --out-dir scratch/iquest-plugin-dist
```

GPU tests skip when CUDA is unavailable. The existing legacy parser
interfaces are retained for compatibility; migration does not change the
model's output format.

## Local validation (2026-09-27)

Validated against the vLLM baseline above, using the same core files without
IQuest-specific changes:

- CPU tests: 91 passed, 7 skipped. H200 GPU tests: 193 passed, 16 skipped.
- Ruff, Markdown lint, typos, and wheel packaging passed.
- TP8 compiled serving with target iter_0001180 and the mtp_recursive draft.
- Full HumanEval, 164 tasks, thinking enabled, temperature 0, concurrency 8:
  161/164 (98.17%) both with and without recursive MTP. Each run had one
  length-truncated response at the 16,384-token limit and no API errors.
- Recursive MTP mean acceptance length: 5.5584, including the bonus token.
- Chat Completions, Responses, and Anthropic: streaming and non-streaming
  basic text output passed in both modes (12 checks). This does not cover
  every tool-call or strict protocol-schema scenario.

The migration retains the existing parser behavior. Other vLLM releases,
MATH-500, and single-stream performance were not evaluated in this validation.
