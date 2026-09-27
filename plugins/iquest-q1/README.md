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
vllm serve /path/to/target \
  --served-model-name iquest-q1 \
  --tensor-parallel-size 8 \
  --reasoning-parser iquest_q1 \
  --enable-auto-tool-choice \
  --tool-call-parser iquest_q1
```

To enable recursive MTP, append:

```bash
--speculative-config '{
  "method": "eagle",
  "model": "/path/to/draft",
  "num_speculative_tokens": 7
}'
```

Set `num_speculative_tokens` to the draft's `num_draft_slots`. Only serial
drafting is supported; target and draft hidden sizes and vocabularies must match.

## Development

```bash
.venv/bin/python -m pytest tests -q
uv build --wheel --out-dir dist
```
