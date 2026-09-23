# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for the IQuestQ1 chat template."""

from collections.abc import Iterable, Sequence

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike


class IQuestQ1ReasoningParser(ReasoningParser):
    """Extract reasoning from the current iQuest Coder assistant turn."""

    _START_TOKEN = "<think>"
    _END_TOKEN = "</think>"
    _ASSISTANT_TOKEN = "<|iquestcoder_assistant|>"
    _TOOL_CALL_START = "<iquestcoder_tool_call>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._thinking_enabled = chat_kwargs.get("enable_thinking", True)

        missing_tokens = [
            token
            for token in (self._START_TOKEN, self._END_TOKEN, self._ASSISTANT_TOKEN)
            if token not in self.vocab
        ]
        if missing_tokens:
            raise ValueError(
                "Tokenizer is missing required iQuest Coder tokens: "
                + ", ".join(missing_tokens)
            )
        self._start_token_id = self.vocab[self._START_TOKEN]
        self._end_token_id = self.vocab[self._END_TOKEN]
        self._assistant_token_id = self.vocab[self._ASSISTANT_TOKEN]
        self._tool_call_token_id = self.vocab.get(self._TOOL_CALL_START)

    @property
    def reasoning_start_str(self) -> str:
        return self._START_TOKEN

    @property
    def reasoning_end_str(self) -> str:
        return self._END_TOKEN

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if not self._thinking_enabled:
            return True
        for token_id in reversed(input_ids):
            if token_id in (self._start_token_id, self._assistant_token_id):
                return False
            if token_id in (self._end_token_id, self._tool_call_token_id):
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return not self._thinking_enabled or any(
            token_id in (self._end_token_id, self._tool_call_token_id)
            for token_id in delta_ids
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self._thinking_enabled:
            return input_ids
        turn_start = 0
        for index in range(len(input_ids) - 1, -1, -1):
            if input_ids[index] in (self._start_token_id, self._assistant_token_id):
                turn_start = index + 1
                break
        for index in range(turn_start, len(input_ids)):
            if input_ids[index] == self._end_token_id:
                return input_ids[index + 1 :]
            if input_ids[index] == self._tool_call_token_id:
                return input_ids[index:]
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self._thinking_enabled:
            return None, model_output
        start_index = model_output.find(self._START_TOKEN)
        if start_index != -1:
            model_output = model_output[start_index + len(self._START_TOKEN) :]
        reasoning, separator, content = model_output.partition(self._END_TOKEN)
        if not separator and start_index == -1:
            return None, model_output or None
        return reasoning or None, (content or None) if separator else None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not delta_text:
            return None
        if (
            not self._thinking_enabled
            or self._end_token_id in previous_token_ids
            or self._tool_call_token_id in previous_token_ids
            or self._TOOL_CALL_START in previous_text
        ):
            return DeltaMessage(content=delta_text)

        delta_text = delta_text.removeprefix(self._START_TOKEN)
        reasoning, separator, content = delta_text.partition(self._END_TOKEN)
        if separator:
            if not reasoning and not content:
                return None
            return DeltaMessage(reasoning=reasoning or None, content=content or None)
        if self._end_token_id in delta_token_ids or not delta_text:
            return None
        # Adaptive thinking can emit a tool call without a closing think token.
        tool_index = delta_text.find(self._TOOL_CALL_START)
        if tool_index != -1:
            return DeltaMessage(
                reasoning=delta_text[:tool_index] or None,
                content=delta_text[tool_index:],
            )
        return DeltaMessage(reasoning=delta_text)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._thinking_enabled:
            return 0
        count = 0
        in_reasoning = True
        for token_id in token_ids:
            if token_id in (self._assistant_token_id, self._start_token_id):
                in_reasoning = True
            elif token_id in (self._end_token_id, self._tool_call_token_id):
                in_reasoning = False
            elif in_reasoning:
                count += 1
        return count
