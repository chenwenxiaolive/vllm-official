# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import ClientDisconnect, Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)

_IMAGE_PLACEHOLDER = (
    "[Image omitted: this server does not process images. "
    "Use text-only tools or provide a text description instead.]"
)
_PATHS = {
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
}


def _replace_images(content: object) -> int:
    if not isinstance(content, list):
        return 0

    count = 0
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("image", "image_url"):
            content[index] = {
                "type": "text",
                "text": _IMAGE_PLACEHOLDER,
            }
            count += 1
        elif block.get("type") == "tool_result":
            count += _replace_images(block.get("content"))
    return count


class IgnoreImagesMiddleware:
    """Opt-in image placeholders for text-only Chat and Messages API servers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"].removeprefix(scope.get("root_path", "")).rstrip("/")
            not in _PATHS
            or Headers(scope=scope).get("content-type", "").split(";")[0].strip()
            != "application/json"
        ):
            await self.app(scope, receive, send)
            return

        scope = {**scope, "headers": list(scope["headers"])}
        body_received = False

        async def receive_body() -> Message:
            nonlocal body_received
            if body_received:
                return await receive()
            body_received = True

            # Defer buffering until after downstream authentication.
            try:
                body = await Request(scope, receive).body()
            except ClientDisconnect:
                return {"type": "http.disconnect"}

            count = 0
            try:
                payload = json.loads(body)
            except ValueError:
                pass
            else:
                messages = (
                    payload.get("messages") if isinstance(payload, dict) else None
                )
                if isinstance(messages, list):
                    for message in messages:
                        if isinstance(message, dict):
                            count += _replace_images(message.get("content"))
                if count:
                    body = json.dumps(payload).encode("utf-8")
                    headers = MutableHeaders(raw=scope["headers"])
                    headers["content-length"] = str(len(body))
                    if "transfer-encoding" in headers:
                        del headers["transfer-encoding"]
                    logger.warning_once(
                        "IgnoreImagesMiddleware is replacing image inputs with "
                        "text placeholders; image contents are not processed."
                    )

            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, receive_body, send)
