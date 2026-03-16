"""
Converter for Anthropic API format to/from OpenAI API format.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional


def convert_anthropic_to_openai(
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Convert Anthropic messages.create() request to OpenAI chat.completions.create() format.

    Args:
        messages: Anthropic-format messages
        model: Model to use
        max_tokens: Maximum tokens to generate
        system: System prompt (Anthropic-style)
        temperature: Sampling temperature
        top_p: Top-p sampling
        stop_sequences: Stop sequences
        stream: Whether to stream the response

    Returns:
        OpenAI-format request dictionary
    """
    openai_messages = []

    # Add system message if provided
    if system:
        openai_messages.append({"role": "system", "content": system})

    # Convert Anthropic messages to OpenAI format
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Anthropic uses "content" which can be a string or list of content blocks
        if isinstance(content, list):
            # Handle content blocks (text, image, etc.)
            openai_content_parts = []
            text_parts = []

            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        # Convert Anthropic image format to OpenAI format
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            media_type = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            openai_content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{data}"
                                    },
                                }
                            )
                        elif source.get("type") == "url":
                            openai_content_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": source.get("url", "")},
                                }
                            )
                elif isinstance(block, str):
                    text_parts.append(block)

            # If we have images, use the multimodal format
            if openai_content_parts:
                if text_parts:
                    openai_content_parts.insert(
                        0, {"type": "text", "text": "\n".join(text_parts)}
                    )
                content = openai_content_parts
            else:
                content = "\n".join(text_parts)

        openai_messages.append({"role": role, "content": content})

    request = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if temperature is not None:
        request["temperature"] = temperature

    if top_p is not None:
        request["top_p"] = top_p

    if stop_sequences:
        request["stop"] = stop_sequences

    return request


@dataclass
class AnthropicTextBlock:
    """Represents an Anthropic text content block."""

    type: str = "text"
    text: str = ""


@dataclass
class AnthropicUsage:
    """Represents Anthropic usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AnthropicMessage:
    """
    Represents an Anthropic Message response.

    This mimics the structure returned by anthropic.messages.create()
    """

    id: str = ""
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicTextBlock] = field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = field(default_factory=AnthropicUsage)


def convert_openai_to_anthropic(
    openai_response: Any, original_model: str
) -> AnthropicMessage:
    """
    Convert OpenAI chat.completions response to Anthropic Message format.

    Args:
        openai_response: OpenAI ChatCompletion response object
        original_model: The original model requested (for the response)

    Returns:
        AnthropicMessage object mimicking Anthropic's response structure
    """
    choice = openai_response.choices[0] if openai_response.choices else None

    content_text = ""
    stop_reason = None

    if choice:
        content_text = choice.message.content or ""
        # Map OpenAI finish_reason to Anthropic stop_reason
        finish_reason = choice.finish_reason
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "content_filter":
            stop_reason = "content_filter"
        else:
            stop_reason = finish_reason

    usage = AnthropicUsage(
        input_tokens=getattr(openai_response.usage, "prompt_tokens", 0),
        output_tokens=getattr(openai_response.usage, "completion_tokens", 0),
    )

    return AnthropicMessage(
        id=f"msg_{openai_response.id}"
        if openai_response.id
        else f"msg_{uuid.uuid4().hex}",
        type="message",
        role="assistant",
        content=[AnthropicTextBlock(type="text", text=content_text)],
        model=original_model,
        stop_reason=stop_reason,
        usage=usage,
    )


@dataclass
class AnthropicStreamEvent:
    """Represents an Anthropic streaming event."""

    type: str
    message: Optional[AnthropicMessage] = None
    index: int = 0
    content_block: Optional[AnthropicTextBlock] = None
    delta: Optional[Dict[str, Any]] = None


class AnthropicStreamWrapper:
    """Wraps an OpenAI stream to yield Anthropic-format events."""

    def __init__(self, openai_stream: Iterator[Any], model: str):
        self._stream = openai_stream
        self._model = model
        self._message_id = f"msg_{uuid.uuid4().hex}"
        self._started = False
        self._content_started = False
        self._accumulated_text = ""

    def __iter__(self) -> Iterator[AnthropicStreamEvent]:
        return self

    def __next__(self) -> AnthropicStreamEvent:
        if not self._started:
            self._started = True
            return AnthropicStreamEvent(
                type="message_start",
                message=AnthropicMessage(
                    id=self._message_id,
                    model=self._model,
                    content=[],
                ),
            )

        if not self._content_started:
            self._content_started = True
            return AnthropicStreamEvent(
                type="content_block_start",
                index=0,
                content_block=AnthropicTextBlock(type="text", text=""),
            )

        try:
            chunk = next(self._stream)
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                self._accumulated_text += text
                return AnthropicStreamEvent(
                    type="content_block_delta",
                    index=0,
                    delta={"type": "text_delta", "text": text},
                )
            # Skip empty chunks
            return self.__next__()
        except StopIteration:
            raise StopIteration


def create_anthropic_stream_wrapper(
    openai_stream: Any, model: str
) -> AnthropicStreamWrapper:
    """Create an Anthropic-compatible stream wrapper."""
    return AnthropicStreamWrapper(openai_stream, model)


class AsyncAnthropicStreamWrapper:
    """Wraps an async OpenAI stream to yield Anthropic-format events."""

    def __init__(self, client: Any, request: dict):
        self._client = client
        self._request = request
        self._model = request.get("model", "")
        self._message_id = f"msg_{uuid.uuid4().hex}"
        self._started = False
        self._content_started = False
        self._stream = None

    def __aiter__(self) -> AsyncIterator[AnthropicStreamEvent]:
        return self

    async def __anext__(self) -> AnthropicStreamEvent:
        if not self._started:
            self._started = True
            self._stream = await self._client.chat.completions.create(**self._request)
            return AnthropicStreamEvent(
                type="message_start",
                message=AnthropicMessage(
                    id=self._message_id,
                    model=self._model,
                    content=[],
                ),
            )

        if not self._content_started:
            self._content_started = True
            return AnthropicStreamEvent(
                type="content_block_start",
                index=0,
                content_block=AnthropicTextBlock(type="text", text=""),
            )

        try:
            chunk = await self._stream.__anext__()
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                return AnthropicStreamEvent(
                    type="content_block_delta",
                    index=0,
                    delta={"type": "text_delta", "text": text},
                )
            return await self.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration


def create_async_anthropic_stream_wrapper(
    client: Any, request: dict
) -> AsyncAnthropicStreamWrapper:
    """Create an async Anthropic-compatible stream wrapper."""
    return AsyncAnthropicStreamWrapper(client, request)
