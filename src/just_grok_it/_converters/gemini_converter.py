"""
Converter for Google Gemini API format to/from OpenAI API format.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union


def convert_gemini_to_openai(
    contents: Any,
    model: str,
    generation_config: Any = None,
    stream: bool = False,
) -> Dict[str, Any]:
    """
    Convert Gemini generate_content() request to OpenAI chat.completions.create() format.

    Args:
        contents: Gemini-format contents (string, list, or Content objects)
        model: Model to use
        generation_config: Gemini generation configuration
        stream: Whether to stream the response

    Returns:
        OpenAI-format request dictionary
    """
    openai_messages = []

    # Handle different content formats
    if isinstance(contents, str):
        # Simple string input
        openai_messages.append({"role": "user", "content": contents})
    elif isinstance(contents, list):
        # List of contents or conversation history
        for item in contents:
            if isinstance(item, str):
                openai_messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                # Dictionary format
                role = item.get("role", "user")
                # Map Gemini roles to OpenAI roles
                if role == "model":
                    role = "assistant"
                parts = item.get("parts", [])
                content = _extract_text_from_parts(parts)
                openai_messages.append({"role": role, "content": content})
            elif hasattr(item, "role") and hasattr(item, "parts"):
                # Content object
                role = item.role
                if role == "model":
                    role = "assistant"
                content = _extract_text_from_parts(item.parts)
                openai_messages.append({"role": role, "content": content})
    elif hasattr(contents, "parts"):
        # Single Content object
        content = _extract_text_from_parts(contents.parts)
        openai_messages.append({"role": "user", "content": content})

    request = {
        "model": model,
        "messages": openai_messages,
        "stream": stream,
    }

    # Convert generation_config to OpenAI parameters
    if generation_config:
        if (
            hasattr(generation_config, "temperature")
            and generation_config.temperature is not None
        ):
            request["temperature"] = generation_config.temperature
        elif isinstance(generation_config, dict) and "temperature" in generation_config:
            request["temperature"] = generation_config["temperature"]

        if hasattr(generation_config, "top_p") and generation_config.top_p is not None:
            request["top_p"] = generation_config.top_p
        elif isinstance(generation_config, dict) and "top_p" in generation_config:
            request["top_p"] = generation_config["top_p"]

        if (
            hasattr(generation_config, "max_output_tokens")
            and generation_config.max_output_tokens is not None
        ):
            request["max_tokens"] = generation_config.max_output_tokens
        elif (
            isinstance(generation_config, dict)
            and "max_output_tokens" in generation_config
        ):
            request["max_tokens"] = generation_config["max_output_tokens"]

        if (
            hasattr(generation_config, "stop_sequences")
            and generation_config.stop_sequences
        ):
            request["stop"] = generation_config.stop_sequences
        elif (
            isinstance(generation_config, dict)
            and "stop_sequences" in generation_config
        ):
            request["stop"] = generation_config["stop_sequences"]

    return request


def _extract_text_from_parts(parts: Any) -> str:
    """Extract text content from Gemini parts."""
    if isinstance(parts, str):
        return parts

    if isinstance(parts, list):
        text_parts = []
        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif hasattr(part, "text"):
                text_parts.append(part.text)
        return "\n".join(text_parts)

    if hasattr(parts, "text"):
        return parts.text

    return str(parts)


@dataclass
class GeminiPart:
    """Represents a Gemini content part."""

    text: str = ""


@dataclass
class GeminiContent:
    """Represents Gemini content."""

    parts: List[GeminiPart] = field(default_factory=list)
    role: str = "model"


@dataclass
class GeminiCandidate:
    """Represents a Gemini response candidate."""

    content: GeminiContent = field(default_factory=GeminiContent)
    finish_reason: Optional[int] = None  # Gemini uses int enums
    safety_ratings: List[Any] = field(default_factory=list)
    index: int = 0


@dataclass
class GeminiUsageMetadata:
    """Represents Gemini usage metadata."""

    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0


@dataclass
class GeminiGenerateContentResponse:
    """
    Represents a Gemini GenerateContentResponse.

    This mimics the structure returned by GenerativeModel.generate_content()
    """

    candidates: List[GeminiCandidate] = field(default_factory=list)
    usage_metadata: GeminiUsageMetadata = field(default_factory=GeminiUsageMetadata)

    @property
    def text(self) -> str:
        """Get the text from the first candidate."""
        if self.candidates and self.candidates[0].content.parts:
            return self.candidates[0].content.parts[0].text
        return ""

    @property
    def parts(self) -> List[GeminiPart]:
        """Get parts from the first candidate."""
        if self.candidates:
            return self.candidates[0].content.parts
        return []


def convert_openai_to_gemini(
    openai_response: Any, original_model: str
) -> GeminiGenerateContentResponse:
    """
    Convert OpenAI chat.completions response to Gemini GenerateContentResponse format.

    Args:
        openai_response: OpenAI ChatCompletion response object
        original_model: The original model requested

    Returns:
        GeminiGenerateContentResponse object mimicking Gemini's response structure
    """
    choice = openai_response.choices[0] if openai_response.choices else None

    content_text = ""
    finish_reason = None

    if choice:
        content_text = choice.message.content or ""
        # Map OpenAI finish_reason to Gemini finish_reason (int enum)
        fr = choice.finish_reason
        if fr == "stop":
            finish_reason = 1  # STOP
        elif fr == "length":
            finish_reason = 2  # MAX_TOKENS
        elif fr == "content_filter":
            finish_reason = 3  # SAFETY
        else:
            finish_reason = 0  # FINISH_REASON_UNSPECIFIED

    usage = GeminiUsageMetadata(
        prompt_token_count=getattr(openai_response.usage, "prompt_tokens", 0),
        candidates_token_count=getattr(openai_response.usage, "completion_tokens", 0),
        total_token_count=getattr(openai_response.usage, "total_tokens", 0),
    )

    candidate = GeminiCandidate(
        content=GeminiContent(
            parts=[GeminiPart(text=content_text)],
            role="model",
        ),
        finish_reason=finish_reason,
        index=0,
    )

    return GeminiGenerateContentResponse(
        candidates=[candidate],
        usage_metadata=usage,
    )


class GeminiStreamWrapper:
    """Wraps an OpenAI stream to yield Gemini-format responses."""

    def __init__(self, openai_stream: Iterator[Any], model: str):
        self._stream = openai_stream
        self._model = model
        self._accumulated_text = ""

    def __iter__(self) -> Iterator[GeminiGenerateContentResponse]:
        return self

    def __next__(self) -> GeminiGenerateContentResponse:
        try:
            chunk = next(self._stream)
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                self._accumulated_text += text

                return GeminiGenerateContentResponse(
                    candidates=[
                        GeminiCandidate(
                            content=GeminiContent(
                                parts=[GeminiPart(text=text)],
                                role="model",
                            ),
                            index=0,
                        )
                    ],
                )
            # Skip empty chunks
            return self.__next__()
        except StopIteration:
            raise StopIteration


def create_gemini_stream_wrapper(openai_stream: Any, model: str) -> GeminiStreamWrapper:
    """Create a Gemini-compatible stream wrapper."""
    return GeminiStreamWrapper(openai_stream, model)


class AsyncGeminiStreamWrapper:
    """Wraps an async OpenAI stream to yield Gemini-format responses."""

    def __init__(self, client: Any, request: dict, model: str):
        self._client = client
        self._request = request
        self._model = model
        self._stream = None
        self._started = False

    def __aiter__(self) -> AsyncIterator[GeminiGenerateContentResponse]:
        return self

    async def __anext__(self) -> GeminiGenerateContentResponse:
        if not self._started:
            self._started = True
            self._stream = await self._client.chat.completions.create(**self._request)

        try:
            chunk = await self._stream.__anext__()
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content

                return GeminiGenerateContentResponse(
                    candidates=[
                        GeminiCandidate(
                            content=GeminiContent(
                                parts=[GeminiPart(text=text)],
                                role="model",
                            ),
                            index=0,
                        )
                    ],
                )
            return await self.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration


def create_async_gemini_stream_wrapper(
    client: Any, request: dict, model: str
) -> AsyncGeminiStreamWrapper:
    """Create an async Gemini-compatible stream wrapper."""
    return AsyncGeminiStreamWrapper(client, request, model)
