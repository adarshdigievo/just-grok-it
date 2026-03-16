"""
Mock xAI API server for integration testing.

This FastAPI server mimics xAI's OpenAI-compatible API for testing
that requests are properly redirected and formatted.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Mock xAI API Server")

# Store received requests for verification
received_requests: List[Dict[str, Any]] = []


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/v1/requests")
async def get_requests():
    """Get all received requests (for test verification)."""
    return {"requests": received_requests}


@app.delete("/v1/requests")
async def clear_requests():
    """Clear all received requests."""
    received_requests.clear()
    return {"status": "cleared"}


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Mock chat completions endpoint."""
    body = await request.json()

    # Store the request for verification
    received_requests.append(
        {
            "endpoint": "/v1/chat/completions",
            "authorization": authorization,
            "body": body,
            "timestamp": time.time(),
        }
    )

    # Parse request
    model = body.get("model", "grok-4-1-fast-non-reasoning")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Get the last user message for response
    last_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_message = msg.get("content", "")
            break

    response_content = f"Mock response to: {last_message[:50]}..."

    if stream:
        return StreamingResponse(
            generate_stream(model, response_content),
            media_type="text/event-stream",
        )

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    return response.model_dump()


async def generate_stream(model: str, content: str):
    """Generate a streaming response."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Send content in chunks
    words = content.split()
    for i, word in enumerate(words):
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant" if i == 0 else None,
                        "content": word + " ",
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def run_server(host: str = "127.0.0.1", port: int = 8123):
    """Run the mock server."""
    import uvicorn


if __name__ == "__main__":
    run_server()
