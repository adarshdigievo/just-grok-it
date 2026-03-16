"""
Pytest fixtures for integration tests.
"""

import os
import socket
import threading
import time
from typing import Generator

import pytest

# Check if integration test dependencies are installed
try:
    import fastapi
    import uvicorn

    INTEGRATION_DEPS_AVAILABLE = True
except ImportError:
    INTEGRATION_DEPS_AVAILABLE = False

# Skip all tests in this directory if integration deps are not available
pytestmark = pytest.mark.skipif(
    not INTEGRATION_DEPS_AVAILABLE,
    reason="Integration test dependencies not installed (uvicorn, fastapi). Run: uv sync --group dev",
)


def get_free_port() -> int:
    """Get an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_server_port() -> Generator[int, None, None]:
    """Start a mock xAI server and return its port."""
    import uvicorn

    from tests.integration.mock_server import app

    port = get_free_port()
    host = "127.0.0.1"

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Run server in a thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    url = f"http://{host}:{port}"
    max_retries = 50
    for _ in range(max_retries):
        try:
            import httpx

            response = httpx.get(f"{url}/health", timeout=1)
            if response.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("Mock server failed to start")

    yield port


@pytest.fixture(scope="session")
def mock_server_url(mock_server_port: int) -> str:
    """Get the mock server URL."""
    return f"http://127.0.0.1:{mock_server_port}"


@pytest.fixture(autouse=True)
def setup_mock_environment(mock_server_url: str, monkeypatch):
    """Set up environment to use mock server."""
    import just_grok_it

    # First unpatch everything
    just_grok_it.unpatch_all()

    # Set environment variable
    monkeypatch.setenv("XAI_API_KEY", "test-api-key")

    # Patch the XAI_BASE_URL constant BEFORE any patching happens
    mock_base_url = f"{mock_server_url}/v1"
    monkeypatch.setattr("just_grok_it._constants.XAI_BASE_URL", mock_base_url)

    # Also patch it in the providers module since it's imported there
    monkeypatch.setattr(
        "just_grok_it.providers.openai_provider.XAI_BASE_URL", mock_base_url
    )
    monkeypatch.setattr(
        "just_grok_it.providers.anthropic_provider.XAI_BASE_URL", mock_base_url
    )

    # Clear requests before each test
    import httpx

    httpx.delete(f"{mock_server_url}/v1/requests")

    yield

    # Cleanup
    just_grok_it.unpatch_all()


@pytest.fixture
def mock_xai_base_url(mock_server_url: str) -> str:
    """Return the mock xAI base URL."""
    return f"{mock_server_url}/v1"
