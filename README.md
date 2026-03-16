# Just Grok It 🚀

**Auto-redirect LLM API requests to xAI APIs with a single line of code.**

[![PyPI version](https://badge.fury.io/py/just-grok-it.svg)](https://badge.fury.io/py/just-grok-it)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`just-grok-it` is a drop-in solution that redirects LLM API calls from multiple providers to [xAI's OpenAI-compatible API](https://docs.x.ai/docs/tutorial). This allows you to seamlessly switch your existing LLM-powered applications to use Grok models without changing any of your existing code.

### Supported SDKs

| Provider | Package | Redirect Method |
|----------|---------|-----------------|
| **OpenAI** | `openai` | Direct base URL redirect |
| **OpenRouter** | `openrouter` | Official SDK patching + format conversion |
| **Together AI** | `openai` (custom base_url) | Base URL intercept + redirect |
| **Groq** | `openai` (custom base_url) | Base URL intercept + redirect |
| **Perplexity** | `openai` (custom base_url) | Base URL intercept + redirect |
| **Anthropic** | `anthropic` | Format conversion + redirect |
| **Google Gemini** | `google-generativeai` | Format conversion + redirect |
| **Mistral AI** | `mistralai` | Format conversion + redirect |
| **Cohere** | `cohere` | Format conversion + redirect |

> **Note:** OpenRouter, Together AI, Groq, and Perplexity also work when using the `openai` SDK with custom `base_url` - those requests are intercepted and redirected as well.

## Installation

```bash
pip install just-grok-it
```

Or with uv:

```bash
uv add just-grok-it
```

## Quick Start

Add a single line at the entry point of your application:

```python
import just_grok_it
just_grok_it.all()  # Uses default model: grok-4-1-fast-non-reasoning

# Your existing code works as-is - all calls go to xAI!
```

### OpenAI Example

```python
import just_grok_it
just_grok_it.all()

from openai import OpenAI

client = OpenAI()  # Connects to xAI APIs
response = client.chat.completions.create(
    model="grok-4",
    messages=[{"role": "user", "content": "Hello, Grok!"}]
)
print(response.choices[0].message.content)
```

### OpenRouter Example (Official SDK)

```python
import just_grok_it
just_grok_it.all()

from openrouter import OpenRouter

# Your existing OpenRouter code - automatically redirected to xAI!
client = OpenRouter(api_key="your-key")
response = client.chat.send(
    model="anthropic/claude-3-opus",  # Will use Grok instead
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Together / Groq Migration (via OpenAI SDK)

```python
import just_grok_it
just_grok_it.all()

from openai import OpenAI

# Your existing Together/Groq code using OpenAI SDK - automatically redirected!
client = OpenAI(
    base_url="https://api.together.xyz/v1",  # Will be redirected!
    api_key="your-key"
)
# All calls now go to xAI
```

### Anthropic Example

```python
import just_grok_it
just_grok_it.all()

from anthropic import Anthropic

client = Anthropic()  # Calls get converted and sent to xAI
response = client.messages.create(
    model="claude-3-sonnet",  # Will use Grok model instead
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

### Google Gemini Example

```python
import just_grok_it
just_grok_it.all()

import google.generativeai as genai

model = genai.GenerativeModel('gemini-pro')  # Uses xAI instead
response = model.generate_content("Hello!")
print(response.text)
```

### Mistral AI Example

```python
import just_grok_it
just_grok_it.all()

from mistralai import Mistral

client = Mistral(api_key="your-key")
response = client.chat.complete(
    model="mistral-large",  # Will use Grok instead
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Cohere Example

```python
import just_grok_it
just_grok_it.all()

import cohere

client = cohere.Client(api_key="your-key")
response = client.chat(
    message="Hello!",
    model="command"  # Will use Grok instead
)
print(response.text)
```

## Configuration

### API Key

Set your xAI API key as an environment variable:

```bash
export XAI_API_KEY="your-xai-api-key"
```

The library checks for API keys in this order:
1. **`XAI_API_KEY`** - Preferred, checked first
2. **`OPENAI_API_KEY`** - Fallback if XAI_API_KEY is not set

When using the OpenAI SDK without passing an `api_key` to the client constructor, `just-grok-it` will automatically use `XAI_API_KEY` from the environment:

```python
import just_grok_it
just_grok_it.all()

from openai import OpenAI
client = OpenAI()  # Uses XAI_API_KEY from environment automatically
```

Get your API key from the [xAI Console](https://console.x.ai/).

### Default Model

The default model is `grok-4-1-fast-non-reasoning`. You can change it:

```python
import just_grok_it
just_grok_it.all(default_model='grok-4')  # Use grok-4 instead
```

### Available Models

- `grok-4-1-fast-non-reasoning` - Fast, efficient model (default)
- `grok-4` - Latest and most capable model
- `grok-3` - Previous generation flagship model  
- `grok-3-mini` - Faster, lightweight model

See [xAI documentation](https://docs.x.ai/docs/models) for the complete list.

## API Reference

### `just_grok_it.all(default_model=None)`

Main entry point. Activates xAI API redirection for all installed SDKs.

**Parameters:**
- `default_model` (str, optional): Default model. Defaults to `grok-4-1-fast-non-reasoning`.

**Returns:**
- `dict`: Provider names mapped to patch success status.

```python
results = just_grok_it.all()
# {'openai': True, 'openrouter': True, 'anthropic': True, 'gemini': False, 'mistral': False, 'cohere': False}
```

### `just_grok_it.unpatch_all()`

Removes all patches and restores original SDK behavior.

### `just_grok_it.is_patched(provider_name=None)`

Check if SDKs are patched.

### `just_grok_it.get_installed_providers()`

Returns list of installed provider names.

### `just_grok_it.get_patched_providers()`

Returns list of currently patched provider names.

### `just_grok_it.DEFAULT_XAI_MODEL`

The default model constant: `grok-4-1-fast-non-reasoning`

### `just_grok_it.XAI_BASE_URL`

The xAI API base URL: `https://api.x.ai/v1`

## Debug Logging

Enable debug logging to see interception messages:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

import just_grok_it
just_grok_it.all()
```

Output:
```
DEBUG:just_grok_it:[just_grok_it] Initialized with default model: grok-4-1-fast-non-reasoning
DEBUG:just_grok_it:[just_grok_it] Patched providers: openai, openrouter, anthropic, gemini, mistral, cohere
DEBUG:just_grok_it:[just_grok_it] All requests will now be redirected to https://api.x.ai/v1
```

When making API calls:
```
DEBUG:just_grok_it:[just_grok_it] openai: Intercepted OpenAI initialization, redirecting to xAI API: https://api.x.ai/v1
DEBUG:just_grok_it:[just_grok_it] anthropic: Intercepted messages.create request, redirecting to xAI API with model: grok-4
```

## Examples

See the [`examples/`](examples/) directory for complete usage examples:

- `basic_openai.py` - Basic OpenAI SDK usage
- `basic_anthropic.py` - Basic Anthropic SDK usage
- `basic_gemini.py` - Basic Gemini SDK usage
- `openrouter_migration.py` - Migrating from OpenRouter
- `multi_provider.py` - Using multiple providers
- `streaming.py` - Streaming responses
- `async_example.py` - Async client usage
- `debug_logging.py` - Debug logging

## How It Works

### OpenAI SDK (and compatible providers via base_url)
The library monkey-patches the OpenAI client `__init__` to override the `base_url` to `https://api.x.ai/v1`, regardless of the original URL. This covers OpenAI direct usage, and any provider that uses the OpenAI SDK with a custom `base_url` (Together, Groq, Perplexity, etc.).

### Official OpenRouter SDK
The library patches the `openrouter.chat.Chat.send()` method to intercept calls, convert to OpenAI format, send to xAI, and convert the response back to OpenRouter format.

### Native SDKs (Anthropic, Gemini, Mistral, Cohere)
The library intercepts API calls, converts the request format to OpenAI format, sends to xAI, and converts the response back to the original format.

## Compatibility

- **Python**: 3.10, 3.11, 3.12, 3.13, 3.14
- **OpenAI SDK**: 1.0.0+
- **OpenRouter SDK**: 0.1.0+
- **Anthropic SDK**: 0.18.0+
- **Google GenAI SDK**: 0.5.0+
- **Mistral AI SDK**: 0.1.0+
- **Cohere SDK**: 5.0.0+

## Adding New Providers

The library is designed to be extensible:

1. Create a provider class inheriting from `BaseProvider` in `src/just_grok_it/providers/`
2. Implement: `is_installed()`, `patch()`, `unpatch()`
3. Add format converters in `src/just_grok_it/_converters/` if needed
4. Register in `src/just_grok_it/providers/__init__.py`

## Development

### Setting Up Development Environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/just-grok-it/just-grok-it.git
cd just-grok-it

# Install all development dependencies (recommended)
# This includes: pytest, all provider SDKs, and integration test deps
uv sync

# Run tests
uv run pytest
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run tests for a specific provider
uv run pytest tests/openai/
uv run pytest tests/anthropic/
uv run pytest tests/gemini/
uv run pytest tests/mistral/
uv run pytest tests/cohere/
uv run pytest tests/openrouter/

# Run with debug logging
uv run pytest -v --log-cli-level=DEBUG
```

### Installing for End Users

For end users (not development), install with pip:

```bash
# Basic install (just OpenAI SDK included)
pip install just-grok-it

# Install with specific provider SDKs
pip install just-grok-it[anthropic]
pip install just-grok-it[openrouter]
pip install just-grok-it[gemini]
pip install just-grok-it[mistral]
pip install just-grok-it[cohere]

# Install with all provider SDKs
pip install just-grok-it[all-providers]
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
