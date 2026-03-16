# Just Grok It - Examples

This directory contains example scripts demonstrating how to use `just-grok-it` with various LLM SDKs.

## Prerequisites

1. Install just-grok-it:
   ```bash
   pip install just-grok-it
   ```

2. Install the provider SDKs you want to use:
   ```bash
   # OpenAI is included by default
   
   # Install specific providers
   pip install anthropic          # For Anthropic examples
   pip install google-generativeai # For Gemini examples
   pip install openrouter          # For OpenRouter SDK examples
   pip install mistralai           # For Mistral examples
   pip install cohere              # For Cohere examples
   
   # Or install all providers at once
   pip install just-grok-it[all-providers]
   ```

3. Set your xAI API key:
   ```bash
   export XAI_API_KEY="your-xai-api-key"
   ```

## Examples

### Basic Usage

- [`basic_openai.py`](basic_openai.py) - Basic OpenAI SDK redirection
- [`basic_anthropic.py`](basic_anthropic.py) - Basic Anthropic SDK redirection
- [`basic_gemini.py`](basic_gemini.py) - Basic Google Gemini SDK redirection

### Provider Migration

- [`openrouter_migration.py`](openrouter_migration.py) - Migrating from OpenRouter to xAI (both official SDK and OpenAI SDK patterns)
- [`multi_provider.py`](multi_provider.py) - Using multiple providers simultaneously

### Advanced Usage

- [`streaming.py`](streaming.py) - Streaming responses
- [`async_example.py`](async_example.py) - Async client usage

### Debug & Logging

- [`debug_logging.py`](debug_logging.py) - Enable debug logging to see interception

## Running Examples

```bash
# Run any example
python examples/basic_openai.py

# With debug logging
python examples/debug_logging.py
```

## Notes

- All examples require a valid `XAI_API_KEY` environment variable
- Examples will make real API calls to xAI's servers
- Be mindful of API rate limits and costs
