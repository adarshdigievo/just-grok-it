# Local Testing

This folder contains scripts for testing the `just-grok-it` library against the **live xAI API**.

These scripts make **real API calls** that will use your API quota.

## Prerequisites

1. Set your xAI API key:
   ```bash
   export XAI_API_KEY="your-xai-api-key"
   ```

2. Install the package with dev dependencies:
   ```bash
   cd /testbed
   uv sync
   ```

## Running Tests

Each script tests a specific provider and writes results to a markdown file:

```bash
# Test OpenAI SDK (most comprehensive - includes tool calling, structured output)
uv run python local_testing/check_openai.py

# Test OpenRouter SDK (official SDK)
uv run python local_testing/check_openrouter.py

# Test Gemini SDK
uv run python local_testing/check_gemini.py

# Test Anthropic SDK
uv run python local_testing/check_anthropic.py

# Test Mistral SDK
uv run python local_testing/check_mistral.py

# Test Cohere SDK
uv run python local_testing/check_cohere.py

# Run ALL checks (recommended)
uv run python local_testing/run_all_checks.py
```

## Output

Results are written to `local_testing/results/` as markdown files containing:
- Debug logs showing interception and redirection
- API request details
- Model responses (to verify it's Grok responding)
- Error messages if any tests fail

## Test Scenarios

### OpenAI SDK (`check_openai.py`)
1. Basic chat (API key from env)
2. Basic chat (API key passed directly)
3. Structured output (JSON mode)
4. Tool calling (function calling)
5. OpenRouter URL redirect

### OpenRouter SDK (`check_openrouter.py`)
1. Basic chat
2. Different model specification
3. Temperature parameter
4. Developer role conversion

### Gemini SDK (`check_gemini.py`)
1. Basic generate_content
2. With generation_config
3. Conversation history
4. Different model names

### Anthropic SDK (`check_anthropic.py`)
1. Basic message
2. With system prompt
3. API key passed directly
4. Multi-turn conversation

### Mistral SDK (`check_mistral.py`)
1. Basic chat
2. With system message
3. With temperature

### Cohere SDK (`check_cohere.py`)
1. Basic chat
2. With preamble (system prompt)
3. With chat history

## Verifying Results

After running the checks, open the markdown files in `results/` to verify:

1. **Debug logs show interception**: Look for `[just_grok_it]` messages
2. **Model identifies as Grok**: The response should mention "Grok" or "xAI"
3. **All tests pass**: Check the status indicators (✅/❌)

## Notes

- Results are gitignored to avoid committing API responses
- Each script clears and recaptures logs between tests
- Tests use `grok-3-mini` by default for faster responses
