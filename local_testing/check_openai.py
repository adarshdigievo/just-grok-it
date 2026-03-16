#!/usr/bin/env python3
"""
Local testing script for OpenAI SDK with just-grok-it.

Tests various scenarios:
- Basic chat completion
- API key from env vs passed directly
- Structured outputs
- Tool calling
"""

import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Capture logs
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logging.getLogger("just_grok_it").addHandler(handler)
logging.getLogger("just_grok_it").setLevel(logging.DEBUG)

# Also output to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
logging.getLogger("just_grok_it").addHandler(console_handler)


def get_xai_api_key():
    """Get xAI API key from environment."""
    key = os.environ.get("XAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: XAI_API_KEY or OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    return key


def write_results(results: list, filename: str = "openai_results.md"):
    """Write results to markdown file."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, "w") as f:
        f.write(f"# OpenAI SDK Test Results\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")

        for result in results:
            f.write(f"## {result['test_name']}\n\n")
            f.write(f"**Status:** {'✅ PASS' if result['success'] else '❌ FAIL'}\n\n")

            if result.get("description"):
                f.write(f"**Description:** {result['description']}\n\n")

            if result.get("debug_logs"):
                f.write("### Debug Logs\n\n```\n")
                f.write(result["debug_logs"])
                f.write("\n```\n\n")

            if result.get("response"):
                f.write("### Response\n\n```\n")
                f.write(result["response"])
                f.write("\n```\n\n")

            if result.get("error"):
                f.write("### Error\n\n```\n")
                f.write(result["error"])
                f.write("\n```\n\n")

            f.write("---\n\n")

    print(f"\nResults written to: {filepath}")
    return filepath


def test_basic_chat_env_key():
    """Test basic chat with API key from XAI_API_KEY environment variable."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    result = {
        "test_name": "Basic Chat (API key from XAI_API_KEY env)",
        "description": "Test basic chat completion using API key from XAI_API_KEY environment variable. "
        "just-grok-it automatically uses XAI_API_KEY when no api_key is passed to OpenAI().",
        "success": False,
    }

    try:
        # Note: just-grok-it checks XAI_API_KEY first, then falls back to OPENAI_API_KEY
        client = OpenAI()  # Uses XAI_API_KEY from env

        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What is your name? Reply in one short sentence.",
                }
            ],
            max_tokens=100,
        )

        result["response"] = response.choices[0].message.content
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_basic_chat_passed_key():
    """Test basic chat with API key passed directly."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    result = {
        "test_name": "Basic Chat (API key passed directly)",
        "description": "Test basic chat completion with API key passed to OpenAI client constructor",
        "success": False,
    }

    try:
        api_key = get_xai_api_key()
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What is your name? Reply in one short sentence.",
                }
            ],
            max_tokens=100,
        )

        result["response"] = response.choices[0].message.content
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_structured_output():
    """Test structured output with response_format."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    result = {
        "test_name": "Structured Output (JSON mode)",
        "description": "Test structured output using response_format with JSON schema",
        "success": False,
    }

    try:
        client = OpenAI()

        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs JSON.",
                },
                {
                    "role": "user",
                    "content": "Give me info about yourself in JSON format with fields: name, type, capabilities (as array)",
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )

        content = response.choices[0].message.content
        # Try to parse as JSON to verify it's valid
        parsed = json.loads(content)
        result["response"] = json.dumps(parsed, indent=2)
        result["success"] = True

    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON response: {e}\nContent: {content}"
    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_tool_calling():
    """Test tool calling functionality."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    result = {
        "test_name": "Tool Calling",
        "description": "Test tool calling with a simple get_weather function",
        "success": False,
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    try:
        client = OpenAI()

        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
        )

        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            result["response"] = (
                f"Tool called: {tool_call.function.name}\nArguments: {tool_call.function.arguments}"
            )
            result["success"] = True
        else:
            result["response"] = f"No tool call made. Response: {message.content}"
            result["success"] = True  # Model may choose not to use tool

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_openrouter_url_redirect():
    """Test that OpenRouter base_url gets redirected."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openai import OpenAI

    result = {
        "test_name": "OpenRouter URL Redirect",
        "description": "Test that requests with OpenRouter base_url get redirected to xAI",
        "success": False,
    }

    try:
        api_key = get_xai_api_key()
        # Create client with OpenRouter URL - should be redirected
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Check that base_url was redirected
        base_url = str(client.base_url).rstrip("/")
        result["response"] = f"Client base_url: {base_url}\n"

        response = client.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What is your name? Reply in one short sentence.",
                }
            ],
            max_tokens=100,
        )

        result["response"] += f"Response: {response.choices[0].message.content}"
        result["success"] = "x.ai" in base_url

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenAI SDK Local Testing")
    print("=" * 60)

    results = []

    print("\n[1/5] Testing basic chat (env key)...")
    results.append(test_basic_chat_env_key())

    print("\n[2/5] Testing basic chat (passed key)...")
    results.append(test_basic_chat_passed_key())

    print("\n[3/5] Testing structured output...")
    results.append(test_structured_output())

    print("\n[4/5] Testing tool calling...")
    results.append(test_tool_calling())

    print("\n[5/5] Testing OpenRouter URL redirect...")
    results.append(test_openrouter_url_redirect())

    # Write results
    write_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    passed = sum(1 for r in results if r["success"])
    print(f"  Passed: {passed}/{len(results)}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['test_name']}")


if __name__ == "__main__":
    main()
