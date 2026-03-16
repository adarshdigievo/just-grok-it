#!/usr/bin/env python3
"""
Local testing script for OpenRouter SDK with just-grok-it.

Tests the official OpenRouter SDK being redirected to xAI.
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


def write_results(results: list, filename: str = "openrouter_results.md"):
    """Write results to markdown file."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, "w") as f:
        f.write(f"# OpenRouter SDK Test Results\n\n")
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
    """Test basic chat with API key from environment."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openrouter import OpenRouter

    result = {
        "test_name": "Basic Chat (API key from env)",
        "description": "Test basic chat using OpenRouter SDK with API key from XAI_API_KEY env",
        "success": False,
    }

    try:
        # OpenRouter SDK uses OPENROUTER_API_KEY by default, but we redirect to xAI
        api_key = get_xai_api_key()
        client = OpenRouter(api_key=api_key)

        response = client.chat.send(
            model="anthropic/claude-3-opus",  # Will be ignored, Grok is used
            messages=[
                {
                    "role": "user",
                    "content": "What is your name? Reply in one short sentence.",
                }
            ],
        )

        result["response"] = response.choices[0].message.content
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_basic_chat_passed_key():
    """Test basic chat with different model specified."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openrouter import OpenRouter

    result = {
        "test_name": "Basic Chat (Different model)",
        "description": "Test chat with a different model specified - should still use Grok",
        "success": False,
    }

    try:
        api_key = get_xai_api_key()
        client = OpenRouter(api_key=api_key)

        response = client.chat.send(
            model="openai/gpt-4",  # Will be redirected to Grok
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "What is your name? Reply in one short sentence.",
                },
            ],
        )

        result["response"] = response.choices[0].message.content
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_with_temperature():
    """Test chat with temperature parameter."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openrouter import OpenRouter

    result = {
        "test_name": "Chat with Temperature",
        "description": "Test chat with temperature parameter",
        "success": False,
    }

    try:
        api_key = get_xai_api_key()
        client = OpenRouter(api_key=api_key)

        response = client.chat.send(
            model="grok-3-mini",
            messages=[
                {
                    "role": "user",
                    "content": "What is your name? Reply in one short sentence.",
                }
            ],
            temperature=0.7,
            max_tokens=100,
        )

        result["response"] = response.choices[0].message.content
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_developer_role():
    """Test chat with developer role (should be converted to system)."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from openrouter import OpenRouter

    result = {
        "test_name": "Developer Role Conversion",
        "description": "Test that 'developer' role is converted to 'system'",
        "success": False,
    }

    try:
        api_key = get_xai_api_key()
        client = OpenRouter(api_key=api_key)

        response = client.chat.send(
            model="grok-3-mini",
            messages=[
                {
                    "role": "developer",
                    "content": "You always respond with 'I am Grok!'",
                },
                {"role": "user", "content": "What is your name?"},
            ],
            max_tokens=50,
        )

        result["response"] = response.choices[0].message.content
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenRouter SDK Local Testing")
    print("=" * 60)

    results = []

    print("\n[1/4] Testing basic chat (env key)...")
    results.append(test_basic_chat_env_key())

    print("\n[2/4] Testing basic chat (different model)...")
    results.append(test_basic_chat_passed_key())

    print("\n[3/4] Testing with temperature...")
    results.append(test_with_temperature())

    print("\n[4/4] Testing developer role conversion...")
    results.append(test_developer_role())

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
