#!/usr/bin/env python3
"""
Local testing script for Anthropic SDK with just-grok-it.

Tests the Anthropic SDK being redirected to xAI.
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


def write_results(results: list, filename: str = "anthropic_results.md"):
    """Write results to markdown file."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, "w") as f:
        f.write(f"# Anthropic SDK Test Results\n\n")
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


def test_basic_message():
    """Test basic messages.create."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from anthropic import Anthropic

    result = {
        "test_name": "Basic Message",
        "description": "Test basic messages.create call",
        "success": False,
    }

    try:
        # Set env for Anthropic (will be redirected)
        os.environ["ANTHROPIC_API_KEY"] = get_xai_api_key()
        client = Anthropic()

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "What is your name? Reply in one short sentence.",
                }
            ],
        )

        result["response"] = response.content[0].text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_with_system_prompt():
    """Test messages.create with system prompt."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from anthropic import Anthropic

    result = {
        "test_name": "With System Prompt",
        "description": "Test messages.create with system parameter",
        "success": False,
    }

    try:
        os.environ["ANTHROPIC_API_KEY"] = get_xai_api_key()
        client = Anthropic()

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            system="You always respond with 'I am Grok!' no matter what.",
            messages=[{"role": "user", "content": "What is your name?"}],
        )

        result["response"] = response.content[0].text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_with_key_passed():
    """Test messages.create with API key passed directly."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from anthropic import Anthropic

    result = {
        "test_name": "API Key Passed Directly",
        "description": "Test with API key passed to Anthropic constructor",
        "success": False,
    }

    try:
        client = Anthropic(api_key=get_xai_api_key())

        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "What is your name? Reply in one short sentence.",
                }
            ],
        )

        result["response"] = response.content[0].text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_conversation():
    """Test multi-turn conversation."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    from anthropic import Anthropic

    result = {
        "test_name": "Multi-turn Conversation",
        "description": "Test messages.create with conversation history",
        "success": False,
    }

    try:
        client = Anthropic(api_key=get_xai_api_key())

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "My name is Bob."},
                {"role": "assistant", "content": "Nice to meet you, Bob!"},
                {"role": "user", "content": "What is my name and what is your name?"},
            ],
        )

        result["response"] = response.content[0].text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def main():
    """Run all tests."""
    print("=" * 60)
    print("Anthropic SDK Local Testing")
    print("=" * 60)

    results = []

    print("\n[1/4] Testing basic message...")
    results.append(test_basic_message())

    print("\n[2/4] Testing with system prompt...")
    results.append(test_with_system_prompt())

    print("\n[3/4] Testing with key passed directly...")
    results.append(test_with_key_passed())

    print("\n[4/4] Testing multi-turn conversation...")
    results.append(test_conversation())

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
