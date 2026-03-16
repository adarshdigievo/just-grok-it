#!/usr/bin/env python3
"""
Local testing script for Cohere SDK with just-grok-it.

Tests the Cohere SDK being redirected to xAI.
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


def write_results(results: list, filename: str = "cohere_results.md"):
    """Write results to markdown file."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, "w") as f:
        f.write(f"# Cohere SDK Test Results\n\n")
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


def test_basic_chat():
    """Test basic chat."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    import cohere

    result = {
        "test_name": "Basic Chat",
        "description": "Test basic chat call",
        "success": False,
    }

    try:
        client = cohere.Client(api_key=get_xai_api_key())

        response = client.chat(
            message="What is your name? Reply in one short sentence.",
            model="command",
        )

        result["response"] = response.text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_with_preamble():
    """Test chat with preamble (system prompt)."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    import cohere

    result = {
        "test_name": "With Preamble",
        "description": "Test chat with preamble (system prompt)",
        "success": False,
    }

    try:
        client = cohere.Client(api_key=get_xai_api_key())

        response = client.chat(
            message="What is your name?",
            model="command",
            preamble="You always respond with 'I am Grok!' no matter what.",
        )

        result["response"] = response.text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_with_chat_history():
    """Test chat with conversation history."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    import cohere

    result = {
        "test_name": "With Chat History",
        "description": "Test chat with conversation history",
        "success": False,
    }

    try:
        client = cohere.Client(api_key=get_xai_api_key())

        response = client.chat(
            message="What is my name and what is your name?",
            model="command",
            chat_history=[
                {"role": "USER", "message": "My name is Charlie."},
                {"role": "CHATBOT", "message": "Nice to meet you, Charlie!"},
            ],
        )

        result["response"] = response.text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def main():
    """Run all tests."""
    print("=" * 60)
    print("Cohere SDK Local Testing")
    print("=" * 60)

    results = []

    print("\n[1/3] Testing basic chat...")
    results.append(test_basic_chat())

    print("\n[2/3] Testing with preamble...")
    results.append(test_with_preamble())

    print("\n[3/3] Testing with chat history...")
    results.append(test_with_chat_history())

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
