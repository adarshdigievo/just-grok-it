#!/usr/bin/env python3
"""
Local testing script for Google Gemini SDK with just-grok-it.

Tests the google-generativeai SDK being redirected to xAI.
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


def write_results(results: list, filename: str = "gemini_results.md"):
    """Write results to markdown file."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / filename
    with open(filepath, "w") as f:
        f.write(f"# Gemini SDK Test Results\n\n")
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


def test_basic_generate_content():
    """Test basic generate_content with string input."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    import google.generativeai as genai

    result = {
        "test_name": "Basic Generate Content (String)",
        "description": "Test basic generate_content with a simple string prompt",
        "success": False,
    }

    try:
        # Configure genai (not needed since we redirect, but for completeness)
        genai.configure(api_key=get_xai_api_key())

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(
            "What is your name? Reply in one short sentence."
        )

        result["response"] = response.text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_with_generation_config():
    """Test generate_content with generation_config."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    import google.generativeai as genai

    result = {
        "test_name": "Generate Content with Config",
        "description": "Test generate_content with generation_config parameters",
        "success": False,
    }

    try:
        genai.configure(api_key=get_xai_api_key())

        model = genai.GenerativeModel("gemini-pro")
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 100,
        }

        response = model.generate_content(
            "What is your name? Reply in one short sentence.",
            generation_config=generation_config,
        )

        result["response"] = response.text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_conversation_history():
    """Test generate_content with conversation history."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    import google.generativeai as genai

    result = {
        "test_name": "Conversation History",
        "description": "Test generate_content with multi-turn conversation",
        "success": False,
    }

    try:
        genai.configure(api_key=get_xai_api_key())

        model = genai.GenerativeModel("gemini-pro")

        # Multi-turn conversation
        contents = [
            {"role": "user", "parts": ["My name is Alice."]},
            {"role": "model", "parts": ["Nice to meet you, Alice!"]},
            {"role": "user", "parts": ["What is my name and what is your name?"]},
        ]

        response = model.generate_content(contents)

        result["response"] = response.text
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["debug_logs"] = log_capture.getvalue()
    just_grok_it.unpatch_all()
    return result


def test_different_model_name():
    """Test with a different Gemini model name."""
    log_capture.truncate(0)
    log_capture.seek(0)

    import just_grok_it

    just_grok_it.unpatch_all()
    just_grok_it.all()

    import google.generativeai as genai

    result = {
        "test_name": "Different Model Name",
        "description": "Test with gemini-1.5-pro model name (should still use Grok)",
        "success": False,
    }

    try:
        genai.configure(api_key=get_xai_api_key())

        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            "What is your name? Reply in one short sentence."
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
    print("Gemini SDK Local Testing")
    print("=" * 60)

    results = []

    print("\n[1/4] Testing basic generate_content...")
    results.append(test_basic_generate_content())

    print("\n[2/4] Testing with generation_config...")
    results.append(test_with_generation_config())

    print("\n[3/4] Testing conversation history...")
    results.append(test_conversation_history())

    print("\n[4/4] Testing different model name...")
    results.append(test_different_model_name())

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
