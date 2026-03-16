#!/usr/bin/env python3
"""
Run all local testing checks.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run all check scripts."""
    scripts_dir = Path(__file__).parent

    scripts = [
        "check_openai.py",
        "check_openrouter.py",
        "check_gemini.py",
        "check_anthropic.py",
        "check_mistral.py",
        "check_cohere.py",
    ]

    print("=" * 70)
    print("Running All Local Testing Checks")
    print("=" * 70)

    for script in scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"\n{'=' * 70}")
            print(f"Running: {script}")
            print("=" * 70)

            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(scripts_dir),
            )

            if result.returncode != 0:
                print(f"WARNING: {script} exited with code {result.returncode}")
        else:
            print(f"WARNING: {script} not found")

    print("\n" + "=" * 70)
    print("All checks completed!")
    print(f"Results are in: {scripts_dir / 'results'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
