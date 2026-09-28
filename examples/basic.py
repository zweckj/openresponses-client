#!/usr/bin/env python3
"""Create a response and print its text.

Usage: python examples/basic.py "What is the capital of France?"
Configure with OPENRESPONSES_BASE_URL, OPENRESPONSES_API_KEY, OPENRESPONSES_MODEL.
"""

import asyncio
import os
import sys

from openresponses_client import APIError, OpenResponsesClient


async def run(question: str) -> int:
    async with OpenResponsesClient(
        os.environ.get("OPENRESPONSES_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.environ.get("OPENRESPONSES_API_KEY"),
    ) as client:
        try:
            response = await client.create(
                model=os.environ.get("OPENRESPONSES_MODEL", "gpt-oss:20b"),
                instructions="Answer in one short sentence.",
                input=question,
            )
        except APIError as err:
            print(f"Request failed: {err}", file=sys.stderr)
            return 1

    print(response.output_text)
    if response.usage is not None:
        print(f"\n[{response.usage.total_tokens} tokens, status {response.status}]")
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]) or "Say hello in exactly three words."
    return asyncio.run(run(question))


if __name__ == "__main__":
    sys.exit(main())
