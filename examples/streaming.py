#!/usr/bin/env python3
"""Stream a response and print the text as it arrives.

Usage: python examples/streaming.py "Write a haiku about the sea."
Configure with OPENRESPONSES_BASE_URL, OPENRESPONSES_API_KEY, OPENRESPONSES_MODEL.
"""

import asyncio
import os
import sys

from openresponses_client import APIError, OpenResponsesClient
from openresponses_client.models import (
    ErrorEvent,
    ResponseCompletedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)


async def run(prompt: str) -> int:
    async with OpenResponsesClient(
        os.environ.get("OPENRESPONSES_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.environ.get("OPENRESPONSES_API_KEY"),
    ) as client:
        try:
            async with client.stream(
                model=os.environ.get("OPENRESPONSES_MODEL", "gpt-oss:20b"),
                input=prompt,
                reasoning={"summary": "auto"},
            ) as stream:
                async for event in stream:
                    match event:
                        case ResponseReasoningSummaryTextDeltaEvent(delta=delta):
                            print(delta, end="", file=sys.stderr, flush=True)
                        case ResponseOutputTextDeltaEvent(delta=delta):
                            print(delta, end="", flush=True)
                        case ErrorEvent(error=error):
                            print(f"\nError: {error.message}", file=sys.stderr)
                        case ResponseCompletedEvent(response=response):
                            usage = response.usage
                            tokens = usage.total_tokens if usage else "unknown"
                            print(f"\n[completed, {tokens} tokens]")
        except APIError as err:
            print(f"Request failed: {err}", file=sys.stderr)
            return 1
    return 0


def main() -> int:
    prompt = " ".join(sys.argv[1:]) or "Count from 1 to 5."
    return asyncio.run(run(prompt))


if __name__ == "__main__":
    sys.exit(main())
