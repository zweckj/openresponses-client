#!/usr/bin/env python3
"""Run a tool calling loop with a local function.

Usage: python examples/tools.py "What's the weather like in Paris?"
Configure with OPENRESPONSES_BASE_URL, OPENRESPONSES_API_KEY, OPENRESPONSES_MODEL.
"""

import asyncio
import json
import os
import sys
from typing import Any

from openresponses_client import (
    APIError,
    FunctionToolParam,
    InputItem,
    OpenResponsesClient,
)

MAX_TURNS = 5

TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. Paris, France",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    }
]


def get_weather(location: str) -> dict[str, Any]:
    """Return fake weather data."""
    return {"location": location, "temperature_c": 21, "conditions": "sunny"}


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool call and return its JSON result."""
    if name == "get_weather":
        return json.dumps(get_weather(**arguments))
    return json.dumps({"error": f"Unknown tool: {name}"})


async def run(question: str) -> int:
    history: list[InputItem] = [{"role": "user", "content": question}]
    async with OpenResponsesClient(
        os.environ.get("OPENRESPONSES_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.environ.get("OPENRESPONSES_API_KEY"),
    ) as client:
        for _turn in range(MAX_TURNS):
            try:
                response = await client.create(
                    model=os.environ.get("OPENRESPONSES_MODEL", "gpt-oss:20b"),
                    input=history,
                    tools=TOOLS,
                    store=False,
                )
            except APIError as err:
                print(f"Request failed: {err}", file=sys.stderr)
                return 1

            # Output items, including reasoning and function calls, are sent
            # back unchanged as input of the next turn.
            history.extend(response.output)
            if not response.function_calls:
                print(response.output_text)
                return 0

            for call in response.function_calls:
                print(f"-> {call.name}({call.arguments})", file=sys.stderr)
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": call_tool(call.name, call.parse_arguments()),
                    }
                )

    print(f"No answer after {MAX_TURNS} turns", file=sys.stderr)
    return 1


def main() -> int:
    question = " ".join(sys.argv[1:]) or "What's the weather like in Paris?"
    return asyncio.run(run(question))


if __name__ == "__main__":
    sys.exit(main())
