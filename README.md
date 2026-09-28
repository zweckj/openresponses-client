---
title: openresponses-client
description: Async Python client for the Open Responses API, built on aiohttp
---

## openresponses-client

Async Python client for [Open Responses](https://www.openresponses.org), the open,
multi-provider API spec based on the Responses API. Built on [aiohttp](https://docs.aiohttp.org).

* Works with any server that implements the [spec](https://www.openresponses.org/specification)
* Typed dataclasses ([mashumaro](https://github.com/Fatal1ty/mashumaro)) for all items and 24 streaming events
* HTTP streaming, WebSocket mode and compaction
* Keeps provider extensions instead of failing on them
* Python 3.14+, targets spec release `2026-04-24`

[Quick start](#quick-start) | [Streaming](#streaming) | [Tool calling](#tool-calling) |
[WebSocket](#websocket-mode) | [Errors](#errors) | [Configuration](#configuration)

## Install

```bash
pip install openresponses-client
```

## Quick start

```python
import asyncio

from openresponses_client import OpenResponsesClient


async def main() -> None:
    client = OpenResponsesClient("http://localhost:8080/v1", api_key="sk-...")
    async with client:
        response = await client.create(model="gpt-oss:20b", input="Hi!")
        print(response.output_text)


asyncio.run(main())
```

* `base_url` includes the version prefix. Requests go to `{base_url}/responses`.
* `api_key` is sent as a Bearer token. Leave it out for local servers.
* Every request field of the spec is a keyword argument, for example `tools` or `reasoning`.
* Pass `session=` to reuse your own `aiohttp.ClientSession`.

## Input

Input is a string or a list of items. Items are plain dicts or output items of a response.

```python
response = await client.create(
    model="gpt-oss:20b",
    input=[
        {"role": "developer", "content": "You are a pirate."},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What do you see?"},
                {"type": "input_image", "image_url": "data:image/png;base64,..."},
            ],
        },
    ],
)
```

* Messages without `type` get `"type": "message"` added.
* Files use `{"type": "input_file", "filename": "a.pdf", "file_data": "data:..."}`.
* All item shapes are typed as `TypedDict`s in `openresponses_client.params`.
* Models in `input` are checked when sent: a non-string in a `str` field raises `TypeError`.

## Streaming

```python
from openresponses_client.models import ResponseOutputTextDeltaEvent

async with client.stream(model="gpt-oss:20b", input="Write a haiku.") as stream:
    async for event in stream:
        match event:
            case ResponseOutputTextDeltaEvent(delta=delta):
                print(delta, end="", flush=True)
    response = await stream.get_final_response()
```

| Member                       | Gives you                                         |
|------------------------------|---------------------------------------------------|
| `await get_final_response()` | The final response, after consuming the stream    |
| `text_deltas()`              | An async iterator over the text deltas            |
| `snapshot`                   | The response rebuilt from the events so far       |
| `final_response`             | The final response, once a terminal event arrived |
| `error_event`                | The last `error` event, if any                    |

* Failed and incomplete responses are returned, not raised. Check `response.status`.
* `get_final_response()` raises `ResponseStreamError` if no final response arrives.
* `ResponseAccumulator` rebuilds a response from events you received elsewhere.

## Tool calling

Run the function calls of a response and send the results back:

```python
import json

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

history = [{"role": "user", "content": "What's the weather in Paris?"}]
while True:
    response = await client.create(
        model="gpt-oss:20b", input=history, tools=tools, store=False
    )
    history.extend(response.output)
    if not response.function_calls:
        break
    for call in response.function_calls:
        result = get_weather(**call.parse_arguments())
        history.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            }
        )

print(response.output_text)
```

* Output items go back as input unchanged, including reasoning and provider items.
* Servers that store responses can continue with `previous_response_id` instead.
* `tool_choice` takes `"auto"`, `"required"`, `"none"`, a function, `allowed_tools` or a
  hosted tool such as `{"type": "image_generation"}`.

## WebSocket mode

One connection for many turns, with the same events as HTTP streaming:

```python
async with client.websocket() as ws:
    first = await ws.create(model="gpt-oss:20b", store=False, input="Remember: cobalt.")
    second = await ws.create(
        model="gpt-oss:20b",
        store=False,
        previous_response_id=first.id,
        input="What was the code word?",
    )
```

* `ws.stream()` streams a turn, like `client.stream()`.
* One turn runs at a time. Concurrent calls are queued. Open more connections for parallel turns.
* `stream`, `stream_options` and `background` are not accepted.
* A closed connection reopens for the next turn. A turn the server never started is resent once.
* Options: `heartbeat`, `receive_timeout`, `max_msg_size`, `extra_headers`, `auto_reconnect`.

> [!IMPORTANT]
> With `store=False` only the current connection remembers the last response. After a
> reconnect, continuing raises `PreviousResponseNotFoundError`. Resend the full history:

```python
from openresponses_client import PreviousResponseNotFoundError

try:
    response = await ws.create(
        model="gpt-oss:20b",
        store=False,
        previous_response_id=previous.id,
        input=[message],
    )
except PreviousResponseNotFoundError:
    response = await ws.create(
        model="gpt-oss:20b", store=False, input=[*history, message]
    )
```

## Compaction

Shrink a long conversation, then start a new chain from the result:

```python
compacted = await client.compact(model="gpt-oss:20b", input=history)
response = await client.create(
    model="gpt-oss:20b",
    input=[*compacted.output, {"role": "user", "content": "Continue."}],
)
```

## Listing models

A quick way to check the URL and API key:

```python
models = await client.list_models(timeout=10)
print([model.id for model in models])
```

* A bad key raises `AuthenticationError`.
* An unreachable server raises `APIConnectionError`.

> [!NOTE]
> `GET /models` is not part of the spec. Most OpenAI compatible servers provide it, others
> raise `NotFoundError`.

## Errors

All exceptions derive from `OpenResponsesError`.

| Exception                              | Raised when                               |
|----------------------------------------|-------------------------------------------|
| `APIStatusError`                       | The server returned an error (base class) |
| `BadRequestError`                      | Status 400                                |
| `AuthenticationError`                  | Status 401                                |
| `PermissionDeniedError`                | Status 403                                |
| `NotFoundError`                        | Status 404                                |
| `ConflictError`                        | Status 409                                |
| `UnprocessableEntityError`             | Status 422                                |
| `RateLimitError`                       | Status 429                                |
| `InternalServerError`                  | Status 5xx                                |
| `PreviousResponseNotFoundError`        | Code `previous_response_not_found`        |
| `WebSocketConnectionLimitReachedError` | Code `websocket_connection_limit_reached` |
| `APIConnectionError`                   | The server is unreachable or disconnected |
| `APITimeoutError`                      | A request or stream timed out             |
| `WebSocketClosedError`                 | The WebSocket closed during a turn        |
| `APIResponseValidationError`           | The server sent invalid data              |
| `ResponseStreamError`                  | A stream ended without a final response   |

```python
from openresponses_client import APIStatusError, RateLimitError

try:
    response = await client.create(model="gpt-oss:20b", input="Hi")
except RateLimitError:
    ...
except APIStatusError as err:
    print(err.status, err.code, err.message)
```

* `APIStatusError` has `status`, `code`, `type`, `param`, `message`, `body` and `headers`.
* `APITimeoutError` and `WebSocketClosedError` are subclasses of `APIConnectionError`.
* Error messages strip query strings and credentials from URLs, so they are safe to log.

## Extensions

Providers can add their own items, tools, events and fields. Nothing is lost:

* Unknown types become `UnknownItem`, `UnknownContent`, `UnknownTool` or `UnknownEvent`.
* Their fields are attributes, also for type checkers, for example `item.code`.
* Extra fields of known types are in `extra`.
* `to_dict()` returns the data unchanged, ready to send back.
* OpenAI style `error` events and `response.reasoning_text.*` events are mapped to the spec.

Send your own request fields and headers with `extra_body` and `extra_headers`:

```python
response = await client.create(
    model="my-model",
    input="Find documents about climate change.",
    tools=[{"type": "acme:document_search", "index": "papers"}],
    extra_body={"acme_priority": "high"},
)
```

## Configuration

| Argument        | Default  | Description                                                   |
|-----------------|----------|---------------------------------------------------------------|
| `base_url`      | required | API URL including the version prefix, such as `.../v1`        |
| `api_key`       | `None`   | Sent as `Authorization: Bearer <api_key>`                     |
| `session`       | `None`   | Your own `aiohttp.ClientSession`                              |
| `headers`       | `None`   | Headers for every request                                     |
| `timeout`       | `600`    | Idle timeout in seconds, an `aiohttp.ClientTimeout` or `None` |
| `max_retries`   | `2`      | Retries for connection errors and status 408, 409, 429, 5xx   |
| `websocket_url` | derived  | `ws://` or `wss://` version of `{base_url}/responses`         |

* `create()`, `stream()` and `compact()` also take `timeout`, `extra_headers` and `extra_body`.
* Retries respect the `Retry-After` header.

## Examples

Runnable scripts in [examples](examples): basic request, streaming, tool loop and WebSocket.
Set `OPENRESPONSES_BASE_URL`, `OPENRESPONSES_API_KEY` and `OPENRESPONSES_MODEL` to run them.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

Tests use an in-process server and validate every request against the official OpenAPI
document in `tests/fixtures`.

