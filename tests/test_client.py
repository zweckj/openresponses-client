"""Tests for the HTTP client."""

import asyncio
import json
from typing import Any, cast

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from yarl import URL

from openresponses_client import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    CompactionItem,
    ConflictError,
    FunctionCallOutput,
    InternalServerError,
    Message,
    NotFoundError,
    OpenResponsesClient,
    PermissionDeniedError,
    PreviousResponseNotFoundError,
    RateLimitError,
    UnprocessableEntityError,
)
from openresponses_client._serialization import build_body, dumps
from openresponses_client.client import _retry_after, _ws_redirect_target
from openresponses_client.exceptions import describe_client_error, redact_url

from .conftest import API_KEY
from .fake_server import FakeOpenResponsesServer, make_response


async def test_create(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    response = await client.create(
        model="test-model",
        input="Say hello",
        instructions="Be brief",
        temperature=0.2,
        max_output_tokens=None,
        metadata={"session": "1"},
    )

    assert response.output_text == "Hello!"
    request = fake_server.last_request
    assert request.path == "/v1/responses"
    assert request.body == {
        "model": "test-model",
        "input": "Say hello",
        "instructions": "Be brief",
        "temperature": 0.2,
        "metadata": {"session": "1"},
    }
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["User-Agent"].startswith("openresponses-client/")


async def test_create_serializes_input_items(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    fake_server.validate_requests = False  # Sends implementation-specific items.
    previous = Message.from_dict(make_response()["output"][0])

    await client.create(
        model="test-model",
        input=[
            {"role": "system", "content": "You are a pirate."},
            {"type": "message", "role": "user", "content": "Hi"},
            previous,
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
            {"type": "acme:custom_item", "payload": [1, 2]},
        ],
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
            {"type": "acme:document_search", "documents": []},
        ],
        tool_choice={
            "type": "allowed_tools",
            "tools": [{"type": "function", "name": "get_weather"}],
        },
        text={
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
            }
        },
        reasoning={"effort": "low", "summary": "auto"},
        include=["reasoning.encrypted_content"],
        extra_body={"acme_option": True, "temperature": 0.5},
    )

    body = fake_server.last_request.body
    assert body["input"] == [
        {"type": "message", "role": "system", "content": "You are a pirate."},
        {"type": "message", "role": "user", "content": "Hi"},
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello!", "annotations": []}],
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        {"type": "acme:custom_item", "payload": [1, 2]},
    ]
    assert body["tools"][1] == {"type": "acme:document_search", "documents": []}
    assert body["tool_choice"]["type"] == "allowed_tools"
    assert body["text"]["format"]["schema"] == {"type": "object"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["acme_option"] is True
    assert body["temperature"] == 0.5
    assert "stream" not in body


async def test_headers(
    server: TestServer, fake_server: FakeOpenResponsesServer
) -> None:
    async with OpenResponsesClient(
        server.make_url("/v1"), headers={"X-Org": "org"}, max_retries=0
    ) as client:
        await client.create(model="m", input="x", extra_headers={"X-Request": "1"})

    headers = fake_server.last_request.headers
    assert "Authorization" not in headers
    assert headers["X-Org"] == "org"
    assert headers["X-Request"] == "1"


@pytest.mark.parametrize(
    ("status", "body", "error_type", "code"),
    [
        (
            400,
            {
                "error": {
                    "message": "The requested model 'fake-model' does not exist.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
            BadRequestError,
            "model_not_found",
        ),
        (
            401,
            {"error": {"message": "Bad key", "type": "invalid_request"}},
            AuthenticationError,
            None,
        ),
        (403, {"error": {"message": "Forbidden"}}, PermissionDeniedError, None),
        (
            404,
            {"error": {"message": "Not found", "type": "not_found"}},
            NotFoundError,
            None,
        ),
        (409, {"error": {"message": "Conflict"}}, ConflictError, None),
        (422, {"detail": "Unprocessable"}, UnprocessableEntityError, None),
        (
            429,
            {"error": {"message": "Slow down", "type": "too_many_requests"}},
            RateLimitError,
            None,
        ),
        (
            500,
            {"error": {"message": "Boom", "type": "server_error"}},
            InternalServerError,
            None,
        ),
        (
            400,
            {"error": {"message": "Missing", "code": "previous_response_not_found"}},
            PreviousResponseNotFoundError,
            "previous_response_not_found",
        ),
        (418, {"message": "I'm a teapot"}, APIStatusError, None),
    ],
)
async def test_status_errors(
    client: OpenResponsesClient,
    fake_server: FakeOpenResponsesServer,
    status: int,
    body: dict[str, Any],
    error_type: type[APIStatusError],
    code: str | None,
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.json_response(body, status=status)

    fake_server.responses_handler = handler

    with pytest.raises(error_type) as exc_info:
        await client.create(model="fake-model", input="Hi")

    error = exc_info.value
    assert type(error) is error_type
    assert error.status == status
    assert error.code == code
    assert error.body == body
    assert str(status) in str(error)


async def test_error_details(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.json_response(
            {
                "error": {
                    "message": "The requested model 'fake-model' does not exist.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
            status=400,
            headers={"X-Request-Id": "req_1"},
        )

    fake_server.responses_handler = handler

    with pytest.raises(BadRequestError) as exc_info:
        await client.create(model="fake-model", input="Hi")

    error = exc_info.value
    assert error.message == "The requested model 'fake-model' does not exist."
    assert error.type == "invalid_request_error"
    assert error.param == "model"
    assert error.headers is not None
    assert error.headers["X-Request-Id"] == "req_1"


async def test_plain_text_error(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.Response(text="upstream exploded", status=502)

    fake_server.responses_handler = handler

    with pytest.raises(InternalServerError, match="upstream exploded"):
        await client.create(model="m", input="Hi")


async def test_invalid_json(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.Response(text="<html>proxy</html>", content_type="text/html")

    fake_server.responses_handler = handler

    with pytest.raises(APIResponseValidationError) as exc_info:
        await client.create(model="m", input="Hi")
    assert exc_info.value.body == "<html>proxy</html>"


@pytest.mark.parametrize("body", [[1, 2], {"output": "not a list"}])
async def test_invalid_response_object(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer, body: Any
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.json_response(body)

    fake_server.responses_handler = handler

    with pytest.raises(APIResponseValidationError):
        await client.create(model="m", input="Hi")


async def test_retries(
    server: TestServer,
    fake_server: FakeOpenResponsesServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = [429, 503]
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("openresponses_client.client.sleep", fake_sleep)

    async def handler(request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        if statuses:
            return web.json_response(
                {"error": {"message": "retry"}},
                status=statuses.pop(0),
                headers={"Retry-After": "0.25"},
            )
        return web.json_response(make_response())

    fake_server.responses_handler = handler

    async with OpenResponsesClient(server.make_url("/v1"), max_retries=2) as client:
        response = await client.create(model="m", input="Hi")

    assert response.output_text == "Hello!"
    assert len(fake_server.requests) == 3
    assert delays == [0.25, 0.25]


async def test_retries_exhausted(
    server: TestServer,
    fake_server: FakeOpenResponsesServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("openresponses_client.client.sleep", fake_sleep)

    async def handler(request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        return web.json_response({"error": {"message": "down"}}, status=500)

    fake_server.responses_handler = handler

    async with OpenResponsesClient(server.make_url("/v1"), max_retries=1) as client:
        with pytest.raises(InternalServerError):
            await client.create(model="m", input="Hi")

    assert len(fake_server.requests) == 2


async def test_no_retry_for_client_errors(
    server: TestServer, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        return web.json_response({"error": {"message": "bad"}}, status=400)

    fake_server.responses_handler = handler

    async with OpenResponsesClient(server.make_url("/v1"), max_retries=3) as client:
        with pytest.raises(BadRequestError):
            await client.create(model="m", input="Hi")

    assert len(fake_server.requests) == 1


async def test_retry_connection_errors(
    server: TestServer,
    fake_server: FakeOpenResponsesServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("openresponses_client.client.sleep", fake_sleep)
    attempts = 0

    async def handler(request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            assert request.transport is not None
            request.transport.close()
        elif attempts == 2:
            await asyncio.sleep(0.5)
        return web.json_response(make_response())

    fake_server.responses_handler = handler

    async with OpenResponsesClient(
        server.make_url("/v1"), max_retries=2, timeout=0.1
    ) as client:
        response = await client.create(model="m", input="Hi")

    assert response.output_text == "Hello!"
    assert attempts == 3


def test_timeout_configuration() -> None:
    custom = aiohttp.ClientTimeout(total=5)

    assert OpenResponsesClient("http://h", timeout=custom)._timeout is custom
    assert (
        OpenResponsesClient("http://h", timeout=None)._timeout
        == aiohttp.ClientTimeout()
    )
    default = OpenResponsesClient("http://h")._timeout
    assert (default.total, default.sock_read, default.sock_connect) == (None, 600, 30)


def test_retry_after_parsing() -> None:
    assert _retry_after({"retry-after-ms": "1500"}) == 1.5
    assert _retry_after({"retry-after-ms": "soon", "retry-after": "1"}) == 1.0
    assert _retry_after({"retry-after": "2"}) == 2.0
    assert _retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) is None
    assert _retry_after({"retry-after": "not a date"}) is None
    assert _retry_after({"retry-after": "3600"}) is None
    assert _retry_after({}) is None


async def test_timeout(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        await asyncio.sleep(1)
        return web.json_response(make_response())

    fake_server.responses_handler = handler

    with pytest.raises(APITimeoutError):
        await client.create(model="m", input="Hi", timeout=0.05)


async def test_connection_error(unused_tcp_port: int) -> None:
    async with OpenResponsesClient(
        f"http://127.0.0.1:{unused_tcp_port}/v1", max_retries=0
    ) as client:
        with pytest.raises(APIConnectionError):
            await client.create(model="m", input="Hi")


async def test_compact(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    compacted = await client.compact(
        model="test-model",
        input=[{"role": "user", "content": "We launch on Tuesday."}],
        prompt_cache_key="cache-key",
    )

    assert compacted.object == "response.compaction"
    assert isinstance(compacted.output[1], CompactionItem)
    assert compacted.usage is not None
    request = fake_server.last_request
    assert request.path == "/v1/responses/compact"
    assert request.body == {
        "model": "test-model",
        "input": [
            {"type": "message", "role": "user", "content": "We launch on Tuesday."}
        ],
        "prompt_cache_key": "cache-key",
    }

    # The compacted window is the base input of a new response chain.
    await client.create(
        model="test-model", input=[*compacted.output, {"role": "user", "content": "Go"}]
    )
    new_input = fake_server.last_request.body["input"]
    assert new_input[1] == {
        "id": "cmp_item_1",
        "type": "compaction",
        "encrypted_content": "opaque",
    }
    assert "previous_response_id" not in fake_server.last_request.body


async def test_compact_timeout(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, body: dict[str, Any]) -> web.StreamResponse:
        await asyncio.sleep(1)
        return web.json_response({})

    fake_server.compact_handler = handler

    with pytest.raises(APITimeoutError):
        await client.compact(model="m", input="x", timeout=0.05)


async def test_session_management(server: TestServer) -> None:
    client = OpenResponsesClient(server.make_url("/v1"))
    await client.create(model="m", input="Hi")
    session = client._session
    assert session is not None
    await client.close()
    assert session.closed

    async with aiohttp.ClientSession() as own_session:
        async with OpenResponsesClient(
            server.make_url("/v1"), session=own_session
        ) as client:
            await client.create(model="m", input="Hi")
        assert not own_session.closed

    closed_session = aiohttp.ClientSession()
    await closed_session.close()
    client = OpenResponsesClient(server.make_url("/v1"), session=closed_session)
    with pytest.raises(RuntimeError, match="closed"):
        await client.create(model="m", input="Hi")


async def test_session_with_raise_for_status(
    server: TestServer, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.json_response(
            {"error": {"message": "Missing", "code": "previous_response_not_found"}},
            status=400,
        )

    fake_server.responses_handler = handler
    fake_server.ws_reject_status = 401

    async with (
        aiohttp.ClientSession(raise_for_status=True) as session,
        OpenResponsesClient(
            server.make_url("/v1"), session=session, max_retries=2
        ) as client,
    ):
        with pytest.raises(PreviousResponseNotFoundError):
            await client.create(model="m", input="Hi", previous_response_id="r")
        with pytest.raises(AuthenticationError):
            await client.websocket().connect()

    assert len(fake_server.requests) == 1


def test_client_repr_hides_key() -> None:
    client = OpenResponsesClient("https://api.example.com/v1", api_key="secret")

    assert "secret" not in repr(client)
    assert str(client.base_url) == "https://api.example.com/v1"


async def test_list_models(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    models = await client.list_models(timeout=5)

    assert [model.id for model in models] == ["test-model", "other-model"]
    assert (models[0].created, models[0].owned_by) == (1_764_967_971, "test")
    assert models[1].extra == {"context_length": 8192}
    request = fake_server.last_request
    assert (request.method, request.path, request.body) == ("GET", "/v1/models", None)
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert "Content-Type" not in request.headers


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"data": [{"id": "a"}]}, ["a"]),
        (["a", "b"], None),
        ({"models": []}, None),
        ({"data": ["a"]}, None),
        ({"data": [{"id": {"not": "a string"}}]}, None),
    ],
)
async def test_list_models_response_shapes(
    client: OpenResponsesClient,
    fake_server: FakeOpenResponsesServer,
    body: Any,
    expected: list[str] | None,
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.json_response(body)

    fake_server.models_handler = handler

    if expected is None:
        with pytest.raises(APIResponseValidationError):
            await client.list_models()
    else:
        assert [model.id for model in await client.list_models()] == expected


async def test_list_models_not_supported(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    async def handler(request: web.Request, _: dict[str, Any]) -> web.StreamResponse:
        return web.json_response({"error": {"message": "Not found"}}, status=404)

    fake_server.models_handler = handler

    with pytest.raises(NotFoundError):
        await client.list_models()


async def test_hosted_tool_choice(
    client: OpenResponsesClient, fake_server: FakeOpenResponsesServer
) -> None:
    fake_server.validate_requests = False  # Hosted tools are implementation-specific.

    await client.create(
        model="m",
        input="Draw a cat",
        tools=[{"type": "image_generation", "output_format": "png"}],
        tool_choice={"type": "image_generation"},
    )

    assert fake_server.last_request.body["tool_choice"] == {"type": "image_generation"}


def test_credentials_are_redacted() -> None:
    assert (
        redact_url("https://u:p@h.example/v1?api-key=secret#x")
        == "https://h.example/v1"
    )
    assert redact_url(object()) == "<URL>"
    for err in (
        aiohttp.NonHttpUrlRedirectClientError("wss://h.example/v1?api-key=secret"),
        aiohttp.InvalidUrlClientError("https://h.example/v1?api-key=secret"),
    ):
        assert "secret" not in describe_client_error(err)
        assert "h.example/v1" in describe_client_error(err)
    client = OpenResponsesClient("https://h.example/v1?api-key=secret")
    assert "secret" not in repr(client)


async def test_connection_error_hides_credentials(unused_tcp_port: int) -> None:
    async with OpenResponsesClient(
        f"http://127.0.0.1:{unused_tcp_port}/v1?api-key=secret", max_retries=0
    ) as client:
        with pytest.raises(APIConnectionError) as exc_info:
            await client.create(model="m", input="Hi")

    assert "secret" not in str(exc_info.value)


def test_ws_redirect_target() -> None:
    current = URL("wss://h.example/v1/responses")
    same = "wss://h.example/v1/responses?api-key=k"

    assert _ws_redirect_target(current, same) == URL(same)
    assert _ws_redirect_target(URL("https://h.example/v1/responses"), same) == URL(same)
    assert _ws_redirect_target(URL("ws://h.example/v1"), "wss://h.example/v1") is None
    for location in (
        "ws://h.example/v1/responses",
        "wss://other.example/v1/responses",
        "wss://h.example:8443/v1/responses",
        "https://h.example/v1/responses",
        "wss://[invalid",
    ):
        assert _ws_redirect_target(current, location) is None, location


def test_body_encoding() -> None:
    message = Message.from_dict(
        {"type": "message", "role": "assistant", "content": "Hi"}
    )
    body = build_body(
        {
            "model": "m",
            "input": ({"role": "user", "content": "Hi"}, message),
            "top_p": None,
        },
        {"acme": 1},
    )

    assert json.loads(dumps(body)) == {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "Hi"},
            message.to_dict(),
        ],
        "acme": 1,
    }
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps({"value": object()})
    # A tool result must be JSON text, not a number.
    invalid = FunctionCallOutput(call_id="c", output=cast(Any, 21))
    with pytest.raises(TypeError, match="Invalid FunctionCallOutput"):
        dumps({"input": [invalid]})
