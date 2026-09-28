"""Tests for error helpers."""

from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from openresponses_client import (
    APIStatusError,
    BadRequestError,
    ErrorCode,
    InternalServerError,
    PreviousResponseNotFoundError,
    ResponseStreamError,
    WebSocketConnectionLimitReachedError,
)
from openresponses_client.exceptions import describe_client_error, make_status_error
from openresponses_client.models import ErrorEvent


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {"error": {"message": "Nested", "code": "c", "type": "t", "param": "p"}},
            ("Nested", "c", "t", "p"),
        ),
        ({"error": "Plain"}, ("Plain", None, None, None)),
        # Flat errors use `type` for the event type, not the error type.
        ({"type": "error", "message": "Flat", "code": 7}, ("Flat", "7", None, None)),
        (
            {"message": "Bare", "type": "server_error"},
            ("Bare", None, "server_error", None),
        ),
        ({"error": None, "message": "Top"}, ("Top", None, None, None)),
        ("  text body  ", ("text body", None, None, None)),
        ("   ", ("Reason", None, None, None)),
        (None, ("Reason", None, None, None)),
        ([1, 2], ("Reason", None, None, None)),
    ],
)
def test_error_bodies(body: object, expected: tuple[Any, ...]) -> None:
    error = make_status_error(400, body, reason="Reason")
    assert (error.message, error.code, error.type, error.param) == expected
    assert error.body is body


@pytest.mark.parametrize(
    ("status", "code", "error_type"),
    [
        (400, ErrorCode.PREVIOUS_RESPONSE_NOT_FOUND, PreviousResponseNotFoundError),
        (
            429,
            ErrorCode.WEBSOCKET_CONNECTION_LIMIT_REACHED,
            WebSocketConnectionLimitReachedError,
        ),
        (400, "other", BadRequestError),
        (503, None, InternalServerError),
        (418, None, APIStatusError),
    ],
)
def test_error_classes(
    status: int, code: str | None, error_type: type[APIStatusError]
) -> None:
    error = make_status_error(status, {"error": {"code": code}})
    assert type(error) is error_type
    assert str(error) == (
        f"Request failed with status {status} (status={status}, code={code})"
        if code
        else f"Request failed with status {status} (status={status})"
    )


def test_stream_error_details() -> None:
    event = ErrorEvent.from_dict({"error": {"message": "m", "code": "c", "param": "p"}})
    error = ResponseStreamError("m", event=event)
    assert (error.code, error.type, error.param) == ("c", None, "p")
    assert (ResponseStreamError("m").code, ResponseStreamError("m").event) == (
        None,
        None,
    )


def test_client_error_descriptions() -> None:
    url = URL("https://h.example/v1?api-key=secret")
    headers = CIMultiDictProxy(CIMultiDict[str]())
    response_error = aiohttp.ClientResponseError(
        aiohttp.RequestInfo(url, "GET", headers, url), (), status=500, message="boom"
    )
    assert describe_client_error(response_error) == "500, boom"
    assert describe_client_error(aiohttp.NonHttpUrlClientError()) == (
        "NonHttpUrlClientError"
    )
    assert describe_client_error(aiohttp.ClientConnectionError("down")) == "down"
