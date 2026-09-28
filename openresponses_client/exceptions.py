"""Exceptions raised by the client."""

from collections.abc import Mapping
from typing import Any, override

import aiohttp
from yarl import URL

from .const import ErrorCode
from .models import ErrorEvent, ErrorPayload

__all__ = [
    "APIConnectionError",
    "APIError",
    "APIResponseValidationError",
    "APIStatusError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "ConflictError",
    "InternalServerError",
    "NotFoundError",
    "OpenResponsesError",
    "PermissionDeniedError",
    "PreviousResponseNotFoundError",
    "RateLimitError",
    "ResponseStreamError",
    "UnprocessableEntityError",
    "WebSocketClosedError",
    "WebSocketConnectionLimitReachedError",
]


class OpenResponsesError(Exception):
    """Base class of all exceptions of this library."""


class APIError(OpenResponsesError):
    """Error while talking to the server."""

    def __init__(self, message: str, *, body: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.body = body


class APIConnectionError(APIError):
    """The server is unreachable or the connection broke."""


class APITimeoutError(APIConnectionError):
    """A request or stream timed out."""


class WebSocketClosedError(APIConnectionError):
    """The WebSocket connection closed."""

    def __init__(
        self,
        message: str = "WebSocket connection closed",
        *,
        close_code: int | None = None,
        body: object | None = None,
    ) -> None:
        super().__init__(message, body=body)
        self.close_code = close_code


class APIResponseValidationError(APIError):
    """The server sent data that is not valid."""


class ResponseStreamError(APIError):
    """A stream ended without a final response."""

    def __init__(
        self,
        message: str,
        *,
        event: ErrorEvent | None = None,
        body: object | None = None,
    ) -> None:
        super().__init__(message, body=body)
        error = event.error if event is not None else ErrorPayload()
        self.event = event
        self.code = error.code
        self.type = error.type
        self.param = error.param


class APIStatusError(APIError):
    """The server answered with an error status."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        body: object | None = None,
        code: str | None = None,
        type: str | None = None,
        param: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message, body=body)
        self.status = status
        self.code = code
        self.type = type
        self.param = param
        self.headers = headers

    @override
    def __str__(self) -> str:
        code = f", code={self.code}" if self.code else ""
        return f"{self.message} (status={self.status}{code})"


class BadRequestError(APIStatusError):
    """Status 400: the request is invalid."""


class AuthenticationError(APIStatusError):
    """Status 401: the API key is missing or invalid."""


class PermissionDeniedError(APIStatusError):
    """Status 403: the API key lacks permission."""


class NotFoundError(APIStatusError):
    """Status 404: the model or endpoint does not exist."""


class ConflictError(APIStatusError):
    """Status 409: the request conflicts with the current state."""


class UnprocessableEntityError(APIStatusError):
    """Status 422: the request could not be processed."""


class RateLimitError(APIStatusError):
    """Status 429: too many requests."""


class InternalServerError(APIStatusError):
    """Status 5xx: the server failed."""


class PreviousResponseNotFoundError(BadRequestError):
    """The `previous_response_id` is unknown; resend the full history."""


class WebSocketConnectionLimitReachedError(APIStatusError):
    """The WebSocket reached its lifetime limit; reconnect."""


_STATUS_ERRORS: dict[int, type[APIStatusError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}

_CODE_ERRORS: dict[str, type[APIStatusError]] = {
    ErrorCode.PREVIOUS_RESPONSE_NOT_FOUND: PreviousResponseNotFoundError,
    ErrorCode.WEBSOCKET_CONNECTION_LIMIT_REACHED: WebSocketConnectionLimitReachedError,
}


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _error_details(body: object) -> Mapping[str, Any]:
    """Return the error details of an error response body."""
    match body:
        case {"error": Mapping() as nested}:
            return nested
        case {"error": str() as message}:
            return {"message": message}
        case Mapping() if body.get("type") == "error":
            # A flat error, where `type` is the event type.
            return {key: value for key, value in body.items() if key != "type"}
        case Mapping():
            return body
        case str():
            return {"message": body.strip()}
    return {}


def make_status_error(
    status: int,
    body: object | None,
    *,
    headers: Mapping[str, str] | None = None,
    reason: str | None = None,
) -> APIStatusError:
    """Create the most specific `APIStatusError` for an error response."""
    error = _error_details(body)
    code = _optional_str(error.get("code"))
    cls = (
        _CODE_ERRORS.get(code or "")
        or _STATUS_ERRORS.get(status)
        or (InternalServerError if status >= 500 else APIStatusError)
    )
    return cls(
        _optional_str(error.get("message"))
        or reason
        or f"Request failed with status {status}",
        status=status,
        body=body,
        code=code,
        type=_optional_str(error.get("type")),
        param=_optional_str(error.get("param")),
        headers=headers,
    )


def redact_url(url: object) -> str:
    """Remove the query, fragment and credentials of a URL for safe logging."""
    try:
        parsed = URL(str(url))
        return str(parsed.with_query(None).with_fragment(None).with_user(None))
    except TypeError, ValueError:
        return "<URL>"


def describe_client_error(err: BaseException) -> str:
    """Describe an aiohttp error without URLs that may carry credentials."""
    match err:
        case aiohttp.ClientResponseError(status=status, message=message):
            return f"{status}, {message}"
        case aiohttp.InvalidURL(url=url):
            return f"{type(err).__name__}: {redact_url(url)}"
        case aiohttp.NonHttpUrlClientError(args=[location, *_]):
            return f"{type(err).__name__}: {redact_url(location)}"
    return str(err) or type(err).__name__
