"""HTTP client for Open Responses servers."""

import json
import logging
import random
from asyncio import sleep
from collections.abc import Mapping
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from types import TracebackType
from typing import Any, Final, Self, Unpack, override

import aiohttp
from yarl import URL

from ._serialization import build_body, dumps
from .const import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_WS_MAX_MSG_SIZE,
)
from .exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    describe_client_error,
    make_status_error,
    redact_url,
)
from .models import CompactResponse, ModelInfo, ModelList, OpenResponsesModel, Response
from .models.base import PARSE_ERRORS
from .params import CreateResponseParams, ResponseInput, StreamResponseParams
from .streaming import ResponseStream
from .websocket import ResponsesWebSocket

__all__ = ["OpenResponsesClient"]

_LOGGER = logging.getLogger(__name__)

try:
    _VERSION = version("openresponses-client")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    _VERSION = "0.0.0"

USER_AGENT: Final = f"openresponses-client/{_VERSION}"

_INITIAL_RETRY_DELAY: Final = 0.5
_MAX_RETRY_DELAY: Final = 8.0
_MAX_RETRY_AFTER: Final = 60.0
_UNSET: Final[Any] = object()
_MAX_WS_REDIRECTS: Final = 3

type Timeout = float | aiohttp.ClientTimeout | None


def _should_retry(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Return the delay requested by `Retry-After` headers, if reasonable."""
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            delay = float(headers[name]) * scale
        except KeyError, ValueError:
            continue
        return delay if 0 < delay <= _MAX_RETRY_AFTER else None
    return None


def _backoff(attempt: int) -> float:
    """Return an exponential backoff delay with jitter."""
    delay = min(_INITIAL_RETRY_DELAY * 2.0**attempt, _MAX_RETRY_DELAY)
    return delay * (1 - 0.25 * random.random())  # noqa: S311


def _with_path(url: URL, path: str) -> URL:
    """Replace the path of a URL, keeping its query."""
    return url.with_path(path).with_query(url.query)


def _websocket_scheme(url: URL) -> URL:
    """Switch an `http(s)://` URL to `ws(s)://`."""
    return url.with_scheme({"http": "ws", "https": "wss"}.get(url.scheme, url.scheme))


def _ws_redirect_target(current: URL, location: object) -> URL | None:
    """Return the redirect target if it is the same host, without downgrade."""
    try:
        target = URL(str(location))
    except ValueError:
        return None
    same_host = (target.host, target.port) == (current.host, current.port)
    downgrade = current.scheme in ("https", "wss") and target.scheme == "ws"
    websocket = target.scheme in ("ws", "wss")
    return target if websocket and same_host and not downgrade else None


def _to_timeout(value: Timeout) -> aiohttp.ClientTimeout:
    """Convert a timeout option into an `aiohttp.ClientTimeout`."""
    match value:
        case aiohttp.ClientTimeout():
            return value
        case None:
            return aiohttp.ClientTimeout()
        case _:
            return aiohttp.ClientTimeout(
                sock_connect=min(value, DEFAULT_CONNECT_TIMEOUT), sock_read=value
            )


class OpenResponsesClient:
    """Async client for an Open Responses server."""

    def __init__(
        self,
        base_url: str | URL,
        *,
        api_key: str | None = None,
        session: aiohttp.ClientSession | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: Timeout = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        websocket_url: str | URL | None = None,
    ) -> None:
        base = URL(base_url)
        self._base_url = _with_path(base, base.path.rstrip("/"))
        self._api_key = api_key
        self._session = session
        self._owns_session = session is None
        self._headers = dict(headers or {})
        self._timeout = _to_timeout(timeout)
        self._max_retries = max(0, max_retries)
        self._websocket_url = (
            URL(websocket_url)
            if websocket_url is not None
            else _websocket_scheme(self._endpoint("responses"))
        )

    @override
    def __repr__(self) -> str:
        """Return a representation without credentials."""
        return f"{type(self).__name__}(base_url={redact_url(self._base_url)!r})"

    @property
    def base_url(self) -> URL:
        """The base URL of the API."""
        return self._base_url

    @property
    def websocket_url(self) -> URL:
        """The URL used for WebSocket mode."""
        return self._websocket_url

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the session if the client created it."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def create(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: Timeout = _UNSET,
        **params: Unpack[CreateResponseParams],
    ) -> Response:
        """Create a response and wait for it."""
        body = build_body(params, extra_body)
        data = await self._request_json(
            "POST", "responses", body, extra_headers, timeout
        )
        return self._validate(Response, data)

    def stream(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: Timeout = _UNSET,
        **params: Unpack[StreamResponseParams],
    ) -> ResponseStream:
        """Create a response and stream its events."""
        body = {**build_body(params, extra_body), "stream": True}
        headers = self._request_headers(
            extra_headers, accept="text/event-stream", json_body=True
        )
        return ResponseStream(
            partial(
                self._request,
                "POST",
                self._endpoint("responses"),
                body,
                headers,
                self._request_timeout(timeout),
            )
        )

    async def compact(
        self,
        *,
        model: str,
        input: ResponseInput | None = None,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        prompt_cache_key: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: Timeout = _UNSET,
    ) -> CompactResponse:
        """Compact a conversation into a smaller input window."""
        params = {
            "model": model,
            "input": input,
            "previous_response_id": previous_response_id,
            "instructions": instructions,
            "prompt_cache_key": prompt_cache_key,
        }
        data = await self._request_json(
            "POST",
            "responses/compact",
            build_body(params, extra_body),
            extra_headers,
            timeout,
        )
        return self._validate(CompactResponse, data)

    async def list_models(
        self,
        *,
        extra_headers: Mapping[str, str] | None = None,
        timeout: Timeout = _UNSET,
    ) -> list[ModelInfo]:
        """List the models of the server (`GET /models`, not part of the spec)."""
        data = await self._request_json("GET", "models", None, extra_headers, timeout)
        return self._validate(ModelList, data).data

    def websocket(
        self,
        *,
        auto_reconnect: bool = True,
        heartbeat: float | None = None,
        receive_timeout: float | None = None,
        max_msg_size: int = DEFAULT_WS_MAX_MSG_SIZE,
        extra_headers: Mapping[str, str] | None = None,
    ) -> ResponsesWebSocket:
        """Create a WebSocket connection for multi-turn responses."""
        headers = self._request_headers(extra_headers)

        async def _connect() -> aiohttp.ClientWebSocketResponse:
            return await self._ws_connect(
                headers,
                heartbeat=heartbeat,
                max_msg_size=max_msg_size,
            )

        return ResponsesWebSocket(
            _connect, auto_reconnect=auto_reconnect, receive_timeout=receive_timeout
        )

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the session, creating one if the client owns it."""
        if self._session is None or self._session.closed:
            if not self._owns_session:
                raise RuntimeError("The aiohttp session passed to the client is closed")
            self._session = aiohttp.ClientSession()
        return self._session

    def _endpoint(self, path: str) -> URL:
        return _with_path(self._base_url, f"{self._base_url.path.rstrip('/')}/{path}")

    def _request_headers(
        self,
        extra_headers: Mapping[str, str] | None,
        *,
        accept: str | None = None,
        json_body: bool = False,
    ) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        if json_body:
            headers["Content-Type"] = "application/json"
        if accept is not None:
            headers["Accept"] = accept
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers.update(self._headers)
        headers.update(extra_headers or {})
        return headers

    def _request_timeout(self, timeout: Timeout) -> aiohttp.ClientTimeout:
        return self._timeout if timeout is _UNSET else _to_timeout(timeout)

    async def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        extra_headers: Mapping[str, str] | None,
        timeout: Timeout,
    ) -> Any:
        """Send a request and decode its JSON body."""
        headers = self._request_headers(
            extra_headers, accept="application/json", json_body=body is not None
        )
        response = await self._request(
            method, self._endpoint(path), body, headers, self._request_timeout(timeout)
        )
        try:
            raw = await response.read()
        except TimeoutError as err:
            raise APITimeoutError("Timed out while reading the response") from err
        except aiohttp.ClientError as err:
            raise APIConnectionError(
                f"Error while reading the response: {describe_client_error(err)}"
            ) from err
        finally:
            response.release()
        try:
            return json.loads(raw)
        except ValueError as err:
            text = raw.decode("utf-8", errors="replace")
            raise APIResponseValidationError(
                f"Expected a JSON response but received: {text[:200]!r}", body=text
            ) from err

    async def _request(
        self,
        method: str,
        url: URL,
        body: dict[str, Any] | None,
        headers: Mapping[str, str],
        timeout: aiohttp.ClientTimeout,
    ) -> aiohttp.ClientResponse:
        """Send a request with retries and return a successful response."""
        data = dumps(body).encode() if body is not None else None
        session = self._get_session()
        attempt = 0
        while True:
            _LOGGER.debug("%s %s (attempt %s)", method, redact_url(url), attempt + 1)
            last_attempt = attempt >= self._max_retries
            delay = _backoff(attempt)
            try:
                response = await session.request(
                    method,
                    url,
                    data=data,
                    headers=headers,
                    timeout=timeout,
                    raise_for_status=False,
                )
            except TimeoutError as err:
                if last_attempt:
                    raise APITimeoutError(
                        f"Request to {redact_url(url)} timed out"
                    ) from err
            except aiohttp.ClientError as err:
                if last_attempt:
                    raise APIConnectionError(
                        f"Error connecting to {redact_url(url)}: "
                        f"{describe_client_error(err)}"
                    ) from err
            else:
                if response.status < 400:
                    return response
                if last_attempt or not _should_retry(response.status):
                    raise await self._status_error(response)
                delay = _retry_after(response.headers) or delay
                response.release()
                _LOGGER.debug(
                    "Retrying after status %s in %.2fs", response.status, delay
                )
            await sleep(delay)
            attempt += 1

    @staticmethod
    async def _status_error(response: aiohttp.ClientResponse) -> APIStatusError:
        """Build the exception for an error response."""
        try:
            raw = await response.read()
        except aiohttp.ClientError, TimeoutError:
            raw = b""
        finally:
            response.release()
        text = raw.decode("utf-8", errors="replace")
        body: object
        try:
            body = json.loads(text) if text else None
        except ValueError:
            body = text
        return make_status_error(
            response.status, body, headers=response.headers, reason=response.reason
        )

    async def _ws_connect(
        self,
        headers: Mapping[str, str],
        *,
        heartbeat: float | None,
        max_msg_size: int,
    ) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket, following same-host `ws(s)://` redirects."""
        url = self.websocket_url
        for _redirect in range(_MAX_WS_REDIRECTS + 1):
            _LOGGER.debug("Connecting to %s", redact_url(url))
            try:
                return await self._get_session().ws_connect(
                    url,
                    headers=headers,
                    heartbeat=heartbeat,
                    max_msg_size=max_msg_size,
                    timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
                )
            except aiohttp.NonHttpUrlRedirectClientError as err:
                # aiohttp only follows http(s) redirects, but servers such as
                # Azure OpenAI redirect the handshake to a wss:// URL.
                location = err.args[0] if err.args else None
                target = _ws_redirect_target(url, location)
                if target is None:
                    raise APIConnectionError(
                        f"Refusing to follow the WebSocket redirect to "
                        f"{redact_url(location)}"
                    ) from err
                url = target
            except aiohttp.ClientResponseError as err:
                # Handshake failures, also from sessions with raise_for_status.
                raise make_status_error(
                    err.status,
                    None,
                    headers=err.headers,
                    reason=f"WebSocket handshake failed: {err.message}",
                ) from err
            except TimeoutError as err:
                raise APITimeoutError(
                    f"Timed out connecting to {redact_url(url)}"
                ) from err
            except aiohttp.ClientError as err:
                raise APIConnectionError(
                    f"Error connecting to {redact_url(url)}: "
                    f"{describe_client_error(err)}"
                ) from err
        raise APIConnectionError("Too many WebSocket redirects")

    @staticmethod
    def _validate[M: OpenResponsesModel](model: type[M], data: Any) -> M:
        """Validate data into a model or raise `APIResponseValidationError`."""
        try:
            return model.from_dict(data)
        except PARSE_ERRORS as err:
            raise APIResponseValidationError(
                f"Received an invalid {model.__name__}: {err}", body=data
            ) from err
