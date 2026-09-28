"""Shared test fixtures."""

from collections.abc import AsyncIterator, Iterator

import pytest
from aiohttp.test_utils import TestServer
from pytest_aiohttp import AiohttpServer

from openresponses_client import OpenResponsesClient

from .fake_server import FakeOpenResponsesServer

API_KEY = "sk-test-key"


@pytest.fixture
def fake_server() -> Iterator[FakeOpenResponsesServer]:
    """Return the fake server and check that requests matched the spec."""
    fake_server = FakeOpenResponsesServer()
    yield fake_server
    if fake_server.validate_requests:
        assert fake_server.schema_errors == []


@pytest.fixture
async def server(
    aiohttp_server: AiohttpServer, fake_server: FakeOpenResponsesServer
) -> TestServer:
    """Start the fake server."""
    return await aiohttp_server(fake_server.app)


@pytest.fixture
async def client(server: TestServer) -> AsyncIterator[OpenResponsesClient]:
    """Return a client for the fake server without retries."""
    async with OpenResponsesClient(
        server.make_url("/v1"), api_key=API_KEY, max_retries=0
    ) as client:
        yield client
