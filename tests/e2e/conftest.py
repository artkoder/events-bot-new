"""
E2E Test Configuration - pytest fixtures for human-like Telegram testing.

Provides a session-scoped HumanUserClient fixture for E2E bot testing.
"""

import pytest
import pytest_asyncio
import asyncio
import os
import logging

from tests.e2e.human_client import HumanUserClient, create_human_client

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as E2E tests requiring real Telegram credentials"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def human_client() -> HumanUserClient:
    """
    Session-scoped fixture providing HumanUserClient.
    
    Requires environment variables:
    - TELEGRAM_API_ID
    - TELEGRAM_API_HASH
    - TELEGRAM_SESSION
    
    Skips tests if credentials are not available.
    """
    # Check for required credentials
    required = ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION"]
    missing = [var for var in required if not os.environ.get(var)]
    
    if missing:
        pytest.skip(
            f"E2E tests require Telegram credentials. "
            f"Missing: {', '.join(missing)}"
        )
    
    # Create and connect client
    client = create_human_client()
    
    try:
        await client.connect()
        logger.info("HumanUserClient connected for E2E tests")
        yield client
    finally:
        await client.disconnect()
        logger.info("HumanUserClient disconnected")


@pytest.fixture
def bot_username() -> str:
    """Target bot username for testing."""
    return os.environ.get("E2E_BOT_USERNAME", "eventsbotTestBot")
