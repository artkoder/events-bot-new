"""
Behave environment hooks for E2E Telegram bot testing.

Initializes HumanUserClient before test suite and cleans up after.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.e2e.human_client import HumanUserClient, create_human_client

logger = logging.getLogger("e2e.behave")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def before_all(context):
    """Initialize async event loop and HumanUserClient before all tests."""
    logger.info("=" * 60)
    logger.info("E2E BDD Test Suite Starting")
    logger.info("=" * 60)
    
    # Create event loop for async operations
    context.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(context.loop)
    
    # Check required environment variables
    required = ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_SESSION"]
    missing = [var for var in required if not os.environ.get(var)]
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    
    # Initialize HumanUserClient
    context.client = create_human_client()
    context.loop.run_until_complete(context.client.connect())
    
    # Bot username to test
    context.bot_username = os.environ.get(
        "E2E_BOT_USERNAME", "eventsbotTestBot"
    )
    
    # Store for last message/response
    context.last_message = None
    context.last_response = None
    
    logger.info(f"Connected! Testing bot: @{context.bot_username}")


def after_all(context):
    """Disconnect client after all tests."""
    if hasattr(context, "client") and context.client:
        context.loop.run_until_complete(context.client.disconnect())
        logger.info("Client disconnected")
    
    if hasattr(context, "loop") and context.loop:
        context.loop.close()
    
    logger.info("=" * 60)
    logger.info("E2E BDD Test Suite Finished")
    logger.info("=" * 60)


def before_scenario(context, scenario):
    """Log scenario start."""
    logger.info(f"\n📌 Сценарий: {scenario.name}")


def after_scenario(context, scenario):
    """Log scenario result."""
    if scenario.status == "passed":
        logger.info(f"✅ Сценарий PASSED: {scenario.name}")
    else:
        logger.error(f"❌ Сценарий FAILED: {scenario.name}")
