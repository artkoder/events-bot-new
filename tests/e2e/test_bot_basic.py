"""
Basic E2E tests for bot functionality using HumanUserClient.

These tests interact with a real Telegram bot through a real user account,
simulating natural human behavior to avoid anti-fraud detection.
"""

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.asyncio,
]


class TestBotBasicCommands:
    """Test basic bot commands via real Telegram interaction."""
    
    async def test_events_command_returns_response(
        self,
        human_client,
        bot_username,
    ):
        """
        Test that /events command returns a valid response.
        
        Verifies:
        - Bot responds to /events command
        - Response contains meaningful text (>10 chars)
        """
        response = await human_client.human_send_and_wait(
            bot_username,
            "/events",
            timeout=30,
        )
        
        assert response is not None, "No response from bot"
        assert response.text, "Response has no text"
        assert len(response.text) > 10, (
            f"Response too short: {response.text}"
        )
    
    async def test_events_command_has_navigation_buttons(
        self,
        human_client,
        bot_username,
    ):
        """
        Test that /events response includes navigation buttons.
        
        Verifies:
        - Response has inline buttons
        - At least 2 buttons present (left/right navigation)
        """
        response = await human_client.human_send_and_wait(
            bot_username,
            "/events",
            timeout=30,
        )
        
        assert response.buttons, "Message has no inline buttons"
        
        # Flatten button rows
        all_buttons = []
        for row in response.buttons:
            all_buttons.extend(row)
        
        assert len(all_buttons) >= 2, (
            f"Expected at least 2 buttons, got {len(all_buttons)}"
        )
    
    async def test_start_command(
        self,
        human_client,
        bot_username,
    ):
        """
        Test that /start command works.
        
        Verifies:
        - Bot responds to /start
        - Response is non-empty
        """
        response = await human_client.human_send_and_wait(
            bot_username,
            "/start",
            timeout=30,
        )
        
        assert response is not None, "No response from bot"
        assert response.text or response.buttons, (
            "Response has neither text nor buttons"
        )
