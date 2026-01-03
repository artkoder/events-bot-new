"""
HumanUserClient - Telethon wrapper with natural human-like behavior.

This module provides a wrapper around Telethon that simulates realistic
user behavior to avoid triggering Telegram's anti-fraud systems.

Key features:
- Device fingerprinting (consistent per session)
- Gaussian-distributed delays (typing, reaction)
- Typing action simulation before sending
- FloodWait error handling with buffer
"""

import asyncio
import os
import random
import logging
from typing import Optional

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.functions.messages import SetTypingRequest
from telethon.tl.types import SendMessageTypingAction
from telethon.errors import FloodWaitError

logger = logging.getLogger(__name__)


class HumanUserClient:
    """
    Telethon client wrapper with human-like behavior simulation.
    
    Implements:
    - Device fingerprint (iPhone 14 Pro / iOS 17.2)
    - Gaussian-distributed delays
    - Typing action before message send
    - Read acknowledge before interaction
    - FloodWait safety wrapper
    """
    
    # Device fingerprint - consistent for session
    DEVICE_MODEL = "iPhone 14 Pro"
    SYSTEM_VERSION = "iOS 17.2"
    APP_VERSION = "10.5.1"
    LANG_CODE = "ru"
    
    # Timing parameters
    CHARS_PER_MINUTE = 200  # Average typing speed
    MIN_REACTION_SEC = 1.5
    MAX_REACTION_SEC = 4.0
    TYPING_CHUNK_SEC = 3.0  # Re-send typing action every N seconds
    
    def __init__(
        self,
        session_string: str,
        api_id: int,
        api_hash: str,
        trust_level: float = 0.5,
    ):
        """
        Initialize HumanUserClient.
        
        Args:
            session_string: Telethon StringSession
            api_id: Telegram API ID
            api_hash: Telegram API Hash
            trust_level: 0.0-1.0, higher = longer delays (for new accounts)
        """
        self.trust_level = trust_level
        self._connected = False
        
        self.client = TelegramClient(
            StringSession(session_string),
            api_id,
            api_hash,
            device_model=self.DEVICE_MODEL,
            system_version=self.SYSTEM_VERSION,
            app_version=self.APP_VERSION,
            lang_code=self.LANG_CODE,
            system_lang_code=self.LANG_CODE,
        )
    
    async def connect(self) -> None:
        """Connect to Telegram and verify authorization."""
        await self.client.connect()
        
        if not await self.client.is_user_authorized():
            raise ConnectionError(
                "Client is not authorized. Check TELEGRAM_SESSION."
            )
        
        self._connected = True
        me = await self.client.get_me()
        logger.info(f"Connected as: {me.first_name} (ID: {me.id})")
    
    async def disconnect(self) -> None:
        """Disconnect from Telegram."""
        if self._connected:
            await self.client.disconnect()
            self._connected = False
            logger.info("Disconnected from Telegram")
    
    async def _gaussian_delay(self, min_sec: float, max_sec: float) -> None:
        """
        Sleep for Gaussian-distributed duration.
        
        Uses normal distribution centered between min and max,
        adjusted by trust_level (higher trust = longer delays).
        """
        mean = (min_sec + max_sec) / 2
        std = (max_sec - min_sec) / 4
        delay = max(min_sec, random.gauss(mean, std))
        
        # Adjust for trust level (new accounts wait longer)
        adjusted_delay = delay * (1 + self.trust_level * 0.5)
        
        logger.debug(f"Sleeping for {adjusted_delay:.2f}s (gaussian)")
        await asyncio.sleep(adjusted_delay)
    
    async def _simulate_typing(self, chat_id, duration_sec: float) -> None:
        """
        Simulate typing action for specified duration.
        
        Refreshes typing status every TYPING_CHUNK_SEC to maintain
        the "typing..." indicator.
        """
        elapsed = 0.0
        
        while elapsed < duration_sec:
            try:
                await self.client(SetTypingRequest(
                    peer=chat_id,
                    action=SendMessageTypingAction()
                ))
            except Exception as e:
                logger.warning(f"Failed to set typing action: {e}")
            
            chunk = min(self.TYPING_CHUNK_SEC, duration_sec - elapsed)
            await asyncio.sleep(chunk)
            elapsed += chunk
    
    async def human_send_message(
        self,
        chat_id,
        text: str,
        skip_read: bool = False,
    ):
        """
        Send message with human-like behavior simulation.
        
        Flow:
        1. Mark chat as read (acknowledge previous messages)
        2. Wait for "reaction time" (reading/thinking)
        3. Simulate typing based on message length
        4. Send the message
        
        Args:
            chat_id: Chat/user/bot to send to (username or entity)
            text: Message text to send
            skip_read: Skip read acknowledge (for first message)
            
        Returns:
            Sent message object
        """
        entity = await self.client.get_entity(chat_id)
        
        # 1. Mark as read (human reads before responding)
        if not skip_read:
            try:
                await self.client.send_read_acknowledge(entity)
                logger.debug(f"Marked {chat_id} as read")
            except Exception as e:
                logger.warning(f"Failed to mark as read: {e}")
        
        # 2. Reaction time (reading/thinking)
        await self._gaussian_delay(
            self.MIN_REACTION_SEC,
            self.MAX_REACTION_SEC
        )
        
        # 3. Typing simulation
        # Calculate typing time based on message length
        typing_seconds = (len(text) / self.CHARS_PER_MINUTE) * 60
        typing_seconds = max(1.0, typing_seconds)  # Minimum 1 second
        
        # Add jitter (±20%)
        jitter = random.uniform(-0.2, 0.2)
        typing_seconds *= (1 + jitter)
        
        logger.debug(
            f"Simulating typing for {typing_seconds:.1f}s "
            f"({len(text)} chars)"
        )
        await self._simulate_typing(entity, typing_seconds)
        
        # 4. Send message with FloodWait protection
        try:
            message = await self.client.send_message(entity, text)
            logger.info(f"Sent message to {chat_id}: {text[:50]}...")
            return message
            
        except FloodWaitError as e:
            wait_time = e.seconds + random.randint(10, 30)
            logger.warning(
                f"FloodWait! Waiting {wait_time}s "
                f"(original: {e.seconds}s)"
            )
            await asyncio.sleep(wait_time)
            # Retry once
            return await self.client.send_message(entity, text)
    
    async def human_send_and_wait(
        self,
        chat_id,
        text: str,
        timeout: float = 30.0,
    ):
        """
        Send message and wait for response (for bot testing).
        
        Args:
            chat_id: Bot username or entity
            text: Command/message to send
            timeout: Max seconds to wait for response
            
        Returns:
            Response message from bot
        """
        entity = await self.client.get_entity(chat_id)
        
        async with self.client.conversation(entity, timeout=timeout) as conv:
            # Send with human behavior
            await self.human_send_message(chat_id, text, skip_read=True)
            
            # Wait and return response
            response = await conv.get_response()
            logger.info(
                f"Got response from {chat_id}: "
                f"{response.text[:100] if response.text else '[no text]'}..."
            )
            return response


def create_human_client() -> HumanUserClient:
    """
    Factory function to create HumanUserClient from environment variables.
    
    Required env vars:
    - TELEGRAM_API_ID
    - TELEGRAM_API_HASH
    - TELEGRAM_SESSION
    
    Optional:
    - E2E_TRUST_LEVEL (default: 0.5)
    """
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    session = os.environ.get("TELEGRAM_SESSION")
    
    if not all([api_id, api_hash, session]):
        raise EnvironmentError(
            "Missing required env vars: "
            "TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION"
        )
    
    trust_level = float(os.environ.get("E2E_TRUST_LEVEL", "0.5"))
    
    return HumanUserClient(
        session_string=session,
        api_id=int(api_id),
        api_hash=api_hash,
        trust_level=trust_level,
    )
