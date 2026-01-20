#!/usr/bin/env python3
"""Check message type and details."""
import asyncio
import os
from aiogram import Bot

async def test():
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    try:
        chat = await bot.get_chat("@kenigevents")
        print(f"Chat ID: {chat.id}")
        
        if chat.pinned_message:
            msg = chat.pinned_message
            print(f"\nPinned message ID: {msg.message_id}")
            print(f"Content type: {msg.content_type}")
            print(f"Has text: {msg.text is not None}")
            print(f"Has caption: {msg.caption is not None}")
            print(f"Has media group id: {msg.media_group_id}")
            print(f"Reply markup: {msg.reply_markup}")
            print(f"Forward from chat: {msg.forward_from_chat}")
            print(f"Forward from message id: {msg.forward_from_message_id}")
            if msg.text:
                print(f"Text preview: {msg.text[:100]}...")
        else:
            print("No pinned message")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    finally:
        await bot.session.close()

asyncio.run(test())
