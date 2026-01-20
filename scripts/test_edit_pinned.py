#!/usr/bin/env python3
"""Test edit_message_reply_markup on pinned message."""
import asyncio
import os
from aiogram import Bot
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

async def test():
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        return
    
    bot = Bot(token=token)
    
    try:
        # Get chat info
        chat = await bot.get_chat('@kenigevents')
        print(f"Chat ID: {chat.id}")
        print(f"Chat type: {chat.type}")
        pinned_id = chat.pinned_message.message_id if chat.pinned_message else None
        print(f"Pinned message ID: {pinned_id}")
        
        # Create keyboard
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text='📅 Test Button', url='https://telegra.ph/test')]
        ])
        
        # Try edit
        print(f"\nTrying edit_message_reply_markup(chat_id={chat.id}, message_id=4)...")
        result = await bot.edit_message_reply_markup(
            chat_id=chat.id,
            message_id=4,
            reply_markup=kb
        )
        print(f"SUCCESS: {result}")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(test())
