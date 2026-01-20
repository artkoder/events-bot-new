#!/usr/bin/env python3
"""Test: first delete buttons, then add new."""
import asyncio
import os
from aiogram import Bot
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

async def test():
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    chat_id = -1002331532485
    message_id = 4
    
    try:
        # Step 1: Try to remove buttons first (like /delbutton)
        print("Step 1: Removing buttons (empty reply_markup)...")
        try:
            result = await bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=None  # or InlineKeyboardMarkup(inline_keyboard=[])
            )
            print(f"Remove result: {result}")
        except Exception as e:
            print(f"Remove error: {type(e).__name__}: {e}")
        
        # Step 2: Add new button
        print("\nStep 2: Adding new button...")
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📅 Test Button", url="https://telegra.ph/test")]
        ])
        try:
            result = await bot.edit_message_reply_markup(
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=kb
            )
            print(f"Add result: {result}")
        except Exception as e:
            print(f"Add error: {type(e).__name__}: {e}")
        
    finally:
        await bot.session.close()

asyncio.run(test())
