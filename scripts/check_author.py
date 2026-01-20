#!/usr/bin/env python3
"""Check who sent message #4."""
import asyncio
import os
from aiogram import Bot

async def test():
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    try:
        # Forward message to self to check sender
        me = await bot.get_me()
        print(f"events-bot id: {me.id}")
        
        # Get chat info
        chat = await bot.get_chat("@kenigevents")
        msg = chat.pinned_message
        if msg:
            print(f"\nPinned message #{msg.message_id}:")
            print(f"  from: {msg.from_user}")
            print(f"  sender_chat: {msg.sender_chat}")
            print(f"  author_signature: {msg.author_signature}")
            print(f"  via_bot: {msg.via_bot}")
            
            # Check if message was sent by events-bot
            if msg.from_user and msg.from_user.id == me.id:
                print("\n>>> MESSAGE WAS SENT BY events-bot - SHOULD BE EDITABLE")
            else:
                print("\n>>> MESSAGE WAS NOT SENT BY events-bot - CANNOT EDIT")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    finally:
        await bot.session.close()

asyncio.run(test())
