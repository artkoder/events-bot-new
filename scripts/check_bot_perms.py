#!/usr/bin/env python3
"""Check bot permissions in channel."""
import asyncio
import os
from aiogram import Bot

async def test():
    bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
    try:
        me = await bot.get_me()
        print(f"Bot: @{me.username} (id={me.id})")
        
        member = await bot.get_chat_member("@kenigevents", me.id)
        print(f"Status: {member.status}")
        print(f"Can edit messages: {getattr(member, 'can_edit_messages', 'N/A')}")
        print(f"Can post messages: {getattr(member, 'can_post_messages', 'N/A')}")
        print(f"Can delete messages: {getattr(member, 'can_delete_messages', 'N/A')}")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
    finally:
        await bot.session.close()

asyncio.run(test())
