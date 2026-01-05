import asyncio
import os
import sys
from aiogram import Bot

async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not set")
        return

    bot = Bot(token=token)
    try:
        chat = await bot.get_chat("@keniggpt")
        print(f"Channel Title: {chat.title}")
        print(f"Channel ID: {chat.id}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
