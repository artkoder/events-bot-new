#!/usr/bin/env python3
"""Test raw API call like cat-weather-new uses."""
import asyncio
import os
import aiohttp
import json

async def api_request(token, method, payload):
    url = f"https://api.telegram.org/bot{token}/{method}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            return await resp.json()

async def test():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = -1002331532485
    message_id = 4
    
    # Test 1: Try editMessageReplyMarkup with raw request (like cat-weather-new)
    keyboard = {
        "inline_keyboard": [
            [{"text": "📅 Test Raw", "url": "https://telegra.ph/test"}]
        ]
    }
    
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "reply_markup": keyboard
    }
    
    print(f"Calling editMessageReplyMarkup with payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    
    result = await api_request(token, "editMessageReplyMarkup", payload)
    print(f"\nResult: {json.dumps(result, indent=2, ensure_ascii=False)}")

asyncio.run(test())
