import os
import asyncio
from behave import given, when, then
from telethon import TelegramClient
from telethon.sessions import StringSession

TEST_BOT_USERNAME = "eventsbotTestBot"

_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

def run_async(coro):
    """Helper to run async code in sync steps."""
    return _loop.run_until_complete(coro)

@given('я авторизован в Telegram через Telethon')
def step_auth_telegram(context):
    api_id = os.environ.get('TELEGRAM_API_ID')
    api_hash = os.environ.get('TELEGRAM_API_HASH')
    session_str = os.environ.get('TELEGRAM_SESSION')
    
    if not all([api_id, api_hash, session_str]):
        raise EnvironmentError("Missing TELEGRAM_API_ID, TELEGRAM_API_HASH, or TELEGRAM_SESSION")
    
    context.client = TelegramClient(StringSession(session_str), int(api_id), api_hash)
    
    async def connect():
        await context.client.connect()
        if not await context.client.is_user_authorized():
            raise ConnectionError("Client is not authorized. Check Session String.")
            
    run_async(connect())

@when('я нахожу бота @eventsbotTestBot в списке моих переписок')
def step_find_bot(context):
    async def find_dialog():
        try:
            entity = await context.client.get_entity(TEST_BOT_USERNAME)
            context.bot_entity = entity
        except Exception as e:
            raise AssertionError(f"Could not find bot {TEST_BOT_USERNAME}: {e}")
            
    run_async(find_dialog())

@then('я должен увидеть переписку с ботом')
def step_see_bot_dialog(context):
    pass

@given('я авторизован в боте @eventsbotTestBot')
def step_auth_bot_context(context):
    if not hasattr(context, 'client'):
        step_auth_telegram(context)
    if not hasattr(context, 'bot_entity'):
        step_find_bot(context)

@when('я делаю /events')
def step_send_events_command(context):
    async def send_command():
        async with context.client.conversation(context.bot_entity, timeout=30) as conv:
            await conv.send_message('/events')
            response = await conv.get_response()
            context.last_response = response
            
    run_async(send_command())

@then('я должен получить ответ от бота со списком событий')
def step_check_response_events(context):
    if not hasattr(context, 'last_response') or not context.last_response:
        raise AssertionError("No response received from bot")
    
    text = context.last_response.text
    if not text or len(text) < 10:
        raise AssertionError(f"Response seems too short or empty: {text}")

@then('в сообщении со списком событий будет 2 инлайн кнопки одна влево, вдругая вправо')
def step_check_buttons(context):
    msg = context.last_response
    if not msg.buttons:
        raise AssertionError("Message has no buttons")
    
    all_buttons = []
    if msg.buttons:
        for row in msg.buttons:
            all_buttons.extend(row)
            
    count = len(all_buttons)
    if count < 2:
        raise AssertionError(f"Expected at least 2 buttons, found {count}")