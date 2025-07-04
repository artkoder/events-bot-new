import logging
import os
from datetime import datetime
from typing import Optional

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from aiohttp import web, ClientSession
from telegraph import Telegraph
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import Field, SQLModel, select

logging.basicConfig(level=logging.INFO)


DB_PATH = os.getenv("DB_PATH", "/data/db.sqlite")


class User(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    is_superadmin: bool = False


class PendingUser(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    requested_at: datetime = Field(default_factory=datetime.utcnow)


class RejectedUser(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    rejected_at: datetime = Field(default_factory=datetime.utcnow)


class Setting(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: str
    festival: Optional[str] = None
    date: str
    time: str
    location_name: str
    location_address: Optional[str] = None
    city: Optional[str] = None
    source_text: str


class Database:
    def __init__(self, path: str):
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{path}")

    async def init(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    def get_session(self) -> AsyncSession:
        return AsyncSession(self.engine)


async def get_tz_offset(db: Database) -> str:
    async with db.get_session() as session:
        result = await session.get(Setting, "tz_offset")
        return result.value if result else "+00:00"



async def set_tz_offset(db: Database, value: str):
    async with db.get_session() as session:
        setting = await session.get(Setting, "tz_offset")
        if setting:
            setting.value = value
        else:
            setting = Setting(key="tz_offset", value=value)
            session.add(setting)
        await session.commit()



def validate_offset(value: str) -> bool:
    if len(value) != 6 or value[0] not in "+-" or value[3] != ":":
        return False
    try:
        h = int(value[1:3])
        m = int(value[4:6])
        return 0 <= h <= 14 and 0 <= m < 60
    except ValueError:
        return False



async def parse_event_via_4o(text: str) -> dict:
    token = os.getenv("FOUR_O_TOKEN")
    if not token:
        raise RuntimeError("FOUR_O_TOKEN is missing")
    url = os.getenv("FOUR_O_URL", "https://api.example.com/parse")
    prompt_path = os.path.join("docs", "PROMPTS.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    headers = {"Authorization": f"Bearer {token}"}
    async with ClientSession() as session:
        resp = await session.post(url, json={"text": text, "prompt": prompt}, headers=headers)
        resp.raise_for_status()
        return await resp.json()


async def ask_4o(text: str) -> str:
    token = os.getenv("FOUR_O_TOKEN")
    if not token:
        raise RuntimeError("FOUR_O_TOKEN is missing")
    url = os.getenv("FOUR_O_URL", "https://api.example.com/parse")
    headers = {"Authorization": f"Bearer {token}"}
    async with ClientSession() as session:
        resp = await session.post(url, json={"text": text}, headers=headers)
        resp.raise_for_status()
        data = await resp.json()
        if isinstance(data, dict):
            return data.get("response") or str(data)
        return str(data)



async def handle_start(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        result = await session.execute(select(User))
        user_count = len(result.scalars().all())
        user = await session.get(User, message.from_user.id)
        if user:
            await bot.send_message(message.chat.id, "Bot is running")
            return
        if user_count == 0:
            session.add(
                User(
                    user_id=message.from_user.id,
                    username=message.from_user.username,
                    is_superadmin=True,
                )
            )
            await session.commit()
            await bot.send_message(message.chat.id, "You are superadmin")
        else:
            await bot.send_message(message.chat.id, "Use /register to apply")


async def handle_register(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        if await session.get(User, message.from_user.id):
            await bot.send_message(message.chat.id, "Already registered")
            return
        if await session.get(RejectedUser, message.from_user.id):
            await bot.send_message(message.chat.id, "Access denied by administrator")
            return
        if await session.get(PendingUser, message.from_user.id):
            await bot.send_message(message.chat.id, "Awaiting approval")
            return
        result = await session.execute(select(PendingUser))
        if len(result.scalars().all()) >= 10:
            await bot.send_message(
                message.chat.id, "Registration queue full, try later"
            )
            return
        session.add(
            PendingUser(
                user_id=message.from_user.id, username=message.from_user.username
            )
        )
        await session.commit()
        await bot.send_message(message.chat.id, "Registration pending approval")


async def handle_requests(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            return
        result = await session.execute(select(PendingUser))
        pending = result.scalars().all()
        if not pending:
            await bot.send_message(message.chat.id, "No pending users")

            return
        buttons = [
            [
                types.InlineKeyboardButton(
                    text="Approve", callback_data=f"approve:{p.user_id}"
                ),
                types.InlineKeyboardButton(
                    text="Reject", callback_data=f"reject:{p.user_id}"
                ),
            ]
            for p in pending
        ]
        keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
        lines = [f"{p.user_id} {p.username or ''}" for p in pending]
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=keyboard)


async def process_request(callback: types.CallbackQuery, db: Database, bot: Bot):
    uid = int(callback.data.split(":", 1)[1])
    async with db.get_session() as session:
        p = await session.get(PendingUser, uid)
        if not p:
            await callback.answer("Not found", show_alert=True)
            return

        if callback.data.startswith("approve"):
            session.add(User(user_id=uid, username=p.username, is_superadmin=False))
            await bot.send_message(uid, "You are approved")
        else:
            session.add(RejectedUser(user_id=uid, username=p.username))
            await bot.send_message(uid, "Your registration was rejected")
        await session.delete(p)
        await session.commit()
        await callback.answer("Done")


async def handle_tz(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    if len(parts) != 2 or not validate_offset(parts[1]):
        await bot.send_message(message.chat.id, "Usage: /tz +02:00")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    await set_tz_offset(db, parts[1])
    await bot.send_message(message.chat.id, f"Timezone set to {parts[1]}")



async def handle_add_event(message: types.Message, db: Database, bot: Bot):
    text = message.text.split(maxsplit=1)
    if len(text) != 2:
        await bot.send_message(message.chat.id, "Usage: /addevent <text>")
        return
    try:
        data = await parse_event_via_4o(text[1])
    except Exception as e:
        await bot.send_message(message.chat.id, f"LLM error: {e}")
        return
    event = Event(
        title=data.get("title", ""),
        description=data.get("short_description", ""),
        festival=data.get("festival") or None,
        date=data.get("date", ""),
        time=data.get("time", ""),
        location_name=data.get("location_name", ""),
        location_address=data.get("location_address"),
        city=data.get("city"),
        source_text=text[1],
    )
    async with db.get_session() as session:
        session.add(event)
        await session.commit()
    await bot.send_message(message.chat.id, f"Event '{event.title}' added")


async def handle_add_event_raw(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    if len(parts) != 2 or '|' not in parts[1]:
        await bot.send_message(message.chat.id, "Usage: /addevent_raw title|date|time|location")
        return
    title, date, time, location = (p.strip() for p in parts[1].split('|', 3))
    event = Event(
        title=title,
        description="",
        festival=None,
        date=date,
        time=time,
        location_name=location,
        source_text=parts[1],
    )
    async with db.get_session() as session:
        session.add(event)
        await session.commit()
    await bot.send_message(message.chat.id, f"Event '{title}' added")


async def handle_ask_4o(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    if len(parts) != 2:
        await bot.send_message(message.chat.id, "Usage: /ask4o <text>")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    try:
        answer = await ask_4o(parts[1])
    except Exception as e:
        await bot.send_message(message.chat.id, f"LLM error: {e}")
        return
    await bot.send_message(message.chat.id, answer)


async def telegraph_test():
    token = os.getenv("TELEGRAPH_TOKEN")
    if not token:
        logging.error("TELEGRAPH_TOKEN is missing")
        return
    tg = Telegraph()
    tg.access_token = token
    page = await asyncio.to_thread(tg.create_page, "Test Page", html="<p>test</p>")
    logging.info("Created %s", page["url"])
    await asyncio.to_thread(
        tg.edit_page, page["path"], title="Test Page", html_content="<p>updated</p>"
    )
    logging.info("Edited %s", page["url"])



def create_app() -> web.Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")


    webhook = os.getenv("WEBHOOK_URL")
    if not webhook:
        raise RuntimeError("WEBHOOK_URL is missing")

    bot = Bot(token)
    dp = Dispatcher()
    db = Database(DB_PATH)

    async def start_wrapper(message: types.Message):
        await handle_start(message, db, bot)

    async def register_wrapper(message: types.Message):
        await handle_register(message, db, bot)

    async def requests_wrapper(message: types.Message):
        await handle_requests(message, db, bot)

    async def tz_wrapper(message: types.Message):
        await handle_tz(message, db, bot)

    async def callback_wrapper(callback: types.CallbackQuery):
        await process_request(callback, db, bot)

    async def add_event_wrapper(message: types.Message):
        await handle_add_event(message, db, bot)

    async def add_event_raw_wrapper(message: types.Message):
        await handle_add_event_raw(message, db, bot)

    async def ask_4o_wrapper(message: types.Message):
        await handle_ask_4o(message, db, bot)

    dp.message.register(start_wrapper, Command("start"))
    dp.message.register(register_wrapper, Command("register"))
    dp.message.register(requests_wrapper, Command("requests"))
    dp.callback_query.register(
        callback_wrapper,
        lambda c: c.data.startswith("approve") or c.data.startswith("reject"),
    )
    dp.message.register(tz_wrapper, Command("tz"))
    dp.message.register(add_event_wrapper, Command("addevent"))
    dp.message.register(add_event_raw_wrapper, Command("addevent_raw"))
    dp.message.register(ask_4o_wrapper, Command("ask4o"))

    app = web.Application()
    SimpleRequestHandler(dp, bot).register(app, path="/webhook")
    setup_application(app, dp, bot=bot)

    async def on_startup(app: web.Application):
        await db.init()
        await bot.set_webhook(webhook.rstrip("/") + "/webhook")

    async def on_shutdown(app: web.Application):
        await bot.session.close()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


    async def on_shutdown(app: web.Application):
        await bot.session.close()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test_telegraph":
        asyncio.run(telegraph_test())
    else:
        web.run_app(create_app(), port=int(os.getenv("PORT", 8080)))

