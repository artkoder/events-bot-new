async def handle_privet(message: types.Message, db: Database, bot: Bot) -> None:
    await bot.send_message(message.chat.id, "Привет мир!")


    async def privet_wrapper(message: types.Message):
        await handle_privet(message, db, bot)

    dp.message.register(privet_wrapper, Command("privet"))
