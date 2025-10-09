from festival_activities import (
    FestivalActivitiesError,
    activities_to_nodes,
    format_activities_preview,
    parse_festival_activities_yaml,
    save_festival_activities,
)
FESTIVAL_EDIT_FIELD_ACTIVITIES = "activities"
    elif data.startswith("festacts:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        festival_edit_sessions[callback.from_user.id] = (fid, FESTIVAL_EDIT_FIELD_ACTIVITIES)
        preview = format_activities_preview(fest.activities_json or [])
        await callback.message.answer(
            "Пришлите YAML с активностями файлом или текстом.\n"
            "Файл должен соответствовать схеме version: 2.\n\n"
            f"Предпросмотр:\n{preview}"
        )
        await callback.answer()
    activities_nodes = activities_to_nodes(fest.activities_json or [])
    if activities_nodes:
        nodes.extend(telegraph_br())
        nodes.extend(telegraph_br())
        nodes.extend(activities_nodes)

        f"activities: {len(fest.activities_json or [])}",
        [
            types.InlineKeyboardButton(
                text="Активности (YAML)", callback_data=f"festacts:{fest.id}"
            )
        ],
    if field == FESTIVAL_EDIT_FIELD_ACTIVITIES:
        raw_payload = ""
        if message.document:
            mime = (message.document.mime_type or "").lower()
            name = (message.document.file_name or "").lower()
            if not (
                mime.startswith("text/")
                or mime in {"application/x-yaml", "application/yaml"}
                or name.endswith((".yaml", ".yml"))
            ):
                await bot.send_message(
                    message.chat.id,
                    "Документ должен быть YAML или текстовым файлом.",
                )
                return
            bio = BytesIO()
            async with span("tg-send"):
                await bot.download(message.document.file_id, destination=bio)
            raw_payload = bio.getvalue().decode("utf-8", errors="ignore")
        else:
            raw_payload = message.text or message.caption or ""
        if not (raw_payload or "").strip():
            await bot.send_message(message.chat.id, "Текст YAML не найден.")
            return
        try:
            parsed = parse_festival_activities_yaml(raw_payload)
        except FestivalActivitiesError as exc:
            await bot.send_message(message.chat.id, f"Ошибка в YAML: {exc}")
            return
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
            if not fest:
                await bot.send_message(message.chat.id, "Festival not found")
                festival_edit_sessions.pop(message.from_user.id, None)
                return
            previous_site = fest.website_url
            await save_festival_activities(session, fest, parsed)
            fest_view = Festival(**fest.model_dump())  # type: ignore[arg-type]
            fest_name = fest.name
        festival_edit_sessions[message.from_user.id] = (fid, None)
        preview_text = format_activities_preview(parsed.activities)
        response_lines = [
            f"Активности обновлены ({len(parsed.activities)}).",
            "Предпросмотр:",
            preview_text,
        ]
        if parsed.website_url and parsed.website_url != previous_site:
            response_lines.insert(1, f"Сайт фестиваля обновлён: {parsed.website_url}")
        await bot.send_message(message.chat.id, "\n".join(response_lines))
        await show_festival_edit_menu(message.from_user.id, fest_view, bot)
        await sync_festival_page(db, fest_name)
        await sync_festival_vk_post(db, fest_name, bot)
        await rebuild_fest_nav_if_changed(db)
        return
        or c.data.startswith("festacts:")
