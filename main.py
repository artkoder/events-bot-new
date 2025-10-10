from festival_activities import (
    TEMPLATE_YAML as FESTIVAL_ACTIVITIES_TEMPLATE,
    MAX_YAML_SIZE,
    activities_to_telegraph_nodes,
    load_groups_from_json as load_activity_groups,
    FestivalActivitiesError,
    build_activity_card_lines,
    parse_festival_activities_yaml,
)
FESTIVAL_EDIT_FIELD_ACTIVITIES = "activities"
    elif data.startswith("festacttmpl:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            if not await session.get(User, callback.from_user.id):
                await callback.answer("Not authorized", show_alert=True)
                return
            fest = await session.get(Festival, fid)
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        safe_name = re.sub(r"[^\w.-]+", "_", fest.name or "festival")
        data = FESTIVAL_ACTIVITIES_TEMPLATE.encode("utf-8")
        document = types.BufferedInputFile(data, filename=f"{safe_name}_activities.yaml")
        await callback.message.answer_document(
            document, caption="Шаблон YAML для активностей фестиваля"
        )
        await callback.answer()
    elif data.startswith("festactupload:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            if not await session.get(User, callback.from_user.id):
                await callback.answer("Not authorized", show_alert=True)
                return
            fest = await session.get(Festival, fid)
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        festival_edit_sessions[callback.from_user.id] = (
            fid,
            FESTIVAL_EDIT_FIELD_ACTIVITIES,
        )
        await callback.message.answer(
            "Пришлите YAML с активностями текстом или документом (до 256 КБ)."
        )
        await callback.answer()
    activity_nodes = activities_to_telegraph_nodes(
        load_activity_groups(fest.activities_json)
    )
    if activity_nodes:
        nodes.extend(activity_nodes)

        [
            types.InlineKeyboardButton(
                text="Скачать шаблон YAML",
                callback_data=f"festacttmpl:{fest.id}",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Загрузить активности",
                callback_data=f"festactupload:{fest.id}",
            )
        ],
    if field == FESTIVAL_EDIT_FIELD_ACTIVITIES:
        content = ""
        if message.document:
            size = message.document.file_size or 0
            if size > MAX_YAML_SIZE:
                await bot.send_message(
                    message.chat.id,
                    "Файл слишком большой. Максимальный размер — 256 КБ.",
                )
                return
            buffer = BytesIO()
            async with span("tg-send"):
                await bot.download(message.document.file_id, destination=buffer)
            try:
                content = buffer.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                await bot.send_message(
                    message.chat.id,
                    "Файл должен быть в кодировке UTF-8.",
                )
                return
        else:
            content = (message.text or message.caption or "").strip()
        if not content:
            await bot.send_message(
                message.chat.id,
                "Не удалось прочитать YAML. Пришлите текст или документ.",
            )
            return
        try:
            parsed = parse_festival_activities_yaml(content)
        except FestivalActivitiesError as exc:
            await bot.send_message(
                message.chat.id, f"Не удалось разобрать YAML: {exc}"
            )
            return
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
            if not fest:
                await bot.send_message(message.chat.id, "Festival not found")
                festival_edit_sessions.pop(message.from_user.id, None)
                return
            fest.activities_json = parsed.to_json_payload()
            if parsed.festival_site:
                fest.website_url = parsed.festival_site
            await session.commit()
            await session.refresh(fest)
            fest_view = Festival(**fest.model_dump())  # type: ignore[arg-type]
            fest_name = fest.name
        total = sum(len(group.items) for group in parsed.groups)
        preview_lines = [f"Сохранено активностей: {total}."]
        if parsed.groups:
            for group in parsed.groups:
                preview_lines.append(f"• {group.title}: {len(group.items)}")
                for activity in group.items[:3]:
                    preview_lines.append(f"  ◦ {activity.title}")
                    details = build_activity_card_lines(activity)
                    if details:
                        preview_lines.append(f"    {details[0]}")
        else:
            preview_lines.append("Секции пусты.")
        if parsed.warnings:
            preview_lines.append("Предупреждения:")
            preview_lines.extend(f"⚠️ {warn}" for warn in parsed.warnings)
        await bot.send_message(message.chat.id, "\n".join(preview_lines))
        festival_edit_sessions[message.from_user.id] = (fid, None)
        await show_festival_edit_menu(message.from_user.id, fest_view, bot)
        await sync_festival_page(db, fest_name)
        await sync_festival_vk_post(db, fest_name, bot)
        await rebuild_fest_nav_if_changed(db)
        return
