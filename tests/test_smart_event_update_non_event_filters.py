import smart_event_update as su


def test_non_event_notice_tax_deduction_is_detected() -> None:
    title = "Налоговый вычет на занятия спортом"
    text = (
        "С 2022 года в России действует программа, позволяющая гражданам вернуть часть средств.\n"
        "Итоговый перечень на 2026 год утвердят до 1 марта.\n"
        "Калининградским спортивным клубам нужно успеть подать заявку до 16 февраля."
    )
    assert su._looks_like_non_event_notice(title, text) is True


def test_open_call_is_detected_as_non_event() -> None:
    title = "Open Call: Государство — это Я"
    text = (
        "Открыт конкурсный отбор участников художественного проекта.\n"
        "Подать заявку можно по ссылке.\n"
        "Подробности: https://example.com\n"
    )
    assert su._looks_like_open_call_not_event(title, text) is True


def test_course_promo_is_detected() -> None:
    title = "Голос — следствие внутреннего состояния говорящего"
    text = (
        "На этом основана наша авторская программа.\n"
        "Старт курса 2 марта.\n"
        "На каждом занятии мы тренируемся и закрепляем навыки в действии.\n"
        "Записывайтесь на пробное занятие."
    )
    assert su._looks_like_course_promo(title, text) is True


def test_service_promo_is_detected_as_non_event() -> None:
    title = "Выпускные 2026 в АгроПарке «Некрасово Поле»"
    text = (
        "🎓 Выпускные 2026 в АгроПарке «Некрасово Поле»\n\n"
        "АгроПарк — идеальное место для выпускного: свежий воздух, живая природа, животные и формат настоящего праздника.\n\n"
        "Мы подготовили пакетные программы для детских садов и школьников 1–9 классов:\n"
        "— экскурсия и знакомство с животными\n"
        "— мастер-классы и квесты\n"
        "— анимация и праздничная программа\n"
        "— обед или банкет\n"
        "Стоимость — от 2 000 ₽ / чел.\n"
        "Бронирование выпускных 2026 уже открыто.\n"
        "Рекомендуем бронировать даты заранее.\n\n"
        "📞 Телефон: +7 (911) 474-30-04\n"
        "📩 Telegram / MAX: +7 (911) 474-30-04\n"
        "📍 АгроПарк «Некрасово Поле»\n"
        "Калининградская область, пос. Некрасово.\n"
    )
    assert su._looks_like_service_promo_not_event(title, text) is True


def test_service_promo_is_not_flagged_when_has_concrete_datetime() -> None:
    title = "Выпускной вечер"
    text = (
        "1 июня 2026 в 18:00 состоится выпускной вечер.\n"
        "Приглашаем выпускников и родителей.\n"
        "Телефон: +7 (911) 000-00-00\n"
    )
    assert su._looks_like_service_promo_not_event(title, text) is False


def test_utility_outage_is_detected() -> None:
    title = "Временное отключение электроэнергии в Янтарном"
    text = (
        "19 февраля с 09:00 до 13:00 будет отключено электроснабжение по адресам ...\n"
        "Приносим извинения за доставленные неудобства."
    )
    assert su._looks_like_utility_outage_or_road_closure(title, text) == "utility_outage"


def test_road_closure_is_detected() -> None:
    title = "Ограничение движения транспорта"
    text = "В связи с ремонтными работами будет перекрыто движение на участке дороги с 10:00 до 18:00."
    assert su._looks_like_utility_outage_or_road_closure(title, text) == "road_closure"

def test_book_review_is_detected() -> None:
    title = "Этика идентичности"
    text = (
        "Сегодня — «Этика идентичности» Кваме Энтони Аппиа и флэт уайт.\n"
        "Книга о расе, этничности, нации, религии.\n"
        "#книги #кофе"
    )
    assert su._looks_like_book_review_not_event(title, text) is True


def test_too_soon_notice_is_detected() -> None:
    title = "Пресс-конференция"
    text = "Уже через 5 минут стартует пресс-конференция. Подключайтесь!"
    assert su._looks_like_too_soon_notice(title, text) is True


def test_online_event_is_detected() -> None:
    title = "Беседа"
    text = "Онлайн | Яндекс.Телемост. Подключайтесь по ссылке."
    assert su._looks_like_online_event(title, text) is True


def test_photo_day_is_detected_as_non_event() -> None:
    title = "Выставка в музее «Форт №5»"
    text = "📷 Фото дня: красавица Боня встречает гостей музея «Форт №5» 20 февраля 2026, отдельно стоящая экспозиция КОИХМ"
    assert su._looks_like_photo_day_not_event(title, text) is True


def test_photo_day_is_not_flagged_when_strong_event_signals_present() -> None:
    title = "Лекция в музее"
    text = "📷 Фото дня: и всё же приглашаем 25 февраля в 18:30 на лекцию об истории форта."
    assert su._looks_like_photo_day_not_event(title, text) is False


def test_work_schedule_is_detected_as_non_event() -> None:
    title = "График работы Калининградской областной научной библиотеки в феврале"
    text = (
        "Публикуем график работы в выходные и праздничные дни.\n"
        "23 февраля — праздничный день, библиотека не работает.\n"
        "24 февраля — санитарный день.\n"
        "С 25 февраля работаем по обычному графику."
    )
    assert su._looks_like_work_schedule_notice(title, text) is True


def test_work_schedule_holiday_days_is_detected_as_non_event() -> None:
    title = "Музей Курортной Моды: праздничные дни"
    text = (
        "21.02 музей работает с 11:00 до 19:00.\n"
        "07.03 музей работает с 11:00 до 20:00.\n"
        "Расписание посещения в праздничные дни."
    )
    assert su._looks_like_work_schedule_notice(title, text) is True


def test_work_schedule_is_detected_even_with_generic_event_nouns() -> None:
    title = "Музей: расширенный график работы"
    text = (
        "В праздничные дни музей работает по расширенному графику.\n"
        "Выставка и экспозиция доступны с 11:00 до 19:00."
    )
    assert su._looks_like_work_schedule_notice(title, text) is True


def test_work_schedule_is_not_flagged_when_clear_invite_action_exists() -> None:
    title = "Праздничные дни в музее"
    text = (
        "8 марта в 19:00 состоится лекция о курортной моде.\n"
        "Музей работает по обычному графику."
    )
    assert su._looks_like_work_schedule_notice(title, text) is False


def test_congrats_notice_is_detected_as_non_event() -> None:
    title = "Поздравляем мужчин!"
    text = (
        "Поздравляем с Днём защитника Отечества!\n"
        "23 февраля действует акция для мужчин.\n"
        "Подробности в кассе."
    )
    assert su._looks_like_congrats_notice_not_event(title, text) is True


def test_completed_event_report_is_detected_as_non_event() -> None:
    title = "Профориентационная игра с «Ораторами России»"
    text = (
        "💥 Один на один с учениками МАОУ СОШ № 33\n\n"
        "Мы отправились к ребятам, чтобы поговорить о самом важном – о выборе будущего. "
        "И сделали это в формате профориентационного квиза «Востребованные профессии».\n\n"
        "Вместе с учениками 9-11 классов мы:\n"
        "🔍 Исследовали современные и перспективные профессиональные направления.\n"
        "🧠 Решали практические задачи, где пригодилась и логика, и смекалка.\n"
        "🤝 Работали в командах – ведь умение договариваться и быстро принимать решения пригодится в любой сфере.\n\n"
        "Было здорово видеть горящие глаза ребят, их вовлечённость и неподдельный интерес к теме.\n\n"
        "Огромное спасибо администрации и педагогам 33 школы за тёплый приём и сотрудничество. "
        "И, конечно, скоро увидимся вновь, ведь это не последняя наша встреча!"
    )
    candidate = su.EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-2051396_23431",
        source_text=text,
        raw_excerpt="Профориентационный квиз для учеников 9-11 классов школы № 33.",
        title=title,
        date="2026-03-14",
        city="Калининград",
    )
    assert su._looks_like_completed_event_report_not_event(title, text, candidate=candidate) is True


def test_upcoming_school_game_is_not_flagged_as_completed_event_report() -> None:
    title = "Профориентационная игра с «Ораторами России»"
    text = (
        "14 марта в 14:00 в МАОУ СОШ № 33 состоится профориентационная игра.\n"
        "Приглашаем старшеклассников принять участие в квизе об актуальных профессиях.\n"
        "Нужна регистрация у классного руководителя."
    )
    candidate = su.EventCandidate(
        source_type="telegram",
        source_url="https://t.me/kenigevents/2",
        source_text=text,
        raw_excerpt="Игра для старшеклассников о востребованных профессиях.",
        title=title,
        date="2026-03-14",
        time="14:00",
        city="Калининград",
    )
    assert su._looks_like_completed_event_report_not_event(title, text, candidate=candidate) is False


def test_completed_event_report_with_next_show_is_not_flagged() -> None:
    title = "Мысли мудрых людей на каждый день"
    text = (
        "«Мысли мудрых людей на каждый день»👏👏👏 Премьера. День 2.\n"
        "Спасибо всей команде спектакля, театру и нашим любимым зрителям❤️\n"
        "Следующий показ будет 13 января."
    )
    candidate = su.EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-132625599_15632",
        source_text=text,
        raw_excerpt="Премьера прошла, следующий показ 13 января.",
        title=title,
        date="2026-01-13",
        city="Калининград",
    )
    assert su._looks_like_completed_event_report_not_event(title, text, candidate=candidate) is False


def test_completed_event_report_with_next_workshop_is_not_flagged() -> None:
    title = "Мастер-класс «лошадка»"
    text = (
        "🐎🎠Иго-го! Мастер-класс \"лошадка\" состоялся👍\n"
        "В следующий раз встречаемся 6 января на мастер-классе \"ангел\".\n"
        "✍️Запись - Информационно-туристический центр Светлогорска."
    )
    candidate = su.EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-195754292_10555",
        source_text=text,
        raw_excerpt="Мастер-класс состоялся, следующий пройдет 6 января.",
        title=title,
        date="2026-01-06",
        city="Светлогорск",
    )
    assert su._looks_like_completed_event_report_not_event(title, text, candidate=candidate) is False


def test_completed_event_report_with_repeat_show_is_not_flagged() -> None:
    title = "Снегурочка"
    text = (
        "В этот субботний вечер в Музыкальном театре вновь отгремела \"Снегурочка\".\n"
        "Спасибо!\n"
        "\"Снегурочка\" прощается с вами, но ненадолго.\n"
        "27 и 28 февраля вас вновь ждет встреча с героями пьесы."
    )
    candidate = su.EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-131136967_20613",
        source_text=text,
        raw_excerpt="Спектакль прошел, следующий показ 27 и 28 февраля.",
        title=title,
        date="2026-02-27",
        city="Калининград",
    )
    assert su._looks_like_completed_event_report_not_event(title, text, candidate=candidate) is False


def test_real_lecture_is_not_flagged_as_non_event_notice() -> None:
    title = "Лекция об Алексее Леонове"
    text = "26 февраля состоится лекция о жизни и пути Алексея Леонова в Доме китобоя."
    assert su._looks_like_non_event_notice(title, text) is False


def test_legacy_description_fact_has_no_service_prefix() -> None:
    fact = su._legacy_description_to_fact("### Подробности\n\nТекст события.\n")
    assert fact is not None
    assert "Текст до Smart Update:" not in fact


def test_drop_legacy_leak_from_description_removes_paragraph() -> None:
    text = (
        "Первый абзац.\n\n"
        "«Текст до Smart Update: служебная вставка»\n\n"
        "Второй абзац."
    )
    cleaned = su._drop_legacy_leak_from_description(text)
    assert cleaned is not None
    assert "Текст до Smart Update" not in cleaned
    assert "Первый абзац." in cleaned
    assert "Второй абзац." in cleaned
