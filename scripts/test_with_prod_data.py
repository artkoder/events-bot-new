#!/usr/bin/env python3
"""
Скрипт для тестирования отдельных операций на продакшн-данных
Использование: python scripts/test_with_prod_data.py [command] [options]

Команды:
  stats              - Показать статистику по базе данных
  test-llm EVENT_ID  - Протестировать LLM операцию на конкретном событии
  test-vk-review     - Протестировать VK review процесс
  export-sample      - Экспортировать примеры данных для тестов
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))


async def show_stats(db_path: str):
    """Показывает статистику по базе данных"""
    from db import Database
    
    db = Database(db_path)
    await db.init()
    
    async with db.get_session() as session:
        from sqlalchemy import select, func
        from models import Event, Festival, VKInbox
        
        # Количество событий
        events_count = await session.scalar(select(func.count()).select_from(Event))
        
        # Количество фестивалей
        festivals_count = await session.scalar(select(func.count()).select_from(Festival))
        
        # Статистика VK inbox
        try:
            vk_inbox_count = await session.scalar(select(func.count()).select_from(VKInbox))
        except Exception:
            vk_inbox_count = 0
        
        # Последние события
        from sqlalchemy import desc
        recent_events = await session.execute(
            select(Event.id, Event.title, Event.date)
            .order_by(desc(Event.added_at))
            .limit(5)
        )
        
        print("=" * 60)
        print("📊 СТАТИСТИКА БАЗЫ ДАННЫХ")
        print("=" * 60)
        print(f"📅 События: {events_count}")
        print(f"🎭 Фестивали: {festivals_count}")
        print(f"📥 VK Inbox: {vk_inbox_count}")
        print()
        print("Последние 5 событий:")
        print("-" * 60)
        
        for row in recent_events:
            event_id, title, date_val = row
            print(f"  ID {event_id}: {title[:50]}... ({date_val})")
        
        print("=" * 60)


async def test_llm_on_event(db_path: str, event_id: int):
    """Тестирует LLM операцию на конкретном событии"""
    from db import Database
    from models import Event
    from sqlalchemy import select
    
    db = Database(db_path)
    await db.init()
    
    async with db.get_session() as session:
        result = await session.execute(
            select(Event).where(Event.id == event_id)
        )
        event = result.scalar_one_or_none()
        
        if not event:
            print(f"❌ Событие с ID {event_id} не найдено")
            return
        
        print("=" * 60)
        print(f"📋 СОБЫТИЕ #{event_id}")
        print("=" * 60)
        print(f"Название: {event.title}")
        print(f"Описание: {event.description[:200] if event.description else 'Нет'}...")
        print(f"Дата начала: {event.date}")
        print(f"Город: {event.city}")
        print(f"Место: {event.location_name}")
        print("=" * 60)
        print()
        
        # Здесь можно добавить вызов LLM для тестирования
        print("💡 Для тестирования LLM добавьте код вызова нужной функции")
        print("   Например: parse_event_via_4o, compose_story_pitch_via_4o и т.д.")


async def test_vk_review(db_path: str):
    """Тестирует процесс VK review"""
    from db import Database
    
    db = Database(db_path)
    await db.init()
    
    async with db.get_session() as session:
        from sqlalchemy import select, func
        try:
            from models import VKInbox
            
            # Статистика по статусам
            result = await session.execute(
                select(VKInbox.status, func.count())
                .group_by(VKInbox.status)
            )
            
            print("=" * 60)
            print("📥 VK INBOX СТАТИСТИКА")
            print("=" * 60)
            
            for status, count in result:
                print(f"  {status}: {count}")
            
            # Показываем несколько примеров pending
            pending = await session.execute(
                select(VKInbox)
                .where(VKInbox.status == 'pending')
                .limit(3)
            )
            
            print()
            print("Примеры pending записей:")
            print("-" * 60)
            
            for item in pending.scalars():
                print(f"  Post ID: {item.post_id}")
                print(f"  Group ID: {item.group_id}")
                print(f"  Text preview: {(item.text or '')[:100]}...")
                print()
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            print("   Возможно, таблица VKInbox не существует в этой базе")


async def export_sample_data(db_path: str, output_dir: str = "./test_data"):
    """Экспортирует примеры данных для unit-тестов"""
    import json
    from db import Database
    from models import Event, Festival
    from sqlalchemy import select
    
    os.makedirs(output_dir, exist_ok=True)
    
    db = Database(db_path)
    await db.init()
    
    async with db.get_session() as session:
        # Экспортируем несколько событий
        events_result = await session.execute(
            select(Event).limit(10)
        )
        events = events_result.scalars().all()
        
        events_data = []
        for event in events:
            events_data.append({
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "start_date": str(event.date) if event.date else None,
                "end_date": str(event.end_date) if event.end_date else None,
                "city": event.city,
                "location_name": event.location_name,
            })
        
        # Сохраняем в JSON
        output_file = os.path.join(output_dir, "sample_events.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(events_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Экспортировано {len(events_data)} событий в {output_file}")
        
        # Аналогично для фестивалей
        festivals_result = await session.execute(
            select(Festival).limit(10)
        )
        festivals = festivals_result.scalars().all()
        
        festivals_data = []
        for fest in festivals:
            festivals_data.append({
                "id": fest.id,
                "name": fest.name,
                "city": fest.city,
                "start_date": str(fest.start_date) if fest.start_date else None,
                "end_date": str(fest.end_date) if fest.end_date else None,
            })
        
        output_file = os.path.join(output_dir, "sample_festivals.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(festivals_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Экспортировано {len(festivals_data)} фестивалей в {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Утилита для тестирования операций на продакшн-данных"
    )
    parser.add_argument(
        "command",
        choices=["stats", "test-llm", "test-vk-review", "export-sample"],
        help="Команда для выполнения"
    )
    parser.add_argument(
        "--db",
        default="./db_prod_snapshot.sqlite",
        help="Путь к базе данных (по умолчанию: ./db_prod_snapshot.sqlite)"
    )
    parser.add_argument(
        "--event-id",
        type=int,
        help="ID события для команды test-llm"
    )
    parser.add_argument(
        "--output",
        default="./test_data",
        help="Директория для экспорта (команда export-sample)"
    )
    
    args = parser.parse_args()
    
    # Проверяем существует ли база данных
    if not os.path.exists(args.db):
        print(f"❌ База данных не найдена: {args.db}")
        print("   Сначала скачайте БД с продакшена:")
        print("   ./scripts/sync_prod_db.sh --output", args.db)
        sys.exit(1)
    
    # Выполняем команду
    if args.command == "stats":
        asyncio.run(show_stats(args.db))
    
    elif args.command == "test-llm":
        if not args.event_id:
            print("❌ Для команды test-llm требуется --event-id")
            sys.exit(1)
        asyncio.run(test_llm_on_event(args.db, args.event_id))
    
    elif args.command == "test-vk-review":
        asyncio.run(test_vk_review(args.db))
    
    elif args.command == "export-sample":
        asyncio.run(export_sample_data(args.db, args.output))


if __name__ == "__main__":
    main()
