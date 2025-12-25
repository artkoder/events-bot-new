#!/usr/bin/env bash
# Скрипт для получения дампа БД через команду бота /dumpdb
# Требует: TELEGRAM_BOT_TOKEN и ADMIN_CHAT_ID в переменных окружения
# Использование: ./scripts/dump_prod_db.sh

set -euo pipefail

# Проверяем переменные окружения
if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
    echo "❌ Ошибка: не задана переменная TELEGRAM_BOT_TOKEN"
    echo "   Установите: export TELEGRAM_BOT_TOKEN=your_token"
    exit 1
fi

if [ -z "${ADMIN_CHAT_ID:-}" ]; then
    echo "❌ Ошибка: не задана переменная ADMIN_CHAT_ID"
    echo "   Установите: export ADMIN_CHAT_ID=your_chat_id"
    exit 1
fi

BACKUP_DIR="./backups"
mkdir -p "$BACKUP_DIR"

echo "📤 Отправка команды /dumpdb боту..."

# Отправляем команду /dumpdb
SEND_URL="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage"
curl -s -X POST "$SEND_URL" \
    -d chat_id="$ADMIN_CHAT_ID" \
    -d text="/dumpdb" > /dev/null

echo "✅ Команда отправлена!"
echo ""
echo "⏳ Ожидаем получения дампа (обычно занимает 5-30 секунд)..."
echo ""
echo "Бот отправит файл дампа в чат. Вы можете:"
echo "  1. Скачать файл вручную из Telegram"
echo "  2. Использовать API для автоматического скачивания:"
echo ""
echo "     # Получить последнее сообщение с документом"
echo "     curl -s \"https://api.telegram.org/bot\${TELEGRAM_BOT_TOKEN}/getUpdates\" \\"
echo "         | jq '.result[-1].message.document.file_id'"
echo ""
echo "     # Скачать файл по file_id"
echo "     # (см. документацию Telegram Bot API)"
echo ""
echo "💡 Для восстановления дампа:"
echo "   1. Поместите файл dump.sql в проект"
echo "   2. Отправьте команду /restore боту с прикрепленным файлом"
echo "   3. Или импортируйте локально: sqlite3 db.sqlite < dump.sql"
