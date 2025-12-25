#!/usr/bin/env bash
# Скрипт для синхронизации продакшн базы данных с fly.io на локальную машину
# Использование: ./scripts/sync_prod_db.sh [--app APP_NAME] [--output LOCAL_PATH]

set -euo pipefail

# Настройки по умолчанию
APP_NAME="${FLY_APP_NAME:-events-bot-new-wngqia}"
PROD_DB_PATH="/data/db.sqlite"
LOCAL_DB_PATH="./db_prod_snapshot.sqlite"
BACKUP_DIR="./backups"

# Parsing аргументов
while [[ $# -gt 0 ]]; do
  case $1 in
    --app)
      APP_NAME="$2"
      shift 2
      ;;
    --output)
      LOCAL_DB_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Использование: $0 [опции]"
      echo ""
      echo "Опции:"
      echo "  --app APP_NAME       Имя приложения на Fly.io (по умолчанию: events-bot-new-wngqia)"
      echo "  --output LOCAL_PATH  Путь для сохранения базы данных локально (по умолчанию: ./db_prod_snapshot.sqlite)"
      echo "  --help               Показать эту справку"
      exit 0
      ;;
    *)
      echo "Неизвестная опция: $1"
      exit 1
      ;;
  esac
done

echo "📦 Скачивание продакшн БД с Fly.io..."
echo "   App: $APP_NAME"
echo "   Продакшн путь: $PROD_DB_PATH"
echo "   Локальный путь: $LOCAL_DB_PATH"
echo ""

# Проверяем установлен ли flyctl
if ! command -v fly &> /dev/null; then
    echo "❌ Ошибка: flyctl не установлен"
    echo "   Установите: curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Создаём директорию для бэкапов если её нет
mkdir -p "$BACKUP_DIR"

# Если локальная база уже существует, создаём бэкап
if [ -f "$LOCAL_DB_PATH" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/db_snapshot_$TIMESTAMP.sqlite"
    echo "📋 Создаём бэкап существующей базы: $BACKUP_PATH"
    cp "$LOCAL_DB_PATH" "$BACKUP_PATH"
fi

# Скачиваем базу данных с продакшена через fly ssh sftp
echo "⬇️  Скачивание базы данных..."
fly ssh sftp get "$PROD_DB_PATH" "$LOCAL_DB_PATH" -a "$APP_NAME"

if [ $? -eq 0 ]; then
    # Получаем размер файла
    SIZE=$(ls -lh "$LOCAL_DB_PATH" | awk '{print $5}')
    echo ""
    echo "✅ База данных успешно скачана!"
    echo "   Размер: $SIZE"
    echo "   Путь: $LOCAL_DB_PATH"
    echo ""
    
    # Показываем базовую статистику
    echo "📊 Базовая статистика:"
    sqlite3 "$LOCAL_DB_PATH" <<EOF
.mode column
SELECT 
    'events' as table_name, 
    COUNT(*) as count 
FROM event
UNION ALL
SELECT 
    'festivals' as table_name, 
    COUNT(*) as count 
FROM festival
UNION ALL
SELECT 
    'vk_inbox' as table_name, 
    COUNT(*) as count 
FROM vk_inbox;
EOF
    
    echo ""
    echo "💡 Для использования этой базы данных локально:"
    echo "   export DB_PATH=$LOCAL_DB_PATH"
    echo "   export DEV_MODE=1"
    echo "   python main.py"
else
    echo "❌ Ошибка при скачивании базы данных"
    exit 1
fi
