#!/usr/bin/env bash
# Скрипт для синхронизации продакшн базы данных с fly.io на локальную машину
# Использование: ./scripts/sync_prod_db.sh [--app APP_NAME] [--output LOCAL_PATH]

set -euo pipefail

# Настройки по умолчанию
APP_NAME="${FLY_APP_NAME:-events-bot-new-wngqia}"
PROD_DB_PATH="/data/db.sqlite"
REMOTE_TMP_DB_PATH="/tmp/db_backup_sync.sqlite"
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

# Backup/restore bookkeeping (so a failed download doesn't leave us with a broken snapshot)
had_backup=0
BACKUP_PATH=""
MARKER_PATH="${LOCAL_DB_PATH}.downloaded_at"

restore_previous_snapshot() {
    # Best-effort restore if we moved an existing snapshot out of the way.
    if [ "${had_backup:-0}" -ne 1 ] || [ -z "${BACKUP_PATH:-}" ]; then
        return
    fi
    echo "⚠️  Восстанавливаю предыдущий snapshot после ошибки скачивания/проверки…"
    rm -f "$LOCAL_DB_PATH" 2>/dev/null || true
    if [ -f "$BACKUP_PATH" ]; then
        mv "$BACKUP_PATH" "$LOCAL_DB_PATH" 2>/dev/null || true
    fi
    for ext in "-wal" "-shm" "-journal"; do
        if [ -f "${BACKUP_PATH}${ext}" ]; then
            rm -f "${LOCAL_DB_PATH}${ext}" 2>/dev/null || true
            mv "${BACKUP_PATH}${ext}" "${LOCAL_DB_PATH}${ext}" 2>/dev/null || true
        fi
    done
    if [ -f "${BACKUP_PATH}.downloaded_at" ]; then
        rm -f "$MARKER_PATH" 2>/dev/null || true
        mv "${BACKUP_PATH}.downloaded_at" "$MARKER_PATH" 2>/dev/null || true
    fi
}

trap restore_previous_snapshot ERR

# Если локальная база уже существует, создаём бэкап
if [ -f "$LOCAL_DB_PATH" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/db_snapshot_$TIMESTAMP.sqlite"
    echo "📋 Создаём бэкап существующей базы: $BACKUP_PATH"
    mv "$LOCAL_DB_PATH" "$BACKUP_PATH"
    had_backup=1
    # Backup sqlite sidecars if present (WAL/shm/journal).
    for ext in "-wal" "-shm" "-journal"; do
        if [ -f "${LOCAL_DB_PATH}${ext}" ]; then
            mv "${LOCAL_DB_PATH}${ext}" "${BACKUP_PATH}${ext}" || true
        fi
    done
    # Backup the freshness marker as well, so we can restore it on failure.
    if [ -f "$MARKER_PATH" ]; then
        mv "$MARKER_PATH" "${BACKUP_PATH}.downloaded_at" || true
    fi
fi

# Скачиваем базу данных с продакшена через fly ssh sftp
echo "⬇️  Скачивание базы данных..."
echo "🛠️  Подготовка консистентного snapshot на Fly..."
fly ssh console -a "$APP_NAME" -C "python3 -c \"import sqlite3; src=sqlite3.connect('$PROD_DB_PATH'); dst=sqlite3.connect('$REMOTE_TMP_DB_PATH'); src.backup(dst); dst.close(); src.close(); print('backup_ok')\""

echo "⬇️  Скачивание snapshot..."
fly ssh sftp get "$REMOTE_TMP_DB_PATH" "$LOCAL_DB_PATH" -a "$APP_NAME"
sftp_rc=$?

echo "🧹 Удаление временного snapshot на Fly..."
fly ssh console -a "$APP_NAME" -C "rm -f '$REMOTE_TMP_DB_PATH'" >/dev/null 2>&1 || true

if [ "${sftp_rc:-1}" -ne 0 ]; then
    echo "❌ Ошибка при скачивании snapshot (fly ssh sftp get exit_code=$sftp_rc)"
    exit 1
fi

# Verify the downloaded sqlite snapshot is readable and not corrupted.
python3 - "$LOCAL_DB_PATH" <<'PY'
import sqlite3
import sys

db_path = sys.argv[1]
try:
    con = sqlite3.connect(db_path)
    row = con.execute("PRAGMA quick_check").fetchone()
    res = (row[0] if row else "") or ""
    con.close()
except Exception as exc:
    raise SystemExit(f"sqlite_quick_check_failed: {exc}")
if str(res).strip().lower() != "ok":
    raise SystemExit(f"sqlite_quick_check_failed: {res}")
PY

if [ $? -eq 0 ]; then
    # Track the actual download time in a sidecar marker so other scripts can
    # reliably detect snapshot freshness even if the sqlite file is later
    # mutated by local runs (E2E, debugging, etc).
    touch "${LOCAL_DB_PATH}.downloaded_at" 2>/dev/null || true

    # Получаем размер файла
    SIZE=$(ls -lh "$LOCAL_DB_PATH" | awk '{print $5}')
    echo ""
    echo "✅ База данных успешно скачана!"
    echo "   Размер: $SIZE"
    echo "   Путь: $LOCAL_DB_PATH"
    echo ""
    
    # Показываем базовую статистику (best-effort; sqlite3 CLI может отсутствовать).
    echo "📊 Базовая статистика:"
    if command -v sqlite3 &> /dev/null; then
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
    else
        python3 - "$LOCAL_DB_PATH" <<'PY'
import sqlite3
import sys

db_path = sys.argv[1]
con = sqlite3.connect(db_path)
cur = con.cursor()
def count(table):
    try:
        return cur.execute(f"select count(*) from {table}").fetchone()[0]
    except Exception:
        return "n/a"
print(f"{'table_name':<12} {'count':>8}")
for t in ("event", "festival", "vk_inbox"):
    print(f"{t:<12} {count(t):>8}")
con.close()
PY
    fi
    
    echo ""
    echo "💡 Для использования этой базы данных локально:"
    echo "   export DB_PATH=$LOCAL_DB_PATH"
    echo "   export DEV_MODE=1"
    echo "   python main.py"
else
    echo "❌ Ошибка при скачивании базы данных"
    exit 1
fi

# Success: disable restore trap.
trap - ERR
