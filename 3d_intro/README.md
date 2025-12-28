# 3D Intro - Debugging Tools

Инструменты для итеративной отладки 3D intro сцены в Blender.

## Обзор

Этот модуль предоставляет инфраструктуру для:
- Управления шрифтами с fallback системой
- Покадровой генерации и отладки сцен
- Превью паттернов в плоском формате
- Тестирования совместимости с Kaggle

## Файлы

### Основные модули

- **`font_manager.py`** - Централизованное управление шрифтами
- **`debug_scene.py`** - CLI инструмент для отладки сцен
- **`pattern_preview_3d.py`** - Генератор плоских превью
- **`test_blender_setup.ipynb`** - Kaggle тестовый ноутбук

### Вспомогательные скрипты

- **`render_frame.sh`** - Быстрый рендеринг одного кадра
- **`generate_intro.py`** - Генерация полной сцены
- **`visualize_scene.py`** - Визуализация сцены (matplotlib)

## Быстрый старт

### 1. Генерация сцены

```bash
# Создать базовую сцену с анимацией
python /workspaces/events-bot-new/3d_intro/generate_intro.py
```

Создаёт файл: `/tmp/bento_intro_v1_animated.blend`

### 2. Рендер конкретного кадра

```bash
# Способ 1: С помощью helper script
./render_frame.sh 1.5 /tmp/frame_1_5s.png

# Способ 2: Напрямую через debug_scene.py
blender --background --python debug_scene.py -- \
    render --time 1.5 --output /tmp/frame_1_5s.png
```

### 3. Изменение параметров сцены

```bash
# Изменить позицию камеры
blender --background --python debug_scene.py -- \
    camera --position "10,5,3"

# Изменить rotation камеры (в градусах)
blender --background --python debug_scene.py -- \
    camera --rotation "45,0,90"

# Изменить объект
blender --background --python debug_scene.py -- \
    object --name "Cube_Main" --position "0,0,2"
```

### 4. Snapshot/Restore

```bash
# Сохранить текущее состояние
blender --background --python debug_scene.py -- \
    snapshot save --name "camera_test_v1"

# Список всех сохранённых snapshots
blender --background --python debug_scene.py -- \
    snapshot list

# Восстановить snapshot
blender --background --python debug_scene.py -- \
    snapshot restore --name "camera_test_v1"
```

### 5. Получить информацию о сцене

```bash
blender --background --python debug_scene.py -- info
```

## Рабочий процесс итеративной отладки

**Типичный цикл работы:**

1. **Пользователь**: "Покажи кадр на секунде 1.5"
2. **AI**: Запускает `render_frame.sh 1.5`, показывает изображение
3. **Пользователь**: "Камеру немного вправо и вверх"
4. **AI**: Обновляет `camera --position "X+1,Y+0.5,Z"`, перерендерит
5. **Пользователь**: "Окей, сохрани"
6. **AI**: Сохраняет `snapshot save --name v1_camera_approved`
7. **Повторяем для других ключевых моментов**

## Pattern Preview (плоский формат)

### Генерация превью

```python
from 3d_intro.pattern_preview_3d import generate_cube_text_preview

texts_data = {
    "main_cube": {"text": "АФИША", "font": "druk_cyr_bold"},
    "info_cubes": [
        {"text": "НА ВЫХОДНЫЕ", "font": "benzin_bold"},
        {"text": "27-28", "font": "bebas_neue_regular"},
        {"text": "ДЕКАБРЯ", "font": "bebas_neue_regular"},
    ],
    "cities": "Калининград, Светлогорск"
}

preview_bytes = generate_cube_text_preview(texts_data)

# Сохранить в файл
with open("preview.png", "wb") as f:
    f.write(preview_bytes)
```

### Генерация развёртки куба

```python
from 3d_intro.pattern_preview_3d import generate_cube_unwrap

textures = {
    "front": "АФИША",
    "top": "/path/to/image.jpg",
    "left": "TEXT",
    # ...
}

unwrap_bytes = generate_cube_unwrap("main_cube", textures)
```

## Управление шрифтами

### Проверка доступных шрифтов

```python
from 3d_intro.font_manager import print_font_info

print_font_info()
```

Вывод:
```
Font Manager Status:
Search paths: ['/workspaces/events-bot-new/3d_intro/assets/fonts', ...]

Available fonts:
  ⚠ druk_cyr_bold: MISSING (using fallback: bebas_neue_bold)
  ⚠ benzin_bold: MISSING (using fallback: bebas_neue_bold)
  ✓ bebas_neue_regular: BebasNeue-Regular.ttf
  ✓ bebas_neue_pro_middle: Bebas-Neue-Pro-SemiExpanded-Bold-BF66cf3d78c1f4e.ttf
  ✓ bebas_neue_bold: BebasNeue-Bold.ttf
```

### Получение пути к шрифту

```python
from 3d_intro.font_manager import get_font_str, DRUK_CYR_BOLD

font_path = get_font_str(DRUK_CYR_BOLD, allow_fallback=True)
# Вернет путь к fallback шрифту если основной не найден
```

## Kaggle Setup

### Тестирование на Kaggle

1. Загрузить `/kaggle/TestBlenderSetup/test_blender_setup.ipynb` на Kaggle
2. Запустить ноутбук
3. Проверить вывод:
   - ✓ Blender установлен
   - ✓ Базовый рендеринг работает
   - ✓ Шрифты загружаются
   - ✓ Текстуры применяются
4. Скачать `kaggle_requirements.txt` для синхронизации версий

## Структура директорий

```
3d_intro/
├── assets/
│   ├── fonts/              # Кастомные шрифты (добавить Druk Cyr, Benzin)
│   └── posters/            # Примеры афиш
├── font_manager.py         # Управление шрифтами
├── debug_scene.py          # CLI инструмент отладки
├── pattern_preview_3d.py   # Плоские превью
├── generate_intro.py       # Генерация сцены
├── bento_scene.py          # Генерация Bento-сетки
├── animation_system.py     # Система анимации
├── render_frame.sh         # Helper скрипт
├── visualize_scene.py      # Matplotlib визуализация
└── README.md               # Этот файл
```

## Snapshots

Snapshots сохраняются в `/tmp/3d_intro_snapshots/`:
- `{name}.blend` - Файл сцены
- `{name}.json` - Метаданные (timestamp, оригинальный путь)

## Параметры по умолчанию

- **FPS**: 30
- **Разрешение**: 1080x1920 (vertical для mobile)
- **Продолжительность**: 2.5 секунды (75 кадров)
- **Render Engine**: EEVEE (для скорости), может быть переключён на Cycles

## Ключевые моменты анимации

- **0.0-0.4s**: Фаза 1 - Крупный план главного куба
- **0.4-0.6s**: Фаза 2 - Отъезд камеры
- **0.6-1.2s**: Фаза 3 - Вращение конструкции
- **1.2-1.8s**: Фаза 4 - Наезд на выбранный куб
- **1.8-2.5s**: Фаза 5 - Трансформация в сцену

## Troubleshooting

### Blender не найден

```bash
# Установить Blender
sudo apt-get update
sudo apt-get install blender
```

### bpy module not available

Скрипт должен запускаться через Blender Python:
```bash
blender --background --python script.py -- args
```

Не запускайте напрямую через `python script.py`

### Шрифты не найдены

Добавьте файлы шрифтов в:
- `/workspaces/events-bot-new/3d_intro/assets/fonts/`

Или используйте fallback (Bebas Neue Bold) - автоматически.

## Следующие шаги

1. **Добавить недостающие шрифты**: Druk Cyr Bold, Benzin Bold
2. **Протестировать на Kaggle**: Запустить test_blender_setup.ipynb
3. **Итеративная отладка сцены**: Использовать debug_scene.py
4. **Создать полный pipeline**: Интегрировать с video_announce
