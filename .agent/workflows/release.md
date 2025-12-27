---
description: Процедура релиза на продакшн
---

# Релиз на прод

// turbo-all

1. Проверить что все тесты проходят:
   ```bash
   pytest tests/ -v
   ```

2. Определить тип версии (в VERSION файле):
   - MAJOR (X.0.0) — breaking changes
   - MINOR (0.X.0) — новый функционал
   - PATCH (0.0.X) — багфиксы

3. Обновить VERSION файл с новой версией

4. Обновить CHANGELOG.md:
   - Переименовать [Unreleased] в [X.Y.Z] – YYYY-MM-DD
   - Создать новую пустую секцию [Unreleased]

5. Закоммитить изменения:
   ```bash
   git add -A
   git commit -m "chore: release vX.Y.Z"
   ```

6. Создать git tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   ```

7. Запушить с тегами:
   ```bash
   git push origin main --tags
   ```

8. Задеплоить на Fly.io:
   ```bash
   fly deploy
   ```
