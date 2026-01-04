---
description: Процедура релиза на продакшн
---

# Релиз на прод

// turbo-all

1. Убедиться что мы на ветке dev с актуальными изменениями:
   ```bash
   git checkout dev
   git status
   ```

2. Проверить что все тесты проходят:
   ```bash
   pytest tests/ -v
   ```

3. Определить тип версии (в VERSION файле):
   - MAJOR (X.0.0) — breaking changes
   - MINOR (0.X.0) — новый функционал
   - PATCH (0.0.X) — багфиксы

4. Обновить VERSION файл с новой версией

5. Обновить CHANGELOG.md:
   - Переименовать [Unreleased] в [X.Y.Z] – YYYY-MM-DD
   - Создать новую пустую секцию [Unreleased]

6. Закоммитить изменения в dev:
   ```bash
   git add -A
   git commit -m "chore: release vX.Y.Z"
   git push origin dev
   ```

7. Смержить dev в main:
   ```bash
   git checkout main
   git merge dev
   ```

8. Создать git tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   ```

9. Запушить main с тегами:
   ```bash
   git push origin main --tags
   ```

10. Задеплоить на Fly.io:
    ```bash
    fly deploy
    ```
