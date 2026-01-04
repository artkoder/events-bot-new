Добавил баннер версии в первую code‑ячейку и усилил логирование вокруг выбора/запуска kernel и push, чтобы было видно, какая версия и что именно уходит на Kaggle.
- `kaggle/VideoAfisha/video_afisha.ipynb` — вставил `NOTEBOOK_VERSION` и print в первой code‑ячейке.
- `video_announce/scenario.py` — логируется выбранный `kernel_ref` при клике пользователя.
- `video_announce/kaggle_client.py` — логируется полный путь локального kernel и список файлов при push (и в `deploy_kernel_update`, и в `push_kernel`).
- `CHANGELOG.md` — добавил запись в [Unreleased].
- Проверка выбора: кнопка "Video Afisha (репозиторий)" формируется из папки `kaggle/VideoAfisha` и даёт `kernel_ref=local:VideoAfisha`; деплой берёт код из `kaggle/VideoAfisha/` через `KERNELS_ROOT_PATH`.

Тесты не запускал.

Вижу неожиданный untracked файл `​.codex/reports/debug_kaggle_notebook_upload.md` — он не мой. Как поступаем с ним перед коммитом?

Готов закоммитить в `dev`, сейчас на `main`. Подтвердите, пожалуйста:
1) игнорировать этот файл и просто переключиться на `dev` и закоммитить мои изменения;
2) включить файл в коммит;
3) удалить/игнорировать его отдельно.