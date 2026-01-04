Проблема оказалась не в `rgba_image_to_clip`, а в загрузке сценовых изображений: `ImageClip(image_path)` может возвращать 2D кадры для grayscale/палитровых файлов, из‑за чего `CompositeVideoClip` пытается вставить (H,W) в (H,W,3) и падает. Я перевёл загрузку сценовых картинок на PIL→RGBA→`rgba_image_to_clip`, чтобы гарантировать 3D кадры и корректную маску альфа. Изменение в `kaggle/VideoAfisha/video_afisha.ipynb`.

**Где в ноутбуке создаются маски**
- `kaggle/VideoAfisha/video_afisha.ipynb`: `rgba_image_to_clip` создаёт `mask = ImageClip(alpha, ismask=True)` и назначает `clip.set_mask(mask)`.
- `kaggle/VideoAfisha/video_afisha.ipynb`: `create_overlay_cover` вызывает `clip.add_mask()` для бенто‑картинок без маски.
- `kaggle/VideoAfisha/video_afisha.ipynb`: `dimmer = ColorClip(...).set_opacity(...)` — MoviePy внутри создаёт маску непрозрачности.

**Где возникает несовместимость размерностей**
- `kaggle/VideoAfisha/video_afisha.ipynb`: `create_advanced_scene` использовал `ImageClip(image_path)` напрямую. Если исходная картинка grayscale/палитровая, MoviePy отдаёт 2D кадр (примерно 535×432), и при композите это даёт ошибку `could not broadcast ... into shape (H,W,3)` — как раз на старте второй сцены.

**Проверка PIL → MoviePy мест**
- `generate_kinetic_text` → `create_word_clip_fixed_height` → `rgba_image_to_clip`: ок, всегда RGB + 2D mask.
- `create_overlay_cover` (интро): все PIL‑изображения идут через `rgba_image_to_clip`, так что размерности корректные.
- `create_single_word_clip`/`create_sticker_strip`: тоже через `rgba_image_to_clip`, корректно.

**Исправление**
- `kaggle/VideoAfisha/video_afisha.ipynb`: в `create_advanced_scene` заменил прямой `ImageClip(image_path)` на PIL‑загрузку с `convert('RGBA')` и `rgba_image_to_clip`, чтобы исключить 2D кадры.

Коммит сделан в `dev`: `586bbf8`.

Если нужно проверить прямо сейчас:
1. Перезапусти рендер ноутбука до проблемного кадра (примерно 6–7 сек).  
2. Если где‑то ещё используется `ImageClip(path)` с файлами, лучше тоже прогнать через PIL→RGBA по этому же шаблону.