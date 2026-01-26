# CrumpleVideo

Фича генерации видео-анонсов `/v` (intro + афиши + outro) через пайплайн
CrumpleVideo/Blender. Этот документ собирает требования и проблемы тестового рендера
в одну точку.

## Контекст запуска

### Тестовый запуск (`/v - Тест завтра`)

- Состав: intro + до 7 афиш + outro.
- Афиши берутся из базы анонсов (без тестовых подстановок).
- Выборка и генерация ограничены 7 афишами.
- Если на завтра мало афиш, окно подбора расширяется волнами:
  завтра → завтра+послезавтра → завтра+послезавтра+послепослезавтра → ...
  (лимит: 5 дней включая завтра).

### Боевой запуск (`/v - Запуск завтра`)

- Состав: intro + 2-12 афиш + outro.
- Афиши берутся из базы анонсов (без тестовых подстановок).
- При нехватке афиш окно подбора расширяется волнами:
  завтра → завтра+послезавтра → завтра+послезавтра+послепослезавтра → ...
  (лимит: 5 дней включая завтра).

## Проблемы и наблюдения (последний тестовый прогон)

- Рендер занял `7314.6s` (ускорение требуется только для тестового запуска).
- В тестовом запуске появилось 2 афиши, и это тестовые афиши, а не из БД.
- Intro не соответствует актуальному макету (похоже на старый паттерн).
- Логи показывают `Using legacy dataset: /kaggle/input/afisha-dataset-2`,
  `Posters to render: 4`, и таймауты Blender по 30 минут на каждый рендер.

## Требования

### Производительность (только тестовый запуск)

- Цель: ускорение рендера тестового запуска минимум в 10 раз.
- Недопустимо: таймаут Blender 30 минут на один постер.

### Аудио

- использовать только `The_xx_-_Intro.mp3`.
- начало аудио с 1:11 от начала аудиозаписи

### Состав и источник афиш

- Тестовый запуск:
  - до 7 афиш;
  - из базы анонсов (реальные, выбранные на завтра).
- Боевой запуск:
  - 2-12 афиш;
  - все из базы анонсов;
  - отсутствие тестовых картинок в финальном видео.
- Для `/v - Запуск завтра` и `/v - Тест завтра` события с распроданными билетами не попадают в выборку.

### Качество афиш (OCR / полнота данных)

- В видео попадают только афиши с непустым `EventPoster.ocr_text` (пустые/пунктуация считаются пустыми).
- На афише должны присутствовать: название события, дата+время, место проведения.
- Если `ocr_text` есть, но на афише не хватает части данных, пайплайн добавляет плашку с недостающей
  информацией (best-effort размещение в зоне с низкой «плотностью текста»).
- Проверка полноты `ocr_text` делается через `Gemma-3-27b` (Google AI клиент + общий rate-limit фреймворк).

### Подпись поста

- Формат: `Видео-анонс #{номер} на завтра {дата или диапазон дат}`.
- Дата берётся из выбранных событий: одна дата или диапазон.

### Intro: актуальный макет

#### Intro ref (weekend)

```css
position: relative;
width: 1080px;
height: 1572px;

background: #F1E44B;

/* 24-25 */
position: absolute;
width: 945px;
height: 308px;
left: 55px;
top: 270px;

font-family: 'Benzin-Bold';
font-style: normal;
font-weight: 400;
font-size: 224px;
line-height: 308px;
text-align: right;

color: #100E0E;

/* января */
position: absolute;
width: 476px;
height: 200px;
left: 850px;
top: 541px;

font-family: 'Bebas Neue';
font-style: normal;
font-weight: 400;
font-size: 200px;
line-height: 200px;

color: #100E0E;

transform: rotate(-90deg);

/* КАЛИНИНГРАД СВЕТЛОГОРСК ЗЕЛЕНОГРАДСК */
position: absolute;
width: 357px;
height: 267px;
left: 435px;
top: 1058px;

font-family: 'Oswald';
font-style: normal;
font-weight: 400;
font-size: 60px;
line-height: 89px;
text-align: right;

color: #100E0E;

/* ВЫХОДНЫЕ */
position: absolute;
width: 710px;
height: 279px;
left: 82px;
top: 779px;

font-family: 'Druk Cyr';
font-style: normal;
font-weight: 700;
font-size: 220px;
line-height: 279px;

color: #100E0E;
```

#### Intro ref (day)

```css
position: relative;
width: 1080px;
height: 1572px;

background: #F1E44B;

/* 19 */
position: absolute;
width: 324px;
height: 308px;
left: 676px;
top: 270px;

font-family: 'Benzin-Bold';
font-style: normal;
font-weight: 400;
font-size: 224px;
line-height: 308px;
text-align: right;

color: #100E0E;

/* января */
position: absolute;
width: 476px;
height: 200px;
left: 850px;
top: 541px;

font-family: 'Bebas Neue';
font-style: normal;
font-weight: 400;
font-size: 200px;
line-height: 200px;

color: #100E0E;

transform: rotate(-90deg);

/* КАЛИНИНГРАД СВЕТЛОГОРСК ЗЕЛЕНОГРАДСК */
position: absolute;
width: 357px;
height: 267px;
left: 435px;
top: 1058px;

font-family: 'Oswald';
font-style: normal;
font-weight: 400;
font-size: 60px;
line-height: 89px;
text-align: right;

color: #100E0E;

/* ПОНЕДЕЛЬНИК */
position: absolute;
width: 724px;
height: 228px;
left: 73px;
top: 827px;

font-family: 'Druk Cyr';
font-style: normal;
font-weight: 700;
font-size: 180px;
line-height: 228px;
text-align: right;

color: #100E0E;
```

#### Intro ref (different months)

```css
/* intro ref (different monhes) */

position: relative;
width: 1080px;
height: 1572px;

background: #F1E44B;


/* КАЛИНИНГРАД СВЕТЛОГОРСК ЗЕЛЕНОГРАДСК */

position: absolute;
width: 357px;
height: 267px;
left: 435px;
top: 1058px;

font-family: 'Oswald';
font-style: normal;
font-weight: 400;
font-size: 60px;
line-height: 89px;
text-align: right;

color: #100E0E;



/* ФЕВРАЛЯ */

position: absolute;
width: 480px;
height: 228px;
left: 317px;
top: 827px;

font-family: 'Druk Cyr';
font-style: normal;
font-weight: 700;
font-size: 180px;
line-height: 228px;

color: #100E0E;



/* ЯНВАРЯ — */

position: absolute;
width: 504px;
height: 228px;
left: 317px;
top: 637px;

font-family: 'Druk Cyr';
font-style: normal;
font-weight: 700;
font-size: 180px;
line-height: 228px;

color: #100E0E;



/* 31 */

position: absolute;
width: 107px;
height: 228px;
left: 157px;
top: 637px;

font-family: 'Druk Cyr';
font-style: normal;
font-weight: 700;
font-size: 180px;
line-height: 228px;
text-align: right;

color: #100E0E;



/* 1 */

position: absolute;
width: 44px;
height: 228px;
left: 220px;
top: 827px;

font-family: 'Druk Cyr';
font-style: normal;
font-weight: 700;
font-size: 180px;
line-height: 228px;
text-align: right;

color: #100E0E;
```

## Критерии приемки (готово для нового тестирования)

- Тестовый запуск `/v - Тест завтра`:
  - intro соответствует одному из макетов (weekend/day);
  - до 7 афиш из БД;
  - события с распроданными билетами исключены;
  - подпись соответствует шаблону;
  - авто-расширение окна подбора не превышает 5 дней;
  - нет таймаутов Blender.
