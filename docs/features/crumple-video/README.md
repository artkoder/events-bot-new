# CrumpleVideo

Фича генерации видео-анонсов `/v` (intro + афиши + outro) через пайплайн
CrumpleVideo/Blender. Этот документ собирает требования и проблемы тестового рендера
в одну точку.

## Контекст запуска

### Тестовый запуск (`/v - Тест завтра`)

- Состав: intro + 1 афиша + outro.
- Должна использоваться одна реальная афиша из базы анонсов (не тестовый набор).

### Боевой запуск (`/v - Запуск завтра`)

- Состав: intro + 2-12 афиш + outro.
- Афиши берутся из базы анонсов (без тестовых подстановок).

## Проблемы и наблюдения (последний тестовый прогон)

- Рендер занял `7314.6s` (ускорение требуется только для тестового запуска).
- В тестовом запуске появилось 2 афиши, и это тестовые афиши, а не из БД.
- Intro не соответствует актуальному макету (похоже на старый паттерн).
- Логи показывают `Using legacy dataset: /kaggle/input/afisha-dataset-2`,
  `Posters to render: 4`, и таймауты Blender по 30 минут на каждый рендер.

## Требования

### Производительность (только тестовый запуск)

- Цель: ускорение рендера тестового запуска минимум в 10 раз.
- Ориентир: `~731s` на полный тестовый прогон (intro + 1 афиша + outro).
- Недопустимо: таймаут Blender 30 минут на один постер.

### Аудио

- использовать только `The_xx_-_Intro.mp3`.
- начало аудио с 1:11 от начала аудиозаписи

### Состав и источник афиш

- Тестовый запуск:
  - ровно 1 афиша;
  - из базы анонсов (реальная, выбранная на завтра).
- Боевой запуск:
  - 2-12 афиш;
  - все из базы анонсов;
  - отсутствие тестовых картинок в финальном видео.

### Качество афиш (OCR / полнота данных)

- В видео попадают только афиши с непустым `EventPoster.ocr_text` (пустые/пунктуация считаются пустыми).
- На афише должны присутствовать: название события, дата+время, место проведения.
- Если `ocr_text` есть, но на афише не хватает части данных, пайплайн добавляет плашку с недостающей
  информацией (best-effort размещение в зоне с низкой «плотностью текста»).
- Проверка полноты `ocr_text` делается через `Gemma-3-27b` (Google AI клиент + общий rate-limit фреймворк).

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

## Критерии приемки (готово для нового тестирования)

- Тестовый запуск `/v - Тест завтра`:
  - intro соответствует одному из макетов (weekend/day);
  - ровно 1 афиша из БД;
  - итоговый рендер <= 731s;
  - нет таймаутов Blender.
