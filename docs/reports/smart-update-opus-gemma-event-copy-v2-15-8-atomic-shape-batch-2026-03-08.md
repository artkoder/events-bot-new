# Smart Update Gemma Event Copy V2.15.8 Atomic Shape Batch

Date: 2026-03-08

Этот batch-цикл расширяет step-by-step tuning на новые shape-классы: `screening_card`, `party_theme_program`, `exhibition_context_collection`.

Цель остаётся прежней: не подгонка под один event id, а поиск переносимых атомарных контрактов для Gemma.

## Event 2659 — Посторонний

- Shape: `screening_card`
- Type: `кинопоказ`
- Note: single-film screening with adaptation/festival card and plot hook
- Baseline missing deterministic: `8`
- Final slot score: `6/6`
- Lead ok: `no`
- Generic filler hits: `0`
- Hallucination hits: `0`
- Missing deterministic: `7`

### Winning Stages

- `normalize_card`: chosen `normalize_card_v1`, win=`yes`
- `extract_plot`: chosen `extract_plot_v1`, win=`yes`
- `normalize_support`: chosen `normalize_support_v1`, win=`yes`
- `plan_lead`: chosen `plan_v1_basic`, win=`no`
- `generate_lead`: chosen `lead_v1`, win=`no`
- `generate_body`: chosen `body_v1`, win=`no`
- `repair`: chosen `repair_v1`, win=`no`

### Final Text

```md
Экранизация романа Альбера Камю «Посторонний» от Франсуа Озона погружает зрителя в мир человека, отчужденного от общества и собственных чувств, где равнодушие становится предвестником трагических событий. История Мерсо – это исследование абсурдности существования и столкновение личности с безразличным миром.

Действие фильма разворачивается в Алжире 1938 года, перенося зрителя в эпоху, предшествующую колониальным войнам. Продолжительность картины составляет 125 минут. Фильм представлен в формате дублированного перевода и имеет возрастной рейтинг 18+. Показ состоится в кинозале филиала Третьяковской галереи, а мировая премьера запланирована на Венецианском кинофестивале в 2025 году.
```

## Event 2747 — Киноклуб: «Последнее метро»

- Shape: `screening_card`
- Type: `кинопоказ`
- Note: classic film-club screening with award/cast card and premise
- Baseline missing deterministic: `5`
- Final slot score: `6/6`
- Lead ok: `no`
- Generic filler hits: `0`
- Hallucination hits: `0`
- Missing deterministic: `7`

### Winning Stages

- `normalize_card`: chosen `normalize_card_v1`, win=`yes`
- `extract_plot`: chosen `extract_plot_v1`, win=`yes`
- `normalize_support`: chosen `normalize_support_v1`, win=`yes`
- `plan_lead`: chosen `plan_v1_basic`, win=`no`
- `generate_lead`: chosen `lead_v1`, win=`no`
- `generate_body`: chosen `body_v1`, win=`no`
- `repair`: chosen `repair_v1`, win=`no`

### Final Text

```md
В оккупированном Париже, сквозь призму личной драмы и художественного гения Франсуа Трюффо, разворачивается история «Последнего метро» – фильма, получившего десять наград «Сезар» и повествующего о любви, предательстве и силе искусства во времена Второй мировой войны. Это история о театре, который продолжает жить, несмотря на тьму, и о людях, готовых рисковать всем ради искусства и друг друга.

В центре сюжета – театр, оказавшийся под жестким контролем нацистской администрации. Директор театра, вынужденный сотрудничать с оккупантами, скрывает от всех своего любовный треугольник с актрисой и еврейским пианистом, который вынужден скрываться в подвале театра. В главных ролях блистают Катрин Денев и Жерар Депардье, воплотившие сложные и противоречивые характеры героев. Продолжительность фильма – 2 часа 10 минут. Приглашаем вас на показ «Последнего метро»! Вход – 100 рублей.
```

## Event 2701 — «Татьяна танцует»

- Shape: `party_theme_program`
- Type: `party`
- Note: holiday party manifesto with dress code and atmosphere program
- Baseline missing deterministic: `5`
- Final slot score: `6/6`
- Lead ok: `no`
- Generic filler hits: `0`
- Hallucination hits: `0`
- Missing deterministic: `10`

### Winning Stages

- `normalize_identity`: chosen `normalize_identity_v2_strict`, win=`yes`
- `normalize_participation`: chosen `normalize_participation_v1`, win=`yes`
- `normalize_program`: chosen `normalize_program_v1`, win=`yes`
- `plan_lead`: chosen `plan_v1_basic`, win=`no`
- `generate_lead`: chosen `lead_v1`, win=`no`
- `generate_body`: chosen `body_v2_literal`, win=`no`
- `repair`: chosen `repair_v1`, win=`no`

### Final Text

```md
8 марта в этом году мы не будем ждать подарков – мы сами устроим себе праздник, полный энергии, танцев и настоящей женской силы. «Татьяна танцует» – это вечеринка для тех, кто строит свой мир, поддерживает друг друга и помнит, что женская солидарность способна творить чудеса.

Организатор вечеринки – бар «Татьяна». Дресс-код прост: «примерь образ, который всегда хотела». Вас ждет красная дорожка с репортерами и фоточками для соцсетей, а также тематические фотозоны. В программе вечера – музыка 2000-х и 2010-х годов. Мероприятие «Татьяна танцует» пройдет 8 марта с 22:00 до 04:00 в Колонном зале «Зари» по адресу Мира 41-43.
```

## Event 2732 — Вечер в русском стиле

- Shape: `party_theme_program`
- Type: `party`
- Note: museum costume evening with exhibition context and literal activity program
- Baseline missing deterministic: `3`
- Final slot score: `6/6`
- Lead ok: `no`
- Generic filler hits: `0`
- Hallucination hits: `0`
- Missing deterministic: `8`

### Winning Stages

- `normalize_identity`: chosen `normalize_identity_v1`, win=`yes`
- `normalize_participation`: chosen `normalize_participation_v1`, win=`yes`
- `normalize_program`: chosen `normalize_program_v1`, win=`yes`
- `plan_lead`: chosen `plan_v1_basic`, win=`yes`
- `generate_lead`: chosen `lead_v1`, win=`no`
- `generate_body`: chosen `body_v1`, win=`no`
- `repair`: chosen `repair_v1`, win=`no`

### Final Text

```md
Погрузитесь в мир загадочной русской души и космических горизонтов на вечере, вдохновленном выставкой «Космос красного». Этот костюмированный бал перенесет вас в эпоху, где традиции переплетаются с мечтами о будущем, а атмосфера таинственности и праздника царит в воздухе.

Организованный Музеем изобразительных искусств, вечер предлагает гостям нарядиться в исконно русские костюмы: кокошники, косоворотки, сарафаны, шелковые пояса и другие аксессуары в русском стиле. Будет проведен конкурс на лучший образ с ценными призами. В программе вечера – экскурсия, мастер-классы, викторина, фотозона и, конечно, конкурс костюмов. Дополнят атмосферу волшебные предсказания в стиле народной мудрости. Стоимость посещения мероприятия составляет 500 рублей.
```

## Event 2759 — Выставка «Королева Луиза: идеал или красивая легенда?»

- Shape: `exhibition_context_collection`
- Type: `выставка`
- Note: historical-person exhibition with anniversary and archival object cluster
- Baseline missing deterministic: `12`
- Final slot score: `4/5`
- Lead ok: `yes`
- Generic filler hits: `0`
- Hallucination hits: `0`
- Missing deterministic: `7`

### Winning Stages

- `normalize_theme`: chosen `normalize_theme_v1`, win=`yes`
- `normalize_objects`: chosen `normalize_objects_v1`, win=`yes`
- `normalize_signature`: chosen `normalize_signature_v1`, win=`no`
- `plan_lead`: chosen `plan_v1_basic`, win=`no`
- `generate_lead`: chosen `lead_v1`, win=`yes`
- `generate_body`: chosen `body_v1`, win=`no`
- `repair`: chosen `repair_v1`, win=`yes`

### Final Text

```md
В год 250-летия со дня рождения королевы Луизы, выставка в «Пакгаузе» (набережная Петра Великого, 5) предлагает заново взглянуть на личность, окутанную легендами и восхищением, и отделить исторический портрет от идеализированного образа. Погружаясь в судьбу этой выдающейся женщины, выставка «Королева Луиза: идеал или красивая легенда?» исследует её биографию и вклад в историю Пруссии.

В экспозиции представлены письма и дневники королевы Луизы, позволяющие увидеть её мысли и переживания напрямую. Наряду с личными документами, посетители смогут ознакомиться с произведениями искусства XVIII–XIX веков, отражающими эпоху и культурный контекст жизни королевы. Дополняют экспозицию редкие предметы быта XVIII–XIX веков, воссоздающие атмосферу прусского двора и повседневной жизни того времени. Выставка организована Музеем Мирового океана совместно с государственным музеем-заповедником «Царское Село».
```

## Event 2657 — Коллекция украшений 1930–1960-х годов

- Shape: `exhibition_context_collection`
- Type: `выставка`
- Note: collection exhibition with historical-fashion context and significance line
- Baseline missing deterministic: `11`
- Final slot score: `5/5`
- Lead ok: `yes`
- Generic filler hits: `0`
- Hallucination hits: `0`
- Missing deterministic: `7`

### Winning Stages

- `normalize_theme`: chosen `normalize_theme_v1`, win=`yes`
- `normalize_objects`: chosen `normalize_objects_v1`, win=`no`
- `normalize_signature`: chosen `normalize_signature_v1`, win=`yes`
- `plan_lead`: chosen `plan_v1_basic`, win=`yes`
- `generate_lead`: chosen `lead_v1`, win=`yes`
- `generate_body`: chosen `body_v1`, win=`yes`

### Final Text

```md
Выставка «Доступная роскошь» открывает уникальную страницу истории ювелирного и бижутерийного искусства 1930–1960-х годов, представляя собой знаковое собрание для российских музеев и отражая дух эпохи, когда элегантность стала ближе к широкому кругу людей.

В экспозиции представлено 240 предметов из музейной коллекции, созданных в период «золотого века» костюмных украшений в США – 1930–1960-е годы. Выставка отражает влияние таких значительных исторических событий, как Великая депрессия и Вторая мировая война, на развитие моды и ювелирного искусства того времени.
```

## Batch Findings

- `screening_card` требует отдельного шага на plot hook, иначе Gemma либо теряет фабульную опору, либо уходит в cinephile filler.
- `party_theme_program` требует разнесения identity / participation / program; иначе dress code и literal activities смешиваются в одно атмосферное описание.
- `exhibition_context_collection` требует отдельного object/signature слоя; иначе объектная конкретика проваливается в generic museum prose.
- Exact-match missing остаётся полезным только как secondary signal; slot coverage точнее показывает перенос semantic contracts.
