# [EVE-11](https://linear.app/events-bot-new/issue/EVE-11/globalnyj-frejmvork-upravleniya-limitami-llm)

**Title:** Глобальный фреймворк управления лимитами LLM  
**Status:** Backlog  
**Priority:** High  
**Assignee:** Zigo Maro

### Description
В системе уже есть процесс передачи информации о каждом вызове LLM в supabase. Нужно обеспечить соблюдение лимитов на запросы в LLM без конкуренции и с ретраями (3 ретрая максимум). Перед кадждым запросом в том числе ретраем нужно проверить доступность лимитов по своей модели.

Для запросов к Gemma использовать Google API, переменная GOOGLE_API_KEY, и переменная указывающая название аккаунта GOOGLE_API_LOCALNAME (лимит считается по конкретному аккаунту)

Лимиты на Google прикладываю:
![image.png](assets/eve-11-limits-google.png)

Схема БД в supabase уже настроена под эту задачу:
![image.png](assets/eve-11-supabase-schema.png)
