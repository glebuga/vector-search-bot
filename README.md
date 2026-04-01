<img width="1264" height="842" alt="image" src="https://github.com/user-attachments/assets/2ddfbde0-fe75-4e37-8763-6c3d10692dd1" />


# RAG (PDF -> Chunks -> Vectors -> LLM) Microservices

Проект разбит на микросервисы и предназначен для RAG-поиска по одному PDF-документу, загруженному пользователем в Telegram.

## Состав сервисов

1. `telegram-bot`
   - Telegram-интерфейс: принимает команды/файлы от пользователя и показывает ответы.
   - Загружает PDF в `rag-api`.
   - Отправляет вопросы в `rag-api` и отображает ответ.
   - Кнопка `Закончить` вызывает удаление данных пользователя из БД.

2. `rag-api`
   - Единственный сервис, который делает тяжёлую обработку:
     - извлечение текста из PDF
     - “сборку” текста
     - чанкинг
     - генерацию эмбеддингов
     - сохранение в PostgreSQL + `pgvector`
     - retrieval по `ORDER BY embedding <-> query LIMIT k`
     - формирование контекста (конкатенация топ-чанков)
   - После retrieval делает запрос в `llm-service` и получает финальный ответ.

3. `llm-service`
   - Языковая модель/LLM слой.
   - Получает JSON `{ question, context }`.
   - Сам формирует prompt по `SYSTEM_PROMPT`.
   - Возвращает готовый текст ответа.

4. `postgres`
   - PostgreSQL с расширением `pgvector`.
   - Хранит:
     - `documents` (по одному активному документу на `telegram_user_id`)
     - `chunks` с векторами `vector(768)`

## Изоляция данных

- Изоляция реализована по `telegram_user_id`.
- Правило “один активный документ”:
  - при загрузке нового PDF для пользователя `rag-api` удаляет предыдущие записи (`documents` -> автоматически удаляются `chunks` из-за `ON DELETE CASCADE`)
  - затем сохраняет новый документ и чанки
- Кнопка `Закончить` удаляет все записи пользователя из БД.

## Что хранится в БД

- Оригинальный PDF на диск не сохраняется (в `documents.file_path` хранится `NULL`).
- В БД сохраняются:
  - текст чанков (`chunks.chunk_text`)
  - эмбеддинги чанков (`chunks.embedding vector(768)`)
  - метаданные (`documents.metadata`, `chunks.metadata`, keywords опционально)

## Эндпоинты `rag-api`

### Health

`GET /api/health`

Ответ:
```json
{ "status": "ok" }
```

### Загрузка PDF

`POST /api/documents/upload`

Тип: `multipart/form-data`

Поля:
- `telegram_user_id` (required, form field)
- `pdf_file` (required, file field)
- `doc_name` (optional, form field)
- `doc_number` (optional, form field)

Ответ:
```json
{ "document_id": 1, "chunks_count": 42 }
```

### Вопрос по документу

`POST /api/qa/ask`

Body (JSON):
```json
{
  "telegram_user_id": 123456789,
  "question": "Вопрос...",
  "top_k": 5
}
```

Ответ:
```json
{ "answer": "..." }
```

Если информация не найдена — ответ будет:
`Информация по данному вопросу не найдена в базе`

### Завершить работу с документом

`POST /api/documents/finish`

Тип: `application/x-www-form-urlencoded` (или form-data)

Поля:
- `telegram_user_id` (required)

Ответ:
```json
{ "status": "ok", "result": "..." }
```

## Эндпоинт `llm-service`

`POST /api/llm/answer`

Body:
```json
{ "question": "Вопрос...", "context": "Контекст..." }
```

Ответ:
```json
{ "answer": "..." }
```

## Переменные окружения (.env)

В корне проекта нужен файл `.env`.
Пример шаблона: `.env.example`.

### Что обязательно

- `BOT_TOKEN` — токен Telegram-бота
- `OPENROUTER_API_KEY` — ключ OpenRouter

- `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME` — настройки PostgreSQL
  - для Docker-сети обычно `DB_HOST=postgres` (имя сервиса из `docker-compose.yml`)

Пример:
```env
BOT_TOKEN=PASTE_TELEGRAM_BOT_TOKEN_HERE
OPENROUTER_API_KEY=PASTE_OPENROUTER_API_KEY_HERE

DB_USER=rag_bot_user
DB_PASSWORD=1234512345
DB_HOST=postgres
DB_PORT=5432
DB_NAME=rag_bot_db
```

## Подготовка init SQL для БД

SQL для создания таблиц лежит тут:
- `postgres/init/001_init.sql`

Там:
- создаётся `vector` расширение (`pgvector`)
- создаются таблицы `documents` и `chunks`
- `chunks` ссылается на `documents` через `document_id` с `ON DELETE CASCADE`

## Запуск

### 1) Убедись, что есть Docker

В терминале (PowerShell) проверь:
```powershell
docker --version
docker compose version
```

### 2) Подними сервисы

Из корня проекта (где лежит `docker-compose.yml`):
```powershell
docker compose up --build
```

После первого старта могут быть долгие задержки из-за загрузки моделей эмбеддингов/torch.

### 3) Логи
```powershell
docker compose logs -f rag-api
docker compose logs -f llm-service
docker compose logs -f telegram-bot
docker compose logs -f postgres
```

## Остановка

Обычная остановка:
```powershell
docker compose down
```

Если нужно полностью очистить данные БД:
```powershell
docker compose down -v
```

## Как это работает в Telegram (поток)

1. Пользователь нажимает `/start`
2. Нажимает `Загрузить PDF`
3. Бот отправляет PDF в `rag-api` и ждёт, пока обработка завершится (синхронно)
4. Пользователь задаёт вопросы кнопкой/сообщениями
5. Ответ приходит обратно от `rag-api` (через `llm-service`)
6. По кнопке `Закончить` удаляются все данные пользователя из БД, и можно загружать новый PDF

## Заметки

- Пока идёт обработка PDF, бот не принимает вопросы.
- Оригинальный PDF не хранится (только извлечённый текст/чанки/вектора).

