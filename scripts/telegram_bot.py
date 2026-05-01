#!/usr/bin/env python3
"""
Telegram-бот ИИ-Юрист (Hellen26_bot).
Фаза 2 — семантический поиск + OpenRouter Claude через Telegram.

Запуск:
    pip3 install aiogram openai python-dotenv numpy --break-system-packages
    python3 scripts/telegram_bot.py

Нужен файл .env:
    OPENROUTER_API_KEY=sk-or-...
    TELEGRAM_BOT_TOKEN=...
    ADMIN_USER_ID=...       (ваш Telegram user ID, узнать: @userinfobot)

Команды бота:
    Пользователь:
        /start — приветствие
        /help  — справка
    Админ:
        /stats   — статистика базы знаний
        /list    — список документов
        /rebuild — пересоздать индекс эмбеддингов
        + отправка .txt файла — загрузка в базу знаний
"""

import os
import json
import hashlib
import asyncio
import numpy as np
from pathlib import Path
from datetime import datetime

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode

# === Конфигурация ===

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "initial_docs"
CACHE_FILE = ROOT / "data" / "embeddings_cache.json"
STATS_FILE = ROOT / "data" / "bot_stats.json"

EMBED_MODEL = "openai/text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# === Системный промпт ===

SYSTEM_PROMPT = """Ты — ИИ-юрист (Hellen26_bot), специализирующийся на российском праве в сфере e-commerce.

ПРАВИЛА:
1. Отвечай точно, ссылайся на конкретные статьи законов и документы из контекста.
2. Если информации в контексте недостаточно — скажи об этом прямо.
3. НИКОГДА не выдумывай номера статей, законов или нормативных актов.
4. Если вопрос выходит за рамки предоставленного контекста — так и скажи.
5. Используй простой, понятный язык. Юридические термины объясняй.
6. Структурируй ответ: сначала прямой ответ, потом обоснование со ссылками.
7. Отвечай кратко и по существу — это Telegram, не статья.

ВАЖНЫЕ ФАКТЫ ДЛЯ ВАЛИДАЦИИ:
- В Законе о защите прав потребителей (ЗоЗПП) всего 46 статей. Если спрашивают о статье с номером больше 46 — такой статьи НЕ СУЩЕСТВУЕТ, скажи об этом прямо.
- Статья 26.1 ЗоЗПП — дистанционная торговля, срок возврата 7 дней.
- Статья 18 ЗоЗПП — права при обнаружении недостатков, отсутствие чека не основание для отказа.
- Статья 22 ЗоЗПП — срок возврата денег 10 дней.
- Статья 23 ЗоЗПП — неустойка 1% в день.

ФОРМАТ ОТВЕТА (для Telegram — без markdown заголовков):
- Прямой ответ на вопрос (1-2 предложения, жирным)
- Обоснование со ссылками на конкретные статьи/документы
- Практический совет (что делать дальше)
"""


# === Разбивка на чанки ===

def chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Разбивает текст на перекрывающиеся чанки."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break

        chunk_content = text[start:end].strip()
        if len(chunk_content) > 50:
            chunks.append({"title": title, "content": chunk_content})

        start = end - overlap if end < len(text) else len(text)
    return chunks


# === Эмбеддинги через OpenRouter ===

def get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """Получает эмбеддинги через OpenRouter API."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    all_embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# === Семантический поисковый движок ===

class SemanticSearchEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chunks = []
        self.embeddings = None
        self._cache_hash = None
        self._doc_titles = {}  # filename -> title

    def load_documents(self) -> dict:
        """Загрузка и разбивка документов. Возвращает статистику."""
        self.chunks = []
        self._doc_titles = {}

        stats = {"loaded": 0, "chunks": 0, "errors": []}

        if not DOCS_DIR.exists():
            stats["errors"].append(f"Папка {DOCS_DIR} не найдена")
            return stats

        # Загружаем все .txt файлы из папки
        all_content = ""
        for filepath in sorted(DOCS_DIR.glob("*.txt")):
            try:
                content = filepath.read_text(encoding="utf-8")
                # Берём заголовок из первой строки или имени файла
                first_line = content.strip().split("\n")[0][:80]
                title = first_line if first_line else filepath.stem

                all_content += content
                doc_chunks = chunk_text(content, title)
                self.chunks.extend(doc_chunks)
                self._doc_titles[filepath.name] = title
                stats["loaded"] += 1
            except Exception as e:
                stats["errors"].append(f"{filepath.name}: {e}")

        stats["chunks"] = len(self.chunks)
        self._cache_hash = hashlib.md5(all_content.encode()).hexdigest()
        return stats

    def build_index(self) -> str:
        """Создание или загрузка эмбеддингов. Возвращает статус."""
        if not self.chunks:
            return "❌ Нет документов для индексации"

        # Проверяем кэш
        if CACHE_FILE.exists():
            try:
                cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                if cache.get("hash") == self._cache_hash:
                    self.embeddings = np.array(cache["embeddings"])
                    return f"📦 Загружен кэш ({len(self.chunks)} чанков)"
            except Exception:
                pass

        # Создаём эмбеддинги
        texts = [chunk["content"] for chunk in self.chunks]
        embeddings = get_embeddings(texts, self.api_key)
        self.embeddings = np.array(embeddings)

        # Сохраняем кэш
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "hash": self._cache_hash,
            "embeddings": [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings],
        }
        CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
        return f"✅ Создано {len(self.chunks)} эмбеддингов, кэш сохранён"

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Семантический поиск по cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_embedding = np.array(get_embeddings([query], self.api_key)[0])

        scores = []
        for i in range(len(self.chunks)):
            score = cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        title_count = {}
        for idx, score in scores[:top_k * 3]:
            chunk = self.chunks[idx]
            title = chunk["title"]
            title_count[title] = title_count.get(title, 0) + 1
            if title_count[title] > 2:
                continue
            results.append({
                "title": title,
                "content": chunk["content"],
                "score": round(score, 4),
            })
            if len(results) >= top_k:
                break

        return results

    def add_document_file(self, filename: str, content: str) -> str:
        """Добавляет новый документ и перестраивает индекс."""
        filepath = DOCS_DIR / filename
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

        # Перезагружаем всё
        stats = self.load_documents()
        index_status = self.build_index()
        return f"✅ Добавлен: {filename}\n📊 {stats['loaded']} документов, {stats['chunks']} чанков\n{index_status}"

    def delete_document(self, filename: str) -> str:
        """Удаляет документ и перестраивает индекс."""
        filepath = DOCS_DIR / filename
        if not filepath.exists():
            return f"❌ Файл {filename} не найден"

        filepath.unlink()
        stats = self.load_documents()
        index_status = self.build_index()
        return f"🗑 Удалён: {filename}\n📊 {stats['loaded']} документов, {stats['chunks']} чанков\n{index_status}"

    def get_doc_list(self) -> list[str]:
        """Список документов."""
        if not DOCS_DIR.exists():
            return []
        return sorted([f.name for f in DOCS_DIR.glob("*.txt")])


# === LLM ===

def ask_llm(query: str, context: str, api_key: str, model: str) -> str:
    """Отправляет запрос в OpenRouter."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/amapemom-rgb/ai-lawyer",
            "X-Title": "AI-Lawyer Hellen26",
        },
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "user", "content": f"Контекст из базы знаний:\n{context}"})
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
    )
    return response.choices[0].message.content


# === Статистика ===

class BotStats:
    def __init__(self):
        self.data = {"queries": 0, "users": [], "started": datetime.now().isoformat()}
        self._load()

    def _load(self):
        if STATS_FILE.exists():
            try:
                self.data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _save(self):
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_query(self, user_id: int):
        self.data["queries"] = self.data.get("queries", 0) + 1
        if user_id not in self.data.get("users", []):
            self.data.setdefault("users", []).append(user_id)
        self._save()


# === Конвертация Markdown → HTML ===

import re

def md_to_html(text: str) -> str:
    """Конвертирует базовый Markdown в HTML для Telegram."""
    # Убираем ## заголовки → жирный текст
    text = re.sub(r'^#{1,3}\s+(.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    # **жирный** → <b>жирный</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # *курсив* → <i>курсив</i> (но не внутри уже обработанных тегов)
    text = re.sub(r'(?<!</b>)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    # `код` → <code>код</code>
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # Экранируем HTML-спецсимволы которые могут сломать парсинг
    # (но не трогаем уже вставленные теги)
    return text


# === Telegram-бот ===

router = Router()

# Глобальные объекты
engine: SemanticSearchEngine = None
bot_stats: BotStats = None
API_KEY: str = ""
MODEL: str = ""
ADMIN_ID: int = 0


def is_admin(message: Message) -> bool:
    return message.from_user.id == ADMIN_ID


# --- Команды для всех ---

@router.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "⚖️ <b>ИИ-Юрист — Hellen26</b>\n\n"
        "Привет! Я — юридический ассистент по e-commerce.\n\n"
        "Я помогу с:\n"
        "• Возвратом товаров на маркетплейсах\n"
        "• Правами потребителей при онлайн-покупках\n"
        "• Составлением претензий\n"
        "• Разбором спорных ситуаций\n\n"
        "Просто напишите свой вопрос!",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    text = (
        "📖 <b>Как пользоваться</b>\n\n"
        "Просто напишите вопрос — я найду нужные законы и дам ответ со ссылками.\n\n"
        "<b>Примеры вопросов:</b>\n"
        "• Можно ли вернуть товар без чека?\n"
        "• Какой срок возврата на Wildberries?\n"
        "• Продавец не возвращает деньги, что делать?\n\n"
        "/start — начать сначала\n"
        "/help — эта справка"
    )
    if is_admin(message):
        text += (
            "\n\n<b>Команды админа:</b>\n"
            "/stats — статистика\n"
            "/list — документы в базе\n"
            "/rebuild — пересоздать индекс\n"
            "📎 Отправьте .txt файл — добавлю в базу знаний"
        )
    await message.answer(text, parse_mode=ParseMode.HTML)


# --- Команды админа ---

@router.message(Command("stats"))
async def cmd_stats(message: Message):
    if not is_admin(message):
        await message.answer("⛔ Команда только для администратора.")
        return

    docs = engine.get_doc_list()
    await message.answer(
        f"📊 <b>Статистика Hellen26</b>\n\n"
        f"📚 Документов: {len(docs)}\n"
        f"🧩 Чанков: {len(engine.chunks)}\n"
        f"❓ Запросов обработано: {bot_stats.data.get('queries', 0)}\n"
        f"👥 Уникальных пользователей: {len(bot_stats.data.get('users', []))}\n"
        f"🤖 Модель: {MODEL}\n"
        f"🔍 Эмбеддинги: {EMBED_MODEL}",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("list"))
async def cmd_list(message: Message):
    if not is_admin(message):
        await message.answer("⛔ Команда только для администратора.")
        return

    docs = engine.get_doc_list()
    if not docs:
        await message.answer("📂 База знаний пуста.")
        return

    lines = [f"📂 <b>Документы в базе ({len(docs)}):</b>\n"]
    for i, name in enumerate(docs, 1):
        title = engine._doc_titles.get(name, name)
        lines.append(f"{i}. <code>{name}</code>\n   └ {title}")

    await message.answer("\n".join(lines), parse_mode=ParseMode.HTML)


@router.message(Command("rebuild"))
async def cmd_rebuild(message: Message):
    if not is_admin(message):
        await message.answer("⛔ Команда только для администратора.")
        return

    await message.answer("🔄 Пересоздаю индекс...")
    stats = engine.load_documents()
    index_status = engine.build_index()
    await message.answer(
        f"✅ <b>Индекс пересоздан</b>\n\n"
        f"📚 Документов: {stats['loaded']}\n"
        f"🧩 Чанков: {stats['chunks']}\n"
        f"{index_status}",
        parse_mode=ParseMode.HTML,
    )


@router.message(Command("delete"))
async def cmd_delete(message: Message):
    if not is_admin(message):
        await message.answer("⛔ Команда только для администратора.")
        return

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer("Использование: /delete имя_файла.txt")
        return

    filename = args[1].strip()
    result = engine.delete_document(filename)
    await message.answer(result)


# --- Загрузка документов (админ) ---

@router.message(F.document)
async def handle_document(message: Message):
    if not is_admin(message):
        await message.answer(
            "📎 Загрузка документов доступна только администратору.\n"
            "Просто напишите свой юридический вопрос!"
        )
        return

    doc = message.document
    file_name = doc.file_name or "unknown.txt"
    file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    supported = {"txt"}
    if file_ext not in supported:
        await message.answer(
            f"⚠️ Пока поддерживается только .txt\n"
            f"Формат .{file_ext} будет добавлен позже."
        )
        return

    await message.answer(f"📥 Принимаю <code>{file_name}</code>...", parse_mode=ParseMode.HTML)

    try:
        file = await message.bot.get_file(doc.file_id)
        file_bytes = await message.bot.download_file(file.file_path)
        content = file_bytes.read().decode("utf-8")

        result = engine.add_document_file(file_name, content)
        await message.answer(f"📄 <b>{file_name}</b>\n{result}", parse_mode=ParseMode.HTML)
    except Exception as e:
        await message.answer(f"❌ Ошибка: {e}")


# --- Основной обработчик вопросов ---

@router.message(F.text)
async def handle_question(message: Message):
    """Обработка юридических вопросов."""
    query = message.text.strip()
    if not query or query.startswith("/"):
        return

    # Показываем "печатает..."
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")

    bot_stats.record_query(message.from_user.id)

    try:
        # Семантический поиск
        results = engine.search(query, top_k=5)

        if results:
            context_parts = []
            sources_text = []
            for i, r in enumerate(results[:3], 1):
                context_parts.append(f"--- {r['title']} ---\n{r['content']}")
                sources_text.append(f"  {i}. {r['title']} ({r['score']})")
            context = "\n\n".join(context_parts)
        else:
            context = ""
            sources_text = []

        # Генерация ответа через Claude
        answer = ask_llm(query, context, API_KEY, MODEL)

        # Конвертируем Markdown → HTML для Telegram
        answer = md_to_html(answer)

        # Добавляем источники
        if sources_text:
            answer += "\n\n📄 <i>Источники:</i>\n" + "\n".join(sources_text)

        # Telegram лимит — 4096 символов
        if len(answer) > 4000:
            parts = [answer[i:i + 4000] for i in range(0, len(answer), 4000)]
            for part in parts:
                await message.answer(part, parse_mode=ParseMode.HTML)
        else:
            await message.answer(answer, parse_mode=ParseMode.HTML)

    except Exception as e:
        error_msg = str(e)
        if "402" in error_msg:
            await message.answer("❌ Недостаточно средств на OpenRouter. Обратитесь к администратору.")
        elif "401" in error_msg:
            await message.answer("❌ Ошибка авторизации API. Обратитесь к администратору.")
        else:
            await message.answer(f"❌ Произошла ошибка. Попробуйте переформулировать вопрос.\n\n<i>{error_msg[:200]}</i>", parse_mode=ParseMode.HTML)


# === Запуск ===

async def main():
    global engine, bot_stats, API_KEY, MODEL, ADMIN_ID

    print()
    print("=" * 60)
    print("  ⚖️  HELLEN26 — ИИ-Юрист в Telegram")
    print("  Фаза 2: семантический поиск + OpenRouter Claude")
    print("=" * 60)
    print()

    # Проверяем переменные окружения
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not API_KEY:
        print("❌ OPENROUTER_API_KEY не найден в .env")
        return
    if not tg_token:
        print("❌ TELEGRAM_BOT_TOKEN не найден в .env")
        return

    MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
    admin_id_str = os.getenv("ADMIN_USER_ID", "0")
    ADMIN_ID = int(admin_id_str) if admin_id_str.isdigit() else 0

    print(f"🤖 Модель: {MODEL}")
    print(f"🔍 Эмбеддинги: {EMBED_MODEL}")
    print(f"👤 Админ ID: {ADMIN_ID if ADMIN_ID else 'не задан (команды админа отключены)'}")
    print()

    # Загрузка базы знаний
    print("📚 Загрузка базы знаний...")
    engine = SemanticSearchEngine(API_KEY)
    stats = engine.load_documents()
    print(f"   📊 Загружено: {stats['loaded']} документов, {stats['chunks']} чанков")

    if stats["errors"]:
        for err in stats["errors"]:
            print(f"   ❌ {err}")

    # Индексация
    index_status = engine.build_index()
    print(f"   {index_status}")
    print()

    # Статистика
    bot_stats = BotStats()

    # Запуск бота
    bot = Bot(token=tg_token)
    dp = Dispatcher()
    dp.include_router(router)

    print("✅ Бот Hellen26 запущен! Ожидаю сообщения...")
    print("   Для остановки: Ctrl+C")
    print()

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
