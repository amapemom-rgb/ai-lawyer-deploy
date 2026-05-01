"""
LightRAG-адаптер — Понятийная Графовая Система (ПГС).

Использует LightRAG для построения графа знаний из документов.
Бот видит связи между понятиями: статьи закона ↔ правила маркетплейсов
↔ шаблоны претензий ↔ способы отправки.

Поиск идёт не только по похожести текста (как semantic),
а по структурным связям в графе.

Требования:
    pip install lightrag-hku
"""

import os
import numpy as np
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


class LightRAGEngine:
    """LightRAG-движок — реализация KnowledgeEngine."""

    ENGINE_NAME = "lightrag"

    def __init__(
        self,
        api_key: str,
        model: str,
        docs_dir: Path,
        working_dir: Path,
        embedding_model: str = "openai/text-embedding-3-small",
        embedding_dim: int = 1536,
    ):
        self.api_key = api_key
        self.model = model
        self.docs_dir = docs_dir
        self.working_dir = working_dir
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._rag: LightRAG | None = None
        self._doc_titles: dict[str, str] = {}
        self._indexed_docs: set[str] = set()

    async def initialize(self) -> str:
        """Инициализировать LightRAG и проиндексировать документы."""
        self.working_dir.mkdir(parents=True, exist_ok=True)

        api_key = self.api_key
        model = self.model
        embedding_model = self.embedding_model

        # LLM-функция через OpenRouter (OpenAI-совместимый API)
        async def llm_func(
            prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
        ) -> str:
            return await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                **kwargs,
            )

        # Эмбеддинг-функция через OpenRouter
        async def embed_func(texts: list[str]) -> np.ndarray:
            return await openai_embed.func(
                texts,
                model=embedding_model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        self._rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=8192,
                func=embed_func,
            ),
            # Лёгкие хранилища по умолчанию — идеально для VPS 1GB RAM
            # kv_storage="JsonKVStorage",        (default)
            # vector_storage="NanoVectorDBStorage", (default)
            # graph_storage="NetworkXStorage",    (default)
            addon_params={
                "language": "Russian",
                "entity_types": [
                    "закон", "статья", "право", "обязанность",
                    "маркетплейс", "продавец", "потребитель",
                    "претензия", "суд", "неустойка", "возврат",
                    "товар", "организация", "срок",
                ],
            },
        )

        await self._rag.initialize_storages()

        # Загрузить и проиндексировать документы
        status = await self._index_documents()
        return status

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Поиск через граф знаний LightRAG (hybrid mode)."""
        if self._rag is None:
            return []

        try:
            # hybrid = local + global поиск по графу
            response = await self._rag.aquery(
                query,
                param=QueryParam(
                    mode="hybrid",
                    top_k=top_k * 3,
                    only_need_context=True,  # Получаем только контекст, без генерации
                ),
            )

            if not response:
                return []

            # LightRAG возвращает контекст как строку — разбиваем на результаты
            return [{
                "title": "Граф знаний (LightRAG)",
                "content": response if isinstance(response, str) else str(response),
                "score": 1.0,
            }]

        except Exception as e:
            print(f"⚠️ LightRAG search error: {e}")
            return []

    async def add_document(self, filename: str, content: str) -> str:
        """Добавить документ в базу знаний и проиндексировать."""
        if self._rag is None:
            return "❌ LightRAG не инициализирован"

        # Сохранить файл на диск
        filepath = self.docs_dir / filename
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

        # Добавить в граф знаний
        try:
            await self._rag.ainsert(content)
            self._indexed_docs.add(filename)

            first_line = content.strip().split("\n")[0][:80]
            self._doc_titles[filename] = first_line if first_line else filename

            return (
                f"✅ Добавлен в граф знаний: {filename}\n"
                f"📊 {len(self._indexed_docs)} документов проиндексировано"
            )
        except Exception as e:
            return f"❌ Ошибка индексации: {e}"

    async def delete_document(self, filename: str) -> str:
        """Удалить документ из файловой системы."""
        filepath = self.docs_dir / filename
        if not filepath.exists():
            return f"❌ Файл {filename} не найден"

        filepath.unlink()
        self._indexed_docs.discard(filename)
        self._doc_titles.pop(filename, None)

        return (
            f"🗑 Удалён: {filename}\n"
            f"⚠️ Для полной очистки графа выполните /rebuild"
        )

    async def rebuild_index(self) -> str:
        """Пересоздать граф знаний с нуля."""
        if self._rag is None:
            return "❌ LightRAG не инициализирован"

        # Очистить рабочую директорию LightRAG (кроме кэша LLM)
        import shutil
        for item in self.working_dir.iterdir():
            if item.name == "kv_store_llm_response_cache.json":
                continue  # Сохраняем кэш LLM для экономии токенов
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        # Переинициализировать
        await self._rag.initialize_storages()

        # Переиндексировать все документы
        status = await self._index_documents()
        return f"✅ Граф знаний пересоздан\n{status}"

    def get_doc_list(self) -> list[str]:
        if not self.docs_dir.exists():
            return []
        return sorted([f.name for f in self.docs_dir.glob("*.txt")])

    def get_stats(self) -> dict:
        return {
            "engine_name": self.ENGINE_NAME,
            "docs": len(self.get_doc_list()),
            "indexed": len(self._indexed_docs),
            "doc_titles": dict(self._doc_titles),
            "graph_dir": str(self.working_dir),
        }

    async def shutdown(self) -> None:
        """Корректное завершение LightRAG."""
        if self._rag is not None:
            try:
                await self._rag.finalize_storages()
            except Exception:
                pass

    # --- Внутренние методы ---

    async def _index_documents(self) -> str:
        """Проиндексировать все .txt файлы из docs_dir."""
        if not self.docs_dir.exists():
            return "⚠️ Папка документов не найдена"

        self._indexed_docs = set()
        self._doc_titles = {}
        errors = []
        total_text = ""

        for filepath in sorted(self.docs_dir.glob("*.txt")):
            try:
                content = filepath.read_text(encoding="utf-8")
                first_line = content.strip().split("\n")[0][:80]
                self._doc_titles[filepath.name] = first_line if first_line else filepath.stem
                total_text += f"\n\n--- {filepath.name} ---\n{content}"
                self._indexed_docs.add(filepath.name)
            except Exception as e:
                errors.append(f"{filepath.name}: {e}")

        if total_text:
            try:
                await self._rag.ainsert(total_text)
            except Exception as e:
                errors.append(f"Индексация: {e}")

        status = (
            f"📊 Движок: LightRAG (граф знаний)\n"
            f"📚 Документов: {len(self._indexed_docs)}\n"
            f"🔗 Граф: {self.working_dir}"
        )
        if errors:
            status += "\n❌ Ошибки: " + "; ".join(errors)

        return status
