"""
Модульная архитектура ИИ-Юрист.

Каждый компонент (движок знаний, LLM, эмбеддинги) реализует
абстрактный интерфейс (Protocol). Замена любого модуля —
это просто написать новый адаптер, не трогая остальной код.

Выбор движка через .env:
    KNOWLEDGE_ENGINE=lightrag   — граф знаний LightRAG (ПГС)
    KNOWLEDGE_ENGINE=semantic   — семантический поиск (эмбеддинги)
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class KnowledgeEngine(Protocol):
    """Интерфейс движка знаний.

    Любой движок (LightRAG, semantic, будущий) должен реализовать
    эти методы. Телеграм-бот работает только через этот интерфейс.
    """

    async def initialize(self) -> str:
        """Загрузить документы и построить индекс. Вернуть статус."""
        ...

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Найти релевантные фрагменты. Каждый элемент: {title, content, score}."""
        ...

    async def add_document(self, filename: str, content: str) -> str:
        """Добавить документ в базу знаний. Вернуть статус."""
        ...

    async def delete_document(self, filename: str) -> str:
        """Удалить документ из базы знаний. Вернуть статус."""
        ...

    async def rebuild_index(self) -> str:
        """Принудительно пересоздать индекс. Вернуть статус."""
        ...

    def get_doc_list(self) -> list[str]:
        """Список файлов в базе знаний."""
        ...

    def get_stats(self) -> dict:
        """Статистика: {docs, chunks, engine_name, ...}."""
        ...

    async def shutdown(self) -> None:
        """Корректное завершение (закрыть соединения и т.д.)."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Интерфейс LLM-провайдера.

    Замена модели/провайдера (OpenRouter → Ollama → OpenAI) —
    это просто новый класс с этим интерфейсом.
    """

    async def complete(
        self,
        query: str,
        context: str,
        system_prompt: str = "",
    ) -> str:
        """Сгенерировать ответ на основе запроса и контекста."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Интерфейс провайдера эмбеддингов."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Получить эмбеддинги для списка текстов."""
        ...

    @property
    def dimension(self) -> int:
        """Размерность эмбеддингов."""
        ...
