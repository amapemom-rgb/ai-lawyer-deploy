"""
Семантический поисковый движок (fallback).

Это обёртка вокруг оригинального SemanticSearchEngine из Фазы 1.5,
приведённая к интерфейсу KnowledgeEngine.

Использует: OpenRouter Embeddings + cosine similarity.
Хранение: JSON-кэш эмбеддингов на диске.
"""

import json
import hashlib
import numpy as np
from pathlib import Path


EMBED_MODEL = "openai/text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
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


def _get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """Получает эмбеддинги через OpenRouter API."""
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    all_embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SemanticEngine:
    """Семантический поисковый движок — реализация KnowledgeEngine."""

    ENGINE_NAME = "semantic"

    def __init__(self, api_key: str, docs_dir: Path, cache_file: Path):
        self.api_key = api_key
        self.docs_dir = docs_dir
        self.cache_file = cache_file
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._cache_hash: str | None = None
        self._doc_titles: dict[str, str] = {}

    async def initialize(self) -> str:
        """Загрузить документы и построить индекс."""
        stats = self._load_documents()
        index_status = self._build_index()
        return (
            f"📊 Движок: Semantic Search\n"
            f"📚 Документов: {stats['loaded']}, чанков: {stats['chunks']}\n"
            f"{index_status}"
        )

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Семантический поиск по cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_embedding = np.array(_get_embeddings([query], self.api_key)[0])
        scores = []
        for i in range(len(self.chunks)):
            score = _cosine_similarity(query_embedding, self.embeddings[i])
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        title_count: dict[str, int] = {}
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

    async def add_document(self, filename: str, content: str) -> str:
        """Добавить документ и пересоздать индекс."""
        filepath = self.docs_dir / filename
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        stats = self._load_documents()
        index_status = self._build_index()
        return f"✅ Добавлен: {filename}\n📊 {stats['loaded']} документов, {stats['chunks']} чанков\n{index_status}"

    async def delete_document(self, filename: str) -> str:
        """Удалить документ и пересоздать индекс."""
        filepath = self.docs_dir / filename
        if not filepath.exists():
            return f"❌ Файл {filename} не найден"
        filepath.unlink()
        stats = self._load_documents()
        index_status = self._build_index()
        return f"🗑 Удалён: {filename}\n📊 {stats['loaded']} документов, {stats['chunks']} чанков\n{index_status}"

    async def rebuild_index(self) -> str:
        """Принудительно пересоздать индекс."""
        if self.cache_file.exists():
            self.cache_file.unlink()
        stats = self._load_documents()
        index_status = self._build_index()
        return f"✅ Индекс пересоздан\n📚 {stats['loaded']} документов, {stats['chunks']} чанков\n{index_status}"

    def get_doc_list(self) -> list[str]:
        if not self.docs_dir.exists():
            return []
        return sorted([f.name for f in self.docs_dir.glob("*.txt")])

    def get_stats(self) -> dict:
        return {
            "engine_name": self.ENGINE_NAME,
            "docs": len(self.get_doc_list()),
            "chunks": len(self.chunks),
            "doc_titles": dict(self._doc_titles),
        }

    async def shutdown(self) -> None:
        """Ничего закрывать не нужно."""
        pass

    # --- Внутренние методы ---

    def _load_documents(self) -> dict:
        self.chunks = []
        self._doc_titles = {}
        stats = {"loaded": 0, "chunks": 0, "errors": []}

        if not self.docs_dir.exists():
            stats["errors"].append(f"Папка {self.docs_dir} не найдена")
            return stats

        all_content = ""
        for filepath in sorted(self.docs_dir.glob("*.txt")):
            try:
                content = filepath.read_text(encoding="utf-8")
                first_line = content.strip().split("\n")[0][:80]
                title = first_line if first_line else filepath.stem
                all_content += content
                doc_chunks = _chunk_text(content, title)
                self.chunks.extend(doc_chunks)
                self._doc_titles[filepath.name] = title
                stats["loaded"] += 1
            except Exception as e:
                stats["errors"].append(f"{filepath.name}: {e}")

        stats["chunks"] = len(self.chunks)
        self._cache_hash = hashlib.md5(all_content.encode()).hexdigest()
        return stats

    def _build_index(self) -> str:
        if not self.chunks:
            return "❌ Нет документов для индексации"

        if self.cache_file.exists():
            try:
                cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
                if cache.get("hash") == self._cache_hash:
                    self.embeddings = np.array(cache["embeddings"])
                    return f"📦 Загружен кэш ({len(self.chunks)} чанков)"
            except Exception:
                pass

        texts = [chunk["content"] for chunk in self.chunks]
        embeddings = _get_embeddings(texts, self.api_key)
        self.embeddings = np.array(embeddings)

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "hash": self._cache_hash,
            "embeddings": [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings],
        }
        self.cache_file.write_text(json.dumps(cache), encoding="utf-8")
        return f"✅ Создано {len(self.chunks)} эмбеддингов, кэш сохранён"
