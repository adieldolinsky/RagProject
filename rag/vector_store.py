import json
import shutil
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNKING_VERSION,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    SIMILARITY_THRESHOLD,
)


class VectorStore:
    """
    FAISS vector store using cosine similarity (IndexFlatIP + L2-normalized embeddings).
    Supports optional disk persistence to skip re-embedding on restart.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"[*] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_data: list[dict[str, Any]] = []
        self._index_dir = Path(FAISS_INDEX_DIR)

    @classmethod
    def clear_cache(cls) -> None:
        """Remove persisted FAISS index files so the next build starts fresh."""
        index_dir = Path(FAISS_INDEX_DIR)
        if index_dir.exists():
            shutil.rmtree(index_dir)
            print(f"[*] Cleared FAISS index cache at {index_dir}.")

    @classmethod
    def invalidate_cache_if_stale(cls) -> None:
        """Clear cache when chunking strategy/version or index files are outdated."""
        meta_path = Path(FAISS_INDEX_DIR) / "meta.json"
        if not meta_path.exists():
            if Path(FAISS_INDEX_DIR).exists():
                cls.clear_cache()
            return

        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("chunking_version") != CHUNKING_VERSION:
                print("[*] Chunking version changed — rebuilding FAISS index.")
                cls.clear_cache()
        except (json.JSONDecodeError, OSError):
            cls.clear_cache()

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def build_index(self, chunks: list[dict[str, Any]], *, force_rebuild: bool = False) -> None:
        if not chunks:
            raise ValueError("No data chunks provided to build the index.")

        if not force_rebuild and self._load_from_disk(len(chunks)):
            return

        print(f"[*] Embedding {len(chunks)} chunks (cosine / inner-product index)...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self._encode(texts, show_progress=len(texts) > 50)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.chunks_data = chunks

        self._save_to_disk()
        print(f"[*] Indexed {self.index.ntotal} vectors in FAISS.")

    def _index_paths(self) -> tuple[Path, Path, Path]:
        base = self._index_dir
        return base / "index.faiss", base / "chunks.json", base / "meta.json"

    def _save_to_disk(self) -> None:
        index_path, chunks_path, meta_path = self._index_paths()
        self._index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        with chunks_path.open("w", encoding="utf-8") as f:
            json.dump(self.chunks_data, f, ensure_ascii=False)

        meta = {
            "model": EMBEDDING_MODEL,
            "dimension": self.dimension,
            "count": len(self.chunks_data),
            "chunking_version": CHUNKING_VERSION,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _load_from_disk(self, expected_chunk_count: int) -> bool:
        index_path, chunks_path, meta_path = self._index_paths()
        if not (index_path.exists() and chunks_path.exists() and meta_path.exists()):
            return False

        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("chunking_version") != CHUNKING_VERSION:
                print("[*] Cached FAISS index is stale (chunking version changed). Rebuilding...")
                return False

            if meta.get("count") != expected_chunk_count:
                print("[*] Cached FAISS index is stale (chunk count changed). Rebuilding...")
                return False

            self.index = faiss.read_index(str(index_path))
            with chunks_path.open(encoding="utf-8") as f:
                self.chunks_data = json.load(f)

            print(f"[*] Loaded cached FAISS index ({self.index.ntotal} vectors).")
            return True
        except Exception as exc:
            print(f"[*] Could not load cached index ({exc}). Rebuilding...")
            return False

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            raise RuntimeError("The FAISS index is empty. Build the index first.")

        query_vector = self._encode([query])
        search_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks_data):
                continue
            if float(score) < SIMILARITY_THRESHOLD:
                continue

            chunk = self.chunks_data[idx]
            results.append(
                {
                    "score": float(score),
                    "source": chunk["source"],
                    "type": chunk["type"],
                    "content": chunk["content"],
                }
            )

        return results
