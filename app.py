import os
from dotenv import load_dotenv

load_dotenv()

from app_factory import create_app
from rag.llm_generator import LLMGenerator
from rag.vector_store import VectorStore
from rag_engine import load_documents

print("=== Initializing RAG System ===")
VectorStore.invalidate_cache_if_stale()
docs = load_documents()
v_store = VectorStore()
v_store.build_index(docs, force_rebuild=False)
llm = LLMGenerator()
print("=== System Ready ===")

app = create_app(v_store, llm)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
