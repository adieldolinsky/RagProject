import os

# Paths (relative to project root)
DATA_FOLDER = os.environ.get("RAG_DATA_FOLDER", "data")
INSTANCE_FOLDER = "instance"
CHAT_DB_NAME = "chat.db"
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "instance/faiss_index")

# Document chunking
CHUNK_MAX_CHARACTERS = int(os.environ.get("RAG_CHUNK_MAX_CHARACTERS", "1000"))
CHUNK_COMBINE_UNDER_CHARS = int(os.environ.get("RAG_CHUNK_COMBINE_UNDER_CHARS", "200"))
CHUNKING_VERSION = os.environ.get("RAG_CHUNKING_VERSION", "by_title_1000_v1")

# Embedding & retrieval
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("RAG_TOP_K", "10"))
SIMILARITY_THRESHOLD = float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.22"))
MAX_HISTORY_MESSAGES = int(os.environ.get("RAG_MAX_HISTORY_MESSAGES", "20"))
MAX_HISTORY_FOR_LLM = int(os.environ.get("RAG_MAX_HISTORY_FOR_LLM", "10"))

# LLM
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
