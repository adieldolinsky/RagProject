from typing import Any

from config import MAX_HISTORY_FOR_LLM, TOP_K


class ChatService:
    """Orchestrates retrieval-augmented chat with conversational memory."""

    def __init__(self, vector_store, llm_generator, memory):
        self.vector_store = vector_store
        self.llm = llm_generator
        self.memory = memory

    @staticmethod
    def _build_retrieval_query(query: str, history: list[dict[str, Any]]) -> str:
        """Fuse recent user turns with the latest question for better retrieval."""
        if not history:
            return query

        recent_user = [
            msg["content"]
            for msg in history[-6:]
            if msg.get("role") == "user"
        ]
        if not recent_user:
            return query

        context = " ".join(recent_user[-2:])
        combined = f"{context} {query}".strip()
        return combined[:800]

    def process_message(self, session_id: str, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        history = self.memory.get_history(session_id)
        retrieval_query = self._build_retrieval_query(query, history)
        chunks = self.vector_store.search(retrieval_query, top_k=TOP_K)

        answer = self.llm.answer_query(
            query=query,
            retrieved_chunks=chunks,
            chat_history=history[-MAX_HISTORY_FOR_LLM:],
        )

        self.memory.add_message(session_id, "user", query)
        assistant_message = self.memory.add_message(
            session_id,
            "assistant",
            answer,
            chunks=chunks,
        )

        return {
            "answer": answer,
            "chunks": chunks,
            "message": assistant_message,
        }
