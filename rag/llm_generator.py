import os
from typing import Any

from google import genai
from google.genai import types

from config import GEMINI_MODEL, MAX_HISTORY_FOR_LLM


class LLMGenerator:
    def __init__(self, model_name: str = GEMINI_MODEL):
        print(f"[*] Initializing LLM generator: {model_name}...")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL ERROR: GEMINI_API_KEY is missing from .env")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @staticmethod
    def _format_history(chat_history: list[dict[str, Any]] | None) -> str:
        if not chat_history:
            return "(No prior conversation in this session.)"

        lines = []
        for msg in chat_history[-MAX_HISTORY_FOR_LLM:]:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "(No prior conversation in this session.)"

    def answer_query(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None = None,
    ) -> str:
        if not retrieved_chunks:
            return (
                "I could not find relevant passages in the uploaded documents for this question. "
                "Try rephrasing or asking about a topic covered in the financial report."
            )

        context_parts = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "unknown")
            chunk_type = chunk.get("type", "text")
            context_parts.append(
                f"--- Chunk {index} [{chunk_type}] (source: {source}) ---\n{chunk['content']}\n"
            )
        full_context = "\n".join(context_parts)
        history_text = self._format_history(chat_history)

        prompt = f"""You are a helpful and precise RAG assistant analyzing financial documents.

Use the retrieved context below to answer the user's question. If the context only partially answers the question, provide what you know from the context and note any gaps clearly. Only say you do not have enough information if the context is completely irrelevant to the question.

Use the conversation history to resolve follow-up questions (e.g. "What about Europe?" refers to prior topics). Keep answers clear, objective, and concise. Do not invent financial figures that are not supported by the context.

Conversation history (this session):
{history_text}

Retrieved document context:
{full_context}

User's latest question:
{query}

Answer:"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return response.text.strip()
