import os
from google import genai
from google.genai import types

# ==========================================================
# CONFIGURATION
# ==========================================================
GEMINI_MODEL = "gemini-3-flash-preview"
TOP_K = 3

# ---------- Gemini ----------
class LLMGenerator:
    def __init__(self, model_name=GEMINI_MODEL):
        print(f"[*] Initializing LLM Generator: {model_name}...")
        
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL ERROR: GEMINI_API_KEY is missing from .env")
        
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name


    def answer_query(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return "I cannot answer this based on the provided document. No relevant context found."
        
        context_parts = []
        for index, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"--- Data Chunk {index + 1} ({chunk['type']}) ---\n{chunk['content']}\n")
        full_context = "\n".join(context_parts)

        prompt = f"""You are a helpful and precise RAG assistant analyzing a financial document.

Rules:
1. First and foremost, answer the user's question using ONLY the provided context below.
2. If the context does not contain the answer, say EXACTLY: "I do not have enough information in the documents, but based on general knowledge..." and then provide a brief general answer.
3. Keep the answer simple, clear, and objective.
4. Do not invent financial facts or numbers not present in the context.

Context retrieved from documents:
{full_context}

User's latest question:
{query}

Answer:"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3, 
            ),
        )
        return response.text.strip()