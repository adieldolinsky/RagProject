import os
from flask import Flask, request, render_template
from dotenv import load_dotenv

from rag_engine import load_documents
from vector_store import VectorStore
from llm_generator import LLMGenerator

load_dotenv()

print("=== Initializing RAG System ===")
docs = load_documents()
v_store = VectorStore()
v_store.build_index(docs)
llm = LLMGenerator()
print("=== System Ready ===")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    answer = None
    chunks = []

    if request.method == "POST": 
        query = request.form.get("query", "")
        
        if query:
           
            chunks = v_store.search(query, top_k=3)
            
            answer = llm.answer_query(query, chunks)

    
    return render_template("index.html", query=query, answer=answer, chunks=chunks)

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=True)