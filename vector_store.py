from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"[*] Loading Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.dimension = self.model.get_embedding_dimension()
        
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.chunks_data = []

    def build_index(self, chunks):
        if not chunks:
            raise ValueError("No data chunks provided to build the index.")
            
        print(f"[*] Embedding {len(chunks)} chunks into vectors. This might take a moment...")
        
        texts_to_embed = [chunk["content"] for chunk in chunks]
        
        embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True)
        
        
        self.index.add(embeddings)
        self.chunks_data = chunks
        
        print(f"[*] Successfully indexed {self.index.ntotal} vectors in FAISS.")

    def search(self, query, top_k=3):
        if self.index.ntotal == 0:
            raise RuntimeError("The FAISS index is empty. Build the index first.")
            
        
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
    
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []

        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks_data):
                chunk = self.chunks_data[idx]
                results.append({
                    "score": float(distances[0][i]), 
                    "source": chunk["source"],
                    "type": chunk["type"],
                    "content": chunk["content"]
                })
                
        return results