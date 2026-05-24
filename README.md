# RAG Project: Financial Document Assistant

## Project Overview

This is a RAG (Retrieval-Augmented Generation) web application I built with Flask to answer questions based on PDF documents. It loads local PDFs, chunks the text, embeds it into a FAISS vector database, and uses the retrieved passages as context for the Google Gemini model.

## System Architecture

1. **Offline indexing:** At startup, my app parses PDFs in the `data/` folder using **Unstructured**. I embedded the extracted text chunks using **sentence-transformers** and stored in a **FAISS** index, which is cached on disk for fast restarts.
2. **Online chat:** A **Flask** backend handles user queries. It embeds the query, runs a FAISS similarity search, and passes the most relevant chunks to **Gemini** to generate a grounded answer. Chat history is stored locally in SQLite.

```text
PDFs → Unstructured → Chunks → Sentence-Transformers → FAISS (Disk Cache)
                                                           ↓
User → Flask API → FAISS Search → Top Chunks → Gemini → Answer & Sources
```

---

## Data Source

### Location & ingestion

I configured the system to process all `.pdf` files in the `data/` directory .To validate the system i used a real Amdocs annual financial report (`amdocs_annoul_report_example.pdf`)which contains extensive tables, competitive analysis, and geographical breakdowns.

### Parsing strategy

I used `unstructured.partition.pdf.partition_pdf` configured with:

- `strategy="hi_res"` — deep learning layout detection and OCR
- `infer_table_structure=True` — maps text, structural sections, and embedded tables

### Data representation

I mapped the parsed elements into a standardized dictionary:

```python
{"source": "filename.pdf", "type": "text|header|table", "content": "..."}
```

- **Titles** → `header`
- **Tables** → `table` (preserved as full text_as_html to keep rows and columns intact).
- **Everything else** → `text`

---

## Chunking Strategy

Rather than fixed-size character splitting (which can cut paragraphs mid-sentence), I implemented semantic chunking via `unstructured.chunking.title.chunk_by_title`.

### Pipeline logic

1. Raw structural elements from the parser are analyzed.
2. Elements are grouped under their closest heading or section title.
3. Chunks may span multiple pages (`multipage_sections=True`).

### Granularity control

I Configured these parameters in `config.py`:

| Variable | Default | Effect |
|----------|---------|--------|
| `RAG_CHUNK_MAX_CHARACTERS` | `1000` | Hard maximum size per chunk |
| `RAG_CHUNK_COMBINE_UNDER_CHARS` | `200` | Merge small snippets with neighbors |
| `RAG_CHUNKING_VERSION` | `by_title_1000_v1` | Cache invalidation tag |

---

## Embedding Model

### Model selection

I selected the **`all-MiniLM-L6-v2`** model from the `sentence-transformers` library.

### Mathematical alignment

The text is mapped to a **384-dimensional** dense vector space. By setting `normalize_embeddings=True` in `VectorStore._encode()`, I ensured the inner product equals **cosine similarity**.

### Performance

It has a small ~80MB footprint—runs on CPU without a GPU.

```python
embeddings = self.model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=64,
    convert_to_numpy=True,
)
```

---

## FAISS Indexing Implementation

### Index type

I used `faiss.IndexFlatIP(384)` (`faiss-cpu`) — an exact flat inner-product search, which is highly suitable for a single-document or small corpus.

### Disk persistence & caching

To optimize startup times, my `VectorStore` writes a three-file cache under `instance/faiss_index/`:

| File | Purpose |
|------|---------|
| `index.faiss` | Binary vector index |
| `chunks.json` | Chunk dicts; position `i` maps to FAISS vector `i` |
| `meta.json` | Model name, dimension, chunk count, `chunking_version` |

### Cache invalidation

On startup, `invalidate_cache_if_stale()` checks `meta.json`. If I change the chunking parameters or version, the cache is automatically purged and documents are re-indexed.

### Search & thresholding

1. The user's query is encoded with the same model.
2. The index fetches the top **10** results (`RAG_TOP_K=10`).
3. Results below **0.22** (`RAG_SIMILARITY_THRESHOLD`) are discarded.

---

## Brief Reflection

### What worked well

- **Semantic section boundaries:** `chunk_by_title` kept financial metrics coupled to their headers, improving retrieval for seasonal/annual statistics.
- **HTML table preservation:** Tables stored as HTML in chunks helped Gemini read columns and rows with fewer row-mixing errors.
- **Index caching:** Saving the FAISS index to disk saved me a lot of startup time during development.

### Future improvements

- **Dynamic frontend document uploads:** Currently, PDFs are loaded statically from disk. I'd like to add a secure upload button in the web UI so users can expand the knowledge base on the fly.
- **Cloud-offloaded extraction pipelines:** Local layout detection and OCR are CPU-heavy and slow down the initial load. Moving the parsing step to a cloud function would cut local memory use and speed up boot times significantly.

---

## Run Instructions

### Prerequisites

- Python 3.14+
- Document rendering libraries: `poppler-utils`, `tesseract-ocr`, `libmagic-dev` (included in the Dockerfile)

### Local installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
FLASK_SECRET_KEY=your_custom_flask_secret_string
HF_TOKEN=your_huggingface_access_token
GEMINI_API_KEY=your_google_gemini_api_key
```

Place PDF files in `data/`, then start the server:

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Docker deployment

```bash
# Build the container image
docker build -t financial-rag-assistant .

# Run with env file and data volume mounted
docker run -it -p 5000:5000 --env-file .env  financial-rag-assistant
```
