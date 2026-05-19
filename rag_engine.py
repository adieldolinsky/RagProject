import os
from dotenv import load_dotenv
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from config import CHUNK_COMBINE_UNDER_CHARS, CHUNK_MAX_CHARACTERS, DATA_FOLDER

# import my id_tocken from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("CRITICAL ERROR: HF_TOKEN is missing. Please check your .env file.")

os.environ["HF_TOKEN"] = hf_token


def _element_to_chunk(element, file_name: str) -> dict | None:
    """Convert an unstructured element into a retrieval chunk dict."""
    text = (element.text or "").strip()
    if not text:
        return None

    category = getattr(element, "category", None)
    chunk_data = {"source": file_name, "type": "text", "content": text}

    if category == "Title":
        chunk_data["type"] = "header"
    elif category == "Table":
        chunk_data["type"] = "table"
        html = getattr(getattr(element, "metadata", None), "text_as_html", None)
        chunk_data["content"] = html if html else text
    else:
        chunk_data["type"] = "text"

    return chunk_data


def load_documents(folder: str = DATA_FOLDER) -> list[dict]:
    """Load PDF files and extract contextual chunks grouped by section."""
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist. Create it and put .pdf files inside."
        )

    chunks: list[dict] = []

    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith(".pdf"):
            continue

        file_path = os.path.join(folder, file_name)
        try:
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng"],
            )

            section_chunks = chunk_by_title(
                elements,
                max_characters=CHUNK_MAX_CHARACTERS,
                combine_text_under_n_chars=CHUNK_COMBINE_UNDER_CHARS,
                multipage_sections=True,
            )

            for element in section_chunks:
                chunk = _element_to_chunk(element, file_name)
                if chunk:
                    chunks.append(chunk)

            print(
                f"[*] {file_name}: {len(section_chunks)} contextual chunks "
                f"(max {CHUNK_MAX_CHARACTERS} chars, by_title)"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to process {file_name}. Original error: {str(e)}"
            ) from e

    return chunks
