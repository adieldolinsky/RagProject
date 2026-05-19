import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

# import my id_tocken from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("CRITICAL ERROR: HF_TOKEN is missing. Please check your .env file.")

os.environ["HF_TOKEN"] = hf_token
# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FOLDER = "data"
# ==========================================================
# DOCUMENT LOADING
# ==========================================================
#the function extricating the data from pdf file
def load_documents(folder=DATA_FOLDER):
    #Load .pdf files from the data folder and extract logical chunks.
    if not os.path.exists(folder):
        raise FileNotFoundError( f"Folder '{folder}' does not exist. Create it and put .pdf files inside.")

    chunks = []

    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith(".pdf"):
            continue

        file_path = os.path.join(folder, file_name)
        try:
            #tocknizatia for data
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                languages=["eng"]
            )

            for element in elements:
                chunk_data = {
                    "source": file_name,
                    "type": "",
                    "content": ""
                }

                if element.category == 'Title':
                    chunk_data["type"] = "header"
                    chunk_data["content"] = element.text
                    chunks.append(chunk_data)

                elif element.category in ['NarrativeText', 'ListItem', 'UncategorizedText']:
                    chunk_data["type"] = "text"
                    chunk_data["content"] = element.text
                    chunks.append(chunk_data)

                elif element.category == 'Table':
                    chunk_data["type"] = "table"
                    chunk_data["content"] = element.metadata.text_as_html if hasattr(element.metadata,
                                                                                     'text_as_html') and element.metadata.text_as_html else element.text
                    chunks.append(chunk_data)

                else:
                    print(f"DEBUG - Ignored element category: {element.category} | Preview: {element.text[:30]}...")
        #if the procces failed
        except Exception as e:
            raise RuntimeError(f"Failed to process {file_name}. Original error: {str(e)}")

    return chunks