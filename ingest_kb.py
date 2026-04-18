import os
import json
import ollama
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple

# Configuration
EMBEDDING_MODEL = "embeddinggemma"
DOCS_DIR = "docs"
INDEX_FILE = "vector_store.faiss"
METADATA_FILE = "vector_store.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def get_pdf_pages(path: str) -> List[Tuple[str, int]]:
    """Extract text from a PDF file, keeping track of page numbers."""
    try:
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append((text, i + 1))
        return pages
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return []

def load_all_documents() -> List[Dict]:
    """Load documents from the PDF folder."""
    documents = []

    # 2. Load from PDFs
    if os.path.exists(DOCS_DIR):
        print(f"Scanning directory for PDFs: {DOCS_DIR}")
        for filename in os.listdir(DOCS_DIR):
            if filename.endswith(".pdf"):
                path = os.path.join(DOCS_DIR, filename)
                print(f"Processing PDF: {filename}")
                pages = get_pdf_pages(path)
                for text, page_num in pages:
                    documents.append({
                        "source": filename,
                        "content": text,
                        "type": "pdf",
                        "page": page_num
                    })
    
    return documents

def ingest():
    """Chunk, embed, and store knowledge base."""
    documents = load_all_documents()
    if not documents:
        print("No documents found to ingest.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks_metadata = []
    texts_to_embed = []

    print("Chunking documents...")
    for doc in documents:
        chunks = splitter.split_text(doc['content'])
        for i, chunk in enumerate(chunks):
            chunks_metadata.append({
                "source": doc['source'],
                "type": doc['type'],
                "text": chunk,
                "chunk_id": i,
                "page": doc['page']
            })
            texts_to_embed.append(chunk)

    print(f"Generating embeddings for {len(texts_to_embed)} chunks...")
    embeddings = []
    batch_size = 5
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i : i + batch_size]
        try:
            response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend(response['embeddings'])
        except Exception as e:
            print(f"Error in embedding batch: {e}")
            embeddings.extend([np.zeros(768).tolist()] * len(batch))

    # Build FAISS index
    embeddings_np = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    # Save
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w') as f:
        json.dump(chunks_metadata, f)

    return len(chunks_metadata)

if __name__ == "__main__":
    count = ingest()
    print(f"Ingestion complete! Indexed {count} chunks.")
