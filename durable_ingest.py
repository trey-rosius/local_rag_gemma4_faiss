import os
import json
import ollama
import numpy as np
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple
from dbos import DBOS

# Configuration
EMBEDDING_MODEL = "embeddinggemma"
DOCS_DIR = "docs"
INDEX_FILE = "vector_store.faiss"
METADATA_FILE = "vector_store.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

with open("dbos-config.yaml", "r") as f:
    import yaml
    config = yaml.safe_load(f)
dbos = DBOS(config=config)

# We only launch DBOS here if this script is run directly.
# If imported by api.py, api.py will be responsible for launching or sharing the instance.
if __name__ == "__main__":
    dbos.launch()

def get_pdf_pages(path: str) -> List[Tuple[str, int]]:
    """Extract text from a PDF file, keeping track of page numbers."""
    try:
        import re
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Clean up whitespace: replace multiple spaces/newlines with single ones
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    pages.append((text, i + 1))
        return pages
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return []

@DBOS.step()
def list_document_files() -> List[str]:
    """List all new PDF files in the documents directory that are not yet indexed."""
    files = []
    
    # Check existing metadata to avoid re-embedding
    indexed_files = set()
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                meta = json.load(f)
                for item in meta:
                    if "source" in item:
                        indexed_files.add(item["source"])
        except Exception as e:
            print(f"Metadata read error: {e}")
            
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            if filename.endswith(".pdf") and filename not in indexed_files:
                files.append(filename)
                
    return sorted(files)

@DBOS.step()
def process_single_document(filename: str) -> List[Dict]:
    """Process a single document file and extract its content."""
    path = os.path.join(DOCS_DIR, filename)
    print(f"Processing document: {filename}")
    pages = get_pdf_pages(path)
    docs = []
    for text, page_num in pages:
        docs.append({
            "source": filename,
            "content": text,
            "type": "pdf",
            "page": page_num
        })
    return docs

@DBOS.step()
def embed_batch(batch: List[str]) -> List[List[float]]:
    """Embed a batch of text chunks using Ollama."""
    try:
        # Note: Ollama client will use OLLAMA_HOST from environment
        response = ollama.embed(model=EMBEDDING_MODEL, input=batch)
        return response['embeddings']
    except Exception as e:
        print(f"Error in embedding batch: {e}")
        # In case of failure, return zero embeddings to keep the pipeline moving
        # (DBOS will automatically retry this step if it raises an exception, 
        # so we only catch here if we want to proceed with degraded state)
        raise e

@DBOS.step()
def save_vector_store(embeddings: List[List[float]], chunks_metadata: List[Dict]):
    """Save the FAISS index and metadata to disk."""
    if not embeddings:
        return
        
    embeddings_np = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_np)
    
    existing_meta = []
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        # Append to existing
        index = faiss.read_index(INDEX_FILE)
        index.add(embeddings_np)
        with open(METADATA_FILE, 'r') as f:
            existing_meta = json.load(f)
    else:
        # Create new
        index = faiss.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)

    combined_metadata = existing_meta + chunks_metadata

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w') as f:
        json.dump(combined_metadata, f)

@DBOS.workflow()
def ingest_workflow():
    """Durable workflow for chunking, embedding, and storing knowledge base."""
    # Step 1: List all files
    filenames = list_document_files()
    if not filenames:
        print("No documents found to ingest.")
        return 0

    # Step 2: Process each file as a separate durable step
    all_documents = []
    for filename in filenames:
        docs = process_single_document(filename)
        all_documents.extend(docs)

    if not all_documents:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks_metadata = []
    texts_to_embed = []

    print("Chunking documents...")
    for doc in all_documents:
        chunks = splitter.split_text(doc['content'])
        for i, chunk in enumerate(chunks):
            # Skip degenerate chunks
            if len(chunk.strip()) < 100:
                continue

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
        # Each batch embedding is a durable step
        batch_embeddings = embed_batch(batch)
        embeddings.extend(batch_embeddings)

    # Save vector store as a durable step
    save_vector_store(embeddings, chunks_metadata)

    return len(chunks_metadata)

if __name__ == "__main__":
    # Launch the workflow
    count = ingest_workflow()
    print(f"Ingestion complete! Indexed {count} chunks.")
