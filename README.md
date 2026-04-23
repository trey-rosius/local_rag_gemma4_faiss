# Archivist - Durable Local RAG Pipeline 📚

This project is a full-stack, responsive Local RAG (Retrieval-Augmented Generation) application. It allows you to build a local Intelligence Chatbot that queries your own PDF documents, keeping your data entirely local and private.

It leverages:

- **FastAPI** for the backend architecture.
- **Ollama** for running open-source Local LLMs for both embedding generation and conversational extraction.
- **FAISS** for fast, local vector database indexing.
- **DBOS (Database-Oriented Operating System)** with **PostgreSQL** to make the heavy extraction and embedding pipelines fully durable and crash-resistant.

---

## 🧠 Why DBOS? (The Durable Workflow Philosophy)

When processing large quantities of PDF documents, pulling out chunks of text, and asking an LLM to generate dense neural embeddings, processes can easily fail due to memory limits, LLM timeouts, or system crashes. 

Instead of dealing with corrupt states or having to manually re-ingest thousands of pages from scratch if a failure occurs, we wrapped the RAG ingestion pipeline in **DBOS Workflows**.

### 🛠️ Durable Workflow Steps

The pipeline in `durable_ingest.py` is broken down into atomic, durable steps:

1.  **`list_document_files`**: Scans the `docs/` directory and identifies new PDF files by comparing them against the existing metadata.
2.  **`process_single_document`**: Extracts text from each PDF page-by-page, preserving metadata like page numbers.
3.  **Text Chunking**: Uses `RecursiveCharacterTextSplitter` to break down large documents into manageable 1000-character segments with overlap.
4.  **`embed_batch`**: Sends chunks in batches (default size: 5) to the local **Ollama** instance. If a batch fails, DBOS ensures that only that specific batch is retried, preserving progress.
5.  **`save_vector_store`**: Durably saves the new embeddings and metadata to disk, updating the local FAISS index.

Because every `@DBOS.step()` return value is durably saved in Postgres, if your computer shuts down midway through processing your library, restarting the server will cause the workflow to instantly jump back to the exact chunk batch it left off on!

---

## 💾 Storage Architecture: RAM vs. Disk

The application uses a hybrid storage approach for the FAISS vector database to ensure both speed and persistence:

*   **Disk Persistence**:
    *   **`vector_store.faiss`**: Stores the numerical vector representations.
    *   **`vector_store.json`**: Stores the raw text chunks and metadata (source file, page numbers).
    *   *Benefit*: Your data persists across application restarts without needing to re-embed documents.

*   **RAM Performance**:
    *   Upon startup, the `VectorDB` class loads both the FAISS index and the JSON metadata into memory.
    *   *Benefit*: Semantic search queries are executed in-memory, providing sub-millisecond retrieval times during chat sessions.

---

## 🚀 Step-by-Step Setup Guide

Follow these steps to deploy the server locally on your machine.

### 1. Prerequisites

You must have the following installed on your machine:

- **Python 3.10+**
- **Docker** & **Docker Compose**
- **Ollama** (Running locally)
  - _Note_: Ensure you pull the correct embeddings model: `ollama pull embeddinggemma`

### 2. Launch PostgreSQL for DBOS

DBOS requires a Postgres instance to store workflow states. We have a `docker-compose.yaml` ready for you.

```bash
# Start the database in the background
docker-compose up -d
```

### 3. Initialize Python Virtual Environment

Create a clean environment for your dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Initialize DBOS Tables

Run the DBOS migration command to construct the required internal schemas inside your PostgreSQL instance.

```bash
dbos migrate
```

### 5. Launch the Archivist Backend

Start the FastAPI server. DBOS will automatically launch within the application to listen for workflow triggers.

```bash
python api.py
```

### 6. Open the App in your Browser

Navigate to: [http://localhost:8000](http://localhost:8000)

- Head to the **Sync Library** tab.
- Click **Upload Documents** to drop in your PDFs.
- Watch the durable pipeline step progress dynamically in the UI!

## Plan of action

- how to run gemma4 on your computer(Introduction to Ollama)
- Introduction to FAISS
- introduction to DBOS and durable workflows
- Why we need durable workflows for local RAG
- Why DBOS specifically? What advantages does it bring to the table?
- Installing Postgres locally
- Building the local RAG pipeline with DBOS
- Introduction to AWS Strands(Why do i need strands)
- creating the search knowledge base tool and attaching to strands
- test the pipeline
- test querying the knowledge base through strands
- conclusion
