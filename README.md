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

Here is how the durable pipeline (`durable_ingest.py`) works under the hood:

1. **`ingest_workflow` (@DBOS.workflow)**: The main orchestrator. It manages the steps and automatically saves the state of the execution into PostgreSQL.
2. **`list_document_files` (@DBOS.step)**: Automatically calculates the delta of new `docs/` vs already ingested files.
3. **`process_single_document` (@DBOS.step)**: Extracts text from PDFs and chunks it safely.
4. **`embed_batch` (@DBOS.step)**: Interacts with the local Ollama instance to generate tensor embeddings. If this fails, DBOS automatically retries just this batch without throwing away the rest of the workflow.
5. **`save_vector_store` (@DBOS.step)**: Incremental checkpointing appended to the FAISS index.

Because every `@DBOS.step()` return value is durably saved in Postgres, if your computer shuts down midway through processing your library, restarting the server will cause the workflow to instantly jump back to the exact chunk batch it left off on!

---

## 🚀 Step-by-Step Setup Guide

Follow these steps to deploy the server locally on your machine.

### 1. Prerequisites
You must have the following installed on your machine:
- **Python 3.10+**
- **Docker** & **Docker Compose**
- **Ollama** (Running locally)
  - *Note*: Ensure you pull the correct embeddings model: `ollama pull embeddinggemma`

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
