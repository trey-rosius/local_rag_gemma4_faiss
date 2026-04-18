import ollama
import numpy as np
import json
import os
import faiss
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands_tools.calculator import calculator
from strands_tools.current_time import current_time

# Configuration
LLM_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "embeddinggemma"
INDEX_FILE = "vector_store.faiss"
METADATA_FILE = "vector_store.json"

class VectorDB:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.load()

    def load(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            print(f"Reloading Vector DB from {INDEX_FILE}...")
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)

    def reload(self):
        self.load()

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None: return []
        try:
            response = ollama.embed(model=EMBEDDING_MODEL, input=query)
            query_vector = np.array(response['embeddings'][0]).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)
            distances, indices = self.index.search(query_vector, top_k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1: continue
                if distances[0][i] < 0.3: continue 
                results.append(self.metadata[idx])
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

VECTOR_DB = VectorDB()

@tool
def search_knowledge_base(query: str) -> str:
    """Search knowledge base and documents."""
    results = VECTOR_DB.search(query, top_k=5)
    if not results: return "No relevant information found."
    formatted_results = []
    for res in results:
        page_info = f" (Page {res['page']})" if res.get('page') else ""
        formatted_results.append(f"Source: {res['source']}{page_info}\nContent: {res['text']}")
        search_knowledge_base.last_doc = res
    return "\n---\n".join(formatted_results)

ollama_model = OllamaModel(host="http://localhost:11434", model_id=LLM_MODEL)

agent = Agent(
    model=ollama_model,
    tools=[search_knowledge_base, calculator, current_time],
    system_prompt="""You are Archivist AI, a precise technical librarian and research assistant. 
Always start by calling 'search_knowledge_base' to gather facts.
Refer to yourself as "Archivist AI" in your responses.

VERY IMPORTANT: When providing code examples, ALWAYS use triple backticks with the language identifier 
(e.g., ```python) and put each code block on its own line. Never write code as plain text."""
)

async def run_rag_stream(query: str) -> AsyncGenerator[str, None]:
    """Run agentic RAG pipeline with streaming chunks."""
    search_knowledge_base.last_doc = {}
    
    try:
        async for event in agent.stream_async(query):
            # Nested event format check
            ev = event.get('event', {})
            
            # Tool call detection
            # Format: {'event': {'contentBlockStart': {'start': {'toolUse': {'name': 'search_knowledge_base'}}}}}
            c_start = ev.get('contentBlockStart', {}).get('start', {})
            tool_name = c_start.get('toolUse', {}).get('name')
            if tool_name == 'search_knowledge_base':
                last_doc = getattr(search_knowledge_base, 'last_doc', {})
                if last_doc:
                    yield f"Metadata: {json.dumps(last_doc)}\n"

            # Text chunk detection: Only using the nested contentBlockDelta to avoid duplication
            # with top-level 'data' keys yielded by the Strands SDK events.
            delta_text = ev.get('contentBlockDelta', {}).get('delta', {}).get('text')
            if delta_text:
                yield delta_text

    except Exception as e:
        print(f"Stream Error: {e}")
        yield f"Error: {str(e)}"

def reload_vector_db():
    VECTOR_DB.reload()
