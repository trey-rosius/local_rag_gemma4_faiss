from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import local_rag

app = FastAPI(title="Local AI RAG & Transcription API")

# Configure CORS
# In production, specify exact origins. For local dev, we are more permissive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RagRequest(BaseModel):
    query: str

from fastapi.responses import FileResponse, StreamingResponse

@app.get("/")
async def root():
    """
    Serve the frontend UI.
    """
    return FileResponse("index.html")

@app.post("/rag")
async def run_rag_endpoint(request: RagRequest):
    """
    Endpoint for streaming RAG responses.
    """
    try:
        return StreamingResponse(
            local_rag.run_rag_stream(request.query),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kb")
async def get_knowledge_base():
    """
    Get the knowledge base structure dynamically from the indexed documents.
    """
    try:
        import json
        import os
        from datetime import datetime
        
        metadata_file = "vector_store.json"
        if not os.path.exists(metadata_file):
            return {"folders": []}
            
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        unique_sources = list(set(m['source'] for m in metadata))
        
        # Simple structural grouping
        folders = {}
        for src in unique_sources:
            # metadata extraction for folder
            root = "General Documents"
            description = "Miscellaneous documents and indexed resources."
            icon = "file-text"
            
            if ' > ' in src:
                parts = src.split(' > ')
                root = parts[0]
                icon = "package"
                description = f"Repository for {root.replace('_', ' ')} related intelligence."
            subfolder_name = parts[1] if ' > ' in src else src
            
            # File metadata
            file_meta = {
                "name": src.split(' > ')[-1] if ' > ' in src else src,
                "type": "pdf" if src.lower().endswith(".pdf") else "file",
                "size": "N/A",
                "modified": "N/A"
            }
            
            try:
                # search in docs/
                for f_name in os.listdir("docs"):
                    if src.endswith(f_name):
                        path = os.path.join("docs", f_name)
                        stats = os.stat(path)
                        file_meta["size"] = f"{stats.st_size / 1024:.1f} KB"
                        file_meta["modified"] = datetime.fromtimestamp(stats.st_mtime).strftime("%b %d, %Y")
                        break
            except:
                pass

            if root not in folders: 
                folders[root] = {
                    "name": root, 
                    "description": description,
                    "icon": icon,
                    "updated": "Just now", 
                    "subfolders": {}
                }
            
            if subfolder_name not in folders[root]["subfolders"]:
                folders[root]["subfolders"][subfolder_name] = {
                    "name": subfolder_name,
                    "files": [file_meta]
                }
            else:
                # avoid duplicates
                if not any(f["name"] == file_meta["name"] for f in folders[root]["subfolders"][subfolder_name]["files"]):
                    folders[root]["subfolders"][subfolder_name]["files"].append(file_meta)
                    
            if file_meta["modified"] != "N/A":
                folders[root]["updated"] = file_meta["modified"]
        
        # Format for frontend
        result = []
        for f_name, f_data in folders.items():
            result.append({
                "name": f_data["name"],
                "description": f_data["description"],
                "icon": f_data["icon"],
                "updated": f_data["updated"],
                "subfolders": list(f_data["subfolders"].values())
            })
            
        return {"folders": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"folders": []}


@app.post("/ingest")
async def ingest_endpoint():
    """
    Trigger document re-scan and FAISS index update.
    """
    try:
        import ingest_kb
        count = ingest_kb.ingest()
        local_rag.reload_vector_db()
        return {"status": "success", "message": f"Successfully indexed {count} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
