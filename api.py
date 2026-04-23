from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import json
import os
import shutil
import local_rag
import durable_ingest

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
    Trigger document re-scan and FAISS index update using a durable workflow.
    Returns the workflow ID for progress tracking.
    """
    try:
        # Start the workflow asynchronously
        handle = durable_ingest.dbos.start_workflow(durable_ingest.ingest_workflow)
        return {"status": "success", "workflow_id": handle.workflow_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ingest/status/{workflow_id}")
def get_ingest_status(workflow_id: str):
    """
    Get the status and execution steps of a specific ingestion workflow.
    """
    try:
        status = durable_ingest.dbos.get_workflow_status(workflow_id)
        if not status:
            return {"status": "not_found"}
        
        # If completed successfully, ensure local_rag reloads the index from disk
        if status.status == "SUCCESS":
            print(f"Workflow {workflow_id} successful. Reloading vector database...")
            local_rag.reload_vector_db()

        steps = durable_ingest.dbos.list_workflow_steps(workflow_id)
        
        # Format steps for frontend
        formatted_steps = []
        import json
        for step in steps:
            fn_name = step.get("function_name", "Unknown Step")
            out_val = step.get("output", [])
            
            # Safe output for frontend to extract metadata but not blow up memory
            safe_out = []
            if out_val:
                try:
                    if isinstance(out_val, str):
                        parsed = json.loads(out_val)
                    else:
                        parsed = out_val
                        
                    if fn_name == "process_single_document" and isinstance(parsed, list) and len(parsed) > 0:
                        safe_out = [{"source": parsed[0].get("source", "Unknown")}]
                except:
                    pass

            formatted_steps.append({
                "name": fn_name,
                "status": "COMPLETED" if not step.get("error") else "ERROR",
                "output": safe_out
            })

        # Inject inferred running step if workflow is still progressing
        if status.status == "PENDING":
            expected_order = ['list_document_files', 'process_single_document', 'embed_batch', 'save_vector_store']
            last_completed = formatted_steps[-1]["name"] if formatted_steps else None
            
            # Figure out what's next
            if last_completed in expected_order:
                idx = expected_order.index(last_completed)
                # It could be we are looping with process_single_document or embed_batch,
                # but if we are moving forward, we can just highlight the last completed as potentially still running logic,
                # or add a dummy RUNNING state. The frontend accepts duplicate names in list.
                # Actually, the simplest is to mark the last known step as RUNNING in frontend 
                # if we have it, or add the next step as RUNNING.
                # Since we don't know exactly, we'll append the next logical step as RUNNING.
                if idx + 1 < len(expected_order) and last_completed not in ['process_single_document', 'embed_batch']:
                    next_step = expected_order[idx + 1]
                    formatted_steps.append({"name": next_step, "status": "RUNNING", "output": []})
                else:
                    # If it's a batching step, another instance of the same step is likely running
                    formatted_steps.append({"name": last_completed, "status": "RUNNING", "output": formatted_steps[-1]["output"]})
            elif not formatted_steps:
                formatted_steps.append({"name": "list_document_files", "status": "RUNNING", "output": []})

        return {
            "workflow_id": workflow_id,
            "status": status.status,
            "steps": formatted_steps
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Handle multiple file uploads and save them to the docs/ directory.
    """
    try:
        if not os.path.exists("docs"):
            os.makedirs("docs")
            
        saved_files = []
        for file in files:
            file_path = os.path.join("docs", file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            
        return {"status": "success", "message": f"Successfully uploaded: {', '.join(saved_files)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Launch DBOS for durable workflows
    durable_ingest.dbos.launch()
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
