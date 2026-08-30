"""
AutoBrain Lite — FastAPI Web Server

Handles WebSocket chat connections, PDF uploads, background ingestion, 
and serves the static web UI.

Usage:
    python backend/main.py
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent import AutoBrainAgent
from backend.query import Retriever
from backend.ingest import extract_text_from_pdf, chunk_text, build_index, save_index

app = FastAPI(title="AutoBrain Lite API")

# Shared state
agent = None
MODELS_DIR = "models"
DATA_DIR = "data"
DEFAULT_MODEL = os.path.join(MODELS_DIR, "Phi-3-mini-4k-instruct-q4.gguf")


# ---------------------------------------------------------------------------
# Background Ingestion Task
# ---------------------------------------------------------------------------

def run_background_ingest(pdf_path: str, output_dir: str):
    """
    Ingests a newly uploaded PDF in the background.
    """
    try:
        print(f"[Server Ingest] Starting ingestion for {pdf_path}")
        # Extract
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            print("[Server Ingest] Error: No text extracted from PDF")
            return
            
        # Chunk
        chunks = chunk_text(pages, chunk_size=500, overlap=50)
        
        # Embed
        index, embeddings = build_index(chunks, model_name="all-MiniLM-L6-v2")
        
        # Save
        save_index(index, chunks, output_dir)
        print(f"[Server Ingest] Ingestion complete. Index saved at {output_dir}")
        
        # Clean up uploaded PDF to save local space
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            
    except Exception as e:
        print(f"[Server Ingest] Error processing index: {str(e)}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

class SelectIndexRequest(BaseModel):
    index_name: str


@app.on_event("startup")
def startup_event():
    """Load model and initialize retriever on startup."""
    global agent
    try:
        # Verify default model exists
        if not os.path.exists(DEFAULT_MODEL):
            print(f"[Startup Warning] Model file not found at {DEFAULT_MODEL}. Downloader will need to run.")
            
        # Initialize agent with the first available database in data/
        os.makedirs(DATA_DIR, exist_ok=True)
        agent = AutoBrainAgent(model_path=DEFAULT_MODEL, index_dir=DATA_DIR)
        print("[Startup] AutoBrain Agent ready!")
    except Exception as e:
        print(f"[Startup Error] Could not initialize agent: {str(e)}")
        print("Note: Agent will attempt to lazy-load once a database/model is ready.")


@app.get("/api/indices")
def list_indices():
    """List all available vehicle manuals that have been ingested."""
    os.makedirs(DATA_DIR, exist_ok=True)
    indices = []
    for item in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, item)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "index.faiss")):
            indices.append(item)
    return {"indices": indices, "active": os.path.basename(agent.index_dir) if agent else None}


@app.post("/api/select-index")
def select_index(req: SelectIndexRequest):
    """Switch active index to a different manual."""
    global agent
    target_dir = os.path.join(DATA_DIR, req.index_name)
    if not os.path.exists(os.path.join(target_dir, "index.faiss")):
        raise HTTPException(status_code=404, detail="Index not found")

    try:
        if agent is None:
            agent = AutoBrainAgent(model_path=DEFAULT_MODEL, index_dir=target_dir)
        else:
            # Reconnect retriever to new directory without reloading LLM (very fast!)
            agent.index_dir = target_dir
            agent.retriever = Retriever(target_dir)
        return {"status": "success", "active": req.index_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load index: {str(e)}")


@app.post("/api/upload")
def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a new vehicle manual PDF and queue background ingestion."""
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"message": "Only PDF files are supported"})

    temp_dir = os.path.join(DATA_DIR, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save file temporarily
    temp_pdf_path = os.path.join(temp_dir, file.filename)
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Output directory name based on file stem
    index_name = Path(file.filename).stem.replace(" ", "_")
    output_dir = os.path.join(DATA_DIR, index_name)

    # Queue background processing
    background_tasks.add_task(run_background_ingest, temp_pdf_path, output_dir)

    return {
        "status": "processing",
        "message": f"Manual queued for ingestion. Index name: {index_name}",
        "index_name": index_name
    }


# ---------------------------------------------------------------------------
# WebSocket Chat Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global agent
    await websocket.accept()
    print("[WebSocket] Client connected")

    try:
        if agent is None:
            await websocket.send_json({
                "error": "Agent not initialized. Please ensure a model and an index exist."
            })

        while True:
            # Receive query text from frontend
            data = await websocket.receive_text()

            if agent is None:
                try:
                    await websocket.send_json({
                        "answer": "System Error: The Local AI model or vector index database is not loaded yet.",
                        "retrieved": [],
                        "follow_ups": []
                    })
                except (WebSocketDisconnect, RuntimeError):
                    break
                continue

            try:
                # Run CPU-heavy inference in a separate thread so WebSocket pings still get handled!
                result = await run_in_threadpool(agent.answer_query, data)

                # Send result back as JSON. If the client disconnected while we were working,
                # gracefully stop instead of crashing the server loop.
                try:
                    await websocket.send_json({
                        "answer": result["answer"],
                        "retrieved": [
                            {"text": c["text"], "page": c["page"]}
                            for c in result["retrieved"]
                        ],
                        "follow_ups": result["follow_ups"]
                    })
                except (WebSocketDisconnect, RuntimeError):
                    break
            except Exception as e:
                try:
                    await websocket.send_json({
                        "answer": f"Error during query execution: {str(e)}",
                        "retrieved": [],
                        "follow_ups": []
                    })
                except (WebSocketDisconnect, RuntimeError):
                    break

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except RuntimeError:
        print("[WebSocket] Connection closed while sending response")
    finally:
        print("[WebSocket] Client disconnected")


# ---------------------------------------------------------------------------
# Serve Web UI
# ---------------------------------------------------------------------------

# Mount frontend files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # Start on all interfaces to allow emulator connections
    uvicorn.run(app, host="0.0.0.0", port=8000)
