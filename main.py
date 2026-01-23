import json
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Generator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

app = FastAPI(title="Ollama Chat")

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SAVED_CHATS_DIR = Path("./saved_chats")
SAVED_CHATS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "general": "qwen2.5:3b",
    "code": "qwen2.5-coder:3b"
}

# Pydantic Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model_key: str  # 'general' or 'code'
    messages: List[Message]

class SaveChatRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    model: str
    messages: List[Message]

# Utilities
def get_model_name(key: str) -> str:
    return MODELS.get(key, MODELS["general"])

# Endpoints

@app.get("/api/models")
async def list_models():
    """Return available models configuration."""
    return {
        "models": [
            {"key": "general", "name": MODELS["general"], "label": "💬 General"},
            {"key": "code", "name": MODELS["code"], "label": "💻 Code"}
        ]
    }

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """Stream chat response from Ollama."""
    model_name = get_model_name(request.model_key)
    
    # Convert Pydantic messages to dicts
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Keep last 20 messages context
    if len(messages) > 20:
        messages = messages[-20:]

    async def generate():
        try:
            # Use synchronous ollama client in async generator (blocking, but simple for local setup)
            # ideally use asyncio.to_thread or async client if available
            stream = ollama.chat(
                model=model_name,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/api/chats")
async def get_chats():
    """List all saved chats."""
    chats = []
    for filepath in SAVED_CHATS_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                chats.append({
                    "id": data.get("id", filepath.stem),
                    "title": data.get("title", "Untitled"),
                    "timestamp": data.get("timestamp", ""),
                    "model": data.get("model", "general"),
                })
        except (json.JSONDecodeError, KeyError):
            continue
    
    # Sort by timestamp desc
    chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return chats

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Load a specific chat."""
    filepath = SAVED_CHATS_DIR / f"{chat_id}.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat not found")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats")
async def save_chat(chat: SaveChatRequest):
    """Save a chat session."""
    chat_id = chat.id if chat.id else str(uuid.uuid4())
    
    # Generate title if missing
    title = chat.title
    if not title and chat.messages:
        # User messages only
        user_msgs = [m for m in chat.messages if m.role == "user"]
        if user_msgs:
            first_msg = user_msgs[0].content
            title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
        else:
            title = "Untitled Chat"
    
    data = {
        "id": chat_id,
        "title": title,
        "model": chat.model,
        "timestamp": datetime.now().isoformat(),
        "messages": [m.dict() for m in chat.messages] # Convert model to dict
    }
    
    filepath = SAVED_CHATS_DIR / f"{chat_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return {"id": chat_id, "title": title}

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat."""
    filepath = SAVED_CHATS_DIR / f"{chat_id}.json"
    if filepath.exists():
        filepath.unlink()
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Chat not found")

# Serve Frontend - index.html
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# Run functionality
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
