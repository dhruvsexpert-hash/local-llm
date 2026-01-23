# =============================================================================
# LOCAL LLM - FULLY SELF-CONTAINED DOCKER IMAGE
# =============================================================================
# This Dockerfile contains EVERYTHING - no other files needed!
# 
# BUILD:  docker build -t local-llm .
# RUN:    docker run -d -p 8000:8000 --name local-llm local-llm
# ACCESS: http://localhost:8000
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=http://localhost:11434

# Install system dependencies + Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    zstd \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn ollama pydantic

# Create saved_chats directory
RUN mkdir -p /app/saved_chats

# =============================================================================
# EMBEDDED: main.py
# =============================================================================
RUN cat > /app/main.py << 'PYTHON_EOF'
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
        "messages": [m.dict() for m in chat.messages]
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
PYTHON_EOF

# =============================================================================
# EMBEDDED: index.html
# =============================================================================
RUN cat > /app/index.html << 'HTML_EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama Chat</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/11.1.1/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        :root {
            --bg-color: #131314;
            --sidebar-bg: #0D0D0E;
            --input-bg: #1E1F22;
            --user-msg-bg: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            --text-color: #E3E3E3;
            --text-muted: #9ca3af;
            --border-color: rgba(255, 255, 255, 0.06);
            --sidebar-width: 280px;
        }
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0f10 0%, #1a1a1d 50%, #0f0f10 100%);
            color: var(--text-color);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }
        #sidebar {
            width: var(--sidebar-width);
            background: linear-gradient(180deg, #0a0a0b 0%, #111113 100%);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            z-index: 2000;
        }
        #sidebar.hidden { transform: translateX(-100%); }
        .sidebar-header {
            padding: 1.5rem 1.5rem 1.5rem 5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .new-chat-btn {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.06) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-color);
            padding: 0.8rem 1rem;
            border-radius: 12px;
            cursor: pointer;
            width: calc(100% - 2rem);
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin: 0 1rem;
        }
        .new-chat-btn:hover {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.12) 100%);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-1px);
        }
        .chat-list { flex: 1; overflow-y: auto; padding: 1rem; }
        .chat-item {
            padding: 0.35rem 0.6rem;
            border-radius: 8px;
            cursor: pointer;
            color: var(--text-muted);
            transition: background 0.2s;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .chat-item:hover { background: rgba(255, 255, 255, 0.05); color: var(--text-color); }
        .chat-item.active { background: rgba(255, 255, 255, 0.08); color: white; }
        .chat-title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 180px; }
        .delete-btn {
            opacity: 0.6;
            background: none;
            border: none;
            color: #ef4444;
            cursor: pointer;
            padding: 6px;
            border-radius: 4px;
            font-size: 1rem;
            transition: all 0.2s;
        }
        .delete-btn:hover { opacity: 1; background: rgba(239, 68, 68, 0.1); }
        .chat-item:hover .delete-btn { opacity: 1; }
        #main {
            flex: 1;
            display: flex;
            flex-direction: column;
            margin-left: var(--sidebar-width);
            transition: margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
            position: relative;
        }
        body.sidebar-closed #main { margin-left: 0; }
        #toggle-btn {
            position: fixed;
            top: 16px;
            left: 16px;
            z-index: 2001;
            background: rgba(30, 30, 34, 0.8);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            width: 44px;
            height: 44px;
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            transition: left 0.3s ease;
        }
        #toggle-btn:hover { color: white; background: rgba(40, 40, 45, 1); }
        header { text-align: center; padding: 2rem 1rem; }
        h1 {
            font-size: 1.8rem;
            margin: 0;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 50px 50px 8rem 50px;
            scroll-behavior: smooth;
            max-width: 100%;
            margin: 0;
            width: 100%;
            box-sizing: border-box;
        }
        .message { margin-bottom: 1.5rem; display: flex; flex-direction: column; }
        .message.user { align-items: flex-end; }
        .message-content {
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            max-width: 80%;
            line-height: 1.6;
            position: relative;
        }
        .message.user .message-content {
            background: var(--user-msg-bg);
            color: white;
            border-radius: 1rem 1rem 0.25rem 1rem;
            box-shadow: 0 2px 12px rgba(37, 99, 235, 0.3) !important;
        }
        .message.assistant .message-content {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: #e5e5e5;
            border-radius: 0.5rem 1rem 1rem 1rem;
        }
        .message-content pre { background: #1a1a1e; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; margin: 0.5rem 0; }
        .message-content code { font-family: 'Consolas', 'Monaco', monospace; font-size: 0.9em; }
        .message-content p { margin: 0.5rem 0; }
        .message-content p:first-child { margin-top: 0; }
        .message-content p:last-child { margin-bottom: 0; }
        #input-area {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1.5rem;
            background: linear-gradient(to top, #0f0f10 80%, transparent);
            display: flex;
            justify-content: center;
        }
        .input-wrapper {
            max-width: 850px;
            width: 100%;
            background: var(--input-bg);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 28px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            transition: border-color 0.2s;
            position: relative;
        }
        .input-wrapper:focus-within {
            border-color: rgba(59, 130, 246, 0.5);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 20px rgba(59, 130, 246, 0.15);
        }
        .model-select {
            background: #3a3d45;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 8px 12px;
            font-size: 0.85rem;
            margin-right: 12px;
            outline: none;
            cursor: pointer;
        }
        textarea {
            flex: 1;
            background: transparent;
            border: none;
            color: #f5f5f5;
            font-size: 1rem;
            resize: none;
            height: 24px;
            max-height: 200px;
            padding: 8px 0;
            font-family: inherit;
            outline: none;
        }
        #send-btn {
            background: var(--user-msg-bg);
            border: none;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s;
            margin-left: 8px;
        }
        #send-btn:hover { transform: scale(1.05); }
        #send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .typing-indicator { display: inline-block; margin-top: 4px; }
        .typing-indicator span {
            display: inline-block;
            width: 4px;
            height: 4px;
            background-color: #aaa;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out both;
            margin: 0 2px;
        }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="sidebar-header">
            <h2 style="margin: 0; font-size: 1.1rem; color: #fff;">Chats</h2>
        </div>
        <button class="new-chat-btn" onclick="startNewChat()">✨ New Chat</button>
        <div class="chat-list" id="chat-list"></div>
    </div>
    <div id="main">
        <button id="toggle-btn" onclick="toggleSidebar()">☰</button>
        <div id="chat-container">
            <div id="empty-state" class="empty-chat" style="display: none; flex-direction: column; align-items: center; justify-content: center; gap: 1rem; margin-top: 10vh;">
                <div style="font-size: 4rem; opacity: 0.5;">💬</div>
                <h2 style="color: #e5e5e5; margin: 0;">How can I help you today?</h2>
                <p style="color: #6b7280; margin: 0;">Start a conversation below.</p>
            </div>
        </div>
        <div id="input-area">
            <div class="input-wrapper">
                <select class="model-select" id="model-select">
                    <option value="general">Gen</option>
                    <option value="code">Code</option>
                </select>
                <textarea id="user-input" placeholder="Message Ollama..." rows="1" oninput="autoResize(this)" onkeydown="handleEnter(event)"></textarea>
                <button id="send-btn" onclick="sendMessage()">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>
        </div>
    </div>
    <script>
        let currentChatId = null;
        let messages = [];
        let sidebarVisible = true;
        marked.setOptions({
            highlight: function (code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language }).value;
            },
            langPrefix: 'hljs language-'
        });
        document.addEventListener('DOMContentLoaded', () => {
            loadChats();
            startNewChat();
            document.getElementById('user-input').focus();
        });
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebarVisible = !sidebarVisible;
            document.body.classList.toggle('sidebar-closed');
            if (sidebarVisible) {
                sidebar.classList.remove('hidden');
            } else {
                sidebar.classList.add('hidden');
            }
        }
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }
        function handleEnter(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }
        async function loadChats() {
            try {
                const res = await fetch('/api/chats');
                const chats = await res.json();
                renderChatList(chats);
            } catch (err) {
                console.error('Failed to load chats:', err);
            }
        }
        function renderChatList(chats) {
            const container = document.getElementById('chat-list');
            container.innerHTML = '';
            if (chats.length === 0) {
                container.innerHTML = '<div style="padding:2rem 1rem; color:var(--text-muted); text-align:center; opacity:0.6;"><div style="font-size:1.5rem; margin-bottom:0.5rem;">📭</div><div style="font-size:0.9rem;">No saved chats</div></div>';
                return;
            }
            chats.forEach(chat => {
                const div = document.createElement('div');
                div.className = `chat-item ${chat.id === currentChatId ? 'active' : ''}`;
                div.onclick = () => loadChat(chat.id);
                div.innerHTML = `<div class="chat-title">${chat.title || 'Untitled'}</div><button type="button" class="delete-btn" onclick="deleteChat(event, '${chat.id}')">🗑</button>`;
                container.appendChild(div);
            });
        }
        async function loadChat(id) {
            try {
                const res = await fetch(`/api/chats/${id}`);
                const data = await res.json();
                currentChatId = data.id;
                messages = data.messages || [];
                document.getElementById('model-select').value = data.model;
                renderMessages();
                loadChats();
            } catch (err) {
                console.error('Failed to load chat:', err);
            }
        }
        async function startNewChat() {
            currentChatId = null;
            messages = [];
            renderMessages();
            loadChats();
        }
        async function deleteChat(e, id) {
            e.stopPropagation();
            e.preventDefault();
            if (!confirm('Delete chat?')) return;
            try {
                await fetch(`/api/chats/${id}`, { method: 'DELETE' });
                if (currentChatId === id) startNewChat();
                loadChats();
            } catch (err) {
                console.error('Failed to delete chat:', err);
            }
        }
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const text = input.value.trim();
            const model = document.getElementById('model-select').value;
            if (!text) return;
            messages.push({ role: 'user', content: text });
            input.value = '';
            input.style.height = 'auto';
            renderMessages();
            const assistantMsgIndex = messages.push({ role: 'assistant', content: '' }) - 1;
            renderMessages();
            input.disabled = true;
            document.getElementById('send-btn').disabled = true;
            try {
                const payload = { model: model, messages: messages.slice(0, -1) };
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_key: model, messages: payload.messages })
                });
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantContent = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    assistantContent += chunk;
                    messages[assistantMsgIndex].content = assistantContent;
                    updateLastMessage(assistantContent);
                }
                await saveCurrentChat(model);
            } catch (err) {
                messages[assistantMsgIndex].content = '⚠️ Error: ' + err.message;
                renderMessages();
            } finally {
                input.disabled = false;
                document.getElementById('send-btn').disabled = false;
                input.focus();
            }
        }
        async function saveCurrentChat(model) {
            const payload = { id: currentChatId, model: model, messages: messages, title: null };
            try {
                const res = await fetch('/api/chats', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                currentChatId = data.id;
                loadChats();
            } catch (err) {
                console.error('Failed to save chat', err);
            }
        }
        function renderMessages() {
            const container = document.getElementById('chat-container');
            const emptyState = document.getElementById('empty-state');
            container.innerHTML = '';
            container.appendChild(emptyState);
            if (messages.length === 0) {
                emptyState.style.display = 'flex';
                return;
            } else {
                emptyState.style.display = 'none';
            }
            messages.forEach((msg, index) => {
                const div = document.createElement('div');
                div.className = `message ${msg.role}`;
                div.id = `msg-${index}`;
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                if (msg.role === 'assistant' && !msg.content) {
                    contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
                } else {
                    contentDiv.innerHTML = marked.parse(msg.content);
                }
                div.appendChild(contentDiv);
                container.appendChild(div);
            });
            scrollToBottom();
        }
        function updateLastMessage(content) {
            const index = messages.length - 1;
            const msgDiv = document.getElementById(`msg-${index}`);
            if (msgDiv) {
                const contentDiv = msgDiv.querySelector('.message-content');
                contentDiv.innerHTML = marked.parse(content);
                scrollToBottom();
            }
        }
        function scrollToBottom() {
            const container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        }
    </script>
</body>
</html>
HTML_EOF

# =============================================================================
# Download Ollama models during build
# =============================================================================
RUN ollama serve & \
    sleep 5 && \
    echo "Downloading qwen2.5:3b..." && \
    ollama pull qwen2.5:3b && \
    echo "Downloading qwen2.5-coder:3b..." && \
    ollama pull qwen2.5-coder:3b && \
    pkill ollama || true

# =============================================================================
# Create startup script
# =============================================================================
RUN echo '#!/bin/bash\nollama serve &\nsleep 3\nexec uvicorn main:app --host 0.0.0.0 --port 8000' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["/app/start.sh"]
