import json
import uuid
import os
import calendar
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Generator, Any, Dict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

# Load environment variables
load_dotenv()
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "")

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

CONFIG_DIR = Path("./config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = CONFIG_DIR / "selected_models.json"

def load_models_config():
    """Load selected models from the setup wizard config file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            models = {}
            # General models: use first as default "general" key, rest as named keys
            general = config.get("general", [])
            for i, m in enumerate(general):
                key = "general" if i == 0 else f"general_{i}"
                models[key] = {"tag": m["tag"], "name": m["name"], "category": "general"}
            # Code models
            code = config.get("code", [])
            for i, m in enumerate(code):
                key = "code" if i == 0 else f"code_{i}"
                models[key] = {"tag": m["tag"], "name": m["name"], "category": "code"}
            if models:
                return models
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    # Fallback defaults
    return {
        "general": {"tag": "qwen3:1.7b", "name": "Qwen 3", "category": "general"},
        "code": {"tag": "qwen2.5-coder:3b", "name": "Qwen 2.5 Coder", "category": "code"},
    }

MODELS = load_models_config()

# ============================================================================
# TOOLS DEFINITION
# ============================================================================

def get_current_time(timezone: str = "local") -> str:
    """Get the current time."""
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def get_current_date(format: str = "full") -> str:
    """Get the current date."""
    now = datetime.now()
    if format == "short":
        return now.strftime("%Y-%m-%d")
    elif format == "long":
        return now.strftime("%B %d, %Y")
    else:  # full
        return now.strftime("%A, %B %d, %Y")

def get_day_of_week() -> str:
    """Get the current day of the week."""
    return datetime.now().strftime("%A")

def get_calendar_month(year: int = None, month: int = None) -> str:
    """Get a text calendar for a specific month."""
    now = datetime.now()
    year = year or now.year
    month = month or now.month
    return calendar.month(year, month)

def get_timestamp() -> str:
    """Get the current Unix timestamp."""
    return str(int(datetime.now().timestamp()))

def get_latest_news(category: str = None, country: str = "in", max_results: int = 5, language: str = "en") -> str:
    """Fetch latest news headlines from newsdata.io API."""
    if not NEWSDATA_API_KEY:
        return "Error: NEWSDATA_API_KEY not configured in .env file"
    
    try:
        url = "https://newsdata.io/api/1/latest"
        params = {
            "apikey": NEWSDATA_API_KEY,
            "country": country,
            "language": language
        }
        
        if category:
            params["category"] = category
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") != "success":
            error_msg = data.get("results", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", "Unknown error")
            return f"Error from news API: {error_msg}"
        
        articles = data.get("results", [])
        
        # Filter out duplicates
        articles = [a for a in articles if not a.get("duplicate", False)][:max_results]
        
        if not articles:
            return "No news articles found."
        
        # Format news for readability
        news_output = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            source = article.get("source_name", article.get("source_id", "Unknown"))
            pub_date = article.get("pubDate", "")
            link = article.get("link", "")
            categories = article.get("category", [])
            category_str = ", ".join(categories) if categories else ""
            
            # Clean up description
            description = article.get("description", "")
            if description:
                # Truncate if too long
                if len(description) > 200:
                    description = description[:200] + "..."
            
            news_item = f"{i}. {title}\n   📰 Source: {source}"
            if pub_date:
                news_item += f" | 📅 {pub_date}"
            if category_str:
                news_item += f"\n   🏷️ Categories: {category_str}"
            if description:
                news_item += f"\n   {description}"
            if link:
                news_item += f"\n   🔗 {link}"
            
            news_output.append(news_item)
        
        total = data.get("totalResults", len(articles))
        header = f"📰 Found {total:,} total articles. Showing top {len(articles)}:\n\n"
        
        return header + "\n\n".join(news_output)
    except Exception as e:
        return f"Error: {str(e)}"

# Tool registry - maps tool names to functions
TOOL_FUNCTIONS: Dict[str, callable] = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "get_day_of_week": get_day_of_week,
    "get_calendar_month": get_calendar_month,
    "get_timestamp": get_timestamp,
    "get_latest_news": get_latest_news,
}

# Tool definitions for Ollama
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time. Use this when the user asks what time it is.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (currently only 'local' is supported)",
                        "default": "local"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date. Use this when the user asks what date or day it is today.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["short", "long", "full"],
                        "description": "Date format: 'short' (2024-01-15), 'long' (January 15, 2024), 'full' (Monday, January 15, 2024)",
                        "default": "full"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_day_of_week",
            "description": "Get what day of the week it is (Monday, Tuesday, etc.)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_month",
            "description": "Get a text calendar for a specific month. Useful when user asks to see a calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "The year (defaults to current year)"
                    },
                    "month": {
                        "type": "integer",
                        "description": "The month (1-12, defaults to current month)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_timestamp",
            "description": "Get the current Unix timestamp (seconds since epoch).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_news",
            "description": "Fetch the latest news headlines. Use this when user asks for news, current events, or what's happening in the world.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["business", "entertainment", "environment", "food", "health", "politics", "science", "sports", "technology", "top", "world"],
                        "description": "News category to filter by (optional)"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code (e.g., 'in' for India, 'us' for USA). Defaults to 'in'",
                        "default": "in"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of news articles to return (1-10)",
                        "default": 5
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for news (e.g., 'en' for English, 'hi' for Hindi). Defaults to 'en'",
                        "default": "en"
                    }
                },
                "required": []
            }
        }
    }
]

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool and return its result."""
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'"
    
    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

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
    model_entry = MODELS.get(key, MODELS.get("general", next(iter(MODELS.values()))))
    if isinstance(model_entry, dict):
        return model_entry["tag"]
    return model_entry

# Endpoints

@app.get("/api/models")
async def list_models():
    """Return available models configuration."""
    model_list = []
    for key, entry in MODELS.items():
        if isinstance(entry, dict):
            category = entry.get("category", "general")
            emoji = "💻" if category == "code" else "💬"
            label = f"{emoji} {entry['name']}"
            model_list.append({"key": key, "name": entry["tag"], "label": label})
        else:
            model_list.append({"key": key, "name": entry, "label": key.title()})
    return {"models": model_list}

@app.post("/api/models/reload")
async def reload_models():
    """Reload models config from disk (after re-running setup)."""
    global MODELS
    MODELS = load_models_config()
    return {"status": "ok", "models": list(MODELS.keys())}

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """Stream chat response from Ollama with tool support."""
    model_name = get_model_name(request.model_key)
    
    # Convert Pydantic messages to dicts
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Keep last 20 messages context
    if len(messages) > 20:
        messages = messages[-20:]

    async def generate():
        try:
            # First, make a non-streaming call to check for tool usage
            initial_response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                stream=False
            )
            
            # Check if the model wants to use tools
            if initial_response.get("message", {}).get("tool_calls"):
                tool_calls = initial_response["message"]["tool_calls"]
                
                # Build conversation with tool results
                updated_messages = messages.copy()
                updated_messages.append(initial_response["message"])
                
                # Process each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"].get("arguments", {})
                    
                    # Execute the tool
                    tool_result = execute_tool(tool_name, tool_args)
                    
                    # Add tool result to conversation
                    updated_messages.append({
                        "role": "tool",
                        "content": tool_result
                    })
                
                # Get final response with tool results (streaming)
                final_stream = ollama.chat(
                    model=model_name,
                    messages=updated_messages,
                    stream=True
                )
                
                for chunk in final_stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
            else:
                # No tool calls - make a streaming call for real-time output
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

# ============================================================================
# SETUP WIZARD ENDPOINTS
# ============================================================================

class SetupSelectRequest(BaseModel):
    general: List[Dict[str, str]]
    code: List[Dict[str, str]]

class SetupPullRequest(BaseModel):
    tag: str

@app.get("/setup")
async def serve_setup():
    """Serve the model setup wizard page."""
    return FileResponse("setup.html")

@app.post("/api/setup/select")
async def setup_select(request: SetupSelectRequest):
    """Save the user's model selection to config file."""
    config = {
        "general": [dict(m) for m in request.general],
        "code": [dict(m) for m in request.code],
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return {"status": "ok"}

# Track active pull processes for cancellation
active_pulls: Dict[str, subprocess.Popen] = {}

@app.post("/api/setup/pull")
async def setup_pull(request: SetupPullRequest):
    """Pull a single model using Ollama CLI (cancellable)."""
    tag = request.tag
    try:
        proc = subprocess.Popen(
            ["ollama", "pull", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        active_pulls[tag] = proc

        # Wait for completion
        stdout, stderr = proc.communicate(timeout=600)

        # Clean up tracking
        active_pulls.pop(tag, None)

        if proc.returncode == 0:
            return {"status": "ok", "tag": tag}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to pull {tag}: {stderr}"
            )
    except subprocess.TimeoutExpired:
        proc.kill()
        active_pulls.pop(tag, None)
        raise HTTPException(status_code=504, detail=f"Timeout pulling {tag}")
    except Exception as e:
        active_pulls.pop(tag, None)
        if "cancelled" in str(e).lower():
            raise HTTPException(status_code=499, detail="Download cancelled")
        raise HTTPException(status_code=500, detail=str(e))

class CancelPullRequest(BaseModel):
    tag: str

@app.post("/api/setup/cancel-pull")
async def cancel_pull(request: CancelPullRequest):
    """Cancel an active model pull."""
    tag = request.tag
    proc = active_pulls.pop(tag, None)
    if proc and proc.poll() is None:
        proc.kill()
        proc.wait()
        # Clean up partial download
        try:
            subprocess.run(
                ["ollama", "rm", tag],
                capture_output=True, text=True, timeout=30
            )
        except Exception:
            pass
        return {"status": "cancelled", "tag": tag}
    return {"status": "not_found", "tag": tag}

@app.get("/api/setup/installed")
async def get_installed_models():
    """Return currently installed models from config."""
    if not CONFIG_FILE.exists():
        return {"general": [], "code": []}
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config
    except Exception:
        return {"general": [], "code": []}

@app.get("/api/setup/storage")
async def get_storage_info():
    """Return disk usage for each installed Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return {"models": {}, "total_bytes": 0, "total_display": "0 B"}

        models = {}
        total_bytes = 0

        for line in result.stdout.strip().split("\n")[1:]:  # skip header
            parts = line.split()
            if len(parts) < 4:
                continue
            tag = parts[0]
            # Size is typically like "1.1 GB" or "500 MB" — find it
            size_str = ""
            size_bytes = 0
            for i, p in enumerate(parts):
                if p in ("B", "KB", "MB", "GB", "TB") and i > 0:
                    try:
                        num = float(parts[i - 1])
                        size_str = f"{parts[i-1]} {p}"
                        multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
                        size_bytes = int(num * multipliers.get(p, 1))
                    except ValueError:
                        pass
                    break

            models[tag] = {"size": size_str, "bytes": size_bytes}
            total_bytes += size_bytes

        # Format total
        if total_bytes >= 1024**3:
            total_display = f"{total_bytes / 1024**3:.1f} GB"
        elif total_bytes >= 1024**2:
            total_display = f"{total_bytes / 1024**2:.0f} MB"
        else:
            total_display = f"{total_bytes} B"

        return {"models": models, "total_bytes": total_bytes, "total_display": total_display}
    except Exception:
        return {"models": {}, "total_bytes": 0, "total_display": "0 B"}

class DeleteModelRequest(BaseModel):
    tag: str
    category: str  # "general" or "code"

@app.post("/api/setup/delete")
async def delete_model(request: DeleteModelRequest):
    """Delete a model from config and from Ollama."""
    global MODELS
    # Remove from Ollama
    try:
        subprocess.run(
            ["ollama", "rm", request.tag],
            capture_output=True, text=True, timeout=60
        )
    except Exception:
        pass  # Model might not exist in Ollama, that's ok

    # Remove from config
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)

        category = request.category
        if category in config:
            config[category] = [
                m for m in config[category] if m.get("tag") != request.tag
            ]

        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

    # Reload models in memory
    MODELS = load_models_config()
    return {"status": "ok"}

class AddModelRequest(BaseModel):
    name: str
    tag: str
    ram: str
    category: str  # "general" or "code"

@app.post("/api/setup/add")
async def add_model(request: AddModelRequest):
    """Add a model to config (does not pull it — use /api/setup/pull for that)."""
    global MODELS
    # Load or create config
    config = {"general": [], "code": []}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception:
            pass

    entry = {
        "name": request.name,
        "tag": request.tag,
        "ram": request.ram,
        "category": request.category,
    }

    category = request.category
    if category not in config:
        config[category] = []

    # Avoid duplicates
    existing_tags = [m.get("tag") for m in config[category]]
    if request.tag not in existing_tags:
        config[category].append(entry)

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    # Reload models in memory
    MODELS = load_models_config()
    return {"status": "ok"}

# ============================================================================
# SERVE FRONTEND
# ============================================================================

@app.get("/")
async def serve_index():
    """Serve chat UI, or redirect to setup wizard if not configured."""
    if not CONFIG_FILE.exists():
        return RedirectResponse(url="/setup")
    return FileResponse("index.html")

# Run functionality
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
