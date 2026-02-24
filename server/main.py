"""
DocBro Content Save Server
Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime

app = FastAPI(title="DocBro Content Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

SAVE_DIR = Path("saved_content")
SAVE_DIR.mkdir(exist_ok=True)


class SaveRequest(BaseModel):
    content: str
    filename: str | None = None


@app.post("/save")
async def save_content(req: SaveRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content is empty")

    if req.filename:
        safe_name = Path(req.filename).name
        if not safe_name.endswith(".md"):
            safe_name += ".md"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"content_{timestamp}.md"

    filepath = SAVE_DIR / safe_name
    filepath.write_text(req.content, encoding="utf-8")

    return {"status": "ok", "filename": safe_name, "path": str(filepath)}
