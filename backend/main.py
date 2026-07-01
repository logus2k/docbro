"""docbro-api — minimal publish backend.

Receives a dropped file plus a target name/category and makes it "definitive":
  * writes the file into categories/<category-slug>/<filename>
  * appends an entry to documents.json (preserving the file's existing layout)

Both paths are bind-mounted from the docbro source tree (see docker-compose),
so published content is served live by Caddy without an image rebuild.
"""

import json
import os
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

CONTENT_ROOT = Path(os.environ.get("CONTENT_ROOT", "/content"))
DOCS_JSON = CONTENT_ROOT / "documents.json"
CATEGORIES_DIR = CONTENT_ROOT / "categories"

VIDEO_EXT = {"mp4", "webm", "ogg", "ogv", "mov", "m4v"}
ALLOWED_EXT = {"pdf", "md", "markdown"} | VIDEO_EXT

app = FastAPI(title="docbro-api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return value.strip("_") or "misc"


def safe_filename(name: str) -> str:
    name = os.path.basename(name or "")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "file"


def sanitize_rel_path(p: str) -> str:
    """Turn a user-supplied folder path (possibly nested, e.g. 'sdlc/articles')
    into a safe relative path under categories/. Slugifies each segment and
    drops empty / '.' / '..' segments (no traversal)."""
    segments = []
    for seg in (p or "").replace("\\", "/").split("/"):
        seg = seg.strip()
        if not seg or seg in (".", ".."):
            continue
        slug = slugify(seg)
        if slug:
            segments.append(slug)
    return "/".join(segments)


def _append_entry_preserving_layout(text: str, entry: dict) -> str:
    """Insert a new document entry just before the closing ] of the documents
    array, keeping the rest of the file (tabs, blank-line grouping) intact.
    Falls back to a full reformat if the textual insert would be invalid."""
    idx = text.rfind("]")
    if idx != -1:
        before = text[:idx].rstrip()
        after = text[idx:]
        body = json.dumps(entry, indent="\t", ensure_ascii=False)
        block = "\n".join(("\t\t" + ln) if ln else ln for ln in body.split("\n"))
        sep = "" if before.endswith("[") else ","
        candidate = f"{before}{sep}\n{block}\n\t{after}"
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    data = json.loads(text)
    data.setdefault("documents", []).append(entry)
    return json.dumps(data, indent="\t", ensure_ascii=False)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/publish")
async def publish(
    file: UploadFile = File(...),
    name: str = Form(...),
    category: str = Form(...),
    path: str = Form(""),
):
    name = name.strip()
    category = category.strip()
    if not name or not category:
        raise HTTPException(status_code=400, detail="name and category are required")

    filename = safe_filename(file.filename or name)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"unsupported file type: .{ext}")

    if not DOCS_JSON.exists():
        raise HTTPException(status_code=500, detail="documents.json not found (volume not mounted?)")

    # Folder path is independent of the (display) category, and may be nested
    # (e.g. "sdlc/articles"). If omitted, fall back to a slugged, nested path
    # derived from the (possibly nested, e.g. "SDLC/Articles") category.
    folder = sanitize_rel_path(path) or sanitize_rel_path(category) or "misc"
    dest_dir = CATEGORIES_DIR / folder
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / filename
    stem, dot, suffix = filename.rpartition(".")
    counter = 1
    while dest.exists():
        base = stem or filename
        dest = dest_dir / (f"{base}_{counter}{dot}{suffix}" if dot else f"{base}_{counter}")
        counter += 1

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # Video isn't a first-class document type — wrap it in a small markdown file
    # that embeds the player (relative src works both locally and under /docbro).
    if ext in VIDEO_EXT:
        video_rel = f"categories/{folder}/{dest.name}"
        md_base = dest.stem or "video"
        md_path = dest_dir / f"{md_base}.md"
        c = 1
        while md_path.exists():
            md_path = dest_dir / f"{md_base}_{c}.md"
            c += 1
        md_path.write_text(
            f"# {name}\n\n"
            '<div class="embedded-video">\n'
            "    <video controls>\n"
            f'        <source src="{video_rel}" type="{file.content_type or "video/mp4"}">\n'
            "    </video>\n"
            "</div>\n",
            encoding="utf-8",
        )
        location = f"categories/{folder}/{md_path.name}"
    else:
        location = f"categories/{folder}/{dest.name}"

    entry = {
        "name": name,
        "category": category,
        "location": location,
    }

    try:
        text = DOCS_JSON.read_text(encoding="utf-8")
        new_text = _append_entry_preserving_layout(text, entry)
        # documents.json is a single-file bind mount, so we can't rename over it
        # (Errno 16). Write in place — new_text is already validated JSON above.
        DOCS_JSON.write_text(new_text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — surface any write failure to the client
        # roll back the saved file so we don't leave an orphan
        try:
            dest.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"failed to update documents.json: {exc}")

    return {"ok": True, "entry": entry}
