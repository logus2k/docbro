# Notebook Collaboration Platform - Implementation Specification

## 1. Overview

A web-based collaborative notebook platform that allows multiple users to edit and execute Jupyter notebooks in real-time. Built with FastAPI, Socket.IO, and vanilla JavaScript (ES6 classes), it provides per-project Python virtual environments, live kernel execution with streamed outputs, and cell-level collaborative editing.

---

## 2. Architecture Summary

### Stack

- **Backend:** FastAPI + python-socketio (async)
- **Frontend:** Vanilla JS (ES6 classes), modular CSS
- **Code Editor:** CodeMirror 6
- **Notebook Format:** Native .ipynb JSON
- **Execution:** jupyter_client (ZMQ) bridged to Socket.IO
- **Environments:** Python venvs (project-scoped + shared)
- **Collaboration:** Socket.IO rooms, cell-level locking with lease-based TTL

### High-Level Flow

```
Browser (Client)
  |
  |-- HTTP (REST) -----> FastAPI (notebooks CRUD, venv management)
  |
  |-- Socket.IO -------> python-socketio (execution, collaboration, sync)
                              |
                              |-- ZMQ --> Jupyter Kernel (per session, bound to a venv)
```

---

## 3. Directory Structure

```
notebook-server/
  backend/
    app/
      __init__.py
      main.py                  # FastAPI + Socket.IO setup, CORS, static mount
      config.py                # Paths, constants, defaults
      managers/
        __init__.py
        notebook_manager.py    # Notebook file CRUD
        venv_manager.py        # Venv lifecycle + package management
        kernel_manager.py      # Kernel process lifecycle
        execution_bridge.py    # ZMQ <-> Socket.IO bridge
        collaboration.py       # Rooms, cell locks, change broadcast
      routers/
        __init__.py
        notebooks.py           # REST endpoints for notebook CRUD
        venvs.py               # REST endpoints for venv management
    requirements.txt
  frontend/
    index.html
    css/
      base.css                 # Reset, variables, typography
      notebook.css             # Notebook container layout
      cell.css                 # Cell editor + output styling
      toolbar.css              # Notebook toolbar
      venv-panel.css           # Venv management panel
      output.css               # Cell output rendering
    js/
      app.js                   # Entry point, initialization
      NotebookEditor.js        # Main notebook container, manages cells
      CellEditor.js            # Wraps CodeMirror instance per cell
      CellOutput.js            # Renders cell outputs (text, images, HTML, plots)
      NotebookToolbar.js       # Kernel controls, venv selector, save
      VenvPanel.js             # Venv creation, package management UI
      KernelClient.js          # Socket.IO communication layer
  data/
    projects/                  # Per-project notebooks + venvs
    shared_venvs/              # Reusable shared venvs
```

---

## 4. Backend Components

### 4.1 Notebook Manager

**Responsibility:** CRUD operations for .ipynb files on disk.

**Operations:**

| Operation | Method | Endpoint | Description |
|-----------|--------|----------|-------------|
| List notebooks | GET | `/api/projects/{id}/notebooks` | List all .ipynb files in a project |
| Get notebook | GET | `/api/projects/{id}/notebooks/{path}` | Read and return notebook JSON |
| Create notebook | POST | `/api/projects/{id}/notebooks` | Create a new empty notebook |
| Update notebook | PUT | `/api/projects/{id}/notebooks/{path}` | Write updated notebook JSON to disk |
| Delete notebook | DELETE | `/api/projects/{id}/notebooks/{path}` | Remove notebook file |

**Notebook JSON structure (standard .ipynb v4):**

```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": {},
    "venv_ref": {
      "type": "project | shared",
      "name": "default"
    }
  },
  "cells": [
    {
      "cell_type": "code | markdown | raw",
      "source": ["line1\n", "line2\n"],
      "metadata": {},
      "outputs": [],
      "execution_count": null
    }
  ]
}
```

---

### 4.2 Venv Manager

**Responsibility:** Create, delete, and manage Python virtual environments and their packages.

**Operations:**

| Operation | Method | Endpoint | Description |
|-----------|--------|----------|-------------|
| List project venvs | GET | `/api/projects/{id}/venvs` | List venvs scoped to a project |
| Create project venv | POST | `/api/projects/{id}/venvs` | Create venv in project directory |
| Delete project venv | DELETE | `/api/projects/{id}/venvs/{name}` | Remove project venv |
| List shared venvs | GET | `/api/venvs` | List globally shared venvs |
| Create shared venv | POST | `/api/venvs` | Create a shared venv |
| Delete shared venv | DELETE | `/api/venvs/{name}` | Remove shared venv |
| List packages | GET | `.../{name}/packages` | List installed packages |
| Install packages | POST | `.../{name}/packages` | pip install one or more packages |
| Remove packages | DELETE | `.../{name}/packages` | pip uninstall packages |

**Internals:**

- Venvs created via `python -m venv <path>`
- Package operations via subprocess calls to the venv's pip
- Project venvs stored at `/data/projects/<id>/venvs/<name>/`
- Shared venvs stored at `/data/shared_venvs/<name>/`
- Resolves venv reference to absolute Python interpreter path: `<venv_path>/bin/python`

---

### 4.3 Kernel Manager

**Responsibility:** Start, stop, restart, and monitor Jupyter kernel processes.

**Registry (in-memory):**

```python
kernels = {
    "<session_id>": {
        "kernel_manager": KernelManager,
        "kernel_client": KernelClient,
        "venv_path": str,
        "project_id": str,
        "notebook_path": str,
        "client_sid": str,
        "last_heartbeat": datetime,
        "status": "idle | busy | starting | dead"
    }
}
```

**Lifecycle:**

- **Start:** Receives venv path, creates a KernelManager with the venv's Python as the
  kernel executable, starts the kernel process, creates a KernelClient for ZMQ communication.
- **Stop:** Shuts down kernel process, cleans up ZMQ connections, removes from registry.
- **Restart:** Stops then starts with same configuration.
- **Idle timeout:** Background task checks `last_heartbeat`. If exceeded (configurable,
  default 10 minutes), kernel is stopped automatically.
- **Interrupt:** Sends SIGINT to kernel process to interrupt running execution.

---

### 4.4 Execution Bridge

**Responsibility:** Bridge between Socket.IO events and Jupyter kernel ZMQ channels.

**Flow:**

```
Client emits cell:execute { cell_index, code }
  -> Server sends execute_request to kernel via ZMQ shell channel
  -> Server listens on kernel IOPub channel for output messages
  -> For each output message (stream, display_data, execute_result, error):
       -> Server emits cell:output { cell_index, output } to client
       -> Server broadcasts to all clients in the notebook room
  -> On execute_reply (shell channel):
       -> Server emits cell:execute_complete { cell_index, execution_count }
```

**Jupyter message types handled:**

| ZMQ Message Type | Maps To | Content |
|------------------|---------|---------|
| stream | stdout/stderr text | `{ name: "stdout", text: "..." }` |
| display_data | Rich output | `{ data: { "image/png": "...", "text/html": "..." } }` |
| execute_result | Cell result | `{ data: { "text/plain": "..." }, execution_count: N }` |
| error | Traceback | `{ ename, evalue, traceback[] }` |
| status | Kernel state | `{ execution_state: "busy | idle" }` |

**IOPub listener:** An asyncio task per active kernel that continuously reads from the
IOPub channel and dispatches messages via Socket.IO.

---

### 4.5 Collaboration Manager

**Responsibility:** Manage real-time multi-user editing of notebooks.

**Rooms:**

- Each open notebook gets a Socket.IO room: `notebook:<project_id>:<notebook_path>`
- Clients join on `notebook:open`, leave on `notebook:close` or disconnect
- Room tracks connected client SIDs and cell locks

**Cell-level locking:**

```python
cell_locks = {
    "<cell_index>": {
        "owner_sid": str,
        "owner_name": str,        # display name for UI
        "acquired_at": datetime,
        "expires_at": datetime     # TTL, default 60s
    }
}
```

- Client requests lock via `cell:lock { cell_index }`
- Server grants if cell is unlocked or lock expired, denies otherwise
- Lock renewed via heartbeat (every 30s)
- Released explicitly via `cell:unlock` or on disconnect / TTL expiry
- All lock state changes broadcast to room

**Change synchronization:**

| Event | Payload | Broadcast |
|-------|---------|-----------|
| `cell:update` | `{ cell_index, source, version }` | Room except sender |
| `cell:add` | `{ cell_index, cell_type }` | Room except sender |
| `cell:delete` | `{ cell_index }` | Room except sender |
| `cell:move` | `{ from_index, to_index }` | Room except sender |
| `cell:output` | `{ cell_index, output }` | All room members |
| `cell:lock_changed` | `{ cell_index, owner, locked }` | All room members |

**Conflict resolution (MVP):** Last write wins at the cell level. Since cells are locked
to one editor at a time, conflicts should be rare and limited to edge cases around lock expiry.

---

## 5. Socket.IO Events - Complete Reference

### Client -> Server

```
notebook:open        { project_id, notebook_path }
notebook:close       { project_id, notebook_path }
notebook:save        { project_id, notebook_path }

cell:lock            { cell_index }
cell:unlock          { cell_index }
cell:update          { cell_index, source }
cell:add             { cell_index, cell_type }
cell:delete          { cell_index }
cell:move            { from_index, to_index }
cell:execute         { cell_index }

kernel:start         { venv_ref }
kernel:stop          {}
kernel:restart       {}
kernel:interrupt     {}

heartbeat            {}
```

### Server -> Client

```
notebook:state       { notebook_json, locks, connected_users }
notebook:saved       { success, error? }

cell:updated         { cell_index, source, by_sid }
cell:added           { cell_index, cell_type, by_sid }
cell:deleted         { cell_index, by_sid }
cell:moved           { from_index, to_index, by_sid }
cell:output          { cell_index, output }
cell:execute_complete { cell_index, execution_count }
cell:lock_changed    { cell_index, owner, locked }

kernel:status        { status }

user:joined          { sid, name }
user:left            { sid, name }

error                { message, code }
```

---

## 6. Frontend Components

### 6.1 app.js

- Entry point
- Initializes KernelClient (Socket.IO connection)
- Creates NotebookEditor and mounts to DOM
- Handles routing/project selection (basic for PoC)

### 6.2 KernelClient.js

- Wraps Socket.IO client
- Exposes methods: `connect()`, `openNotebook()`, `executeCell()`, `startKernel()`, etc.
- Event emitter pattern for incoming events
- Handles heartbeat interval
- Reconnection logic

### 6.3 NotebookEditor.js

- Main container class
- Manages array of CellEditor instances
- Handles cell add/delete/move operations
- Coordinates with KernelClient for execution
- Applies remote changes from collaboration events

### 6.4 CellEditor.js

- Wraps a CodeMirror 6 EditorView instance
- Manages cell type (code/markdown)
- Handles lock acquisition on focus, release on blur
- Emits content changes to NotebookEditor
- Shows lock indicator (who is editing)
- Run button per cell

### 6.5 CellOutput.js

- Renders outputs below each code cell
- Handles output types:
  - `text/plain` - rendered as preformatted text
  - `text/html` - rendered in sandboxed container
  - `image/png`, `image/jpeg` - rendered as `<img>` from base64
  - `image/svg+xml` - rendered inline
  - `application/json` - rendered as collapsible tree
  - `error` - rendered with traceback styling
- Appends streamed outputs incrementally

### 6.6 NotebookToolbar.js

- Kernel status indicator (idle/busy/dead/disconnected)
- Venv selector dropdown (project + shared, grouped)
- Buttons: Run All, Restart Kernel, Interrupt, Save
- Connected users display

### 6.7 VenvPanel.js

- Slide-out or modal panel
- Lists venvs with installed packages
- Create new venv (name + optional requirements.txt paste)
- Install/uninstall packages via search input
- Loading/progress indicators for async operations


---

## 7. Data Model

### 7.1 Filesystem Layout

    data/
      projects/
        <project_id>/
          notebooks/
            analysis.ipynb
          venvs/
            default/
              bin/python
            ml-env/
              bin/python
      shared_venvs/
        data-science-base/
          bin/python

### 7.2 Server-Side State (In-Memory)

    # Kernel registry
    kernels: dict[str, KernelSession] = {}

    # Room state
    rooms: dict[str, RoomState] = {
        "notebook:<project>:<path>": {
            "clients": set[str],
            "cell_locks": dict[int, CellLock],
            "notebook_hash": str
        }
    }

### 7.3 Notebook Metadata Extension

Standard .ipynb metadata extended with venv_ref (type: project or shared, name) and
collaboration metadata (last_modified_by, last_modified_at).

---

## 8. Implementation Order

### Phase 1 - Notebook Manager
- Project scaffolding (FastAPI app, config, directory structure)
- Notebook CRUD endpoints
- Basic frontend: list and open notebooks, render cells as read-only

### Phase 2 - Venv Manager
- Venv CRUD endpoints
- Package install/uninstall/list
- Frontend venv panel and selector

### Phase 3 - Kernel Manager + Execution Bridge
- Kernel lifecycle management
- ZMQ to Socket.IO bridge
- Frontend cell execution with streamed output rendering
- Kernel status indicators

### Phase 4 - Cell Editing
- CodeMirror 6 integration per cell
- Cell add/delete/move operations
- Save notebook back to server
- Markdown cell rendering (edit/preview toggle)

### Phase 5 - Collaboration
- Socket.IO rooms per notebook
- Cell-level locking with lease TTL
- Change broadcast and remote patching
- Connected users display
- Heartbeat and disconnect cleanup

---

## 9. Dependencies

### Backend (requirements.txt)

    fastapi
    uvicorn[standard]
    python-socketio
    aiofiles
    jupyter_client
    pyzmq

### Frontend (CDN)

- CodeMirror 6 (core + Python language + themes)
- Socket.IO client
- Marked.js (markdown rendering)
- highlight.js (syntax highlighting in output)

---

## 10. Configuration (config.py)

    DATA_DIR = "/data"
    PROJECTS_DIR = f"{DATA_DIR}/projects"
    SHARED_VENVS_DIR = f"{DATA_DIR}/shared_venvs"
    KERNEL_IDLE_TIMEOUT_SECONDS = 600
    CELL_LOCK_TTL_SECONDS = 60
    HEARTBEAT_INTERVAL_SECONDS = 30
    DEFAULT_VENV_NAME = "default"
    SYSTEM_PYTHON = "/usr/bin/python3"

---

## 11. Open Questions / Future Considerations

- **Authentication:** Not in PoC scope. Clients identified by Socket.IO SID only.
- **File upload:** Uploading notebooks via the UI (drag and drop). Deferred.
- **Terminal access:** Providing a web terminal to the venv. Deferred.
- **Git integration:** Version control for notebooks. Deferred.
- **Resource limits:** CPU/RAM caps per kernel. Deferred until containerized isolation.
- **Export:** Export notebooks as HTML/PDF. Deferred.
- **Autocomplete:** Python autocomplete in CodeMirror via kernel introspection (complete_request). Nice to have.
