# 6. Frontend Architecture
## 6.1 Concept primer

noted's frontend is a single-page web application written in vanilla ES modules with no framework. The design choice that deserves the most attention is the deliberate *absence* of a bundler for the application code: all JS files are served directly to the browser as ES modules, and vendor libraries (CodeMirror, Wunderbaum, jsPanel, socket.io, KaTeX, echarts) are loaded via classic script tags from `frontend/vendor/`. The result is a zero-build dev loop - edit a file, hard-refresh the tab, see the change - which is the primary reason a project-scoped, rapidly iterated tool chose vanilla over React or Vue.

The shell layout is VS Code-shaped for a reason: the target user is a developer, the target task is "edit code, run it, inspect outputs", and the Code-style layout (icon bar + sidebar + tabs + right panel + status bar) is a well-understood mental model for that task. noted does not try to hide that it is a developer tool.

Four libraries do most of the heavy visual work: **Wunderbaum** for trees (Explorer, Data Catalog, Model Registry), **jsPanel** for floating/undockable windows, **CodeMirror 6** for every code editor (including every notebook cell), and **socket.io** for every backend event stream. Together they explain 80% of the frontend's surface area.

## 6.2 Entry point and module graph

`frontend/index.html` (121 lines) is the entry document. Lines 101-116 load 18 vendor scripts via classic `<script>` tags: socket.io, marked, hljs, jsPanel, KaTeX, Wunderbaum, xterm, notyf, echarts, dagre. Line 119 is the one and only ES-module entry point:

```html
<script type="module" src="static/js/app.js"></script>
```

The `/static/` prefix is a FastAPI mount (`backend/app/main.py:1202`) that points at `/frontend/` on disk. There is no `dist/` directory. Every `.js` file under `frontend/js/` is the exact file the browser fetches.

`frontend/js/app.js` is the entry module. Lines 1-50 import the top-level classes: `KernelClient`, `NotebookEditor`, `ChatPanel`, `ExplorerPanel`, `GitPanel`, `TabBar`, `MenuBar`. The `App` class constructor calls six initialization helpers in sequence (lines 43-49):

```javascript
initStatusBar(this);
initMenuCommands(this);
initChat(this);
initFileEditors(this);
initNotebooks(this);
initTabs(this);
```

Each helper registers panel-specific event handlers onto the singleton `app` instance. The app object is the central bus - every panel has a reference to it, every command is dispatched through it, and its maps (`_notebookEditors`, `_fileEditors`, `_documentTabs`, `_undockedPanels`) are the canonical state of which UI surfaces are alive.

**No framework.** No virtual DOM, no reactivity, no component lifecycle. State is mutable; DOM updates are direct; events are explicit. The trade-off: every update path is visible in the code (a feature), but sharing UI between panels requires explicit wiring (a cost).

## 6.3 Layout: the VS Code-style shell

`frontend/css/base.css:93-179` defines the top-level layout with flexbox:

- `#app` - flex column, full viewport height.
- `#menu-bar`, `#toolbar`, `#info-bar` - stacked at the top.
- `#below-bar` - flex row taking remaining height.
- `#icon-bar` - left vertical strip (~50 px), always visible.
- `#sidebar-panel` - Explorer / Git / TOC / Settings, collapsible.
- `#content-area` - the center, holds the tab bar and the active tab's content.
- `#right-panel` - Run Manager, Chat, or Doc view, collapsible.
- `#status-bar` - bottom bar (20 px, dark theme), always visible.

CSS is split across ~27 feature-scoped files (`cell.css`, `notebook.css`, `tab-bar.css`, `icon-bar.css`, `sidebar.css`, `right-panel.css`, ...) that all import from `base.css`. No CSS-in-JS, no preprocessor, no utility framework - plain CSS with a conventional BEM-ish class naming.

The layout's load-bearing affordance is that *every* content area (`#sidebar-panel`, `#content-area`, `#right-panel`) is independently resizable and collapsible. This is what lets the user dedicate most of their viewport to the thing they are focused on - a notebook cell, a Compose panel, a chat thread - without losing quick access to the others.

## 6.4 Wunderbaum trees

Wunderbaum is a vanilla-JS tree library bundled into `frontend/vendor/wunderbaum/wunderbaum.umd.min.js`. It exposes `mar10.Wunderbaum` on the global namespace; noted does not import it as an ES module.

Tree instances are created throughout the Explorer panels with a common pattern. `ExplorerPanel.js:515-552` is a representative example:

```javascript
this._tree = new mar10.Wunderbaum({
  element: treeEl,
  source: source,
  adjustHeight: false,
  selectMode: 'single',
  checkbox: false,
  icon: true,
  iconMap: FA_ICON_MAP,
  render: (e) => {...},
  lazyLoad: (e) => {...},
  activate: (e) => {...},
  click: (e) => {...},
  dblclick: (e) => {...},
});
```

**Key conventions:**

- `source` is either a static array (for known-at-build-time trees) or an async function that returns children on demand (for lazy-loaded trees).
- `lazyLoad` returns a promise resolving to the children when a user expands a node.
- `activate` is the primary click handler; `dblclick` handles the "open in tab" action.
- noted *never* calls `resetLazy()` on static-children nodes - a hard-learned lesson documented in memory. For dynamic trees, the correct refresh is `addChildren()` after removing the old children.

The Explorer tree has nodes for Projects, Git, TOC, Docs, Hydra, MLflow, DVC, Evidently, Serving. Each has its own view module under `frontend/js/panels/explorer/Explorer*Views.js` that handles rendering the detail pane when a node is activated.

## 6.5 jsPanel and the TabBar

`TabBar` (`frontend/js/TabBar.js:1-282`) manages tabs above the content area. Tabs are stored in a `Map` keyed by a stable key (e.g. `notebook:<project>:<file>`, `doc:<category>:<name>`, `pyfile:<project>:<filename>`). Tab properties include `closable`, `preview`, `undockable`, and an `undocked` flag.

Preview tabs (lines 55-57) replace the previous preview tab instead of stacking. This is the VS Code "single click = preview, double click = pinned" pattern. A preview tab is promoted to permanent when the user edits it or explicitly pins it.

**Undocking** (`frontend/js/app-tabs.js:855-940`) is implemented via jsPanel. The flow:

1. User clicks the undock icon on a tab.
2. `app-tabs.js` calls `jsPanel.create({...})` with `panelSize: '70vw 70vh'`, centered with offset, `addCloseControl`, `boxShadow: 3`.
3. The tab's DOM element is *moved* (not cloned, except for canvas-heavy PDF tabs - see below) into the panel's content area.
4. The tab bar shows the tab as undocked (greyed out or with an icon indicator).
5. The panel's `onclosed` callback re-docks unless the user clicked close with the `_docking` flag cleared.

A custom dock button is added to the jsPanel header (lines 922-936) so the user can re-dock without closing.

**The PDF-undock bug** (fixed 2026-04-15) was a reminder that `cloneNode(true)` does not copy canvas pixel data. After cloning a PDF tab's DOM into a jsPanel, every page's canvas was blank. The fix: after cloneNode, walk the canvases in the original and the clone in parallel and blit pixels with `ctx.drawImage(orig, 0, 0)`. The same principle applies to any tab that holds imperative canvas state - cloning the DOM is necessary but not sufficient; the bitmap has to be explicitly copied.

## 6.6 CodeMirror 6 and the cell editor

Every code-edit surface in noted is a CodeMirror 6 instance. Unlike everything else in the frontend, CodeMirror *is* bundled - it has enough internal module structure that shipping its loose ESM would require ~30 HTTP requests per editor.

`frontend/js/CellEditor.js:1-78` is the wrapper. Imports from the bundled `codemirror.bundle.js`:

- `EditorView`, `EditorState`, `keymap`
- Gutters: `lineNumbers`, `lintGutter`, `highlightActiveLine`
- Languages: `python`, `javascript`, `markdown` (plus YAML/JSON via `legacy-modes`)
- Themes: `ayuLight`, `clouds`, `espresso`, `smoothy`, `tomorrow`, `oneDark`
- `autocompletion`, `syntaxHighlighting`

The bundle is built via esbuild at `scripts/build-codemirror/` and checked into the repo. `package.json` in that dir lists the exact dependencies. Rebuilding is a one-line `npm run build` and only happens when a new language or extension is added.

Theme switching uses CodeMirror's `Compartment` (line 64) so the theme can be reconfigured live without recreating the editor. Each cell gets its own editor; the cell type (`code` vs `markdown`) decides which language extension is mounted.

LSP integration is wired via `codemirror-languageserver` (dependency in `scripts/build-codemirror/package.json:27`). When a Python or R cell is focused, the editor connects to the backend's LSP proxy, which forwards requests to the language-specific LSP server (Pyright for Python, R LSP for R). Diagnostics render in the lint gutter; completions come through the autocomplete extension.

## 6.7 Panels and tabs: the lifecycle

There is no central panel registry. Panels are created *on demand* when their corresponding action fires and are tracked via `app._*` maps for lookup on re-open.

The typical lifecycle:

1. **User action** - click a tree node, a menu item, or a button.
2. **Handler in `app-*.js`** - checks if a tab for this resource already exists (`app._notebookEditors.get(key)`, etc.).
3. **Create or focus.** If it exists, `tabBar.activate(key)`. If not, instantiate the panel class (`NotebookEditor`, `DocumentViewer`, ...) and call `tabBar.addTab(key, title, element, opts)`.
4. **Activation handler** (`onActivateTab` registered with TabBar) shows the activated tab's DOM element and hides the previous one.
5. **Close.** Either via the tab's X button or a close-all command. The panel's `destroy()` method (if any) releases listeners and DOM; the entry is removed from the `app._*` map.

Panel types currently in use:

- `NotebookEditor` - a full notebook with cell editors, toolbar, metrics bar.
- `FileEditor` - a single file edited as a CodeMirror buffer (non-notebook code).
- `MediaViewer` - images, video.
- `DocumentViewer` - markdown / PDF, rendered via `marked` + `pdfjs`.
- `ChatPanel` - the AI assistant conversation.
- `ExplorerPanel` - the left sidebar tree.
- Service tabs - iframe wrappers for MLflow, Airflow, MinIO, Evidently.

The deliberate absence of a framework means every panel has an unambiguous owner and an unambiguous cleanup path.

## 6.8 Key dispatchers and shortcuts

`MenuBar.js:316-382` installs a global `keydown` listener. It parses modifier+key combinations (`Ctrl+S`, `Ctrl+Shift+F`, `F12`, ...) and looks them up in a `shortcutMap` populated from `frontend/menu.json`.

`frontend/menu.json` is the canonical source of truth for menu items, labels, shortcuts, and their command ids. Example entries:

```json
{"id": "file.save", "label": "Save", "shortcut": "Ctrl+S"},
{"id": "edit.findReplace", "label": "Find and Replace", "shortcut": "Ctrl+H"},
{"id": "edit.formatDocument", "label": "Format Document", "shortcut": "Ctrl+Shift+F"},
{"id": "edit.goToDefinition", "label": "Go to Definition", "shortcut": "F12"}
```

Commands are looked up in a registry populated by `initMenuCommands` and executed via `executeCommand(id)`.

**The CodeMirror guard** (MenuBar.js:363-371) excludes standard editing shortcuts (Ctrl+Z/X/C/V/A) when the focused element is inside a CodeMirror editor. This prevents the menu system from stealing shortcuts that the editor handles natively.

Cell execution shortcuts (`Shift+Enter` for run, `Ctrl+Shift+Enter` for debug) are *not* in the MenuBar - they live on the `NotebookEditor` because they are cell-scoped rather than app-scoped. `app-notebooks.js:896-911` also installs debug keys (F5 continue, Shift+F5 stop, F10/F11 step) scoped to the active notebook.

## 6.9 Status bar, icon bar, menu bar

`frontend/js/app-status-bar.js` is the status bar. On startup it fetches `/api/system/info` and populates pills: Host OS (golden), Container OS (green), Python version, branch, project, pipeline status. The Problems indicator counts diagnostics. Cursor info (`Ln 42, Col 5`) updates on cell focus. A socket listener on `pipeline:status` updates the pipeline pill live.

`frontend/js/IconBar.js:7-154` is the left vertical icon strip. Two groups separated by a flex spacer:

- **Top**: Projects, Git, TOC, Assistant, Debug, Docs.
- **Bottom**: Airflow, MLflow, MinIO, Evidently, Settings.

Each icon is either SVG or a FontAwesome glyph. Click delegates to `app._onIconBarClick(key)`, which either toggles the sidebar to the matching view or opens a service tab in the content area.

`frontend/js/MenuBar.js` renders the top menu bar from `menu.json`. File, Edit, View, Terminal, Help menus. `Alt+F/E/V/T/H` toggles the corresponding dropdown via keyboard. Menu items that lack a `shortcut` are click-only.

## 6.10 Socket.io and the event surface

`frontend/js/KernelClient.js:1-80` initializes socket.io. The `connect()` method derives the socket path from the page URL (so the frontend works behind any reverse proxy) and sets transports to `['websocket', 'polling']` with 10 reconnection attempts.

Events consumed by the frontend:

- Connection: `connect`, `disconnect`, `connect_error`.
- Notebook: `notebook:state`, `notebook:saved`.
- Cell: `cell:updated`, `cell:added`, `cell:deleted`, `cell:moved`, `cell:output`, `cell:execute_start`, `cell:execute_complete`, `cell:lock_changed`, `cell:diagnostics`.
- Kernel: `kernel:status`.
- Users: `user:joined`, `user:left`.
- Runs: `run:started`, `run:complete`.
- Metrics: `metrics:update` (from the MLflow monkey-patch - see Chapter 2.3.2).
- Pipeline: `pipeline:status`, `pipeline:task_status`.

A custom `on(event, callback)` emitter (lines 194-215) lets panels subscribe to specific events without knowing about socket.io directly. The KernelClient acts as a pub/sub intermediary so that replacing socket.io in the future would not require touching every panel.

## 6.11 Discussion-ready talking points

**Q: Why no framework?**
A: Because the project's iteration velocity depends on a zero-build dev loop. Adding React or Vue would require a bundler, a dev server, a source-map pipeline, and a mental model of component lifecycles. For a single-developer, rapidly iterated project, vanilla ES modules plus three or four well-chosen vendor libraries is strictly simpler. The cost is more explicit wiring; the payoff is that every update path is inspectable by reading the code rather than reasoning about a framework's abstractions.

**Q: Why bundle CodeMirror but not the rest?**
A: Because CodeMirror 6 is internally modular to the point of being un-ship-able as loose files (~30 transitive imports per editor). Bundling it once into `codemirror.bundle.js` avoids 30 HTTP round-trips per notebook load. Wunderbaum and jsPanel ship as pre-bundled UMD files; socket.io and KaTeX have minified builds. The application code itself is cheap to load (native ES modules + HTTP/2 multiplexing) so there is no bundling win.

**Q: How does a new panel get added?**
A: (1) Write the class at `frontend/js/panels/YourPanel.js`. (2) Import and instantiate it in an `initXxx(app)` helper in `app-*.js`. (3) Register the tab via `app._tabBar.addTab(key, title, element, opts)`. (4) Handle activation/deactivation in `onActivateTab`. No framework ceremony; no manifest; no registration map. If you want it to appear in the icon bar or menu, add it to `IconBar.js` or `menu.json`.

**Q: Why is jsPanel used for undocking but not for the main layout?**
A: Because the main layout is fixed, predictable, and always visible - flexbox is the right primitive. jsPanel is designed for floating, draggable, resizable windows that the user summons on demand. Using it for the main chrome would add unnecessary state (position, size, z-index) to surfaces that should not be moved.

**Q: How does the frontend stay in sync with backend state?**
A: Through socket.io events that are *authoritative* for state transitions and *advisory* for polling. When a cell executes, the backend emits `cell:execute_start` and `cell:output` events; the frontend's NotebookEditor subscribes to these and updates the DOM directly. State that does not change often (project list, document catalog) is fetched via REST on demand and cached in memory. The rule: anything that could change during a user's session without them initiating it is delivered via socket.io; anything that only changes when the user clicks a thing is REST.

**Q: Why vanilla CSS instead of Tailwind or a component library?**
A: Same reason as no framework. Tailwind adds a build step; a component library locks in visual decisions. noted's visual language is small and consistent enough that ~27 hand-written CSS files, organized by feature, are easier to reason about than an atomic-class system. The themes are swappable via CSS custom properties, which is all the theming surface noted needs.

**Q: What happens when socket.io disconnects?**
A: The KernelClient attempts 10 reconnects with exponential backoff. The status bar's connection pill turns red. Cell execution commands are buffered in the frontend and replayed on reconnect. Long-running execution results that were in-flight when the socket dropped are re-requested on reconnect via a backfill query. Silent degradation is explicitly avoided - a stale state is always visible to the user.

**Q: Is the frontend tested?**
A: Not yet. The project is in its demo-driven iteration phase; the test pyramid is backend-first (see Module 7). Adding Playwright or Cypress for frontend E2E tests is in the post-demo backlog. The trade-off has been accepted: fewer safety nets, faster iteration.
