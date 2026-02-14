# DocBro

A static, client-side documentation browser for organizing and viewing technical Markdown content. It provides a split-pane interface with hierarchical tree navigation, tabbed document switching, and rich rendering of math, diagrams, and code.

## Features

- Split-pane layout with a resizable divider (tree navigation on the left, content on the right)
- Hierarchical tree view organized by category, document, and in-document headers
- Tab bar for quick switching between documents in the same category
- Markdown rendering with KaTeX math, Mermaid diagrams, and syntax-highlighted code blocks
- Copy-to-clipboard buttons on code blocks
- Image lightbox viewer
- Deep linking via URL hash parameters (e.g. `/#category=CUDA&tab=Glossary`)
- Scroll-synced header tracking in the tree view
- No build step — all rendering happens in the browser

## Project Structure

```
docbro/
├── index.html            # Main entry point
├── documents.json        # Document catalog (categories, titles, file paths)
├── categories/           # Markdown content organized by topic
├── script/               # Application JavaScript
├── styles/               # CSS stylesheets
├── libraries/            # Third-party JS/CSS (Marked, KaTeX, Highlight.js, etc.)
├── fonts/                # Custom fonts
├── images/               # Static images
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Docker Compose configuration
├── Caddyfile             # Caddy web server configuration
└── build.sh              # Build and deploy script
```

## Adding Content

1. Create a Markdown file in the appropriate `categories/<topic>/` directory.
2. Add an entry to `documents.json`:

```json
{
  "category": "CUDA",
  "title": "My New Document",
  "file": "categories/cuda/my-new-document.md"
}
```

3. Rebuild and restart the container (see below), or simply refresh the browser if running locally.

## Deployment

DocBro runs inside a Docker container using [Caddy](https://caddyserver.com/) as the web server.

### Build and run

```bash
./build.sh
```

This stops any existing container, rebuilds the image, and starts a new container on port **8765**.

### Manual steps

```bash
docker build --no-cache -t docbro:1.0 .
docker compose up -d docbro
```

Once running, open `http://localhost:8765` in a browser.

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md) for details.
