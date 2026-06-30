# How to Publish to DocBro

This guide describes every step required to publish a **new document** (PDF or
Markdown) or a **new video / audio** clip to DocBro, and how to make the change
go live.

> **The single most important thing to know:** DocBro is a static site whose
> content is **baked into the Docker image at build time**. In
> [`docker-compose.yml`](../docker-compose.yml) the bind-mounts for
> `documents.json` and `categories/` are **commented out**. That means editing
> files on disk does **not** change the running site. Every publish requires a
> **rebuild + restart** of the container (see
> [Step 4 — Rebuild & go live](#step-4--rebuild--go-live)).

---

## How content is structured

- **Catalog:** [`documents.json`](../documents.json) is the master index. Every
  document that appears in the navigation tree is one entry in the `documents`
  array.
- **Content files:** live under `categories/<category-folder>/`.
  - Documents: `*.pdf` or `*.md`
  - Videos: `categories/<category-folder>/videos/*.mp4`
  - Audio: `categories/<category-folder>/audio/*.m4a`
- **Categories** shown in the tree are derived automatically from the
  `category` field of each document entry — there is **no separate category
  list to edit** (the `categories` object at the top of `documents.json` is only
  used for the special `noted` intro and should be left alone).

A `documents.json` entry looks like this:

```json
{
    "name": "Domain Adaptation",
    "category": "ISCTE - Advanced Techniques for Language Models",
    "location": "categories/reports/atlm/atlm_ma1.pdf"
}
```

| Field      | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| `name`     | Label shown in the navigation tree (the leaf node).                     |
| `category` | Tree branch the document is grouped under. Identical strings group together. |
| `location` | Path to the file, **relative to the site root** (the repo root).        |

---

## Publishing a NEW DOCUMENT (PDF or Markdown)

### Step 1 — Add the content file

Place the file under the appropriate category folder:

```
categories/<category-folder>/<your-file>.pdf      # or .md
```

- Course reports live under `categories/reports/<course>/` (e.g. `taap`, `cgad`,
  `atlm`).
- Use a short, lowercase, descriptive filename. Keep related files together
  (one folder per course / topic).

### Step 2 — Register it in `documents.json`

Add one object to the `documents` array. Match the conventions of the
surrounding entries:

- **`category`** — reuse the exact existing string to add to an existing branch,
  or introduce a new one. Course categories follow the pattern
  `"ISCTE - <Full Course Name>"`.
- **Ordering** — entries are grouped on screen by `category`. Keep each
  category's entries **contiguous**, and place new categories in the same
  alphabetical order the file already uses (e.g. `ISCTE - Advanced Concepts…`
  comes before `ISCTE - Advanced Techniques…` comes before `ISCTE - Deep
  Learning…`).
- **`name`** — a concise topic label, consistent with siblings (e.g.
  `"Project Report"`, `"Domain Adaptation"`).

Example — adding a three-part report as a new category:

```json
{
    "name": "Domain Adaptation",
    "category": "ISCTE - Advanced Techniques for Language Models",
    "location": "categories/reports/atlm/atlm_ma1.pdf"
},
{
    "name": "RLAIF and DPO Alignment",
    "category": "ISCTE - Advanced Techniques for Language Models",
    "location": "categories/reports/atlm/atlm_ma2.pdf"
},
{
    "name": "Agentic RAG System",
    "category": "ISCTE - Advanced Techniques for Language Models",
    "location": "categories/reports/atlm/atlm_ma3.pdf"
}
```

> ⚠️ **Strict JSON** — no trailing comma after the **last** element of the
> array, and a comma **between** every element. A broken catalog stops the whole
> tree from loading.

### Step 3 — Validate the catalog

```bash
python3 -c "import json; d=json.load(open('documents.json')); print('OK', len(d['documents']), 'documents')"
```

If this prints `OK` and a count, the JSON is well-formed. If it raises, fix the
syntax before continuing.

### Step 4 — Rebuild & go live

See [Step 4 — Rebuild & go live](#step-4--rebuild--go-live) below (shared by
documents and videos).

---

## Publishing a NEW VIDEO (or AUDIO)

Videos are **not** catalog entries. A video is an `.mp4` file that is **embedded
inside a Markdown document** which is itself in the catalog.

### Step 1 — Add the media file

```
categories/<category-folder>/videos/<your-video>.mp4
categories/<category-folder>/audio/<your-clip>.m4a    # audio, same idea
```

### Step 2 — Embed it in a Markdown document

In the `.md` document where the video should appear, add the embed block. Use
the exact wrapper used elsewhere in the site:

```html
<div class="embedded-video">
    <video controls>
        <source src="https://logus2k.com/docbro/categories/<category-folder>/videos/<your-video>.mp4" type="video/mp4">
    </video>
</div>
```

> **Note the `src` is an absolute production URL**
> (`https://logus2k.com/docbro/...`), not a relative path. This is the pattern
> used by all existing videos — the media is served from the production host.
> Make sure the file is also deployed there for the embed to play in production.

If the Markdown document is brand new, also register it in `documents.json`
exactly as in [Publishing a new document](#publishing-a-new-document-pdf-or-markdown).
If you are only adding a video to an **existing** document, no `documents.json`
change is needed.

### Step 3 — Rebuild & go live

Same as below.

---

## Step 4 — Rebuild & go live

Because content is baked into the image, apply changes by rebuilding and
recreating the container:

```bash
cd /home/logus/env/assets/docbro

# 1. Rebuild the image with the new content
docker compose --profile default build docbro

# 2. Recreate the running container from the new image
docker compose --profile default up -d docbro
```

The site runs on **port 8765** (`http://localhost:8765`).

---

## Step 5 — Verify it is live

Confirm the running container actually serves the new content (not the previous
baked image):

```bash
# Container is up
docker ps --filter name=docbro --format '{{.Status}}'

# Site responds
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/

# The LIVE catalog contains your new entry (search a unique token)
curl -s http://localhost:8765/documents.json | grep -c "atlm"

# Each new file is reachable (expect 200)
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/categories/reports/atlm/atlm_ma1.pdf
```

Expected: container `Up`, site `200`, the catalog grep returns the number of new
entries you added, and every file returns `200`. If the grep returns `0`, the
rebuild/restart did not take effect — repeat [Step 4](#step-4--rebuild--go-live).

Finally, open `http://localhost:8765/` in a browser and confirm the new
category/document appears in the tree and that PDFs render and videos play.

---

## Quick checklist

**New document**
- [ ] File copied to `categories/<cat>/<file>.pdf|.md`
- [ ] Entry added to `documents.json` (`name`, `category`, `location`), correct ordering, valid JSON
- [ ] `python3 -c "import json; json.load(open('documents.json'))"` passes
- [ ] `docker compose --profile default build docbro && docker compose --profile default up -d docbro`
- [ ] Verified live: catalog grep + file returns `200`

**New video / audio**
- [ ] Media copied to `categories/<cat>/videos/<file>.mp4` (or `audio/…m4a`)
- [ ] `<div class="embedded-video">` block added to the target `.md` (absolute `https://logus2k.com/docbro/...` src)
- [ ] If the host doc is new: also registered in `documents.json`
- [ ] Media also deployed to the production host for production playback
- [ ] Rebuilt, restarted, and verified live

---

## Notes & gotchas

- **Baked image, not live volume.** The compose bind-mounts are commented out;
  on-disk edits are invisible until you rebuild. This is the #1 cause of "I
  changed it but nothing happened".
- **Categories are implicit.** They come from the `category` strings — no
  separate registry. A typo in `category` silently creates a new branch.
- **JSON is strict.** A stray/missing comma breaks the entire tree.
- **Git.** Commits and pushes are performed manually by the maintainer; this
  guide stops at "verified live locally".
