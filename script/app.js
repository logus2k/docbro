// app.js — orchestrator
import { GlobalWorkerOptions } from '../libraries/pdf.js/pdf.min.mjs';
import { SelectionMode } from './selection-mode.js';
import { ContentPanel } from './content-panel.js';
import { PdfTocSync } from './pdf-toc-sync.js';
import { Lightbox } from './lightbox.js';
import { DocumentLoader } from './document-loader.js';
import { TabManager } from './tab-manager.js';
import { LayoutManager } from './layout-manager.js';
import { PdfRenderer } from './pdf-renderer.js';
import { PdfControls } from './pdf-controls.js';
import { TocManager } from './toc-manager.js';
import { DropHandler } from './drop-handler.js';

GlobalWorkerOptions.workerSrc = './libraries/pdf.js/pdf.worker.min.mjs';

class DocumentBrowser {

    constructor(configPath) {
        this.activeCategory = null;
        this.activeDocumentIndex = null;
        this.editMode = false;
        this._activationVersion = 0;
        this._isRendering = false;
        this._showingIntro = false;
        this.contentContainer = document.getElementById('contentContainer');

        this.loader = new DocumentLoader(configPath);
        this.pdfTocSync = new PdfTocSync();
        this.selectionMode = null;
        this.contentPanel = null;

        this.init();
    }

    get documents() { return this.loader.documents; }
    get categories() { return this.loader.categories; }

    isPdf(doc) {
        return this.loader.isPdf(doc);
    }

    async init() {
        try {
            // UI setup
            this.selectionMode = new SelectionMode(
                this.contentContainer,
                (text) => this.handleCopyToEditor(text)
            );
            this.contentPanel = new ContentPanel();
            document.addEventListener('content-panel-closed', () => {
                this.setEditMode(false);
            });

            this.lightbox = new Lightbox(this.contentContainer);

            // Data
            await this.loader.loadConfiguration();
            this.loader.extractCategories();

            // Modules that depend on data
            this.tabManager = new TabManager({
                tabsContainer: document.getElementById('tabsContainer'),
                onActivateDocument: (globalIndex, headerId, category, isIntro) => {
                    if (isIntro || (category && globalIndex === null)) {
                        this.activateCategoryIntro(category || this.activeCategory);
                    } else if (globalIndex !== null && globalIndex !== undefined) {
                        this.activateDocument(globalIndex, headerId);
                    }
                }
            });

            this.pdfRenderer = new PdfRenderer({
                contentContainer: this.contentContainer,
                selectionMode: this.selectionMode
            });

            this.layoutManager = new LayoutManager({
                contentContainer: this.contentContainer,
                getPdfPageDivs: () => this.pdfRenderer.pdfPageDivs,
                onZoomApplied: () => this.pdfRenderer.refreshResolution(),
                onStateChanged: () => this.pdfControls?.syncReadouts()
            });

            this.pdfControls = new PdfControls({
                contentPane: document.getElementById('contentPane'),
                layoutManager: this.layoutManager,
                pdfRenderer: this.pdfRenderer,
                getScrollContainer: () => this.contentContainer.querySelector('.pdf-content')
            });
            this.pdfControls.attach();

            this.tocManager = new TocManager({
                contentContainer: this.contentContainer,
                pdfTocSync: this.pdfTocSync,
                isPdf: (doc) => this.isPdf(doc),
                onActivateDocument: (docIndex, headerId, category, isIntro, isDoubleClick) => {
                    if (category !== undefined && category !== null) {
                        // Category click — show intro
                        this.activeCategory = category;
                        if (isDoubleClick && this._showingIntro) {
                            // Double-click on category while showing intro — make intro sticky
                            // (intro is virtual, no doc index to pin)
                        }
                        this.activateCategoryIntro(category);
                    } else if (docIndex !== null && docIndex !== undefined) {
                        // Double-click on a document — make it sticky
                        if (isDoubleClick) {
                            this.tabManager.state.makeSticky(docIndex);
                            const introInfo = this._introCacheSync(this.activeCategory);
                            this.tabManager.renderTabs(this.documents, this.activeCategory, docIndex, introInfo);
                            this.tabManager.updateActiveState(docIndex);
                            return;
                        }

                        if (headerId && this.activeDocumentIndex === docIndex && !this._isRendering) {
                            this.tocManager.jumpToHeader(headerId, this.documents[docIndex]);
                        } else if (!headerId && this.activeDocumentIndex === docIndex && !this._isRendering) {
                            // Scroll to top WITHOUT letting scroll-sync snap the
                            // selection back to the first section — keep the
                            // document (parent) node selected.
                            this.tocManager.scrollToTop(this.documents[docIndex]);
                            this.tocManager.setNodeActive(`doc-${docIndex}`);
                        } else {
                            this.activateDocument(docIndex, headerId);
                        }
                    }
                }
            });

            this.layoutManager.initSplitPane();
            this.tocManager.buildTree(this.categories, this.documents);

            // Drag & drop local files (PDF / Markdown / video) to view them.
            this.dropHandler = new DropHandler({
                overlay: document.getElementById('dropOverlay'),
                onFiles: (files) => this.openLocalFiles(files)
            });
            this.dropHandler.attach();
            // Chat panel disabled (pending rework)
            // this.chatPanel = new ChatPanel(document.getElementById('rightPane'));
            // this.chatService = new ChatService(this.chatPanel);
            // this.chatService.connect().catch(err => {
            //     console.error('Chat service connection failed:', err);
            // });
            this.setupHashChangeListener();

            // Navigate to initial document
            const hashParams = this.parseHash();
            if (hashParams.category && this.categories.includes(hashParams.category)) {
                if (hashParams.tab !== null) {
                    await this.navigateToDocument(hashParams.category, hashParams.tab);
                } else {
                    await this.activateCategoryIntro(hashParams.category);
                }
            } else if (this.categories.length > 0) {
                await this.activateCategoryIntro(this.categories[0]);
            }
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize document browser');
        }
    }


    setEditMode(enabled) {
        this.editMode = enabled;

        if (enabled) {
            this.selectionMode.activate();
            this.contentPanel.open();
        } else {
            this.selectionMode.deactivate();
            this.contentPanel.close();
        }
    }

    handleCopyToEditor(text) {
        if (this.contentPanel) {
            this.contentPanel.appendText(text);
        }
    }

    // --- Local (dropped) files ---

    async openLocalFiles(files) {
        let firstIndex = null;
        for (const file of files) {
            const idx = this.registerLocalFile(file);
            if (idx !== null && firstIndex === null) firstIndex = idx;
        }
        if (firstIndex !== null) {
            await this.activateDocument(firstIndex);
        }
    }

    // Classify a dropped file, wrap it as a temporary document + TOC node.
    // Returns its globalIndex, or null if the type is unsupported.
    registerLocalFile(file) {
        const name = file.name || 'Untitled';
        const ext = (name.split('.').pop() || '').toLowerCase();
        const mime = file.type || '';

        let type = null;
        if (ext === 'pdf' || mime === 'application/pdf') type = 'pdf';
        else if (['md', 'markdown', 'txt'].includes(ext) || mime === 'text/markdown' || mime === 'text/plain') type = 'md';
        else if (['mp4', 'webm', 'ogg', 'ogv', 'mov', 'm4v'].includes(ext) || mime.startsWith('video/')) type = 'video';

        if (!type) {
            console.warn('Unsupported dropped file:', name, mime);
            return null;
        }

        const blobUrl = URL.createObjectURL(file);
        const doc = this.loader.registerLocalDocument({ name, type, blobUrl, mime });
        this.tocManager.addLocalDocument(doc.globalIndex, name);
        return doc.globalIndex;
    }

    // --- Navigation ---

    parseHash() {
        const hash = window.location.hash.slice(1);
        const params = {};
        if (!hash) return params;

        hash.split('&').forEach(part => {
            const [key, value] = part.split('=');
            if (key && value) {
                params[decodeURIComponent(key)] = decodeURIComponent(value);
            }
        });

        if (params.tab !== undefined) {
            const asNumber = parseInt(params.tab, 10);
            params.tab = isNaN(asNumber) ? params.tab : asNumber;
        } else {
            params.tab = null;
        }
        return params;
    }

    updateHash(category, docName) {
        const hash = `#category=${encodeURIComponent(category)}&tab=${encodeURIComponent(docName)}`;
        if (window.location.hash !== hash) {
            window.history.replaceState(null, '', hash);
        }
    }

    setupHashChangeListener() {
        window.addEventListener('hashchange', async () => {
            const hashParams = this.parseHash();
            if (hashParams.category && hashParams.tab !== null) {
                await this.navigateToDocument(hashParams.category, hashParams.tab);
            }
        });
    }

    async navigateToDocument(category, tabNameOrIndex) {
        let doc;
        if (typeof tabNameOrIndex === 'number') {
            const categoryDocs = this.documents.filter(d => d.category === category);
            doc = categoryDocs[tabNameOrIndex];
        } else {
            doc = this.documents.find(d => d.category === category && d.name === tabNameOrIndex);
        }
        if (doc) {
            await this.activateDocument(doc.globalIndex);
        }
    }

    // --- Document activation & rendering ---

    async activateDocument(globalIndex, headerId = null) {
        const doc = this.documents[globalIndex];
        if (!doc) return;

        const activationVersion = ++this._activationVersion;
        this._showingIntro = false;

        const isNewDocument = this.activeDocumentIndex !== globalIndex || this._isRendering;
        const categoryChanged = this.activeCategory !== doc.category;

        if (categoryChanged) {
            this.activeCategory = doc.category;
        }

        // Ensure the intro cache is populated for this category (non-blocking if already cached).
        if (!(doc.category in this.loader._introCache)) {
            await this.loader.loadCategoryIntro(doc.category);
        }
        if (this._activationVersion !== activationVersion) return;

        const introInfo = this._introCacheSync(doc.category);
        this.tabManager.renderTabs(this.documents, this.activeCategory, globalIndex, introInfo);
        this.tabManager.updateActiveState(globalIndex);

        if (!doc.loaded || (this.isPdf(doc) && !doc.pdfDoc)) {
            await this.loader.loadDocument(globalIndex);
        }

        // A newer activation started while loading — bail out
        if (this._activationVersion !== activationVersion) return;

        if (isNewDocument) {
            this.activeDocumentIndex = globalIndex;
            this.pdfRenderer.incrementRenderVersion();
            this._isRendering = true;

            try {
                await this.renderDocument(globalIndex);
                if (this._activationVersion !== activationVersion) return;
                this._isRendering = false;
                // Local (dropped) docs aren't in the catalog, so don't write a
                // deep-link hash that would 404 on reload.
                if (!doc.isLocal) this.updateHash(doc.category, doc.name);
                await this.tocManager.extractAndUpdateHeaders(globalIndex, doc);
                if (this._activationVersion !== activationVersion) return;
                this.tocManager.setupScrollSync(doc, () => this.documents[this.activeDocumentIndex]);
            } catch (e) {
                if (this._activationVersion === activationVersion) {
                    this._isRendering = false;
                }
                console.error('Error activating document:', e);
            }
        }

        if (this._activationVersion !== activationVersion) return;

        if (headerId) {
            this.tocManager.jumpToHeader(headerId, doc);
        }

        if (!headerId) {
            this.tocManager.setNodeActive(`doc-${globalIndex}`);
        }
    }

    async _getIntroInfo(category) {
        const intro = await this.loader.loadCategoryIntro(category);
        if (intro) {
            return { category, label: category };
        }
        return null;
    }

    /** Synchronous version — only works if the intro was already fetched and cached. */
    _introCacheSync(category) {
        const cached = this.loader._introCache[category];
        return cached ? { category, label: category } : null;
    }

    async activateCategoryIntro(category) {
        this.activeCategory = category;
        this._showingIntro = true;
        this.activeDocumentIndex = null;

        const introInfo = await this._getIntroInfo(category);
        this.tabManager.renderTabs(this.documents, category, null, introInfo);

        if (introInfo) {
            this.tabManager.setIntroActive();

            // Render the intro content
            const intro = await this.loader.loadCategoryIntro(category);
            if (intro && !intro.error) {
                this.pdfRenderer.cleanup();
                this.layoutManager.disconnectLayoutObserver();
                this.contentContainer.innerHTML = '';

                const contentDiv = document.createElement('div');
                contentDiv.className = 'document-content active md-doc';

                const innerDiv = document.createElement('div');
                innerDiv.className = 'document-content-inner';
                innerDiv.innerHTML = intro.content;

                contentDiv.appendChild(innerDiv);
                this.contentContainer.appendChild(contentDiv);

                this.contentContainer.querySelectorAll('pre code').forEach((block) => {
                    if (!block.classList.contains('language-mermaid')) {
                        hljs.highlightElement(block);
                    }
                });
                this.renderMermaidBlocks(innerDiv);
                this.renderDrawioBlocks(innerDiv);
                this.setupCodeCopyButtons();
            }
        } else {
            // No intro — load the first document's content as a fallback, but
            // keep the CATEGORY (root) node selected (see below).
            const firstDoc = this.documents.find(d => d.category === category);
            if (firstDoc) {
                this._showingIntro = false;
                await this.activateDocument(firstDoc.globalIndex);
            }
        }

        // Keep the clicked category (root) node selected in the tree — otherwise
        // the no-intro fallback's activateDocument() leaves the first document
        // node selected instead of the category the user clicked.
        this.tocManager.setNodeActive(`cat-${category}`);
    }

    async renderDocument(globalIndex) {
        const doc = this.documents[globalIndex];

        if (this.selectionMode) {
            this.selectionMode.reset();
        }

        this.pdfRenderer.cleanup();
        this.layoutManager.disconnectLayoutObserver();
        this.pdfControls?.showForDocument(false);
        this.contentContainer.innerHTML = '';

        const contentDiv = document.createElement('div');
        contentDiv.className = this.isPdf(doc)
            ? 'document-content active pdf-doc'
            : 'document-content active md-doc';
        contentDiv.setAttribute('data-doc-index', globalIndex);

        const innerDiv = document.createElement('div');
        innerDiv.className = 'document-content-inner';

        if (this.isPdf(doc)) {
            innerDiv.classList.add('pdf-content');
        }

        if (doc.error) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = 'Loading error';
            innerDiv.appendChild(errorDiv);
        } else if (this.isPdf(doc) && doc.pdfDoc) {
            contentDiv.appendChild(innerDiv);
            this.contentContainer.appendChild(contentDiv);
            await this.pdfRenderer.setupPlaceholders(doc.pdfDoc, innerDiv);
            // Bail if a newer document activation superseded this one
            if (this.activeDocumentIndex !== globalIndex) return;
            this.layoutManager.initForDocument();
            if (this.editMode && this.selectionMode) {
                this.selectionMode.activate();
            }
            this.pdfRenderer.startLazyRendering(doc.pdfDoc, innerDiv);
            this.pdfControls.showForDocument(true);
            return;
        } else {
            innerDiv.innerHTML = doc.content;
        }

        contentDiv.appendChild(innerDiv);
        this.contentContainer.appendChild(contentDiv);

        if (doc.headers && doc.headers.length > 0) {
            const headers = Array.from(innerDiv.querySelectorAll('h1, h2, h3, h4, h5, h6'));
            headers.forEach((h, i) => {
                if (i < doc.headers.length) {
                    h.id = doc.headers[i].id;
                }
            });
        }

        this.contentContainer.querySelectorAll('pre code').forEach((block) => {
            if (!block.classList.contains('language-mermaid')) {
                hljs.highlightElement(block);
            }
        });

        this.renderMermaidBlocks(innerDiv);
        this.renderDrawioBlocks(innerDiv);
        this.setupCodeCopyButtons();
    }

    renderDrawioBlocks(container) {
        const drawioBlocks = container.querySelectorAll('.mxgraph');
        if (drawioBlocks.length === 0 || typeof GraphViewer === 'undefined') return;

        // Use createViewerForElement directly (instead of processElements)
        // so we get a reference to each viewer instance for live resizing.
        const viewers = [];
        for (const block of drawioBlocks) {
            block.innerText = '';
            GraphViewer.createViewerForElement(block, (viewer) => {
                viewers.push(viewer);
            });
        }

        // Call fitGraph on each viewer when the container resizes
        const ro = new ResizeObserver(() => {
            for (const v of viewers) {
                if (v.fitGraph) v.fitGraph();
            }
        });
        ro.observe(container);
    }

    renderMermaidBlocks(container) {
        const mermaidBlocks = container.querySelectorAll('pre code.language-mermaid');
        if (mermaidBlocks.length === 0 || typeof mermaid === 'undefined') return;

        mermaidBlocks.forEach((codeBlock) => {
            const pre = codeBlock.parentElement;
            const div = document.createElement('div');
            div.className = 'mermaid';
            div.textContent = codeBlock.textContent;
            pre.replaceWith(div);
        });

        mermaid.run({ nodes: container.querySelectorAll('.mermaid') });
    }

    setupCodeCopyButtons() {
        this.contentContainer.querySelectorAll('pre').forEach((pre) => {
            if (pre.parentElement.classList.contains('code-block-wrapper')) return;

            const wrapper = document.createElement('div');
            wrapper.className = 'code-block-wrapper';
            pre.parentNode.insertBefore(wrapper, pre);
            wrapper.appendChild(pre);

            const btn = document.createElement('button');
            btn.className = 'code-copy-btn';
            btn.textContent = 'Copy';
            btn.addEventListener('click', async () => {
                const code = pre.querySelector('code')?.textContent || pre.textContent;
                await navigator.clipboard.writeText(code);
                btn.textContent = 'Copied';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy';
                    btn.classList.remove('copied');
                }, 2000);
            });
            wrapper.appendChild(btn);
        });
    }

    showError(message) {
        this.contentContainer.innerHTML = `
            <div class="error-message">
                ${message}
            </div>
        `;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new DocumentBrowser('documents.json');
});
