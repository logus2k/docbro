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
import { TocManager } from './toc-manager.js';
import { ChatPanel } from './chat-panel.js';
import { ChatService } from './chat-service.js';

GlobalWorkerOptions.workerSrc = './libraries/pdf.js/pdf.worker.min.mjs';

class DocumentBrowser {

    constructor(configPath) {
        this.activeCategory = null;
        this.activeDocumentIndex = null;
        this.editMode = false;
        this._activationVersion = 0;
        this._isRendering = false;
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
            this.setupSettings();
            this.setupModeToggle();

            // Data
            await this.loader.loadConfiguration();
            this.loader.extractCategories();

            // Modules that depend on data
            this.tabManager = new TabManager({
                tabsContainer: document.getElementById('tabsContainer'),
                onActivateDocument: (i) => this.activateDocument(i)
            });

            this.pdfRenderer = new PdfRenderer({
                contentContainer: this.contentContainer,
                selectionMode: this.selectionMode
            });

            this.layoutManager = new LayoutManager({
                contentContainer: this.contentContainer,
                getPdfPageDivs: () => this.pdfRenderer.pdfPageDivs
            });

            this.tocManager = new TocManager({
                contentContainer: this.contentContainer,
                pdfTocSync: this.pdfTocSync,
                isPdf: (doc) => this.isPdf(doc),
                onActivateDocument: (docIndex, headerId, category) => {
                    if (category !== undefined && category !== null) {
                        // Category click
                        if (this.activeCategory !== category) {
                            this.activeCategory = category;
                            this.tabManager.renderTabs(this.documents, this.activeCategory);
                            const firstDoc = this.documents.find(d => d.category === category);
                            if (firstDoc) {
                                this.activateDocument(firstDoc.globalIndex);
                            }
                        }
                    } else if (docIndex !== null && docIndex !== undefined) {
                        if (headerId && this.activeDocumentIndex === docIndex && !this._isRendering) {
                            this.tocManager.jumpToHeader(headerId, this.documents[docIndex]);
                        } else if (!headerId && this.activeDocumentIndex === docIndex && !this._isRendering) {
                            // Document root node clicked — scroll to top (cover/first page)
                            const sc = this.tocManager.getScrollContainer(this.documents[docIndex]);
                            if (sc) sc.scrollTo({ top: 0 });
                        } else {
                            this.activateDocument(docIndex, headerId);
                        }
                    }
                }
            });

            this.layoutManager.initSplitPane();
            this.tocManager.buildTree(this.categories, this.documents);
            this.chatPanel = new ChatPanel(document.getElementById('rightPane'));
            this.chatService = new ChatService(this.chatPanel);
            this.chatService.connect().catch(err => {
                console.error('Chat service connection failed:', err);
            });
            this.setupHashChangeListener();

            // Navigate to initial document
            const hashParams = this.parseHash();
            if (hashParams.category && this.categories.includes(hashParams.category)) {
                this.activeCategory = hashParams.category;
                this.tabManager.renderTabs(this.documents, this.activeCategory);
                if (hashParams.tab !== null) {
                    await this.navigateToDocument(hashParams.category, hashParams.tab);
                } else {
                    const firstDoc = this.documents.find(d => d.category === hashParams.category);
                    if (firstDoc) {
                        await this.activateDocument(firstDoc.globalIndex);
                    }
                }
            } else if (this.categories.length > 0) {
                this.activeCategory = this.categories[0];
                this.tabManager.renderTabs(this.documents, this.activeCategory);
                const firstDoc = this.documents.find(d => d.category === this.activeCategory);
                if (firstDoc) {
                    await this.activateDocument(firstDoc.globalIndex);
                }
            }
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize document browser');
        }
    }

    // --- Settings ---

    setupSettings() {
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsMenu = document.getElementById('settingsMenu');
        const layoutRadios = settingsMenu.querySelectorAll('input[name="pageLayout"]');

        settingsBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            settingsMenu.classList.toggle('active');
        });

        document.addEventListener('click', (e) => {
            if (!settingsMenu.contains(e.target) && e.target !== settingsBtn) {
                settingsMenu.classList.remove('active');
            }
        });

        layoutRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.layoutManager.pageLayoutMode = e.target.value;
                this.layoutManager.applyLayoutAndZoom();
            });
        });

        const zoomSlider = document.getElementById('pdfZoomSlider');
        const zoomValue = document.getElementById('pdfZoomValue');
        zoomSlider.addEventListener('input', (e) => {
            if (this.layoutManager.pageLayoutMode !== 'custom') {
                this.layoutManager.pageLayoutMode = 'custom';
                const customRadio = settingsMenu.querySelector('input[name="pageLayout"][value="custom"]');
                if (customRadio) customRadio.checked = true;
                this.layoutManager.applyPageLayout();
                this.layoutManager._setupLayoutResizeObserver();
            }
            this.layoutManager.pdfZoom = parseInt(e.target.value, 10) / 100;
            zoomValue.textContent = e.target.value + '%';
            this.layoutManager.applyZoom();
        });
    }

    // --- Mode toggle ---

    setupModeToggle() {
        const container = document.createElement('div');
        container.className = 'mode-toggle-container';

        const editBtn = document.createElement('button');
        editBtn.className = 'mode-toggle-btn';
        editBtn.id = 'editModeBtn';
        editBtn.title = 'Selection mode';
        editBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M14.236 1.76386C13.2123 0.740172 11.5525 0.740171 10.5289 1.76386L2.65722 9.63549C2.28304 10.0097 2.01623 10.4775 1.88467 10.99L1.01571 14.3755C0.971767 14.5467 1.02148 14.7284 1.14646 14.8534C1.27144 14.9783 1.45312 15.028 1.62432 14.9841L5.00978 14.1151C5.52234 13.9836 5.99015 13.7168 6.36433 13.3426L14.236 5.47097C15.2596 4.44728 15.2596 2.78755 14.236 1.76386ZM11.236 2.47097C11.8691 1.8378 12.8957 1.8378 13.5288 2.47097C14.162 3.10413 14.162 4.1307 13.5288 4.76386L12.75 5.54269L10.4571 3.24979L11.236 2.47097ZM9.75002 3.9569L12.0429 6.24979L5.65722 12.6355C5.40969 12.883 5.10023 13.0595 4.76117 13.1465L2.19447 13.8053L2.85327 11.2386C2.9403 10.8996 3.1168 10.5901 3.36433 10.3426L9.75002 3.9569Z"/></svg>';
        editBtn.addEventListener('click', () => this.setEditMode(true));

        const readBtn = document.createElement('button');
        readBtn.className = 'mode-toggle-btn';
        readBtn.id = 'readModeBtn';
        readBtn.title = 'Read mode';
        readBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M2.5 2C1.67157 2 1 2.67157 1 3.5V12.5C1 13.3284 1.67157 14 2.5 14H6C6.8178 14 7.54389 13.6073 8 13.0002C8.45612 13.6073 9.1822 14 10 14H13.5C14.3284 14 15 13.3284 15 12.5V3.5C15 2.67157 14.3284 2 13.5 2H10C9.1822 2 8.45612 2.39267 8 2.99976C7.54389 2.39267 6.8178 2 6 2H2.5ZM7.5 4.5V11.5C7.5 12.3284 6.82843 13 6 13H2.5C2.22386 13 2 12.7761 2 12.5V3.5C2 3.22386 2.22386 3 2.5 3H6C6.82843 3 7.5 3.67157 7.5 4.5ZM8.5 11.5V4.5C8.5 3.67157 9.17157 3 10 3H13.5C13.7761 3 14 3.22386 14 3.5V12.5C14 12.7761 13.7761 13 13.5 13H10C9.17157 13 8.5 12.3284 8.5 11.5Z"/></svg>';
        readBtn.addEventListener('click', () => this.setEditMode(false));

        readBtn.classList.add('active');

        container.appendChild(readBtn);
        container.appendChild(editBtn);
        document.body.appendChild(container);
    }

    setEditMode(enabled) {
        this.editMode = enabled;
        const editBtn = document.getElementById('editModeBtn');
        const readBtn = document.getElementById('readModeBtn');

        if (enabled) {
            editBtn.classList.add('active');
            readBtn.classList.remove('active');
            this.selectionMode.activate();
            this.contentPanel.open();
        } else {
            readBtn.classList.add('active');
            editBtn.classList.remove('active');
            this.selectionMode.deactivate();
            this.contentPanel.close();
        }
    }

    handleCopyToEditor(text) {
        if (this.contentPanel) {
            this.contentPanel.appendText(text);
        }
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

        const wasClosed = this.tabManager.closedTabs.has(globalIndex);
        if (wasClosed) {
            this.tabManager.closedTabs.delete(globalIndex);
        }

        const isNewDocument = this.activeDocumentIndex !== globalIndex || this._isRendering;

        if (this.activeCategory !== doc.category) {
            this.activeCategory = doc.category;
            this.tabManager.renderTabs(this.documents, this.activeCategory);
        } else if (wasClosed) {
            this.tabManager.renderTabs(this.documents, this.activeCategory);
        }

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
                this.updateHash(doc.category, doc.name);
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

    async renderDocument(globalIndex) {
        const doc = this.documents[globalIndex];

        if (this.selectionMode) {
            this.selectionMode.reset();
        }

        this.pdfRenderer.cleanup();
        this.layoutManager.disconnectLayoutObserver();
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
            this.layoutManager.applyLayoutAndZoom();
            if (this.editMode && this.selectionMode) {
                this.selectionMode.activate();
            }
            this.pdfRenderer.startLazyRendering(doc.pdfDoc, innerDiv);
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
