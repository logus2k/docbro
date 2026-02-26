// app.js
import { getDocument, GlobalWorkerOptions, TextLayer, OutputScale } from '../libraries/pdf.js/pdf.min.mjs';
import { SelectionMode } from './selection-mode.js';
import { ContentPanel } from './content-panel.js';
import { PdfTocSync } from './pdf-toc-sync.js';
GlobalWorkerOptions.workerSrc = './libraries/pdf.js/pdf.worker.min.mjs';

class DocumentBrowser {

    constructor(configPath) {
        this.configPath = configPath;
        this.documents = [];
        this.categories = [];
        this.activeCategory = null;
        this.activeDocumentIndex = null; // Global index in this.documents
        this.treeInstance = null;
        this.tocPane = document.getElementById('tocPane');
        this.contentPane = document.getElementById('contentPane');
        this.tabsContainer = document.getElementById('tabsContainer');
        this.contentContainer = document.getElementById('contentContainer');
        this.scrollSyncEnabled = true;
        this.closedTabs = new Set();
        this.pageLayoutMode = 'single';
        this.pdfZoom = 1;
        this._layoutResizeObserver = null;
        this.selectionMode = null;
        this.contentPanel = null;
        this.pdfTocSync = new PdfTocSync();
        this._renderVersion = 0;
        this.editMode = false;

        // PDF rendering state
        this._pdfResizeObserver = null;
        this._pdfOverlayEntries = [];
        this._pdfPageDivs = [];
        this._pdfDoc = null;
        this._pdfContainer = null;

        this.init();
    }

    isPdf(doc) {
        return doc.location.toLowerCase().endsWith('.pdf');
    }

    async init() {
        try {
            this.createLightbox();
            this.setupSettings();
            this.setupModeToggle();
            this.selectionMode = new SelectionMode(
                this.contentContainer,
                (text) => this.handleCopyToEditor(text)
            );
            this.contentPanel = new ContentPanel();
            document.addEventListener('content-panel-closed', () => {
                this.setEditMode(false);
            });
            await this.loadConfiguration();
            this.extractCategories();
            this.initSplitPane();
            this.buildTree();
            this.setupImageClickHandlers();
            this.setupHashChangeListener();
            
            // Check if there's a hash in the URL
            const hashParams = this.parseHash();
            if (hashParams.category && this.categories.includes(hashParams.category)) {
                this.activeCategory = hashParams.category;
                this.renderTabs();
                if (hashParams.tab !== null) {
                    await this.navigateToDocument(hashParams.category, hashParams.tab);
                } else {
                    // Activate first document in category
                    const firstDoc = this.documents.find(d => d.category === hashParams.category);
                    if (firstDoc) {
                        await this.activateDocument(firstDoc.globalIndex);
                    }
                }
            } else if (this.categories.length > 0) {
                // Default to first category and first document
                this.activeCategory = this.categories[0];
                this.renderTabs();
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

    initSplitPane() {
        const tocPane = document.getElementById('tocPane');
        const rightPane = document.getElementById('rightPane');
        this._splitInstance = Split(['#tocPane', '#contentPane', '#rightPane'], {
            sizes: [20, 75, 5],
            minSize: [5, 5, 5],
            gutterSize: 6,
            cursor: 'col-resize',
            onDragEnd: () => {
                this._tocPixelWidth = tocPane.getBoundingClientRect().width;
                this._rightPanePixelWidth = rightPane.getBoundingClientRect().width;
            }
        });
        // Store initial pixel widths
        this._tocPixelWidth = tocPane.getBoundingClientRect().width;
        this._rightPanePixelWidth = rightPane.getBoundingClientRect().width;

        // On window resize, maintain both TOC and right pane pixel widths
        window.addEventListener('resize', () => {
            if (!this._splitInstance) return;
            const container = tocPane.parentElement;
            const containerWidth = container.getBoundingClientRect().width;
            if (containerWidth <= 0) return;
            const tocPct = (this._tocPixelWidth / containerWidth) * 100;
            const rightPct = (this._rightPanePixelWidth / containerWidth) * 100;
            const clampedToc = Math.min(Math.max(tocPct, 1), 90);
            const clampedRight = Math.min(Math.max(rightPct, 1), 90);
            const contentPct = 100 - clampedToc - clampedRight;
            this._splitInstance.setSizes([clampedToc, Math.max(contentPct, 1), clampedRight]);
        });
    }

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
                this.pageLayoutMode = e.target.value;
                this.applyLayoutAndZoom();
            });
        });

        const zoomSlider = document.getElementById('pdfZoomSlider');
        const zoomValue = document.getElementById('pdfZoomValue');
        zoomSlider.addEventListener('input', (e) => {
            // When user moves slider, switch to custom mode
            if (this.pageLayoutMode !== 'custom') {
                this.pageLayoutMode = 'custom';
                const customRadio = settingsMenu.querySelector('input[name="pageLayout"][value="custom"]');
                if (customRadio) customRadio.checked = true;
                this.applyPageLayout();
                this._setupLayoutResizeObserver(); // disconnects the observer
            }
            this.pdfZoom = parseInt(e.target.value, 10) / 100;
            zoomValue.textContent = e.target.value + '%';
            this.applyZoom();
        });
    }

    applyPageLayout() {
        const pdfContent = this.contentContainer.querySelector('.pdf-content');
        if (!pdfContent) return;

        pdfContent.classList.remove('custom-layout', 'dual-page');
        if (this.pageLayoutMode === 'custom') {
            pdfContent.classList.add('custom-layout');
        } else if (this.pageLayoutMode === 'dual') {
            pdfContent.classList.add('dual-page');
        }
    }

    computeFitZoom(mode) {
        const pdfContent = this.contentContainer.querySelector('.pdf-content');
        if (!pdfContent) return 1;

        const style = getComputedStyle(pdfContent);
        const padLeft = parseFloat(style.paddingLeft) || 0;
        const padRight = parseFloat(style.paddingRight) || 0;
        const padTop = parseFloat(style.paddingTop) || 0;
        const padBottom = parseFloat(style.paddingBottom) || 0;
        const availableWidth = pdfContent.clientWidth - padLeft - padRight;
        const availableHeight = pdfContent.clientHeight - padTop - padBottom;

        // Use first page's aspect ratio (width/height) for height-fitting
        let pageAspect = 900 / 1165; // default ~letter proportions
        const firstDiv = this._pdfPageDivs[0];
        if (firstDiv && firstDiv._pdfViewport) {
            const vp = firstDiv._pdfViewport;
            pageAspect = vp.width / vp.height;
        }

        if (mode === 'single') {
            // Page width = 900 * zoom, page height = 900 * zoom / pageAspect
            const zoomByWidth = availableWidth / 900;
            const zoomByHeight = (availableHeight * pageAspect) / 900;
            return Math.min(zoomByWidth, zoomByHeight);
        }
        if (mode === 'dual') {
            // Each page width = 450 * zoom, height = 450 * zoom / pageAspect
            const zoomByWidth = (availableWidth - 6) / 900; // 2 × 450
            const zoomByHeight = (availableHeight * pageAspect) / 450;
            return Math.min(zoomByWidth, zoomByHeight);
        }
        return this.pdfZoom;
    }

    applyLayoutAndZoom() {
        const zoomSlider = document.getElementById('pdfZoomSlider');
        const zoomValue = document.getElementById('pdfZoomValue');

        // Apply layout class first so container dimensions are correct for zoom computation
        this.applyPageLayout();

        if (this.pageLayoutMode === 'single' || this.pageLayoutMode === 'dual') {
            this.pdfZoom = this.computeFitZoom(this.pageLayoutMode);
        }

        // Update slider and label to reflect current zoom
        if (zoomSlider) {
            const pct = Math.round(this.pdfZoom * 100);
            zoomSlider.value = pct;
            if (zoomValue) zoomValue.textContent = pct + '%';
        }

        this.applyZoom();
        this._setupLayoutResizeObserver();
    }

    _setupLayoutResizeObserver() {
        // Disconnect any existing observer
        if (this._layoutResizeObserver) {
            this._layoutResizeObserver.disconnect();
            this._layoutResizeObserver = null;
        }

        if (this.pageLayoutMode !== 'single' && this.pageLayoutMode !== 'dual') return;

        const pdfContent = this.contentContainer.querySelector('.pdf-content');
        if (!pdfContent) return;

        this._layoutResizeObserver = new ResizeObserver(() => {
            if (this.pageLayoutMode !== 'single' && this.pageLayoutMode !== 'dual') return;
            const newZoom = this.computeFitZoom(this.pageLayoutMode);
            // Only apply if zoom changed meaningfully (prevents scrollbar oscillation loop)
            if (Math.abs(newZoom - this.pdfZoom) < 0.005) return;
            this.pdfZoom = newZoom;
            this.applyZoom();

            const zoomSlider = document.getElementById('pdfZoomSlider');
            const zoomValue = document.getElementById('pdfZoomValue');
            if (zoomSlider) {
                const pct = Math.round(this.pdfZoom * 100);
                zoomSlider.value = pct;
                if (zoomValue) zoomValue.textContent = pct + '%';
            }
        });
        this._layoutResizeObserver.observe(pdfContent);
    }

    applyZoom() {
        const pdfContent = this.contentContainer.querySelector('.pdf-content');
        if (!pdfContent) return;
        pdfContent.style.setProperty('--pdf-zoom', this.pdfZoom);
    }

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

        // Read mode is active by default
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

    createLightbox() {
        const lightbox = document.createElement('div');
        lightbox.id = 'imageLightbox';
        lightbox.className = 'lightbox';
        lightbox.innerHTML = `
            <div class="lightbox-content">
                <span class="lightbox-close">&times;</span>
                <img class="lightbox-image" src="" alt="">
            </div>
        `;
        document.body.appendChild(lightbox);

        this.lightbox = lightbox;
        this.lightboxImage = lightbox.querySelector('.lightbox-image');
        this.lightboxClose = lightbox.querySelector('.lightbox-close');

        this.lightboxClose.addEventListener('click', () => this.closeLightbox());
        this.lightbox.addEventListener('click', (e) => {
            if (e.target === this.lightbox) {
                this.closeLightbox();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.lightbox.classList.contains('active')) {
                this.closeLightbox();
            }
        });
    }

    setupImageClickHandlers() {
        this.contentContainer.addEventListener('click', (e) => {
            if (e.target.tagName === 'IMG') {
                this.openLightbox(e.target.src, e.target.alt);
            }
        });
    }

    openLightbox(src, alt) {
        this.lightboxImage.src = src;
        this.lightboxImage.alt = alt || '';
        this.lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    closeLightbox() {
        this.lightbox.classList.remove('active');
        document.body.style.overflow = '';
    }

    async loadConfiguration() {
        try {
            const response = await fetch(this.configPath);
            if (!response.ok) {
                throw new Error(`Failed to load configuration: ${response.status}`);
            }
            const config = await response.json();
            this.documents = (config.documents || []).map((doc, index) => ({
                ...doc,
                globalIndex: index,
                loaded: false,
                content: '',
                error: false,
                headers: null,
                configHeaders: doc.headers || null,
                pdfDoc: null
            }));
        } catch (error) {
            console.error('Configuration loading error:', error);
            throw error;
        }
    }

    async loadDocument(globalIndex) {
        const doc = this.documents[globalIndex];
        if (doc.loaded) return true;

        try {
            if (this.isPdf(doc)) {
                const pdfDoc = await getDocument({
                    url: doc.location,
                    disableRange: true,
                    disableStream: true,
                }).promise;
                doc.pdfDoc = pdfDoc;
                doc.error = false;
                doc.loaded = true;
                return true;
            }

            const response = await fetch(doc.location);
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            const markdown = await response.text();
            doc.content = this.renderMarkdown(markdown);
            doc.error = false;
            doc.loaded = true;
            return true;
        } catch (error) {
            console.error(`Error loading document ${doc.name}:`, error);
            doc.content = '';
            doc.error = true;
            doc.loaded = true;
            return false;
        }
    }

    extractCategories() {
        const categorySet = new Set();
        this.documents.forEach(doc => {
            if (doc.category) {
                categorySet.add(doc.category);
            }
        });
        this.categories = Array.from(categorySet);
    }

    renderMarkdown(markdown) {
        const mathExpressions = [];
        let mathIndex = 0;

        markdown = markdown.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => {
            const placeholder = `MATH_DISPLAY_${mathIndex}`;
            mathExpressions.push({ type: 'display', math: math.trim(), placeholder });
            mathIndex++;
            return placeholder;
        });

        markdown = markdown.replace(/\$([^\$\n]+?)\$/g, (match, math) => {
            const placeholder = `MATH_INLINE_${mathIndex}`;
            mathExpressions.push({ type: 'inline', math: math.trim(), placeholder });
            mathIndex++;
            return placeholder;
        });

        let html = marked.parse(markdown);

        mathExpressions.forEach(item => {
            try {
                const rendered = katex.renderToString(item.math, { 
                    displayMode: item.type === 'display', 
                    throwOnError: false 
                });
                html = html.replace(item.placeholder, rendered);
            } catch (e) {
                console.error('KaTeX rendering error:', e);
                html = html.replace(item.placeholder, item.type === 'display' ? `$$${item.math}$$` : `$${item.math}$`);
            }
        });

        return html;
    }

    buildTree() {
        const treeData = this.categories.map(category => {
            const categoryDocs = this.documents.filter(d => d.category === category);
            return {
                title: category,
                key: `cat-${category}`,
                folder: true,
                expanded: false,
                children: categoryDocs.map(doc => ({
                    title: doc.name,
                    key: `doc-${doc.globalIndex}`,
                    folder: true,
                    expanded: false,
                    children: []
                }))
            };
        });

        this.treeInstance = new mar10.Wunderbaum({
            element: document.getElementById('tocTree'),
            source: treeData,
            selectMode: 'single',
            checkbox: false,
            icon: true,
            iconMap: {
                folder: "fa-solid fa-folder",
                folderOpen: "fa-solid fa-folder-open",
                doc: "fa-regular fa-file",
                expanderExpanded: "fa-solid fa-chevron-down",
                expanderCollapsed: "fa-solid fa-chevron-right",
            },
            render: (e) => {
                const node = e.node;
                const row = e.nodeElem;
                if (row) {
                    // Determine type from key prefix
                    const key = node.key || '';
                    let type = '';
                    if (key.startsWith('cat-')) type = 'category';
                    else if (key.match(/^doc-\d+$/)) type = 'document';
                    else if (key.match(/^doc-\d+-header-/)) type = 'header';
                    row.setAttribute('data-type', type);
                }
            },
            click: (e) => {
                const node = e.node;
                const key = node.key || '';
                
                // For expander clicks, let Wunderbaum handle if it recognizes them
                // (dynamically added nodes may not report targetType correctly)
                if (e.targetType === 'expander') {
                    return;
                }
                
                // Header node - toggle expand and jump to header
                if (key.match(/^doc-\d+-header-/)) {
                    const match = key.match(/^doc-(\d+)-header-/);
                    if (match) {
                        const docIndex = parseInt(match[1], 10);
                        if (this.activeDocumentIndex !== docIndex) {
                            this.activateDocument(docIndex, key);
                        } else {
                            this.jumpToHeader(key);
                        }
                    }
                    // Toggle expand if node has children
                    if (node.children && node.children.length > 0) {
                        node.setExpanded(!node.isExpanded());
                    }
                    // Activate the clicked node (return false prevents Wunderbaum's default)
                    node.setActive(true, { noEvents: true });
                    return false;
                }
                
                // Category node - expand and load first document
                if (key.startsWith('cat-')) {
                    const category = key.replace('cat-', '');
                    node.setExpanded(!node.isExpanded());
                    if (this.activeCategory !== category) {
                        this.activeCategory = category;
                        this.renderTabs();
                        const firstDoc = this.documents.find(d => d.category === category);
                        if (firstDoc) {
                            this.activateDocument(firstDoc.globalIndex);
                        }
                    }
                    return false;
                }
                
                // Document node - load document and expand
                if (key.match(/^doc-\d+$/)) {
                    const docIndex = parseInt(key.replace('doc-', ''), 10);
                    node.setExpanded(!node.isExpanded());
                    this.activateDocument(docIndex);
                    return false;
                }
            }
        });
    }

    renderTabs() {
        this.tabsContainer.innerHTML = '';
        const categoryDocs = this.documents.filter(d => d.category === this.activeCategory);
        const openDocs = categoryDocs.filter(d => !this.closedTabs.has(d.globalIndex));

        openDocs.forEach((doc) => {
            const tab = document.createElement('div');
            tab.className = 'tab';
            tab.textContent = doc.name;
            tab.title = doc.name;
            tab.setAttribute('data-doc-index', doc.globalIndex);
            tab.addEventListener('click', () => this.activateDocument(doc.globalIndex));

            // Add close button unless this is the last open tab
            if (openDocs.length > 1) {
                const closeBtn = document.createElement('span');
                closeBtn.className = 'tab-close-btn';
                closeBtn.textContent = '\u00d7';
                closeBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.closeTab(doc.globalIndex);
                });
                tab.appendChild(closeBtn);
            }

            this.tabsContainer.appendChild(tab);
        });
    }

    updateTabsActiveState(globalIndex) {
        const tabs = this.tabsContainer.querySelectorAll('.tab');
        tabs.forEach(tab => {
            const tabDocIndex = parseInt(tab.getAttribute('data-doc-index'), 10);
            if (tabDocIndex === globalIndex) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
    }

    closeTab(globalIndex) {
        const categoryDocs = this.documents.filter(d => d.category === this.activeCategory);
        const openDocs = categoryDocs.filter(d => !this.closedTabs.has(d.globalIndex));

        // Don't close the last open tab
        if (openDocs.length <= 1) return;

        this.closedTabs.add(globalIndex);

        if (this.activeDocumentIndex === globalIndex) {
            // Find next open tab to the right in the full category order
            const closedPos = categoryDocs.findIndex(d => d.globalIndex === globalIndex);
            let nextDoc = null;

            for (let i = closedPos + 1; i < categoryDocs.length; i++) {
                if (!this.closedTabs.has(categoryDocs[i].globalIndex)) {
                    nextDoc = categoryDocs[i];
                    break;
                }
            }
            // If none to the right, pick nearest to the left
            if (!nextDoc) {
                for (let i = closedPos - 1; i >= 0; i--) {
                    if (!this.closedTabs.has(categoryDocs[i].globalIndex)) {
                        nextDoc = categoryDocs[i];
                        break;
                    }
                }
            }

            this.renderTabs();
            if (nextDoc) {
                this.activateDocument(nextDoc.globalIndex);
            }
        } else {
            this.renderTabs();
            this.updateTabsActiveState(this.activeDocumentIndex);
        }
    }

    async activateDocument(globalIndex, headerId = null) {
        const doc = this.documents[globalIndex];
        if (!doc) return;

        // Reopen tab if it was closed
        const wasClosed = this.closedTabs.has(globalIndex);
        if (wasClosed) {
            this.closedTabs.delete(globalIndex);
        }

        const isNewDocument = this.activeDocumentIndex !== globalIndex;

        // Check if category changed
        if (this.activeCategory !== doc.category) {
            this.activeCategory = doc.category;
            this.renderTabs();
        } else if (wasClosed) {
            this.renderTabs();
        }

        // Update active tab
        this.updateTabsActiveState(globalIndex);

        // Load document if needed
        if (!doc.loaded) {
            await this.loadDocument(globalIndex);
        }

        // Only re-render and extract headers if document changed
        if (isNewDocument) {
            this.activeDocumentIndex = globalIndex;
            this._renderVersion++;

            try {
                // Render document content
                await this.renderDocument(globalIndex);

                // Update hash
                this.updateHash(doc.category, doc.name);

                // Extract headers and update tree
                await this.extractAndUpdateHeaders(globalIndex);

                // Setup scroll sync for this document
                this.setupScrollSync();
            } catch (e) {
                console.error('Error activating document:', e);
            }
        }
        
        // Jump to header if specified
        if (headerId) {
            this.jumpToHeader(headerId);
        }
        
        // Set the document node as active in the tree
        const docNode = this.treeInstance.findKey(`doc-${globalIndex}`);
        if (docNode) {
            try {
                docNode.setActive(true, { noEvents: true });
            } catch (e) { /* ignore */ }
        }
    }

    async renderDocument(globalIndex) {
        const doc = this.documents[globalIndex];

        // Reset selection mode when document changes
        if (this.selectionMode) {
            this.selectionMode.reset();
        }

        // Clean up PDF rendering state before clearing DOM
        this._cleanupPdf();

        // Clear content container
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
            // Create placeholders immediately, then render pages progressively
            await this.setupPdfPlaceholders(doc.pdfDoc, innerDiv);
            this.applyLayoutAndZoom();
            // Re-activate selection mode overlays if edit mode is on
            if (this.editMode && this.selectionMode) {
                this.selectionMode.activate();
            }
            // Render actual page content in the background (don't await)
            this.renderPdfPagesProgressively(doc.pdfDoc, innerDiv);
            return;
        } else {
            innerDiv.innerHTML = doc.content;
        }

        contentDiv.appendChild(innerDiv);
        this.contentContainer.appendChild(contentDiv);

        // Re-apply header IDs if already extracted
        if (doc.headers && doc.headers.length > 0) {
            const headers = Array.from(innerDiv.querySelectorAll('h1, h2, h3, h4, h5, h6'));
            headers.forEach((h, i) => {
                if (i < doc.headers.length) {
                    h.id = doc.headers[i].id;
                }
            });
        }

        // Apply syntax highlighting (skip mermaid blocks)
        this.contentContainer.querySelectorAll('pre code').forEach((block) => {
            if (!block.classList.contains('language-mermaid')) {
                hljs.highlightElement(block);
            }
        });

        // Render mermaid diagrams
        this.renderMermaidBlocks(innerDiv);

        // Add copy buttons
        this.setupCodeCopyButtons();
    }

    async setupPdfPlaceholders(pdfDoc, container) {
        const numPages = pdfDoc.numPages;
        const scale = 1.5;

        // Clean up previous state
        this._cleanupPdf();
        this._pdfDoc = pdfDoc;
        this._pdfContainer = container;

        // Batch-fetch all page metadata (lightweight, no GPU memory)
        const pages = await Promise.all(
            Array.from({ length: numPages }, (_, i) =>
                pdfDoc.getPage(i + 1).catch(e => {
                    console.warn(`Failed to get page ${i + 1}:`, e);
                    return null;
                })
            )
        );

        // Create placeholder divs with correct aspect ratios
        const pageDivs = [];
        for (let i = 0; i < pages.length; i++) {
            const page = pages[i];
            const pageDiv = document.createElement('div');
            pageDiv.className = 'pdf-page';

            if (page) {
                const viewport = page.getViewport({ scale });
                pageDiv.style.aspectRatio = `${viewport.width} / ${viewport.height}`;
                pageDiv._pdfPage = page;
                pageDiv._pdfViewport = viewport;
                pageDiv._pdfOutputScale = new OutputScale();
            } else {
                pageDiv.style.aspectRatio = '8.5 / 11';
            }

            container.appendChild(pageDiv);
            pageDivs.push(pageDiv);
        }

        this._pdfPageDivs = pageDivs;

        // Set up overlay scaling for annotations
        if (this._pdfResizeObserver) this._pdfResizeObserver.disconnect();
        this._pdfResizeObserver = new ResizeObserver(() => {
            for (const entry of this._pdfOverlayEntries) {
                const pd = entry.div.parentElement;
                if (pd) {
                    const dw = pd.clientWidth;
                    if (dw > 0) {
                        entry.div.style.transform = `scale(${dw / entry.viewport.width})`;
                    }
                }
            }
        });
        this._pdfResizeObserver.observe(container);
    }

    async renderPdfPagesProgressively(pdfDoc, container) {
        const renderVersion = this._renderVersion;
        const pageDivs = this._pdfPageDivs;

        for (let i = 0; i < pageDivs.length; i++) {
            if (this._renderVersion !== renderVersion) return;

            const pageDiv = pageDivs[i];
            const page = pageDiv._pdfPage;
            if (!page) continue;

            const viewport = pageDiv._pdfViewport;
            const outputScale = pageDiv._pdfOutputScale;

            // Create a fresh canvas, render, convert to <img>, then destroy canvas
            const canvas = document.createElement('canvas');
            canvas.width = Math.floor(viewport.width * outputScale.sx);
            canvas.height = Math.floor(viewport.height * outputScale.sy);
            const ctx = canvas.getContext('2d');

            try {
                await page.render({
                    canvasContext: ctx,
                    viewport,
                    transform: outputScale.scaled ? [outputScale.sx, 0, 0, outputScale.sy, 0, 0] : null
                }).promise;
            } catch (e) {
                console.warn(`Failed to render page ${i + 1}:`, e);
            }
            if (this._renderVersion !== renderVersion) return;

            // Convert canvas to <img>
            const img = document.createElement('img');
            img.style.width = Math.floor(viewport.width) + 'px';
            img.style.height = Math.floor(viewport.height) + 'px';
            try {
                const blob = await new Promise((resolve, reject) => {
                    canvas.toBlob(b => b ? resolve(b) : reject(new Error('toBlob failed')), 'image/jpeg', 0.92);
                });
                img.src = URL.createObjectURL(blob);
            } catch (e) {
                console.warn(`Failed to convert page ${i + 1} to image:`, e);
            }
            // Free the canvas GPU buffer immediately
            canvas.width = 0;
            canvas.height = 0;

            pageDiv.appendChild(img);
            pageDiv.style.aspectRatio = ''; // let the img drive sizing now

            // Text layer
            try {
                const textContent = await page.getTextContent();
                const displayedWidth = pageDiv.clientWidth || img.getBoundingClientRect().width;
                const textScale = displayedWidth / page.getViewport({ scale: 1 }).width;
                const textViewport = page.getViewport({ scale: textScale });

                const textLayerDiv = document.createElement('div');
                textLayerDiv.className = 'textLayer';
                textLayerDiv.style.setProperty('--scale-factor', textScale);
                pageDiv.appendChild(textLayerDiv);

                const textLayer = new TextLayer({
                    textContentSource: textContent,
                    container: textLayerDiv,
                    viewport: textViewport
                });
                await textLayer.render();

                if (this.selectionMode) {
                    this.selectionMode.registerPage(pageDiv, textContent, textScale, textViewport);
                }
            } catch (e) {
                console.warn(`Failed to render text layer for page ${i + 1}:`, e);
            }

            // Context menu — re-render to a temporary canvas on demand
            pageDiv.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                this._showPdfPageContextMenu(e.clientX, e.clientY, pageDiv);
            });

            // Link overlay (annotations)
            try {
                const annotations = await page.getAnnotations();
                const linkAnnotations = annotations.filter(a => a.subtype === 'Link' && (a.dest || a.url));
                if (linkAnnotations.length > 0) {
                    const annotationDiv = document.createElement('div');
                    annotationDiv.className = 'annotationLayer';
                    annotationDiv.style.width = viewport.width + 'px';
                    annotationDiv.style.height = viewport.height + 'px';
                    pageDiv.appendChild(annotationDiv);

                    for (const annot of linkAnnotations) {
                        const [x1, y1, x2, y2] = viewport.convertToViewportRectangle(annot.rect);
                        const left = Math.min(x1, x2);
                        const top = Math.min(y1, y2);
                        const width = Math.abs(x2 - x1);
                        const height = Math.abs(y2 - y1);

                        const link = document.createElement('a');
                        link.style.position = 'absolute';
                        link.style.left = left + 'px';
                        link.style.top = top + 'px';
                        link.style.width = width + 'px';
                        link.style.height = height + 'px';

                        if (annot.url) {
                            link.href = annot.url;
                            link.target = '_blank';
                            link.rel = 'noopener noreferrer';
                        } else if (annot.dest) {
                            link.href = '#';
                            link.addEventListener('click', async (e) => {
                                e.preventDefault();
                                try {
                                    let dest = annot.dest;
                                    if (typeof dest === 'string') {
                                        dest = await pdfDoc.getDestination(dest);
                                    }
                                    if (!Array.isArray(dest)) return;
                                    const ref = dest[0];
                                    const pageIndex = typeof ref === 'number' ? ref : await pdfDoc.getPageIndex(ref);
                                    const targetDiv = pageDivs[pageIndex];
                                    if (targetDiv) {
                                        const containerRect = container.getBoundingClientRect();
                                        const targetRect = targetDiv.getBoundingClientRect();
                                        const offset = targetRect.top - containerRect.top + container.scrollTop;
                                        container.scrollTo({ top: offset, behavior: 'smooth' });
                                    }
                                } catch (err) {
                                    console.error('PDF link navigation error:', err);
                                }
                            });
                        }

                        annotationDiv.appendChild(link);
                    }

                    this._pdfOverlayEntries.push({ div: annotationDiv, viewport });
                }
            } catch (e) {
                console.warn(`Failed to process annotations for page ${i + 1}:`, e);
            }
        }
    }

    async _showPdfPageContextMenu(x, y, pageDiv) {
        const page = pageDiv._pdfPage;
        const viewport = pageDiv._pdfViewport;
        const outputScale = pageDiv._pdfOutputScale;
        if (!page || !viewport || !outputScale) return;

        // Re-render to a temporary canvas for copy/save
        const canvas = document.createElement('canvas');
        canvas.width = Math.floor(viewport.width * outputScale.sx);
        canvas.height = Math.floor(viewport.height * outputScale.sy);
        const ctx = canvas.getContext('2d');
        try {
            await page.render({
                canvasContext: ctx,
                viewport,
                transform: outputScale.scaled ? [outputScale.sx, 0, 0, outputScale.sy, 0, 0] : null
            }).promise;
        } catch (e) {
            console.warn('Failed to re-render page for context menu:', e);
            return;
        }
        this.showPdfContextMenu(x, y, canvas);
        // Canvas will be GC'd after the menu closes
    }

    _cleanupPdf() {
        if (this._pdfResizeObserver) {
            this._pdfResizeObserver.disconnect();
            this._pdfResizeObserver = null;
        }
        if (this._layoutResizeObserver) {
            this._layoutResizeObserver.disconnect();
            this._layoutResizeObserver = null;
        }
        // Revoke blob URLs to free memory
        for (const pageDiv of this._pdfPageDivs) {
            const img = pageDiv.querySelector('img');
            if (img && img.src.startsWith('blob:')) {
                URL.revokeObjectURL(img.src);
            }
        }
        this._pdfOverlayEntries = [];
        this._pdfPageDivs = [];
        this._pdfDoc = null;
        this._pdfContainer = null;
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

    showPdfContextMenu(x, y, canvas) {
        // Remove any existing menu
        document.querySelector('.pdf-context-menu')?.remove();

        const menu = document.createElement('div');
        menu.className = 'pdf-context-menu';
        menu.style.left = x + 'px';
        menu.style.top = y + 'px';

        const copyItem = document.createElement('div');
        copyItem.className = 'pdf-context-menu-item';
        copyItem.textContent = 'Copy page as image';
        copyItem.addEventListener('click', () => {
            menu.remove();
            canvas.toBlob(async (blob) => {
                try {
                    await navigator.clipboard.write([
                        new ClipboardItem({ 'image/png': blob })
                    ]);
                } catch (err) {
                    console.error('Copy failed:', err);
                }
            });
        });

        const saveItem = document.createElement('div');
        saveItem.className = 'pdf-context-menu-item';
        saveItem.textContent = 'Save page as image';
        saveItem.addEventListener('click', () => {
            menu.remove();
            const link = document.createElement('a');
            link.download = 'page.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        });

        menu.appendChild(copyItem);
        menu.appendChild(saveItem);
        document.body.appendChild(menu);

        // Close on click elsewhere or Escape
        const close = () => {
            menu.remove();
            document.removeEventListener('click', close);
            document.removeEventListener('keydown', onKey);
        };
        const onKey = (e) => { if (e.key === 'Escape') close(); };
        setTimeout(() => {
            document.addEventListener('click', close);
            document.addEventListener('keydown', onKey);
        }, 0);
    }

    async extractAndUpdateHeaders(globalIndex) {
        const doc = this.documents[globalIndex];

        // Skip if headers already extracted (null = not yet attempted)
        if (doc.headers !== null) {
            return;
        }

        if (this.isPdf(doc)) {
            if (!doc.pdfDoc) {
                doc.headers = [];
                return;
            }
            try {
                doc.headers = await this.pdfTocSync.extractHeaders(doc.pdfDoc, globalIndex, doc.configHeaders);
            } catch (e) {
                console.error('PDF header extraction error:', e);
                doc.headers = [];
            }
            if (doc.headers.length > 0) {
                const headerTree = this.buildHeaderTree(doc.headers, globalIndex);
                const docNode = this.treeInstance.findKey(`doc-${globalIndex}`);
                if (docNode) {
                    docNode.removeChildren();
                    docNode.addChildren(headerTree);
                }
            }
            return;
        }

        const contentInner = this.contentContainer.querySelector('.document-content-inner');
        if (!contentInner) return;

        const headers = Array.from(contentInner.querySelectorAll('h1, h2, h3, h4, h5, h6'));
        doc.headers = headers.map((h, i) => {
            const id = `doc-${globalIndex}-header-${i}`;
            h.id = id;
            return {
                id: id,
                level: parseInt(h.tagName.substring(1)),
                text: h.innerText
            };
        });

        const headerTree = this.buildHeaderTree(doc.headers, globalIndex);

        const docNode = this.treeInstance.findKey(`doc-${globalIndex}`);
        if (docNode) {
            docNode.removeChildren();
            if (headerTree.length > 0) {
                docNode.addChildren(headerTree);
            }
        }
    }

    buildHeaderTree(headers, docIndex) {
        const root = [];
        const stack = [{ level: 0, children: root }];

        headers.forEach(h => {
            const node = {
                title: h.text,
                key: h.id,
                children: []
            };

            while (stack.length > 1 && stack[stack.length - 1].level >= h.level) {
                stack.pop();
            }

            stack[stack.length - 1].children.push(node);
            stack.push({ level: h.level, children: node.children });
        });

        return root;
    }

    getScrollContainer() {
        const doc = this.documents[this.activeDocumentIndex];
        if (doc && !this.isPdf(doc)) {
            return this.contentContainer.querySelector('.document-content.md-doc');
        }
        return this.contentContainer.querySelector('.document-content-inner');
    }

    jumpToHeader(headerId) {
        this.scrollSyncEnabled = false;
        this.pdfTocSync.syncEnabled = false;

        const doc = this.documents[this.activeDocumentIndex];
        const scrollContainer = this.getScrollContainer();

        if (doc && this.isPdf(doc) && doc.headers) {
            this.pdfTocSync.jumpToPage(doc.headers, headerId, scrollContainer);
        } else {
            const header = document.getElementById(headerId);
            if (header && scrollContainer) {
                const containerRect = scrollContainer.getBoundingClientRect();
                const headerRect = header.getBoundingClientRect();
                const offset = headerRect.top - containerRect.top + scrollContainer.scrollTop;
                scrollContainer.scrollTo({ top: Math.max(0, offset - 15) });
            }
        }

        // Re-enable scroll sync after scroll settles (scrollTo is async)
        setTimeout(() => {
            this.scrollSyncEnabled = true;
            this.pdfTocSync.syncEnabled = true;
        }, 300);
    }

    setupScrollSync() {
        const scrollContainer = this.getScrollContainer();
        if (!scrollContainer) return;

        const doc = this.documents[this.activeDocumentIndex];

        // For PDFs, delegate to PdfTocSync
        if (doc && this.isPdf(doc)) {
            this.pdfTocSync.setupScrollSync(doc.headers, scrollContainer, this.treeInstance);
            return;
        }

        if (this._scrollHandler) {
            scrollContainer.removeEventListener('scroll', this._scrollHandler);
        }

        this._scrollHandler = () => {
            if (!this.scrollSyncEnabled) return;

            const doc = this.documents[this.activeDocumentIndex];
            if (!doc || !doc.headers || !doc.headers.length) return;

            const scrollTop = scrollContainer.scrollTop;
            const containerTop = scrollContainer.offsetTop;

            let currentHeaderId = null;
            for (const h of doc.headers) {
                const headerEl = document.getElementById(h.id);
                if (headerEl) {
                    const headerTop = headerEl.offsetTop - containerTop;
                    if (headerTop <= scrollTop + 50) {
                        currentHeaderId = h.id;
                    } else {
                        break;
                    }
                }
            }

            if (currentHeaderId) {
                const headerNode = this.treeInstance.findKey(currentHeaderId);
                if (headerNode && !headerNode.isActive()) {
                    try {
                        headerNode.visitParents((p) => {
                            if (!p.isExpanded()) {
                                p.setExpanded(true);
                            }
                        });
                        headerNode.setActive(true, { noEvents: true });
                    } catch (e) { /* ignore */ }
                }
            }
        };

        scrollContainer.addEventListener('scroll', this._scrollHandler);
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

// Initialize the document browser when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DocumentBrowser('documents.json');
});
