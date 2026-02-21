// app.js

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

        this.init();
    }

    async init() {
        try {
            this.createLightbox();
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
        Split(['#tocPane', '#contentPane'], {
            sizes: [20, 80],
            minSize: [5, 400],
            gutterSize: 6,
            cursor: 'col-resize'
        });
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
                headers: []
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
            // Render document content
            this.renderDocument(globalIndex);
            this.activeDocumentIndex = globalIndex;
            
            // Update hash
            this.updateHash(doc.category, doc.name);
            
            // Extract headers and update tree
            this.extractAndUpdateHeaders(globalIndex);
            
            // Setup scroll sync for this document
            this.setupScrollSync();
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

    renderDocument(globalIndex) {
        const doc = this.documents[globalIndex];
        
        // Clear content container
        this.contentContainer.innerHTML = '';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'document-content active';
        contentDiv.setAttribute('data-doc-index', globalIndex);
        
        const innerDiv = document.createElement('div');
        innerDiv.className = 'document-content-inner';
        
        if (doc.error) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = 'Loading error';
            innerDiv.appendChild(errorDiv);
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

    extractAndUpdateHeaders(globalIndex) {
        const doc = this.documents[globalIndex];
        
        // Skip if headers already extracted
        if (doc.headers && doc.headers.length > 0) {
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

    jumpToHeader(headerId) {
        this.scrollSyncEnabled = false;
        
        const header = document.getElementById(headerId);
        if (header) {
            const contentInner = this.contentContainer.querySelector('.document-content-inner');
            if (contentInner) {
                const containerRect = contentInner.getBoundingClientRect();
                const headerRect = header.getBoundingClientRect();
                const offset = headerRect.top - containerRect.top + contentInner.scrollTop;
                contentInner.scrollTo({ top: Math.max(0, offset - 15) });
            }
        }
        
        this.scrollSyncEnabled = true;
    }

    setupScrollSync() {
        const contentInner = this.contentContainer.querySelector('.document-content-inner');
        if (!contentInner) return;

        if (this._scrollHandler) {
            contentInner.removeEventListener('scroll', this._scrollHandler);
        }

        this._scrollHandler = () => {
            if (!this.scrollSyncEnabled) return;
            
            const doc = this.documents[this.activeDocumentIndex];
            if (!doc || !doc.headers || !doc.headers.length) return;

            const scrollTop = contentInner.scrollTop;
            const containerTop = contentInner.offsetTop;
            
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
                        // Expand all parent nodes so the header is visible
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

        contentInner.addEventListener('scroll', this._scrollHandler);
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
