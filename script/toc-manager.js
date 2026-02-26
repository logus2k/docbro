export class TocManager {

    constructor({ contentContainer, pdfTocSync, onActivateDocument, isPdf }) {
        this.contentContainer = contentContainer;
        this.pdfTocSync = pdfTocSync;
        this.onActivateDocument = onActivateDocument;
        this.isPdf = isPdf;
        this.treeInstance = null;
        this.scrollSyncEnabled = true;
        this._scrollHandler = null;
    }

    buildTree(categories, documents) {
        const treeData = categories.map(category => {
            const categoryDocs = documents.filter(d => d.category === category);
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

                if (e.targetType === 'expander') {
                    return;
                }

                // Header node
                if (key.match(/^doc-\d+-header-/)) {
                    const match = key.match(/^doc-(\d+)-header-/);
                    if (match) {
                        const docIndex = parseInt(match[1], 10);
                        this.onActivateDocument(docIndex, key);
                    }
                    if (node.children && node.children.length > 0) {
                        node.setExpanded(!node.isExpanded());
                    }
                    node.setActive(true, { noEvents: true });
                    return false;
                }

                // Category node
                if (key.startsWith('cat-')) {
                    const category = key.replace('cat-', '');
                    node.setExpanded(!node.isExpanded());
                    this.onActivateDocument(null, null, category);
                    return false;
                }

                // Document node
                if (key.match(/^doc-\d+$/)) {
                    const docIndex = parseInt(key.replace('doc-', ''), 10);
                    node.setExpanded(!node.isExpanded());
                    this.onActivateDocument(docIndex);
                    return false;
                }
            }
        });
    }

    buildHeaderTree(headers) {
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

    async extractAndUpdateHeaders(globalIndex, doc) {
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
                const headerTree = this.buildHeaderTree(doc.headers);
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

        const headerTree = this.buildHeaderTree(doc.headers);

        const docNode = this.treeInstance.findKey(`doc-${globalIndex}`);
        if (docNode) {
            docNode.removeChildren();
            if (headerTree.length > 0) {
                docNode.addChildren(headerTree);
            }
        }
    }

    getScrollContainer(doc) {
        if (doc && !this.isPdf(doc)) {
            return this.contentContainer.querySelector('.document-content.md-doc');
        }
        return this.contentContainer.querySelector('.document-content-inner');
    }

    jumpToHeader(headerId, doc) {
        this.scrollSyncEnabled = false;
        this.pdfTocSync.syncEnabled = false;

        const scrollContainer = this.getScrollContainer(doc);

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

        setTimeout(() => {
            this.scrollSyncEnabled = true;
            this.pdfTocSync.syncEnabled = true;
        }, 300);
    }

    setupScrollSync(doc, getActiveDoc) {
        const scrollContainer = this.getScrollContainer(doc);
        if (!scrollContainer) return;

        if (doc && this.isPdf(doc)) {
            this.pdfTocSync.setupScrollSync(doc.headers, scrollContainer, this.treeInstance);
            return;
        }

        if (this._scrollHandler) {
            scrollContainer.removeEventListener('scroll', this._scrollHandler);
        }

        this._scrollHandler = () => {
            if (!this.scrollSyncEnabled) return;

            const currentDoc = getActiveDoc();
            if (!currentDoc || !currentDoc.headers || !currentDoc.headers.length) return;

            const scrollTop = scrollContainer.scrollTop;
            const containerTop = scrollContainer.offsetTop;

            let currentHeaderId = null;
            for (const h of currentDoc.headers) {
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

    setNodeActive(key) {
        const node = this.treeInstance.findKey(key);
        if (node) {
            try {
                node.setActive(true, { noEvents: true });
            } catch (e) { /* ignore */ }
        }
    }
}
