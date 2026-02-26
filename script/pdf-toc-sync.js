export class PdfTocSync {

    constructor() {
        this._scrollHandler = null;
    }

    async extractHeaders(pdfDoc, globalIndex, configHeaders) {
        // Try to get the embedded PDF outline (bookmarks)
        const outline = await pdfDoc.getOutline();

        if (outline && outline.length > 0) {
            return await this._flattenOutline(pdfDoc, outline, globalIndex, 1);
        }

        // Fall back to manually defined configHeaders
        if (configHeaders && configHeaders.length > 0) {
            return configHeaders.map((h, i) => ({
                id: `doc-${globalIndex}-header-${i}`,
                level: h.level,
                text: h.text,
                page: h.page != null ? h.page - 1 : null // Convert 1-indexed to 0-indexed
            }));
        }

        return [];
    }

    async _flattenOutline(pdfDoc, items, globalIndex, level) {
        const results = [];
        for (const item of items) {
            let page = null;
            try {
                let dest = item.dest;
                if (typeof dest === 'string') {
                    dest = await pdfDoc.getDestination(dest);
                }
                if (Array.isArray(dest)) {
                    const ref = dest[0];
                    page = typeof ref === 'number' ? ref : await pdfDoc.getPageIndex(ref);
                }
            } catch (e) {
                // Ignore unresolvable destinations
            }

            results.push({ level, text: item.title, page });

            if (item.items && item.items.length > 0) {
                const children = await this._flattenOutline(pdfDoc, item.items, globalIndex, level + 1);
                results.push(...children);
            }
        }

        // Assign sequential IDs at the top level only
        if (level === 1) {
            results.forEach((h, i) => {
                h.id = `doc-${globalIndex}-header-${i}`;
            });
        }

        return results;
    }

    jumpToPage(headers, headerId, scrollContainer) {
        const header = headers.find(h => h.id === headerId);
        if (!header || header.page == null) return;

        const pages = scrollContainer.querySelectorAll('.pdf-page');
        const targetPage = pages[header.page];
        if (!targetPage) return;

        const containerRect = scrollContainer.getBoundingClientRect();
        const pageRect = targetPage.getBoundingClientRect();
        const offset = pageRect.top - containerRect.top + scrollContainer.scrollTop;
        scrollContainer.scrollTo({ top: Math.max(0, offset) });
    }

    setupScrollSync(headers, scrollContainer, treeInstance) {
        this.removeScrollSync(scrollContainer);

        if (!headers || headers.length === 0) return;

        // Build a sorted list of unique pages that have headers
        const headersByPage = headers.filter(h => h.page != null);
        if (headersByPage.length === 0) return;

        this._scrollHandler = () => {
            const pages = scrollContainer.querySelectorAll('.pdf-page');
            if (pages.length === 0) return;

            const containerTop = scrollContainer.getBoundingClientRect().top;
            let visiblePageIndex = 0;

            for (let i = 0; i < pages.length; i++) {
                const pageTop = pages[i].getBoundingClientRect().top - containerTop;
                if (pageTop <= 50) {
                    visiblePageIndex = i;
                } else {
                    break;
                }
            }

            // Find the last header whose page <= visiblePageIndex
            let currentHeaderId = null;
            for (const h of headersByPage) {
                if (h.page <= visiblePageIndex) {
                    currentHeaderId = h.id;
                } else {
                    break;
                }
            }

            if (currentHeaderId) {
                const headerNode = treeInstance.findKey(currentHeaderId);
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

    removeScrollSync(scrollContainer) {
        if (this._scrollHandler && scrollContainer) {
            scrollContainer.removeEventListener('scroll', this._scrollHandler);
            this._scrollHandler = null;
        }
    }
}
