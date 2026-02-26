export class PdfTocSync {

    constructor() {
        this._scrollHandler = null;
    }

    async extractHeaders(pdfDoc, globalIndex, configHeaders) {
        // Try to get the embedded PDF outline (bookmarks)
        let outline = null;
        try {
            outline = await pdfDoc.getOutline();
        } catch (e) {
            console.warn('Could not read PDF outline:', e);
        }

        if (outline && outline.length > 0) {
            console.log(`PDF doc-${globalIndex}: found ${outline.length} outline entries`);
            return await this._flattenOutline(pdfDoc, outline, globalIndex, 1);
        }

        console.log(`PDF doc-${globalIndex}: no embedded outline found, trying text-based extraction`);

        // Fall back to parsing visible TOC text from page content
        const textHeaders = await this._extractTocFromText(pdfDoc, globalIndex);
        if (textHeaders.length > 0) {
            console.log(`PDF doc-${globalIndex}: extracted ${textHeaders.length} headers from text`);
            return textHeaders;
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

    async _extractTocFromText(pdfDoc, globalIndex) {
        const numPages = pdfDoc.numPages;
        const maxScanPages = Math.min(numPages, 10);
        const results = [];
        let foundToc = false;
        let missCount = 0; // consecutive lines without a page number
        let tocFinished = false;

        for (let p = 1; p <= maxScanPages && !tocFinished; p++) {
            let page;
            try { page = await pdfDoc.getPage(p); } catch { continue; }

            let textContent;
            try { textContent = await page.getTextContent(); } catch { continue; }

            const items = textContent.items.filter(it => it.str && it.str.trim());
            if (items.length === 0) continue;

            // Group text items into lines by Y coordinate (tolerance ±3)
            const lines = [];
            for (const item of items) {
                const y = Math.round(item.transform[5]);
                let line = lines.find(l => Math.abs(l.y - y) <= 3);
                if (!line) {
                    line = { y, items: [] };
                    lines.push(line);
                }
                line.items.push({ str: item.str.trim(), x: item.transform[4] });
            }
            // Sort lines top-to-bottom (higher Y = higher on page in PDF coords)
            lines.sort((a, b) => b.y - a.y);

            for (const line of lines) {
                // Sort items left-to-right within the line
                line.items.sort((a, b) => a.x - b.x);
                const fullText = line.items.map(it => it.str).join(' ');

                // Detect "Table of Contents" header
                if (!foundToc) {
                    if (/table\s+of\s+contents/i.test(fullText)) {
                        foundToc = true;
                    }
                    continue;
                }

                // Skip lines that are only dots/leaders
                if (/^[.\s_\-─…]+$/.test(fullText)) continue;

                // Try to parse a TOC entry: title text + page number
                // The rightmost item that's a pure number is the page reference
                const rightItem = line.items[line.items.length - 1];
                const pageNum = parseInt(rightItem.str, 10);
                if (isNaN(pageNum) || pageNum < 1 || pageNum > numPages) {
                    // Track consecutive misses to detect end of TOC section
                    if (results.length > 0) missCount++;
                    if (missCount >= 6) { tocFinished = true; break; }
                    continue;
                }
                missCount = 0; // reset on successful parse

                // Build title from items excluding trailing dots and page number
                const titleParts = [];
                for (let i = 0; i < line.items.length - 1; i++) {
                    const s = line.items[i].str;
                    if (!/^[.\s_\-─…]+$/.test(s)) titleParts.push(s);
                }
                // Also check if the second-to-last item is the page number
                // (sometimes dots are merged with surrounding text)
                let title = titleParts.join(' ').replace(/[.\s_\-─…]+$/, '').trim();
                if (!title) continue;

                // Determine heading level from numbering pattern
                let level = 1;
                const numMatch = title.match(/^(\d+(?:\.\d+)*)[.\s)]/);
                if (numMatch) {
                    level = numMatch[1].split('.').length;
                }

                results.push({
                    id: `doc-${globalIndex}-header-${results.length}`,
                    level,
                    text: title,
                    page: pageNum - 1 // Convert to 0-indexed
                });
            }
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
