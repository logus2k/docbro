import { TextLayer, OutputScale } from '../libraries/pdf.js/pdf.min.mjs';

export class PdfRenderer {

    constructor({ contentContainer, selectionMode }) {
        this.contentContainer = contentContainer;
        this.selectionMode = selectionMode;
        this._pdfDoc = null;
        this._pdfContainer = null;
        this._pdfPageDivs = [];
        this._pdfOverlayEntries = [];
        this._pdfResizeObserver = null;
        this._renderVersion = 0;

        // Lazy rendering state
        this._intersectionObserver = null;
        this._unloadObserver = null;
        this._renderQueue = [];
        this._activeRenders = 0;
        this._maxConcurrentRenders = 2;
        this._intersectingIndices = new Set();
    }

    get pdfPageDivs() {
        return this._pdfPageDivs;
    }

    incrementRenderVersion() {
        this._renderVersion++;
        return this._renderVersion;
    }

    get renderVersion() {
        return this._renderVersion;
    }

    async setupPlaceholders(pdfDoc, container) {
        const numPages = pdfDoc.numPages;
        const scale = 1.5;
        const renderVersion = this._renderVersion;

        this.cleanup();
        this._pdfDoc = pdfDoc;
        this._pdfContainer = container;

        const pages = await Promise.all(
            Array.from({ length: numPages }, (_, i) =>
                pdfDoc.getPage(i + 1).then(
                    page => this._renderVersion === renderVersion ? page : null,
                    e => {
                        if (this._renderVersion === renderVersion) {
                            console.warn(`Failed to get page ${i + 1}:`, e);
                        }
                        return null;
                    }
                )
            )
        );

        // A newer render started while we were fetching pages — discard results
        if (this._renderVersion !== renderVersion) return;

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

            // Lazy rendering state per page
            pageDiv._renderState = 'idle';
            pageDiv._pageRenderVersion = 0;
            pageDiv._pageIndex = i;

            container.appendChild(pageDiv);
            pageDivs.push(pageDiv);
        }

        this._pdfPageDivs = pageDivs;

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

    // --- Lazy rendering system ---

    startLazyRendering(pdfDoc, container) {
        const renderVersion = this._renderVersion;
        const pageDivs = this._pdfPageDivs;

        this._disconnectObservers();

        // Render observer: trigger rendering when pages approach viewport
        this._intersectionObserver = new IntersectionObserver((entries) => {
            if (this._renderVersion !== renderVersion) return;

            for (const entry of entries) {
                const pageDiv = entry.target;
                if (entry.isIntersecting) {
                    this._intersectingIndices.add(pageDiv._pageIndex);
                    if (pageDiv._renderState === 'idle' || pageDiv._renderState === 'unloaded') {
                        this._enqueueRender(pageDiv);
                    }
                } else {
                    this._intersectingIndices.delete(pageDiv._pageIndex);
                }
            }

            this._processRenderQueue(pdfDoc, container, renderVersion);
        }, {
            root: container,
            rootMargin: '200% 0px'
        });

        // Unload observer: reclaim memory when pages are far from viewport
        this._unloadObserver = new IntersectionObserver((entries) => {
            if (this._renderVersion !== renderVersion) return;

            for (const entry of entries) {
                if (!entry.isIntersecting && entry.target._renderState === 'rendered') {
                    this._unloadPage(entry.target);
                }
            }
        }, {
            root: container,
            rootMargin: '500% 0px'
        });

        for (const pageDiv of pageDivs) {
            this._intersectionObserver.observe(pageDiv);
            this._unloadObserver.observe(pageDiv);
        }
    }

    _enqueueRender(pageDiv) {
        if (pageDiv._renderState === 'rendering') return;
        if (this._renderQueue.includes(pageDiv)) return;
        this._renderQueue.push(pageDiv);
    }

    _processRenderQueue(pdfDoc, container, renderVersion) {
        while (this._activeRenders < this._maxConcurrentRenders && this._renderQueue.length > 0) {
            if (this._renderVersion !== renderVersion) return;

            // Sort: pages closest to current scroll center first
            const scrollCenter = container.scrollTop + container.clientHeight / 2;
            this._renderQueue.sort((a, b) => {
                const aDist = Math.abs(a.offsetTop + a.offsetHeight / 2 - scrollCenter);
                const bDist = Math.abs(b.offsetTop + b.offsetHeight / 2 - scrollCenter);
                return aDist - bDist;
            });

            const pageDiv = this._renderQueue.shift();

            // Skip if no longer eligible
            if (pageDiv._renderState === 'rendered' || pageDiv._renderState === 'rendering') continue;
            if (!pageDiv._pdfPage) continue;

            this._activeRenders++;
            pageDiv._renderState = 'rendering';

            this._renderSinglePage(pageDiv, pdfDoc, container, renderVersion).then(() => {
                this._activeRenders--;
                this._processRenderQueue(pdfDoc, container, renderVersion);
            });
        }
    }

    async _renderSinglePage(pageDiv, pdfDoc, container, renderVersion) {
        const page = pageDiv._pdfPage;
        const viewport = pageDiv._pdfViewport;
        const outputScale = pageDiv._pdfOutputScale;
        const pageRenderVersion = ++pageDiv._pageRenderVersion;
        const pageDivs = this._pdfPageDivs;

        if (!page || !viewport || !outputScale) {
            pageDiv._renderState = 'idle';
            return;
        }

        // --- Canvas render (kept in DOM, no JPEG conversion) ---
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
            console.warn(`Failed to render page ${pageDiv._pageIndex + 1}:`, e);
            pageDiv._renderState = pageDiv._renderState === 'rendering' ? 'idle' : pageDiv._renderState;
            return;
        }

        // Staleness checks
        if (this._renderVersion !== renderVersion) return;
        if (pageDiv._pageRenderVersion !== pageRenderVersion) return;
        if (pageDiv._renderState !== 'rendering') return;

        pageDiv.appendChild(canvas);
        pageDiv.style.aspectRatio = '';

        // --- Text layer ---
        try {
            const textContent = await page.getTextContent();
            if (this._renderVersion !== renderVersion || pageDiv._pageRenderVersion !== pageRenderVersion) return;

            const displayedWidth = pageDiv.clientWidth || viewport.width;
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

            if (this._renderVersion !== renderVersion || pageDiv._pageRenderVersion !== pageRenderVersion) return;

            if (this.selectionMode) {
                this.selectionMode.registerPage(pageDiv, textContent, textScale, textViewport);
            }
        } catch (e) {
            console.warn(`Failed to render text layer for page ${pageDiv._pageIndex + 1}:`, e);
        }

        // --- Context menu (attach once) ---
        if (!pageDiv._hasContextMenu) {
            pageDiv.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                this._showPageContextMenu(e.clientX, e.clientY, pageDiv);
            });
            pageDiv._hasContextMenu = true;
        }

        // --- Annotation overlay (links) ---
        try {
            const annotations = await page.getAnnotations();
            if (this._renderVersion !== renderVersion || pageDiv._pageRenderVersion !== pageRenderVersion) return;

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
            console.warn(`Failed to process annotations for page ${pageDiv._pageIndex + 1}:`, e);
        }

        pageDiv._renderState = 'rendered';
    }

    _unloadPage(pageDiv) {
        if (pageDiv._renderState !== 'rendered') return;

        // Cancel any lingering async work for this page
        pageDiv._pageRenderVersion++;

        // Remove canvas and release GPU memory
        const canvas = pageDiv.querySelector('canvas');
        if (canvas) {
            canvas.width = 0;
            canvas.height = 0;
            canvas.remove();
        }

        // Remove text layer
        const textLayer = pageDiv.querySelector('.textLayer');
        if (textLayer) textLayer.remove();

        // Remove annotation layer
        const annotLayer = pageDiv.querySelector('.annotationLayer');
        if (annotLayer) {
            this._pdfOverlayEntries = this._pdfOverlayEntries.filter(e => e.div !== annotLayer);
            annotLayer.remove();
        }

        // Unregister from selection mode
        if (this.selectionMode) {
            this.selectionMode.unregisterPage(pageDiv);
        }

        // Restore aspect-ratio placeholder
        if (pageDiv._pdfViewport) {
            const vp = pageDiv._pdfViewport;
            pageDiv.style.aspectRatio = `${vp.width} / ${vp.height}`;
        }

        pageDiv._renderState = 'unloaded';
    }

    _disconnectObservers() {
        if (this._intersectionObserver) {
            this._intersectionObserver.disconnect();
            this._intersectionObserver = null;
        }
        if (this._unloadObserver) {
            this._unloadObserver.disconnect();
            this._unloadObserver = null;
        }
        this._renderQueue = [];
        this._activeRenders = 0;
        this._intersectingIndices.clear();
    }

    // --- Context menu ---

    async _showPageContextMenu(x, y, pageDiv) {
        const page = pageDiv._pdfPage;
        const viewport = pageDiv._pdfViewport;
        const outputScale = pageDiv._pdfOutputScale;
        if (!page || !viewport || !outputScale) return;

        // Reuse in-DOM canvas if the page is currently rendered
        const existingCanvas = pageDiv.querySelector('canvas');
        if (existingCanvas && pageDiv._renderState === 'rendered') {
            this._showContextMenu(x, y, existingCanvas);
            return;
        }

        // Otherwise render to a temporary canvas
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
        this._showContextMenu(x, y, canvas);
    }

    _showContextMenu(x, y, canvas) {
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

    // --- Cleanup ---

    cleanup() {
        this._disconnectObservers();

        if (this._pdfResizeObserver) {
            this._pdfResizeObserver.disconnect();
            this._pdfResizeObserver = null;
        }
        for (const pageDiv of this._pdfPageDivs) {
            // Revoke any remaining blob URLs (from pre-migration renders)
            const img = pageDiv.querySelector('img');
            if (img && img.src.startsWith('blob:')) {
                URL.revokeObjectURL(img.src);
            }
            // Release canvas GPU memory
            const canvas = pageDiv.querySelector('canvas');
            if (canvas) {
                canvas.width = 0;
                canvas.height = 0;
            }
        }
        this._pdfOverlayEntries = [];
        this._pdfPageDivs = [];
        this._pdfDoc = null;
        this._pdfContainer = null;
    }
}
