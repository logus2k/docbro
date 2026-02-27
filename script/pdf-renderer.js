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
                        // Only log errors for the current render, not stale ones
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

    async renderPagesProgressively(pdfDoc, container) {
        const renderVersion = this._renderVersion;
        const pageDivs = this._pdfPageDivs;

        for (let i = 0; i < pageDivs.length; i++) {
            if (this._renderVersion !== renderVersion) return;

            const pageDiv = pageDivs[i];
            const page = pageDiv._pdfPage;
            if (!page) continue;

            const viewport = pageDiv._pdfViewport;
            const outputScale = pageDiv._pdfOutputScale;

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
            canvas.width = 0;
            canvas.height = 0;

            pageDiv.appendChild(img);
            pageDiv.style.aspectRatio = '';

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

            // Context menu
            pageDiv.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                this._showPageContextMenu(e.clientX, e.clientY, pageDiv);
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

    async _showPageContextMenu(x, y, pageDiv) {
        const page = pageDiv._pdfPage;
        const viewport = pageDiv._pdfViewport;
        const outputScale = pageDiv._pdfOutputScale;
        if (!page || !viewport || !outputScale) return;

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

    cleanup() {
        if (this._pdfResizeObserver) {
            this._pdfResizeObserver.disconnect();
            this._pdfResizeObserver = null;
        }
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
}
