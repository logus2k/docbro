export class LayoutManager {

    constructor({ contentContainer, getPdfPageDivs, onZoomApplied, onStateChanged }) {
        this.contentContainer = contentContainer;
        this.getPdfPageDivs = getPdfPageDivs;
        this.onZoomApplied = onZoomApplied;
        this.onStateChanged = onStateChanged;

        // PDF view state. `columns` = pages per row (1–4). `pdfZoom` is a
        // multiplier where 1 == fit the configured columns across the width.
        // `fitMode` drives how zoom recomputes on resize / rotation.
        this.columns = 1;
        this.pdfZoom = 1;
        this.fitMode = 'page'; // 'width' | 'page' | 'custom'

        this.MIN_ZOOM = 0.1;
        this.MAX_ZOOM = 5;

        this._layoutResizeObserver = null;
        this._zoomRefreshTimer = null;
        this._splitInstance = null;
        this._tocPixelWidth = null;
        this._rightPanePixelWidth = null;
    }

    initSplitPane() {
        const tocPane = document.getElementById('tocPane');
        const rightPane = document.getElementById('rightPane');

        const tocPct = 16;
        const contentPct = 100 - tocPct;

        this._splitInstance = Split(['#tocPane', '#contentPane', '#rightPane'], {
            sizes: [tocPct, contentPct, 0],
            minSize: [5, 5, 0],
            gutterSize: 6,
            cursor: 'col-resize',
            onDragEnd: () => {
                this._tocPixelWidth = tocPane.getBoundingClientRect().width;
                this._rightPanePixelWidth = rightPane.getBoundingClientRect().width;
            }
        });
        this._tocPixelWidth = tocPane.getBoundingClientRect().width;
        this._rightPanePixelWidth = rightPane.getBoundingClientRect().width;

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

    // --- PDF layout (columns + zoom) ---

    _pdfContent() {
        return this.contentContainer.querySelector('.pdf-content');
    }

    applyColumns() {
        const pdfContent = this._pdfContent();
        if (!pdfContent) return;
        pdfContent.classList.add('pdf-cols');
        pdfContent.style.setProperty('--pdf-columns', this.columns);
    }

    applyZoom() {
        const pdfContent = this._pdfContent();
        if (!pdfContent) return;
        pdfContent.style.setProperty('--pdf-zoom', this.pdfZoom);

        // After layout settles, re-rasterize visible pages at the new size so
        // zoomed-in text is rendered sharp rather than CSS-upscaled. Debounced
        // because the zoom buttons/resize can fire a stream of updates.
        if (this.onZoomApplied) {
            clearTimeout(this._zoomRefreshTimer);
            this._zoomRefreshTimer = setTimeout(() => this.onZoomApplied(), 150);
        }
    }

    // Zoom multiplier for a fit mode. 1 == one page exactly fills its column
    // ('width'); 'page' shrinks further so a whole page fits the height.
    computeFitZoom(mode) {
        const pdfContent = this._pdfContent();
        if (!pdfContent) return 1;
        if (mode === 'width') return 1;

        const style = getComputedStyle(pdfContent);
        const padX = (parseFloat(style.paddingLeft) || 0) + (parseFloat(style.paddingRight) || 0);
        const padY = (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0);
        const availableWidth = pdfContent.clientWidth - padX;
        const availableHeight = pdfContent.clientHeight - padY;

        let pageAspect = 900 / 1165; // width / height fallback
        const firstDiv = this.getPdfPageDivs()[0];
        if (firstDiv && firstDiv._pdfViewport) {
            const vp = firstDiv._pdfViewport;
            pageAspect = vp.width / vp.height;
        }

        const gap = 6;
        const colWidth = (availableWidth - (this.columns - 1) * gap) / this.columns;
        if (colWidth <= 0) return 1;
        const pageHeightAtFitWidth = colWidth / pageAspect;
        if (pageHeightAtFitWidth <= 0) return 1;
        return Math.min(1, availableHeight / pageHeightAtFitWidth);
    }

    _clampZoom(z) {
        return Math.min(this.MAX_ZOOM, Math.max(this.MIN_ZOOM, z));
    }

    _notify() {
        if (this.onStateChanged) this.onStateChanged();
    }

    // Called when a PDF is first shown: one column, whole first page visible.
    initForDocument() {
        this.columns = 1;
        this.fitMode = 'page';
        this.applyColumns();
        this.pdfZoom = this._clampZoom(this.computeFitZoom('page'));
        this.applyZoom();
        this._setupLayoutResizeObserver();
        this._notify();
    }

    setColumns(n) {
        this.columns = Math.min(4, Math.max(1, Math.round(n)));
        this.applyColumns();
        // Refit to width so the new column count fills the area cleanly.
        this.fitMode = 'width';
        this.pdfZoom = this._clampZoom(this.computeFitZoom('width'));
        this.applyZoom();
        this._notify();
    }

    fitWidth() {
        this.fitMode = 'width';
        this.pdfZoom = this._clampZoom(this.computeFitZoom('width'));
        this.applyZoom();
        this._notify();
    }

    fitPage() {
        this.fitMode = 'page';
        this.pdfZoom = this._clampZoom(this.computeFitZoom('page'));
        this.applyZoom();
        this._notify();
    }

    setZoom(z) {
        this.fitMode = 'custom';
        this.pdfZoom = this._clampZoom(z);
        this.applyZoom();
        this._notify();
    }

    zoomBy(factor) {
        this.setZoom(this.pdfZoom * factor);
    }

    // Re-fit after an external geometry change (e.g. rotation) using current mode.
    refit() {
        if (this.fitMode !== 'custom') {
            this.pdfZoom = this._clampZoom(this.computeFitZoom(this.fitMode));
        }
        this.applyZoom();
        this._notify();
    }

    _setupLayoutResizeObserver() {
        if (this._layoutResizeObserver) {
            this._layoutResizeObserver.disconnect();
            this._layoutResizeObserver = null;
        }
        const pdfContent = this._pdfContent();
        if (!pdfContent) return;

        this._layoutResizeObserver = new ResizeObserver(() => {
            // 'width'/'custom' are handled by the %-based CSS automatically;
            // just refresh raster + readouts. 'page' must recompute fit zoom.
            if (this.fitMode === 'page') {
                const z = this._clampZoom(this.computeFitZoom('page'));
                if (Math.abs(z - this.pdfZoom) > 0.005) this.pdfZoom = z;
            }
            this.applyZoom();
            this._notify();
        });
        this._layoutResizeObserver.observe(pdfContent);
    }

    disconnectLayoutObserver() {
        if (this._layoutResizeObserver) {
            this._layoutResizeObserver.disconnect();
            this._layoutResizeObserver = null;
        }
    }
}
