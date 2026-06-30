export class LayoutManager {

    constructor({ contentContainer, getPdfPageDivs, onZoomApplied, onStateChanged }) {
        this.contentContainer = contentContainer;
        this.getPdfPageDivs = getPdfPageDivs;
        this.onZoomApplied = onZoomApplied;
        this.onStateChanged = onStateChanged;

        // PDF view state. `columns` = pages per row (1–4). Page sizing is
        // computed in pixels (pageWidthPx) and pushed to CSS; `pdfZoom` is a
        // derived readout where 1 == fit-width. `fitMode` drives how the size
        // recomputes on resize / rotation.
        this.columns = 1;
        this.pdfZoom = 1;
        this.pageWidthPx = 0;
        this.fitMode = 'page'; // 'width' | 'page' | 'custom'
        this.GAP = 6;          // px gap between pages in a row (matches CSS grid gap)

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
    //
    // Page sizing is computed in JS (px) and pushed to CSS via --pdf-page-width;
    // applyColumns sets --pdf-grid-cols to repeat(N, max-content). A centred CSS
    // grid then shows exactly N pages per row, adjacent, with spare width on the
    // outer edges — regardless of how tall/short the pages are.

    _pdfContent() {
        return this.contentContainer.querySelector('.pdf-content');
    }

    applyColumns() {
        const pdfContent = this._pdfContent();
        if (!pdfContent) return;
        pdfContent.classList.add('pdf-cols');
        // N fixed content-sized columns => exactly N pages per row; the grid is
        // centred so spare width sits on the outer edges, not between pages.
        pdfContent.style.setProperty('--pdf-grid-cols', `repeat(${this.columns}, max-content)`);
    }

    _notify() {
        if (this.onStateChanged) this.onStateChanged();
    }

    // Available content box + first page aspect (width / height) + per-column slot.
    _metrics() {
        const pdfContent = this._pdfContent();
        if (!pdfContent) return null;
        const style = getComputedStyle(pdfContent);
        const padX = (parseFloat(style.paddingLeft) || 0) + (parseFloat(style.paddingRight) || 0);
        const padY = (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0);
        const availW = Math.max(1, pdfContent.clientWidth - padX);
        const availH = Math.max(1, pdfContent.clientHeight - padY);

        let aspect = 900 / 1165; // width / height fallback
        const firstDiv = this.getPdfPageDivs()[0];
        if (firstDiv && firstDiv._pdfViewport) {
            aspect = firstDiv._pdfViewport.width / firstDiv._pdfViewport.height;
        }
        // Per-column width: the available width split into N columns with a
        // GAP between them, minus 1px slack so rounding never overflows.
        const slot = Math.max(20, Math.floor((availW - (this.columns - 1) * this.GAP) / this.columns) - 1);
        return { availW, availH, aspect, slot };
    }

    // Push a concrete page width (px) to CSS. The grid (N max-content columns,
    // centred) keeps exactly N pages per row, adjacent, with the spare width on
    // the outer edges — so no per-page margin is needed here.
    _applyPageWidth(pageWidth, m) {
        const pdfContent = this._pdfContent();
        if (!pdfContent) return;
        const pw = Math.max(20, Math.floor(pageWidth));
        this.pageWidthPx = pw;
        // Readout: 100% == fit-width (page fills its column).
        this.pdfZoom = pw / Math.max(1, m.slot);

        pdfContent.style.setProperty('--pdf-page-width', pw + 'px');

        if (this.onZoomApplied) {
            clearTimeout(this._zoomRefreshTimer);
            this._zoomRefreshTimer = setTimeout(() => this.onZoomApplied(), 150);
        }
        this._notify();
    }

    _fitWidthPx(m) {
        return m.slot;
    }

    _fitPagePx(m) {
        // Largest page that fits both its column width and the viewport height.
        return Math.max(20, Math.min(m.slot, Math.floor(m.availH * m.aspect)));
    }

    // Called when a PDF is first shown: one column, whole first page visible.
    initForDocument() {
        this.columns = 1;
        this.applyColumns();
        this.fitPage();
        this._setupLayoutResizeObserver();
    }

    setColumns(n) {
        this.columns = Math.min(4, Math.max(1, Math.round(n)));
        this.applyColumns();
        this.fitPage(); // each column count defaults to whole-page-visible
    }

    fitWidth() {
        const m = this._metrics();
        if (!m) return;
        this.fitMode = 'width';
        this._applyPageWidth(this._fitWidthPx(m), m);
    }

    fitPage() {
        const m = this._metrics();
        if (!m) return;
        this.fitMode = 'page';
        this._applyPageWidth(this._fitPagePx(m), m);
    }

    zoomBy(factor) {
        const m = this._metrics();
        if (!m) return;
        this.fitMode = 'custom';
        const base = this.pageWidthPx || this._fitPagePx(m);
        this._applyPageWidth(Math.max(20, base * factor), m);
    }

    // Re-fit after an external geometry change (e.g. rotation) using current mode.
    refit() {
        const m = this._metrics();
        if (!m) return;
        if (this.fitMode === 'width') this._applyPageWidth(this._fitWidthPx(m), m);
        else if (this.fitMode === 'custom') this._applyPageWidth(this.pageWidthPx || this._fitPagePx(m), m);
        else this._applyPageWidth(this._fitPagePx(m), m);
    }

    _setupLayoutResizeObserver() {
        if (this._layoutResizeObserver) {
            this._layoutResizeObserver.disconnect();
            this._layoutResizeObserver = null;
        }
        const pdfContent = this._pdfContent();
        if (!pdfContent) return;

        // Recompute pixel widths against the new content box for every mode.
        this._layoutResizeObserver = new ResizeObserver(() => this.refit());
        this._layoutResizeObserver.observe(pdfContent);
    }

    disconnectLayoutObserver() {
        if (this._layoutResizeObserver) {
            this._layoutResizeObserver.disconnect();
            this._layoutResizeObserver = null;
        }
    }
}
