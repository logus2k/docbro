// pdf-controls.js — floating PDF control box (navigation, zoom, columns, rotate)
//
// Drives the static #pdfControlBox markup. Shows on mouse activity over the
// content pane and auto-hides ~2s after movement stops. Talks to LayoutManager
// (columns + zoom) and PdfRenderer (rotation + page navigation).

export class PdfControls {

    constructor({ contentPane, layoutManager, pdfRenderer, getScrollContainer }) {
        this.contentPane = contentPane;
        this.layoutManager = layoutManager;
        this.pdfRenderer = pdfRenderer;
        this.getScrollContainer = getScrollContainer;

        this.box = document.getElementById('pdfControlBox');
        this.el = {
            prev: document.getElementById('pcbPrev'),
            next: document.getElementById('pcbNext'),
            pageInput: document.getElementById('pcbPageInput'),
            pageTotal: document.getElementById('pcbPageTotal'),
            zoomOut: document.getElementById('pcbZoomOut'),
            zoomIn: document.getElementById('pcbZoomIn'),
            zoomValue: document.getElementById('pcbZoomValue'),
            fitWidth: document.getElementById('pcbFitWidth'),
            fitPage: document.getElementById('pcbFitPage'),
            columns: document.getElementById('pcbColumns'),
            columnsValue: document.getElementById('pcbColumnsValue'),
            rotate: document.getElementById('pcbRotate'),
        };

        this._hideTimer = null;
        this._hovering = false;
        this._pageInputFocused = false;
        this._isPdf = false;
        this._scrollContainer = null;
        this._scrollHandler = null;
        this._scrollRaf = 0;
    }

    attach() {
        if (!this.box) return;

        // --- Auto-hide on activity ---
        this.contentPane.addEventListener('mousemove', () => {
            if (this._isPdf) this._show();
        });
        this.box.addEventListener('mouseenter', () => {
            this._hovering = true;
            clearTimeout(this._hideTimer);
        });
        this.box.addEventListener('mouseleave', () => {
            this._hovering = false;
            this._resetHideTimer();
        });

        // --- Page navigation ---
        this.el.prev?.addEventListener('click', () => this._goRelative(-1));
        this.el.next?.addEventListener('click', () => this._goRelative(1));
        this.el.pageInput?.addEventListener('focus', () => { this._pageInputFocused = true; });
        this.el.pageInput?.addEventListener('blur', () => { this._pageInputFocused = false; this._commitPageInput(); });
        this.el.pageInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); this.el.pageInput.blur(); }
        });

        // --- Zoom ---
        this.el.zoomOut?.addEventListener('click', () => this.layoutManager.zoomBy(1 / 1.1));
        this.el.zoomIn?.addEventListener('click', () => this.layoutManager.zoomBy(1.1));
        this.el.fitWidth?.addEventListener('click', () => this.layoutManager.fitWidth());
        this.el.fitPage?.addEventListener('click', () => this.layoutManager.fitPage());

        // --- Columns (pages per row) ---
        this.el.columns?.addEventListener('input', (e) => {
            const n = parseInt(e.target.value, 10) || 1;
            if (this.el.columnsValue) this.el.columnsValue.textContent = String(n);
            this.layoutManager.setColumns(n);
        });

        // --- Rotate ---
        this.el.rotate?.addEventListener('click', () => {
            this.pdfRenderer.setRotation(this.pdfRenderer.rotation + 90);
            this.layoutManager.refit();
            this.syncReadouts();
        });
    }

    // Show/reset the box for the active document (PDF) or hide it (non-PDF).
    showForDocument(isPdf) {
        this._isPdf = isPdf;
        if (!this.box) return;

        this._detachScroll();

        if (!isPdf) {
            this.box.classList.remove('visible');
            this.box.style.display = 'none';
            return;
        }

        this.box.style.display = '';
        this._attachScroll();
        this.syncReadouts();
        this._show();
    }

    // Refresh all readouts from the managers (zoom %, columns, page X / N).
    syncReadouts() {
        if (!this.box || !this._isPdf) return;

        if (this.el.zoomValue) {
            this.el.zoomValue.textContent = Math.round(this.layoutManager.pdfZoom * 100) + '%';
        }
        if (this.el.columns) {
            this.el.columns.value = String(this.layoutManager.columns);
        }
        if (this.el.columnsValue) {
            this.el.columnsValue.textContent = String(this.layoutManager.columns);
        }

        const total = this.pdfRenderer.pageCount || 0;
        if (this.el.pageTotal) this.el.pageTotal.textContent = String(total);
        if (this.el.pageInput && !this._pageInputFocused) {
            this.el.pageInput.value = String(this.pdfRenderer.getCurrentPageIndex() + 1);
        }
    }

    // --- internals ---

    _goRelative(direction) {
        // Row-based so it works for any column count (adjacent pages share a row).
        this.pdfRenderer.scrollByRow(direction);
    }

    _commitPageInput() {
        const total = this.pdfRenderer.pageCount || 0;
        let n = parseInt(this.el.pageInput.value, 10);
        if (isNaN(n)) n = this.pdfRenderer.getCurrentPageIndex() + 1;
        n = Math.min(Math.max(n, 1), Math.max(total, 1));
        this.el.pageInput.value = String(n);
        this.pdfRenderer.scrollToPage(n - 1);
    }

    _attachScroll() {
        const sc = this.getScrollContainer();
        if (!sc) return;
        this._scrollContainer = sc;
        this._scrollHandler = () => {
            if (this._scrollRaf) return;
            this._scrollRaf = requestAnimationFrame(() => {
                this._scrollRaf = 0;
                if (this.el.pageInput && !this._pageInputFocused) {
                    this.el.pageInput.value = String(this.pdfRenderer.getCurrentPageIndex() + 1);
                }
            });
        };
        sc.addEventListener('scroll', this._scrollHandler, { passive: true });
    }

    _detachScroll() {
        if (this._scrollContainer && this._scrollHandler) {
            this._scrollContainer.removeEventListener('scroll', this._scrollHandler);
        }
        if (this._scrollRaf) { cancelAnimationFrame(this._scrollRaf); this._scrollRaf = 0; }
        this._scrollContainer = null;
        this._scrollHandler = null;
    }

    _show() {
        this.box.classList.add('visible');
        this._resetHideTimer();
    }

    _resetHideTimer() {
        clearTimeout(this._hideTimer);
        if (this._hovering) return;
        this._hideTimer = setTimeout(() => this.box.classList.remove('visible'), 2000);
    }
}
