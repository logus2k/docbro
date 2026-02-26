export class LayoutManager {

    constructor({ contentContainer, getPdfPageDivs }) {
        this.contentContainer = contentContainer;
        this.getPdfPageDivs = getPdfPageDivs;
        this.pageLayoutMode = 'single';
        this.pdfZoom = 1;
        this._layoutResizeObserver = null;
        this._splitInstance = null;
        this._tocPixelWidth = null;
        this._rightPanePixelWidth = null;
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

        let pageAspect = 900 / 1165;
        const pdfPageDivs = this.getPdfPageDivs();
        const firstDiv = pdfPageDivs[0];
        if (firstDiv && firstDiv._pdfViewport) {
            const vp = firstDiv._pdfViewport;
            pageAspect = vp.width / vp.height;
        }

        if (mode === 'single') {
            const zoomByWidth = availableWidth / 900;
            const zoomByHeight = (availableHeight * pageAspect) / 900;
            return Math.min(zoomByWidth, zoomByHeight);
        }
        if (mode === 'dual') {
            const zoomByWidth = (availableWidth - 6) / 900;
            const zoomByHeight = (availableHeight * pageAspect) / 450;
            return Math.min(zoomByWidth, zoomByHeight);
        }
        return this.pdfZoom;
    }

    applyLayoutAndZoom() {
        const zoomSlider = document.getElementById('pdfZoomSlider');
        const zoomValue = document.getElementById('pdfZoomValue');

        this.applyPageLayout();

        if (this.pageLayoutMode === 'single' || this.pageLayoutMode === 'dual') {
            this.pdfZoom = this.computeFitZoom(this.pageLayoutMode);
        }

        if (zoomSlider) {
            const pct = Math.round(this.pdfZoom * 100);
            zoomSlider.value = pct;
            if (zoomValue) zoomValue.textContent = pct + '%';
        }

        this.applyZoom();
        this._setupLayoutResizeObserver();
    }

    _setupLayoutResizeObserver() {
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

    disconnectLayoutObserver() {
        if (this._layoutResizeObserver) {
            this._layoutResizeObserver.disconnect();
            this._layoutResizeObserver = null;
        }
    }
}
