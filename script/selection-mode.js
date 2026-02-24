export class SelectionMode {
    constructor(contentContainer, onCopyToEditor) {
        this.contentContainer = contentContainer;
        this.onCopyToEditor = onCopyToEditor;
        this.active = false;
        this.paragraphMap = new Map(); // Map<pageDiv, ParagraphInfo[]>
        this.selectedParagraphs = new Set();
    }

    registerPage(pageDiv, textContent, textScale, viewport) {
        const paragraphs = this._groupIntoParagraphs(textContent, textScale, viewport);
        this.paragraphMap.set(pageDiv, paragraphs);

        // Attach pageDiv reference to each paragraph
        for (const para of paragraphs) {
            para.pageDiv = pageDiv;
        }

        // If already active, create overlays immediately
        if (this.active) {
            this._createOverlays(pageDiv);
        }
    }

    activate() {
        this.active = true;
        for (const [pageDiv] of this.paragraphMap) {
            this._createOverlays(pageDiv);
        }
        document.body.classList.add('selection-mode-active');
    }

    deactivate() {
        this.active = false;
        this.selectedParagraphs.clear();
        for (const [pageDiv] of this.paragraphMap) {
            this._removeOverlays(pageDiv);
        }
        document.body.classList.remove('selection-mode-active');
    }

    reset() {
        this.deactivate();
        this.paragraphMap.clear();
    }

    isActive() {
        return this.active;
    }

    _groupIntoParagraphs(textContent, textScale, viewport) {
        const items = textContent.items.filter(item => item.str && item.str.trim().length > 0);
        if (items.length === 0) return [];

        // Convert PDF coordinates to viewport coordinates
        const fragments = items.map(item => {
            const tx = item.transform[4];
            const ty = item.transform[5];
            const [vx, vy] = viewport.convertToViewportPoint(tx, ty);
            const itemHeight = item.height * textScale;
            const itemWidth = item.width * textScale;

            return {
                text: item.str,
                left: vx,
                top: vy - itemHeight, // PDF origin is bottom-left, viewport is top-left
                right: vx + itemWidth,
                bottom: vy,
                height: itemHeight
            };
        });

        // Sort by top position
        fragments.sort((a, b) => a.top - b.top);

        // Group into lines (fragments within ~3px vertically)
        const lineTolerance = 3 * textScale;
        const lines = [];
        let currentLine = [fragments[0]];

        for (let i = 1; i < fragments.length; i++) {
            const frag = fragments[i];
            const lineTop = currentLine[0].top;

            if (Math.abs(frag.top - lineTop) <= lineTolerance) {
                currentLine.push(frag);
            } else {
                lines.push(this._mergeLine(currentLine));
                currentLine = [frag];
            }
        }
        lines.push(this._mergeLine(currentLine));

        // Calculate average line height
        const avgLineHeight = lines.reduce((sum, l) => sum + l.height, 0) / lines.length;

        // Group lines into paragraphs (gap < 1.5x average line height)
        const paragraphGapThreshold = avgLineHeight * 1.5;
        const paragraphs = [];
        let currentParaLines = [lines[0]];

        for (let i = 1; i < lines.length; i++) {
            const prevLine = lines[i - 1];
            const currLine = lines[i];
            const gap = currLine.top - prevLine.bottom;

            if (gap < paragraphGapThreshold) {
                currentParaLines.push(currLine);
            } else {
                paragraphs.push(this._mergeParaLines(currentParaLines));
                currentParaLines = [currLine];
            }
        }
        paragraphs.push(this._mergeParaLines(currentParaLines));

        return paragraphs;
    }

    _mergeLine(fragments) {
        // Sort by left position within line
        fragments.sort((a, b) => a.left - b.left);

        const text = fragments.map(f => f.text).join(' ');
        const left = Math.min(...fragments.map(f => f.left));
        const top = Math.min(...fragments.map(f => f.top));
        const right = Math.max(...fragments.map(f => f.right));
        const bottom = Math.max(...fragments.map(f => f.bottom));

        return {
            text,
            left,
            top,
            right,
            bottom,
            height: bottom - top
        };
    }

    _mergeParaLines(lines) {
        const text = lines.map(l => l.text).join(' ');
        const padding = 4;
        const left = Math.min(...lines.map(l => l.left)) - padding;
        const top = Math.min(...lines.map(l => l.top)) - padding;
        const right = Math.max(...lines.map(l => l.right)) + padding;
        const bottom = Math.max(...lines.map(l => l.bottom)) + padding;

        return {
            text,
            boundingBox: {
                left,
                top,
                width: right - left,
                height: bottom - top
            },
            pageDiv: null,
            overlayEl: null
        };
    }

    _createOverlays(pageDiv) {
        const paragraphs = this.paragraphMap.get(pageDiv);
        if (!paragraphs) return;

        for (const para of paragraphs) {
            if (para.overlayEl) continue; // already created

            const overlay = document.createElement('div');
            overlay.className = 'pdf-paragraph-overlay';
            overlay.style.left = para.boundingBox.left + 'px';
            overlay.style.top = para.boundingBox.top + 'px';
            overlay.style.width = para.boundingBox.width + 'px';
            overlay.style.height = para.boundingBox.height + 'px';

            overlay.addEventListener('mouseenter', () => {
                overlay.classList.add('hovered');
                if (this.selectedParagraphs.has(overlay)) {
                    this._showCopyButton(overlay, para);
                }
            });

            overlay.addEventListener('mouseleave', () => {
                overlay.classList.remove('hovered');
                this._hideCopyButton(overlay);
            });

            overlay.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();

                if (this.selectedParagraphs.has(overlay)) {
                    this.selectedParagraphs.delete(overlay);
                    overlay.classList.remove('selected');
                    this._hideCopyButton(overlay);
                } else {
                    this.selectedParagraphs.add(overlay);
                    overlay.classList.add('selected');
                    this._showCopyButton(overlay, para);
                }
            });

            pageDiv.appendChild(overlay);
            para.overlayEl = overlay;
        }
    }

    _removeOverlays(pageDiv) {
        const paragraphs = this.paragraphMap.get(pageDiv);
        if (!paragraphs) return;

        for (const para of paragraphs) {
            if (para.overlayEl) {
                para.overlayEl.remove();
                para.overlayEl = null;
            }
        }
    }

    _showCopyButton(overlayEl, paragraphInfo) {
        if (overlayEl.querySelector('.para-copy-btn')) return;

        const btn = document.createElement('button');
        btn.className = 'para-copy-btn';
        btn.textContent = 'Copy to editor';
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            this._copyToEditor(paragraphInfo);
        });
        overlayEl.appendChild(btn);
    }

    _hideCopyButton(overlayEl) {
        const btn = overlayEl.querySelector('.para-copy-btn');
        if (btn) btn.remove();
    }

    _copyToEditor(paragraphInfo) {
        this.onCopyToEditor(paragraphInfo.text);
        const overlay = paragraphInfo.overlayEl;
        this.selectedParagraphs.delete(overlay);
        overlay.classList.remove('selected');
        this._hideCopyButton(overlay);
    }
}
