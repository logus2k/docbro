// drop-handler.js — drag & drop of local files (PDF / Markdown / video) onto docbro.
// Shows a full-window overlay while a file is dragged over the app, and hands the
// dropped File objects to onFiles() for rendering.

export class DropHandler {

    constructor({ overlay, onFiles }) {
        this.overlay = overlay;
        this.onFiles = onFiles;
        this._depth = 0; // dragenter/leave nesting counter (avoids overlay flicker)
    }

    _hasFiles(e) {
        return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files');
    }

    _show() { this.overlay && this.overlay.classList.add('visible'); }
    _hide() { this.overlay && this.overlay.classList.remove('visible'); }

    attach() {
        window.addEventListener('dragenter', (e) => {
            if (!this._hasFiles(e)) return;
            e.preventDefault();
            this._depth++;
            this._show();
        });

        window.addEventListener('dragover', (e) => {
            if (!this._hasFiles(e)) return;
            e.preventDefault(); // required so 'drop' fires
            if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
        });

        window.addEventListener('dragleave', (e) => {
            if (!this._hasFiles(e)) return;
            this._depth = Math.max(0, this._depth - 1);
            if (this._depth === 0) this._hide();
        });

        window.addEventListener('drop', (e) => {
            if (!this._hasFiles(e)) return;
            e.preventDefault();
            this._depth = 0;
            this._hide();
            const files = Array.from(e.dataTransfer.files || []);
            if (files.length && this.onFiles) this.onFiles(files);
        });
    }
}
