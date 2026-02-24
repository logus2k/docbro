export class ContentPanel {
    constructor(saveEndpoint = 'http://localhost:8000/save') {
        this.saveEndpoint = saveEndpoint;
        this.panel = null;
        this.editor = null;
        this.isOpen = false;
    }

    open() {
        if (this.panel) {
            this.panel.front();
            this.isOpen = true;
            return;
        }

        this.panel = jsPanel.create({
            id: 'contentEditorPanel',
            headerTitle: 'CONTENT EDITOR',
            theme: '#ffb44f',
            borderRadius: '8px',
            border: '0.5px solid #9a9a9a',
            boxShadow: 1,
            position: {
                my: 'right-top',
                at: 'right-top',
                offsetX: -20,
                offsetY: 60
            },
            panelSize: {
                width: 400,
                height: 500
            },
            headerControls: {
                minimize: 'remove',
                smallify: 'remove',
                normalize: 'remove',
                maximize: 'remove'
            },
            iconfont: [
                'custom-smallify',
                'custom-minimize',
                'custom-normalize',
                'custom-maximize',
                'custom-close'
            ],
            onclosed: () => {
                this.panel = null;
                this.editor = null;
                this.isOpen = false;
                document.dispatchEvent(new CustomEvent('content-panel-closed'));
            },
            callback: (panel) => {
                panel.content.innerHTML = `
                    <div class="content-editor-wrapper">
                        <div class="content-editor-body">
                            <textarea id="contentEditorTextarea"></textarea>
                        </div>
                        <div class="content-editor-toolbar">
                            <button class="content-editor-btn save-btn" id="contentEditorSave">Save</button>
                            <button class="content-editor-btn clear-btn" id="contentEditorClear">Clear</button>
                        </div>
                    </div>
                `;

                this.editor = new EasyMDE({
                    element: document.getElementById('contentEditorTextarea'),
                    spellChecker: false,
                    status: false,
                    toolbar: false,
                    minHeight: '100px',
                    autofocus: false,
                    placeholder: 'Select text blocks from the PDF and copy them here...',
                });

                document.getElementById('contentEditorSave').addEventListener('click', () => this.save());
                document.getElementById('contentEditorClear').addEventListener('click', () => this.clear());
            }
        });

        this.isOpen = true;
    }

    close() {
        if (this.panel) {
            this.panel.close();
        }
    }

    appendText(text) {
        if (!this.editor) return;
        const current = this.editor.value();
        const separator = current.length > 0 ? '\n\n' : '';
        this.editor.value(current + separator + text.trim());
    }

    getContent() {
        return this.editor ? this.editor.value() : '';
    }

    clear() {
        if (this.editor) {
            this.editor.value('');
        }
    }

    async save() {
        if (!this.editor) return;
        const content = this.editor.value();
        if (!content.trim()) return;

        const saveBtn = document.getElementById('contentEditorSave');

        try {
            const response = await fetch(this.saveEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content })
            });

            if (!response.ok) {
                throw new Error(`Save failed: ${response.status}`);
            }

            if (saveBtn) {
                saveBtn.textContent = 'Saved!';
                saveBtn.classList.add('saved');
                setTimeout(() => {
                    saveBtn.textContent = 'Save';
                    saveBtn.classList.remove('saved');
                }, 2000);
            }
        } catch (error) {
            console.error('Save error:', error);
            if (saveBtn) {
                saveBtn.textContent = 'Error';
                saveBtn.classList.add('error');
                setTimeout(() => {
                    saveBtn.textContent = 'Save';
                    saveBtn.classList.remove('error');
                }, 2000);
            }
        }
    }
}
