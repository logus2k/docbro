export class TabManager {

    constructor({ tabsContainer, onActivateDocument }) {
        this.tabsContainer = tabsContainer;
        this.onActivateDocument = onActivateDocument;
        this.closedTabs = new Set();
    }

    renderTabs(documents, activeCategory) {
        this.tabsContainer.innerHTML = '';
        const categoryDocs = documents.filter(d => d.category === activeCategory);
        const openDocs = categoryDocs.filter(d => !this.closedTabs.has(d.globalIndex));

        openDocs.forEach((doc) => {
            const tab = document.createElement('div');
            tab.className = 'tab';
            tab.textContent = doc.name;
            tab.title = doc.name;
            tab.setAttribute('data-doc-index', doc.globalIndex);
            tab.addEventListener('click', () => this.onActivateDocument(doc.globalIndex));

            if (openDocs.length > 1) {
                const closeBtn = document.createElement('span');
                closeBtn.className = 'tab-close-btn';
                closeBtn.textContent = '\u00d7';
                closeBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.closeTab(doc.globalIndex, documents, activeCategory);
                });
                tab.appendChild(closeBtn);
            }

            this.tabsContainer.appendChild(tab);
        });
    }

    updateActiveState(globalIndex) {
        const tabs = this.tabsContainer.querySelectorAll('.tab');
        tabs.forEach(tab => {
            const tabDocIndex = parseInt(tab.getAttribute('data-doc-index'), 10);
            if (tabDocIndex === globalIndex) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
    }

    closeTab(globalIndex, documents, activeCategory) {
        const categoryDocs = documents.filter(d => d.category === activeCategory);
        const openDocs = categoryDocs.filter(d => !this.closedTabs.has(d.globalIndex));

        // Determine which tab is currently active from the DOM
        const activeTab = this.tabsContainer.querySelector('.tab.active');
        const activeDocumentIndex = activeTab
            ? parseInt(activeTab.getAttribute('data-doc-index'), 10)
            : null;

        if (openDocs.length <= 1) return;

        this.closedTabs.add(globalIndex);

        if (activeDocumentIndex === globalIndex) {
            const closedPos = categoryDocs.findIndex(d => d.globalIndex === globalIndex);
            let nextDoc = null;

            for (let i = closedPos + 1; i < categoryDocs.length; i++) {
                if (!this.closedTabs.has(categoryDocs[i].globalIndex)) {
                    nextDoc = categoryDocs[i];
                    break;
                }
            }
            if (!nextDoc) {
                for (let i = closedPos - 1; i >= 0; i--) {
                    if (!this.closedTabs.has(categoryDocs[i].globalIndex)) {
                        nextDoc = categoryDocs[i];
                        break;
                    }
                }
            }

            this.renderTabs(documents, activeCategory);
            if (nextDoc) {
                this.onActivateDocument(nextDoc.globalIndex);
            }
        } else {
            this.renderTabs(documents, activeCategory);
            this.updateActiveState(activeDocumentIndex);
        }
    }
}
