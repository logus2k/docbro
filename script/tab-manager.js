export class TabState {

    constructor() {
        this.stickyTabs = new Set();       // globalIndex — persist across categories
        this.closedDefaults = new Set();   // globalIndex — openByDefault tabs user closed
    }

    isSticky(globalIndex) {
        return this.stickyTabs.has(globalIndex);
    }

    makeSticky(globalIndex) {
        this.stickyTabs.add(globalIndex);
    }

    close(globalIndex, doc) {
        this.stickyTabs.delete(globalIndex);
        if (doc && doc.openByDefault) {
            this.closedDefaults.add(globalIndex);
        }
    }

    shouldAutoOpen(doc) {
        return doc.openByDefault && !this.closedDefaults.has(doc.globalIndex);
    }

    /** Serializable snapshot for future persistence. */
    toJSON() {
        return {
            stickyTabs: [...this.stickyTabs],
            closedDefaults: [...this.closedDefaults]
        };
    }

    static fromJSON(data) {
        const state = new TabState();
        if (data.stickyTabs) data.stickyTabs.forEach(i => state.stickyTabs.add(i));
        if (data.closedDefaults) data.closedDefaults.forEach(i => state.closedDefaults.add(i));
        return state;
    }
}

export class TabManager {

    constructor({ tabsContainer, onActivateDocument }) {
        this.tabsContainer = tabsContainer;
        this.onActivateDocument = onActivateDocument;
        this.state = new TabState();
        this._visibleTabs = new Set();  // globalIndex — tabs shown in current render
        this._introTab = null;          // { category, label } — virtual intro tab
    }

    /**
     * Determine which tabs to show for the given category and render them.
     * @param {Array} documents - full documents array
     * @param {string} activeCategory - currently active category
     * @param {number|null} activatedIndex - the doc the user just clicked (null for category-level click)
     * @param {object|null} introInfo - { category, label } if an intro exists for this category
     */
    renderTabs(documents, activeCategory, activatedIndex = null, introInfo = null) {
        this.tabsContainer.innerHTML = '';
        this._introTab = introInfo;

        const categoryDocs = documents.filter(d => d.category === activeCategory);

        // Build the set of visible tabs
        this._visibleTabs = new Set();

        // 1. Sticky tabs (from any category)
        for (const idx of this.state.stickyTabs) {
            this._visibleTabs.add(idx);
        }

        // 2. openByDefault tabs for the active category (unless user closed them)
        for (const doc of categoryDocs) {
            if (this.state.shouldAutoOpen(doc)) {
                this._visibleTabs.add(doc.globalIndex);
            }
        }

        // 3. The specifically activated tab
        if (activatedIndex !== null && activatedIndex !== undefined) {
            this._visibleTabs.add(activatedIndex);
        }

        // Render intro tab first (if applicable)
        if (introInfo) {
            const tab = this._createTab(introInfo.label, 'intro', () => {
                this.onActivateDocument(null, null, activeCategory, true);
            });
            if (activatedIndex === null || activatedIndex === 'intro') {
                tab.classList.add('active');
            }
            this.tabsContainer.appendChild(tab);
        }

        // Render document tabs — order: active-category docs first, then sticky from other categories
        const activeCategoryVisible = categoryDocs.filter(d => this._visibleTabs.has(d.globalIndex));
        const otherVisible = documents.filter(d =>
            d.category !== activeCategory && this._visibleTabs.has(d.globalIndex)
        );

        const renderList = [...activeCategoryVisible, ...otherVisible];

        renderList.forEach((doc) => {
            const label = doc.category !== activeCategory
                ? `${doc.category} · ${doc.name}`
                : doc.name;

            const tab = this._createTab(label, doc.globalIndex, () => {
                this.onActivateDocument(doc.globalIndex);
            });

            if (this.state.isSticky(doc.globalIndex)) {
                tab.classList.add('sticky');
            }

            // Close button
            const closeBtn = document.createElement('span');
            closeBtn.className = 'tab-close-btn';
            closeBtn.textContent = '\u00d7';
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.closeTab(doc.globalIndex, documents, activeCategory);
            });
            tab.appendChild(closeBtn);

            this.tabsContainer.appendChild(tab);
        });
    }

    _createTab(label, dataIndex, onClick) {
        const tab = document.createElement('div');
        tab.className = 'tab';
        tab.textContent = label;
        tab.title = label;
        tab.setAttribute('data-doc-index', dataIndex);
        tab.addEventListener('click', onClick);
        return tab;
    }

    updateActiveState(globalIndex) {
        const tabs = this.tabsContainer.querySelectorAll('.tab');
        const indexStr = String(globalIndex);
        tabs.forEach(tab => {
            const tabIdx = tab.getAttribute('data-doc-index');
            if (tabIdx === indexStr) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
    }

    setIntroActive() {
        const tabs = this.tabsContainer.querySelectorAll('.tab');
        tabs.forEach(tab => {
            if (tab.getAttribute('data-doc-index') === 'intro') {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });
    }

    closeTab(globalIndex, documents, activeCategory) {
        const doc = documents[globalIndex];
        this.state.close(globalIndex, doc);
        this._visibleTabs.delete(globalIndex);

        // Determine which tab is currently active
        const activeTab = this.tabsContainer.querySelector('.tab.active');
        const activeAttr = activeTab ? activeTab.getAttribute('data-doc-index') : null;
        const activeDocumentIndex = activeAttr === 'intro' ? 'intro' : parseInt(activeAttr, 10);

        if (activeDocumentIndex === globalIndex) {
            // Closing the active tab — find a neighbour
            const remaining = documents.filter(d => this._visibleTabs.has(d.globalIndex));
            const nextDoc = remaining.length > 0 ? remaining[0] : null;

            this.renderTabs(documents, activeCategory, nextDoc ? nextDoc.globalIndex : null, this._introTab);
            if (nextDoc) {
                this.onActivateDocument(nextDoc.globalIndex);
            } else if (this._introTab) {
                this.onActivateDocument(null, null, activeCategory, true);
            }
        } else {
            this.renderTabs(documents, activeCategory,
                activeDocumentIndex === 'intro' ? null : activeDocumentIndex, this._introTab);
            if (activeDocumentIndex === 'intro') {
                this.setIntroActive();
            } else {
                this.updateActiveState(activeDocumentIndex);
            }
        }
    }
}
