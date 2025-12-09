class DocumentBrowser {
    constructor(configPath) {
        this.configPath = configPath;
        this.documents = [];
        this.categories = [];
        this.activeCategory = null;
        this.categoryLastActiveTab = {}; // Remembers last active tab index per category
        this.tabsContainer = document.getElementById('tabsContainer');
        this.contentContainer = document.getElementById('contentContainer');
        this.categorySelect = document.getElementById('categorySelect');
        
        this.init();
    }

    async init() {
        try {
            this.createLightbox();
            await this.loadConfiguration();
            await this.loadDocuments();
            this.extractCategories();
            this.renderCategorySelector();
            this.render();
            this.setActiveCategory(this.categories[0]);
            this.setupImageClickHandlers();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize document browser');
        }
    }

    createLightbox() {
        // Create lightbox elements
        const lightbox = document.createElement('div');
        lightbox.id = 'imageLightbox';
        lightbox.className = 'lightbox';
        lightbox.innerHTML = `
            <div class="lightbox-content">
                <span class="lightbox-close">&times;</span>
                <img class="lightbox-image" src="" alt="">
            </div>
        `;
        document.body.appendChild(lightbox);

        this.lightbox = lightbox;
        this.lightboxImage = lightbox.querySelector('.lightbox-image');
        this.lightboxClose = lightbox.querySelector('.lightbox-close');

        // Close handlers
        this.lightboxClose.addEventListener('click', () => this.closeLightbox());
        this.lightbox.addEventListener('click', (e) => {
            if (e.target === this.lightbox) {
                this.closeLightbox();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.lightbox.classList.contains('active')) {
                this.closeLightbox();
            }
        });
    }

    setupImageClickHandlers() {
        // Use event delegation on the content container
        this.contentContainer.addEventListener('click', (e) => {
            if (e.target.tagName === 'IMG') {
                this.openLightbox(e.target.src, e.target.alt);
            }
        });
    }

    openLightbox(src, alt) {
        this.lightboxImage.src = src;
        this.lightboxImage.alt = alt || '';
        this.lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    closeLightbox() {
        this.lightbox.classList.remove('active');
        document.body.style.overflow = '';
    }

    async loadConfiguration() {
        try {
            const response = await fetch(this.configPath);
            if (!response.ok) {
                throw new Error(`Failed to load configuration: ${response.status}`);
            }
            const config = await response.json();
            this.documents = config.documents || [];
        } catch (error) {
            console.error('Configuration loading error:', error);
            throw error;
        }
    }

    async loadDocuments() {
        const loadPromises = this.documents.map(async (doc, index) => {
            try {
                const response = await fetch(doc.location);
                if (!response.ok) {
                    throw new Error(`HTTP error ${response.status}`);
                }
                const markdown = await response.text();
                this.documents[index].content = this.renderMarkdown(markdown);
                this.documents[index].error = false;
            } catch (error) {
                console.error(`Error loading document ${doc.name}:`, error);
                this.documents[index].content = '';
                this.documents[index].error = true;
            }
        });

        await Promise.all(loadPromises);
    }

    extractCategories() {
        const categorySet = new Set();
        this.documents.forEach(doc => {
            if (doc.category) {
                categorySet.add(doc.category);
            }
        });
        this.categories = Array.from(categorySet);
    }

    renderCategorySelector() {
        this.categorySelect.innerHTML = '';
        
        this.categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            this.categorySelect.appendChild(option);
        });

        this.categorySelect.addEventListener('change', (e) => {
            this.setActiveCategory(e.target.value);
        });
    }

    setActiveCategory(category) {
        this.activeCategory = category;
        this.categorySelect.value = category;
        this.render();
        
        // Show the last active tab for this category, or the first one if none remembered
        const lastActiveIndex = this.categoryLastActiveTab[category];
        if (lastActiveIndex !== undefined) {
            this.showDocument(lastActiveIndex);
        } else {
            this.showDocument(0);
        }
    }

    getFilteredDocuments() {
        return this.documents.filter(doc => doc.category === this.activeCategory);
    }

    renderMarkdown(markdown) {
        // Store math expressions before Marked processes the markdown
        const mathExpressions = [];
        let mathIndex = 0;

        // Extract and replace display math $$...$$
        markdown = markdown.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => {
            const placeholder = `MATH_DISPLAY_${mathIndex}`;
            mathExpressions.push({ type: 'display', math: math.trim(), placeholder });
            mathIndex++;
            return placeholder;
        });

        // Extract and replace inline math $...$
        markdown = markdown.replace(/\$([^\$\n]+?)\$/g, (match, math) => {
            const placeholder = `MATH_INLINE_${mathIndex}`;
            mathExpressions.push({ type: 'inline', math: math.trim(), placeholder });
            mathIndex++;
            return placeholder;
        });

        // Convert markdown to HTML
        let html = marked.parse(markdown);

        // Restore math expressions with KaTeX rendering
        mathExpressions.forEach(item => {
            try {
                const rendered = katex.renderToString(item.math, { 
                    displayMode: item.type === 'display', 
                    throwOnError: false 
                });
                html = html.replace(item.placeholder, rendered);
            } catch (e) {
                console.error('KaTeX rendering error:', e);
                html = html.replace(item.placeholder, item.type === 'display' ? `$$${item.math}$$` : `$${item.math}$`);
            }
        });

        return html;
    }

    processMath(html) {
        // This method is no longer needed but kept for compatibility
        return html;
    }

    render() {
        this.renderTabs();
        this.renderContent();
    }

    renderTabs() {
        this.tabsContainer.innerHTML = '';
        const filteredDocs = this.getFilteredDocuments();
        
        filteredDocs.forEach((doc, index) => {
            const tab = document.createElement('div');
            tab.className = 'tab';
            tab.textContent = doc.name;
            tab.addEventListener('click', () => this.showDocument(index));
            this.tabsContainer.appendChild(tab);
        });
    }

    renderContent() {
        this.contentContainer.innerHTML = '';
        const filteredDocs = this.getFilteredDocuments();
        
        filteredDocs.forEach((doc, index) => {
            const contentDiv = document.createElement('div');
            contentDiv.className = 'document-content';
            contentDiv.setAttribute('data-index', index);
            
            if (doc.error) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = 'Loading error';
                contentDiv.appendChild(errorDiv);
            } else {
                contentDiv.innerHTML = doc.content;
            }
            
            this.contentContainer.appendChild(contentDiv);
        });
    }

    showDocument(index) {
        const filteredDocs = this.getFilteredDocuments();
        
        if (index < 0 || index >= filteredDocs.length) {
            return;
        }

        // Remember this tab for the current category
        this.categoryLastActiveTab[this.activeCategory] = index;

        // Update tabs
        const tabs = this.tabsContainer.querySelectorAll('.tab');
        tabs.forEach((tab, i) => {
            if (i === index) {
                tab.classList.add('active');
            } else {
                tab.classList.remove('active');
            }
        });

        // Update content
        const contents = this.contentContainer.querySelectorAll('.document-content');
        contents.forEach((content, i) => {
            if (i === index) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
    }

    showError(message) {
        this.contentContainer.innerHTML = `
            <div class="error-message">
                ${message}
            </div>
        `;
    }
}

// Initialize the document browser when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DocumentBrowser('documents.json');
});
