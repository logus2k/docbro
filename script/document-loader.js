import { getDocument } from '../libraries/pdf.js/pdf.min.mjs';

export class DocumentLoader {

    constructor(configPath) {
        this.configPath = configPath;
        this.documents = [];
        this.categories = [];
        this.categoryConfig = {};  // per-category config from documents.json
        this._introCache = {};     // category → { content, loaded, error }
    }

    isPdf(doc) {
        if (doc.type) return doc.type === 'pdf';
        return doc.location.toLowerCase().endsWith('.pdf');
    }

    // Register an in-memory (dropped) file as a temporary document backed by a
    // blob URL. PDFs and Markdown load lazily via the blob URL (so switching
    // away/back re-reads it); video content is prebuilt. Returns the new doc.
    registerLocalDocument({ name, type, blobUrl, mime }) {
        const CATEGORY = 'Local files';
        if (!this.categories.includes(CATEGORY)) this.categories.push(CATEGORY);
        // Avoid a 404 probe for a non-existent category intro.
        if (!(CATEGORY in this._introCache)) this._introCache[CATEGORY] = null;

        const isVideo = type === 'video';
        const doc = {
            name,
            category: CATEGORY,
            location: blobUrl,
            type,
            globalIndex: this.documents.length,
            loaded: isVideo,   // video content is ready immediately; pdf/md load lazily
            content: isVideo
                ? `<div class="embedded-video"><video controls><source src="${blobUrl}" type="${mime || 'video/mp4'}"></video></div>`
                : '',
            error: false,
            headers: null,
            configHeaders: null,
            openByDefault: false,
            pdfDoc: null,
            isLocal: true,
        };
        this.documents.push(doc);
        return doc;
    }

    async loadConfiguration() {
        try {
            const response = await fetch(this.configPath);
            if (!response.ok) {
                throw new Error(`Failed to load configuration: ${response.status}`);
            }
            const config = await response.json();
            this.categoryConfig = config.categories || {};
            this.documents = (config.documents || []).map((doc, index) => ({
                ...doc,
                globalIndex: index,
                loaded: false,
                content: '',
                error: false,
                headers: null,
                configHeaders: doc.headers || null,
                openByDefault: doc.openByDefault || false,
                pdfDoc: null
            }));
        } catch (error) {
            console.error('Configuration loading error:', error);
            throw error;
        }
    }

    async loadDocument(globalIndex) {
        const doc = this.documents[globalIndex];

        if (this.isPdf(doc)) {
            // pdf.js uses a PagesMapper singleton shared across all document
            // proxies — only one proxy may be alive at a time, otherwise page
            // validation breaks when switching between documents.
            for (const d of this.documents) {
                if (d.globalIndex !== globalIndex && d.pdfDoc) {
                    d.pdfDoc.destroy();
                    d.pdfDoc = null;
                }
            }

            if (doc.pdfDoc) return true;

            try {
                doc.pdfDoc = await getDocument({
                    url: doc.location,
                }).promise;
                doc.error = false;
                doc.loaded = true;
                return true;
            } catch (error) {
                console.error(`Error loading document ${doc.name}:`, error);
                doc.content = '';
                doc.error = true;
                doc.loaded = true;
                return false;
            }
        }

        if (doc.loaded) return true;

        try {
            const response = await fetch(doc.location);
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            const markdown = await response.text();
            doc.content = this.renderMarkdown(markdown);
            doc.error = false;
            doc.loaded = true;
            return true;
        } catch (error) {
            console.error(`Error loading document ${doc.name}:`, error);
            doc.content = '';
            doc.error = true;
            doc.loaded = true;
            return false;
        }
    }

    /**
     * Resolve the intro file location for a category.
     * Priority: explicit categoryConfig.intro > convention (intro.md in category folder).
     */
    getIntroLocation(category) {
        const catCfg = this.categoryConfig[category];
        if (catCfg && catCfg.intro) {
            return catCfg.intro;
        }
        // Convention: derive folder from the first document in the category
        const firstDoc = this.documents.find(d => d.category === category);
        if (firstDoc) {
            const folder = firstDoc.location.substring(0, firstDoc.location.lastIndexOf('/') + 1);
            return folder + 'intro.md';
        }
        return null;
    }

    /**
     * Load and render the intro markdown for a category.
     * Returns { content, error } or null if no intro exists.
     */
    async loadCategoryIntro(category) {
        if (category in this._introCache) {
            return this._introCache[category];
        }

        const location = this.getIntroLocation(category);
        if (!location) {
            this._introCache[category] = null;
            return null;
        }

        try {
            const response = await fetch(location);
            if (!response.ok) {
                this._introCache[category] = null;
                return null;
            }
            const markdown = await response.text();
            const entry = { content: this.renderMarkdown(markdown), error: false };
            this._introCache[category] = entry;
            return entry;
        } catch {
            this._introCache[category] = null;
            return null;
        }
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

    renderMarkdown(markdown) {
        const mathExpressions = [];
        let mathIndex = 0;

        markdown = markdown.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => {
            const placeholder = `MATH_DISPLAY_${mathIndex}`;
            mathExpressions.push({ type: 'display', math: math.trim(), placeholder });
            mathIndex++;
            return placeholder;
        });

        markdown = markdown.replace(/\$([^\$\n]+?)\$/g, (match, math) => {
            const placeholder = `MATH_INLINE_${mathIndex}`;
            mathExpressions.push({ type: 'inline', math: math.trim(), placeholder });
            mathIndex++;
            return placeholder;
        });

        let html = marked.parse(markdown);

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
}
