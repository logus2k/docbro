import { getDocument } from '../libraries/pdf.js/pdf.min.mjs';

export class DocumentLoader {

    constructor(configPath) {
        this.configPath = configPath;
        this.documents = [];
        this.categories = [];
    }

    isPdf(doc) {
        return doc.location.toLowerCase().endsWith('.pdf');
    }

    async loadConfiguration() {
        try {
            const response = await fetch(this.configPath);
            if (!response.ok) {
                throw new Error(`Failed to load configuration: ${response.status}`);
            }
            const config = await response.json();
            this.documents = (config.documents || []).map((doc, index) => ({
                ...doc,
                globalIndex: index,
                loaded: false,
                content: '',
                error: false,
                headers: null,
                configHeaders: doc.headers || null,
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
