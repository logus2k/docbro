export class Lightbox {

    constructor(contentContainer) {
        this.contentContainer = contentContainer;
        this.lightbox = null;
        this.lightboxImage = null;
        this.lightboxClose = null;
        this._create();
        this._setupImageClickHandlers();
    }

    _create() {
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

        this.lightboxClose.addEventListener('click', () => this.close());
        this.lightbox.addEventListener('click', (e) => {
            if (e.target === this.lightbox) {
                this.close();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.lightbox.classList.contains('active')) {
                this.close();
            }
        });
    }

    _setupImageClickHandlers() {
        this.contentContainer.addEventListener('click', (e) => {
            if (e.target.tagName === 'IMG') {
                this.open(e.target.src, e.target.alt);
            }
        });
    }

    open(src, alt) {
        this.lightboxImage.src = src;
        this.lightboxImage.alt = alt || '';
        this.lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    close() {
        this.lightbox.classList.remove('active');
        document.body.style.overflow = '';
    }
}
