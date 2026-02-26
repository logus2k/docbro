export class ChatPanel {

    constructor(containerElement) {
        this.container = containerElement;
        this._onSendCallback = null;
        this._onSttToggleCallback = null;
        this._onTtsToggleCallback = null;
        this._sttActive = false;
        this._build();
    }

    _build() {
        const panel = document.createElement('div');
        panel.className = 'chat-panel';

        // Header
        const header = document.createElement('div');
        header.className = 'chat-header';
        header.textContent = 'Chat';
        panel.appendChild(header);

        // Messages area
        this._messagesArea = document.createElement('div');
        this._messagesArea.className = 'chat-messages';
        panel.appendChild(this._messagesArea);

        // Typing indicator
        this._typingIndicator = document.createElement('div');
        this._typingIndicator.className = 'chat-typing-indicator';
        this._typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        this._typingIndicator.style.display = 'none';
        this._messagesArea.appendChild(this._typingIndicator);

        // Input area
        const inputArea = document.createElement('div');
        inputArea.className = 'chat-input-area';

        // STT button
        const sttBtn = document.createElement('button');
        sttBtn.className = 'chat-stt-btn';
        sttBtn.title = 'Voice input';
        sttBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M8 1C6.89543 1 6 1.89543 6 3V8C6 9.10457 6.89543 10 8 10C9.10457 10 10 9.10457 10 8V3C10 1.89543 9.10457 1 8 1ZM7 3C7 2.44772 7.44772 2 8 2C8.55228 2 9 2.44772 9 3V8C9 8.55228 8.55228 9 8 9C7.44772 9 7 8.55228 7 8V3ZM4 7.5C4.27614 7.5 4.5 7.72386 4.5 8C4.5 9.933 6.067 11.5 8 11.5C9.933 11.5 11.5 9.933 11.5 8C11.5 7.72386 11.7239 7.5 12 7.5C12.2761 7.5 12.5 7.72386 12.5 8C12.5 10.3688 10.7462 12.3235 8.5 12.6V14.5C8.5 14.7761 8.27614 15 8 15C7.72386 15 7.5 14.7761 7.5 14.5V12.6C5.25381 12.3235 3.5 10.3688 3.5 8C3.5 7.72386 3.72386 7.5 4 7.5Z"/></svg>';
        sttBtn.addEventListener('click', () => {
            this._sttActive = !this._sttActive;
            sttBtn.classList.toggle('active', this._sttActive);
            if (this._onSttToggleCallback) {
                this._onSttToggleCallback(this._sttActive);
            }
        });
        inputArea.appendChild(sttBtn);

        // Text input
        this._input = document.createElement('textarea');
        this._input.className = 'chat-input';
        this._input.placeholder = 'Type a message...';
        this._input.rows = 1;
        this._input.addEventListener('input', () => this._autoGrow());
        this._input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this._handleSend();
            }
        });
        inputArea.appendChild(this._input);

        // Send button
        const sendBtn = document.createElement('button');
        sendBtn.className = 'chat-send-btn';
        sendBtn.title = 'Send message';
        sendBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M1.72386 1.05279C1.49903 0.940376 1.22918 0.959728 1.02382 1.10257C0.818462 1.24542 0.710144 1.4889 0.747826 1.73511L1.63486 7.25H8.5C8.77614 7.25 9 7.47386 9 7.75C9 8.02614 8.77614 8.25 8.5 8.25H1.63486L0.747826 13.7649C0.710144 14.0111 0.818462 14.2546 1.02382 14.3974C1.22918 14.5403 1.49903 14.5596 1.72386 14.4472L14.7239 7.94721C14.893 7.86264 15 7.68996 15 7.5C15 7.31004 14.893 7.13736 14.7239 7.05279L1.72386 1.05279Z"/></svg>';
        sendBtn.addEventListener('click', () => this._handleSend());
        inputArea.appendChild(sendBtn);

        // TTS button
        const ttsBtn = document.createElement('button');
        ttsBtn.className = 'chat-tts-btn';
        ttsBtn.title = 'Text to speech';
        ttsBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M8.29289 1.29289C8.57889 1.00689 9.009 0.921968 9.38268 1.07612C9.75636 1.23028 10 1.59554 10 2V14C10 14.4045 9.75636 14.7697 9.38268 14.9239C9.009 15.078 8.57889 14.9931 8.29289 14.7071L4.58579 11H2C1.44772 11 1 10.5523 1 10V6C1 5.44772 1.44772 5 2 5H4.58579L8.29289 1.29289ZM9 2.41421L5.70711 5.70711C5.51957 5.89464 5.26522 6 5 6H2V10H5C5.26522 10 5.51957 10.1054 5.70711 10.2929L9 13.5858V2.41421ZM12.3588 3.05546C12.5274 2.82296 12.8399 2.76585 13.0724 2.93443C14.2808 3.80897 15 5.26286 15 7C15 8.73714 14.2808 10.191 13.0724 11.0656C12.8399 11.2341 12.5274 11.177 12.3588 10.9445C12.1902 10.712 12.2474 10.3996 12.4799 10.231C13.4137 9.55496 14 8.3876 14 7C14 5.6124 13.4137 4.44504 12.4799 3.76904C12.2474 3.60045 12.1902 3.28795 12.3588 3.05546ZM11.2826 5.11701C11.4448 4.88002 11.7558 4.81373 11.9928 4.97594C12.5869 5.38207 13 6.12078 13 7C13 7.87922 12.5869 8.61793 11.9928 9.02406C11.7558 9.18627 11.4448 9.11998 11.2826 8.88299C11.1204 8.646 11.1867 8.33497 11.4237 8.17276C11.7577 7.94371 12 7.53371 12 7C12 6.46629 11.7577 6.05629 11.4237 5.82724C11.1867 5.66503 11.1204 5.354 11.2826 5.11701Z"/></svg>';
        ttsBtn.addEventListener('click', () => {
            if (this._onTtsToggleCallback) {
                this._onTtsToggleCallback();
            }
        });
        inputArea.appendChild(ttsBtn);

        panel.appendChild(inputArea);
        this.container.appendChild(panel);
    }

    _autoGrow() {
        this._input.style.height = 'auto';
        this._input.style.height = Math.min(this._input.scrollHeight, 120) + 'px';
    }

    _handleSend() {
        const text = this._input.value.trim();
        if (!text) return;

        this.addMessage('user', text);
        this._input.value = '';
        this._input.style.height = 'auto';

        if (this._onSendCallback) {
            this._onSendCallback(text);
        }
    }

    addMessage(role, text) {
        const msg = document.createElement('div');
        msg.className = `chat-message chat-message-${role}`;
        msg.textContent = text;

        // Insert before typing indicator
        this._messagesArea.insertBefore(msg, this._typingIndicator);
        this._messagesArea.scrollTop = this._messagesArea.scrollHeight;
    }

    clearMessages() {
        const messages = this._messagesArea.querySelectorAll('.chat-message');
        messages.forEach(m => m.remove());
    }

    setInputText(text) {
        this._input.value = text;
        this._autoGrow();
    }

    setLoading(loading) {
        this._typingIndicator.style.display = loading ? 'flex' : 'none';
        if (loading) {
            this._messagesArea.scrollTop = this._messagesArea.scrollHeight;
        }
    }

    onSend(callback) {
        this._onSendCallback = callback;
    }

    onSttToggle(callback) {
        this._onSttToggleCallback = callback;
    }

    onTtsToggle(callback) {
        this._onTtsToggleCallback = callback;
    }
}
