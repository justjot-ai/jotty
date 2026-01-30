/**
 * Jotty Web UI - Frontend Application
 *
 * LibreChat-inspired chat interface with:
 * - WebSocket streaming
 * - Session management
 * - Markdown rendering
 */

class JottyApp {
    constructor() {
        this.sessionId = null;
        this.ws = null;
        this.messages = [];
        this.isProcessing = false;
        this.streamingContent = '';

        this.init();
    }

    init() {
        // DOM elements
        this.chatArea = document.getElementById('chat-area');
        this.messagesContainer = document.getElementById('messages-container');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.newChatBtn = document.getElementById('new-chat-btn');
        this.sessionsList = document.getElementById('sessions-list');
        this.sessionIdDisplay = document.getElementById('session-id');
        this.menuToggle = document.getElementById('menu-toggle');
        this.sidebar = document.getElementById('sidebar');
        this.connectionStatus = document.getElementById('connection-status');
        this.welcomeScreen = document.getElementById('welcome-screen');

        // Event listeners
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeydown(e));
        this.messageInput.addEventListener('input', () => this.autoResize());
        this.newChatBtn.addEventListener('click', () => this.newChat());
        this.menuToggle?.addEventListener('click', () => this.toggleSidebar());

        // Example cards
        document.querySelectorAll('.example-card').forEach(card => {
            card.addEventListener('click', () => {
                const text = card.querySelector('.example-text').textContent;
                this.messageInput.value = text;
                this.messageInput.focus();
                this.autoResize();
            });
        });

        // Initialize
        this.loadSessions();
        this.newChat();
    }

    generateSessionId() {
        return 'web_' + Math.random().toString(36).substr(2, 8);
    }

    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            this.renderSessions(data.sessions || []);
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }

    renderSessions(sessions) {
        if (!this.sessionsList) return;

        // Group by date
        const today = new Date().toDateString();
        const yesterday = new Date(Date.now() - 86400000).toDateString();

        const grouped = {
            today: [],
            yesterday: [],
            older: []
        };

        sessions.forEach(session => {
            const date = new Date(session.created_at).toDateString();
            if (date === today) {
                grouped.today.push(session);
            } else if (date === yesterday) {
                grouped.yesterday.push(session);
            } else {
                grouped.older.push(session);
            }
        });

        let html = '';

        if (grouped.today.length > 0) {
            html += this.renderSessionGroup('Today', grouped.today);
        }
        if (grouped.yesterday.length > 0) {
            html += this.renderSessionGroup('Yesterday', grouped.yesterday);
        }
        if (grouped.older.length > 0) {
            html += this.renderSessionGroup('Previous', grouped.older);
        }

        this.sessionsList.innerHTML = html || '<div class="session-item">No sessions yet</div>';

        // Add click handlers
        this.sessionsList.querySelectorAll('.session-item').forEach(item => {
            item.addEventListener('click', () => {
                const id = item.dataset.sessionId;
                if (id) this.loadSession(id);
            });
        });
    }

    renderSessionGroup(title, sessions) {
        return `
            <div class="session-group">
                <div class="session-group-title">${title}</div>
                ${sessions.map(s => `
                    <div class="session-item ${s.session_id === this.sessionId ? 'active' : ''}"
                         data-session-id="${s.session_id}">
                        ${this.getSessionTitle(s)}
                    </div>
                `).join('')}
            </div>
        `;
    }

    getSessionTitle(session) {
        // Try to get title from first user message
        if (session.history && session.history.length > 0) {
            const firstUser = session.history.find(m => m.role === 'user');
            if (firstUser) {
                return firstUser.content.substring(0, 30) + (firstUser.content.length > 30 ? '...' : '');
            }
        }
        return `Session ${session.session_id}`;
    }

    async loadSession(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}`);
            if (!response.ok) throw new Error('Session not found');

            const session = await response.json();
            this.sessionId = sessionId;
            this.messages = session.history || [];

            this.updateSessionDisplay();
            this.renderMessages();
            this.connectWebSocket();
            this.loadSessions(); // Refresh list to show active

            // Hide welcome, show messages
            if (this.welcomeScreen) this.welcomeScreen.style.display = 'none';
            if (this.messagesContainer) this.messagesContainer.style.display = 'block';

        } catch (error) {
            console.error('Failed to load session:', error);
        }
    }

    newChat() {
        this.sessionId = this.generateSessionId();
        this.messages = [];

        this.updateSessionDisplay();
        this.renderMessages();
        this.connectWebSocket();

        // Show welcome screen for new chat
        if (this.welcomeScreen) this.welcomeScreen.style.display = 'flex';
        if (this.messagesContainer) this.messagesContainer.style.display = 'none';

        // Focus input
        this.messageInput.focus();
    }

    updateSessionDisplay() {
        if (this.sessionIdDisplay) {
            this.sessionIdDisplay.textContent = this.sessionId;
        }
    }

    connectWebSocket() {
        // Close existing connection
        if (this.ws) {
            this.ws.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat/${this.sessionId}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
    }

    updateConnectionStatus(connected) {
        if (this.connectionStatus) {
            this.connectionStatus.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
            this.connectionStatus.innerHTML = `
                <span class="connection-dot"></span>
                ${connected ? 'Connected' : 'Disconnected'}
            `;
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'stream':
                this.handleStreamChunk(data.chunk);
                break;
            case 'status':
                this.updateStatus(data.stage, data.detail);
                break;
            case 'complete':
                this.handleComplete(data);
                break;
            case 'error':
                this.handleError(data.error);
                break;
            case 'pong':
                // Heartbeat response
                break;
        }
    }

    handleStreamChunk(chunk) {
        this.streamingContent += chunk;
        this.updateStreamingMessage();
    }

    updateStreamingMessage() {
        const streamingEl = document.getElementById('streaming-message');
        if (streamingEl) {
            const textEl = streamingEl.querySelector('.message-text');
            if (textEl) {
                textEl.innerHTML = this.renderMarkdown(this.streamingContent);
            }
        }
    }

    handleComplete(data) {
        this.isProcessing = false;
        this.streamingContent = '';

        // Remove status indicator
        const statusEl = document.getElementById('status-indicator');
        if (statusEl) statusEl.remove();

        // Update or add the assistant message
        const streamingEl = document.getElementById('streaming-message');
        if (streamingEl) {
            streamingEl.removeAttribute('id');
            const textEl = streamingEl.querySelector('.message-text');
            if (textEl) {
                textEl.innerHTML = this.renderMarkdown(data.content);
            }

            // Add output path if present
            if (data.output_path) {
                const metaEl = streamingEl.querySelector('.message-meta');
                if (metaEl) {
                    metaEl.innerHTML += `<span class="output-path">Saved: ${data.output_path}</span>`;
                }
            }
        }

        // Add to messages array
        this.messages.push({
            role: 'assistant',
            content: data.content,
            interface: 'web',
            output_path: data.output_path
        });

        // Re-enable input
        this.setProcessing(false);

        // Refresh sessions list
        this.loadSessions();
    }

    handleError(error) {
        this.isProcessing = false;
        this.streamingContent = '';
        this.setProcessing(false);

        // Remove status
        const statusEl = document.getElementById('status-indicator');
        if (statusEl) statusEl.remove();

        // Remove streaming message
        const streamingEl = document.getElementById('streaming-message');
        if (streamingEl) streamingEl.remove();

        // Show error
        this.addErrorMessage(error);
    }

    updateStatus(stage, detail) {
        let statusEl = document.getElementById('status-indicator');

        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.id = 'status-indicator';
            statusEl.className = 'status-indicator';
            this.messagesContainer.appendChild(statusEl);
        }

        const emoji = this.getStatusEmoji(stage);
        statusEl.innerHTML = `
            <div class="status-spinner"></div>
            <span>${emoji} ${this.capitalizeFirst(stage)}${detail ? ': ' + detail : ''}</span>
        `;

        this.scrollToBottom();
    }

    getStatusEmoji(stage) {
        const emojis = {
            'processing': '⚙️',
            'analyzing': '🔍',
            'searching': '🌐',
            'reading': '📖',
            'generating': '✨',
            'generated': '✅',
            'saving': '💾',
            'saved': '📁',
            'decision': '🤔'
        };
        return emojis[stage.toLowerCase()] || '⚙️';
    }

    async sendMessage() {
        const content = this.messageInput.value.trim();
        if (!content || this.isProcessing) return;

        // Hide welcome screen
        if (this.welcomeScreen) this.welcomeScreen.style.display = 'none';
        if (this.messagesContainer) this.messagesContainer.style.display = 'block';

        // Add user message
        this.messages.push({
            role: 'user',
            content: content,
            interface: 'web'
        });
        this.renderMessages();

        // Clear input
        this.messageInput.value = '';
        this.autoResize();
        this.setProcessing(true);

        // Add streaming placeholder
        this.addStreamingMessage();

        // Send via WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'message',
                content: content
            }));
        } else {
            // Fallback to REST
            this.sendViaRest(content);
        }
    }

    async sendViaRest(content) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: content,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();

            // Remove streaming message
            const streamingEl = document.getElementById('streaming-message');
            if (streamingEl) streamingEl.remove();

            // Remove status
            const statusEl = document.getElementById('status-indicator');
            if (statusEl) statusEl.remove();

            if (data.success) {
                this.messages.push({
                    role: 'assistant',
                    content: data.content,
                    interface: 'web',
                    output_path: data.output_path
                });
                this.renderMessages();
            } else {
                this.addErrorMessage(data.error || 'Unknown error');
            }

        } catch (error) {
            console.error('REST request failed:', error);
            this.addErrorMessage(error.message);
        } finally {
            this.setProcessing(false);
        }
    }

    addStreamingMessage() {
        const messageEl = document.createElement('div');
        messageEl.id = 'streaming-message';
        messageEl.className = 'message assistant';
        messageEl.innerHTML = `
            <div class="message-avatar">J</div>
            <div class="message-content">
                <div class="message-role">Jotty</div>
                <div class="message-text"></div>
                <div class="message-meta">
                    <span class="message-interface">web</span>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
    }

    addErrorMessage(error) {
        const errorEl = document.createElement('div');
        errorEl.className = 'error-message';
        errorEl.textContent = `Error: ${error}`;
        this.messagesContainer.appendChild(errorEl);
        this.scrollToBottom();
    }

    renderMessages() {
        if (!this.messagesContainer) return;

        this.messagesContainer.innerHTML = this.messages.map(msg => `
            <div class="message ${msg.role}">
                <div class="message-avatar">${msg.role === 'user' ? 'U' : 'J'}</div>
                <div class="message-content">
                    <div class="message-role">${msg.role === 'user' ? 'You' : 'Jotty'}</div>
                    <div class="message-text">${this.renderMarkdown(msg.content)}</div>
                    <div class="message-meta">
                        <span class="message-interface">${msg.interface || 'web'}</span>
                        ${msg.output_path ? `<span class="output-path">Saved: ${msg.output_path}</span>` : ''}
                    </div>
                </div>
            </div>
        `).join('');

        this.scrollToBottom();
    }

    renderMarkdown(text) {
        if (!text) return '';

        // Escape HTML first
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Code blocks
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code class="language-${lang || ''}">${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Bold and italic
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Lists
        html = html.replace(/^- \[x\] (.+)$/gm, '<li class="checked">☑ $1</li>');
        html = html.replace(/^- \[ \] (.+)$/gm, '<li class="unchecked">☐ $1</li>');
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        html = html.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');

        // Wrap consecutive list items
        html = html.replace(/(<li[^>]*>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Paragraphs (lines with content)
        html = html.replace(/^(?!<[huplo]|$)(.+)$/gm, '<p>$1</p>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

        return html;
    }

    setProcessing(processing) {
        this.isProcessing = processing;
        this.sendBtn.disabled = processing;
        this.messageInput.disabled = processing;

        if (!processing) {
            this.messageInput.focus();
        }
    }

    handleKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    autoResize() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    scrollToBottom() {
        if (this.chatArea) {
            this.chatArea.scrollTop = this.chatArea.scrollHeight;
        }
    }

    toggleSidebar() {
        if (this.sidebar) {
            this.sidebar.classList.toggle('visible');
        }
    }

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.jottyApp = new JottyApp();
});
