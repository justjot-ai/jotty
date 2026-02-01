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

        // ==========================================================================
        // NEW: DRY Configuration from API (not hardcoded)
        // ==========================================================================
        this.capabilities = null;
        this.enabledWidgets = [];
        this.enabledTools = [];
        this.agents = [];

        // UI state
        this.chatMode = 'raw';  // raw | agent | swarm
        this.selectedAgent = null;
        this.swarmMode = 'auto';  // auto | manual | workflow
        this.attachments = [];
        this.isListening = false;
        this.recognition = null;

        // Theme (sync with JustJot.ai via postMessage)
        this.theme = localStorage.getItem('jotty_theme') || 'dark';

        this.init();
    }

    async init() {
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

        // ==========================================================================
        // NEW: Initialize new features
        // ==========================================================================
        await this.loadCapabilities();
        this.applyTheme();
        this.initVoice();
        this.initFileInput();
        this.syncThemeFromParent();
        this.loadAgents();

        // Initialize
        this.loadSessions();
        this.newChat();
    }

    // ==========================================================================
    // DRY: Load from unified registry API
    // ==========================================================================

    async loadCapabilities() {
        try {
            const resp = await fetch('/api/capabilities');
            this.capabilities = await resp.json();

            // Use defaults from registry
            if (this.capabilities.defaults) {
                this.enabledWidgets = this.capabilities.defaults.widgets || [];
                this.enabledTools = this.capabilities.defaults.tools || [];
            }

            // Override with localStorage if exists
            const savedWidgets = localStorage.getItem('jotty_widgets');
            if (savedWidgets) this.enabledWidgets = JSON.parse(savedWidgets);

            const savedTools = localStorage.getItem('jotty_tools');
            if (savedTools) this.enabledTools = JSON.parse(savedTools);

            console.log('Capabilities loaded:', this.capabilities);
        } catch (e) {
            console.error('Failed to load capabilities:', e);
        }
    }

    async loadAgents() {
        try {
            const resp = await fetch('/api/agents');
            const data = await resp.json();
            this.agents = data.agents || [];
            this.renderAgentSelector();
        } catch (e) {
            console.error('Failed to load agents:', e);
        }
    }

    // ==========================================================================
    // Theme Support (JustJot.ai sync)
    // ==========================================================================

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('jotty_theme', this.theme);
        this.applyTheme();
        this.updateContextPanel();
    }

    syncThemeFromParent() {
        window.addEventListener('message', (event) => {
            if (event.data && event.data.type === 'theme-change') {
                this.theme = event.data.theme;
                localStorage.setItem('jotty_theme', this.theme);
                this.applyTheme();
            }
        });
    }

    // ==========================================================================
    // Voice Recording (Web Speech API)
    // ==========================================================================

    initVoice() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.log('Voice input not supported');
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;

        this.recognition.onresult = (e) => {
            const transcript = Array.from(e.results)
                .map(r => r[0].transcript)
                .join('');
            this.messageInput.value = transcript;
            this.autoResize();
        };

        this.recognition.onerror = (e) => {
            console.error('Voice error:', e);
            this.stopVoice();
        };

        this.recognition.onend = () => {
            if (this.isListening) {
                // Restart if still listening
                this.recognition.start();
            }
        };
    }

    toggleVoice() {
        if (this.isListening) {
            this.stopVoice();
        } else {
            this.startVoice();
        }
    }

    startVoice() {
        if (!this.recognition) return;
        try {
            this.recognition.start();
            this.isListening = true;
            const voiceBtn = document.getElementById('voice-btn');
            if (voiceBtn) voiceBtn.classList.add('listening');
        } catch (e) {
            console.error('Failed to start voice:', e);
        }
    }

    stopVoice() {
        if (!this.recognition) return;
        this.recognition.stop();
        this.isListening = false;
        const voiceBtn = document.getElementById('voice-btn');
        if (voiceBtn) voiceBtn.classList.remove('listening');
    }

    // ==========================================================================
    // Image Attachments
    // ==========================================================================

    initFileInput() {
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files));
        }
    }

    handleFileSelect(files) {
        Array.from(files).forEach(file => {
            if (!file.type.startsWith('image/')) return;

            const reader = new FileReader();
            reader.onload = (e) => {
                this.attachments.push({
                    type: 'image',
                    name: file.name,
                    data: e.target.result,
                    size: file.size
                });
                this.renderAttachments();
            };
            reader.readAsDataURL(file);
        });
    }

    renderAttachments() {
        const container = document.getElementById('attachment-previews');
        if (!container) return;

        container.innerHTML = this.attachments.map((att, i) => `
            <div class="attachment-preview">
                <img src="${att.data}" alt="${att.name}" />
                <button class="remove-attachment" onclick="jottyApp.removeAttachment(${i})">×</button>
                <span class="attachment-name">${att.name}</span>
            </div>
        `).join('');

        container.style.display = this.attachments.length > 0 ? 'flex' : 'none';
    }

    removeAttachment(index) {
        this.attachments.splice(index, 1);
        this.renderAttachments();
    }

    // ==========================================================================
    // Agent/Swarm Mode Selector
    // ==========================================================================

    setChatMode(mode) {
        this.chatMode = mode;
        this.renderAgentSelector();

        if (mode !== 'raw' && this.agents.length === 0) {
            this.loadAgents();
        }
    }

    setSwarmMode(mode) {
        this.swarmMode = mode;
        this.renderAgentSelector();
    }

    selectAgent(agentId) {
        this.selectedAgent = agentId;
        this.renderAgentSelector();
    }

    renderAgentSelector() {
        const container = document.getElementById('agent-selector');
        if (!container) return;

        container.innerHTML = `
            <div class="mode-toggle">
                <button class="${this.chatMode === 'raw' ? 'active' : ''}"
                        onclick="jottyApp.setChatMode('raw')">LLM</button>
                <button class="${this.chatMode === 'agent' ? 'active' : ''}"
                        onclick="jottyApp.setChatMode('agent')">Agent</button>
                <button class="${this.chatMode === 'swarm' ? 'active' : ''}"
                        onclick="jottyApp.setChatMode('swarm')">Swarm</button>
            </div>

            ${this.chatMode === 'agent' ? `
                <select class="agent-dropdown" onchange="jottyApp.selectAgent(this.value)">
                    <option value="">Select agent...</option>
                    ${this.agents.map(a => `
                        <option value="${a.id}" ${this.selectedAgent === a.id ? 'selected' : ''}>
                            ${a.name}
                        </option>
                    `).join('')}
                </select>
            ` : ''}

            ${this.chatMode === 'swarm' ? `
                <div class="swarm-modes">
                    <button class="${this.swarmMode === 'auto' ? 'active' : ''}"
                            onclick="jottyApp.setSwarmMode('auto')">Auto</button>
                    <button class="${this.swarmMode === 'manual' ? 'active' : ''}"
                            onclick="jottyApp.setSwarmMode('manual')">Manual</button>
                    <button class="${this.swarmMode === 'workflow' ? 'active' : ''}"
                            onclick="jottyApp.setSwarmMode('workflow')">Workflow</button>
                </div>
            ` : ''}
        `;
    }

    // ==========================================================================
    // Configuration Modal (Widgets/Tools/Models)
    // ==========================================================================

    showConfigModal() {
        const modal = document.getElementById('config-modal');
        if (modal) {
            modal.classList.add('open');
            this.renderConfigTabs();
        }
    }

    hideConfigModal() {
        const modal = document.getElementById('config-modal');
        if (modal) modal.classList.remove('open');
    }

    showConfigTab(tab) {
        document.querySelectorAll('.config-tab-content').forEach(el => {
            el.classList.remove('active');
        });
        document.querySelectorAll('.config-tabs button').forEach(btn => {
            btn.classList.remove('active');
        });

        const tabContent = document.getElementById(`config-${tab}`);
        if (tabContent) tabContent.classList.add('active');

        const tabBtn = document.querySelector(`.config-tabs button[onclick*="${tab}"]`);
        if (tabBtn) tabBtn.classList.add('active');
    }

    renderConfigTabs() {
        if (!this.capabilities) return;

        const widgetsHtml = this.renderWidgetConfig();
        const toolsHtml = this.renderToolConfig();

        const container = document.getElementById('config-content');
        if (container) {
            container.innerHTML = `
                <div class="config-tabs">
                    <button class="active" onclick="jottyApp.showConfigTab('widgets')">Widgets</button>
                    <button onclick="jottyApp.showConfigTab('tools')">Tools</button>
                    <button onclick="jottyApp.showConfigTab('models')">Models</button>
                </div>
                <div id="config-widgets" class="config-tab-content active">${widgetsHtml}</div>
                <div id="config-tools" class="config-tab-content">${toolsHtml}</div>
                <div id="config-models" class="config-tab-content">
                    <div class="config-category">
                        <h4>Model selection coming soon</h4>
                        <p style="color: var(--text-secondary); font-size: 13px;">
                            Configure AI providers and models in the next update.
                        </p>
                    </div>
                </div>
            `;
        }
    }

    renderWidgetConfig() {
        if (!this.capabilities || !this.capabilities.widgets) return '<p>Loading widgets...</p>';

        const widgets = this.capabilities.widgets.widgets || [];
        const categories = [...new Set(widgets.map(w => w.category))];

        return categories.map(cat => `
            <div class="config-category">
                <h4>${cat}</h4>
                <div class="config-items">
                    ${widgets.filter(w => w.category === cat).map(w => `
                        <label class="config-item">
                            <input type="checkbox"
                                   ${this.enabledWidgets.includes(w.value) ? 'checked' : ''}
                                   onchange="jottyApp.toggleWidget('${w.value}')" />
                            <span class="item-icon">${w.icon || '📦'}</span>
                            <span class="item-label">${w.label}</span>
                        </label>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }

    renderToolConfig() {
        if (!this.capabilities || !this.capabilities.tools) return '<p>Loading tools...</p>';

        const tools = this.capabilities.tools.tools || [];
        const categories = [...new Set(tools.map(t => t.category))];

        return categories.map(cat => `
            <div class="config-category">
                <h4>${cat}</h4>
                <div class="config-items">
                    ${tools.filter(t => t.category === cat).map(t => `
                        <label class="config-item">
                            <input type="checkbox"
                                   ${this.enabledTools.includes(t.name) ? 'checked' : ''}
                                   onchange="jottyApp.toggleTool('${t.name}')" />
                            <span class="item-label">${t.name}</span>
                        </label>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }

    toggleWidget(value) {
        const idx = this.enabledWidgets.indexOf(value);
        if (idx >= 0) {
            this.enabledWidgets.splice(idx, 1);
        } else {
            this.enabledWidgets.push(value);
        }
        localStorage.setItem('jotty_widgets', JSON.stringify(this.enabledWidgets));
        this.renderConfigTabs();
        this.updateConfigBadge();
    }

    toggleTool(name) {
        const idx = this.enabledTools.indexOf(name);
        if (idx >= 0) {
            this.enabledTools.splice(idx, 1);
        } else {
            this.enabledTools.push(name);
        }
        localStorage.setItem('jotty_tools', JSON.stringify(this.enabledTools));
        this.renderConfigTabs();
        this.updateConfigBadge();
    }

    updateConfigBadge() {
        const btn = document.getElementById('config-btn');
        if (btn) {
            const count = this.enabledWidgets.length + this.enabledTools.length;
            btn.setAttribute('data-count', count);
        }
    }

    // ==========================================================================
    // Context Panel ("i" icon)
    // ==========================================================================

    toggleContextPanel() {
        const panel = document.getElementById('context-panel');
        if (panel) {
            panel.classList.toggle('open');
            if (panel.classList.contains('open')) {
                this.updateContextPanel();
            }
        }
    }

    updateContextPanel() {
        const panel = document.getElementById('context-panel-content');
        if (!panel) return;

        const modeLabel = this.chatMode === 'raw' ? 'Direct LLM' :
                          this.chatMode === 'agent' ? 'Single Agent' : 'Multi-Agent Swarm';

        panel.innerHTML = `
            <div class="context-section">
                <h4>💬 Conversation</h4>
                <p>Messages: ${this.messages.length}</p>
                <p>Session: ${this.sessionId}</p>
            </div>

            <div class="context-section">
                <h4>⚡ Execution Mode</h4>
                <p>Mode: ${modeLabel}</p>
                ${this.selectedAgent ? `<p>Agent: ${this.selectedAgent}</p>` : ''}
                ${this.chatMode === 'swarm' ? `<p>Swarm: ${this.swarmMode}</p>` : ''}
            </div>

            <div class="context-section">
                <h4>🎛️ Capabilities</h4>
                <p>Widgets: ${this.enabledWidgets.length} enabled</p>
                <p>Tools: ${this.enabledTools.length} enabled</p>
            </div>

            <div class="context-section">
                <h4>🎨 Theme</h4>
                <button onclick="jottyApp.toggleTheme()">
                    ${this.theme === 'dark' ? '☀️ Switch to Light' : '🌙 Switch to Dark'}
                </button>
            </div>
        `;
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
        if (!content && this.attachments.length === 0) return;
        if (this.isProcessing) return;

        // Hide welcome screen
        if (this.welcomeScreen) this.welcomeScreen.style.display = 'none';
        if (this.messagesContainer) this.messagesContainer.style.display = 'block';

        // Build message with attachments
        const userMessage = {
            role: 'user',
            content: content,
            interface: 'web',
            attachments: this.attachments.length > 0 ? [...this.attachments] : undefined
        };

        // Add user message
        this.messages.push(userMessage);
        this.renderMessages();

        // Clear input and attachments
        this.messageInput.value = '';
        this.attachments = [];
        this.renderAttachments();
        this.autoResize();
        this.setProcessing(true);

        // Stop voice if active
        if (this.isListening) {
            this.stopVoice();
        }

        // Add streaming placeholder
        this.addStreamingMessage();

        // Build payload with context
        const payload = {
            type: 'message',
            content: content,
            attachments: userMessage.attachments,
            mode: this.chatMode,
            agent: this.selectedAgent,
            swarmMode: this.chatMode === 'swarm' ? this.swarmMode : null,
            enabledWidgets: this.enabledWidgets,
            enabledTools: this.enabledTools
        };

        // Send via WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(payload));
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

// Expose for inline handlers
window.jottyApp = window.jottyApp || null;
