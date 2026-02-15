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
        this.isTemporary = false;

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
                    <button onclick="jottyApp.showConfigTab('mcp')">MCP Tools</button>
                    <button onclick="jottyApp.showConfigTab('models')">Models</button>
                </div>
                <div id="config-widgets" class="config-tab-content active">${widgetsHtml}</div>
                <div id="config-tools" class="config-tab-content">${toolsHtml}</div>
                <div id="config-mcp" class="config-tab-content">
                    <div class="config-category">
                        <h4>🔧 MCP Tools</h4>
                        <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 16px;">
                            External tools connected via Model Context Protocol
                        </p>
                        <div id="mcp-tools-list" class="mcp-tools-list">
                            <p>Loading MCP tools...</p>
                        </div>
                    </div>
                </div>
                <div id="config-models" class="config-tab-content">
                    <div class="config-category">
                        <h4>Model selection coming soon</h4>
                        <p style="color: var(--text-secondary); font-size: 13px;">
                            Configure AI providers and models in the next update.
                        </p>
                    </div>
                </div>
            `;

            // Load MCP tools when tab is shown
            this.loadMCPTools();
        }
    }

    async loadMCPTools() {
        try {
            const response = await fetch('/api/mcp/tools');
            const data = await response.json();

            const container = document.getElementById('mcp-tools-list');
            if (!container) return;

            if (!data.tools || data.tools.length === 0) {
                container.innerHTML = `
                    <p style="color: var(--text-secondary); font-size: 13px;">
                        ${data.connected ? 'No MCP tools available.' : 'MCP server not connected.'}
                        ${data.error ? `<br><span style="color: var(--error);">Error: ${data.error}</span>` : ''}
                    </p>
                `;
                return;
            }

            container.innerHTML = data.tools.map(tool => `
                <div class="mcp-tool-item">
                    <div class="tool-info">
                        <div class="tool-name">${tool.name}</div>
                        <div class="tool-description">${tool.description || 'No description'}</div>
                    </div>
                    <div class="tool-toggle ${tool.enabled ? 'enabled' : ''}"
                         onclick="jottyApp.toggleMCPTool('${tool.name}', this)">
                    </div>
                </div>
            `).join('');
        } catch (e) {
            console.error('Failed to load MCP tools:', e);
            const container = document.getElementById('mcp-tools-list');
            if (container) {
                container.innerHTML = `<p style="color: var(--error);">Failed to load MCP tools: ${e.message}</p>`;
            }
        }
    }

    toggleMCPTool(toolName, element) {
        element.classList.toggle('enabled');
        // Store preference in localStorage
        let enabledMCPTools = JSON.parse(localStorage.getItem('jotty_mcp_tools') || '[]');
        if (element.classList.contains('enabled')) {
            if (!enabledMCPTools.includes(toolName)) enabledMCPTools.push(toolName);
        } else {
            enabledMCPTools = enabledMCPTools.filter(t => t !== toolName);
        }
        localStorage.setItem('jotty_mcp_tools', JSON.stringify(enabledMCPTools));
    }

    async executeMCPTool(toolName, args = {}) {
        try {
            const response = await fetch('/api/mcp/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tool_name: toolName, arguments: args })
            });
            return await response.json();
        } catch (e) {
            console.error('MCP tool execution failed:', e);
            return { success: false, error: e.message };
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
        this.isTemporary = false;

        this.updateSessionDisplay();
        this.renderMessages();
        this.connectWebSocket();

        // Remove temp chat indicator if present
        const tempIndicator = document.querySelector('.temp-chat-indicator');
        if (tempIndicator) tempIndicator.remove();

        // Show welcome screen for new chat
        if (this.welcomeScreen) this.welcomeScreen.style.display = 'flex';
        if (this.messagesContainer) this.messagesContainer.style.display = 'none';

        // Focus input
        this.messageInput.focus();
    }

    updateSessionDisplay() {
        if (this.sessionIdDisplay) {
            const suffix = this.isTemporary ? ' (Temp)' : '';
            this.sessionIdDisplay.textContent = this.sessionId + suffix;
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
            // LibreChat-style events
            case 'searching':
                this.handleSearching(data.query, data.provider);
                break;
            case 'search_results':
                this.handleSearchResults(data.count, data.sources);
                break;
            case 'tool_call_start':
                this.handleToolCallStart(data.tool_name, data.args, data.tool_call_id);
                break;
            case 'tool_call_result':
                this.handleToolCallResult(data.tool_call_id, data.result, data.duration_ms);
                break;
            case 'artifact':
                this.handleArtifact(data.artifact_id, data.artifact_type, data.content);
                break;
            case 'code_output':
                this.handleCodeOutput(data.output, data.is_error, data.execution_id);
                break;
        }
    }

    // =========================================================================
    // Web Search UI Handlers
    // =========================================================================

    handleSearching(query, provider) {
        let statusEl = document.getElementById('search-indicator');
        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.id = 'search-indicator';
            statusEl.className = 'search-indicator';
            this.messagesContainer.appendChild(statusEl);
        }

        statusEl.innerHTML = `
            <div class="search-spinner"></div>
            <span class="search-icon">🔍</span>
            <span class="search-text">Searching: "${query}"</span>
            ${provider ? `<span class="search-provider">${provider}</span>` : ''}
        `;
        this.scrollToBottom();
    }

    handleSearchResults(count, sources) {
        // Remove search indicator
        const searchEl = document.getElementById('search-indicator');
        if (searchEl) searchEl.remove();

        // Add citation cards if sources provided
        if (sources && sources.length > 0) {
            this.pendingSources = sources;
        }
    }

    renderCitationCards(sources) {
        if (!sources || sources.length === 0) return '';

        return `
            <div class="citation-cards">
                <div class="citation-header">
                    <span class="citation-icon">📚</span>
                    <span>Sources (${sources.length})</span>
                </div>
                <div class="citation-list">
                    ${sources.map((s, i) => `
                        <a href="${s.url}" target="_blank" class="citation-card" title="${s.title}">
                            <span class="citation-number">${i + 1}</span>
                            <span class="citation-source">${s.source || this.extractDomain(s.url)}</span>
                            <span class="citation-title">${s.title}</span>
                        </a>
                    `).join('')}
                </div>
            </div>
        `;
    }

    extractDomain(url) {
        try {
            return new URL(url).hostname.replace('www.', '');
        } catch {
            return url;
        }
    }

    // =========================================================================
    // Tool Call UI Handlers
    // =========================================================================

    handleToolCallStart(toolName, args, toolCallId) {
        const toolEl = document.createElement('div');
        toolEl.id = `tool-call-${toolCallId}`;
        toolEl.className = 'tool-call-card';
        toolEl.innerHTML = `
            <div class="tool-call-header">
                <span class="tool-icon">🔧</span>
                <span class="tool-name">${toolName}</span>
                <span class="tool-status running">Running...</span>
            </div>
            <div class="tool-call-args">
                <pre>${JSON.stringify(args, null, 2)}</pre>
            </div>
        `;
        this.messagesContainer.appendChild(toolEl);
        this.scrollToBottom();
    }

    handleToolCallResult(toolCallId, result, durationMs) {
        const toolEl = document.getElementById(`tool-call-${toolCallId}`);
        if (toolEl) {
            const statusEl = toolEl.querySelector('.tool-status');
            if (statusEl) {
                statusEl.className = 'tool-status completed';
                statusEl.textContent = `${durationMs}ms`;
            }

            // Add result preview
            const resultEl = document.createElement('div');
            resultEl.className = 'tool-call-result';
            resultEl.innerHTML = `<pre>${JSON.stringify(result, null, 2).substring(0, 500)}${JSON.stringify(result).length > 500 ? '...' : ''}</pre>`;
            toolEl.appendChild(resultEl);
        }
    }

    // =========================================================================
    // Artifacts UI Handlers
    // =========================================================================

    handleArtifact(artifactId, artifactType, content) {
        // Store artifact for later rendering
        if (!this.artifacts) this.artifacts = {};
        this.artifacts[artifactId] = { type: artifactType, content };
    }

    renderArtifactInline(artifact) {
        const { artifact_id, artifact_type, content, language, title } = artifact;

        switch (artifact_type) {
            case 'mermaid':
                return `
                    <div class="artifact-container mermaid-container" data-artifact-id="${artifact_id}">
                        <div class="artifact-header">
                            <span class="artifact-icon">📊</span>
                            <span class="artifact-title">${title || 'Diagram'}</span>
                            <button class="artifact-action" onclick="jottyApp.openArtifactFullscreen('${artifact_id}')">⛶</button>
                            <button class="artifact-action" onclick="jottyApp.copyArtifact('${artifact_id}')">📋</button>
                        </div>
                        <div class="mermaid-preview" id="mermaid-${artifact_id}">${content}</div>
                    </div>
                `;

            case 'html':
                return `
                    <div class="artifact-container html-container" data-artifact-id="${artifact_id}">
                        <div class="artifact-header">
                            <span class="artifact-icon">🌐</span>
                            <span class="artifact-title">${title || 'HTML Preview'}</span>
                            <button class="artifact-action" onclick="jottyApp.openArtifactFullscreen('${artifact_id}')">⛶</button>
                            <button class="artifact-action" onclick="jottyApp.copyArtifact('${artifact_id}')">📋</button>
                        </div>
                        <iframe class="html-preview" srcdoc="${this.escapeHtml(content)}" sandbox="allow-scripts"></iframe>
                    </div>
                `;

            case 'code':
                const executable = ['python', 'py', 'javascript', 'js'].includes(language?.toLowerCase());
                return `
                    <div class="artifact-container code-container" data-artifact-id="${artifact_id}">
                        <div class="artifact-header">
                            <span class="artifact-icon">💻</span>
                            <span class="artifact-title">${title || language?.toUpperCase() || 'Code'}</span>
                            ${executable ? `<button class="artifact-action run-btn" onclick="jottyApp.runCode('${artifact_id}')">▶ Run</button>` : ''}
                            <button class="artifact-action" onclick="jottyApp.copyArtifact('${artifact_id}')">📋</button>
                        </div>
                        <pre><code class="language-${language || ''}">${this.escapeHtml(content)}</code></pre>
                        <div class="code-output" id="output-${artifact_id}" style="display: none;"></div>
                    </div>
                `;

            case 'svg':
                return `
                    <div class="artifact-container svg-container" data-artifact-id="${artifact_id}">
                        <div class="artifact-header">
                            <span class="artifact-icon">🎨</span>
                            <span class="artifact-title">${title || 'SVG Image'}</span>
                            <button class="artifact-action" onclick="jottyApp.copyArtifact('${artifact_id}')">📋</button>
                        </div>
                        <div class="svg-preview">${content}</div>
                    </div>
                `;

            default:
                return `<pre><code>${this.escapeHtml(content)}</code></pre>`;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async openArtifactFullscreen(artifactId) {
        const artifact = this.artifacts?.[artifactId];
        if (!artifact) return;

        const modal = document.createElement('div');
        modal.className = 'artifact-modal';
        modal.innerHTML = `
            <div class="artifact-modal-content">
                <div class="artifact-modal-header">
                    <span>Artifact View</span>
                    <button onclick="this.closest('.artifact-modal').remove()">✕</button>
                </div>
                <div class="artifact-modal-body">
                    ${artifact.type === 'html'
                        ? `<iframe srcdoc="${this.escapeHtml(artifact.content)}" sandbox="allow-scripts"></iframe>`
                        : artifact.type === 'mermaid'
                        ? `<div class="mermaid">${artifact.content}</div>`
                        : `<pre>${this.escapeHtml(artifact.content)}</pre>`
                    }
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Re-render mermaid if needed
        if (artifact.type === 'mermaid' && window.mermaid) {
            await mermaid.run({ nodes: modal.querySelectorAll('.mermaid') });
        }
    }

    copyArtifact(artifactId) {
        const artifact = this.artifacts?.[artifactId];
        if (!artifact) return;

        navigator.clipboard.writeText(artifact.content).then(() => {
            this.showToast('Copied to clipboard');
        });
    }

    // =========================================================================
    // Code Interpreter UI
    // =========================================================================

    async runCode(artifactId) {
        const artifact = this.artifacts?.[artifactId];
        if (!artifact || artifact.type !== 'code') return;

        const outputEl = document.getElementById(`output-${artifactId}`);
        if (!outputEl) return;

        outputEl.style.display = 'block';
        outputEl.innerHTML = '<div class="code-running">Running...</div>';

        try {
            const response = await fetch('/api/code/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: artifact.content,
                    language: artifact.language || 'python'
                })
            });

            const result = await response.json();

            if (result.success) {
                outputEl.innerHTML = `
                    <div class="code-success">
                        <div class="output-header">Output (${result.duration_ms}ms)</div>
                        <pre>${this.escapeHtml(result.output)}</pre>
                    </div>
                `;
            } else {
                outputEl.innerHTML = `
                    <div class="code-error">
                        <div class="output-header">Error</div>
                        <pre>${this.escapeHtml(result.error || 'Execution failed')}</pre>
                    </div>
                `;
            }
        } catch (e) {
            outputEl.innerHTML = `<div class="code-error">Error: ${e.message}</div>`;
        }
    }

    handleCodeOutput(output, isError, executionId) {
        const outputEl = document.getElementById(`output-${executionId}`);
        if (outputEl) {
            if (isError) {
                outputEl.innerHTML += `<div class="code-error-line">${this.escapeHtml(output)}</div>`;
            } else {
                outputEl.innerHTML += `<div class="code-output-line">${this.escapeHtml(output)}</div>`;
            }
        }
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
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

        // Remove search indicator if present
        const searchEl = document.getElementById('search-indicator');
        if (searchEl) searchEl.remove();

        // Get content from nested result if present (WS format)
        const content = data.result?.content || data.content || '';
        const outputPath = data.result?.output_path || data.output_path;
        const messageId = data.result?.message_id || data.message_id || `msg-${Date.now()}`;

        // Update or add the assistant message
        const streamingEl = document.getElementById('streaming-message');
        if (streamingEl) {
            streamingEl.removeAttribute('id');
            streamingEl.setAttribute('data-message-id', messageId);
            const textEl = streamingEl.querySelector('.message-text');
            if (textEl) {
                // Render with artifact support
                textEl.innerHTML = this.renderMessageWithArtifacts(content, messageId);
            }

            // Add citations if pending
            if (this.pendingSources && this.pendingSources.length > 0) {
                const citationHtml = this.renderCitationCards(this.pendingSources);
                textEl.insertAdjacentHTML('afterend', citationHtml);
                this.pendingSources = null;
            }

            // Add output path if present
            if (outputPath) {
                const metaEl = streamingEl.querySelector('.message-meta');
                if (metaEl) {
                    metaEl.innerHTML += `<span class="output-path">Saved: ${outputPath}</span>`;
                }
            }
        }

        // Initialize mermaid diagrams
        this.initMermaidDiagrams();

        // Add to messages array
        this.messages.push({
            role: 'assistant',
            content: content,
            message_id: messageId,
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

        // Extract and store artifacts for later use
        this.artifacts = {};

        this.messagesContainer.innerHTML = this.messages.map((msg, index) => {
            // Extract artifacts from assistant messages
            let renderedContent = this.renderMarkdown(msg.content);
            if (msg.role === 'assistant') {
                renderedContent = this.renderMessageWithArtifacts(msg.content, msg.message_id || `msg-${index}`);
            }

            // Add citations if available
            const citations = msg.sources ? this.renderCitationCards(msg.sources) : '';

            return `
                <div class="message ${msg.role}" data-message-id="${msg.message_id || ''}" data-branch="${msg.branch_id || 'main'}">
                    <div class="message-avatar">${msg.role === 'user' ? 'U' : 'J'}</div>
                    <div class="message-content">
                        <div class="message-header">
                            <div class="message-role">${msg.role === 'user' ? 'You' : 'Jotty'}</div>
                            <div class="message-actions">
                                ${msg.role === 'user' ? `
                                    <button class="msg-action-btn edit-btn" onclick="jottyApp.editMessage('${msg.message_id || ''}')" title="Edit & Regenerate">
                                        <span>✏️</span>
                                    </button>
                                ` : ''}
                                <button class="msg-action-btn copy-btn" onclick="jottyApp.copyMessage('${msg.message_id || ''}')" title="Copy">
                                    <span>📋</span>
                                </button>
                                ${msg.branch_id && msg.branch_id !== 'main' ? `
                                    <span class="branch-badge">${msg.branch_id}</span>
                                ` : ''}
                            </div>
                        </div>
                        <div class="message-text">${renderedContent}</div>
                        ${citations}
                        <div class="message-meta">
                            <span class="message-interface">${msg.interface || 'web'}</span>
                            ${msg.output_path ? `<span class="output-path">Saved: ${msg.output_path}</span>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        // Render mermaid diagrams
        this.initMermaidDiagrams();

        this.scrollToBottom();
    }

    renderMessageWithArtifacts(content, messageId) {
        if (!content) return '';

        // Extract artifacts from code blocks
        const artifactPattern = /```(\w+)?\n([\s\S]*?)```/g;
        let result = content;
        let match;
        let artifactIndex = 0;

        while ((match = artifactPattern.exec(content)) !== null) {
            const language = match[1] || '';
            const code = match[2].trim();
            const artifactId = `${messageId}-artifact-${artifactIndex++}`;

            // Store artifact
            this.artifacts[artifactId] = {
                type: this.getArtifactType(language),
                content: code,
                language: language
            };

            // Check if it's a special artifact type
            if (language.toLowerCase() === 'mermaid') {
                const artifactHtml = this.renderArtifactInline({
                    artifact_id: artifactId,
                    artifact_type: 'mermaid',
                    content: code,
                    language: 'mermaid',
                    title: 'Diagram'
                });
                result = result.replace(match[0], artifactHtml);
            } else if (language.toLowerCase() === 'html') {
                const artifactHtml = this.renderArtifactInline({
                    artifact_id: artifactId,
                    artifact_type: 'html',
                    content: code,
                    language: 'html',
                    title: 'HTML Preview'
                });
                result = result.replace(match[0], artifactHtml);
            } else if (['python', 'py', 'javascript', 'js'].includes(language.toLowerCase())) {
                const artifactHtml = this.renderArtifactInline({
                    artifact_id: artifactId,
                    artifact_type: 'code',
                    content: code,
                    language: language,
                    title: `${language.toUpperCase()} Code`
                });
                result = result.replace(match[0], artifactHtml);
            }
        }

        return this.renderMarkdown(result);
    }

    getArtifactType(language) {
        const lang = language.toLowerCase();
        if (lang === 'mermaid') return 'mermaid';
        if (['html', 'htm'].includes(lang)) return 'html';
        if (lang === 'svg') return 'svg';
        if (lang === 'json') return 'json';
        return 'code';
    }

    initMermaidDiagrams() {
        // Load mermaid.js if not already loaded
        if (!window.mermaid) {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js';
            script.onload = () => {
                mermaid.initialize({ startOnLoad: false, theme: this.theme === 'dark' ? 'dark' : 'default' });
                this.renderMermaidDiagrams();
            };
            document.head.appendChild(script);
        } else {
            this.renderMermaidDiagrams();
        }
    }

    renderMermaidDiagrams() {
        if (!window.mermaid) return;
        const diagrams = document.querySelectorAll('.mermaid-preview');
        diagrams.forEach(async (el) => {
            try {
                const id = el.id;
                const graphDefinition = el.textContent;
                const { svg } = await mermaid.render(`${id}-svg`, graphDefinition);
                el.innerHTML = svg;
            } catch (e) {
                el.innerHTML = `<div class="mermaid-error">Diagram error: ${e.message}</div>`;
            }
        });
    }

    // =========================================================================
    // Conversation Branching UI
    // =========================================================================

    async editMessage(messageId) {
        if (!messageId) return;

        // Find the message
        const msg = this.messages.find(m => m.message_id === messageId);
        if (!msg) return;

        // Show edit modal
        const modal = document.createElement('div');
        modal.className = 'edit-message-modal';
        modal.innerHTML = `
            <div class="edit-modal-content">
                <div class="edit-modal-header">
                    <span>Edit Message</span>
                    <button onclick="this.closest('.edit-message-modal').remove()">✕</button>
                </div>
                <div class="edit-modal-body">
                    <textarea id="edit-message-text">${msg.content}</textarea>
                    <div class="edit-options">
                        <label>
                            <input type="checkbox" id="create-branch-checkbox" checked>
                            Create new branch (preserve original)
                        </label>
                    </div>
                </div>
                <div class="edit-modal-footer">
                    <button class="btn-secondary" onclick="this.closest('.edit-message-modal').remove()">Cancel</button>
                    <button class="btn-primary" onclick="jottyApp.submitEditMessage('${messageId}')">Save & Regenerate</button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Focus textarea
        const textarea = document.getElementById('edit-message-text');
        if (textarea) {
            textarea.focus();
            textarea.setSelectionRange(textarea.value.length, textarea.value.length);
        }
    }

    async submitEditMessage(messageId) {
        const textarea = document.getElementById('edit-message-text');
        const createBranch = document.getElementById('create-branch-checkbox')?.checked ?? true;

        if (!textarea) return;

        const newContent = textarea.value.trim();
        if (!newContent) return;

        // Close modal
        document.querySelector('.edit-message-modal')?.remove();

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/messages/${messageId}/edit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    new_content: newContent,
                    create_branch: createBranch
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showToast(result.branch_id ? `Created branch: ${result.branch_id}` : 'Message updated');
                // Reload session to get updated messages
                await this.loadSession(this.sessionId);
                // Regenerate response for edited message
                this.messageInput.value = newContent;
                this.sendMessage();
            } else {
                this.showToast('Failed to edit message');
            }
        } catch (e) {
            console.error('Edit failed:', e);
            this.showToast('Error editing message');
        }
    }

    copyMessage(messageId) {
        const msg = this.messages.find(m => m.message_id === messageId);
        if (msg) {
            navigator.clipboard.writeText(msg.content).then(() => {
                this.showToast('Copied to clipboard');
            });
        }
    }

    async loadBranches() {
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/branches`);
            const data = await response.json();

            this.branches = data.branches || [];
            this.activeBranch = data.active_branch || 'main';
            this.branchTree = data.tree || {};

            this.renderBranchSelector();
        } catch (e) {
            console.error('Failed to load branches:', e);
        }
    }

    renderBranchSelector() {
        const container = document.getElementById('branch-selector');
        if (!container) return;
        if (!this.branches || this.branches.length <= 1) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'flex';
        container.innerHTML = `
            <span class="branch-icon">🔀</span>
            <select onchange="jottyApp.switchBranch(this.value)">
                ${this.branches.map(b => `
                    <option value="${b}" ${b === this.activeBranch ? 'selected' : ''}>${b}</option>
                `).join('')}
            </select>
            <button class="branch-tree-btn" onclick="jottyApp.showBranchTree()" title="Show branch tree">
                🌳
            </button>
        `;
    }

    async switchBranch(branchId) {
        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/branch/switch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ branch_id: branchId })
            });

            const result = await response.json();

            if (result.success) {
                this.activeBranch = branchId;
                // Reload session to get branch-specific messages
                await this.loadSession(this.sessionId);
                this.showToast(`Switched to branch: ${branchId}`);
            }
        } catch (e) {
            console.error('Failed to switch branch:', e);
        }
    }

    showBranchTree() {
        if (!this.branchTree) return;

        const modal = document.createElement('div');
        modal.className = 'branch-tree-modal';
        modal.innerHTML = `
            <div class="branch-tree-content">
                <div class="branch-tree-header">
                    <span>Conversation Branches</span>
                    <button onclick="this.closest('.branch-tree-modal').remove()">✕</button>
                </div>
                <div class="branch-tree-body">
                    ${Object.entries(this.branchTree).map(([branchId, info]) => `
                        <div class="branch-item ${branchId === this.activeBranch ? 'active' : ''}" onclick="jottyApp.switchBranch('${branchId}'); this.closest('.branch-tree-modal').remove();">
                            <span class="branch-name">${branchId}</span>
                            <span class="branch-info">${info.message_count} messages</span>
                            ${info.parent_branch ? `<span class="branch-parent">from ${info.parent_branch}</span>` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
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

    // ==========================================================================
    // SHAREABLE LINKS
    // ==========================================================================

    async shareConversation(title = null, expiresInDays = null) {
        if (!this.sessionId) {
            this.showToast('No active session to share', 'error');
            return;
        }

        try {
            const response = await fetch('/api/share/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    title: title,
                    expires_in_days: expiresInDays
                })
            });

            if (!response.ok) throw new Error('Failed to create share link');
            const data = await response.json();

            this.showShareDialog(data);
        } catch (error) {
            console.error('Share error:', error);
            this.showToast('Failed to create share link', 'error');
        }
    }

    showShareDialog(shareData) {
        const shareUrl = `${window.location.origin}${shareData.url}`;

        const dialog = document.createElement('div');
        dialog.className = 'modal-overlay';
        dialog.innerHTML = `
            <div class="modal share-modal">
                <div class="modal-header">
                    <h3>Share Conversation</h3>
                    <button class="modal-close" onclick="this.closest('.modal-overlay').remove()">×</button>
                </div>
                <div class="modal-body">
                    <p>Anyone with this link can view this conversation:</p>
                    <div class="share-url-container">
                        <input type="text" class="share-url" value="${shareUrl}" readonly>
                        <button class="copy-btn" onclick="navigator.clipboard.writeText('${shareUrl}').then(() => window.jottyApp.showToast('Link copied!', 'success'))">Copy</button>
                    </div>
                    <div class="share-qr" id="share-qr">
                        <p>Loading QR code...</p>
                    </div>
                    <div class="share-info">
                        <small>Token: ${shareData.link.token}</small><br>
                        <small>Expires: ${shareData.link.expires_at ? new Date(shareData.link.expires_at).toLocaleString() : 'Never'}</small>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove()">Close</button>
                </div>
            </div>
        `;

        document.body.appendChild(dialog);

        // Load QR code
        this.loadShareQRCode(shareData.link.token);
    }

    async loadShareQRCode(token) {
        try {
            const response = await fetch(`/api/share/${token}/qrcode?base_url=${encodeURIComponent(window.location.origin)}`);
            const data = await response.json();

            const qrDiv = document.getElementById('share-qr');
            if (data.qrcode) {
                qrDiv.innerHTML = `<img src="${data.qrcode}" alt="QR Code" style="max-width: 200px;">`;
            } else {
                qrDiv.innerHTML = `<p style="color: var(--text-muted);">QR code not available</p>`;
            }
        } catch (error) {
            console.error('QR code error:', error);
        }
    }

    async loadSessionShareLinks() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/share/session/${this.sessionId}`);
            const data = await response.json();
            return data.links;
        } catch (error) {
            console.error('Error loading share links:', error);
            return [];
        }
    }

    async revokeShareLink(token) {
        try {
            const response = await fetch(`/api/share/${token}/revoke`, { method: 'POST' });
            if (!response.ok) throw new Error('Failed to revoke');
            this.showToast('Share link revoked', 'success');
        } catch (error) {
            console.error('Revoke error:', error);
            this.showToast('Failed to revoke link', 'error');
        }
    }

    // ==========================================================================
    // TEMPORARY CHAT
    // ==========================================================================

    async newTemporaryChat() {
        try {
            const response = await fetch('/api/sessions/temporary', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ expiry_days: 30 })
            });

            if (!response.ok) throw new Error('Failed to create temporary session');
            const data = await response.json();

            this.sessionId = data.session_id;
            this.isTemporary = true;
            this.messages = [];

            // Update UI
            if (this.sessionIdDisplay) {
                this.sessionIdDisplay.textContent = `${this.sessionId} (Temp)`;
            }
            if (this.messagesContainer) {
                this.messagesContainer.innerHTML = '';
            }

            this.showTempChatIndicator(data.expires_at);
            this.connectWebSocket();
            this.showToast('Temporary chat created', 'info');

            // Hide welcome screen
            if (this.welcomeScreen) {
                this.welcomeScreen.style.display = 'none';
            }
        } catch (error) {
            console.error('Temp chat error:', error);
            this.showToast('Failed to create temporary chat', 'error');
        }
    }

    showTempChatIndicator(expiresAt) {
        // Remove existing indicator
        const existing = document.querySelector('.temp-chat-indicator');
        if (existing) existing.remove();

        if (!this.isTemporary) return;

        const indicator = document.createElement('div');
        indicator.className = 'temp-chat-indicator';
        indicator.innerHTML = `
            <span class="temp-icon">⏱</span>
            <span>Temporary Chat</span>
            <span class="temp-expires">Expires: ${new Date(expiresAt).toLocaleDateString()}</span>
            <button class="temp-convert" onclick="window.jottyApp.convertToRegular()">Keep</button>
        `;

        const chatArea = document.getElementById('chat-area');
        if (chatArea) {
            chatArea.insertBefore(indicator, chatArea.firstChild);
        }
    }

    async convertToRegular() {
        if (!this.sessionId || !this.isTemporary) return;

        try {
            const response = await fetch(`/api/sessions/${this.sessionId}/temporary?is_temporary=false`, {
                method: 'POST'
            });

            if (!response.ok) throw new Error('Failed to convert session');

            this.isTemporary = false;

            // Update UI
            const indicator = document.querySelector('.temp-chat-indicator');
            if (indicator) indicator.remove();

            if (this.sessionIdDisplay) {
                this.sessionIdDisplay.textContent = this.sessionId;
            }

            this.showToast('Chat saved permanently', 'success');
            this.loadSessions(); // Refresh session list
        } catch (error) {
            console.error('Convert error:', error);
            this.showToast('Failed to save chat', 'error');
        }
    }

    async cleanupExpiredSessions() {
        try {
            const response = await fetch('/api/sessions/cleanup', { method: 'POST' });
            const data = await response.json();

            if (data.deleted_count > 0) {
                this.showToast(`Cleaned up ${data.deleted_count} expired sessions`, 'info');
                this.loadSessions();
            }
        } catch (error) {
            console.error('Cleanup error:', error);
        }
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
