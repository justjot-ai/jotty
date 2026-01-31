/**
 * AG-UI Protocol Client for Vanilla JavaScript
 * =============================================
 *
 * Implements the AG-UI (Agent-User Interaction) protocol for Jotty.
 * AG-UI is an open standard by CopilotKit for agent-frontend communication.
 *
 * Protocol: https://docs.ag-ui.com/
 * GitHub: https://github.com/ag-ui-protocol/ag-ui
 *
 * Event Categories:
 * - Lifecycle: RunStarted, RunFinished, RunError
 * - Text: TextMessageStart, TextMessageContent, TextMessageEnd
 * - Tool: ToolCallStart, ToolCallArgs, ToolCallEnd, ToolCallResult
 * - State: StateSnapshot, StateDelta, MessagesSnapshot
 * - Activity: ActivitySnapshot, ActivityDelta
 * - Special: Raw, Custom
 */

class AGUIClient {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || '';
        this.threadId = options.threadId || this.generateId('thread');
        this.onEvent = options.onEvent || (() => {});
        this.onMessage = options.onMessage || (() => {});
        this.onToolCall = options.onToolCall || (() => {});
        this.onStateChange = options.onStateChange || (() => {});
        this.onActivity = options.onActivity || (() => {});
        this.onError = options.onError || console.error;

        // State management
        this.state = {};
        this.messages = [];
        this.activeRun = null;
        this.activeMessages = {};  // messageId -> content
        this.activeToolCalls = {}; // toolCallId -> data
        this.activities = {};      // messageId -> activity

        // Event source
        this.eventSource = null;
    }

    generateId(prefix = 'id') {
        return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Send a message to the agent
     */
    async sendMessage(content, options = {}) {
        const runId = this.generateId('run');
        const messageId = this.generateId('msg');

        // Add user message to local state
        this.messages.push({
            id: messageId,
            role: 'user',
            content: content,
            timestamp: new Date().toISOString()
        });

        // Build request
        const request = {
            threadId: this.threadId,
            runId: runId,
            messages: this.messages,
            state: this.state,
            ...options
        };

        // Connect to SSE endpoint
        return this.connectStream(request);
    }

    /**
     * Connect to AG-UI SSE stream
     */
    connectStream(request) {
        return new Promise((resolve, reject) => {
            // Close existing connection
            if (this.eventSource) {
                this.eventSource.close();
            }

            // Build URL with request as query param or use POST
            const url = new URL(`${this.baseUrl}/api/agui/run`, window.location.origin);

            // Use fetch with streaming for POST requests
            this.streamWithFetch(url.toString(), request, resolve, reject);
        });
    }

    async streamWithFetch(url, request, resolve, reject) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify(request)
            });

            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    resolve({ success: true, messages: this.messages });
                    break;
                }

                buffer += decoder.decode(value, { stream: true });

                // Parse SSE events
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data.trim()) {
                            try {
                                const event = JSON.parse(data);
                                this.handleEvent(event);
                            } catch (e) {
                                console.warn('Failed to parse event:', data);
                            }
                        }
                    }
                }
            }
        } catch (error) {
            this.onError(error);
            reject(error);
        }
    }

    /**
     * Handle AG-UI events
     */
    handleEvent(event) {
        // Call generic event handler
        this.onEvent(event);

        switch (event.type) {
            // === Lifecycle Events ===
            case 'RunStarted':
                this.handleRunStarted(event);
                break;
            case 'RunFinished':
                this.handleRunFinished(event);
                break;
            case 'RunError':
                this.handleRunError(event);
                break;

            // === Text Message Events ===
            case 'TextMessageStart':
                this.handleTextMessageStart(event);
                break;
            case 'TextMessageContent':
                this.handleTextMessageContent(event);
                break;
            case 'TextMessageEnd':
                this.handleTextMessageEnd(event);
                break;
            case 'TextMessageChunk':
                // Convenience event combining start+content+end
                this.handleTextMessageChunk(event);
                break;

            // === Tool Call Events ===
            case 'ToolCallStart':
                this.handleToolCallStart(event);
                break;
            case 'ToolCallArgs':
                this.handleToolCallArgs(event);
                break;
            case 'ToolCallEnd':
                this.handleToolCallEnd(event);
                break;
            case 'ToolCallResult':
                this.handleToolCallResult(event);
                break;

            // === State Management Events ===
            case 'StateSnapshot':
                this.handleStateSnapshot(event);
                break;
            case 'StateDelta':
                this.handleStateDelta(event);
                break;
            case 'MessagesSnapshot':
                this.handleMessagesSnapshot(event);
                break;

            // === Activity Events ===
            case 'ActivitySnapshot':
                this.handleActivitySnapshot(event);
                break;
            case 'ActivityDelta':
                this.handleActivityDelta(event);
                break;

            // === Special Events ===
            case 'Raw':
                this.handleRawEvent(event);
                break;
            case 'Custom':
                this.handleCustomEvent(event);
                break;

            default:
                console.log('Unknown AG-UI event:', event.type, event);
        }
    }

    // === Lifecycle Handlers ===

    handleRunStarted(event) {
        this.activeRun = {
            runId: event.runId,
            threadId: event.threadId,
            startTime: event.timestamp || new Date().toISOString()
        };
        console.log('AG-UI Run started:', event.runId);
    }

    handleRunFinished(event) {
        if (this.activeRun) {
            this.activeRun.endTime = event.timestamp || new Date().toISOString();
            this.activeRun.result = event.result;
        }
        console.log('AG-UI Run finished:', event.runId);
    }

    handleRunError(event) {
        this.onError(new Error(event.message || 'Run failed'));
        console.error('AG-UI Run error:', event.message, event.code);
    }

    // === Text Message Handlers ===

    handleTextMessageStart(event) {
        this.activeMessages[event.messageId] = {
            id: event.messageId,
            role: event.role || 'assistant',
            content: '',
            timestamp: event.timestamp || new Date().toISOString()
        };
    }

    handleTextMessageContent(event) {
        const msg = this.activeMessages[event.messageId];
        if (msg) {
            msg.content += event.delta || '';
            this.onMessage({
                type: 'stream',
                messageId: event.messageId,
                content: msg.content,
                delta: event.delta
            });
        }
    }

    handleTextMessageEnd(event) {
        const msg = this.activeMessages[event.messageId];
        if (msg) {
            this.messages.push(msg);
            delete this.activeMessages[event.messageId];
            this.onMessage({
                type: 'complete',
                messageId: event.messageId,
                content: msg.content
            });
        }
    }

    handleTextMessageChunk(event) {
        // Convenience format - handle as combined event
        if (!this.activeMessages[event.messageId]) {
            this.handleTextMessageStart(event);
        }
        if (event.delta) {
            this.handleTextMessageContent(event);
        }
    }

    // === Tool Call Handlers ===

    handleToolCallStart(event) {
        this.activeToolCalls[event.toolCallId] = {
            id: event.toolCallId,
            name: event.toolCallName,
            parentMessageId: event.parentMessageId,
            args: '',
            status: 'running'
        };
        this.onToolCall({
            type: 'start',
            toolCallId: event.toolCallId,
            toolName: event.toolCallName
        });
    }

    handleToolCallArgs(event) {
        const tool = this.activeToolCalls[event.toolCallId];
        if (tool) {
            tool.args += event.delta || '';
        }
    }

    handleToolCallEnd(event) {
        const tool = this.activeToolCalls[event.toolCallId];
        if (tool) {
            tool.status = 'completed';
            // Parse args if JSON
            try {
                tool.parsedArgs = JSON.parse(tool.args);
            } catch (e) {
                tool.parsedArgs = tool.args;
            }
            this.onToolCall({
                type: 'end',
                toolCallId: event.toolCallId,
                toolName: tool.name,
                args: tool.parsedArgs
            });
        }
    }

    handleToolCallResult(event) {
        const tool = this.activeToolCalls[event.toolCallId];
        if (tool) {
            tool.result = event.content;
            this.onToolCall({
                type: 'result',
                toolCallId: event.toolCallId,
                toolName: tool.name,
                result: event.content
            });
        }
    }

    // === State Management Handlers ===

    handleStateSnapshot(event) {
        this.state = event.snapshot || {};
        this.onStateChange({
            type: 'snapshot',
            state: this.state
        });
    }

    handleStateDelta(event) {
        // Apply JSON Patch operations
        if (event.delta && Array.isArray(event.delta)) {
            for (const op of event.delta) {
                this.applyPatchOperation(this.state, op);
            }
        }
        this.onStateChange({
            type: 'delta',
            delta: event.delta,
            state: this.state
        });
    }

    handleMessagesSnapshot(event) {
        this.messages = event.messages || [];
        this.onMessage({
            type: 'snapshot',
            messages: this.messages
        });
    }

    applyPatchOperation(obj, op) {
        const path = op.path.split('/').filter(p => p);
        let current = obj;

        for (let i = 0; i < path.length - 1; i++) {
            if (!(path[i] in current)) {
                current[path[i]] = {};
            }
            current = current[path[i]];
        }

        const lastKey = path[path.length - 1];
        switch (op.op) {
            case 'add':
            case 'replace':
                current[lastKey] = op.value;
                break;
            case 'remove':
                delete current[lastKey];
                break;
        }
    }

    // === Activity Handlers ===

    handleActivitySnapshot(event) {
        this.activities[event.messageId] = {
            id: event.messageId,
            type: event.activityType,
            content: event.content
        };
        this.onActivity({
            type: 'snapshot',
            activityId: event.messageId,
            activityType: event.activityType,
            content: event.content
        });
    }

    handleActivityDelta(event) {
        const activity = this.activities[event.messageId];
        if (activity && event.patch) {
            for (const op of event.patch) {
                this.applyPatchOperation(activity.content, op);
            }
            this.onActivity({
                type: 'delta',
                activityId: event.messageId,
                activityType: event.activityType,
                content: activity.content
            });
        }
    }

    // === Special Event Handlers ===

    handleRawEvent(event) {
        console.log('AG-UI Raw event from', event.source, ':', event.event);
    }

    handleCustomEvent(event) {
        console.log('AG-UI Custom event:', event.name, event.value);
    }

    /**
     * Close the connection
     */
    close() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
}

/**
 * AG-UI UI Components
 * Render AG-UI events as interactive UI components
 */
class AGUIRenderer {
    constructor(container, options = {}) {
        this.container = container;
        this.options = options;
        this.toolCallElements = {};
        this.activityElements = {};
    }

    /**
     * Render a tool call visualization
     */
    renderToolCall(toolCall) {
        let el = this.toolCallElements[toolCall.toolCallId];

        if (!el) {
            el = document.createElement('div');
            el.className = 'agui-tool-call';
            el.innerHTML = `
                <div class="agui-tool-header">
                    <span class="agui-tool-icon">🔧</span>
                    <span class="agui-tool-name">${toolCall.toolName}</span>
                    <span class="agui-tool-status">Running</span>
                </div>
                <div class="agui-tool-body">
                    <div class="agui-tool-args"></div>
                    <div class="agui-tool-result"></div>
                </div>
            `;
            this.container.appendChild(el);
            this.toolCallElements[toolCall.toolCallId] = el;
        }

        // Update based on event type
        if (toolCall.type === 'start') {
            el.querySelector('.agui-tool-status').textContent = 'Running';
            el.querySelector('.agui-tool-status').className = 'agui-tool-status running';
        } else if (toolCall.type === 'end') {
            el.querySelector('.agui-tool-status').textContent = 'Completed';
            el.querySelector('.agui-tool-status').className = 'agui-tool-status completed';
            if (toolCall.args) {
                el.querySelector('.agui-tool-args').innerHTML = `
                    <details>
                        <summary>Arguments</summary>
                        <pre>${JSON.stringify(toolCall.args, null, 2)}</pre>
                    </details>
                `;
            }
        } else if (toolCall.type === 'result') {
            el.querySelector('.agui-tool-result').innerHTML = `
                <details open>
                    <summary>Result</summary>
                    <pre>${typeof toolCall.result === 'object' ? JSON.stringify(toolCall.result, null, 2) : toolCall.result}</pre>
                </details>
            `;
        }

        return el;
    }

    /**
     * Render activity (like planning steps)
     */
    renderActivity(activity) {
        let el = this.activityElements[activity.activityId];

        if (!el) {
            el = document.createElement('div');
            el.className = `agui-activity agui-activity-${activity.activityType.toLowerCase()}`;
            this.container.appendChild(el);
            this.activityElements[activity.activityId] = el;
        }

        // Render based on activity type
        if (activity.activityType === 'PLAN') {
            el.innerHTML = this.renderPlanActivity(activity.content);
        } else if (activity.activityType === 'PROGRESS') {
            el.innerHTML = this.renderProgressActivity(activity.content);
        } else {
            el.innerHTML = `<pre>${JSON.stringify(activity.content, null, 2)}</pre>`;
        }

        return el;
    }

    renderPlanActivity(content) {
        if (!content || !content.steps) return '';

        return `
            <div class="agui-plan">
                <div class="agui-plan-header">📋 Plan</div>
                <div class="agui-plan-steps">
                    ${content.steps.map((step, i) => `
                        <div class="agui-plan-step ${step.status || ''}">
                            <span class="agui-step-number">${i + 1}</span>
                            <span class="agui-step-text">${step.text || step}</span>
                            ${step.status === 'completed' ? '<span class="agui-step-check">✓</span>' : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    renderProgressActivity(content) {
        const percent = content.percent || 0;
        return `
            <div class="agui-progress">
                <div class="agui-progress-header">${content.label || 'Progress'}</div>
                <div class="agui-progress-bar">
                    <div class="agui-progress-fill" style="width: ${percent}%"></div>
                </div>
                <div class="agui-progress-text">${percent}%</div>
            </div>
        `;
    }

    /**
     * Clear all rendered elements
     */
    clear() {
        Object.values(this.toolCallElements).forEach(el => el.remove());
        Object.values(this.activityElements).forEach(el => el.remove());
        this.toolCallElements = {};
        this.activityElements = {};
    }
}

// Export for use
window.AGUIClient = AGUIClient;
window.AGUIRenderer = AGUIRenderer;
