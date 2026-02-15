/**
 * Web Renderer (TypeScript/React)
 * ================================
 *
 * React-based renderer for Web PWA and Tauri.
 * Implements shared interface for consistent UI.
 */

import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

// Import types from shared models
interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  format: 'text' | 'markdown' | 'json' | 'html' | 'a2ui';
  event_type?: string;
  skill_name?: string;
  agent_name?: string;
  swarm_id?: string;
  progress?: number;
  progress_total?: number;
  progress_current?: number;
  attachments?: any[];
  ephemeral?: boolean;
  priority?: number;
  collapsible?: boolean;
  hidden?: boolean;
  id?: string;
  parent_id?: string;
  metadata?: Record<string, any>;
}

interface Status {
  state: string;
  message?: string;
  icon?: string;
  progress?: number;
  details?: Record<string, any>;
}

interface Error {
  message: string;
  error_type?: string;
  traceback?: string;
  recoverable?: boolean;
  metadata?: Record<string, any>;
}

/**
 * Web Message Renderer Component
 */
export const WebMessageRenderer: React.FC<{
  messages: Message[];
  onMessageUpdate?: (message: Message) => void;
}> = ({ messages, onMessageUpdate }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="message-list flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((message, index) => (
        <MessageBubble key={message.id || index} message={message} />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

/**
 * Single Message Bubble
 */
const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  // Get icon for message
  const icon = getStatusIcon(message.event_type);

  // Get progress text
  const progressText = getProgressText(message);

  return (
    <div
      className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''} ${
        message.hidden ? 'hidden' : ''
      }`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-blue-600' : isSystem ? 'bg-yellow-600' : 'bg-emerald-600'
        }`}
      >
        {isUser ? '👤' : isSystem ? 'ℹ️' : '🤖'}
      </div>

      {/* Message Content */}
      <div
        className={`flex-1 max-w-3xl rounded-lg p-4 ${
          isUser
            ? 'bg-blue-600/20 border border-blue-600/30'
            : isSystem
            ? 'bg-yellow-600/20 border border-yellow-600/30'
            : 'bg-gray-800 border border-gray-700'
        } ${message.ephemeral ? 'animate-pulse' : ''}`}
      >
        {/* Header with icon and progress */}
        {(icon || progressText) && (
          <div className="flex items-center gap-2 mb-2 text-sm text-gray-400">
            {icon && <span>{icon}</span>}
            {message.skill_name && <span>Skill: {message.skill_name}</span>}
            {message.agent_name && <span>Agent: {message.agent_name}</span>}
            {progressText && <span>{progressText}</span>}
          </div>
        )}

        {/* Content */}
        {message.format === 'markdown' ? (
          <MarkdownContent content={message.content} />
        ) : message.format === 'json' ? (
          <pre className="bg-gray-900 p-3 rounded overflow-x-auto">
            <code>{JSON.stringify(JSON.parse(message.content), null, 2)}</code>
          </pre>
        ) : (
          <div className="whitespace-pre-wrap">{message.content}</div>
        )}

        {/* Attachments */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-700">
            <div className="text-sm text-gray-400">
              📎 {message.attachments.length} attachment(s)
            </div>
          </div>
        )}

        {/* Timestamp */}
        <div className="mt-2 text-xs text-gray-500">
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

/**
 * Markdown Content with Syntax Highlighting
 */
const MarkdownContent: React.FC<{ content: string }> = ({ content }) => {
  return (
    <div className="prose prose-invert prose-sm max-w-none">
      <ReactMarkdown
        components={{
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';

            return !inline && language ? (
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={language}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code
                className="bg-gray-900 px-1 py-0.5 rounded text-emerald-400"
                {...props}
              >
                {children}
              </code>
            );
          },
          a({ node, children, href, ...props }: any) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-emerald-400 hover:underline"
                {...props}
              >
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

/**
 * Web Status Renderer Component
 */
export const WebStatusRenderer: React.FC<{
  status?: Status;
  error?: Error;
}> = ({ status, error }) => {
  if (error) {
    return <ErrorDisplay error={error} />;
  }

  if (!status) {
    return null;
  }

  return (
    <div className="status-bar bg-gray-800 border-t border-gray-700 p-3">
      <div className="flex items-center gap-3">
        {/* Icon */}
        {status.icon && <span className="text-xl">{status.icon}</span>}

        {/* Message */}
        <span className="text-sm text-gray-300">{status.message || status.state}</span>

        {/* Progress Bar */}
        {status.progress !== undefined && (
          <div className="flex-1 max-w-xs">
            <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-300"
                style={{ width: `${status.progress * 100}%` }}
              />
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {Math.round(status.progress * 100)}%
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Error Display Component
 */
const ErrorDisplay: React.FC<{ error: Error }> = ({ error }) => {
  const [showTraceback, setShowTraceback] = useState(false);

  return (
    <div className="error-display mx-4 my-3 p-4 bg-red-900/50 border border-red-700 rounded-lg">
      <div className="flex items-start gap-3">
        <span className="text-2xl">❌</span>
        <div className="flex-1">
          <h3 className="font-semibold text-red-300 mb-1">
            {error.error_type || 'Error'}
          </h3>
          <p className="text-sm text-red-200">{error.message}</p>

          {error.traceback && (
            <>
              <button
                onClick={() => setShowTraceback(!showTraceback)}
                className="mt-2 text-xs text-red-400 hover:text-red-300 underline"
              >
                {showTraceback ? 'Hide' : 'Show'} traceback
              </button>

              {showTraceback && (
                <pre className="mt-2 p-3 bg-red-950/50 rounded text-xs overflow-x-auto">
                  <code>{error.traceback}</code>
                </pre>
              )}
            </>
          )}

          {!error.recoverable && (
            <div className="mt-2 text-xs text-yellow-400">
              ⚠️ This error is not recoverable
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Web Input Handler Component
 */
export const WebInputHandler: React.FC<{
  onSend: (text: string) => void;
  onVoiceClick?: () => void;
  disabled?: boolean;
  placeholder?: string;
}> = ({ onSend, onVoiceClick, disabled = false, placeholder = 'Type a message...' }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const message = input.trim();
    if (!message || disabled) return;

    onSend(message);
    setInput('');

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-resize
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  return (
    <div className="input-handler border-t border-gray-700 bg-gray-800 p-4">
      <div className="flex items-end gap-2">
        {/* Voice Button */}
        {onVoiceClick && (
          <button
            onClick={onVoiceClick}
            disabled={disabled}
            className="flex-shrink-0 p-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition-colors"
            title="Voice input"
          >
            🎤
          </button>
        )}

        {/* Text Input */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          placeholder={disabled ? 'Waiting for response...' : placeholder}
          className="flex-1 bg-gray-700 text-gray-100 placeholder-gray-400 rounded-lg px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
          rows={1}
          style={{ minHeight: '48px', maxHeight: '200px' }}
        />

        {/* Send Button */}
        <button
          onClick={handleSend}
          disabled={!input.trim() || disabled}
          className="flex-shrink-0 p-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition-colors"
          title="Send message"
        >
          📤
        </button>
      </div>
    </div>
  );
};

/**
 * Helper Functions
 */

function getStatusIcon(eventType?: string): string {
  if (!eventType) return '';

  const iconMap: Record<string, string> = {
    start: '▶️',
    thinking: '🤔',
    planning: '📋',
    skill_start: '🔧',
    skill_progress: '⏳',
    skill_complete: '✅',
    agent_start: '🤖',
    agent_complete: '✓',
    memory_recall: '🧠',
    memory_store: '💾',
    validation_start: '🔍',
    validation_complete: '✓',
    learning_update: '📚',
    voice_stt_start: '🎤',
    voice_tts_start: '🔊',
    swarm_agent_start: '🐝',
    swarm_coordination: '🔀',
    error: '❌',
    complete: '✅',
  };

  return iconMap[eventType] || '•';
}

function getProgressText(message: Message): string | null {
  if (message.progress !== undefined) {
    return `${Math.round(message.progress * 100)}%`;
  }
  if (message.progress_current && message.progress_total) {
    return `Step ${message.progress_current}/${message.progress_total}`;
  }
  return null;
}

/**
 * Complete Chat Interface Component (combines all renderers)
 */
export const WebChatInterface: React.FC<{
  messages: Message[];
  status?: Status;
  error?: Error;
  onSend: (text: string) => void;
  onVoiceClick?: () => void;
  disabled?: boolean;
}> = ({ messages, status, error, onSend, onVoiceClick, disabled }) => {
  return (
    <div className="web-chat-interface flex flex-col h-full bg-gray-900 text-gray-100">
      {/* Messages */}
      <WebMessageRenderer messages={messages} />

      {/* Status Bar */}
      <WebStatusRenderer status={status} error={error} />

      {/* Input */}
      <WebInputHandler
        onSend={onSend}
        onVoiceClick={onVoiceClick}
        disabled={disabled}
      />
    </div>
  );
};

export default WebChatInterface;
