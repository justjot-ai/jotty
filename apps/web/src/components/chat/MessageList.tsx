/**
 * MessageList - Chat Messages Display
 * ===================================
 *
 * Scrollable message list with markdown rendering.
 * Auto-scrolls to bottom on new messages.
 */

'use client';

import { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Bot, User } from 'lucide-react';

interface Message {
  role: string;
  content: string;
}

interface MessageListProps {
  messages: Message[];
  loading: boolean;
}

export default function MessageList({ messages, loading }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.length === 0 && !loading && (
        <div className="flex flex-col items-center justify-center h-full text-gray-400">
          <Bot size={48} className="mb-4 text-emerald-400" />
          <h3 className="text-lg font-semibold mb-2">Welcome to Jotty AI</h3>
          <p className="text-sm text-center max-w-md">
            AI agent framework with chat, voice, workflows, and multi-agent swarms.
            Start a conversation or try voice input.
          </p>
        </div>
      )}

      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}

      {loading && (
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center">
            <Bot size={16} />
          </div>
          <div className="flex-1 bg-gray-800 rounded-lg p-4">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-blue-600' : 'bg-emerald-600'
        }`}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Message Content */}
      <div
        className={`flex-1 max-w-3xl rounded-lg p-4 ${
          isUser
            ? 'bg-blue-600/20 border border-blue-600/30'
            : 'bg-gray-800 border border-gray-700'
        }`}
      >
        <div className="prose prose-invert prose-sm max-w-none">
          <ReactMarkdown
            components={{
              // Customize markdown rendering
              code: ({ node, inline, className, children, ...props }: any) => {
                return inline ? (
                  <code className="bg-gray-900 px-1 py-0.5 rounded text-emerald-400" {...props}>
                    {children}
                  </code>
                ) : (
                  <pre className="bg-gray-900 p-3 rounded-lg overflow-x-auto">
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                );
              },
              a: ({ node, children, href, ...props }: any) => (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-400 hover:underline"
                  {...props}
                >
                  {children}
                </a>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
