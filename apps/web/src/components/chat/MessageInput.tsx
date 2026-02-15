/**
 * MessageInput - Text Input with Voice Button
 * ===========================================
 *
 * Text input bar with send button and voice toggle.
 * Supports Enter to send, Shift+Enter for newline.
 */

'use client';

import { useState, useRef, KeyboardEvent } from 'react';
import { Send, Mic } from 'lucide-react';

interface MessageInputProps {
  onSendMessage: (content: string) => Promise<void>;
  onVoiceClick: () => void;
  disabled?: boolean;
}

export default function MessageInput({
  onSendMessage,
  onVoiceClick,
  disabled = false,
}: MessageInputProps) {
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = async () => {
    const message = input.trim();
    if (!message || sending || disabled) return;

    setInput('');
    setSending(true);

    try {
      await onSendMessage(message);
    } finally {
      setSending(false);
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  return (
    <div className="flex items-end gap-2 p-4">
      {/* Voice Button */}
      <button
        onClick={onVoiceClick}
        disabled={disabled || sending}
        className="flex-shrink-0 p-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition-colors"
        title="Voice input"
      >
        <Mic size={20} />
      </button>

      {/* Text Input */}
      <textarea
        ref={textareaRef}
        value={input}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        disabled={disabled || sending}
        placeholder={disabled ? 'Waiting for response...' : 'Type a message... (Enter to send, Shift+Enter for newline)'}
        className="flex-1 bg-gray-700 text-gray-100 placeholder-gray-400 rounded-lg px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
        rows={1}
        style={{ minHeight: '48px', maxHeight: '200px' }}
      />

      {/* Send Button */}
      <button
        onClick={handleSend}
        disabled={!input.trim() || disabled || sending}
        className="flex-shrink-0 p-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition-colors"
        title="Send message"
      >
        <Send size={20} />
      </button>
    </div>
  );
}
