/**
 * Chat Page - Main PWA Chat Interface
 * ====================================
 *
 * Main chat page using Jotty TypeScript SDK.
 * Supports text and voice input, streaming responses.
 */

'use client';

import { useState } from 'react';
import ChatLayout from '@/components/chat/ChatLayout';
import { useChat } from '@/lib/jotty/hooks';

export default function ChatPage() {
  const [sessionId] = useState<string>(() => `session-${Date.now()}`);
  const chat = useChat(sessionId);

  return (
    <ChatLayout
      messages={chat.messages}
      loading={chat.loading}
      error={chat.error}
      onSendMessage={chat.sendMessage}
      onClearMessages={chat.clearMessages}
    />
  );
}
