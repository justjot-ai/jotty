/**
 * ChatLayout - Main Layout Component
 * ==================================
 *
 * Three-column layout: Sidebar | Messages | (Settings)
 * Responsive: collapses to single column on mobile
 */

'use client';

import { useState } from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import VoiceRecorder from './VoiceRecorder';
import { Menu, Trash2, Settings as SettingsIcon } from 'lucide-react';

interface ChatLayoutProps {
  messages: Array<{role: string, content: string}>;
  loading: boolean;
  error: string | null;
  onSendMessage: (content: string) => Promise<void>;
  onClearMessages: () => void;
}

export default function ChatLayout({
  messages,
  loading,
  error,
  onSendMessage,
  onClearMessages,
}: ChatLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Sidebar */}
      <div
        className={`
          fixed md:relative z-20 h-full w-64 bg-gray-800 border-r border-gray-700
          transition-transform duration-200 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-700">
            <h1 className="text-xl font-bold text-emerald-400">Jotty AI</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              className="md:hidden p-1 hover:bg-gray-700 rounded"
            >
              ×
            </button>
          </div>

          {/* Conversations List */}
          <div className="flex-1 overflow-y-auto p-2">
            <button
              className="w-full p-3 text-left bg-emerald-600 hover:bg-emerald-500 rounded-lg mb-2"
              onClick={() => {
                onClearMessages();
                setSidebarOpen(false);
              }}
            >
              + New Chat
            </button>
            <div className="text-sm text-gray-400 p-2">
              Recent Chats
            </div>
            {/* Placeholder for recent chats */}
            <div className="text-xs text-gray-500 p-2">
              No recent chats
            </div>
          </div>

          {/* Sidebar Footer */}
          <div className="p-4 border-t border-gray-700">
            <button className="w-full flex items-center gap-2 p-2 hover:bg-gray-700 rounded">
              <SettingsIcon size={16} />
              <span className="text-sm">Settings</span>
            </button>
          </div>
        </div>
      </div>

      {/* Overlay for mobile sidebar */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-10 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-800">
          <button
            onClick={() => setSidebarOpen(true)}
            className="md:hidden p-2 hover:bg-gray-700 rounded"
          >
            <Menu size={20} />
          </button>
          <div className="flex-1 text-center md:text-left">
            <h2 className="font-semibold">Chat</h2>
            <p className="text-xs text-gray-400">
              {messages.length} messages
            </p>
          </div>
          <button
            onClick={onClearMessages}
            className="p-2 hover:bg-gray-700 rounded"
            title="Clear chat"
          >
            <Trash2 size={18} />
          </button>
        </div>

        {/* Messages */}
        <MessageList messages={messages} loading={loading} />

        {/* Error Display */}
        {error && (
          <div className="mx-4 mb-2 p-3 bg-red-900/50 border border-red-700 rounded-lg text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Input Area */}
        <div className="border-t border-gray-700 bg-gray-800">
          {voiceMode ? (
            <VoiceRecorder
              onTranscript={(text) => {
                onSendMessage(text);
                setVoiceMode(false);
              }}
              onCancel={() => setVoiceMode(false)}
            />
          ) : (
            <MessageInput
              onSendMessage={onSendMessage}
              onVoiceClick={() => setVoiceMode(true)}
              disabled={loading}
            />
          )}
        </div>
      </div>
    </div>
  );
}
