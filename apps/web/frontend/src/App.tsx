/**
 * Jotty Web App
 * =============
 *
 * Standalone React app for testing.
 * Connects to backend via WebSocket.
 */

import React, { useState, useEffect, useRef } from 'react';
import './App.css';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [connected, setConnected] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [status, setStatus] = useState('');

  const wsRef = useRef<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Connect to WebSocket
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setStatus('Connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received:', data);

      switch (data.type) {
        case 'connected':
          setSessionId(data.session_id);
          setMessages([{
            role: 'system',
            content: `Welcome to Jotty AI!\n\n${data.message}\n\nCommands: ${data.commands_available}`,
            timestamp: new Date()
          }]);
          break;

        case 'message':
          setMessages(prev => [...prev, {
            role: data.role || 'assistant',
            content: data.content,
            timestamp: new Date(data.timestamp || Date.now())
          }]);
          setStatus('');
          break;

        case 'status':
          setStatus(data.message || data.state);
          break;

        case 'error':
          setMessages(prev => [...prev, {
            role: 'system',
            content: `❌ Error: ${data.message}`,
            timestamp: new Date()
          }]);
          setStatus('');
          break;

        case 'command_result':
          if (data.output) {
            setMessages(prev => [...prev, {
              role: 'system',
              content: data.output,
              timestamp: new Date()
            }]);
          }
          setStatus('');
          break;
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
      setStatus('Connection error');
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      setConnected(false);
      setStatus('Disconnected');
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();

    if (!input.trim() || !connected) return;

    // Add user message immediately
    setMessages(prev => [...prev, {
      role: 'user',
      content: input,
      timestamp: new Date()
    }]);

    // Send to server
    wsRef.current?.send(JSON.stringify({
      type: input.startsWith('/') ? 'command' : 'chat',
      content: input
    }));

    setInput('');
    setStatus('Sending...');
  };

  return (
    <div className="App">
      {/* Header */}
      <div className="header">
        <h1>🤖 Jotty AI</h1>
        <div className="status">
          {connected ? (
            <span className="connected">✅ Connected | Session: {sessionId.slice(0, 8)}</span>
          ) : (
            <span className="disconnected">❌ Disconnected</span>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-role">
              {msg.role === 'user' ? '👤' : msg.role === 'assistant' ? '🤖' : 'ℹ️'} {msg.role}
            </div>
            <div className="message-content">{msg.content}</div>
            <div className="message-time">
              {msg.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Status */}
      {status && (
        <div className="status-bar">
          {status}
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSend} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={connected ? "Type a message or /help for commands..." : "Connecting..."}
          disabled={!connected}
          className="input"
        />
        <button type="submit" disabled={!connected || !input.trim()} className="send-button">
          Send
        </button>
      </form>
    </div>
  );
}

export default App;
