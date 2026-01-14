/**
 * Example: Using DSPy Agents from TypeScript/Next.js
 * Demonstrates all three methods: execute, stream, and React hook
 */

import {
  executeDSPyAgent,
  streamDSPyAgent,
  useDSPyAgent,
  checkDSPyHealth,
  listDSPyAgents,
} from '@/lib/ai/agents/dspy-client';

// ============================================================================
// Example 1: Non-streaming execution (simple queries)
// ============================================================================

async function exampleNonStreaming() {
  try {
    const result = await executeDSPyAgent({
      agentId: 'research-assistant',
      query: 'Search for ideas about machine learning and summarize the top 3',
    });

    console.log('Reasoning:', result.reasoning);
    console.log('Tool Calls:', result.tool_calls);
    console.log('Response:', result.response);
  } catch (error) {
    console.error('DSPy execution failed:', error);
  }
}

// ============================================================================
// Example 2: Streaming execution (real-time UI updates)
// ============================================================================

async function exampleStreaming() {
  try {
    await streamDSPyAgent(
      {
        agentId: 'research-assistant',
        query: 'Analyze all ideas tagged with "AI" and create a summary',
      },
      (event) => {
        switch (event.type) {
          case 'reasoning':
            console.log('🤔 Reasoning:', event.data.reasoning);
            // Update UI: Show thinking indicator
            break;

          case 'tool_call':
            console.log('🔧 Tool Call:', event.data.name);
            // Update UI: Show tool execution status
            break;

          case 'tool_result':
            console.log('✅ Tool Result:', event.data.tool);
            // Update UI: Show tool result
            break;

          case 'response':
            console.log('💬 Response:', event.data.response);
            // Update UI: Show final response
            break;

          case 'done':
            console.log('✓ Completed');
            // Update UI: Hide loading indicators
            break;

          case 'error':
            console.error('❌ Error:', event.data.error);
            // Update UI: Show error message
            break;
        }
      }
    );
  } catch (error) {
    console.error('DSPy streaming failed:', error);
  }
}

// ============================================================================
// Example 3: React Component with DSPy Agent (Next.js)
// ============================================================================

'use client';

import React, { useState } from 'react';

export function ResearchAssistantDemo() {
  const { execute, loading, error } = useDSPyAgent('research-assistant');
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const response = await execute({ query });

    if (response) {
      setResult(response.response);
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Research Assistant (DSPy)</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your ideas..."
          className="w-full p-3 border rounded-lg"
          rows={4}
        />

        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50"
        >
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-800">Error: {error}</p>
        </div>
      )}

      {result && (
        <div className="mt-6 p-6 bg-gray-50 rounded-lg">
          <h2 className="font-semibold mb-2">Response:</h2>
          <p className="whitespace-pre-wrap">{result}</p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Example 4: Streaming UI with Real-time Updates
// ============================================================================

export function StreamingResearchDemo() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [reasoning, setReasoning] = useState('');
  const [toolCalls, setToolCalls] = useState<string[]>([]);
  const [response, setResponse] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setReasoning('');
    setToolCalls([]);
    setResponse('');

    try {
      await streamDSPyAgent({ agentId: 'research-assistant', query }, (event) => {
        switch (event.type) {
          case 'reasoning':
            setReasoning(event.data.reasoning);
            break;

          case 'tool_call':
            setToolCalls((prev) => [...prev, `${event.data.name}(${JSON.stringify(event.data.arguments)})`]);
            break;

          case 'response':
            setResponse(event.data.response);
            break;

          case 'done':
            setLoading(false);
            break;

          case 'error':
            alert(`Error: ${event.data.error}`);
            setLoading(false);
            break;
        }
      });
    } catch (error) {
      alert(`Error: ${error}`);
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Streaming Research Assistant</h1>

      <form onSubmit={handleSubmit}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
          className="w-full p-3 border rounded-lg"
          rows={3}
        />
        <button
          type="submit"
          disabled={loading}
          className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-lg"
        >
          {loading ? 'Processing...' : 'Ask'}
        </button>
      </form>

      {reasoning && (
        <div className="p-4 bg-blue-50 rounded-lg">
          <h3 className="font-semibold mb-2">🤔 Reasoning:</h3>
          <p className="text-sm">{reasoning}</p>
        </div>
      )}

      {toolCalls.length > 0 && (
        <div className="p-4 bg-yellow-50 rounded-lg">
          <h3 className="font-semibold mb-2">🔧 Tool Calls:</h3>
          <ul className="text-sm space-y-1">
            {toolCalls.map((call, i) => (
              <li key={i} className="font-mono">
                {call}
              </li>
            ))}
          </ul>
        </div>
      )}

      {response && (
        <div className="p-4 bg-green-50 rounded-lg">
          <h3 className="font-semibold mb-2">💬 Response:</h3>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Example 5: Health Check & Agent Discovery
// ============================================================================

async function exampleHealthAndDiscovery() {
  // Check DSPy bridge health
  const health = await checkDSPyHealth();
  console.log('DSPy Bridge Status:', health);

  if (!health.dspy_available) {
    console.warn('DSPy is not available');
    return;
  }

  // List all available agents
  const agents = await listDSPyAgents();
  console.log('Available Agents:', agents);

  // Use discovered agents
  for (const agent of agents) {
    console.log(`Agent: ${agent.name} (${agent.tools} tools)`);
  }
}

// ============================================================================
// Example 6: Integration with Existing Agent System
// ============================================================================

import type { AgentDefinition } from '@/lib/ai/agents/registry';

export const researchAgentDSPy: AgentDefinition = {
  id: 'research-assistant',
  name: 'Research Assistant (DSPy)',
  description: 'Helps research ideas using DSPy framework',
  category: 'research',
  useDSPy: true, // Flag to route to DSPy

  async execute(request) {
    if (request.stream) {
      // Streaming mode
      return new ReadableStream({
        async start(controller) {
          await streamDSPyAgent(
            {
              agentId: 'research-assistant',
              query: request.query,
              conversationHistory: request.conversationHistory,
            },
            (event) => {
              if (event.type === 'response') {
                controller.enqueue(new TextEncoder().encode(event.data.response));
              }
              if (event.type === 'done') {
                controller.close();
              }
            }
          );
        },
      });
    } else {
      // Non-streaming mode
      const result = await executeDSPyAgent({
        agentId: 'research-assistant',
        query: request.query,
        conversationHistory: request.conversationHistory,
      });

      return result.response;
    }
  },
};

// ============================================================================
// Example 7: Custom Agent Configuration
// ============================================================================

async function exampleCustomAgent() {
  const result = await executeDSPyAgent({
    agentId: 'my-custom-agent',
    query: 'Analyze sentiment of recent ideas',
    config: {
      name: 'Sentiment Analysis Agent',
      description: 'Analyzes sentiment and emotional tone of ideas',
      base_url: 'http://localhost:3000', // Override MCP server URL
    },
  });

  console.log('Custom agent result:', result);
}
