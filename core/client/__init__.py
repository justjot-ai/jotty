"""
Jotty Client SDK

TypeScript/JavaScript SDK for easy integration with Jotty servers.
Provides type-safe client for chat and workflow operations.

Minimal client code:
    import { JottyClient } from '@jotty/client';
    
    const client = new JottyClient('http://localhost:8080');
    
    // Chat
    const stream = await client.chat.stream('Hello', { history: [...] });
    
    // Workflow
    const result = await client.workflow.execute('Analyze data', { context: {...} });
"""

# Python client for testing (TypeScript SDK would be separate package)
from .python_client import JottyClient, JottyClientConfig

__all__ = [
    "JottyClient",
    "JottyClientConfig",
]
