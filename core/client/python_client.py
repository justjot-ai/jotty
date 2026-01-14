"""
Jotty Python Client

Python client SDK for interacting with Jotty HTTP servers.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass
import aiohttp


@dataclass
class JottyClientConfig:
    """Configuration for Jotty client."""
    base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    timeout: float = 300.0
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
            if self.api_key:
                self.headers['Authorization'] = f'Bearer {self.api_key}'


class JottyClient:
    """
    Python client for Jotty HTTP server.
    
    Usage:
        client = JottyClient('http://localhost:8080', api_key='...')
        
        # Chat
        result = await client.chat.execute('Hello', history=[])
        
        # Stream
        async for event in client.chat.stream('Hello'):
            print(event)
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        config: Optional[JottyClientConfig] = None
    ):
        """
        Initialize Jotty client.
        
        Args:
            base_url: Base URL of Jotty server
            api_key: Optional API key for authentication
            config: Optional client configuration
        """
        self.config = config or JottyClientConfig(
            base_url=base_url,
            api_key=api_key
        )
        self.base_url = self.config.base_url.rstrip('/')
    
    async def chat_execute(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute chat synchronously."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat/execute",
                json={
                    "message": message,
                    "history": history or [],
                    "agentId": agent_id
                },
                headers=self.config.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                return await response.json()
    
    async def chat_stream(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        agent_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat response."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat/stream",
                json={
                    "message": message,
                    "history": history or [],
                    "agentId": agent_id
                },
                headers=self.config.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            if data_str == '[DONE]':
                                return
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                pass
    
    async def workflow_execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        mode: str = "dynamic",
        agent_order: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute workflow synchronously."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/workflow/execute",
                json={
                    "goal": goal,
                    "context": context or {},
                    "mode": mode,
                    "agent_order": agent_order
                },
                headers=self.config.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                return await response.json()
    
    async def workflow_stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream workflow execution."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/workflow/stream",
                json={
                    "goal": goal,
                    "context": context or {}
                },
                headers=self.config.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                pass
    
    async def list_agents(self) -> Dict[str, Any]:
        """List available agents."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/agents",
                headers=self.config.headers
            ) as response:
                return await response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/health",
                headers=self.config.headers
            ) as response:
                return await response.json()
