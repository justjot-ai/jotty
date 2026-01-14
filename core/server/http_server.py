"""
Jotty HTTP Server

Production-ready HTTP server that provides:
- Chat endpoints (/api/chat/stream, /api/chat/execute)
- Workflow endpoints (/api/workflow/execute, /api/workflow/stream)
- Agent management (/api/agents, /api/health)
- Automatic SSE formatting for different clients
- Built-in authentication middleware
- Error handling and resilience
- Request logging

Minimal client integration - just configure and run!
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from pathlib import Path
from dataclasses import dataclass, field

try:
    from flask import Flask, jsonify, request, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from ..api import JottyAPI
from ..foundation.agent_config import AgentConfig
from ..foundation.data_structures import JottyConfig
from .middleware import AuthMiddleware, LoggingMiddleware, ErrorMiddleware
from .formats import SSEFormatter, useChatFormatter, OpenAIFormatter, AnthropicFormatter

logger = logging.getLogger(__name__)


@dataclass
class JottyServerConfig:
    """Configuration for Jotty HTTP Server."""
    port: int = 8080
    host: str = "0.0.0.0"
    debug: bool = False
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Authentication
    auth_enabled: bool = True
    auth_type: str = "bearer"  # "bearer", "clerk", "none"
    auth_validate_fn: Optional[Callable[[str], bool]] = None
    
    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Error handling
    enable_error_handling: bool = True
    show_error_details: bool = False  # Don't expose internal errors to clients
    
    # SSE Format
    sse_format: str = "usechat"  # "usechat", "openai", "anthropic", "raw"
    
    # Health check
    health_check_path: str = "/api/health"


class JottyHTTPServer:
    """
    Production-ready HTTP server for Jotty.
    
    Provides all necessary endpoints with minimal client code.
    
    Usage:
        from Jotty.server import JottyHTTPServer, JottyServerConfig
        from Jotty import AgentConfig, JottyConfig
        
        # Configure agents
        agents = [
            AgentConfig(
                name="Research Assistant",
                agent=my_dspy_module,
                architect_prompts=[],
                auditor_prompts=[]
            )
        ]
        
        # Create server
        server = JottyHTTPServer(
            agents=agents,
            config=JottyConfig(),
            server_config=JottyServerConfig(
                port=8080,
                auth_type="clerk"  # or "bearer", "none"
            )
        )
        
        # Run server
        server.run()
    """
    
    def __init__(
        self,
        agents: List[AgentConfig],
        config: Optional[JottyConfig] = None,
        server_config: Optional[JottyServerConfig] = None
    ):
        """
        Initialize Jotty HTTP Server.
        
        Args:
            agents: List of agent configurations
            config: Jotty configuration
            server_config: Server-specific configuration
        """
        if not FLASK_AVAILABLE:
            raise RuntimeError(
                "Flask is required for JottyHTTPServer. "
                "Install with: pip install flask flask-cors"
            )
        
        self.agents = agents
        self.config = config or JottyConfig()
        self.server_config = server_config or JottyServerConfig()
        
        # Create Flask app
        self.app = Flask(__name__)
        
        # Enable CORS if configured
        if self.server_config.enable_cors:
            CORS(self.app, origins=self.server_config.cors_origins)
        
        # Create Jotty API
        self.jotty_api = JottyAPI(
            agents=agents,
            config=config
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Register routes
        self._register_routes()
        
        logger.info(f"✅ Jotty HTTP Server initialized on port {self.server_config.port}")
    
    def _setup_middleware(self):
        """Setup middleware for authentication, logging, and error handling."""
        # Authentication middleware
        if self.server_config.auth_enabled:
            auth_middleware = AuthMiddleware(
                auth_type=self.server_config.auth_type,
                validate_fn=self.server_config.auth_validate_fn
            )
            self.app.before_request(auth_middleware.before_request)
        
        # Logging middleware
        if self.server_config.enable_logging:
            logging_middleware = LoggingMiddleware(
                log_level=self.server_config.log_level
            )
            self.app.before_request(logging_middleware.before_request)
            self.app.after_request(logging_middleware.after_request)
        
        # Error handling middleware
        if self.server_config.enable_error_handling:
            error_middleware = ErrorMiddleware(
                show_details=self.server_config.show_error_details
            )
            self.app.register_error_handler(Exception, error_middleware.handle_error)
    
    def _register_routes(self):
        """Register all HTTP routes."""
        
        # Health check
        @self.app.route(self.server_config.health_check_path, methods=['GET'])
        def health():
            """Health check endpoint."""
            return jsonify({
                "status": "ok",
                "service": "jotty",
                "version": "1.0.0",
                "agents": len(self.agents)
            })
        
        # Chat endpoints
        @self.app.route('/api/chat/stream', methods=['POST'])
        def chat_stream():
            """Stream chat response (SSE)."""
            return self._handle_chat_stream()
        
        @self.app.route('/api/chat/execute', methods=['POST'])
        def chat_execute():
            """Execute chat synchronously."""
            return self._handle_chat_execute()
        
        # Workflow endpoints
        @self.app.route('/api/workflow/execute', methods=['POST'])
        def workflow_execute():
            """Execute workflow synchronously."""
            return self._handle_workflow_execute()
        
        @self.app.route('/api/workflow/stream', methods=['POST'])
        def workflow_stream():
            """Stream workflow execution (SSE)."""
            return self._handle_workflow_stream()
        
        # Agent management
        @self.app.route('/api/agents', methods=['GET'])
        def list_agents():
            """List all available agents."""
            return jsonify({
                "success": True,
                "agents": [
                    {
                        "id": agent.name,
                        "name": agent.name,
                        "description": getattr(agent, 'description', '')
                    }
                    for agent in self.agents
                ]
            })
    
    def _handle_chat_stream(self) -> Response:
        """Handle chat streaming request."""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "Request body required"}), 400
            
            # Extract message and history
            messages_data = data.get('messages', [])
            message = data.get('message')
            history_data = data.get('history', [])
            
            # Handle useChat format (messages array)
            if messages_data:
                if not messages_data or len(messages_data) == 0:
                    return jsonify({"error": "messages array is required"}), 400
                last_msg = messages_data[-1]
                if last_msg.get('role') != 'user':
                    return jsonify({"error": "Last message must be from user"}), 400
                message = last_msg.get('content', '')
                if isinstance(message, list):
                    message = ' '.join([
                        item.get('text', '') for item in message
                        if item.get('type') == 'text'
                    ])
                history_data = messages_data[:-1]
            elif not message:
                return jsonify({"error": "message or messages is required"}), 400
            
            agent_id = data.get('agentId')
            
            # Get SSE formatter based on config
            formatter = self._get_sse_formatter()
            
            def generate_events():
                """Generate SSE events."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Convert history to ChatMessage objects
                    try:
                        from ..use_cases.chat import ChatMessage
                        chat_history = []
                        for msg in history_data:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            if isinstance(content, list):
                                content = ' '.join([
                                    item.get('text', '') for item in content
                                    if item.get('type') == 'text'
                                ])
                            chat_history.append(ChatMessage(
                                role=role,
                                content=content,
                                timestamp=msg.get('timestamp')
                            ))
                    except ImportError:
                        # Fallback if ChatMessage not available
                        chat_history = []
                    
                    # Stream chat response
                    async_gen = self.jotty_api.chat_stream(
                        message=message,
                        history=chat_history,
                        agent_id=agent_id
                    )
                    
                    # Format and yield events
                    while True:
                        try:
                            event = loop.run_until_complete(async_gen.__anext__())
                            for formatted_event in formatter.format_event(event):
                                yield formatted_event
                        except StopAsyncIteration:
                            break
                    
                    # End marker
                    yield formatter.end_marker()
                
                except Exception as e:
                    logger.error(f"Chat streaming error: {e}", exc_info=True)
                    yield formatter.error_event(str(e))
                
                finally:
                    loop.close()
            
            return Response(
                generate_events(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        except Exception as e:
            logger.error(f"Chat stream endpoint error: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e) if self.server_config.show_error_details else "Internal server error"
            }), 500
    
    def _handle_chat_execute(self) -> Response:
        """Handle synchronous chat execution."""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "Request body required"}), 400
            
            # Extract message and history (same logic as stream)
            messages_data = data.get('messages', [])
            message = data.get('message')
            history_data = data.get('history', [])
            
            if messages_data:
                if not messages_data or len(messages_data) == 0:
                    return jsonify({"error": "messages array is required"}), 400
                last_msg = messages_data[-1]
                if last_msg.get('role') != 'user':
                    return jsonify({"error": "Last message must be from user"}), 400
                message = last_msg.get('content', '')
                if isinstance(message, list):
                    message = ' '.join([
                        item.get('text', '') for item in message
                        if item.get('type') == 'text'
                    ])
                history_data = messages_data[:-1]
            elif not message:
                return jsonify({"error": "message or messages is required"}), 400
            
            agent_id = data.get('agentId')
            
            # Execute chat
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                try:
                    from ..use_cases.chat import ChatMessage
                    chat_history = []
                    for msg in history_data:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            content = ' '.join([
                                item.get('text', '') for item in content
                                if item.get('type') == 'text'
                            ])
                        chat_history.append(ChatMessage(
                            role=role,
                            content=content,
                            timestamp=msg.get('timestamp')
                        ))
                except ImportError:
                    chat_history = []
                
                result = loop.run_until_complete(
                    self.jotty_api.chat_execute(
                        message=message,
                        history=chat_history,
                        agent_id=agent_id
                    )
                )
                
                return jsonify({
                    "success": True,
                    **result
                })
            finally:
                loop.close()
        
        except Exception as e:
            logger.error(f"Chat execute endpoint error: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e) if self.server_config.show_error_details else "Internal server error"
            }), 500
    
    def _handle_workflow_execute(self) -> Response:
        """Handle workflow execution."""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "Request body required"}), 400
            
            goal = data.get('goal')
            if not goal:
                return jsonify({"error": "goal is required"}), 400
            
            context = data.get('context', {})
            mode = data.get('mode', 'dynamic')
            agent_order = data.get('agent_order')
            
            # Execute workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.jotty_api.workflow_execute(
                        goal=goal,
                        context=context,
                        mode=mode,
                        agent_order=agent_order
                    )
                )
                
                return jsonify({
                    "success": True,
                    **result
                })
            finally:
                loop.close()
        
        except Exception as e:
            logger.error(f"Workflow execute endpoint error: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e) if self.server_config.show_error_details else "Internal server error"
            }), 500
    
    def _handle_workflow_stream(self) -> Response:
        """Handle workflow streaming."""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "Request body required"}), 400
            
            goal = data.get('goal')
            if not goal:
                return jsonify({"error": "goal is required"}), 400
            
            context = data.get('context', {})
            
            def generate_events():
                """Generate workflow SSE events."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    async_gen = self.jotty_api.workflow_stream(
                        goal=goal,
                        context=context
                    )
                    
                    while True:
                        try:
                            event = loop.run_until_complete(async_gen.__anext__())
                            yield f"event: workflow\ndata: {json.dumps(event)}\n\n"
                        except StopAsyncIteration:
                            break
                    
                    yield f"event: done\ndata: {json.dumps({'success': True})}\n\n"
                
                except Exception as e:
                    logger.error(f"Workflow streaming error: {e}", exc_info=True)
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                
                finally:
                    loop.close()
            
            return Response(
                generate_events(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        
        except Exception as e:
            logger.error(f"Workflow stream endpoint error: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e) if self.server_config.show_error_details else "Internal server error"
            }), 500
    
    def _get_sse_formatter(self) -> SSEFormatter:
        """Get SSE formatter based on config."""
        format_type = self.server_config.sse_format.lower()
        
        if format_type == "usechat":
            return useChatFormatter()
        elif format_type == "openai":
            return OpenAIFormatter()
        elif format_type == "anthropic":
            return AnthropicFormatter()
        else:
            # Raw format (default)
            return SSEFormatter()
    
    def run(self, **kwargs):
        """
        Run the HTTP server.
        
        Args:
            **kwargs: Additional arguments passed to Flask.run()
        """
        port = kwargs.pop('port', self.server_config.port)
        host = kwargs.pop('host', self.server_config.host)
        debug = kwargs.pop('debug', self.server_config.debug)
        
        logger.info(f"🚀 Starting Jotty HTTP Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, **kwargs)


# Convenience alias
JottyServer = JottyHTTPServer
