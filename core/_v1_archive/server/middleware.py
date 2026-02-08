"""
Jotty HTTP Server Middleware

Provides authentication, logging, and error handling middleware.
"""

import logging
import time
from typing import Callable, Optional
from functools import wraps

try:
    from flask import request, g, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """
    Authentication middleware for Jotty HTTP Server.
    
    Supports:
    - Bearer token authentication
    - Clerk JWT tokens
    - Custom validation functions
    """
    
    def __init__(
        self,
        auth_type: str = "bearer",
        validate_fn: Optional[Callable[[str], bool]] = None
    ):
        """
        Initialize auth middleware.
        
        Args:
            auth_type: "bearer", "clerk", or "none"
            validate_fn: Custom validation function (token) -> bool
        """
        self.auth_type = auth_type
        self.validate_fn = validate_fn
    
    def before_request(self):
        """Validate authentication before request."""
        # Skip auth for health check endpoint
        if request.path == '/api/health':
            return None
            
        if self.auth_type == "none":
            return None
        
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header:
            return jsonify({"error": "Authorization header required"}), 401
        
        if self.auth_type == "bearer":
            if not auth_header.startswith('Bearer '):
                return jsonify({"error": "Invalid authorization format"}), 401
            
            token = auth_header[7:]
            
            # Use custom validation if provided
            if self.validate_fn:
                if not self.validate_fn(token):
                    return jsonify({"error": "Invalid authorization token"}), 401
            else:
                # Basic validation: check token exists and has minimum length
                if not token or len(token) < 10:
                    return jsonify({"error": "Invalid authorization token"}), 401
        
        elif self.auth_type == "clerk":
            # Clerk token validation
            if not auth_header.startswith('Bearer '):
                return jsonify({"error": "Invalid authorization format"}), 401
            
            token = auth_header[7:]
            
            # Use custom validation if provided
            if self.validate_fn:
                if not self.validate_fn(token):
                    return jsonify({"error": "Invalid Clerk token"}), 401
            else:
                # Basic validation: check token exists and has minimum length
                if not token or len(token) < 10:
                    return jsonify({"error": "Invalid Clerk token"}), 401
        
        # Store token in request context for use in handlers
        g.auth_token = token if 'token' in locals() else auth_header
        
        return None


class LoggingMiddleware:
    """
    Logging middleware for request/response tracking.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize logging middleware."""
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(self.log_level)
    
    def before_request(self):
        """Log request start."""
        g.start_time = time.time()
        logger.info(
            f"📥 {request.method} {request.path} "
            f"(IP: {request.remote_addr})"
        )
        return None
    
    def after_request(self, response):
        """Log request completion."""
        duration = time.time() - g.start_time
        logger.info(
            f"📤 {request.method} {request.path} "
            f"-> {response.status_code} ({duration:.3f}s)"
        )
        return response


class ErrorMiddleware:
    """
    Error handling middleware for graceful error responses.
    """
    
    def __init__(self, show_details: bool = False):
        """
        Initialize error middleware.
        
        Args:
            show_details: Whether to show error details to clients
        """
        self.show_details = show_details
    
    def handle_error(self, error: Exception):
        """Handle exceptions and return appropriate response."""
        logger.error(f"❌ Request error: {error}", exc_info=True)
        
        error_message = str(error) if self.show_details else "Internal server error"
        
        return jsonify({
            "success": False,
            "error": error_message
        }), 500
