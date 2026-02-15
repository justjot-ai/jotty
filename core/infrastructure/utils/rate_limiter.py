"""
Rate Limiting for Jotty

Provides:
- Per-skill rate limiting
- Per-user rate limiting  
- Per-API-key rate limiting
- Token bucket algorithm
- Sliding window algorithm
"""
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import deque


class RateLimitStrategy(Enum):
    """Rate limiting strategy."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int  # Max requests
    period: float  # Time period in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Allows bursts while maintaining average rate.
    """

    def __init__(self, rate: int, period: float) -> None:
        """
        Initialize token bucket.
        
        Args:
            rate: Maximum requests per period
            period: Time period in seconds
        """
        self.rate = rate
        self.period = period
        self.tokens = float(rate)
        self.last_update = time.time()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        """
        Check if request is allowed.
        
        Returns:
            True if request allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens for elapsed time
            self.tokens = min(
                self.rate,
                self.tokens + elapsed * (self.rate / self.period)
            )
            self.last_update = now

            # Check if we have tokens
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                return False

    def tokens_available(self) -> float:
        """Get number of tokens currently available."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            return min(
                self.rate,
                self.tokens + elapsed * (self.rate / self.period)
            )


class SlidingWindowLimiter:
    """
    Sliding window rate limiter.
    
    More accurate than fixed window, prevents boundary issues.
    """

    def __init__(self, rate: int, period: float) -> None:
        """
        Initialize sliding window limiter.
        
        Args:
            rate: Maximum requests per period
            period: Time period in seconds
        """
        self.rate = rate
        self.period = period
        self.requests = deque()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        """Check if request is allowed."""
        with self.lock:
            now = time.time()

            # Remove requests outside window
            while self.requests and self.requests[0] < now - self.period:
                self.requests.popleft()

            # Check if under limit
            if len(self.requests) < self.rate:
                self.requests.append(now)
                return True
            else:
                return False

    def requests_in_window(self) -> int:
        """Get number of requests in current window."""
        with self.lock:
            now = time.time()
            while self.requests and self.requests[0] < now - self.period:
                self.requests.popleft()
            return len(self.requests)


class RateLimiter:
    """
    Multi-level rate limiter for Jotty.
    
    Supports rate limiting by:
    - Skill name
    - User ID
    - API key
    - Global
    
    Usage:
        limiter = get_rate_limiter()
        limiter.add_limit("skill:web-search", RateLimit(100, 60))  # 100/minute
        limiter.add_limit("user:alice", RateLimit(1000, 3600))  # 1000/hour
        
        if limiter.allow("skill:web-search", user_id="alice"):
            # Execute skill
            pass
        else:
            # Rate limited
            pass
    """

    def __init__(self) -> None:
        """Initialize rate limiter."""
        self.limits: Dict[str, RateLimit] = {}
        self.limiters: Dict[str, object] = {}
        self.lock = threading.Lock()

    def add_limit(self, key: str, limit: RateLimit) -> None:
        """
        Add rate limit.
        
        Args:
            key: Limit key (e.g., "skill:web-search", "user:alice")
            limit: RateLimit configuration
        """
        with self.lock:
            self.limits[key] = limit

            # Create limiter based on strategy
            if limit.strategy == RateLimitStrategy.TOKEN_BUCKET:
                self.limiters[key] = TokenBucket(limit.requests, limit.period)
            elif limit.strategy == RateLimitStrategy.SLIDING_WINDOW:
                self.limiters[key] = SlidingWindowLimiter(limit.requests, limit.period)
            else:
                raise ValueError(f"Unsupported strategy: {limit.strategy}")

    def allow(self, *keys: str) -> bool:
        """
        Check if request is allowed for all given keys.
        
        Args:
            *keys: Limit keys to check
        
        Returns:
            True if allowed, False if any limit exceeded
        """
        with self.lock:
            for key in keys:
                if key in self.limiters:
                    if not self.limiters[key].allow():
                        return False
            return True

    def get_status(self, key: str) -> Optional[Dict]:
        """
        Get rate limit status.
        
        Args:
            key: Limit key
        
        Returns:
            Dict with status info or None if not found
        """
        if key not in self.limiters:
            return None

        limiter = self.limiters[key]
        limit = self.limits[key]

        if isinstance(limiter, TokenBucket):
            return {
                "type": "token_bucket",
                "rate": limit.requests,
                "period": limit.period,
                "tokens_available": limiter.tokens_available()
            }
        elif isinstance(limiter, SlidingWindowLimiter):
            return {
                "type": "sliding_window",
                "rate": limit.requests,
                "period": limit.period,
                "requests_in_window": limiter.requests_in_window()
            }

        return None


# Singleton instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """
    Get singleton rate limiter instance.
    
    Returns:
        RateLimiter instance
    """
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = RateLimiter()

        # Add default limits
        _rate_limiter.add_limit(
            "global",
            RateLimit(requests=1000, period=60, strategy=RateLimitStrategy.SLIDING_WINDOW)
        )

    return _rate_limiter


# Decorator for rate-limited functions
def rate_limit(key: str, requests: int, period: float, strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET) -> Any:
    """
    Decorator for rate limiting functions.
    
    Usage:
        @rate_limit("api_call", requests=100, period=60)
        def make_api_call():
            ...
    """
    from functools import wraps

    def decorator(func: Any) -> Any:
        # Add limit to global limiter
        limiter = get_rate_limiter()
        limiter.add_limit(key, RateLimit(requests, period, strategy))

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not limiter.allow(key):
                raise Exception(f"Rate limit exceeded for {key}. Limit: {requests}/{period}s")
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not limiter.allow(key):
                raise Exception(f"Rate limit exceeded for {key}. Limit: {requests}/{period}s")
            return await func(*args, **kwargs)

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator
