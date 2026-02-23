-- Atomic token bucket update
-- 1. Calculate tokens to add based on elapsed time
-- 2. Update bucket with new token count
-- 3. Attempt to consume token for request
-- 4. Return success/failure with remaining tokens

RateLimiter(redis_client, key_prefix, rate, capacity, window_size_seconds)

if limiter.allow_request('user_123'):
    process_request()
else:
    return_rate_limit_error()

remaining = limiter.get_remaining_tokens('user_123')
response_headers['X-RateLimit-Remaining'] = str(remaining)

# Admin override or quota reset
limiter.reset_bucket('premium_user_456')

# Conservative API limits
conservative_limiter = RateLimiter(
    redis_client=redis_client,
    key_prefix='api_v1',
    rate=10,           # 10 requests/second
    capacity=50,       # Burst of 50
    window_size_seconds=60
)

# High-throughput service
high_throughput_limiter = RateLimiter(
    redis_client=redis_client,
    key_prefix='internal_api',
    rate=1000,         # 1000 requests/second
    capacity=5000,     # Large burst capacity
    window_size_seconds=30
)

# Strict rate limiting
strict_limiter = RateLimiter(
    redis_client=redis_client,
    key_prefix='public_api',
    rate=5,            # 5 requests/second
    capacity=5,        # No burst allowance
    window_size_seconds=60
)
