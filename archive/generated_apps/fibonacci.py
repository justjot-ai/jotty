def fibonacci(n, memo=None):
    """
    Calculate the nth Fibonacci number using memoization.
    
    The Fibonacci sequence is defined as:
    F(0) = 0
    F(1) = 1
    F(n) = F(n-1) + F(n-2) for n > 1
    
    Args:
        n (int): The position in the Fibonacci sequence (0-indexed)
        memo (dict, optional): Dictionary for memoization. Defaults to None.
    
    Returns:
        int: The nth Fibonacci number
    
    Raises:
        ValueError: If n is negative
    
    Example:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]