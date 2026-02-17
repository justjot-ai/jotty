def fibonacci_sequence(n: int) -> list[int]:
    """
    Calculate the Fibonacci sequence up to n terms using dynamic programming.

    The Fibonacci sequence is defined as:
    F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2) for n > 1

    This function uses an iterative bottom-up dynamic programming approach
    to efficiently compute the sequence by storing previously calculated values.

    Parameters:
    -----------
    n : int
        The number of Fibonacci terms to generate. Must be a non-negative integer.

    Returns:
    --------
    list[int]
        A list containing the first n Fibonacci numbers in sequence.
        Returns empty list if n <= 0.

    Examples:
    ---------
    >>> fibonacci_sequence(0)
    []
    >>> fibonacci_sequence(1)
    [0]
    >>> fibonacci_sequence(2)
    [0, 1]
    >>> fibonacci_sequence(5)
    [0, 1, 1, 2, 3]
    >>> fibonacci_sequence(10)
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    # Handle edge case: n <= 0
    if n <= 0:
        return []

    # Handle edge case: n == 1
    if n == 1:
        return [0]

    # Handle edge case: n == 2
    if n == 2:
        return [0, 1]

    # Initialize DP array with base cases
    # dp[i] will store the i-th Fibonacci number
    dp: list[int] = [0] * n
    dp[0] = 0  # F(0) = 0
    dp[1] = 1  # F(1) = 1

    # Bottom-up DP: calculate each Fibonacci number using previously computed values
    # Time complexity: O(n), Space complexity: O(n)
    for i in range(2, n):
        # Current Fibonacci number = sum of two previous numbers
        # This is the core DP recurrence relation: F(i) = F(i-1) + F(i-2)
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp
