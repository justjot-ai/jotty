"""Batch 6a — 10 algorithm & CS skills with real implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_skills import create_skill

# ──────────────────────────────────────────────
# 1. matrix-calculator
# ──────────────────────────────────────────────
create_skill(
    name="matrix-calculator",
    frontmatter_name="matrix-calculator",
    description="Matrix operations: add, multiply, transpose, determinant, inverse.",
    category="mathematics",
    capabilities=["Matrix addition/subtraction", "Matrix multiplication", "Transpose", "Determinant", "Inverse"],
    triggers=["matrix multiply", "matrix inverse", "determinant", "transpose matrix"],
    tool_docs="### matrix_tool\nPerform matrix operations (add, multiply, transpose, determinant, inverse).",
    eval_tool="matrix_tool",
    eval_input={"operation": "determinant", "matrix_a": [[1, 2], [3, 4]]},
    tools_code=r'''"""Matrix calculator — add, multiply, transpose, determinant, inverse."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("matrix-calculator")

def _det(m: List[List[float]]) -> float:
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    d = 0.0
    for c in range(n):
        sub = [[m[r][j] for j in range(n) if j != c] for r in range(1, n)]
        d += ((-1) ** c) * m[0][c] * _det(sub)
    return d

def _transpose(m: List[List[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*m)]

def _minor(m: List[List[float]], i: int, j: int) -> List[List[float]]:
    return [[m[r][c] for c in range(len(m)) if c != j] for r in range(len(m)) if r != i]

def _inverse(m: List[List[float]]) -> List[List[float]]:
    n = len(m)
    d = _det(m)
    if abs(d) < 1e-12:
        raise ValueError("Matrix is singular")
    if n == 1:
        return [[1.0 / d]]
    cofactors = [[(-1) ** (i + j) * _det(_minor(m, i, j)) for j in range(n)] for i in range(n)]
    adj = _transpose(cofactors)
    return [[adj[i][j] / d for j in range(n)] for i in range(n)]

@tool_wrapper(required_params=["operation"])
def matrix_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform matrix operations: add, subtract, multiply, transpose, determinant, inverse."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    a = params.get("matrix_a")
    b = params.get("matrix_b")
    if not a:
        return tool_error("matrix_a is required")
    try:
        if op == "transpose":
            return tool_response(result=_transpose(a))
        if op == "determinant":
            return tool_response(result=_det(a))
        if op == "inverse":
            return tool_response(result=_inverse(a))
        if op in ("add", "subtract"):
            if not b:
                return tool_error("matrix_b required for add/subtract")
            sign = 1 if op == "add" else -1
            result = [[a[i][j] + sign * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
            return tool_response(result=result)
        if op == "multiply":
            if not b:
                return tool_error("matrix_b required for multiply")
            ra, ca, cb = len(a), len(a[0]), len(b[0])
            result = [[sum(a[i][k] * b[k][j] for k in range(ca)) for j in range(cb)] for i in range(ra)]
            return tool_response(result=result)
        return tool_error(f"Unknown operation: {op}. Use add/subtract/multiply/transpose/determinant/inverse")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["matrix_tool"]
''',
)

# ──────────────────────────────────────────────
# 2. statistics-calculator
# ──────────────────────────────────────────────
create_skill(
    name="statistics-calculator",
    frontmatter_name="statistics-calculator",
    description="Calculate mean, median, mode, std dev, variance, percentiles, quartiles.",
    category="mathematics",
    capabilities=["Mean/median/mode", "Standard deviation & variance", "Percentiles & quartiles", "Summary statistics"],
    triggers=["statistics", "standard deviation", "calculate mean", "percentile"],
    tool_docs="### statistics_tool\nCompute descriptive statistics for a list of numbers.",
    eval_tool="statistics_tool",
    eval_input={"numbers": [1, 2, 3, 4, 5], "operation": "summary"},
    tools_code=r'''"""Statistics calculator — descriptive stats using stdlib statistics."""
import statistics as st
import math
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("statistics-calculator")

def _percentile(data: List[float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f, c = int(math.floor(k)), int(math.ceil(k))
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)

@tool_wrapper(required_params=["numbers"])
def statistics_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute descriptive statistics: mean, median, mode, stdev, variance, percentiles."""
    status.set_callback(params.pop("_status_callback", None))
    nums = [float(x) for x in params["numbers"]]
    if not nums:
        return tool_error("numbers list is empty")
    op = params.get("operation", "summary").lower()
    try:
        if op == "mean":
            return tool_response(result=st.mean(nums))
        if op == "median":
            return tool_response(result=st.median(nums))
        if op == "mode":
            return tool_response(result=st.mode(nums))
        if op == "stdev":
            return tool_response(result=st.stdev(nums) if len(nums) > 1 else 0.0)
        if op == "variance":
            return tool_response(result=st.variance(nums) if len(nums) > 1 else 0.0)
        if op == "percentile":
            p = float(params.get("percentile", 50))
            return tool_response(result=_percentile(nums, p))
        if op == "quartiles":
            return tool_response(q1=_percentile(nums, 25), q2=_percentile(nums, 50), q3=_percentile(nums, 75))
        # summary
        result = {
            "count": len(nums), "mean": st.mean(nums), "median": st.median(nums),
            "min": min(nums), "max": max(nums),
            "stdev": st.stdev(nums) if len(nums) > 1 else 0.0,
            "variance": st.variance(nums) if len(nums) > 1 else 0.0,
            "q1": _percentile(nums, 25), "q3": _percentile(nums, 75),
        }
        return tool_response(**result)
    except Exception as e:
        return tool_error(str(e))

__all__ = ["statistics_tool"]
''',
)

# ──────────────────────────────────────────────
# 3. prime-number-tool
# ──────────────────────────────────────────────
create_skill(
    name="prime-number-tool",
    frontmatter_name="prime-number-tool",
    description="Check primality, generate primes, factorize, find nth prime.",
    category="mathematics",
    capabilities=["Primality testing", "Prime generation (sieve)", "Prime factorization", "Nth prime finder"],
    triggers=["is prime", "prime factors", "generate primes", "nth prime"],
    tool_docs="### prime_tool\nPrimality check, sieve, factorization, nth prime.",
    eval_tool="prime_tool",
    eval_input={"operation": "is_prime", "number": 97},
    tools_code=r'''"""Prime number tool — primality, sieve, factorization, nth prime."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("prime-number-tool")

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _sieve(limit: int) -> List[int]:
    if limit < 2:
        return []
    s = [True] * (limit + 1)
    s[0] = s[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if s[i]:
            for j in range(i * i, limit + 1, i):
                s[j] = False
    return [i for i, v in enumerate(s) if v]

def _factorize(n: int) -> List[int]:
    if n < 2:
        return []
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def _nth_prime(n: int) -> int:
    if n < 1:
        return 2
    count, candidate = 0, 1
    while count < n:
        candidate += 1
        if _is_prime(candidate):
            count += 1
    return candidate

@tool_wrapper(required_params=["operation"])
def prime_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Prime operations: is_prime, sieve, factorize, nth_prime."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    try:
        if op == "is_prime":
            n = int(params.get("number", 0))
            return tool_response(number=n, is_prime=_is_prime(n))
        if op == "sieve":
            limit = int(params.get("limit", 100))
            if limit > 1_000_000:
                return tool_error("Limit capped at 1000000")
            return tool_response(primes=_sieve(limit), count=len(_sieve(limit)))
        if op == "factorize":
            n = int(params.get("number", 0))
            f = _factorize(n)
            return tool_response(number=n, factors=f, expression=" x ".join(map(str, f)))
        if op == "nth_prime":
            n = int(params.get("n", 1))
            if n > 10000:
                return tool_error("n capped at 10000")
            return tool_response(n=n, prime=_nth_prime(n))
        return tool_error(f"Unknown op: {op}. Use is_prime/sieve/factorize/nth_prime")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["prime_tool"]
''',
)

# ──────────────────────────────────────────────
# 4. fibonacci-generator
# ──────────────────────────────────────────────
create_skill(
    name="fibonacci-generator",
    frontmatter_name="fibonacci-generator",
    description="Generate Fibonacci numbers, check membership, golden ratio.",
    category="mathematics",
    capabilities=["Generate Fibonacci sequence", "Check if number is Fibonacci", "Golden ratio calculation", "Nth Fibonacci number"],
    triggers=["fibonacci", "golden ratio", "fibonacci sequence"],
    tool_docs="### fibonacci_tool\nGenerate Fibonacci numbers and related operations.",
    eval_tool="fibonacci_tool",
    eval_input={"operation": "sequence", "count": 10},
    tools_code=r'''"""Fibonacci generator — sequence, membership check, golden ratio."""
import math
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("fibonacci-generator")

def _fib_seq(n: int) -> List[int]:
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

def _is_fib(n: int) -> bool:
    if n < 0:
        return False
    def _is_perfect_square(x: int) -> bool:
        s = int(math.isqrt(x))
        return s * s == x
    return _is_perfect_square(5 * n * n + 4) or _is_perfect_square(5 * n * n - 4)

def _nth_fib(n: int) -> int:
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

@tool_wrapper(required_params=["operation"])
def fibonacci_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fibonacci operations: sequence, nth, is_fibonacci, golden_ratio."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    try:
        if op == "sequence":
            count = int(params.get("count", 10))
            if count > 1000:
                return tool_error("Count capped at 1000")
            seq = _fib_seq(count)
            return tool_response(sequence=seq, count=len(seq))
        if op == "nth":
            n = int(params.get("n", 1))
            return tool_response(n=n, fibonacci=_nth_fib(n))
        if op == "is_fibonacci":
            num = int(params.get("number", 0))
            return tool_response(number=num, is_fibonacci=_is_fib(num))
        if op == "golden_ratio":
            n = int(params.get("precision", 20))
            a, b = _nth_fib(n), _nth_fib(n - 1)
            ratio = a / b if b != 0 else float("inf")
            phi = (1 + math.sqrt(5)) / 2
            return tool_response(approximation=ratio, exact_phi=phi, terms_used=n, error=abs(ratio - phi))
        return tool_error(f"Unknown op: {op}. Use sequence/nth/is_fibonacci/golden_ratio")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["fibonacci_tool"]
''',
)

# ──────────────────────────────────────────────
# 5. sorting-visualizer
# ──────────────────────────────────────────────
create_skill(
    name="sorting-visualizer",
    frontmatter_name="sorting-visualizer",
    description="Demonstrate sorting algorithms step-by-step with text visualization.",
    category="education",
    capabilities=["Bubble sort steps", "Insertion sort steps", "Merge sort steps", "Quick sort steps"],
    triggers=["sorting algorithm", "bubble sort", "visualize sort", "sort step by step"],
    tool_docs="### sorting_tool\nVisualize sorting algorithms step by step.",
    eval_tool="sorting_tool",
    eval_input={"algorithm": "bubble", "array": [5, 3, 1, 4, 2]},
    tools_code=r'''"""Sorting visualizer — step-by-step sorting demonstrations."""
from typing import Dict, Any, List, Tuple
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("sorting-visualizer")

def _bubble(arr: List[int]) -> Tuple[List[int], List[str]]:
    a, steps = arr[:], []
    for i in range(len(a)):
        for j in range(len(a) - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                steps.append(f"Swap {a[j+1]} and {a[j]} -> {a[:]}")
    return a, steps

def _insertion(arr: List[int]) -> Tuple[List[int], List[str]]:
    a, steps = arr[:], []
    for i in range(1, len(a)):
        key, j = a[i], i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
        steps.append(f"Insert {key} at pos {j+1} -> {a[:]}")
    return a, steps

def _merge_sort(arr: List[int], steps: List[str]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = _merge_sort(arr[:mid], steps)
    right = _merge_sort(arr[mid:], steps)
    merged, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    steps.append(f"Merge {left} + {right} -> {merged}")
    return merged

def _quick_sort(arr: List[int], steps: List[str]) -> List[int]:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    steps.append(f"Pivot={pivot}: left={left}, mid={mid}, right={right}")
    return _quick_sort(left, steps) + mid + _quick_sort(right, steps)

@tool_wrapper(required_params=["algorithm", "array"])
def sorting_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Visualize sorting: bubble, insertion, merge, quick."""
    status.set_callback(params.pop("_status_callback", None))
    algo = params["algorithm"].lower()
    arr = [int(x) for x in params["array"]]
    if len(arr) > 50:
        return tool_error("Array capped at 50 elements for visualization")
    try:
        if algo == "bubble":
            result, steps = _bubble(arr)
        elif algo == "insertion":
            result, steps = _insertion(arr)
        elif algo == "merge":
            steps: list = []
            result = _merge_sort(arr, steps)
        elif algo == "quick":
            steps: list = []
            result = _quick_sort(arr, steps)
        else:
            return tool_error(f"Unknown algorithm: {algo}. Use bubble/insertion/merge/quick")
        return tool_response(original=arr, sorted=result, steps=steps, step_count=len(steps), algorithm=algo)
    except Exception as e:
        return tool_error(str(e))

__all__ = ["sorting_tool"]
''',
)

# ──────────────────────────────────────────────
# 6. graph-algorithms
# ──────────────────────────────────────────────
create_skill(
    name="graph-algorithms",
    frontmatter_name="graph-algorithms",
    description="Shortest path, BFS, cycle detection, topological sort on adjacency lists.",
    category="algorithms",
    capabilities=["Dijkstra shortest path", "BFS traversal", "Cycle detection", "Topological sort"],
    triggers=["shortest path", "dijkstra", "topological sort", "graph cycle"],
    tool_docs="### graph_tool\nGraph algorithms on adjacency list representations.",
    eval_tool="graph_tool",
    eval_input={"operation": "bfs", "graph": {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}, "start": "A"},
    tools_code=r'''"""Graph algorithms — Dijkstra, BFS, cycle detection, topological sort."""
import heapq
from typing import Dict, Any, List, Optional
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("graph-algorithms")

def _bfs(graph: Dict, start: str) -> List[str]:
    visited, queue, order = set(), [start], []
    visited.add(start)
    while queue:
        node = queue.pop(0)
        order.append(node)
        for nb in graph.get(node, []):
            n = nb if isinstance(nb, str) else nb[0]
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return order

def _dijkstra(graph: Dict, start: str, end: Optional[str] = None) -> Dict:
    dist = {start: 0}
    prev: Dict[str, Optional[str]] = {start: None}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for edge in graph.get(u, []):
            v, w = (edge[0], edge[1]) if isinstance(edge, (list, tuple)) else (edge, 1)
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if end and end in dist:
        path, cur = [], end
        while cur is not None:
            path.append(cur)
            cur = prev.get(cur)
        path.reverse()
        return {"distances": dist, "path": path, "cost": dist[end]}
    return {"distances": dist}

def _has_cycle(graph: Dict) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    def dfs(u: str) -> bool:
        color[u] = GRAY
        for nb in graph.get(u, []):
            n = nb if isinstance(nb, str) else nb[0]
            if color.get(n, WHITE) == GRAY:
                return True
            if color.get(n, WHITE) == WHITE and dfs(n):
                return True
        color[u] = BLACK
        return False
    return any(color.get(n, WHITE) == WHITE and dfs(n) for n in graph)

def _topo_sort(graph: Dict) -> List[str]:
    visited, stack = set(), []
    def dfs(u: str) -> None:
        visited.add(u)
        for nb in graph.get(u, []):
            n = nb if isinstance(nb, str) else nb[0]
            if n not in visited:
                dfs(n)
        stack.append(u)
    for n in graph:
        if n not in visited:
            dfs(n)
    return stack[::-1]

@tool_wrapper(required_params=["operation", "graph"])
def graph_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Graph operations: bfs, dijkstra, has_cycle, topological_sort."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    graph = params["graph"]
    try:
        if op == "bfs":
            start = params.get("start", next(iter(graph)))
            return tool_response(order=_bfs(graph, start))
        if op == "dijkstra":
            start = params.get("start", next(iter(graph)))
            end = params.get("end")
            return tool_response(**_dijkstra(graph, start, end))
        if op == "has_cycle":
            return tool_response(has_cycle=_has_cycle(graph))
        if op == "topological_sort":
            if _has_cycle(graph):
                return tool_error("Graph has a cycle; topological sort not possible")
            return tool_response(order=_topo_sort(graph))
        return tool_error(f"Unknown op: {op}. Use bfs/dijkstra/has_cycle/topological_sort")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["graph_tool"]
''',
)

# ──────────────────────────────────────────────
# 7. encryption-tool
# ──────────────────────────────────────────────
create_skill(
    name="encryption-tool",
    frontmatter_name="encryption-tool",
    description="Caesar cipher, Vigenere cipher, ROT13, XOR encryption (educational).",
    category="cryptography",
    capabilities=["Caesar cipher encrypt/decrypt", "Vigenere cipher", "ROT13", "XOR encryption"],
    triggers=["caesar cipher", "encrypt text", "rot13", "vigenere"],
    tool_docs="### encryption_tool\nEducational ciphers: Caesar, Vigenere, ROT13, XOR.",
    eval_tool="encryption_tool",
    eval_input={"operation": "caesar", "text": "hello", "shift": 3},
    tools_code=r'''"""Encryption tool — educational ciphers (Caesar, Vigenere, ROT13, XOR)."""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("encryption-tool")

def _caesar(text: str, shift: int, decrypt: bool = False) -> str:
    if decrypt:
        shift = -shift
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)

def _vigenere(text: str, key: str, decrypt: bool = False) -> str:
    if not key.isalpha():
        raise ValueError("Key must be alphabetic")
    key = key.lower()
    result, ki = [], 0
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            shift = ord(key[ki % len(key)]) - ord("a")
            if decrypt:
                shift = -shift
            result.append(chr((ord(ch) - base + shift) % 26 + base))
            ki += 1
        else:
            result.append(ch)
    return "".join(result)

def _xor(text: str, key: str) -> str:
    return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(text))

@tool_wrapper(required_params=["operation", "text"])
def encryption_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Educational ciphers: caesar, vigenere, rot13, xor."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    text = params["text"]
    decrypt = params.get("decrypt", False)
    try:
        if op == "caesar":
            shift = int(params.get("shift", 3))
            result = _caesar(text, shift, decrypt)
            return tool_response(result=result, cipher="caesar", shift=shift)
        if op == "rot13":
            result = _caesar(text, 13)
            return tool_response(result=result, cipher="rot13")
        if op == "vigenere":
            key = params.get("key", "")
            if not key:
                return tool_error("key required for Vigenere cipher")
            result = _vigenere(text, key, decrypt)
            return tool_response(result=result, cipher="vigenere", key=key)
        if op == "xor":
            key = params.get("key", "")
            if not key:
                return tool_error("key required for XOR")
            result = _xor(text, key)
            hex_result = result.encode("utf-8", errors="replace").hex()
            return tool_response(result=hex_result, cipher="xor", note="Hex-encoded output")
        return tool_error(f"Unknown op: {op}. Use caesar/rot13/vigenere/xor")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["encryption_tool"]
''',
)

# ──────────────────────────────────────────────
# 8. compression-analyzer
# ──────────────────────────────────────────────
create_skill(
    name="compression-analyzer",
    frontmatter_name="compression-analyzer",
    description="Analyze text compressibility, entropy, run-length encode/decode, Huffman tree.",
    category="algorithms",
    capabilities=["Shannon entropy", "Run-length encoding/decoding", "Huffman tree visualization", "Compression ratio analysis"],
    triggers=["entropy", "run-length encoding", "huffman", "compression"],
    tool_docs="### compression_tool\nAnalyze text compression: entropy, RLE, Huffman.",
    eval_tool="compression_tool",
    eval_input={"operation": "entropy", "text": "hello world"},
    tools_code=r'''"""Compression analyzer — entropy, RLE, Huffman tree visualization."""
import math
from collections import Counter
from typing import Dict, Any, List, Tuple
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("compression-analyzer")

def _entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())

def _rle_encode(text: str) -> str:
    if not text:
        return ""
    result, count, prev = [], 1, text[0]
    for ch in text[1:]:
        if ch == prev:
            count += 1
        else:
            result.append(f"{count}{prev}" if count > 1 else prev)
            prev, count = ch, 1
    result.append(f"{count}{prev}" if count > 1 else prev)
    return "".join(result)

def _rle_decode(encoded: str) -> str:
    import re
    return "".join(ch * int(n) if n else ch for n, ch in re.findall(r"(\d*)(\D)", encoded))

def _huffman_tree(text: str) -> Tuple[Dict[str, str], List[str]]:
    import heapq
    freq = Counter(text)
    if len(freq) <= 1:
        ch = next(iter(freq))
        return {ch: "0"}, [f"'{ch}': 0"]
    heap: List = [[f, [ch, ""]] for ch, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    codes = {ch: code for ch, code in heap[0][1:]}
    tree_lines = [f"  '{ch}' ({freq[ch]}x) -> {code}" for ch, code in sorted(codes.items())]
    return codes, tree_lines

@tool_wrapper(required_params=["operation", "text"])
def compression_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compression analysis: entropy, rle_encode, rle_decode, huffman, analyze."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    text = params["text"]
    try:
        if op == "entropy":
            e = _entropy(text)
            return tool_response(entropy=round(e, 4), max_entropy=round(math.log2(max(len(set(text)), 1)), 4), length=len(text))
        if op == "rle_encode":
            enc = _rle_encode(text)
            return tool_response(encoded=enc, original_len=len(text), encoded_len=len(enc), ratio=round(len(enc) / max(len(text), 1), 3))
        if op == "rle_decode":
            return tool_response(decoded=_rle_decode(text))
        if op == "huffman":
            codes, tree = _huffman_tree(text)
            total_bits = sum(len(codes[ch]) * cnt for ch, cnt in Counter(text).items())
            return tool_response(codes=codes, tree="\n".join(tree), total_bits=total_bits, bits_per_char=round(total_bits / max(len(text), 1), 3))
        if op == "analyze":
            e = _entropy(text)
            rle = _rle_encode(text)
            codes, _ = _huffman_tree(text)
            huff_bits = sum(len(codes[ch]) * cnt for ch, cnt in Counter(text).items())
            return tool_response(length=len(text), unique_chars=len(set(text)), entropy=round(e, 4),
                                 rle_ratio=round(len(rle) / max(len(text), 1), 3), huffman_bits=huff_bits,
                                 huffman_bpc=round(huff_bits / max(len(text), 1), 3), naive_bits=len(text) * 8)
        return tool_error(f"Unknown op: {op}. Use entropy/rle_encode/rle_decode/huffman/analyze")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["compression_tool"]
''',
)

# ──────────────────────────────────────────────
# 9. data-structure-tool
# ──────────────────────────────────────────────
create_skill(
    name="data-structure-tool",
    frontmatter_name="data-structure-tool",
    description="Demonstrate stack, queue, linked list, binary tree operations step-by-step.",
    category="education",
    capabilities=["Stack operations", "Queue operations", "Linked list operations", "Binary search tree operations"],
    triggers=["stack push pop", "queue operations", "linked list", "binary tree"],
    tool_docs="### data_structure_tool\nStep-by-step data structure demonstrations.",
    eval_tool="data_structure_tool",
    eval_input={"structure": "stack", "operations": [{"op": "push", "value": 1}, {"op": "push", "value": 2}, {"op": "pop"}]},
    tools_code=r'''"""Data structure tool — stack, queue, linked list, BST demonstrations."""
from typing import Dict, Any, List, Optional
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("data-structure-tool")

def _run_stack(ops: List[Dict]) -> Dict:
    stack: List = []
    steps = []
    for o in ops:
        action = o.get("op", "").lower()
        if action == "push":
            stack.append(o["value"])
            steps.append(f"push({o['value']}) -> stack={stack[:]}")
        elif action == "pop":
            if stack:
                v = stack.pop()
                steps.append(f"pop() -> {v}, stack={stack[:]}")
            else:
                steps.append("pop() -> EMPTY (underflow)")
        elif action == "peek":
            steps.append(f"peek() -> {stack[-1] if stack else 'EMPTY'}")
    return {"final": stack[:], "steps": steps, "size": len(stack)}

def _run_queue(ops: List[Dict]) -> Dict:
    queue: List = []
    steps = []
    for o in ops:
        action = o.get("op", "").lower()
        if action == "enqueue":
            queue.append(o["value"])
            steps.append(f"enqueue({o['value']}) -> queue={queue[:]}")
        elif action == "dequeue":
            if queue:
                v = queue.pop(0)
                steps.append(f"dequeue() -> {v}, queue={queue[:]}")
            else:
                steps.append("dequeue() -> EMPTY (underflow)")
        elif action == "peek":
            steps.append(f"peek() -> {queue[0] if queue else 'EMPTY'}")
    return {"final": queue[:], "steps": steps, "size": len(queue)}

def _run_linked_list(ops: List[Dict]) -> Dict:
    ll: List = []
    steps = []
    for o in ops:
        action = o.get("op", "").lower()
        if action == "append":
            ll.append(o["value"])
            steps.append(f"append({o['value']}) -> {' -> '.join(map(str, ll))} -> None")
        elif action == "prepend":
            ll.insert(0, o["value"])
            steps.append(f"prepend({o['value']}) -> {' -> '.join(map(str, ll))} -> None")
        elif action == "delete":
            val = o["value"]
            if val in ll:
                ll.remove(val)
                steps.append(f"delete({val}) -> {' -> '.join(map(str, ll)) or 'empty'} -> None")
            else:
                steps.append(f"delete({val}) -> not found")
        elif action == "search":
            val = o["value"]
            idx = ll.index(val) if val in ll else -1
            steps.append(f"search({val}) -> {'found at index ' + str(idx) if idx >= 0 else 'not found'}")
    return {"final": ll[:], "steps": steps, "representation": " -> ".join(map(str, ll)) + " -> None" if ll else "empty"}

def _run_bst(ops: List[Dict]) -> Dict:
    nodes: Dict[int, Dict] = {}
    root: Optional[int] = None
    steps = []
    def insert(val: int) -> None:
        nonlocal root
        if root is None:
            root = val; nodes[val] = {"left": None, "right": None}; return
        cur = root
        while True:
            if val < cur:
                if nodes[cur]["left"] is None:
                    nodes[cur]["left"] = val; nodes[val] = {"left": None, "right": None}; return
                cur = nodes[cur]["left"]
            else:
                if nodes[cur]["right"] is None:
                    nodes[cur]["right"] = val; nodes[val] = {"left": None, "right": None}; return
                cur = nodes[cur]["right"]
    def inorder(n: Optional[int]) -> List[int]:
        if n is None:
            return []
        return inorder(nodes[n]["left"]) + [n] + inorder(nodes[n]["right"])
    for o in ops:
        action = o.get("op", "").lower()
        if action == "insert":
            val = int(o["value"])
            insert(val)
            steps.append(f"insert({val}) -> inorder: {inorder(root)}")
        elif action == "search":
            val, cur = int(o["value"]), root
            path = []
            while cur is not None:
                path.append(cur)
                if val == cur:
                    break
                cur = nodes[cur]["left"] if val < cur else nodes[cur]["right"]
            found = cur == val if cur is not None else False
            steps.append(f"search({val}) -> {'found' if found else 'not found'}, path: {path}")
        elif action == "inorder":
            steps.append(f"inorder() -> {inorder(root)}")
    return {"inorder": inorder(root), "steps": steps, "root": root}

@tool_wrapper(required_params=["structure", "operations"])
def data_structure_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Demonstrate data structures: stack, queue, linked_list, bst."""
    status.set_callback(params.pop("_status_callback", None))
    struct = params["structure"].lower()
    ops = params["operations"]
    try:
        if struct == "stack":
            return tool_response(**_run_stack(ops))
        if struct == "queue":
            return tool_response(**_run_queue(ops))
        if struct in ("linked_list", "linkedlist"):
            return tool_response(**_run_linked_list(ops))
        if struct in ("bst", "binary_tree", "binarytree"):
            return tool_response(**_run_bst(ops))
        return tool_error(f"Unknown structure: {struct}. Use stack/queue/linked_list/bst")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["data_structure_tool"]
''',
)

# ──────────────────────────────────────────────
# 10. algorithm-complexity
# ──────────────────────────────────────────────
create_skill(
    name="algorithm-complexity",
    frontmatter_name="algorithm-complexity",
    description="Analyze Big-O complexity, compare growth rates, generate ASCII runtime plots.",
    category="education",
    capabilities=["Big-O lookup for common algorithms", "Growth rate comparison", "ASCII runtime plots", "Complexity classification"],
    triggers=["big o", "algorithm complexity", "time complexity", "growth rate"],
    tool_docs="### complexity_tool\nAnalyze and visualize algorithm complexity.",
    eval_tool="complexity_tool",
    eval_input={"operation": "lookup", "algorithm": "binary_search"},
    tools_code=r'''"""Algorithm complexity — Big-O analysis, comparison, ASCII plots."""
import math
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("algorithm-complexity")

COMPLEXITIES = {
    "binary_search": {"best": "O(1)", "average": "O(log n)", "worst": "O(log n)", "space": "O(1)"},
    "linear_search": {"best": "O(1)", "average": "O(n)", "worst": "O(n)", "space": "O(1)"},
    "bubble_sort": {"best": "O(n)", "average": "O(n^2)", "worst": "O(n^2)", "space": "O(1)"},
    "insertion_sort": {"best": "O(n)", "average": "O(n^2)", "worst": "O(n^2)", "space": "O(1)"},
    "merge_sort": {"best": "O(n log n)", "average": "O(n log n)", "worst": "O(n log n)", "space": "O(n)"},
    "quick_sort": {"best": "O(n log n)", "average": "O(n log n)", "worst": "O(n^2)", "space": "O(log n)"},
    "heap_sort": {"best": "O(n log n)", "average": "O(n log n)", "worst": "O(n log n)", "space": "O(1)"},
    "hash_table_lookup": {"best": "O(1)", "average": "O(1)", "worst": "O(n)", "space": "O(n)"},
    "bst_search": {"best": "O(log n)", "average": "O(log n)", "worst": "O(n)", "space": "O(n)"},
    "bfs": {"best": "O(V+E)", "average": "O(V+E)", "worst": "O(V+E)", "space": "O(V)"},
    "dfs": {"best": "O(V+E)", "average": "O(V+E)", "worst": "O(V+E)", "space": "O(V)"},
    "dijkstra": {"best": "O(V+E log V)", "average": "O(V+E log V)", "worst": "O(V+E log V)", "space": "O(V)"},
}

GROWTH_FNS = {
    "O(1)": lambda n: 1, "O(log n)": lambda n: math.log2(max(n, 1)),
    "O(n)": lambda n: n, "O(n log n)": lambda n: n * math.log2(max(n, 1)),
    "O(n^2)": lambda n: n * n, "O(n^3)": lambda n: n ** 3,
    "O(2^n)": lambda n: 2 ** min(n, 30), "O(n!)": lambda n: math.factorial(min(n, 12)),
}

def _ascii_plot(fns: List[str], max_n: int = 20, height: int = 15) -> str:
    points = {}
    for fn_name in fns:
        fn = GROWTH_FNS.get(fn_name)
        if fn:
            points[fn_name] = [fn(n) for n in range(1, max_n + 1)]
    if not points:
        return "No valid functions to plot"
    all_vals = [v for vals in points.values() for v in vals]
    max_val = max(all_vals) if all_vals else 1
    lines = []
    for row in range(height, -1, -1):
        threshold = max_val * row / height
        line = f"{threshold:>8.0f} |"
        for col in range(max_n):
            chars = []
            for i, fn_name in enumerate(fns):
                if fn_name in points and col < len(points[fn_name]):
                    if abs(points[fn_name][col] - threshold) <= max_val / height / 2 or (row == 0 and points[fn_name][col] <= threshold + max_val / height):
                        chars.append("*#@$%"[i % 5])
            line += chars[0] if chars else " "
        lines.append(line)
    lines.append("         +" + "-" * max_n)
    lines.append("          " + "".join(str(i % 10) for i in range(1, max_n + 1)) + " -> n")
    legend = "  Legend: " + ", ".join(f"{'*#@$%'[i % 5]}={fn}" for i, fn in enumerate(fns) if fn in GROWTH_FNS)
    lines.append(legend)
    return "\n".join(lines)

@tool_wrapper(required_params=["operation"])
def complexity_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Big-O analysis: lookup, compare, plot, list."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    try:
        if op == "lookup":
            algo = params.get("algorithm", "").lower().replace(" ", "_")
            if algo not in COMPLEXITIES:
                return tool_error(f"Unknown algorithm: {algo}. Known: {', '.join(COMPLEXITIES.keys())}")
            return tool_response(algorithm=algo, **COMPLEXITIES[algo])
        if op == "compare":
            algos = params.get("algorithms", [])
            results = {}
            for a in algos:
                key = a.lower().replace(" ", "_")
                if key in COMPLEXITIES:
                    results[key] = COMPLEXITIES[key]
            return tool_response(comparison=results)
        if op == "plot":
            fns = params.get("functions", ["O(n)", "O(n log n)", "O(n^2)"])
            max_n = int(params.get("max_n", 20))
            chart = _ascii_plot(fns, max_n)
            return tool_response(plot=chart, functions=fns)
        if op == "list":
            return tool_response(algorithms=list(COMPLEXITIES.keys()), growth_functions=list(GROWTH_FNS.keys()))
        if op == "classify":
            expr = params.get("expression", "")
            for name in ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)"]:
                if name.lower() in expr.lower():
                    category = "constant" if "1" == name[2:-1] else name[2:-1].replace(" ", "-")
                    return tool_response(expression=expr, complexity=name, category=category)
            return tool_response(expression=expr, complexity="unknown", note="Could not classify. Provide standard Big-O notation.")
        return tool_error(f"Unknown op: {op}. Use lookup/compare/plot/list/classify")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["complexity_tool"]
''',
)

print("\nBatch 6a: 10 skills generated!")
