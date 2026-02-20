#!/usr/bin/env python3
"""
Jotty COMPLEX Learning Evaluation
===================================

Tests learning in realistic, multi-faceted scenarios:

1. CROSS-DOMAIN TRANSFER: Learn patterns in coding → apply to data science
2. ESCALATION LADDER: Easy → Medium → Hard within same domain, verify quality rises
3. RETRY WITH LEARNING: Deliberately hard task → fail → retry with learning context
4. A/B MULTI-DOMAIN: Same task, with vs without learning, across 3 domains
5. STRUCTURAL DEPTH: Checks demand code, math, tests, citations — not just keywords

Uses Haiku (weakest model) to stress-test learning signal.
"""

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_complex")
logger.setLevel(logging.INFO)

EVAL_MODEL = "claude-3-haiku-20240307"

# =============================================================================
# HELPERS
# =============================================================================

def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def subsep(title: str) -> None:
    print(f"\n  --- {title} ---\n")


def check_regex(content: str, pattern: str, flags: int = re.IGNORECASE | re.DOTALL) -> bool:
    return bool(re.search(pattern, content, flags))


def check_code_blocks(content: str, min_blocks: int = 3, min_lines: int = 5) -> bool:
    blocks = re.findall(r'```[\w]*\n(.*?)```', content, re.DOTALL)
    substantial = [b for b in blocks if len(b.strip().split('\n')) >= min_lines]
    return len(substantial) >= min_blocks


def check_assertions(content: str, n: int = 5) -> bool:
    return len(re.findall(r'assert\s+\w+', content)) >= n


def check_math(content: str, n: int = 2) -> bool:
    formulas = re.findall(r'O\([^)]+\)|Θ\([^)]+\)|n\s*log\s*n|≤|≥|∑|∫|∀|∃|→|⟹|\\frac|R\^2|MSE|∂', content)
    return len(formulas) >= n


# =============================================================================
# SCENARIO 1: ESCALATION LADDER (Easy → Medium → Hard, same domain)
# =============================================================================

ESCALATION_TASKS = [
    {
        "id": "ESC_easy_sorting",
        "level": "easy",
        "domain": "algorithms",
        "goal": (
            "Explain merge sort and quicksort. For each:\n"
            "1. Write the algorithm in Python\n"
            "2. Prove time complexity (average and worst case)\n"
            "3. Explain when each is preferred\n"
            "4. Write a test function with at least 3 test cases\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("Merge sort implementation", lambda c: check_regex(c, r'def\s+merge_sort')),
            ("Quick sort implementation", lambda c: check_regex(c, r'def\s+quick_?sort')),
            ("O(n log n) analysis", lambda c: check_regex(c, r'O\(\s*n\s*log\s*n\s*\)')),
            ("O(n²) worst case", lambda c: check_regex(c, r'O\(\s*n\s*[²2\^]\s*\)')),
            ("Test function", lambda c: check_regex(c, r'def\s+test_')),
            ("Code blocks", lambda c: check_code_blocks(c, 2, 5)),
        ],
    },
    {
        "id": "ESC_med_balanced_tree",
        "level": "medium",
        "domain": "algorithms",
        "goal": (
            "Implement a self-balancing AVL tree in Python with:\n\n"
            "1. THEORY: Prove AVL invariant (height difference ≤ 1) ensures O(log n) height.\n"
            "   Show the math: h ≤ 1.44 * log₂(n+2). Derive from Fibonacci recurrence.\n"
            "2. IMPLEMENTATION:\n"
            "   - AVLNode with value, left, right, height\n"
            "   - insert(value) with automatic rebalancing\n"
            "   - delete(value) with rebalancing\n"
            "   - Left rotation, right rotation, left-right, right-left rotations\n"
            "   - Balance factor computation\n"
            "3. TESTS:\n"
            "   - test_insert_maintains_balance: insert 1..20 in order, verify max height ≤ 6\n"
            "   - test_delete_rebalances: delete root, verify tree stays balanced\n"
            "   - test_rotations: verify each rotation type with specific trigger case\n"
            "   - At least 5 assertions\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("AVLNode class", lambda c: check_regex(c, r'class\s+AVL(?:Node|Tree)')),
            ("Insert with rebalance", lambda c: check_regex(c, r'def\s+insert.*(?:balance|rotat)')),
            ("Left rotation", lambda c: check_regex(c, r'def\s+(?:left_)?rotat\w+.*(?:left|right)')),
            ("Balance factor", lambda c: check_regex(c, r'(?:balance_factor|height.*left.*right|left.*height.*right.*height)')),
            ("Height O(log n) proof", lambda c: check_regex(c, r'O\(\s*log\s*n\s*\).*(?:height|depth|balanc)')),
            ("Fibonacci recurrence", lambda c: check_regex(c, r'(?:fibonacci|fib|N_h|recurrence).*(?:h-1|h-2|\+\s*1)')),
            ("Delete implementation", lambda c: check_regex(c, r'def\s+delete')),
            ("Test functions", lambda c: check_regex(c, r'def\s+test_.*assert')),
            ("5+ assertions", lambda c: check_assertions(c, 5)),
            ("Code blocks", lambda c: check_code_blocks(c, 3, 5)),
        ],
    },
    {
        "id": "ESC_hard_btree",
        "level": "hard",
        "domain": "algorithms",
        "goal": (
            "Implement a B+ tree (order m=4) with disk-oriented storage semantics.\n\n"
            "PART A — THEORY:\n"
            "1. Define B+ tree invariants: all values in leaves, internal nodes store keys only, "
            "   leaves form a doubly-linked list. Prove height = O(log_m(n)).\n"
            "2. Explain why B+ trees are optimal for disk: node size = page size (4KB), "
            "   fanout calculation (if key=8B, pointer=8B, page=4096B, then max keys per node = ?).\n"
            "3. Compare B+ tree vs B-tree vs LSM-tree for:\n"
            "   - Point reads, range scans, sequential writes, random writes\n"
            "   - Calculate amplification factors (read amp, write amp, space amp)\n\n"
            "PART B — IMPLEMENTATION (m=4):\n"
            "- BPlusNode: is_leaf, keys, children (internal) / values (leaf), next/prev (leaf)\n"
            "- BPlusTree: insert(key, value), search(key), range_query(low, high)\n"
            "- Split logic: when node has >m keys, split into two nodes, push median up\n"
            "- Handle: root splits (tree grows taller), leaf splits vs internal splits\n"
            "- Bulk loading optimization: sort keys, fill leaves left-to-right, build internal bottom-up\n\n"
            "PART C — TESTS:\n"
            "- test_insert_1000: insert 1..1000, verify all searchable\n"
            "- test_range_query: insert random keys, range_query(100, 200) returns sorted subset\n"
            "- test_leaf_linked_list: verify all leaves connected in order\n"
            "- test_height_bound: after 10000 inserts, height ≤ log_4(10000) ≈ 7\n"
            "- test_split_correctness: insert m+1 keys, verify split produces valid tree\n"
            "- 10+ assertions total\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("B+ tree invariants", lambda c: check_regex(c, r'(?:leaf|leaves).*(?:linked list|doubly|pointer|next|prev)')),
            ("Height O(log_m(n))", lambda c: check_regex(c, r'O\(\s*log.*[_mM].*n\s*\)|height.*log.*base.*m')),
            ("Disk page size calculation", lambda c: check_regex(c, r'(?:4096|4\s*KB|page\s*size).*(?:key|pointer|fanout|node)')),
            ("Read/write amplification", lambda c: check_regex(c, r'(?:read|write|space)\s+(?:amplif|amp)')),
            ("BPlusNode class", lambda c: check_regex(c, r'class\s+BPlus(?:Node|Tree)')),
            ("Insert with split", lambda c: check_regex(c, r'def\s+insert.*(?:split|overflow|median)')),
            ("Split logic", lambda c: check_regex(c, r'def\s+(?:_?split|_?split_child|_?split_node)')),
            ("Range query", lambda c: check_regex(c, r'def\s+range_query')),
            ("Leaf linked list traversal", lambda c: check_regex(c, r'(?:next|prev|forward|linked).*(?:leaf|node|pointer)')),
            ("Bulk loading", lambda c: check_regex(c, r'(?:bulk|batch).*(?:load|insert|build)')),
            ("Substantial code", lambda c: check_code_blocks(c, 3, 8)),
            ("test_range_query", lambda c: check_regex(c, r'def\s+test_\w*range')),
            ("test_split", lambda c: check_regex(c, r'def\s+test_\w*split')),
            ("5+ assertions", lambda c: check_assertions(c, 5)),
            ("Math reasoning", lambda c: check_math(c, 2)),
        ],
    },
]


# =============================================================================
# SCENARIO 2: CROSS-DOMAIN TRANSFER (coding patterns → data science)
# =============================================================================

TRANSFER_SEED_TASK = {
    "id": "XFER_seed_ml_pipeline",
    "domain": "coding",
    "goal": (
        "Implement a complete ML pipeline framework in Python:\n"
        "1. A Pipeline class with add_step(name, fn), run(data) → transformed_data\n"
        "2. Steps: StandardScaler (z-score), OneHotEncoder, TrainTestSplit\n"
        "3. Each step must be reversible (inverse_transform)\n"
        "4. Add caching: if same input hash, return cached output\n"
        "5. Add logging: each step logs input shape → output shape\n"
        "6. Write tests for 3-step pipeline with numpy arrays\n"
        "Do NOT use any tools."
    ),
    "checks": [
        ("Pipeline class", lambda c: check_regex(c, r'class\s+Pipeline')),
        ("add_step method", lambda c: check_regex(c, r'def\s+add_step')),
        ("StandardScaler", lambda c: check_regex(c, r'(?:class\s+)?Standard(?:Scaler|ize)')),
        ("Caching logic", lambda c: check_regex(c, r'(?:cache|hash|memoiz)')),
        ("Test function", lambda c: check_regex(c, r'def\s+test_')),
        ("Code blocks", lambda c: check_code_blocks(c, 2, 5)),
    ],
}

TRANSFER_TARGET_TASK = {
    "id": "XFER_target_bayesian_opt",
    "domain": "data_science",
    "goal": (
        "Implement Bayesian Optimization from scratch in Python for hyperparameter tuning.\n\n"
        "PART A — THEORY:\n"
        "1. Define the optimization problem: minimize f(x) where f is expensive (e.g., model training).\n"
        "2. Gaussian Process (GP) surrogate: define the kernel (RBF), posterior mean μ(x) and "
        "   variance σ²(x) formulas. Show the matrix equations.\n"
        "3. Acquisition functions: implement Expected Improvement (EI):\n"
        "   EI(x) = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z) where Z = (μ(x) - f_best - ξ) / σ(x)\n"
        "4. Prove EI balances exploration (high σ) vs exploitation (low μ).\n\n"
        "PART B — IMPLEMENTATION:\n"
        "- GaussianProcess: fit(X, y), predict(X_new) → (μ, σ²), rbf_kernel(X1, X2, l, σ_f)\n"
        "- BayesianOptimizer: suggest_next(n_candidates=1000), update(x, y), optimize(f, n_iters=20)\n"
        "- Kernel matrix inversion with Cholesky (numerically stable)\n"
        "- Handle: noisy observations, multiple restarts, convergence detection\n\n"
        "PART C — TESTS:\n"
        "- test_gp_interpolation: fit GP to sin(x), verify predictions match at training points\n"
        "- test_ei_prefers_uncertain: high σ region should have higher EI than low σ with same μ\n"
        "- test_optimize_quadratic: optimize f(x)=x², verify finds minimum near 0\n"
        "- test_convergence: verify optimizer converges (best_y improves over iterations)\n"
        "- 5+ assertions\n\n"
        "Do NOT use any tools."
    ),
    "checks": [
        ("GP posterior formulas", lambda c: check_regex(c, r'(?:μ|mu|mean).*(?:K|kernel|posterior).*(?:σ|sigma|variance)')),
        ("RBF kernel", lambda c: check_regex(c, r'(?:rbf|radial|squared.*exponential).*(?:kernel|exp|l\^2)')),
        ("Expected Improvement formula", lambda c: check_regex(c, r'(?:EI|expected.*improvement).*(?:Φ|phi|cdf|μ|sigma)')),
        ("Cholesky decomposition", lambda c: check_regex(c, r'(?:cholesky|cho_factor|linalg\.cho|np\.linalg)')),
        ("GaussianProcess class", lambda c: check_regex(c, r'class\s+Gaussian(?:Process|GP)')),
        ("BayesianOptimizer class", lambda c: check_regex(c, r'class\s+Bayesian(?:Optim|Opt)')),
        ("suggest_next method", lambda c: check_regex(c, r'def\s+suggest_next|def\s+suggest')),
        ("Exploration vs exploitation", lambda c: check_regex(c, r'(?:explor|exploit).*(?:σ|uncertainty|variance|mean)')),
        ("test_gp or test_optimize", lambda c: check_regex(c, r'def\s+test_\w*(?:gp|optim|convergence)')),
        ("Substantial code", lambda c: check_code_blocks(c, 3, 5)),
        ("5+ assertions", lambda c: check_assertions(c, 5)),
        ("Math reasoning", lambda c: check_math(c, 2)),
    ],
}


# =============================================================================
# SCENARIO 3: RETRY IMPROVEMENT (fail → learn → retry → improve)
# =============================================================================

RETRY_TASK = {
    "id": "RETRY_compiler",
    "domain": "compiler_design",
    "goal": (
        "Implement a complete recursive descent parser and AST-walking interpreter for a "
        "small programming language with the following features:\n\n"
        "LANGUAGE SPEC:\n"
        "- Types: int, float, bool, string\n"
        "- Variables: let x = expr;\n"
        "- Arithmetic: +, -, *, /, % with correct precedence\n"
        "- Comparison: ==, !=, <, >, <=, >=\n"
        "- Control flow: if/else, while loops\n"
        "- Functions: fn name(params) { body } with return statement\n"
        "- Closures: functions capture enclosing scope\n\n"
        "IMPLEMENTATION REQUIREMENTS:\n"
        "1. LEXER: Token class with type + value, tokenize(source) → List[Token]\n"
        "   Handle: strings with escapes, multi-char operators (==, !=, <=, >=)\n"
        "2. PARSER: Recursive descent with Pratt parsing for expressions\n"
        "   - parse_expression() with precedence climbing (prefix/infix/postfix)\n"
        "   - parse_statement() → LetStmt | IfStmt | WhileStmt | ReturnStmt | ExprStmt\n"
        "   - parse_function() → FnDecl with params and body\n"
        "   - Produce AST nodes: BinaryExpr, UnaryExpr, CallExpr, IfExpr, etc.\n"
        "3. INTERPRETER: Tree-walking evaluator with environment/scope chain\n"
        "   - Environment with parent scope (for closures)\n"
        "   - evaluate(node, env) → Value\n"
        "   - Function calls create new scope with closure capture\n"
        "   - Type checking: division by zero, type mismatch errors\n"
        "4. TESTS:\n"
        "   - test_arithmetic: 2 + 3 * 4 == 14 (precedence)\n"
        "   - test_variables: let x = 10; let y = x + 5; y == 15\n"
        "   - test_if_else: if (x > 0) { 1 } else { -1 }\n"
        "   - test_while_loop: factorial via while loop\n"
        "   - test_functions: fn add(a, b) { return a + b; }; add(3, 4) == 7\n"
        "   - test_closures: fn make_adder(x) { return fn(y) { return x + y; }; }; make_adder(5)(3) == 8\n"
        "   - test_fibonacci: recursive fibonacci function\n"
        "   - 10+ assertions total\n\n"
        "This must be COMPLETE and RUNNABLE. Do NOT use any tools."
    ),
    "checks": [
        # Lexer
        ("Token class", lambda c: check_regex(c, r'class\s+Token')),
        ("Tokenize function", lambda c: check_regex(c, r'def\s+(?:tokenize|lex)\s*\(')),
        ("Multi-char operators", lambda c: check_regex(c, r'(?:==|!=|<=|>=).*(?:token|Token|operator)')),

        # Parser
        ("AST node classes", lambda c: check_regex(c, r'class\s+(?:Binary|Unary|Call|If|While|Let|Fn)(?:Expr|Stmt|Node|Decl)')),
        ("parse_expression", lambda c: check_regex(c, r'def\s+parse_(?:expression|expr)\s*\(')),
        ("Precedence handling", lambda c: check_regex(c, r'(?:precedence|prec|binding_power|pratt)')),
        ("parse_statement", lambda c: check_regex(c, r'def\s+parse_(?:statement|stmt)\s*\(')),
        ("Function parsing", lambda c: check_regex(c, r'def\s+parse_(?:function|fn)\s*\(')),

        # Interpreter
        ("Environment/scope", lambda c: check_regex(c, r'class\s+(?:Environment|Scope|Env)')),
        ("evaluate method", lambda c: check_regex(c, r'def\s+(?:evaluate|eval|interpret|visit)\s*\(')),
        ("Closure capture", lambda c: check_regex(c, r'(?:closure|enclos|parent|captured).*(?:scope|env|environment)')),
        ("Function call execution", lambda c: check_regex(c, r'(?:call|invoke|apply).*(?:arg|param|scope|env)')),

        # Tests
        ("test_arithmetic", lambda c: check_regex(c, r'def\s+test_\w*(?:arithmetic|precedence|expr)')),
        ("test_functions", lambda c: check_regex(c, r'def\s+test_\w*(?:function|fn|call)')),
        ("test_closures", lambda c: check_regex(c, r'def\s+test_\w*(?:closure|make_adder|nested)')),
        ("Substantial code", lambda c: check_code_blocks(c, 4, 8)),
        ("5+ assertions", lambda c: check_assertions(c, 5)),
    ],
}


# =============================================================================
# SCENARIO 4: A/B MULTI-DOMAIN (test across coding, math, system_design)
# =============================================================================

AB_TASKS = [
    {
        "id": "AB_graph_algorithms",
        "domain": "algorithms",
        "goal": (
            "Implement Dijkstra's algorithm with Fibonacci heap and A* search.\n\n"
            "1. Dijkstra with min-heap: O((V+E) log V). Implement with heapq.\n"
            "2. Dijkstra with Fibonacci heap: O(V log V + E). Implement the Fibonacci heap "
            "   with decrease-key in O(1) amortized. Explain why this matters for dense graphs.\n"
            "3. A* search: implement with heuristic function. Prove admissibility guarantees optimality.\n"
            "4. Compare on a 100-node grid graph: show Dijkstra explores more nodes than A*.\n"
            "5. Tests: test_shortest_path, test_negative_no_path, test_astar_fewer_nodes, 5+ assertions.\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("Dijkstra implementation", lambda c: check_regex(c, r'def\s+dijkstra')),
            ("Fibonacci heap", lambda c: check_regex(c, r'(?:class|def)\s+(?:Fibonacci|Fib)(?:Heap|_heap)')),
            ("A* implementation", lambda c: check_regex(c, r'def\s+a_?star')),
            ("Admissibility proof", lambda c: check_regex(c, r'(?:admiss|heuristic).*(?:optimal|underestimate|≤|<=|never\s+over)')),
            ("decrease-key", lambda c: check_regex(c, r'decrease.?key.*O\(\s*1\s*\)|O\(\s*1\s*\).*decrease.?key')),
            ("Complexity analysis", lambda c: check_regex(c, r'O\(\s*(?:V|E).*log')),
            ("Grid graph comparison", lambda c: check_regex(c, r'(?:grid|graph).*(?:node|explored|visited).*(?:fewer|less|more)')),
            ("Code blocks", lambda c: check_code_blocks(c, 3, 5)),
            ("Test functions", lambda c: check_regex(c, r'def\s+test_')),
            ("5+ assertions", lambda c: check_assertions(c, 5)),
        ],
    },
]


# =============================================================================
# EVALUATOR
# =============================================================================

class ComplexEval:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self._orch = None
        self._learning = None

    def _get_orchestrator(self) -> Any:
        if self._orch is None:
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
            self._orch = Orchestrator()
        return self._orch

    def _get_learning(self) -> Any:
        if self._learning is None:
            from Jotty.core.intelligence.learning.learning_service import LearningService
            self._learning = LearningService.get_instance()
        return self._learning

    def _snapshot(self, domain: str = "") -> Dict[str, Any]:
        ls = self._get_learning()
        query_domain = domain or "general"
        guidance = ls.query(query_domain, "")
        ctx = ls.build_context_string(query_domain, "")
        ret = ls.build_retrieval_context(query_domain, "")
        # Also get global episode count across all domains
        try:
            conn = ls._store._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM episodes").fetchone()
            global_episodes = row["cnt"] if row else 0
        except Exception:
            global_episodes = guidance.get("total_episodes", 0)
        return {
            "episodes": guidance.get("total_episodes", 0),
            "global_episodes": global_episodes,
            "rate": guidance.get("success_rate", 0.0),
            "ctx_len": len(ctx),
            "ret_len": len(ret),
            "ctx": ctx[:200] if ctx else "",
        }

    async def run_task(
        self, task: Dict, learn: bool = True, tag: str = ""
    ) -> Dict[str, Any]:
        label = f"{task['id']}{f' ({tag})' if tag else ''}"
        no_learn_tag = " [NO LEARNING]" if not learn else ""
        subsep(f"{label}{no_learn_tag}")

        orch = self._get_orchestrator()
        domain = task.get("domain", "general")
        snap_before = self._snapshot(domain)

        print(f"  Model: {EVAL_MODEL}")
        print(f"  Episodes: {snap_before['episodes']} (domain) / {snap_before['global_episodes']} (global) | Rate: {snap_before['rate']:.0%}")
        print(f"  Context injected: {snap_before['ctx_len']} chars")
        if snap_before['ctx']:
            for line in snap_before['ctx'].split('\n')[:3]:
                print(f"    > {line[:100]}")
        print(f"  Retrieval: {snap_before['ret_len']} chars")

        start = time.time()
        content = ""
        error = None

        try:
            result = await orch.chat(
                message=task["goal"],
                provider="anthropic",
                model=EVAL_MODEL,
                learn=learn,
                enabled_tools=[],
                max_steps=1,
                temperature=0,
            )
            content = getattr(result, "content", str(result))
        except Exception as e:
            error = str(e)
            print(f"  ERROR: {error[:200]}")

        elapsed = time.time() - start
        snap_after = self._snapshot(domain)

        check_results = {}
        for check_name, check_fn in task["checks"]:
            try:
                check_results[check_name] = check_fn(content)
            except Exception:
                check_results[check_name] = False

        passed = sum(1 for v in check_results.values() if v)
        total = len(check_results)
        check_ratio = passed / max(total, 1)

        code_blocks = len(re.findall(r'```[\w]*\n.+?```', content, re.DOTALL))
        quality = check_ratio * 0.5 + min(1.0, len(content) / 6000) * 0.25 + min(1.0, code_blocks / 3) * 0.25

        success = check_ratio >= 0.65 and len(content) >= 2000

        record = {
            "task_id": task["id"],
            "tag": tag,
            "learn": learn,
            "domain": domain,
            "level": task.get("level", ""),
            "success": success,
            "quality": round(quality, 3),
            "checks_passed": passed,
            "checks_total": total,
            "check_ratio": round(check_ratio, 3),
            "content_length": len(content),
            "code_blocks": code_blocks,
            "elapsed": round(elapsed, 1),
            "ep_before": snap_before["episodes"],
            "ep_after": snap_after["episodes"],
            "ctx_injected": snap_before["ctx_len"],
            "ret_injected": snap_before["ret_len"],
            "error": error,
        }
        self.results.append(record)

        status = "PASS" if success else "FAIL"
        print(
            f"\n  [{status}] Q={quality:.2f} | Checks={passed}/{total} ({check_ratio:.0%}) | "
            f"Len={len(content)} | Code={code_blocks} | {elapsed:.0f}s"
        )
        print(f"  Episodes: {snap_before['episodes']}→{snap_after['episodes']} | "
              f"Rate: {snap_before['rate']:.0%}→{snap_after['rate']:.0%}")

        for name, ok in check_results.items():
            print(f"    [{'+'if ok else '-'}] {name}")

        return record

    def _print_results(self) -> None:
        separator("FINAL RESULTS")

        for r in self.results:
            s = "PASS" if r["success"] else "FAIL"
            tags = []
            if r["tag"]:
                tags.append(r["tag"])
            if not r["learn"]:
                tags.append("BASELINE")
            tag_str = f" [{','.join(tags)}]" if tags else ""
            ctx = r["ctx_injected"] + r["ret_injected"]
            ctx_str = f" ctx={ctx}" if ctx > 0 else ""
            print(
                f"  [{s}] {r['task_id']:25s} Q={r['quality']:.2f} "
                f"Chk={r['checks_passed']}/{r['checks_total']} ({r['check_ratio']:.0%}) "
                f"Len={r['content_length']:5d} Ep={r['ep_before']}→{r['ep_after']}"
                f"{tag_str}{ctx_str}"
            )

        # ─── ESCALATION LADDER ───
        esc = [r for r in self.results if r["task_id"].startswith("ESC_")]
        if len(esc) >= 2:
            subsep("ESCALATION LADDER (Easy → Medium → Hard)")
            for r in esc:
                bar_len = int(r["check_ratio"] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(
                    f"  {r['level']:6s} {bar} {r['check_ratio']:.0%} "
                    f"({r['checks_passed']}/{r['checks_total']}) "
                    f"ctx={r['ctx_injected']}ch"
                )
            first, last = esc[0]["check_ratio"], esc[-1]["check_ratio"]
            delta = last - first
            print(f"\n  Trend: {first:.0%} → {last:.0%} ({delta:+.0%})")
            verdict = "LEARNING HELPS" if delta > 0.05 else "MAINTAINED" if delta >= -0.05 else "DEGRADED"
            print(f"  Verdict: {verdict}")

        # ─── CROSS-DOMAIN TRANSFER ───
        seed = [r for r in self.results if r["task_id"] == "XFER_seed_ml_pipeline"]
        target = [r for r in self.results if r["task_id"] == "XFER_target_bayesian_opt"]
        if seed and target:
            subsep("CROSS-DOMAIN TRANSFER (coding → data_science)")
            s, t = seed[0], target[0]
            print(f"  Seed (coding):       Q={s['quality']:.2f} Chk={s['checks_passed']}/{s['checks_total']}")
            print(f"  Target (data_sci):   Q={t['quality']:.2f} Chk={t['checks_passed']}/{t['checks_total']} ctx={t['ctx_injected']}ch")
            if t["ctx_injected"] > 0:
                print(f"  Transfer signal: YES — learning context from seed domain reached target")
            else:
                print(f"  Transfer signal: NO — no cross-domain context injection")

        # ─── RETRY IMPROVEMENT ───
        retries = [r for r in self.results if r["task_id"] == "RETRY_compiler"]
        if len(retries) >= 2:
            subsep("RETRY IMPROVEMENT")
            for i, r in enumerate(retries):
                attempt = i + 1
                bar_len = int(r["check_ratio"] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(
                    f"  Attempt {attempt}: {bar} {r['check_ratio']:.0%} "
                    f"({r['checks_passed']}/{r['checks_total']}) "
                    f"ctx={r['ctx_injected']}ch"
                )
            first, last = retries[0]["check_ratio"], retries[-1]["check_ratio"]
            delta = last - first
            print(f"\n  Improvement: {first:.0%} → {last:.0%} ({delta:+.0%})")
            if delta > 0.05:
                print("  LEARNING IMPROVED RETRY ✓")
            elif delta >= -0.05:
                print("  ~ NEUTRAL")
            else:
                print("  ✗ REGRESSION")

        # ─── A/B MULTI-DOMAIN ───
        ab_base = [r for r in self.results if "AB_" in r["task_id"] and not r["learn"]]
        ab_learn = [r for r in self.results if "AB_" in r["task_id"] and r["learn"]]
        if ab_base and ab_learn:
            subsep("A/B MULTI-DOMAIN")
            for b in ab_base:
                match = [l for l in ab_learn if l["task_id"] == b["task_id"]]
                if match:
                    l = match[0]
                    delta = l["check_ratio"] - b["check_ratio"]
                    win = "LEARN WINS" if delta > 0.05 else "NEUTRAL" if abs(delta) <= 0.05 else "BASE WINS"
                    print(
                        f"  {b['task_id']:25s} Base={b['check_ratio']:.0%} Learn={l['check_ratio']:.0%} "
                        f"Δ={delta:+.0%} → {win}"
                    )

        # ─── OVERALL ───
        subsep("OVERALL METRICS")
        all_learned = [r for r in self.results if r["learn"]]
        avg_q = sum(r["quality"] for r in all_learned) / max(len(all_learned), 1)
        avg_chk = sum(r["check_ratio"] for r in all_learned) / max(len(all_learned), 1)
        pass_rate = sum(1 for r in all_learned if r["success"]) / max(len(all_learned), 1)
        total_ep = self.results[-1]["ep_after"] if self.results else 0
        retrieval_hits = sum(1 for r in self.results if r["ret_injected"] > 0)
        ctx_hits = sum(1 for r in self.results if r["ctx_injected"] > 0)

        print(f"  Tasks run:           {len(self.results)}")
        print(f"  Pass rate:           {pass_rate:.0%}")
        print(f"  Avg quality:         {avg_q:.2f}")
        print(f"  Avg check rate:      {avg_chk:.0%}")
        print(f"  Episodes accumulated: {total_ep}")
        print(f"  Context injected:    {ctx_hits}/{len(self.results)} tasks")
        print(f"  Retrieval injected:  {retrieval_hits}/{len(self.results)} tasks")

    async def run_all(self) -> None:
        separator("JOTTY COMPLEX LEARNING EVALUATION")
        now = datetime.now().isoformat()
        snap = self._snapshot()
        print(f"  Date: {now}")
        print(f"  Model: {EVAL_MODEL}")
        print(f"  Initial episodes: {snap['episodes']}")
        print(f"  Scenarios: Escalation + Cross-Domain Transfer + Retry + A/B")

        # ─── SCENARIO 1: ESCALATION LADDER ───
        separator("SCENARIO 1: ESCALATION LADDER (Easy → Medium → Hard)")
        for task in ESCALATION_TASKS:
            await self.run_task(task, tag=task["level"])
            print("  Waiting for learning...")
            await asyncio.sleep(15)

        # ─── SCENARIO 2: CROSS-DOMAIN TRANSFER ───
        separator("SCENARIO 2: CROSS-DOMAIN TRANSFER")
        print("  Step 1: Seed with coding task...")
        await self.run_task(TRANSFER_SEED_TASK, tag="seed")
        await asyncio.sleep(15)

        print("\n  Step 2: Test transfer to data science...")
        await self.run_task(TRANSFER_TARGET_TASK, tag="target")
        await asyncio.sleep(15)

        # ─── SCENARIO 3: RETRY IMPROVEMENT ───
        separator("SCENARIO 3: RETRY WITH LEARNING")
        print("  Attempt 1 (cold for this domain)...")
        await self.run_task(RETRY_TASK, tag="attempt_1")
        await asyncio.sleep(20)  # Let learning process

        print("\n  Attempt 2 (with learning from attempt 1)...")
        await self.run_task(RETRY_TASK, tag="attempt_2")
        await asyncio.sleep(15)

        # ─── SCENARIO 4: A/B MULTI-DOMAIN ───
        separator("SCENARIO 4: A/B TEST")
        for task in AB_TASKS:
            print(f"\n  Running BASELINE (no learning)...")
            await self.run_task(task, learn=False, tag="baseline")
            print(f"\n  Running WITH LEARNING...")
            await self.run_task(task, learn=True, tag="learning")
            await asyncio.sleep(10)

        # ─── RESULTS ───
        self._print_results()

        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"{ts}_complex_learning.json"
        with open(out, "w") as f:
            json.dump(
                {"timestamp": now, "model": EVAL_MODEL, "results": self.results},
                f, indent=2, default=str,
            )
        print(f"\n  Saved: {out}")


async def main() -> None:
    await ComplexEval().run_all()


if __name__ == "__main__":
    asyncio.run(main())
