"""
Domain Classifier - Keyword-based domain detection for learning tasks.

Pure functions and constants for classifying task text into domains.
No dependencies on LearningService or any stateful components.

Extracted from learning_service.py for modularity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# =============================================================================
# DOMAIN CLASSIFIER - Detects actual domain from task text
# =============================================================================

# Keyword sets per domain, ordered by specificity (most specific first)
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coding": [
        "implement",
        "function",
        "class ",
        "code",
        "python",
        "javascript",
        "typescript",
        "algorithm",
        "data structure",
        "api",
        "rest ",
        "refactor",
        "debug",
        "unit test",
        "integration test",
        "lru",
        "cache",
        "rate limit",
        "database",
        "sql",
        "redis",
        "docker",
        "kubernetes",
        "deploy",
        "ci/cd",
        "git",
        "compile",
        "runtime",
        "bug",
        "exception",
        "stack trace",
        "leetcode",
        "linked list",
        "binary tree",
        "hash map",
        "thread",
        "async",
        "concurren",
    ],
    "research": [
        "research",
        "analyze",
        "analysis",
        "study",
        "survey",
        "paper",
        "literature",
        "findings",
        "evidence",
        "hypothesis",
        "methodology",
        "experiment",
        "dataset",
        "statistical",
        "quantitative",
        "qualitative",
        "peer review",
        "citation",
        "journal",
        "academic",
        "scholar",
    ],
    "system_design": [
        "system design",
        "design a",
        "architect",
        "scalab",
        "distributed",
        "microservice",
        "load balanc",
        "failover",
        "availability",
        "latency",
        "throughput",
        "kafka",
        "message queue",
        "event driven",
        "cqrs",
        "cap theorem",
        "consensus",
        "replication",
        "sharding",
        "partition",
        "p99",
        "infrastructure",
        "capacity planning",
        "concurrent users",
        "concurrent players",
        "crdt",
        "server mesh",
        "auto-scal",
        "multiplayer",
        "requests/sec",
        "req/s",
        "rps",
        "qps",
        "semilattice",
        "raft",
        "paxos",
        "vector clock",
        "gossip",
        "failure detect",
        "cluster member",
        "service discover",
        "byzantine",
        "fault toleran",
        "lock-free",
        "lock free",
        "linearizab",
        "two-phase commit",
        "two phase commit",
        "2pc",
        "saga pattern",
        "quorum",
        "leader election",
        "state machine replicat",
        "causal broadcast",
        "happens-before",
        "snapshot isolation",
        "serializab",
        "compare-and-swap",
        "compare and swap",
        "load shed",
        "circuit break",
        "bulkhead",
        "backpressure",
        "health check",
        "service mesh",
        "tick rate",
        "netcode",
        "matchmaking",
        "real-time",
    ],
    "data_science": [
        "machine learning",
        "neural network",
        "deep learning",
        "model",
        "training",
        "inference",
        "xgboost",
        "random forest",
        "regression",
        "classification",
        "clustering",
        "feature engineer",
        "backpropagation",
        "gradient",
        "loss function",
        "optimizer",
        "epoch",
        "batch",
        "transformer",
        "attention",
        "embedding",
        "fine-tun",
        "rl ",
        "reinforcement learning",
        "continual learning",
        "catastrophic forgetting",
        "pipeline",
        "scaler",
        "preprocessing",
        "cross-validation",
        "hyperparameter",
        "bayesian",
        "gaussian process",
    ],
    "economics": [
        "economic",
        "gdp",
        "inflation",
        "labor market",
        "monetary",
        "fiscal",
        "trade",
        "tariff",
        "supply chain",
        "market",
        "investment",
        "portfolio",
        "stock",
        "bond",
        "interest rate",
        "unemployment",
        "productivity",
        "inequality",
        "policy",
    ],
    "writing": [
        "write",
        "essay",
        "article",
        "blog",
        "content",
        "copywriting",
        "narrative",
        "storytelling",
        "creative writing",
        "proofread",
        "grammar",
        "tone",
        "audience",
        "persuasive",
        "report",
    ],
    "math": [
        "prove",
        "theorem",
        "lemma",
        "equation",
        "integral",
        "derivative",
        "matrix",
        "eigenvalue",
        "probability",
        "combinatorics",
        "topology",
        "group theory",
        "number theory",
        "optimization",
        "convex",
    ],
    "algorithms": [
        "sorting",
        "merge sort",
        "quick sort",
        "heap sort",
        "binary search",
        "depth-first",
        "breadth-first",
        "dfs",
        "bfs",
        "dijkstra",
        "bellman-ford",
        "dynamic programming",
        "greedy algorithm",
        "backtracking",
        "divide and conquer",
        "b-tree",
        "b+ tree",
        "avl tree",
        "red-black tree",
        "balanced tree",
        "trie",
        "graph algorithm",
        "shortest path",
        "minimum spanning",
        "topological sort",
        "time complexity",
        "space complexity",
        "big-o",
        "O(n",
        "amortized",
        "recurrence",
        "fibonacci heap",
    ],
    "compiler_design": [
        "compiler",
        "parser",
        "lexer",
        "tokenizer",
        "interpreter",
        "abstract syntax tree",
        "ast ",
        "token",
        "grammar",
        "recursive descent",
        "precedence",
        "parse expression",
        "scanner",
        "code generation",
        "semantic analysis",
        "scope",
        "closure",
        "type checking",
        "ir ",
        "intermediate representation",
    ],
}

# Domain affinity map: for cross-domain transfer, related domains share patterns.
# Key = source domain, value = list of related domains ordered by affinity.
_DOMAIN_AFFINITY: Dict[str, List[str]] = {
    "coding": ["system_design", "data_science", "devops"],
    "system_design": ["coding", "devops", "data_science"],
    "data_science": ["coding", "math", "research"],
    "math": ["data_science", "economics", "research"],
    "economics": ["math", "data_science", "research"],
    "research": ["data_science", "economics", "writing"],
    "devops": ["coding", "system_design"],
    "writing": ["research", "economics"],
    "algorithms": ["coding", "math", "system_design"],
    "compiler_design": ["coding", "algorithms"],
}


def _get_related_domains(domain: str) -> List[str]:
    """Get related domains for cross-domain transfer, ordered by affinity."""
    related = _DOMAIN_AFFINITY.get(domain, [])
    if not related:
        # Fall back to domains that list this domain as related
        related = [d for d, affinities in _DOMAIN_AFFINITY.items() if domain in affinities]
    return related


def classify_domain(text: str) -> Tuple[str, str]:
    """
    Classify task text into (domain, task_type) using keyword matching.

    Returns the most specific matching domain, with ties broken by
    match count. Falls back to 'general' if no strong match.
    """
    text_lower = text.lower()
    scores: Dict[str, float] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = float(score)

    if not scores:
        return "general", "general"

    # Generic implementation keywords appear in almost every technical task.
    # When a specialized domain (system_design, data_science, economics) also
    # matches, demote generic coding keywords to fractional weight so the
    # specialized domain wins.
    _generic_coding = {
        "implement",
        "function",
        "class ",
        "code",
        "python",
        "javascript",
        "typescript",
        "unit test",
        "integration test",
    }
    specialized = {d for d in scores if d not in ("coding", "writing", "general")}
    if "coding" in scores and specialized:
        coding_hits = [kw for kw in _DOMAIN_KEYWORDS["coding"] if kw in text_lower]
        generic_count = sum(1 for kw in coding_hits if kw in _generic_coding)
        specific_count = len(coding_hits) - generic_count
        scores["coding"] = specific_count + generic_count * 0.25

    specificity_bonus = {
        "system_design": 1.0,
        "economics": 0.8,
        "algorithms": 0.7,
        "compiler_design": 0.9,
        "data_science": 0.5,
    }
    adjusted = {d: s + specificity_bonus.get(d, 0) for d, s in scores.items()}
    best_domain = max(adjusted, key=adjusted.get)  # type: ignore[arg-type]

    keywords_hit = [kw for kw in _DOMAIN_KEYWORDS[best_domain] if kw in text_lower]
    task_type = keywords_hit[0].strip() if keywords_hit else best_domain

    # Cross-domain detection: if 2+ domains score highly, mark as synthesis
    high_scorers = [d for d, s in scores.items() if s >= 3]
    if len(high_scorers) >= 3:
        return "synthesis", "cross_domain"
    if len(high_scorers) == 2:
        return best_domain, f"{high_scorers[0]}+{high_scorers[1]}"

    return best_domain, task_type
