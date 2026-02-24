#!/usr/bin/env python3
"""
SQL Agent Evaluation: DSPy optimization with sqlglot library validation.

End-to-end test:
  1. Seed a realistic e-commerce SQLite database
  2. Generate gold SQL examples from Sonnet (teacher)
  3. Optimize DSPy SQL module with sqlglot validation cascade
  4. A/B test: Haiku baseline vs Haiku+DSPy vs Sonnet
  5. Execute every generated query against the real DB to verify correctness

Usage:
    python scripts/eval_sql_agent.py
"""

import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Load .env if ANTHROPIC_API_KEY not in environment
if not os.environ.get("ANTHROPIC_API_KEY"):
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DB_PATH = Path(__file__).parent / "eval_results" / "ecommerce_test.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. SEED DATABASE
# ============================================================================

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    city TEXT,
    country TEXT DEFAULT 'US',
    tier TEXT CHECK(tier IN ('free','premium','enterprise')) DEFAULT 'free',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    status TEXT CHECK(status IN ('pending','shipped','delivered','cancelled')) DEFAULT 'pending',
    total REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    comment TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

SEED_DATA = """
INSERT OR IGNORE INTO customers (id, name, email, city, country, tier, created_at) VALUES
(1, 'Alice Johnson', 'alice@example.com', 'New York', 'US', 'premium', '2025-01-15'),
(2, 'Bob Smith', 'bob@example.com', 'London', 'UK', 'enterprise', '2025-02-01'),
(3, 'Charlie Brown', 'charlie@example.com', 'Tokyo', 'JP', 'free', '2025-03-10'),
(4, 'Diana Prince', 'diana@example.com', 'Paris', 'FR', 'premium', '2025-01-20'),
(5, 'Eve Davis', 'eve@example.com', 'New York', 'US', 'free', '2025-04-01'),
(6, 'Frank Miller', 'frank@example.com', 'Berlin', 'DE', 'enterprise', '2025-02-15'),
(7, 'Grace Lee', 'grace@example.com', 'Seoul', 'KR', 'premium', '2025-05-01'),
(8, 'Hank Wilson', 'hank@example.com', 'Sydney', 'AU', 'free', '2025-03-20'),
(9, 'Ivy Chen', 'ivy@example.com', 'San Francisco', 'US', 'enterprise', '2025-01-05'),
(10, 'Jack Taylor', 'jack@example.com', 'Toronto', 'CA', 'premium', '2025-06-01');

INSERT OR IGNORE INTO products (id, name, category, price, stock, is_active) VALUES
(1, 'Laptop Pro 15', 'Electronics', 1299.99, 50, 1),
(2, 'Wireless Mouse', 'Electronics', 29.99, 200, 1),
(3, 'Standing Desk', 'Furniture', 549.00, 30, 1),
(4, 'Noise-Cancelling Headphones', 'Electronics', 349.99, 75, 1),
(5, 'Ergonomic Chair', 'Furniture', 899.00, 15, 1),
(6, 'USB-C Hub', 'Electronics', 49.99, 300, 1),
(7, 'Mechanical Keyboard', 'Electronics', 159.99, 100, 1),
(8, 'Monitor 27"', 'Electronics', 449.99, 40, 1),
(9, 'Desk Lamp', 'Furniture', 79.99, 120, 1),
(10, 'Webcam HD', 'Electronics', 89.99, 80, 0);

INSERT OR IGNORE INTO orders (id, customer_id, status, total, created_at) VALUES
(1, 1, 'delivered', 1329.98, '2025-02-01'),
(2, 1, 'delivered', 549.00, '2025-03-15'),
(3, 2, 'shipped', 1599.98, '2025-04-01'),
(4, 3, 'pending', 29.99, '2025-05-10'),
(5, 4, 'delivered', 899.00, '2025-02-20'),
(6, 5, 'cancelled', 159.99, '2025-04-15'),
(7, 6, 'delivered', 2148.98, '2025-03-01'),
(8, 7, 'shipped', 449.99, '2025-05-20'),
(9, 2, 'delivered', 349.99, '2025-04-15'),
(10, 9, 'delivered', 1849.98, '2025-02-10'),
(11, 1, 'pending', 79.99, '2025-06-01'),
(12, 4, 'delivered', 209.98, '2025-05-01'),
(13, 6, 'delivered', 549.00, '2025-04-20'),
(14, 10, 'shipped', 1299.99, '2025-06-10'),
(15, 8, 'pending', 129.98, '2025-06-15');

INSERT OR IGNORE INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 1299.99), (1, 2, 1, 29.99),
(2, 3, 1, 549.00),
(3, 1, 1, 1299.99), (3, 7, 1, 159.99), (3, 6, 1, 49.99),
(4, 2, 1, 29.99),
(5, 5, 1, 899.00),
(6, 7, 1, 159.99),
(7, 1, 1, 1299.99), (7, 5, 1, 899.00),
(8, 8, 1, 449.99),
(9, 4, 1, 349.99),
(10, 1, 1, 1299.99), (10, 3, 1, 549.00),
(11, 9, 1, 79.99),
(12, 7, 1, 159.99), (12, 6, 1, 49.99),
(13, 3, 1, 549.00),
(14, 1, 1, 1299.99),
(15, 2, 1, 29.99), (15, 6, 1, 49.99), (15, 9, 1, 79.99);

INSERT OR IGNORE INTO reviews (customer_id, product_id, rating, comment, created_at) VALUES
(1, 1, 5, 'Amazing laptop, very fast', '2025-02-15'),
(1, 3, 4, 'Great desk but took a while to assemble', '2025-03-20'),
(2, 1, 4, 'Good build quality', '2025-04-10'),
(2, 4, 5, 'Best headphones I have ever owned', '2025-04-20'),
(4, 5, 5, 'Worth every penny for my back', '2025-03-01'),
(6, 1, 4, 'Solid performance', '2025-03-15'),
(6, 5, 3, 'Decent chair but armrests are stiff', '2025-04-25'),
(7, 8, 4, 'Great monitor for the price', '2025-05-25'),
(9, 1, 5, 'Perfect for development work', '2025-02-20'),
(9, 3, 4, 'Sturdy desk', '2025-02-25'),
(10, 1, 3, 'Battery life could be better', '2025-06-15');
"""


def seed_database() -> sqlite3.Connection:
    """Create and seed the e-commerce test database."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA_DDL)
    conn.executescript(SEED_DATA)
    conn.commit()

    # Print schema summary
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cur.fetchone()[0]
        logger.info(f"  {t}: {cnt} rows")
    return conn


def get_schema_ddl() -> str:
    """Return the DDL string for LLM context."""
    return SCHEMA_DDL.strip()


def execute_sql(conn: sqlite3.Connection, sql: str) -> Tuple[bool, Any]:
    """Execute SQL and return (success, result_or_error)."""
    try:
        # Strip markdown fences
        clean = sql.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()

        cur = conn.cursor()
        cur.execute(clean)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return True, {"columns": cols, "rows": rows[:20], "row_count": len(rows)}
    except Exception as e:
        return False, str(e)


# ============================================================================
# 2. GOLD GENERATION + OPTIMIZATION
# ============================================================================

GOLD_TASKS = [
    "Write a SQL query to find the top 5 customers by total spending, showing customer name, total spent, and number of orders.",
    "Write a SQL query to find products that have never been ordered.",
    "Write a SQL query to calculate monthly revenue for 2025, showing month and total revenue, only for delivered orders.",
    "Write a SQL query to find the average rating per product category, only for products with at least 2 reviews.",
    "Write a SQL query using a CTE to find customers who have placed orders in at least 2 different months.",
    "Write a SQL query to find the top 3 products by revenue (quantity * unit_price from order_items), with product name and total revenue.",
    "Write a SQL query to find customers who have ordered products from both Electronics and Furniture categories.",
    "Write a SQL query to rank customers by their average order value using a window function, showing name, avg order value, and rank.",
    "Write a SQL query to find all orders where the total exceeds the average order total across all orders.",
    "Write a SQL query to show each product with its review count, average rating, and total quantity sold, using LEFT JOINs.",
]

EVAL_TASKS = [
    (
        "Find the total revenue per country for delivered orders, sorted descending.",
        "Should show country and total. US should have highest revenue.",
    ),
    (
        "Which customers have never left a review?",
        "Should list customers with no entries in reviews table.",
    ),
    (
        "Show the top 3 product categories by number of distinct customers who ordered them.",
        "Need to join orders → order_items → products, count distinct customers.",
    ),
    (
        "Find orders that contain more than 2 items.",
        "Need GROUP BY order_id HAVING COUNT(*) > 2 on order_items.",
    ),
    (
        "Calculate each customer's lifetime value (total spending) and classify as 'high' (>$1000), 'medium' ($500-$1000), or 'low' (<$500).",
        "Requires CASE/WHEN and SUM with GROUP BY.",
    ),
]


def generate_gold_and_optimize(conn: sqlite3.Connection) -> Any:
    """Generate gold SQL from Sonnet, validate with sqlglot + execution, optimize DSPy module."""
    from core.intelligence.learning.advanced_learning import (
        DomainDSPyOptimizer,
        validate_output,
    )

    optimizer = DomainDSPyOptimizer.get_instance()
    schema_ddl = get_schema_ddl()

    # Generate gold using Sonnet — with schema context
    import anthropic

    client = anthropic.Anthropic()
    added = 0

    logger.info("\n=== Generating gold SQL from Sonnet ===")
    for i, task in enumerate(GOLD_TASKS):
        prompt = (
            f"Given this SQLite database schema:\n\n{schema_ddl}\n\n"
            f"{task}\n\n"
            f"Output ONLY the SQL query, no explanation. Use SQLite-compatible syntax."
        )
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            sql = resp.content[0].text.strip()

            # Level 1: sqlglot validation
            vr = validate_output(sql, "sql")
            # Level 1b: Execute against real DB
            ok, result = execute_sql(conn, sql)

            status = "OK" if (vr.valid and ok) else "FAIL"
            if vr.valid and ok:
                optimizer.add_gold_examples("sql", [{"task": task, "output": sql}])
                added += 1
            detail = f"sqlglot={vr.valid}" + (
                f" exec={ok}" if vr.valid else f" err={vr.errors[:1]}"
            )
            logger.info(f"  [{i+1}/{len(GOLD_TASKS)}] {status}: {detail} — {task[:60]}...")
        except Exception as e:
            logger.error(f"  [{i+1}] ERROR: {e}")

    logger.info(f"\nGold: {added}/{len(GOLD_TASKS)} valid (sqlglot + execution)")

    # Optimize DSPy module
    logger.info("\n=== Optimizing DSPy SQL module ===")
    optimized = optimizer.optimize("sql", num_candidate_programs=4)
    return optimized


# ============================================================================
# 3. A/B TEST
# ============================================================================


def run_ab_test(conn: sqlite3.Connection) -> List[Dict]:
    """Compare Haiku baseline vs Haiku+DSPy vs Sonnet on eval tasks."""
    import anthropic
    import dspy

    from core.intelligence.learning.advanced_learning import (
        DomainDSPyOptimizer,
        validate_output,
    )
    from core.infrastructure.foundation.unified_lm_provider import UnifiedLMProvider

    schema_ddl = get_schema_ddl()
    client = anthropic.Anthropic()
    optimizer = DomainDSPyOptimizer.get_instance()
    dspy_module = optimizer.load_optimized("sql")

    student_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="haiku")
    sonnet_lm = UnifiedLMProvider.create_lm(provider="anthropic", model="sonnet")

    results = []

    for task_desc, expected_hint in EVAL_TASKS:
        row: Dict[str, Any] = {"task": task_desc, "hint": expected_hint}

        prompt = (
            f"Given this SQLite database schema:\n\n{schema_ddl}\n\n"
            f"{task_desc}\n\n"
            f"Output ONLY the SQL query, no explanation."
        )

        # --- A: Haiku baseline ---
        t0 = time.time()
        try:
            resp = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            sql_a = resp.content[0].text.strip()
        except Exception as e:
            sql_a = f"ERROR: {e}"
        time_a = time.time() - t0

        vr_a = validate_output(sql_a, "sql")
        exec_a, res_a = execute_sql(conn, sql_a) if vr_a.valid else (False, "sqlglot rejected")

        row["haiku"] = {
            "sql": sql_a[:500],
            "time": round(time_a, 2),
            "sqlglot_valid": vr_a.valid,
            "sqlglot_method": vr_a.method,
            "sqlglot_confidence": vr_a.confidence,
            "exec_ok": exec_a,
            "rows": res_a.get("row_count", 0) if isinstance(res_a, dict) else 0,
            "error": res_a if not exec_a else None,
        }

        # --- B: Haiku + DSPy ---
        t0 = time.time()
        if dspy_module:
            try:
                task_with_schema = f"Schema:\n{schema_ddl}\n\nTask: {task_desc}\nOutput ONLY SQL."
                with dspy.context(lm=student_lm):
                    pred = dspy_module(task_description=task_with_schema, domain="sql")
                sql_b = getattr(pred, "output", "")
            except Exception as e:
                sql_b = f"ERROR: {e}"
        else:
            sql_b = "NO OPTIMIZED MODULE"
        time_b = time.time() - t0

        vr_b = validate_output(sql_b, "sql")
        exec_b, res_b = execute_sql(conn, sql_b) if vr_b.valid else (False, "sqlglot rejected")

        row["haiku_dspy"] = {
            "sql": sql_b[:500],
            "time": round(time_b, 2),
            "sqlglot_valid": vr_b.valid,
            "sqlglot_method": vr_b.method,
            "sqlglot_confidence": vr_b.confidence,
            "exec_ok": exec_b,
            "rows": res_b.get("row_count", 0) if isinstance(res_b, dict) else 0,
            "error": res_b if not exec_b else None,
        }

        # --- C: Sonnet (gold standard) ---
        t0 = time.time()
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            sql_c = resp.content[0].text.strip()
        except Exception as e:
            sql_c = f"ERROR: {e}"
        time_c = time.time() - t0

        vr_c = validate_output(sql_c, "sql")
        exec_c, res_c = execute_sql(conn, sql_c) if vr_c.valid else (False, "sqlglot rejected")

        row["sonnet"] = {
            "sql": sql_c[:500],
            "time": round(time_c, 2),
            "sqlglot_valid": vr_c.valid,
            "sqlglot_method": vr_c.method,
            "sqlglot_confidence": vr_c.confidence,
            "exec_ok": exec_c,
            "rows": res_c.get("row_count", 0) if isinstance(res_c, dict) else 0,
            "error": res_c if not exec_c else None,
        }

        results.append(row)

    return results


def print_results(results: List[Dict]) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("SQL AGENT A/B TEST: Haiku vs Haiku+DSPy vs Sonnet")
    print("Validation: sqlglot (L1 library) + real DB execution")
    print("=" * 100)

    for i, row in enumerate(results):
        print(f"\n{'─' * 100}")
        print(f"Task {i+1}: {row['task']}")
        print(f"  Hint: {row['hint']}")

        for label, key in [
            ("A) Haiku", "haiku"),
            ("B) Haiku+DSPy", "haiku_dspy"),
            ("C) Sonnet", "sonnet"),
        ]:
            d = row[key]
            sqlglot_str = f"sqlglot={'PASS' if d['sqlglot_valid'] else 'FAIL'}({d['sqlglot_method']},{d['sqlglot_confidence']:.2f})"
            exec_str = f"exec={'OK' if d['exec_ok'] else 'FAIL'}"
            rows_str = f"rows={d['rows']}"
            time_str = f"{d['time']:.1f}s"

            print(f"\n  {label}: {sqlglot_str} | {exec_str} | {rows_str} | {time_str}")
            sql_preview = d["sql"].replace("\n", " ")[:120]
            print(f"    SQL: {sql_preview}")
            if d.get("error") and not d["exec_ok"]:
                err = str(d["error"])[:120]
                print(f"    ERR: {err}")

    # Summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")

    headers = ["Metric", "Haiku", "Haiku+DSPy", "Sonnet"]
    print(f"\n  {headers[0]:<25} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15}")
    print(f"  {'─'*70}")

    for metric_name, key_fn in [
        ("sqlglot valid", lambda d: d["sqlglot_valid"]),
        ("Execution OK", lambda d: d["exec_ok"]),
        ("Avg time (s)", lambda d: d["time"]),
    ]:
        vals = []
        for model_key in ["haiku", "haiku_dspy", "sonnet"]:
            raw = [key_fn(r[model_key]) for r in results]
            if isinstance(raw[0], bool):
                vals.append(f"{sum(raw)}/{len(raw)}")
            else:
                vals.append(f"{sum(raw)/len(raw):.2f}")
        print(f"  {metric_name:<25} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15}")

    # Determine winner per task
    wins = {"haiku": 0, "haiku_dspy": 0, "sonnet": 0}
    for row in results:
        scores = {}
        for key in ["haiku", "haiku_dspy", "sonnet"]:
            d = row[key]
            scores[key] = (2 if d["exec_ok"] else 0) + (1 if d["sqlglot_valid"] else 0)
        best = max(scores, key=scores.get)
        wins[best] += 1

    print(
        f"\n  {'Task wins':<25} {wins['haiku']:<15} {wins['haiku_dspy']:<15} {wins['sonnet']:<15}"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():
    logger.info("=== SQL Agent Evaluation ===")
    logger.info(f"Database: {DB_PATH}")

    # Step 1: Seed DB
    logger.info("\n--- Step 1: Seeding database ---")
    conn = seed_database()

    # Step 2: Generate gold + optimize
    logger.info("\n--- Step 2: Gold generation + DSPy optimization ---")
    optimized = generate_gold_and_optimize(conn)

    # Step 3: A/B test
    logger.info("\n--- Step 3: A/B test ---")
    results = run_ab_test(conn)

    # Step 4: Print results
    print_results(results)

    # Save raw results
    out_path = DB_PATH.parent / f"sql_agent_eval_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nRaw results saved to: {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
