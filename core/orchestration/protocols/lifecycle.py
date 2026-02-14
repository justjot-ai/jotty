"""
Lifecycle protocol mixin: priority queue, task decomposition, leader election, scaling, caching, parallel execution.

Extracted from SwarmIntelligence for modularity.
These are mixed into SwarmIntelligence at class definition.
"""

import asyncio
import time
import logging
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict

from ..swarm_data_structures import (
    AgentSpecialization, AgentProfile, ConsensusVote, SwarmDecision,
    AgentSession, HandoffContext, Coalition, AuctionBid, GossipMessage, SupervisorNode,
)

logger = logging.getLogger(__name__)


class LifecycleMixin:
    """Lifecycle protocol mixin: priority queue, task decomposition, leader election, scaling, caching, parallel execution."""


    # =========================================================================
    # PRIORITY QUEUE (Handle urgent tasks first)
    # =========================================================================

    def __init_priority_queue(self):
        """Initialize priority queue if not exists."""
        if not hasattr(self, 'priority_queue'):
            self.priority_queue: List[Dict] = []



    def enqueue_task(
        self,
        task_id: str,
        task_type: str,
        priority: int = 5,
        deadline: float = None,
        context: Dict = None
    ):
        """
        Add task to priority queue.

        Priority 1-10 (10 = most urgent).
        """
        self.__init_priority_queue()

        task = {
            "task_id": task_id,
            "task_type": task_type,
            "priority": priority,
            "deadline": deadline or (time.time() + 3600),
            "context": context or {},
            "enqueued_at": time.time()
        }

        # Insert in priority order (higher priority first)
        inserted = False
        for i, t in enumerate(self.priority_queue):
            if priority > t["priority"]:
                self.priority_queue.insert(i, task)
                inserted = True
                break
            elif priority == t["priority"]:
                # Same priority: earlier deadline first
                if task["deadline"] < t["deadline"]:
                    self.priority_queue.insert(i, task)
                    inserted = True
                    break

        if not inserted:
            self.priority_queue.append(task)

        logger.debug(f"Task enqueued: {task_id} (priority={priority})")



    def dequeue_task(self) -> Optional[Dict]:
        """Get highest priority task from queue."""
        self.__init_priority_queue()

        if not self.priority_queue:
            return None

        return self.priority_queue.pop(0)



    def peek_queue(self, n: int = 5) -> List[Dict]:
        """Peek at top N tasks in queue."""
        self.__init_priority_queue()
        return self.priority_queue[:n]



    def escalate_priority(self, task_id: str, new_priority: int) -> None:
        """Escalate task priority (reposition in queue)."""
        self.__init_priority_queue()

        # Find and remove task
        task = None
        for i, t in enumerate(self.priority_queue):
            if t["task_id"] == task_id:
                task = self.priority_queue.pop(i)
                break

        if task:
            # Re-enqueue with new priority (only pass valid params)
            self.enqueue_task(
                task_id=task["task_id"],
                task_type=task["task_type"],
                priority=new_priority,
                deadline=task.get("deadline"),
                context=task.get("context")
            )
            logger.info(f"Task escalated: {task_id} → priority {new_priority}")

    # =========================================================================
    # TASK DECOMPOSITION (Split complex tasks)
    # =========================================================================



    # =========================================================================
    # TASK DECOMPOSITION (Split complex tasks)
    # =========================================================================

    def decompose_task(
        self,
        task_id: str,
        task_type: str,
        subtasks: List[Dict],
        parallel: bool = True
    ) -> List[str]:
        """
        Decompose complex task into subtasks.

        Args:
            task_id: Parent task ID
            task_type: Type of parent task
            subtasks: List of {"type": str, "context": dict, "priority": int}
            parallel: Whether subtasks can run in parallel

        Returns:
            List of subtask IDs assigned to agents.
        """
        subtask_ids = []

        for i, sub in enumerate(subtasks):
            sub_id = f"{task_id}_sub_{i}"
            sub_type = sub.get("type", task_type)
            sub_priority = sub.get("priority", 5)
            sub_context = sub.get("context", {})
            sub_context["parent_task"] = task_id
            sub_context["subtask_index"] = i
            sub_context["parallel"] = parallel

            # Route subtask to best agent
            route = self.smart_route(
                task_id=sub_id,
                task_type=sub_type,
                task_description=sub_context.get("description", "")
            )

            if route["assigned_agent"]:
                # Create handoff for subtask
                self.initiate_handoff(
                    task_id=sub_id,
                    from_agent="decomposer",
                    to_agent=route["assigned_agent"],
                    task_type=sub_type,
                    context=sub_context,
                    priority=sub_priority
                )
                subtask_ids.append(sub_id)

        logger.info(f"Task decomposed: {task_id} → {len(subtask_ids)} subtasks")
        return subtask_ids



    def aggregate_subtask_results(
        self,
        parent_task_id: str,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from completed subtasks.

        Args:
            parent_task_id: Parent task ID
            results: Dict of subtask_id -> result

        Returns:
            Aggregated result dict.
        """
        aggregated = {
            "parent_task": parent_task_id,
            "subtask_count": len(results),
            "successful": sum(1 for r in results.values() if r.get("success", False)),
            "results": results,
            "aggregated_at": time.time()
        }

        # Calculate overall success
        aggregated["overall_success"] = aggregated["successful"] == aggregated["subtask_count"]

        return aggregated

    # =========================================================================
    # BYZANTINE CONSENSUS (Fault-tolerant agreement)
    # =========================================================================



    # =========================================================================
    # EMERGENT LEADERSHIP (Dynamic leader election)
    # =========================================================================

    def elect_leader(self, candidates: List[str] = None, task_type: str = None) -> Optional[str]:
        """
        Elect leader based on trust, success rate, and availability.

        Emergent leadership pattern: Best performer leads.
        """
        candidates = candidates or list(self.agent_profiles.keys())
        available = self.get_available_agents(candidates)

        if not available:
            return None

        def score_candidate(agent: str) -> float:
            profile = self.agent_profiles.get(agent, AgentProfile(agent))
            score = profile.trust_score * 0.4

            if task_type:
                score += profile.get_success_rate(task_type) * 0.3
            else:
                # Overall success
                total_s = sum(s for s, t in profile.task_success.values())
                total_t = sum(t for s, t in profile.task_success.values())
                score += (total_s / total_t if total_t > 0 else 0.5) * 0.3

            # Low load bonus
            load = self.get_agent_load(agent)
            score += (1 - load) * 0.2

            # Experience bonus
            score += min(0.1, profile.total_tasks / 100)

            return score

        leader = max(available, key=score_candidate)
        logger.info(f"Leader elected: {leader} (score: {score_candidate(leader):.2f})")
        return leader

    # =========================================================================
    # ADAPTIVE TIMEOUT (Adjust based on task/agent history)
    # =========================================================================



    # =========================================================================
    # ADAPTIVE TIMEOUT (Adjust based on task/agent history)
    # =========================================================================

    def get_adaptive_timeout(self, agent: str, task_type: str, base_timeout: float = 30.0) -> float:
        """
        Calculate adaptive timeout based on agent's historical performance.

        Slow agents get more time, fast agents get less.
        """
        profile = self.agent_profiles.get(agent)
        if not profile or profile.total_tasks < 3:
            return base_timeout

        # Use agent's average execution time
        avg_time = profile.avg_execution_time

        # Add buffer based on variance (if we had it, use 1.5x for now)
        timeout = avg_time * 2.0

        # Clamp to reasonable bounds
        return max(base_timeout * 0.5, min(base_timeout * 3, timeout))

    # =========================================================================
    # AGENT LIFECYCLE (Spawn/retire dynamically)
    # =========================================================================



    # =========================================================================
    # AGENT LIFECYCLE (Spawn/retire dynamically)
    # =========================================================================

    def should_spawn_agent(self, task_type: str = None) -> bool:
        """
        Determine if new agent should be spawned.

        Based on: load, queue size, specialization gaps.
        """
        backpressure = self.calculate_backpressure()
        if backpressure < 0.7:
            return False

        # Check specialization gap
        if task_type:
            specialists = [a for a, p in self.agent_profiles.items()
                         if p.get_success_rate(task_type) > 0.8]
            if len(specialists) < 2:
                return True

        return backpressure > 0.85



    def should_retire_agent(self, agent: str) -> bool:
        """
        Determine if agent should be retired.

        Based on: low trust, high failure rate, idle time.
        """
        profile = self.agent_profiles.get(agent)
        if not profile:
            return False

        # Low trust
        if profile.trust_score < 0.2:
            return True

        # High failure rate with enough history
        if profile.total_tasks >= 10:
            total_s = sum(s for s, t in profile.task_success.values())
            total_t = sum(t for s, t in profile.task_success.values())
            if total_t > 0 and total_s / total_t < 0.3:
                return True

        # Circuit breaker open for too long
        cb = self.circuit_breakers.get(agent, {}) if hasattr(self, 'circuit_breakers') else {}
        if cb.get('state') == 'open' and time.time() - cb.get('last_failure', 0) > 300:
            return True

        return False



    def retire_agent(self, agent: str) -> None:
        """Remove agent from swarm."""
        if agent in self.agent_profiles:
            del self.agent_profiles[agent]

        # Clean up related state
        self.agent_coalitions.pop(agent, None)
        if hasattr(self, 'circuit_breakers'):
            self.circuit_breakers.pop(agent, None)

        # Reassign pending handoffs
        for task_id, handoff in list(self.pending_handoffs.items()):
            if handoff.to_agent == agent:
                available = [a for a in self.agent_profiles.keys() if a != agent]
                if available:
                    new_agent = self.get_best_agent_for_task(handoff.task_type, available)
                    if new_agent:
                        handoff.to_agent = new_agent

        logger.info(f"Agent retired: {agent}")

    # =========================================================================
    # PARALLEL EXECUTION (Speed up multi-agent tasks)
    # =========================================================================



    # =========================================================================
    # PARALLEL EXECUTION (Speed up multi-agent tasks)
    # =========================================================================

    async def execute_parallel(
        self,
        tasks: List[Dict],
        timeout_per_task: float = 30.0
    ) -> List[Dict]:
        """
        Execute multiple tasks in parallel across agents.

        Args:
            tasks: List of {"task_id", "task_type", "func", "args", "kwargs"}
            timeout_per_task: Timeout per task in seconds

        Returns:
            List of results with success/failure status
        """
        import asyncio

        async def run_task(task: Dict) -> Dict:
            task_id = task.get("task_id", "unknown")
            func = task.get("func")
            args = task.get("args", [])
            kwargs = task.get("kwargs", {})

            start = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_per_task
                    )
                else:
                    result = func(*args, **kwargs)

                return {
                    "task_id": task_id,
                    "success": True,
                    "result": result,
                    "execution_time": time.time() - start
                }
            except asyncio.TimeoutError:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": "timeout",
                    "execution_time": timeout_per_task
                }
            except Exception as e:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start
                }

        # Run all tasks concurrently
        results = await asyncio.gather(*[run_task(t) for t in tasks], return_exceptions=True)

        # Process results
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed.append({
                    "task_id": tasks[i].get("task_id", f"task_{i}"),
                    "success": False,
                    "error": str(r)
                })
            else:
                processed.append(r)

        return processed



    async def parallel_map(
        self,
        items: List[Any],
        func,
        max_concurrent: int = 5
    ) -> List[Any]:
        """
        Apply function to items in parallel with concurrency limit.

        Useful for processing multiple concepts, papers, etc.

        Args:
            items: Items to process
            func: Async function to apply
            max_concurrent: Max concurrent executions

        Returns:
            List of results in order
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_func(item, idx):
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(item)
                    else:
                        return func(item)
                except Exception as e:
                    logger.warning(f"parallel_map item {idx} failed: {e}")
                    return None

        results = await asyncio.gather(*[
            limited_func(item, i) for i, item in enumerate(items)
        ])

        return list(results)

    # =========================================================================
    # SMART CACHING (Reduce redundant LLM calls)
    # =========================================================================



    # =========================================================================
    # SMART CACHING (Reduce redundant LLM calls)
    # =========================================================================

    def __init_cache(self):
        """Initialize result cache."""
        if not hasattr(self, '_result_cache'):
            self._result_cache: Dict[str, Dict] = {}
            self._cache_hits = 0
            self._cache_misses = 0



    def cache_result(self, key: str, result: Any, ttl: float = 3600.0) -> None:
        """
        Cache a result with TTL.

        Args:
            key: Cache key
            result: Result to cache
            ttl: Time-to-live in seconds (default 1 hour)
        """
        self.__init_cache()
        self._result_cache[key] = {
            "result": result,
            "cached_at": time.time(),
            "ttl": ttl
        }



    def get_cached(self, key: str) -> Optional[Any]:
        """
        Get cached result if not expired.

        Returns None if not cached or expired.
        """
        self.__init_cache()

        entry = self._result_cache.get(key)
        if not entry:
            self._cache_misses += 1
            return None

        # Check TTL
        age = time.time() - entry["cached_at"]
        if age > entry["ttl"]:
            del self._result_cache[key]
            self._cache_misses += 1
            return None

        self._cache_hits += 1
        return entry["result"]



    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self.__init_cache()
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0,
            "size": len(self._result_cache)
        }



    def clear_cache(self, pattern: str = None) -> None:
        """Clear cache entries (optionally matching pattern)."""
        self.__init_cache()
        if pattern:
            import fnmatch
            keys_to_delete = [k for k in self._result_cache if fnmatch.fnmatch(k, pattern)]
            for k in keys_to_delete:
                del self._result_cache[k]
        else:
            self._result_cache.clear()

    # =========================================================================
    # INCREMENTAL PROCESSING (Stream results as they complete)
    # =========================================================================



    # =========================================================================
    # INCREMENTAL PROCESSING (Stream results as they complete)
    # =========================================================================

    async def execute_incremental(
        self,
        tasks: List[Dict],
        on_complete=None,
        on_error=None
    ):
        """
        Execute tasks and yield results incrementally as they complete.

        Args:
            tasks: List of {"task_id", "func", "args", "kwargs"}
            on_complete: Callback(task_id, result) called on each completion
            on_error: Callback(task_id, error) called on each error

        Yields:
            Results as they complete (not in order)
        """
        import asyncio

        async def run_task(task: Dict):
            task_id = task.get("task_id", "unknown")
            func = task.get("func")
            args = task.get("args", [])
            kwargs = task.get("kwargs", {})

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if on_complete:
                    on_complete(task_id, result)
                return {"task_id": task_id, "success": True, "result": result}

            except Exception as e:
                if on_error:
                    on_error(task_id, e)
                return {"task_id": task_id, "success": False, "error": str(e)}

        # Use as_completed for incremental results
        pending = [asyncio.create_task(run_task(t)) for t in tasks]

        for coro in asyncio.as_completed(pending):
            result = await coro
            yield result

    # =========================================================================
    # CHUNKED PROCESSING (Process large batches efficiently)
    # =========================================================================



    # =========================================================================
    # CHUNKED PROCESSING (Process large batches efficiently)
    # =========================================================================

    async def process_in_chunks(
        self,
        items: List[Any],
        chunk_size: int,
        process_func,
        delay_between_chunks: float = 0.1
    ) -> List[Any]:
        """
        Process items in chunks to avoid overwhelming LLM.

        Useful for processing many concepts without timeouts.

        Args:
            items: All items to process
            chunk_size: Items per chunk
            process_func: Async function to process a chunk
            delay_between_chunks: Delay between chunks

        Returns:
            All results combined
        """
        import asyncio

        results = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]

            # Check backpressure
            if not self.should_accept_task(priority=5):
                logger.warning(f"Backpressure high, waiting before chunk {i // chunk_size}")
                await asyncio.sleep(1.0)

            chunk_results = await process_func(chunk)
            results.extend(chunk_results if isinstance(chunk_results, list) else [chunk_results])

            if delay_between_chunks > 0 and i + chunk_size < len(items):
                await asyncio.sleep(delay_between_chunks)

        return results

    # =========================================================================
    # SWARM HEALTH MONITORING
    # =========================================================================

