"""Graph algorithms — Dijkstra, BFS, cycle detection, topological sort."""
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
