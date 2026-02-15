"""Data structure tool — stack, queue, linked list, BST demonstrations."""

from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

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
            steps.append(
                f"search({val}) -> {'found at index ' + str(idx) if idx >= 0 else 'not found'}"
            )
    return {
        "final": ll[:],
        "steps": steps,
        "representation": " -> ".join(map(str, ll)) + " -> None" if ll else "empty",
    }


def _run_bst(ops: List[Dict]) -> Dict:
    nodes: Dict[int, Dict] = {}
    root: Optional[int] = None
    steps = []

    def insert(val: int) -> None:
        nonlocal root
        if root is None:
            root = val
            nodes[val] = {"left": None, "right": None}
            return
        cur = root
        while True:
            if val < cur:
                if nodes[cur]["left"] is None:
                    nodes[cur]["left"] = val
                    nodes[val] = {"left": None, "right": None}
                    return
                cur = nodes[cur]["left"]
            else:
                if nodes[cur]["right"] is None:
                    nodes[cur]["right"] = val
                    nodes[val] = {"left": None, "right": None}
                    return
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
