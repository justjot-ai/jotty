"""Sorting visualizer — step-by-step sorting demonstrations."""

from typing import Any, Dict, List, Tuple

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

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
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
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
        return tool_response(
            original=arr, sorted=result, steps=steps, step_count=len(steps), algorithm=algo
        )
    except Exception as e:
        return tool_error(str(e))


__all__ = ["sorting_tool"]
