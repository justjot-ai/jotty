"""Compression analyzer — entropy, RLE, Huffman tree visualization."""

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

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
            return tool_response(
                entropy=round(e, 4),
                max_entropy=round(math.log2(max(len(set(text)), 1)), 4),
                length=len(text),
            )
        if op == "rle_encode":
            enc = _rle_encode(text)
            return tool_response(
                encoded=enc,
                original_len=len(text),
                encoded_len=len(enc),
                ratio=round(len(enc) / max(len(text), 1), 3),
            )
        if op == "rle_decode":
            return tool_response(decoded=_rle_decode(text))
        if op == "huffman":
            codes, tree = _huffman_tree(text)
            total_bits = sum(len(codes[ch]) * cnt for ch, cnt in Counter(text).items())
            return tool_response(
                codes=codes,
                tree="\n".join(tree),
                total_bits=total_bits,
                bits_per_char=round(total_bits / max(len(text), 1), 3),
            )
        if op == "analyze":
            e = _entropy(text)
            rle = _rle_encode(text)
            codes, _ = _huffman_tree(text)
            huff_bits = sum(len(codes[ch]) * cnt for ch, cnt in Counter(text).items())
            return tool_response(
                length=len(text),
                unique_chars=len(set(text)),
                entropy=round(e, 4),
                rle_ratio=round(len(rle) / max(len(text), 1), 3),
                huffman_bits=huff_bits,
                huffman_bpc=round(huff_bits / max(len(text), 1), 3),
                naive_bits=len(text) * 8,
            )
        return tool_error(f"Unknown op: {op}. Use entropy/rle_encode/rle_decode/huffman/analyze")
    except Exception as e:
        return tool_error(str(e))


__all__ = ["compression_tool"]
