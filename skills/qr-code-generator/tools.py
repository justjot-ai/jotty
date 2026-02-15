"""QR Code Generator Skill — generate QR codes as SVG or ASCII (pure Python)."""
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("qr-code-generator")

# Minimal QR Code encoder for alphanumeric/byte mode, version 1-4
# Uses mode indicator + character count + data + error correction

# Pre-computed generator polynomials and GF(256) tables for Reed-Solomon
GF_EXP = [0] * 512
GF_LOG = [0] * 256

def _init_gf():
    x = 1
    for i in range(255):
        GF_EXP[i] = x
        GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11d
    for i in range(255, 512):
        GF_EXP[i] = GF_EXP[i - 255]

_init_gf()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def _rs_encode(data: List[int], nsym: int) -> List[int]:
    gen = [1]
    for i in range(nsym):
        ng = [0] * (len(gen) + 1)
        for j, g in enumerate(gen):
            ng[j] ^= g
            ng[j + 1] ^= _gf_mul(g, GF_EXP[i])
        gen = ng

    remainder = [0] * (len(data) + nsym)
    remainder[:len(data)] = data
    for i in range(len(data)):
        coef = remainder[i]
        if coef != 0:
            for j in range(1, len(gen)):
                remainder[i + j] ^= _gf_mul(gen[j], coef)
    return remainder[len(data):]


def _encode_data_bits(text: str) -> tuple:
    """Encode data to bit string, return (bits, version, ec_codewords)."""
    data_bytes = text.encode("utf-8")
    byte_len = len(data_bytes)

    # Version capacity (byte mode, L error correction): (version, data_cap, ec_codewords, total_codewords)
    versions = [
        (1, 17, 7, 26), (2, 32, 10, 44), (3, 53, 15, 70), (4, 78, 20, 100),
        (5, 106, 26, 134), (6, 134, 36, 172),
    ]
    version = ec_cw = total_cw = None
    for v, cap, ec, tot in versions:
        if byte_len <= cap:
            version = v
            ec_cw = ec
            total_cw = tot
            break
    if version is None:
        return None, 0, 0

    data_cw = total_cw - ec_cw

    # Mode indicator (0100 = byte mode) + char count
    bits = "0100"
    count_bits = 8 if version <= 9 else 16
    bits += format(byte_len, f"0{count_bits}b")
    for b in data_bytes:
        bits += format(b, "08b")

    # Terminator
    bits += "0000"
    while len(bits) % 8 != 0:
        bits += "0"

    # Pad codewords
    pads = ["11101100", "00010001"]
    idx = 0
    while len(bits) < data_cw * 8:
        bits += pads[idx % 2]
        idx += 1

    bits = bits[:data_cw * 8]
    codewords = [int(bits[i:i+8], 2) for i in range(0, len(bits), 8)]

    # Reed-Solomon error correction
    ec_bytes = _rs_encode(codewords, ec_cw)
    all_cw = codewords + ec_bytes

    return all_cw, version, ec_cw


def _place_modules(codewords: List[int], version: int) -> List[List[int]]:
    """Place modules in QR matrix. Returns 2D grid (-1=unset, 0=white, 1=black)."""
    size = 17 + version * 4
    grid = [[-1] * size for _ in range(size)]
    reserved = [[False] * size for _ in range(size)]

    def set_mod(r, c, val):
        if 0 <= r < size and 0 <= c < size:
            grid[r][c] = val
            reserved[r][c] = True

    # Finder patterns
    for (cr, cc) in [(0, 0), (0, size - 7), (size - 7, 0)]:
        for r in range(7):
            for c in range(7):
                if (r in (0, 6) or c in (0, 6) or (2 <= r <= 4 and 2 <= c <= 4)):
                    set_mod(cr + r, cc + c, 1)
                else:
                    set_mod(cr + r, cc + c, 0)

    # Separators
    for i in range(8):
        for (cr, cc) in [(7, i), (i, 7), (7, size - 8 + i), (i, size - 8),
                          (size - 8, i), (size - 8 + i, 7)]:
            if 0 <= cr < size and 0 <= cc < size:
                set_mod(cr, cc, 0)

    # Timing patterns
    for i in range(8, size - 8):
        val = 1 if i % 2 == 0 else 0
        if not reserved[6][i]:
            set_mod(6, i, val)
        if not reserved[i][6]:
            set_mod(i, 6, val)

    # Dark module
    set_mod(size - 8, 8, 1)

    # Reserve format info areas
    for i in range(9):
        if not reserved[8][i]:
            reserved[8][i] = True
            grid[8][i] = 0
        if not reserved[i][8]:
            reserved[i][8] = True
            grid[i][8] = 0
        if i < 8:
            if not reserved[8][size - 1 - i]:
                reserved[8][size - 1 - i] = True
                grid[8][size - 1 - i] = 0
            if not reserved[size - 1 - i][8]:
                reserved[size - 1 - i][8] = True
                grid[size - 1 - i][8] = 0

    # Alignment patterns (version 2+)
    if version >= 2:
        positions = {2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34]}
        if version in positions:
            for ar in positions[version]:
                for ac in positions[version]:
                    if reserved[ar][ac]:
                        continue
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            val = 1 if (abs(dr) == 2 or abs(dc) == 2 or (dr == 0 and dc == 0)) else 0
                            set_mod(ar + dr, ac + dc, val)

    # Place data bits
    all_bits = []
    for cw in codewords:
        all_bits.extend([(cw >> (7 - i)) & 1 for i in range(8)])

    bit_idx = 0
    col = size - 1
    going_up = True
    while col >= 0:
        if col == 6:
            col -= 1
            continue
        rows = range(size - 1, -1, -1) if going_up else range(size)
        for row in rows:
            for dc in [0, -1]:
                c = col + dc
                if 0 <= c < size and not reserved[row][c]:
                    if bit_idx < len(all_bits):
                        grid[row][c] = all_bits[bit_idx]
                        bit_idx += 1
                    else:
                        grid[row][c] = 0
        col -= 2
        going_up = not going_up

    # Apply mask 0 (checkerboard) and format info
    for r in range(size):
        for c in range(size):
            if not reserved[r][c] and grid[r][c] != -1:
                if (r + c) % 2 == 0:
                    grid[r][c] ^= 1

    # Write format info for mask 0, EC level L
    fmt_bits = "111011111000100"
    positions_h = [(8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
                   (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8)]
    positions_v = [(size-1, 8), (size-2, 8), (size-3, 8), (size-4, 8), (size-5, 8),
                   (size-6, 8), (size-7, 8), (8, size-8), (8, size-7), (8, size-6),
                   (8, size-5), (8, size-4), (8, size-3), (8, size-2), (8, size-1)]
    for i, bit in enumerate(fmt_bits):
        val = int(bit)
        r, c = positions_h[i]
        grid[r][c] = val
        r, c = positions_v[i]
        grid[r][c] = val

    return grid


def _grid_to_svg(grid: list, module_size: int = 10) -> str:
    size = len(grid)
    total = size * module_size + module_size * 8
    margin = module_size * 4
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" width="{total}" height="{total}">']
    parts.append(f'<rect width="{total}" height="{total}" fill="white"/>')
    for r in range(size):
        for c in range(size):
            if grid[r][c] == 1:
                x = margin + c * module_size
                y = margin + r * module_size
                parts.append(f'<rect x="{x}" y="{y}" width="{module_size}" height="{module_size}" fill="black"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _grid_to_ascii(grid: list) -> str:
    lines = []
    for row in grid:
        line = ""
        for cell in row:
            line += "##" if cell == 1 else "  "
        lines.append(line)
    return "\n".join(lines)


@tool_wrapper(required_params=["data"])
def generate_qr_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a QR code from text or URL."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    fmt = params.get("format", "svg").lower()
    module_size = int(params.get("size", 10))

    if len(data) > 100:
        return tool_error("Data too long. Maximum 100 characters for built-in encoder.")

    codewords, version, ec_cw = _encode_data_bits(data)
    if codewords is None:
        return tool_error("Data too long for QR code generation")

    grid = _place_modules(codewords, version)

    if fmt == "ascii":
        qr_out = _grid_to_ascii(grid)
    else:
        qr_out = _grid_to_svg(grid, module_size)

    return tool_response(qr_code=qr_out, format=fmt, version=version,
                         size=len(grid), data_length=len(data))


__all__ = ["generate_qr_tool"]
