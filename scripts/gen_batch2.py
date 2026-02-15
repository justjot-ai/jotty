"""Batch 2: Skills 21-40 (networking, data, devops, generators)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from generate_skills import create_skill

# ── 21. ssl-certificate-checker ───────────────────────────────────
create_skill(
    name="ssl-certificate-checker",
    frontmatter_name="checking-ssl-certificates",
    description="Check SSL certificate validity, expiry date, issuer, and chain. Use when the user wants to check SSL, certificate expiry, HTTPS cert.",
    category="development",
    capabilities=["data-fetch", "devops"],
    triggers=["ssl", "certificate", "cert expiry", "https check", "tls"],
    eval_tool="check_ssl_tool",
    eval_input={"hostname": "example.com"},
    tool_docs="""### check_ssl_tool
Check SSL certificate for a hostname.

**Parameters:**
- `hostname` (str, required): Domain to check
- `port` (int, optional): Port (default: 443)

**Returns:**
- `success` (bool)
- `subject` (str): Certificate subject
- `issuer` (str): Certificate issuer
- `expires` (str): Expiry date
- `days_remaining` (int): Days until expiry
- `valid` (bool): Whether cert is currently valid""",
    tools_code='''"""SSL Certificate Checker Skill — check cert validity and expiry."""
import ssl
import socket
from datetime import datetime, timezone
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("ssl-certificate-checker")


@tool_wrapper(required_params=["hostname"])
def check_ssl_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check SSL certificate for a hostname."""
    status.set_callback(params.pop("_status_callback", None))
    hostname = params["hostname"].strip().lower()
    port = int(params.get("port", 443))

    # Strip protocol if provided
    for prefix in ("https://", "http://"):
        if hostname.startswith(prefix):
            hostname = hostname[len(prefix):]
    hostname = hostname.rstrip("/").split("/")[0]

    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.settimeout(10)
            s.connect((hostname, port))
            cert = s.getpeercert()

        if not cert:
            return tool_error("No certificate returned")

        subject = dict(x[0] for x in cert.get("subject", ()))
        issuer = dict(x[0] for x in cert.get("issuer", ()))
        not_after = cert.get("notAfter", "")
        not_before = cert.get("notBefore", "")

        # Parse expiry date
        expire_dt = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_remaining = (expire_dt - now).days
        is_valid = days_remaining > 0

        san_list = []
        for entry_type, value in cert.get("subjectAltName", ()):
            san_list.append(value)

        return tool_response(
            hostname=hostname,
            subject=subject.get("commonName", ""),
            issuer=issuer.get("organizationName", issuer.get("commonName", "")),
            not_before=not_before,
            expires=not_after,
            days_remaining=days_remaining,
            valid=is_valid,
            serial_number=cert.get("serialNumber", ""),
            version=cert.get("version", ""),
            san=san_list[:20],
        )
    except ssl.SSLCertVerificationError as e:
        return tool_error(f"SSL verification failed: {e}")
    except socket.gaierror:
        return tool_error(f"Cannot resolve hostname: {hostname}")
    except socket.timeout:
        return tool_error(f"Connection timed out: {hostname}:{port}")
    except Exception as e:
        return tool_error(f"SSL check failed: {e}")


__all__ = ["check_ssl_tool"]
''')

# ── 22. uptime-monitor ────────────────────────────────────────────
create_skill(
    name="uptime-monitor",
    frontmatter_name="monitoring-uptime",
    description="Check HTTP endpoint availability, response time, status codes. Use when the user wants to check uptime, ping website, monitor endpoint.",
    category="development",
    capabilities=["data-fetch", "devops"],
    triggers=["uptime", "ping", "health check", "endpoint status", "website up"],
    eval_tool="check_endpoint_tool",
    eval_input={"url": "https://example.com"},
    tool_docs="""### check_endpoint_tool
Check HTTP endpoint availability.

**Parameters:**
- `url` (str, required): URL to check
- `method` (str, optional): HTTP method (default: GET)
- `timeout` (int, optional): Timeout seconds (default: 10)
- `expected_status` (int, optional): Expected HTTP status (default: 200)

**Returns:**
- `success` (bool)
- `status_code` (int): HTTP status code
- `response_time_ms` (float): Response time in milliseconds
- `available` (bool): Whether endpoint is available""",
    tools_code='''"""Uptime Monitor Skill — check HTTP endpoint availability."""
import time
import requests
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("uptime-monitor")


@tool_wrapper(required_params=["url"])
def check_endpoint_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check HTTP endpoint availability and response time."""
    status.set_callback(params.pop("_status_callback", None))
    url = params["url"].strip()
    method = params.get("method", "GET").upper()
    timeout = min(int(params.get("timeout", 10)), 30)
    expected_status = int(params.get("expected_status", 200))

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    start = time.monotonic()
    try:
        resp = requests.request(method, url, timeout=timeout, allow_redirects=True)
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)

        available = resp.status_code == expected_status
        headers_dict = {k: v for k, v in list(resp.headers.items())[:20]}

        return tool_response(
            url=url,
            status_code=resp.status_code,
            response_time_ms=elapsed_ms,
            available=available,
            content_length=len(resp.content),
            headers=headers_dict,
            redirect_url=resp.url if resp.url != url else None,
        )
    except requests.ConnectionError:
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        return tool_response(url=url, status_code=0, response_time_ms=elapsed_ms,
                             available=False, error="Connection refused")
    except requests.Timeout:
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        return tool_response(url=url, status_code=0, response_time_ms=elapsed_ms,
                             available=False, error="Timeout")
    except requests.RequestException as e:
        return tool_error(f"Request failed: {e}")


@tool_wrapper(required_params=["urls"])
def check_multiple_endpoints_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check multiple endpoints at once."""
    status.set_callback(params.pop("_status_callback", None))
    urls = params["urls"]
    if not isinstance(urls, list):
        return tool_error("urls must be a list")

    results = []
    for url in urls[:20]:
        result = check_endpoint_tool({"url": url, "timeout": params.get("timeout", 10)})
        results.append(result)

    available_count = sum(1 for r in results if r.get("available"))
    return tool_response(results=results, total=len(results),
                         available=available_count, unavailable=len(results) - available_count)


__all__ = ["check_endpoint_tool", "check_multiple_endpoints_tool"]
''')

# ── 23. webhook-dispatcher ────────────────────────────────────────
create_skill(
    name="webhook-dispatcher",
    frontmatter_name="dispatching-webhooks",
    description="Send HTTP webhook payloads with retry logic and authentication. Use when the user wants to send webhook, POST payload, HTTP callback.",
    category="development",
    capabilities=["data-fetch", "code"],
    triggers=["webhook", "send webhook", "POST", "callback", "http post"],
    eval_tool="send_webhook_tool",
    eval_input={"url": "https://httpbin.org/post", "payload": {"event": "test"}},
    tool_docs="""### send_webhook_tool
Send an HTTP webhook with retry logic.

**Parameters:**
- `url` (str, required): Webhook URL
- `payload` (dict, required): JSON payload
- `method` (str, optional): HTTP method (default: POST)
- `headers` (dict, optional): Custom headers
- `retries` (int, optional): Max retries (default: 3)
- `timeout` (int, optional): Timeout seconds (default: 10)

**Returns:**
- `success` (bool)
- `status_code` (int): HTTP status code
- `attempts` (int): Number of attempts made""",
    tools_code='''"""Webhook Dispatcher Skill — send webhooks with retry logic."""
import time
import json
import hashlib
import hmac
import requests
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("webhook-dispatcher")


@tool_wrapper(required_params=["url", "payload"])
def send_webhook_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Send HTTP webhook payload with retry logic."""
    status.set_callback(params.pop("_status_callback", None))
    url = params["url"]
    payload = params["payload"]
    method = params.get("method", "POST").upper()
    custom_headers = params.get("headers", {})
    max_retries = min(int(params.get("retries", 3)), 5)
    timeout = min(int(params.get("timeout", 10)), 30)
    secret = params.get("secret")

    if not url.startswith(("http://", "https://")):
        return tool_error("URL must start with http:// or https://")

    headers = {"Content-Type": "application/json", "User-Agent": "Jotty-Webhook/1.0"}
    headers.update(custom_headers)

    body = json.dumps(payload)

    if secret:
        sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={sig}"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(method, url, data=body, headers=headers, timeout=timeout)
            if resp.status_code < 500:
                return tool_response(
                    status_code=resp.status_code,
                    attempts=attempt,
                    response_body=resp.text[:1000],
                    url=url,
                    delivered=200 <= resp.status_code < 300,
                )
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))

    return tool_response(
        status_code=0,
        attempts=max_retries,
        delivered=False,
        error=f"All {max_retries} attempts failed. Last error: {last_error}",
        url=url,
    )


__all__ = ["send_webhook_tool"]
''')

print(f"\nBatch 2 progress: skills 21-23 created.")

# ── 24. currency-converter ────────────────────────────────────────
create_skill(
    name="currency-converter",
    frontmatter_name="converting-currency",
    description="Convert currencies using live exchange rates from frankfurter.app. Use when the user wants to convert currency, exchange rate, USD to EUR.",
    category="data-analysis",
    capabilities=["data-fetch"],
    triggers=["currency", "exchange rate", "convert currency", "USD", "EUR", "forex"],
    eval_tool="convert_currency_tool",
    eval_input={"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
    tool_docs="""### convert_currency_tool
Convert between currencies using live rates.

**Parameters:**
- `amount` (float, required): Amount to convert
- `from_currency` (str, required): Source currency code (e.g. USD)
- `to_currency` (str, required): Target currency code (e.g. EUR)

**Returns:**
- `success` (bool)
- `converted` (float): Converted amount
- `rate` (float): Exchange rate used
- `from_currency` (str): Source currency
- `to_currency` (str): Target currency""",
    tools_code='''"""Currency Converter Skill — convert currencies using frankfurter.app."""
import requests
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("currency-converter")


@tool_wrapper(required_params=["amount", "from_currency", "to_currency"])
def convert_currency_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert between currencies using live exchange rates."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        amount = float(params["amount"])
    except (ValueError, TypeError):
        return tool_error("amount must be a number")

    from_cur = params["from_currency"].upper().strip()
    to_cur = params["to_currency"].upper().strip()

    if from_cur == to_cur:
        return tool_response(converted=amount, rate=1.0,
                             from_currency=from_cur, to_currency=to_cur, amount=amount)

    try:
        resp = requests.get(
            f"https://api.frankfurter.app/latest",
            params={"amount": amount, "from": from_cur, "to": to_cur},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if "rates" not in data or to_cur not in data["rates"]:
            return tool_error(f"Could not get rate for {from_cur} to {to_cur}")

        converted = data["rates"][to_cur]
        rate = converted / amount if amount != 0 else 0

        return tool_response(
            converted=round(converted, 4),
            rate=round(rate, 6),
            from_currency=from_cur,
            to_currency=to_cur,
            amount=amount,
            date=data.get("date", ""),
        )
    except requests.RequestException as e:
        return tool_error(f"Currency conversion failed: {e}")


@tool_wrapper(required_params=["base"])
def list_rates_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """List all exchange rates for a base currency."""
    status.set_callback(params.pop("_status_callback", None))
    base = params["base"].upper().strip()
    try:
        resp = requests.get(
            f"https://api.frankfurter.app/latest",
            params={"from": base},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return tool_response(base=base, rates=data.get("rates", {}),
                             date=data.get("date", ""))
    except requests.RequestException as e:
        return tool_error(f"Failed to fetch rates: {e}")


__all__ = ["convert_currency_tool", "list_rates_tool"]
''')

# ── 25. crypto-price-tracker ──────────────────────────────────────
create_skill(
    name="crypto-price-tracker",
    frontmatter_name="tracking-crypto-prices",
    description="Fetch cryptocurrency prices from CoinGecko free API. Use when the user wants to check crypto price, bitcoin price, ethereum price.",
    category="data-analysis",
    capabilities=["data-fetch"],
    triggers=["crypto", "bitcoin", "ethereum", "crypto price", "BTC", "ETH", "cryptocurrency"],
    eval_tool="crypto_price_tool",
    eval_input={"coin": "bitcoin"},
    tool_docs="""### crypto_price_tool
Get current cryptocurrency price.

**Parameters:**
- `coin` (str, required): Coin ID or symbol (bitcoin, ethereum, BTC, ETH)
- `currency` (str, optional): Fiat currency (default: usd)

**Returns:**
- `success` (bool)
- `coin` (str): Coin name
- `price` (float): Current price
- `change_24h` (float): 24h price change percentage""",
    tools_code='''"""Crypto Price Tracker Skill — fetch prices from CoinGecko."""
import requests
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("crypto-price-tracker")

SYMBOL_MAP = {
    "btc": "bitcoin", "eth": "ethereum", "bnb": "binancecoin",
    "xrp": "ripple", "ada": "cardano", "sol": "solana",
    "dot": "polkadot", "doge": "dogecoin", "avax": "avalanche-2",
    "matic": "matic-network", "ltc": "litecoin", "link": "chainlink",
    "uni": "uniswap", "atom": "cosmos", "xlm": "stellar",
}


@tool_wrapper(required_params=["coin"])
def crypto_price_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get current cryptocurrency price from CoinGecko."""
    status.set_callback(params.pop("_status_callback", None))
    coin = params["coin"].strip().lower()
    currency = params.get("currency", "usd").strip().lower()

    coin_id = SYMBOL_MAP.get(coin, coin)

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": currency,
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
            },
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        if coin_id not in data:
            return tool_error(f"Coin not found: {coin}. Use full name (e.g. bitcoin) or symbol (e.g. btc)")

        coin_data = data[coin_id]
        return tool_response(
            coin=coin_id,
            symbol=coin,
            currency=currency,
            price=coin_data.get(currency),
            change_24h=coin_data.get(f"{currency}_24h_change"),
            market_cap=coin_data.get(f"{currency}_market_cap"),
            volume_24h=coin_data.get(f"{currency}_24h_vol"),
        )
    except requests.RequestException as e:
        return tool_error(f"Failed to fetch crypto price: {e}")


@tool_wrapper()
def crypto_top_coins_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get top cryptocurrencies by market cap."""
    status.set_callback(params.pop("_status_callback", None))
    currency = params.get("currency", "usd").lower()
    limit = min(int(params.get("limit", 10)), 50)

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": currency, "order": "market_cap_desc",
                    "per_page": limit, "page": 1},
            timeout=10,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        coins = []
        for c in resp.json():
            coins.append({
                "name": c.get("name"), "symbol": c.get("symbol"),
                "price": c.get("current_price"),
                "market_cap": c.get("market_cap"),
                "change_24h": c.get("price_change_percentage_24h"),
            })
        return tool_response(coins=coins, currency=currency, count=len(coins))
    except requests.RequestException as e:
        return tool_error(f"Failed to fetch top coins: {e}")


__all__ = ["crypto_price_tool", "crypto_top_coins_tool"]
''')

# ── 26. qr-code-generator ────────────────────────────────────────
create_skill(
    name="qr-code-generator",
    frontmatter_name="generating-qr-codes",
    description="Generate QR codes as SVG or ASCII art from text or URLs. Pure Python, no external deps. Use when the user wants to generate QR code, create QR.",
    category="content-creation",
    capabilities=["generate"],
    triggers=["qr", "qr code", "generate qr", "barcode"],
    eval_tool="generate_qr_tool",
    eval_input={"data": "https://example.com"},
    tool_docs="""### generate_qr_tool
Generate a QR code from text or URL.

**Parameters:**
- `data` (str, required): Text or URL to encode
- `format` (str, optional): Output format: svg, ascii (default: svg)
- `size` (int, optional): Module size in pixels for SVG (default: 10)

**Returns:**
- `success` (bool)
- `qr_code` (str): QR code as SVG string or ASCII art
- `format` (str): Output format used""",
    tools_code='''"""QR Code Generator Skill — generate QR codes as SVG or ASCII (pure Python)."""
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
    return "\\n".join(parts)


def _grid_to_ascii(grid: list) -> str:
    lines = []
    for row in grid:
        line = ""
        for cell in row:
            line += "##" if cell == 1 else "  "
        lines.append(line)
    return "\\n".join(lines)


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
''')

print(f"Batch 2 progress: skills 24-26 created.")

# ── 27. chart-generator ───────────────────────────────────────────
create_skill(
    name="chart-generator",
    frontmatter_name="generating-charts",
    description="Generate ASCII bar and line charts from data. Pure Python. Use when the user wants to create chart, bar chart, line chart, visualize data.",
    category="data-analysis",
    capabilities=["generate", "analyze"],
    triggers=["chart", "bar chart", "line chart", "ascii chart", "visualize"],
    eval_tool="bar_chart_tool",
    eval_input={"data": {"Python": 35, "JavaScript": 28, "Go": 15, "Rust": 12}},
    tool_docs="""### bar_chart_tool
Generate ASCII horizontal bar chart.

**Parameters:**
- `data` (dict, required): Label-value pairs
- `width` (int, optional): Chart width in chars (default: 50)
- `sort` (bool, optional): Sort by value descending (default: true)

**Returns:**
- `success` (bool)
- `chart` (str): ASCII bar chart""",
    tools_code='''"""Chart Generator Skill — ASCII bar and line charts (pure Python)."""
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("chart-generator")

BLOCK_CHARS = " ▏▎▍▌▋▊▉█"


@tool_wrapper(required_params=["data"])
def bar_chart_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ASCII horizontal bar chart."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    width = int(params.get("width", 50))
    sort_desc = params.get("sort", True)

    if not isinstance(data, dict) or not data:
        return tool_error("data must be a non-empty dict of label: value pairs")

    items = list(data.items())
    if sort_desc:
        items.sort(key=lambda x: float(x[1]), reverse=True)

    max_val = max(float(v) for _, v in items)
    max_label = max(len(str(k)) for k, _ in items)

    lines = []
    for label, value in items:
        val = float(value)
        if max_val > 0:
            filled = val / max_val * width
            full_blocks = int(filled)
            fraction = filled - full_blocks
            frac_idx = int(fraction * 8)
            bar = "█" * full_blocks
            if frac_idx > 0 and full_blocks < width:
                bar += BLOCK_CHARS[frac_idx]
        else:
            bar = ""
        lines.append(f"{str(label):>{max_label}} │ {bar} {val:g}")

    chart = "\\n".join(lines)
    return tool_response(chart=chart, items=len(items))


@tool_wrapper(required_params=["data"])
def line_chart_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ASCII line chart from sequential data."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    height = int(params.get("height", 15))
    width = int(params.get("width", 60))

    if isinstance(data, dict):
        labels = list(data.keys())
        values = [float(v) for v in data.values()]
    elif isinstance(data, list):
        values = [float(v) for v in data]
        labels = [str(i) for i in range(len(values))]
    else:
        return tool_error("data must be a list of numbers or dict of label: value")

    if not values:
        return tool_error("No data points provided")

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1

    # Build grid
    grid = [[" "] * len(values) for _ in range(height)]
    for i, v in enumerate(values):
        row = height - 1 - int((v - min_val) / val_range * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][i] = "●"
        # Draw line to next point
        if i < len(values) - 1:
            next_row = height - 1 - int((values[i+1] - min_val) / val_range * (height - 1))
            next_row = max(0, min(height - 1, next_row))
            if next_row != row:
                step = 1 if next_row > row else -1
                for r in range(row + step, next_row, step):
                    if grid[r][i] == " ":
                        grid[r][i] = "│"

    # Render
    lines = []
    for r in range(height):
        y_val = max_val - r * val_range / (height - 1) if height > 1 else max_val
        row_str = "".join(grid[r])
        lines.append(f"{y_val:>8.1f} ┤ {row_str}")
    lines.append(" " * 10 + "└" + "─" * len(values))

    chart = "\\n".join(lines)
    return tool_response(chart=chart, points=len(values),
                         min_value=min_val, max_value=max_val)


__all__ = ["bar_chart_tool", "line_chart_tool"]
''')

# ── 28. data-schema-inferrer ──────────────────────────────────────
create_skill(
    name="data-schema-inferrer",
    frontmatter_name="inferring-data-schemas",
    description="Infer JSON Schema from sample data. Pure Python. Use when the user wants to generate schema, infer types, JSON schema from data.",
    category="data-analysis",
    capabilities=["analyze", "code"],
    triggers=["schema", "json schema", "infer schema", "data types", "type inference"],
    eval_tool="infer_schema_tool",
    eval_input={"data": {"name": "Alice", "age": 30, "active": True, "tags": ["a", "b"]}},
    tool_docs="""### infer_schema_tool
Infer JSON Schema from sample data.

**Parameters:**
- `data` (any, required): Sample JSON data (object, array, or primitive)
- `title` (str, optional): Schema title

**Returns:**
- `success` (bool)
- `schema` (dict): Inferred JSON Schema (draft-07)""",
    tools_code='''"""Data Schema Inferrer Skill — infer JSON Schema from data (pure Python)."""
import json
from typing import Dict, Any, List, Optional

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("data-schema-inferrer")


def _infer_type(value: Any) -> Dict[str, Any]:
    """Infer JSON Schema for a single value."""
    if value is None:
        return {"type": "null"}
    elif isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, int):
        return {"type": "integer"}
    elif isinstance(value, float):
        return {"type": "number"}
    elif isinstance(value, str):
        schema = {"type": "string"}
        if len(value) > 0:
            # Detect common formats
            import re
            if re.match(r"^\\d{4}-\\d{2}-\\d{2}$", value):
                schema["format"] = "date"
            elif re.match(r"^\\d{4}-\\d{2}-\\d{2}T", value):
                schema["format"] = "date-time"
            elif re.match(r"^[^@]+@[^@]+\\.[^@]+$", value):
                schema["format"] = "email"
            elif re.match(r"^https?://", value):
                schema["format"] = "uri"
            elif re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value, re.I):
                schema["format"] = "uuid"
        return schema
    elif isinstance(value, list):
        if not value:
            return {"type": "array", "items": {}}
        item_schemas = [_infer_type(item) for item in value]
        # If all items same type, use single schema
        types = set(json.dumps(s, sort_keys=True) for s in item_schemas)
        if len(types) == 1:
            return {"type": "array", "items": item_schemas[0]}
        else:
            return {"type": "array", "items": {"oneOf": item_schemas}}
    elif isinstance(value, dict):
        properties = {}
        required = []
        for k, v in value.items():
            properties[k] = _infer_type(v)
            if v is not None:
                required.append(k)
        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = sorted(required)
        return schema
    else:
        return {"type": "string"}


def _merge_schemas(samples: List[Any]) -> Dict[str, Any]:
    """Merge schemas from multiple samples."""
    if not samples:
        return {}
    if len(samples) == 1:
        return _infer_type(samples[0])

    # For array of objects, merge property sets
    all_props = {}
    required_counts = {}
    total = len(samples)

    for sample in samples:
        schema = _infer_type(sample)
        if schema.get("type") == "object":
            for prop, prop_schema in schema.get("properties", {}).items():
                if prop not in all_props:
                    all_props[prop] = prop_schema
                required_counts[prop] = required_counts.get(prop, 0) + 1

    required = [k for k, count in required_counts.items() if count == total]
    result = {"type": "object", "properties": all_props}
    if required:
        result["required"] = sorted(required)
    return result


@tool_wrapper(required_params=["data"])
def infer_schema_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Infer JSON Schema from sample data."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    title = params.get("title", "Inferred Schema")

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid JSON: {e}")

    if isinstance(data, list) and data and isinstance(data[0], dict):
        schema = _merge_schemas(data)
    else:
        schema = _infer_type(data)

    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = title

    return tool_response(schema=schema)


@tool_wrapper(required_params=["data", "schema"])
def validate_against_schema_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data against a JSON Schema (basic validation)."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    schema = params["schema"]
    errors = []

    def _validate(value, sch, path="$"):
        expected_type = sch.get("type")
        type_map = {"string": str, "integer": int, "number": (int, float),
                    "boolean": bool, "array": list, "object": dict, "null": type(None)}
        if expected_type and expected_type in type_map:
            if not isinstance(value, type_map[expected_type]):
                errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
                return
        if expected_type == "object" and isinstance(value, dict):
            for req in sch.get("required", []):
                if req not in value:
                    errors.append(f"{path}: missing required property '{req}'")
            for prop, prop_schema in sch.get("properties", {}).items():
                if prop in value:
                    _validate(value[prop], prop_schema, f"{path}.{prop}")
        elif expected_type == "array" and isinstance(value, list):
            item_schema = sch.get("items", {})
            for i, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{i}]")

    _validate(data, schema)
    return tool_response(valid=len(errors) == 0, errors=errors, error_count=len(errors))


__all__ = ["infer_schema_tool", "validate_against_schema_tool"]
''')

# ── 29. ab-test-analyzer ─────────────────────────────────────────
create_skill(
    name="ab-test-analyzer",
    frontmatter_name="analyzing-ab-tests",
    description="Calculate statistical significance, p-values, and confidence intervals for A/B tests. Use when the user wants to analyze A/B test, p-value, significance.",
    category="data-analysis",
    capabilities=["analyze"],
    triggers=["a/b test", "p-value", "significance", "conversion rate", "hypothesis test"],
    eval_tool="ab_test_tool",
    eval_input={"visitors_a": 1000, "conversions_a": 50, "visitors_b": 1000, "conversions_b": 65},
    tool_docs="""### ab_test_tool
Analyze A/B test results for statistical significance.

**Parameters:**
- `visitors_a` (int, required): Visitors in control group
- `conversions_a` (int, required): Conversions in control group
- `visitors_b` (int, required): Visitors in test group
- `conversions_b` (int, required): Conversions in test group
- `confidence_level` (float, optional): Confidence level (default: 0.95)

**Returns:**
- `success` (bool)
- `significant` (bool): Whether result is statistically significant
- `p_value` (float): Two-tailed p-value
- `lift` (float): Relative improvement percentage""",
    tools_code='''"""A/B Test Analyzer Skill — calculate significance using stdlib math."""
import math
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("ab-test-analyzer")


def _normal_cdf(x: float) -> float:
    """Approximate the cumulative distribution function of the standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _z_to_p(z: float) -> float:
    """Convert z-score to two-tailed p-value."""
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


def _confidence_to_z(confidence: float) -> float:
    """Get z critical value for confidence level."""
    alpha = 1.0 - confidence
    # Newton's method approximation for inverse normal
    # Common values
    z_table = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576, 0.999: 3.291}
    if confidence in z_table:
        return z_table[confidence]
    # Approximation for other values
    p = alpha / 2.0
    t = math.sqrt(-2.0 * math.log(p))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)


@tool_wrapper(required_params=["visitors_a", "conversions_a", "visitors_b", "conversions_b"])
def ab_test_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze A/B test for statistical significance."""
    status.set_callback(params.pop("_status_callback", None))
    n_a = int(params["visitors_a"])
    c_a = int(params["conversions_a"])
    n_b = int(params["visitors_b"])
    c_b = int(params["conversions_b"])
    confidence = float(params.get("confidence_level", 0.95))

    if n_a <= 0 or n_b <= 0:
        return tool_error("Visitor counts must be positive")
    if c_a < 0 or c_b < 0:
        return tool_error("Conversion counts must be non-negative")
    if c_a > n_a or c_b > n_b:
        return tool_error("Conversions cannot exceed visitors")

    rate_a = c_a / n_a
    rate_b = c_b / n_b

    # Pooled proportion
    p_pool = (c_a + c_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) if p_pool > 0 and p_pool < 1 else 0.0001

    z_score = (rate_b - rate_a) / se if se > 0 else 0
    p_value = _z_to_p(z_score)
    z_crit = _confidence_to_z(confidence)
    significant = abs(z_score) > z_crit

    # Confidence interval for difference
    se_diff = math.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    diff = rate_b - rate_a
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    lift = ((rate_b - rate_a) / rate_a * 100) if rate_a > 0 else 0

    return tool_response(
        significant=significant,
        p_value=round(p_value, 6),
        z_score=round(z_score, 4),
        rate_a=round(rate_a, 6),
        rate_b=round(rate_b, 6),
        lift=round(lift, 2),
        confidence_interval={"lower": round(ci_lower, 6), "upper": round(ci_upper, 6)},
        confidence_level=confidence,
        recommendation="B is better" if significant and z_score > 0 else "A is better" if significant and z_score < 0 else "No significant difference",
    )


@tool_wrapper(required_params=["baseline_rate", "minimum_effect"])
def sample_size_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate required sample size for an A/B test."""
    status.set_callback(params.pop("_status_callback", None))
    p1 = float(params["baseline_rate"])
    mde = float(params["minimum_effect"])  # relative effect
    confidence = float(params.get("confidence_level", 0.95))
    power = float(params.get("power", 0.80))

    if not (0 < p1 < 1):
        return tool_error("baseline_rate must be between 0 and 1")

    p2 = p1 * (1 + mde)
    if not (0 < p2 < 1):
        return tool_error("minimum_effect produces invalid rate")

    z_alpha = _confidence_to_z(confidence)
    z_beta = _confidence_to_z(0.5 + power / 2)

    p_avg = (p1 + p2) / 2
    n = ((z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
          z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p2 - p1)) ** 2
    n = math.ceil(n)

    return tool_response(
        sample_size_per_group=n,
        total_sample_size=n * 2,
        baseline_rate=p1,
        expected_rate=round(p2, 6),
        minimum_effect=mde,
        confidence_level=confidence,
        power=power,
    )


__all__ = ["ab_test_tool", "sample_size_tool"]
''')

# ── 30. pivot-table-builder ──────────────────────────────────────
create_skill(
    name="pivot-table-builder",
    frontmatter_name="building-pivot-tables",
    description="Create pivot tables from data with row/column grouping and aggregation. Pure Python. Use when the user wants to pivot table, group by, aggregate data.",
    category="data-analysis",
    capabilities=["analyze"],
    triggers=["pivot", "pivot table", "group by", "aggregate", "crosstab"],
    eval_tool="pivot_table_tool",
    eval_input={"data": [{"region": "North", "product": "A", "sales": 100}, {"region": "North", "product": "B", "sales": 150}, {"region": "South", "product": "A", "sales": 200}], "rows": "region", "columns": "product", "values": "sales", "aggfunc": "sum"},
    tool_docs="""### pivot_table_tool
Create a pivot table from data.

**Parameters:**
- `data` (list, required): List of objects
- `rows` (str, required): Field for row grouping
- `columns` (str, optional): Field for column grouping
- `values` (str, required): Field to aggregate
- `aggfunc` (str, optional): sum, mean, count, min, max (default: sum)

**Returns:**
- `success` (bool)
- `table` (dict): Pivot table data
- `formatted` (str): ASCII formatted table""",
    tools_code='''"""Pivot Table Builder Skill — create pivot tables (pure Python)."""
import statistics
from typing import Dict, Any, List, Callable

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("pivot-table-builder")

AGG_FUNCS = {
    "sum": sum,
    "mean": lambda vals: statistics.mean(vals) if vals else 0,
    "count": len,
    "min": min,
    "max": max,
    "median": lambda vals: statistics.median(vals) if vals else 0,
}


@tool_wrapper(required_params=["data", "rows", "values"])
def pivot_table_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a pivot table from data."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    row_field = params["rows"]
    col_field = params.get("columns")
    val_field = params["values"]
    agg_name = params.get("aggfunc", "sum").lower()

    if not isinstance(data, list):
        return tool_error("data must be a list of objects")
    if agg_name not in AGG_FUNCS:
        return tool_error(f"aggfunc must be one of: {list(AGG_FUNCS.keys())}")

    agg_func = AGG_FUNCS[agg_name]

    # Collect values into buckets
    buckets: Dict[str, Dict[str, List[float]]] = {}
    col_keys = set()

    for record in data:
        if not isinstance(record, dict):
            continue
        row_key = str(record.get(row_field, ""))
        val = record.get(val_field)
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue

        if col_field:
            col_key = str(record.get(col_field, ""))
            col_keys.add(col_key)
        else:
            col_key = val_field

        if row_key not in buckets:
            buckets[row_key] = {}
        if col_key not in buckets[row_key]:
            buckets[row_key][col_key] = []
        buckets[row_key][col_key].append(val)

    # Aggregate
    table = {}
    for row_key, cols in sorted(buckets.items()):
        table[row_key] = {}
        for col_key, vals in sorted(cols.items()):
            result = agg_func(vals)
            table[row_key][col_key] = round(result, 2) if isinstance(result, float) else result

    # Format as ASCII table
    all_cols = sorted(col_keys) if col_field else [val_field]
    col_width = max(max((len(c) for c in all_cols), default=5), 10)
    row_width = max(max((len(r) for r in table.keys()), default=5), 10)

    header = f"{'':<{row_width}} | " + " | ".join(f"{c:>{col_width}}" for c in all_cols)
    sep = "-" * len(header)
    lines = [header, sep]
    for row_key in sorted(table.keys()):
        vals = " | ".join(
            f"{table[row_key].get(c, 0):>{col_width}}" if isinstance(table[row_key].get(c, 0), int)
            else f"{table[row_key].get(c, 0):>{col_width}.2f}"
            for c in all_cols
        )
        lines.append(f"{row_key:<{row_width}} | {vals}")

    formatted = "\\n".join(lines)
    return tool_response(table=table, formatted=formatted,
                         rows=len(table), columns=len(all_cols), aggfunc=agg_name)


__all__ = ["pivot_table_tool"]
''')

print(f"Batch 2 progress: skills 27-30 created.")

# ── 31. log-analyzer ──────────────────────────────────────────────
create_skill(
    name="log-analyzer",
    frontmatter_name="analyzing-logs",
    description="Parse and summarize log files, find errors, count patterns. Pure Python. Use when the user wants to analyze logs, find errors, parse log file.",
    category="development",
    capabilities=["analyze", "code"],
    triggers=["log", "analyze log", "log file", "error log", "parse logs"],
    eval_tool="analyze_log_tool",
    eval_input={"content": "2024-01-01 10:00:00 ERROR Database connection failed\\n2024-01-01 10:00:01 INFO Retrying...\\n2024-01-01 10:00:02 ERROR Database connection failed\\n2024-01-01 10:00:03 INFO Connected successfully"},
    tool_docs="""### analyze_log_tool
Parse and summarize log content.

**Parameters:**
- `content` (str, optional): Log text content
- `file_path` (str, optional): Path to log file
- `level_filter` (str, optional): Filter by level (ERROR, WARN, INFO, DEBUG)

**Returns:**
- `success` (bool)
- `summary` (dict): Level counts, error messages, time range
- `errors` (list): Error log lines""",
    tools_code='''"""Log Analyzer Skill — parse and summarize log files (pure Python)."""
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("log-analyzer")

LOG_PATTERNS = [
    # Standard: 2024-01-01 10:00:00 ERROR message
    re.compile(r"^(\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}:\\d{2}[^ ]*)\\s+(ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL|TRACE)\\s+(.+)$", re.IGNORECASE),
    # Bracketed: [2024-01-01 10:00:00] [ERROR] message
    re.compile(r"^\\[(\\d{4}-\\d{2}-\\d{2}[T ]\\d{2}:\\d{2}:\\d{2}[^\\]]*)\\]\\s*\\[(ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL)\\]\\s+(.+)$", re.IGNORECASE),
    # Syslog-ish: Jan  1 10:00:00 hostname service: message
    re.compile(r"^(\\w{3}\\s+\\d+\\s+\\d{2}:\\d{2}:\\d{2})\\s+\\S+\\s+(\\S+):\\s+(.+)$"),
    # Nginx/Apache access log
    re.compile(r'^(\\S+)\\s+\\S+\\s+\\S+\\s+\\[([^\\]]+)\\]\\s+"(\\S+)\\s+(\\S+)\\s+\\S+"\\s+(\\d{3})\\s+(\\d+)'),
]

ERROR_LEVELS = {"error", "fatal", "critical"}
WARN_LEVELS = {"warn", "warning"}


@tool_wrapper()
def analyze_log_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and summarize log content."""
    status.set_callback(params.pop("_status_callback", None))
    content = params.get("content", "")
    file_path = params.get("file_path")
    level_filter = params.get("level_filter", "").upper()

    if file_path:
        p = Path(file_path)
        if not p.exists():
            return tool_error(f"File not found: {file_path}")
        content = p.read_text(errors="replace")

    if not content:
        return tool_error("Provide either content or file_path")

    lines = content.splitlines()
    level_counts = Counter()
    errors = []
    warnings = []
    timestamps = []
    parsed_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        matched = False
        for pattern in LOG_PATTERNS[:2]:
            m = pattern.match(line)
            if m:
                ts, level, msg = m.group(1), m.group(2).upper(), m.group(3)
                timestamps.append(ts)
                level_norm = level.replace("WARNING", "WARN")
                level_counts[level_norm] += 1
                if level.lower() in ERROR_LEVELS:
                    errors.append({"timestamp": ts, "message": msg[:200]})
                elif level.lower() in WARN_LEVELS:
                    warnings.append({"timestamp": ts, "message": msg[:200]})
                parsed_count += 1
                matched = True
                break
        if not matched:
            level_counts["UNPARSED"] += 1

    # Apply filter
    filtered_errors = errors
    if level_filter == "ERROR":
        pass  # already filtered
    elif level_filter == "WARN":
        filtered_errors = warnings

    # Find common error patterns
    error_messages = [e["message"] for e in errors]
    common_errors = Counter(error_messages).most_common(10)

    return tool_response(
        total_lines=len(lines),
        parsed_lines=parsed_count,
        level_counts=dict(level_counts),
        errors=errors[:50],
        warnings=warnings[:50],
        common_errors=[{"message": msg, "count": cnt} for msg, cnt in common_errors],
        time_range={"start": timestamps[0] if timestamps else None,
                     "end": timestamps[-1] if timestamps else None},
    )


@tool_wrapper(required_params=["content", "pattern"])
def search_log_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Search log content for a pattern."""
    status.set_callback(params.pop("_status_callback", None))
    content = params["content"]
    pattern = params["pattern"]
    limit = int(params.get("limit", 50))

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return tool_error(f"Invalid pattern: {e}")

    matches = []
    for i, line in enumerate(content.splitlines()):
        if regex.search(line):
            matches.append({"line_number": i + 1, "content": line.strip()[:300]})
            if len(matches) >= limit:
                break

    return tool_response(matches=matches, count=len(matches), pattern=pattern)


__all__ = ["analyze_log_tool", "search_log_tool"]
''')

# ── 32. nginx-config-generator ────────────────────────────────────
create_skill(
    name="nginx-config-generator",
    frontmatter_name="generating-nginx-config",
    description="Generate Nginx reverse proxy and server block configurations. Pure Python templates. Use when the user wants to generate nginx config, reverse proxy, server block.",
    category="development",
    capabilities=["code", "devops"],
    triggers=["nginx", "reverse proxy", "nginx config", "server block", "proxy pass"],
    eval_tool="nginx_reverse_proxy_tool",
    eval_input={"domain": "example.com", "upstream_port": 3000},
    tool_docs="""### nginx_reverse_proxy_tool
Generate Nginx reverse proxy configuration.

**Parameters:**
- `domain` (str, required): Domain name
- `upstream_port` (int, required): Backend port
- `ssl` (bool, optional): Enable SSL (default: true)
- `upstream_host` (str, optional): Backend host (default: 127.0.0.1)
- `websocket` (bool, optional): Enable WebSocket support (default: false)

**Returns:**
- `success` (bool)
- `config` (str): Nginx configuration""",
    tools_code='''"""Nginx Config Generator Skill — generate reverse proxy configs."""
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("nginx-config-generator")


@tool_wrapper(required_params=["domain", "upstream_port"])
def nginx_reverse_proxy_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Nginx reverse proxy configuration."""
    status.set_callback(params.pop("_status_callback", None))
    domain = params["domain"].strip()
    port = int(params["upstream_port"])
    use_ssl = params.get("ssl", True)
    upstream_host = params.get("upstream_host", "127.0.0.1")
    websocket = params.get("websocket", False)
    max_body = params.get("max_body_size", "10m")
    cache_static = params.get("cache_static", True)

    upstream_name = domain.replace(".", "_")
    lines = []

    lines.append(f"upstream {upstream_name} {{")
    lines.append(f"    server {upstream_host}:{port};")
    lines.append("}")
    lines.append("")

    if use_ssl:
        # HTTP -> HTTPS redirect
        lines.append("server {")
        lines.append("    listen 80;")
        lines.append("    listen [::]:80;")
        lines.append(f"    server_name {domain} www.{domain};")
        lines.append(f"    return 301 https://{domain}$request_uri;")
        lines.append("}")
        lines.append("")

    lines.append("server {")
    if use_ssl:
        lines.append("    listen 443 ssl http2;")
        lines.append("    listen [::]:443 ssl http2;")
        lines.append(f"    server_name {domain} www.{domain};")
        lines.append("")
        lines.append(f"    ssl_certificate /etc/letsencrypt/live/{domain}/fullchain.pem;")
        lines.append(f"    ssl_certificate_key /etc/letsencrypt/live/{domain}/privkey.pem;")
        lines.append("    ssl_protocols TLSv1.2 TLSv1.3;")
        lines.append("    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;")
        lines.append("    ssl_prefer_server_ciphers off;")
        lines.append("    ssl_session_cache shared:SSL:10m;")
        lines.append("    ssl_session_timeout 1d;")
    else:
        lines.append("    listen 80;")
        lines.append("    listen [::]:80;")
        lines.append(f"    server_name {domain} www.{domain};")

    lines.append("")
    lines.append(f"    client_max_body_size {max_body};")
    lines.append("")
    lines.append("    # Security headers")
    lines.append("    add_header X-Frame-Options SAMEORIGIN;")
    lines.append("    add_header X-Content-Type-Options nosniff;")
    lines.append("    add_header X-XSS-Protection \"1; mode=block\";")
    if use_ssl:
        lines.append("    add_header Strict-Transport-Security \"max-age=31536000; includeSubDomains\" always;")
    lines.append("")

    if cache_static:
        lines.append("    # Static file caching")
        lines.append("    location ~* \\\\.(jpg|jpeg|png|gif|ico|css|js|woff2?|ttf|svg)$ {")
        lines.append(f"        proxy_pass http://{upstream_name};")
        lines.append("        expires 30d;")
        lines.append("        add_header Cache-Control \"public, immutable\";")
        lines.append("    }")
        lines.append("")

    lines.append("    location / {")
    lines.append(f"        proxy_pass http://{upstream_name};")
    lines.append("        proxy_http_version 1.1;")
    lines.append("        proxy_set_header Host $host;")
    lines.append("        proxy_set_header X-Real-IP $remote_addr;")
    lines.append("        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;")
    lines.append("        proxy_set_header X-Forwarded-Proto $scheme;")
    if websocket:
        lines.append("        proxy_set_header Upgrade $http_upgrade;")
        lines.append("        proxy_set_header Connection \"upgrade\";")
        lines.append("        proxy_read_timeout 86400;")
    lines.append("    }")
    lines.append("}")

    config = "\\n".join(lines)
    return tool_response(config=config, domain=domain, upstream=f"{upstream_host}:{port}",
                         ssl=use_ssl, websocket=websocket)


@tool_wrapper(required_params=["domain", "root_path"])
def nginx_static_site_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Nginx config for static site hosting."""
    status.set_callback(params.pop("_status_callback", None))
    domain = params["domain"]
    root_path = params["root_path"]
    use_ssl = params.get("ssl", True)
    spa = params.get("spa", False)

    lines = []
    if use_ssl:
        lines.append("server {")
        lines.append("    listen 80;")
        lines.append(f"    server_name {domain};")
        lines.append(f"    return 301 https://{domain}$request_uri;")
        lines.append("}")
        lines.append("")

    lines.append("server {")
    if use_ssl:
        lines.append("    listen 443 ssl http2;")
        lines.append(f"    ssl_certificate /etc/letsencrypt/live/{domain}/fullchain.pem;")
        lines.append(f"    ssl_certificate_key /etc/letsencrypt/live/{domain}/privkey.pem;")
    else:
        lines.append("    listen 80;")
    lines.append(f"    server_name {domain};")
    lines.append(f"    root {root_path};")
    lines.append("    index index.html;")
    lines.append("")
    if spa:
        lines.append("    location / {")
        lines.append("        try_files $uri $uri/ /index.html;")
        lines.append("    }")
    else:
        lines.append("    location / {")
        lines.append("        try_files $uri $uri/ =404;")
        lines.append("    }")
    lines.append("")
    lines.append("    location ~* \\\\.(jpg|jpeg|png|gif|ico|css|js|woff2?|ttf|svg)$ {")
    lines.append("        expires 30d;")
    lines.append("        add_header Cache-Control \"public, immutable\";")
    lines.append("    }")
    lines.append("")
    lines.append("    gzip on;")
    lines.append("    gzip_types text/plain text/css application/json application/javascript text/xml;")
    lines.append("}")

    config = "\\n".join(lines)
    return tool_response(config=config, domain=domain, root=root_path, ssl=use_ssl)


__all__ = ["nginx_reverse_proxy_tool", "nginx_static_site_tool"]
''')

# ── 33. ci-cd-pipeline-builder ────────────────────────────────────
create_skill(
    name="ci-cd-pipeline-builder",
    frontmatter_name="building-ci-cd-pipelines",
    description="Generate GitHub Actions workflow YAML configurations. Pure Python. Use when the user wants to create CI/CD pipeline, GitHub Actions, workflow yaml.",
    category="development",
    capabilities=["code", "devops"],
    triggers=["ci/cd", "github actions", "workflow", "pipeline", "continuous integration"],
    eval_tool="github_actions_tool",
    eval_input={"language": "python", "features": ["test", "lint"]},
    tool_docs="""### github_actions_tool
Generate GitHub Actions workflow YAML.

**Parameters:**
- `language` (str, required): Programming language (python, node, go, rust, java)
- `features` (list, optional): Features: test, lint, build, deploy, docker (default: [test])
- `name` (str, optional): Workflow name
- `branches` (list, optional): Trigger branches (default: [main])

**Returns:**
- `success` (bool)
- `yaml` (str): GitHub Actions workflow YAML
- `file_path` (str): Suggested file path""",
    tools_code='''"""CI/CD Pipeline Builder Skill — generate GitHub Actions YAML."""
import json
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("ci-cd-pipeline-builder")

LANGUAGE_CONFIGS = {
    "python": {
        "versions": ["3.10", "3.11", "3.12"],
        "setup": "actions/setup-python@v5",
        "install": "pip install -r requirements.txt",
        "test": "pytest tests/ -v",
        "lint": "flake8 . && black --check .",
        "version_key": "python-version",
    },
    "node": {
        "versions": ["18", "20", "22"],
        "setup": "actions/setup-node@v4",
        "install": "npm ci",
        "test": "npm test",
        "lint": "npm run lint",
        "version_key": "node-version",
    },
    "go": {
        "versions": ["1.21", "1.22"],
        "setup": "actions/setup-go@v5",
        "install": "go mod download",
        "test": "go test ./...",
        "lint": "golangci-lint run",
        "version_key": "go-version",
    },
    "rust": {
        "versions": ["stable"],
        "setup": "dtolnay/rust-toolchain@stable",
        "install": "",
        "test": "cargo test",
        "lint": "cargo clippy -- -D warnings",
        "version_key": "toolchain",
    },
    "java": {
        "versions": ["17", "21"],
        "setup": "actions/setup-java@v4",
        "install": "",
        "test": "mvn test",
        "lint": "",
        "version_key": "java-version",
    },
}


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\\n".join(prefix + line if line.strip() else line for line in text.splitlines())


@tool_wrapper(required_params=["language"])
def github_actions_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate GitHub Actions workflow YAML."""
    status.set_callback(params.pop("_status_callback", None))
    lang = params["language"].lower().strip()
    features = params.get("features", ["test"])
    if isinstance(features, str):
        features = [features]
    wf_name = params.get("name", f"CI ({lang.title()})")
    branches = params.get("branches", ["main"])

    if lang not in LANGUAGE_CONFIGS:
        return tool_error(f"Unsupported language: {lang}. Use: {list(LANGUAGE_CONFIGS.keys())}")

    cfg = LANGUAGE_CONFIGS[lang]
    branch_list = ", ".join(branches)

    lines = []
    lines.append(f"name: {wf_name}")
    lines.append("")
    lines.append("on:")
    lines.append("  push:")
    lines.append(f"    branches: [{branch_list}]")
    lines.append("  pull_request:")
    lines.append(f"    branches: [{branch_list}]")
    lines.append("")
    lines.append("jobs:")

    if "test" in features or "lint" in features:
        lines.append("  test:")
        lines.append("    runs-on: ubuntu-latest")
        if len(cfg["versions"]) > 1:
            lines.append("    strategy:")
            lines.append("      matrix:")
            vers = ", ".join(f'"{v}"' for v in cfg["versions"])
            lines.append(f"        {cfg['version_key']}: [{vers}]")
        lines.append("    steps:")
        lines.append("      - uses: actions/checkout@v4")
        lines.append(f"      - uses: {cfg['setup']}")
        if len(cfg["versions"]) > 1:
            lines.append("        with:")
            lines.append(f"          {cfg['version_key']}: ${{{{ matrix.{cfg['version_key']} }}}}")
        elif cfg.get("version_key") and cfg["versions"]:
            lines.append("        with:")
            lines.append(f"          {cfg['version_key']}: \"{cfg['versions'][0]}\"")

        if cfg["install"]:
            lines.append(f"      - run: {cfg['install']}")
            lines.append("        name: Install dependencies")

        if "lint" in features and cfg["lint"]:
            if lang == "python":
                lines.append("      - run: pip install flake8 black")
                lines.append("        name: Install linters")
            lines.append(f"      - run: {cfg['lint']}")
            lines.append("        name: Lint")

        if "test" in features:
            lines.append(f"      - run: {cfg['test']}")
            lines.append("        name: Test")

    if "docker" in features:
        lines.append("")
        lines.append("  docker:")
        lines.append("    runs-on: ubuntu-latest")
        if "test" in features:
            lines.append("    needs: test")
        lines.append("    steps:")
        lines.append("      - uses: actions/checkout@v4")
        lines.append("      - uses: docker/setup-buildx-action@v3")
        lines.append("      - uses: docker/login-action@v3")
        lines.append("        with:")
        lines.append("          registry: ghcr.io")
        lines.append("          username: ${{ github.actor }}")
        lines.append("          password: ${{ secrets.GITHUB_TOKEN }}")
        lines.append("      - uses: docker/build-push-action@v5")
        lines.append("        with:")
        lines.append("          push: ${{ github.ref == \'refs/heads/main\' }}")
        lines.append("          tags: ghcr.io/${{ github.repository }}:latest")

    if "deploy" in features:
        lines.append("")
        lines.append("  deploy:")
        lines.append("    runs-on: ubuntu-latest")
        lines.append("    needs: [test]")
        lines.append("    if: github.ref == \'refs/heads/main\'")
        lines.append("    steps:")
        lines.append("      - uses: actions/checkout@v4")
        lines.append("      - name: Deploy")
        lines.append("        run: echo \"Add deployment steps here\"")
        lines.append("        env:")
        lines.append("          DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}")

    yaml_content = "\\n".join(lines)
    return tool_response(yaml=yaml_content,
                         file_path=f".github/workflows/{lang}-ci.yml",
                         language=lang, features=features)


__all__ = ["github_actions_tool"]
''')

# ── 34. dockerfile-generator ─────────────────────────────────────
create_skill(
    name="dockerfile-generator",
    frontmatter_name="generating-dockerfiles",
    description="Generate Dockerfiles from language and framework specifications. Pure Python. Use when the user wants to generate Dockerfile, containerize, Docker image.",
    category="development",
    capabilities=["code", "devops"],
    triggers=["dockerfile", "docker", "containerize", "docker image", "container"],
    eval_tool="generate_dockerfile_tool",
    eval_input={"language": "python", "framework": "fastapi"},
    tool_docs="""### generate_dockerfile_tool
Generate a Dockerfile for a project.

**Parameters:**
- `language` (str, required): Language (python, node, go, rust, java)
- `framework` (str, optional): Framework (fastapi, flask, express, nextjs, gin)
- `port` (int, optional): Exposed port (default: auto-detected)
- `multi_stage` (bool, optional): Use multi-stage build (default: true)

**Returns:**
- `success` (bool)
- `dockerfile` (str): Dockerfile content
- `dockerignore` (str): .dockerignore content""",
    tools_code='''"""Dockerfile Generator Skill — generate Dockerfiles from specs."""
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("dockerfile-generator")

TEMPLATES = {
    "python": {
        "base": "python:{version}-slim",
        "default_version": "3.12",
        "default_port": 8000,
        "install": "COPY requirements.txt .\\nRUN pip install --no-cache-dir -r requirements.txt",
        "copy": "COPY . .",
        "frameworks": {
            "fastapi": {"cmd": "uvicorn main:app --host 0.0.0.0 --port {port}", "port": 8000},
            "flask": {"cmd": "gunicorn -w 4 -b 0.0.0.0:{port} app:app", "port": 5000},
            "django": {"cmd": "gunicorn -w 4 -b 0.0.0.0:{port} project.wsgi:application", "port": 8000},
            "default": {"cmd": "python main.py", "port": 8000},
        },
    },
    "node": {
        "base": "node:{version}-alpine",
        "default_version": "20",
        "default_port": 3000,
        "install": "COPY package*.json ./\\nRUN npm ci --only=production",
        "copy": "COPY . .",
        "frameworks": {
            "express": {"cmd": "node server.js", "port": 3000},
            "nextjs": {"cmd": "npm start", "port": 3000, "build": "RUN npm run build"},
            "nestjs": {"cmd": "node dist/main.js", "port": 3000, "build": "RUN npm run build"},
            "default": {"cmd": "node index.js", "port": 3000},
        },
    },
    "go": {
        "base": "golang:{version}-alpine",
        "default_version": "1.22",
        "default_port": 8080,
        "install": "COPY go.mod go.sum ./\\nRUN go mod download",
        "copy": "COPY . .",
        "frameworks": {
            "gin": {"cmd": "./app", "port": 8080},
            "fiber": {"cmd": "./app", "port": 3000},
            "default": {"cmd": "./app", "port": 8080},
        },
    },
    "rust": {
        "base": "rust:{version}-slim",
        "default_version": "1.75",
        "default_port": 8080,
        "install": "COPY Cargo.toml Cargo.lock ./\\nRUN mkdir src && echo \\"fn main() {}\\" > src/main.rs && cargo build --release && rm -rf src",
        "copy": "COPY . .\\nRUN cargo build --release",
        "frameworks": {
            "actix": {"cmd": "./target/release/app", "port": 8080},
            "axum": {"cmd": "./target/release/app", "port": 3000},
            "default": {"cmd": "./target/release/app", "port": 8080},
        },
    },
}

DOCKERIGNORE = """node_modules/
.git/
.gitignore
.env
.env.*
__pycache__/
*.pyc
.venv/
venv/
target/
dist/
build/
*.log
.DS_Store
Dockerfile
docker-compose*.yml
.dockerignore
README.md
"""


@tool_wrapper(required_params=["language"])
def generate_dockerfile_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a Dockerfile for a project."""
    status.set_callback(params.pop("_status_callback", None))
    lang = params["language"].lower().strip()
    framework = params.get("framework", "default").lower().strip()
    multi_stage = params.get("multi_stage", True)

    if lang not in TEMPLATES:
        return tool_error(f"Unsupported language: {lang}. Use: {list(TEMPLATES.keys())}")

    tmpl = TEMPLATES[lang]
    fw = tmpl["frameworks"].get(framework, tmpl["frameworks"]["default"])
    port = int(params.get("port", fw.get("port", tmpl["default_port"])))
    version = params.get("version", tmpl["default_version"])
    base_image = tmpl["base"].format(version=version)

    lines = []
    lines.append(f"# Auto-generated Dockerfile for {lang}/{framework}")
    lines.append("")

    if multi_stage and lang in ("go", "rust"):
        # Builder stage
        lines.append(f"FROM {base_image} AS builder")
        lines.append("WORKDIR /build")
        lines.append(tmpl["install"])
        lines.append(tmpl["copy"])
        if lang == "go":
            lines.append("RUN CGO_ENABLED=0 go build -o app .")
        lines.append("")
        # Runtime stage
        if lang == "go":
            lines.append("FROM alpine:3.19")
            lines.append("RUN apk --no-cache add ca-certificates")
        else:
            lines.append("FROM debian:bookworm-slim")
        lines.append("WORKDIR /app")
        lines.append("COPY --from=builder /build/target/release/app ./app" if lang == "rust" else "COPY --from=builder /build/app ./app")
    else:
        lines.append(f"FROM {base_image}")
        lines.append("WORKDIR /app")
        lines.append(tmpl["install"])
        if fw.get("build"):
            lines.append("COPY . .")
            lines.append(fw["build"])
        else:
            lines.append(tmpl["copy"])

    lines.append("")
    lines.append(f"EXPOSE {port}")
    lines.append("")
    cmd = fw["cmd"].format(port=port)
    cmd_parts = cmd.split()
    cmd_json = ", ".join(f'"{p}"' for p in cmd_parts)
    lines.append(f"CMD [{cmd_json}]")

    dockerfile = "\\n".join(lines)
    return tool_response(dockerfile=dockerfile, dockerignore=DOCKERIGNORE,
                         language=lang, framework=framework, port=port)


__all__ = ["generate_dockerfile_tool"]
''')

print(f"Batch 2 progress: skills 31-34 created.")
