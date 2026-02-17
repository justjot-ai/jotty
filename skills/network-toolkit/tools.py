"""
Network Toolkit — Unified network diagnostics and security skill.

Consolidates: dns-lookup, ip-lookup, port-scanner, ssl-certificate-checker,
url-parser, uptime-monitor, http-status-lookup.
"""

import socket
import ssl
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse, urlunparse

import requests
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("network-toolkit")

# =============================================================================
# CONSTANTS
# =============================================================================

_SERVICES: Dict[int, str] = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    143: "IMAP",
    443: "HTTPS",
    465: "SMTPS",
    587: "SMTP-TLS",
    993: "IMAPS",
    995: "POP3S",
    3306: "MySQL",
    3389: "RDP",
    5432: "PostgreSQL",
    5672: "RabbitMQ",
    6379: "Redis",
    8080: "HTTP-Alt",
    8443: "HTTPS-Alt",
    9200: "Elasticsearch",
    27017: "MongoDB",
}


# =============================================================================
# DNS TOOLS
# =============================================================================


@tool_wrapper(required_params=["domain"])
def dns_lookup_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve DNS records using Cloudflare DNS-over-HTTPS."""
    status.set_callback(params.pop("_status_callback", None))
    domain = params["domain"].strip().rstrip(".")
    record_type = params.get("record_type", "A").upper()
    valid_types = {"A", "AAAA", "MX", "CNAME", "TXT", "NS", "SOA", "SRV", "PTR"}
    if record_type not in valid_types:
        return tool_error(f"Invalid record type. Use one of: {sorted(valid_types)}")
    try:
        resp = requests.get(
            "https://cloudflare-dns.com/dns-query",
            params={"name": domain, "type": record_type},
            headers={"Accept": "application/dns-json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        records = [
            {
                "name": a.get("name", ""),
                "type": a.get("type", 0),
                "ttl": a.get("TTL", 0),
                "data": a.get("data", ""),
            }
            for a in data.get("Answer", [])
        ]
        return tool_response(
            domain=domain, record_type=record_type, records=records, count=len(records)
        )
    except requests.RequestException as e:
        return tool_error(f"DNS lookup failed: {e}")


@tool_wrapper(required_params=["domain"])
def dns_all_records_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get all common DNS record types for a domain."""
    status.set_callback(params.pop("_status_callback", None))
    domain = params["domain"]
    all_records: Dict[str, List] = {}
    for rtype in ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]:
        result = dns_lookup_tool({"domain": domain, "record_type": rtype})
        if result.get("success") and result.get("records"):
            all_records[rtype] = result["records"]
    return tool_response(domain=domain, records=all_records)


# =============================================================================
# IP LOOKUP
# =============================================================================


@tool_wrapper()
def ip_lookup_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """IP geolocation and network info (country, city, ISP, coordinates)."""
    status.set_callback(params.pop("_status_callback", None))
    ip = params.get("ip", "")
    try:
        url = f"http://ip-api.com/json/{ip}" if ip else "http://ip-api.com/json/"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "fail":
            return tool_error(data.get("message", "Lookup failed"))
        return tool_response(
            ip=data.get("query", ip),
            country=data.get("country", ""),
            country_code=data.get("countryCode", ""),
            region=data.get("regionName", ""),
            city=data.get("city", ""),
            zip=data.get("zip", ""),
            lat=data.get("lat"),
            lon=data.get("lon"),
            timezone=data.get("timezone", ""),
            isp=data.get("isp", ""),
            org=data.get("org", ""),
            as_number=data.get("as", ""),
        )
    except requests.RequestException as e:
        return tool_error(f"Lookup failed: {e}")


# =============================================================================
# PORT SCANNING
# =============================================================================


def _check_port(host: str, port: int, timeout: float) -> Dict[str, Any]:
    service = _SERVICES.get(port, "unknown")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            return {"port": port, "open": result == 0, "service": service}
    except socket.gaierror:
        return {"port": port, "open": False, "service": service, "error": "DNS resolution failed"}
    except OSError as e:
        return {"port": port, "open": False, "service": service, "error": str(e)}


@tool_wrapper(required_params=["host"])
def scan_ports_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check if specific ports are open on a host."""
    status.set_callback(params.pop("_status_callback", None))
    host = params["host"]
    ports: List[int] = params.get("ports", [80, 443, 22, 8080])
    timeout = float(params.get("timeout", 1.0))
    if len(ports) > 100:
        return tool_error("Maximum 100 ports per scan")
    results = [_check_port(host, int(p), timeout) for p in ports]
    open_ports = [r for r in results if r["open"]]
    return tool_response(
        host=host,
        results=results,
        open_count=len(open_ports),
        closed_count=len(results) - len(open_ports),
    )


# =============================================================================
# SSL CERTIFICATE CHECK
# =============================================================================


@tool_wrapper(required_params=["hostname"])
def check_ssl_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check SSL/TLS certificate validity, issuer, and expiry."""
    status.set_callback(params.pop("_status_callback", None))
    hostname = params["hostname"].strip().lower()
    port = int(params.get("port", 443))
    for prefix in ("https://", "http://"):
        if hostname.startswith(prefix):
            hostname = hostname[len(prefix) :]
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
        expire_dt = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(
            tzinfo=timezone.utc
        )
        days_remaining = (expire_dt - datetime.now(timezone.utc)).days
        san_list = [v for _, v in cert.get("subjectAltName", ())]
        return tool_response(
            hostname=hostname,
            subject=subject.get("commonName", ""),
            issuer=issuer.get("organizationName", issuer.get("commonName", "")),
            not_before=cert.get("notBefore", ""),
            expires=not_after,
            days_remaining=days_remaining,
            valid=days_remaining > 0,
            serial_number=cert.get("serialNumber", ""),
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


# =============================================================================
# URL TOOLS
# =============================================================================


@tool_wrapper(required_params=["url"])
def parse_url_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a URL into components (scheme, host, port, path, query, fragment)."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        parsed = urlparse(params["url"])
        query_params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}
        return tool_response(
            scheme=parsed.scheme,
            host=parsed.hostname or "",
            port=parsed.port,
            path=parsed.path,
            query=query_params,
            fragment=parsed.fragment,
            username=parsed.username,
            password=parsed.password,
        )
    except Exception as e:
        return tool_error(f"Failed to parse URL: {e}")


@tool_wrapper()
def build_url_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a URL from components."""
    status.set_callback(params.pop("_status_callback", None))
    host = params.get("host", "")
    if not host:
        return tool_error("host parameter required")
    scheme = params.get("scheme", "https")
    port = params.get("port")
    netloc = f"{host}:{port}" if port else host
    qs = urlencode(params.get("query", {})) if params.get("query") else ""
    url = urlunparse((scheme, netloc, params.get("path", "/"), "", qs, params.get("fragment", "")))
    return tool_response(url=url)


@tool_wrapper(required_params=["text"])
def url_encode_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """URL-encode or decode text."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    if params.get("mode", "encode") == "decode":
        return tool_response(result=unquote(text))
    return tool_response(result=quote(text, safe=params.get("safe", "")))


__all__ = [
    "dns_lookup_tool",
    "dns_all_records_tool",
    "ip_lookup_tool",
    "scan_ports_tool",
    "check_ssl_tool",
    "parse_url_tool",
    "build_url_tool",
    "url_encode_tool",
]
