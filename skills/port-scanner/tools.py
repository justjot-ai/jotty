"""Check if ports are open on a host using socket connections."""
import socket
from typing import Dict, Any, List
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("port-scanner")

_SERVICES: Dict[int, str] = {
    21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
    80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 465: "SMTPS",
    587: "SMTP-TLS", 993: "IMAPS", 995: "POP3S", 3306: "MySQL",
    3389: "RDP", 5432: "PostgreSQL", 5672: "RabbitMQ", 6379: "Redis",
    8080: "HTTP-Alt", 8443: "HTTPS-Alt", 9200: "Elasticsearch",
    27017: "MongoDB",
}


def _check_port(host: str, port: int, timeout: float) -> Dict[str, Any]:
    service = _SERVICES.get(port, "unknown")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            is_open = result == 0
    except socket.gaierror:
        return {"port": port, "open": False, "service": service, "error": "DNS resolution failed"}
    except OSError as e:
        return {"port": port, "open": False, "service": service, "error": str(e)}
    return {"port": port, "open": is_open, "service": service}


@tool_wrapper(required_params=["host"])
def scan_ports(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check if specific ports are open on a host."""
    status.set_callback(params.pop("_status_callback", None))
    host = params["host"]
    ports: List[int] = params.get("ports", [80, 443, 22, 8080])
    timeout = float(params.get("timeout", 1.0))
    if len(ports) > 100:
        return tool_error("Maximum 100 ports per scan")
    results = [_check_port(host, int(p), timeout) for p in ports]
    open_ports = [r for r in results if r["open"]]
    return tool_response(host=host, results=results, open_count=len(open_ports),
                         closed_count=len(results) - len(open_ports))


@tool_wrapper()
def list_common_ports(params: Dict[str, Any]) -> Dict[str, Any]:
    """List common ports and their associated services."""
    status.set_callback(params.pop("_status_callback", None))
    return tool_response(ports={str(k): v for k, v in sorted(_SERVICES.items())},
                         count=len(_SERVICES))


__all__ = ["scan_ports", "list_common_ports"]
