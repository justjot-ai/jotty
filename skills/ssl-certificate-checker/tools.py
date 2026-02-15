"""SSL Certificate Checker Skill — check cert validity and expiry."""

import socket
import ssl
from datetime import datetime, timezone
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

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
        not_before = cert.get("notBefore", "")

        # Parse expiry date
        expire_dt = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(
            tzinfo=timezone.utc
        )
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
