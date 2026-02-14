"""Nginx Config Generator Skill — generate reverse proxy configs."""
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

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
    lines.append('    add_header X-XSS-Protection "1; mode=block";')
    if use_ssl:
        lines.append('    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;')
    lines.append("")

    if cache_static:
        lines.append("    # Static file caching")
        lines.append("    location ~* \\.(jpg|jpeg|png|gif|ico|css|js|woff2?|ttf|svg)$ {")
        lines.append(f"        proxy_pass http://{upstream_name};")
        lines.append("        expires 30d;")
        lines.append('        add_header Cache-Control "public, immutable";')
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
        lines.append("        proxy_set_header Connection "upgrade";")
        lines.append("        proxy_read_timeout 86400;")
    lines.append("    }")
    lines.append("}")

    config = "\n".join(lines)
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
    lines.append("    location ~* \\.(jpg|jpeg|png|gif|ico|css|js|woff2?|ttf|svg)$ {")
    lines.append("        expires 30d;")
    lines.append("        add_header Cache-Control "public, immutable";")
    lines.append("    }")
    lines.append("")
    lines.append("    gzip on;")
    lines.append("    gzip_types text/plain text/css application/json application/javascript text/xml;")
    lines.append("}")

    config = "\n".join(lines)
    return tool_response(config=config, domain=domain, root=root_path, ssl=use_ssl)


__all__ = ["nginx_reverse_proxy_tool", "nginx_static_site_tool"]
