---
name: network-toolkit
description: "Unified network diagnostics and security toolkit. DNS lookup, IP geolocation, port scanning, SSL certificate checking, URL parsing, and HTTP status reference."
---

# Network Toolkit

Unified network diagnostics, security, and URL manipulation skill. Consolidates dns-lookup,
ip-lookup, port-scanner, ssl-certificate-checker, url-parser, uptime-monitor, and http-status-lookup.

## Type
base

## Capabilities
- network
- dns
- security
- diagnostics
- url

## Triggers
- "dns lookup"
- "ip lookup"
- "geolocation"
- "port scan"
- "ssl certificate"
- "ssl check"
- "parse url"
- "url encode"
- "whois"

## Category
network

## Tools

### dns_lookup_tool
Resolve DNS records for a domain using Cloudflare DNS-over-HTTPS.

**Parameters:**
- `domain` (str, required): Domain name to look up
- `record_type` (str, optional): Record type: A, AAAA, MX, CNAME, TXT, NS, SOA, SRV, PTR (default: A)

### dns_all_records_tool
Get all common DNS record types (A, AAAA, MX, NS, TXT, CNAME) for a domain.

**Parameters:**
- `domain` (str, required): Domain name

### ip_lookup_tool
IP geolocation and network info (country, city, ISP, coordinates).

**Parameters:**
- `ip` (str, optional): IP address (default: your public IP)

### scan_ports_tool
Check if specific ports are open on a host.

**Parameters:**
- `host` (str, required): Hostname or IP
- `ports` (array, optional): Ports to check (default: [80, 443, 22, 8080])
- `timeout` (float, optional): Connection timeout in seconds (default: 1.0)

### check_ssl_tool
Check SSL/TLS certificate validity, issuer, and expiry for a hostname.

**Parameters:**
- `hostname` (str, required): Hostname to check
- `port` (int, optional): Port (default: 443)

### parse_url_tool
Parse a URL into its components (scheme, host, port, path, query, fragment).

**Parameters:**
- `url` (str, required): URL to parse

### build_url_tool
Build a URL from components.

**Parameters:**
- `scheme` (str, optional): URL scheme (default: https)
- `host` (str, required): Hostname
- `port` (int, optional): Port
- `path` (str, optional): Path (default: /)
- `query` (object, optional): Query parameters
- `fragment` (str, optional): Fragment

### url_encode_tool
URL-encode or decode text.

**Parameters:**
- `text` (str, required): Text to encode/decode
- `mode` (str, optional): encode or decode (default: encode)

## Dependencies
- requests
