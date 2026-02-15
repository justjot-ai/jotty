# Jotty MCP Server - Python 3.11 for mcp package (requires 3.10+)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt server.py ./

RUN pip install --no-cache-dir -r requirements.txt

ENV JOTTY_GATEWAY_URL=http://localhost:8766
# stdio: for n8n spawning via wrapper script
CMD ["python", "server.py"]
