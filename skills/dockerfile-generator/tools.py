"""Dockerfile Generator Skill — generate Dockerfiles from specs."""
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("dockerfile-generator")

TEMPLATES = {
    "python": {
        "base": "python:{version}-slim",
        "default_version": "3.12",
        "default_port": 8000,
        "install": "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt",
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
        "install": "COPY package*.json ./\nRUN npm ci --only=production",
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
        "install": "COPY go.mod go.sum ./\nRUN go mod download",
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
        "install": "COPY Cargo.toml Cargo.lock ./\nRUN mkdir src && echo \"fn main() {}\" > src/main.rs && cargo build --release && rm -rf src",
        "copy": "COPY . .\nRUN cargo build --release",
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

    dockerfile = "\n".join(lines)
    return tool_response(dockerfile=dockerfile, dockerignore=DOCKERIGNORE,
                         language=lang, framework=framework, port=port)


__all__ = ["generate_dockerfile_tool"]
