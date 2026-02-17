"""
Dev Generator Toolkit — Unified developer config and scaffold generation skill.

Consolidates: dockerfile-generator, nginx-config-generator, ci-cd-pipeline-builder,
gitignore-generator, readme-generator, license-generator, robots-txt-generator,
sitemap-generator, api-docs-generator, env-config-manager, db-seed-generator.
"""

from datetime import datetime
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("dev-generator-toolkit")

# =============================================================================
# DOCKERFILE TEMPLATES
# =============================================================================

_DOCKER = {
    "python": {
        "base": "python:{version}-slim",
        "default_version": "3.12",
        "default_port": 8000,
        "install": "COPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt",
        "copy": "COPY . .",
        "frameworks": {
            "fastapi": {"cmd": "uvicorn main:app --host 0.0.0.0 --port {port}", "port": 8000},
            "flask": {"cmd": "gunicorn -w 4 -b 0.0.0.0:{port} app:app", "port": 5000},
            "django": {
                "cmd": "gunicorn -w 4 -b 0.0.0.0:{port} project.wsgi:application",
                "port": 8000,
            },
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
            "default": {"cmd": "./app", "port": 8080},
        },
    },
    "rust": {
        "base": "rust:{version}-slim",
        "default_version": "1.75",
        "default_port": 8080,
        "install": 'COPY Cargo.toml Cargo.lock ./\nRUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src',
        "copy": "COPY . .\nRUN cargo build --release",
        "frameworks": {
            "actix": {"cmd": "./target/release/app", "port": 8080},
            "default": {"cmd": "./target/release/app", "port": 8080},
        },
    },
}

_DOCKERIGNORE = "node_modules/\n.git/\n.env\n.env.*\n__pycache__/\n*.pyc\n.venv/\nvenv/\ntarget/\ndist/\nbuild/\n*.log\n.DS_Store\nDockerfile\ndocker-compose*.yml\n"

# =============================================================================
# GITIGNORE PATTERNS
# =============================================================================

_GITIGNORE = {
    "python": "__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nbuild/\ndevelop-eggs/\ndist/\n*.egg-info/\n*.egg\n.venv/\nvenv/\n.env\n.mypy_cache/\n.pytest_cache/\n",
    "node": "node_modules/\ndist/\nbuild/\n.env\n.env.local\n*.log\nnpm-debug.log*\ncoverage/\n.next/\n",
    "go": "*.exe\n*.exe~\n*.dll\n*.so\n*.dylib\n*.test\n*.out\nvendor/\n",
    "rust": "target/\n**/*.rs.bk\nCargo.lock\n",
    "java": "*.class\n*.jar\n*.war\n*.ear\ntarget/\n.gradle/\nbuild/\n",
    "ruby": "*.gem\n*.rbc\n/.config\n/coverage/\n/InstalledFiles\n/pkg/\n/spec/reports/\n/test/tmp/\n",
    "common": ".DS_Store\nThumbs.db\n*.swp\n*.swo\n*~\n.idea/\n.vscode/\n*.sublime-*\n",
}

# =============================================================================
# LICENSE TEMPLATES
# =============================================================================

_LICENSES = {
    "mit": """MIT License

Copyright (c) {year} {author}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""",
    "apache2": """Apache License, Version 2.0

Copyright {year} {author}

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.""",
    "isc": """ISC License

Copyright (c) {year} {author}

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.""",
}


# =============================================================================
# TOOLS
# =============================================================================


@tool_wrapper(required_params=["language"])
def generate_dockerfile_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a Dockerfile for Python, Node.js, Go, or Rust with multi-stage builds."""
    status.set_callback(params.pop("_status_callback", None))
    lang = params["language"].lower().strip()
    framework = params.get("framework", "default").lower().strip()
    multi_stage = params.get("multi_stage", True)
    if lang not in _DOCKER:
        return tool_error(f"Unsupported language: {lang}. Use: {list(_DOCKER.keys())}")
    tmpl = _DOCKER[lang]
    fw = tmpl["frameworks"].get(framework, tmpl["frameworks"]["default"])
    port = int(params.get("port", fw.get("port", tmpl["default_port"])))
    version = params.get("version", tmpl["default_version"])
    base_image = tmpl["base"].format(version=version)
    lines = [f"# Dockerfile for {lang}/{framework}", ""]
    if multi_stage and lang in ("go", "rust"):
        lines += [f"FROM {base_image} AS builder", "WORKDIR /build", tmpl["install"], tmpl["copy"]]
        if lang == "go":
            lines.append("RUN CGO_ENABLED=0 go build -o app .")
        lines += ["", "FROM alpine:3.19" if lang == "go" else "FROM debian:bookworm-slim"]
        if lang == "go":
            lines.append("RUN apk --no-cache add ca-certificates")
        lines.append("WORKDIR /app")
        lines.append(
            "COPY --from=builder /build/target/release/app ./app"
            if lang == "rust"
            else "COPY --from=builder /build/app ./app"
        )
    else:
        lines += [f"FROM {base_image}", "WORKDIR /app", tmpl["install"]]
        if fw.get("build"):
            lines += ["COPY . .", fw["build"]]
        else:
            lines.append(tmpl["copy"])
    cmd = fw["cmd"].format(port=port)
    cmd_json = ", ".join(f'"{p}"' for p in cmd.split())
    lines += ["", f"EXPOSE {port}", "", f"CMD [{cmd_json}]"]
    return tool_response(
        dockerfile="\n".join(lines),
        dockerignore=_DOCKERIGNORE,
        language=lang,
        framework=framework,
        port=port,
    )


@tool_wrapper(required_params=["server_name"])
def generate_nginx_config_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate nginx configuration for reverse proxy, static site, or load balancer."""
    status.set_callback(params.pop("_status_callback", None))
    server = params["server_name"]
    mode = params.get("mode", "reverse_proxy")
    upstream_port = int(params.get("upstream_port", 3000))
    use_ssl = params.get("ssl", True)
    root = params.get("root_path", "/var/www/html")
    lines = [f"# nginx config for {server}", ""]
    if mode == "reverse_proxy":
        lines += [
            "server {",
            f"    server_name {server};",
            "    listen 80;",
            "",
            "    location / {",
            f"        proxy_pass http://127.0.0.1:{upstream_port};",
            "        proxy_set_header Host $host;",
            "        proxy_set_header X-Real-IP $remote_addr;",
            "        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;",
            "        proxy_set_header X-Forwarded-Proto $scheme;",
            "    }",
            "}",
        ]
    elif mode == "static":
        lines += [
            "server {",
            f"    server_name {server};",
            "    listen 80;",
            f"    root {root};",
            "    index index.html;",
            "",
            "    location / {",
            "        try_files $uri $uri/ /index.html;",
            "    }",
            "",
            "    location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff2?)$ {",
            "        expires 30d;",
            '        add_header Cache-Control "public, immutable";',
            "    }",
            "}",
        ]
    else:
        return tool_error(f"Unknown mode: {mode}. Use: reverse_proxy, static")
    if use_ssl:
        lines += [
            "",
            "# SSL config (uncomment after certbot setup):",
            f"# listen 443 ssl;",
            f"# ssl_certificate /etc/letsencrypt/live/{server}/fullchain.pem;",
            f"# ssl_certificate_key /etc/letsencrypt/live/{server}/privkey.pem;",
        ]
    return tool_response(config="\n".join(lines), server_name=server, mode=mode)


@tool_wrapper(required_params=["platform", "language"])
def generate_ci_cd_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate CI/CD pipeline for GitHub Actions, GitLab CI, or CircleCI."""
    status.set_callback(params.pop("_status_callback", None))
    platform = params["platform"].lower()
    lang = params["language"].lower()
    steps = params.get("steps", ["lint", "test", "build"])
    if platform == "github":
        node_ver = params.get("node_version", "20")
        py_ver = params.get("python_version", "3.12")
        lines = [
            "name: CI",
            "",
            "on:",
            "  push:",
            "    branches: [main]",
            "  pull_request:",
            "    branches: [main]",
            "",
            "jobs:",
            "  build:",
            "    runs-on: ubuntu-latest",
            "    steps:",
            "      - uses: actions/checkout@v4",
        ]
        if lang == "python":
            lines += [
                f"      - uses: actions/setup-python@v5",
                f"        with:",
                f"          python-version: '{py_ver}'",
                "      - run: pip install -r requirements.txt",
            ]
            if "lint" in steps:
                lines.append("      - run: ruff check .")
            if "test" in steps:
                lines.append("      - run: pytest")
        elif lang == "node":
            lines += [
                f"      - uses: actions/setup-node@v4",
                f"        with:",
                f"          node-version: '{node_ver}'",
                "      - run: npm ci",
            ]
            if "lint" in steps:
                lines.append("      - run: npm run lint")
            if "test" in steps:
                lines.append("      - run: npm test")
            if "build" in steps:
                lines.append("      - run: npm run build")
        elif lang == "go":
            lines += [
                "      - uses: actions/setup-go@v5",
                "        with:",
                "          go-version: '1.22'",
            ]
            if "lint" in steps:
                lines.append("      - run: go vet ./...")
            if "test" in steps:
                lines.append("      - run: go test ./...")
            if "build" in steps:
                lines.append("      - run: go build ./...")
        config = "\n".join(lines)
    elif platform == "gitlab":
        lines = ["stages:"]
        for s in steps:
            lines.append(f"  - {s}")
        lines += [""]
        if "test" in steps:
            lines += [
                f"test:",
                f"  stage: test",
                f"  image: {'python:3.12' if lang == 'python' else 'node:20'}",
                f"  script:",
            ]
            if lang == "python":
                lines += ["    - pip install -r requirements.txt", "    - pytest"]
            else:
                lines += ["    - npm ci", "    - npm test"]
        config = "\n".join(lines)
    else:
        return tool_error(f"Unknown platform: {platform}. Use: github, gitlab")
    return tool_response(config=config, platform=platform, language=lang, steps=steps)


@tool_wrapper(required_params=["languages"])
def generate_gitignore_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate .gitignore for specified languages/frameworks."""
    status.set_callback(params.pop("_status_callback", None))
    languages = params["languages"]
    if isinstance(languages, str):
        languages = [l.strip() for l in languages.split(",")]
    sections = [_GITIGNORE.get("common", "")]
    for lang in languages:
        section = _GITIGNORE.get(lang.lower())
        if section:
            sections.append(f"# {lang}\n{section}")
    return tool_response(gitignore="\n".join(sections), languages=languages)


@tool_wrapper(required_params=["license_type"])
def generate_license_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an open source license file."""
    status.set_callback(params.pop("_status_callback", None))
    lt = params["license_type"].lower().replace("-", "").replace(" ", "")
    author = params.get("author", "[Author Name]")
    year = params.get("year", datetime.now().year)
    template = _LICENSES.get(lt)
    if not template:
        return tool_error(
            f"Unknown license: {params['license_type']}. Available: {list(_LICENSES.keys())}"
        )
    return tool_response(license=template.format(year=year, author=author), license_type=lt)


@tool_wrapper(required_params=["project_name"])
def generate_readme_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a README.md template."""
    status.set_callback(params.pop("_status_callback", None))
    name = params["project_name"]
    desc = params.get("description", f"A brief description of {name}.")
    lang = params.get("language", "")
    features = params.get("features", [])
    sections = params.get("sections", ["installation", "usage"])
    lines = [f"# {name}", "", desc, ""]
    if features:
        lines += ["## Features", ""] + [f"- {f}" for f in features] + [""]
    if "installation" in sections:
        lines += [
            "## Installation",
            "",
            "```bash",
            f"git clone https://github.com/user/{name.lower().replace(' ', '-')}.git",
            f"cd {name.lower().replace(' ', '-')}",
        ]
        if lang == "python":
            lines += ["pip install -r requirements.txt"]
        elif lang == "node":
            lines += ["npm install"]
        lines += ["```", ""]
    if "usage" in sections:
        lines += ["## Usage", "", "```bash", f"# Run {name}", "```", ""]
    lines += ["## License", "", "MIT"]
    return tool_response(readme="\n".join(lines), project_name=name)


@tool_wrapper(required_params=["stack"])
def generate_env_template_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate .env.example template for a tech stack."""
    status.set_callback(params.pop("_status_callback", None))
    stack = params["stack"].lower()
    extras = params.get("extras", [])
    common = [
        "# Application\nNODE_ENV=development\nPORT=3000\n\n# Database\nDATABASE_URL=postgresql://user:pass@localhost:5432/dbname\n"
    ]
    stack_vars = {
        "node": "# Node.js\nNODE_ENV=development\nPORT=3000\n\n# Database\nDATABASE_URL=postgresql://user:pass@localhost:5432/db\n\n# Auth\nJWT_SECRET=your-secret-key\nJWT_EXPIRES_IN=7d\n",
        "python": "# Python\nDEBUG=True\nSECRET_KEY=your-secret-key\nDATABASE_URL=postgresql://user:pass@localhost:5432/db\n\n# Redis\nREDIS_URL=redis://localhost:6379\n",
        "django": "# Django\nDJANGO_SECRET_KEY=your-secret-key\nDJANGO_DEBUG=True\nDJANGO_ALLOWED_HOSTS=localhost,127.0.0.1\nDATABASE_URL=postgresql://user:pass@localhost:5432/db\n",
        "fastapi": "# FastAPI\nDEBUG=True\nSECRET_KEY=your-secret-key\nDATABASE_URL=postgresql://user:pass@localhost:5432/db\nREDIS_URL=redis://localhost:6379\nCORS_ORIGINS=http://localhost:3000\n",
        "nextjs": "# Next.js\nNEXTAUTH_URL=http://localhost:3000\nNEXTAUTH_SECRET=your-secret\nDATABASE_URL=postgresql://user:pass@localhost:5432/db\n",
    }
    env = stack_vars.get(stack, common[0])
    if extras:
        env += "\n# Additional\n" + "\n".join(f"{e}=" for e in extras) + "\n"
    return tool_response(env_template=env, stack=stack)


__all__ = [
    "generate_dockerfile_tool",
    "generate_nginx_config_tool",
    "generate_ci_cd_tool",
    "generate_gitignore_tool",
    "generate_license_tool",
    "generate_readme_tool",
    "generate_env_template_tool",
]
