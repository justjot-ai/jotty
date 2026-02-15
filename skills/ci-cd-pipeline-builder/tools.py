"""CI/CD Pipeline Builder Skill — generate GitHub Actions YAML."""

import json
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

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
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


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
            lines.append(f'          {cfg["version_key"]}: "{cfg["versions"][0]}"')

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
        lines.append("          push: ${{ github.ref == 'refs/heads/main' }}")
        lines.append("          tags: ghcr.io/${{ github.repository }}:latest")

    if "deploy" in features:
        lines.append("")
        lines.append("  deploy:")
        lines.append("    runs-on: ubuntu-latest")
        lines.append("    needs: [test]")
        lines.append("    if: github.ref == 'refs/heads/main'")
        lines.append("    steps:")
        lines.append("      - uses: actions/checkout@v4")
        lines.append("      - name: Deploy")
        lines.append('        run: echo "Add deployment steps here"')
        lines.append("        env:")
        lines.append("          DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}")

    yaml_content = "\n".join(lines)
    return tool_response(
        yaml=yaml_content,
        file_path=f".github/workflows/{lang}-ci.yml",
        language=lang,
        features=features,
    )


__all__ = ["github_actions_tool"]
