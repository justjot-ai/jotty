---
name: dev-generator-toolkit
description: "Unified developer config and scaffold generator. Generate Dockerfiles, nginx configs, CI/CD pipelines, .gitignore, README, LICENSE, robots.txt, sitemaps, API docs, .env templates, and database seeds."
---

# Dev Generator Toolkit

Unified developer config and scaffold generation skill. Consolidates dockerfile-generator,
nginx-config-generator, ci-cd-pipeline-builder, gitignore-generator, readme-generator,
license-generator, robots-txt-generator, sitemap-generator, api-docs-generator,
env-config-manager, and db-seed-generator.

## Type
base

## Capabilities
- generate
- scaffold
- devops
- configuration

## Triggers
- "generate dockerfile"
- "dockerfile"
- "nginx config"
- "ci cd pipeline"
- "github actions"
- "gitignore"
- "readme"
- "license"
- "robots.txt"
- "sitemap"
- "api docs"
- "env file"
- "database seed"

## Category
devops

## Tools

### generate_dockerfile_tool
Generate a Dockerfile for Python, Node.js, Go, or Rust projects with multi-stage builds.

**Parameters:**
- `language` (str, required): Language: python, node, go, rust
- `framework` (str, optional): Framework: fastapi, flask, django, express, nextjs, gin, actix (default: default)
- `port` (int, optional): Exposed port
- `version` (str, optional): Language version
- `multi_stage` (bool, optional): Use multi-stage builds (default: true)

### generate_nginx_config_tool
Generate nginx configuration for reverse proxy, static site, or load balancer.

**Parameters:**
- `server_name` (str, required): Server domain name
- `mode` (str, optional): Config mode: reverse_proxy, static, load_balancer (default: reverse_proxy)
- `upstream_port` (int, optional): Upstream port for proxy (default: 3000)
- `ssl` (bool, optional): Enable SSL with Let's Encrypt paths (default: true)
- `root_path` (str, optional): Document root for static mode (default: /var/www/html)

### generate_ci_cd_tool
Generate CI/CD pipeline configuration for GitHub Actions, GitLab CI, or CircleCI.

**Parameters:**
- `platform` (str, required): CI platform: github, gitlab, circleci
- `language` (str, required): Project language: python, node, go, rust
- `steps` (array, optional): Pipeline steps: lint, test, build, deploy (default: [lint, test, build])
- `node_version` (str, optional): Node.js version (default: 20)
- `python_version` (str, optional): Python version (default: 3.12)

### generate_gitignore_tool
Generate .gitignore for a project type.

**Parameters:**
- `languages` (array, required): Languages/frameworks: python, node, go, rust, java, ruby, swift, react, vue, angular

### generate_license_tool
Generate an open source license file.

**Parameters:**
- `license_type` (str, required): License: mit, apache2, gpl3, bsd2, bsd3, isc, unlicense, mpl2
- `author` (str, optional): Author name (default: [Author Name])
- `year` (int, optional): Copyright year (default: current year)

### generate_readme_tool
Generate a README.md template.

**Parameters:**
- `project_name` (str, required): Project name
- `description` (str, optional): Project description
- `language` (str, optional): Primary language
- `features` (array, optional): Feature list
- `sections` (array, optional): Extra sections: api, contributing, changelog, faq (default: [installation, usage])

### generate_env_template_tool
Generate a .env.example template with common variables for a stack.

**Parameters:**
- `stack` (str, required): Tech stack: node, python, django, fastapi, nextjs, rails, docker
- `extras` (array, optional): Additional env vars to include

## Dependencies
None
