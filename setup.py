"""
Custom setup.py for package discovery.

The repo root IS the Jotty package (has __init__.py). setuptools' find_packages()
discovers core/, apps/, sdk/ as top-level, but they need to be namespaced
under Jotty.core, Jotty.apps, etc.

Skills directories use hyphens (e.g. web-search/) so they're not valid Python
packages. They're included explicitly and loaded dynamically at runtime by
the skills registry.
"""

import os

from setuptools import find_packages, setup

EXCLUDE = [
    "tests",
    "tests.*",
    "scripts",
    "scripts.*",
    "examples",
    "examples.*",
    "docs",
    "docs.*",
    # Non-Python directories that leak into wheel
    "apps.web.frontend",
    "apps.web.frontend.*",
    "apps.whatsapp.node_modules",
    "apps.whatsapp.node_modules.*",
    "apps._archived",
    "apps._archived.*",
    "sdk.generated",
    "sdk.generated.*",
]

found = find_packages(where=".", exclude=EXCLUDE)
packages = ["Jotty"] + [f"Jotty.{p}" for p in found]
package_dir = {"Jotty": "."}

# Skills use hyphens so find_packages() misses them. Walk and add manually.
skills_root = os.path.join(os.path.dirname(__file__) or ".", "skills")
for entry in os.listdir(skills_root):
    full = os.path.join(skills_root, entry)
    if os.path.isdir(full) and not entry.startswith((".", "_")):
        pkg_name = f"Jotty.skills.{entry}"
        if pkg_name not in packages:
            packages.append(pkg_name)
        # Include subdirectories too
        for sub in os.listdir(full):
            sub_full = os.path.join(full, sub)
            if os.path.isdir(sub_full) and not sub.startswith((".", "_")):
                sub_pkg = f"{pkg_name}.{sub}"
                if sub_pkg not in packages:
                    packages.append(sub_pkg)

# Package data — include non-.py files needed at runtime.
# Skills need SKILL.md (parsed by registry) and eval.json.
# Core needs prompt .md files and config .yml/.yaml.
DATA_PATTERNS = ["*.md", "*.yml", "*.yaml", "*.json"]

package_data = {
    "Jotty": ["py.typed"],
}
for pkg in packages:
    if pkg.startswith("Jotty.skills.") or pkg.startswith("Jotty.core."):
        package_data[pkg] = DATA_PATTERNS

setup(
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
)
