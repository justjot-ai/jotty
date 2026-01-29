#!/usr/bin/env python3
"""
Multi-Language SDK Generator for Jotty

Automatically generates SDKs for multiple languages from OpenAPI specification.
Uses OpenAPI Generator (https://openapi-generator.tech/) to create client libraries.

Supported languages:
- TypeScript/JavaScript (Node.js, Browser)
- Python
- Go
- Java
- Ruby
- PHP
- Swift
- Kotlin
- Rust
- C#
- Dart
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import shutil


# Configuration for each language SDK
SDK_CONFIGS = {
    "typescript-node": {
        "generator": "typescript-node",
        "package_name": "@jotty/sdk-node",
        "description": "Jotty SDK for Node.js",
        "output_dir": "sdk/generated/typescript-node",
        "additional_properties": {
            "npmName": "@jotty/sdk-node",
            "npmVersion": "1.0.0",
            "supportsES6": "true",
            "withNodeImports": "true"
        }
    },
    "typescript-fetch": {
        "generator": "typescript-fetch",
        "package_name": "@jotty/sdk-browser",
        "description": "Jotty SDK for Browser (Fetch API)",
        "output_dir": "sdk/generated/typescript-fetch",
        "additional_properties": {
            "npmName": "@jotty/sdk-browser",
            "npmVersion": "1.0.0",
            "supportsES6": "true"
        }
    },
    "python": {
        "generator": "python",
        "package_name": "jotty-sdk",
        "description": "Jotty SDK for Python",
        "output_dir": "sdk/generated/python",
        "additional_properties": {
            "packageName": "jotty_sdk",
            "packageVersion": "1.0.0",
            "pythonVersion": "3.8"
        }
    },
    "go": {
        "generator": "go",
        "package_name": "github.com/jotty/jotty-sdk-go",
        "description": "Jotty SDK for Go",
        "output_dir": "sdk/generated/go",
        "additional_properties": {
            "packageName": "jotty",
            "packageVersion": "1.0.0",
            "withGoCodegenComment": "true"
        }
    },
    "java": {
        "generator": "java",
        "package_name": "com.jotty.sdk",
        "description": "Jotty SDK for Java",
        "output_dir": "sdk/generated/java",
        "additional_properties": {
            "groupId": "com.jotty",
            "artifactId": "jotty-sdk",
            "artifactVersion": "1.0.0",
            "library": "okhttp-gson",
            "java8": "true"
        }
    },
    "ruby": {
        "generator": "ruby",
        "package_name": "jotty-sdk",
        "description": "Jotty SDK for Ruby",
        "output_dir": "sdk/generated/ruby",
        "additional_properties": {
            "gemName": "jotty_sdk",
            "gemVersion": "1.0.0"
        }
    },
    "php": {
        "generator": "php",
        "package_name": "jotty/sdk",
        "description": "Jotty SDK for PHP",
        "output_dir": "sdk/generated/php",
        "additional_properties": {
            "packageName": "jotty/sdk",
            "composerVendorName": "jotty",
            "composerPackageName": "sdk"
        }
    },
    "swift": {
        "generator": "swift5",
        "package_name": "JottySDK",
        "description": "Jotty SDK for Swift",
        "output_dir": "sdk/generated/swift",
        "additional_properties": {
            "projectName": "JottySDK"
        }
    },
    "kotlin": {
        "generator": "kotlin",
        "package_name": "com.jotty.sdk",
        "description": "Jotty SDK for Kotlin",
        "output_dir": "sdk/generated/kotlin",
        "additional_properties": {
            "packageName": "com.jotty.sdk"
        }
    },
    "rust": {
        "generator": "rust",
        "package_name": "jotty-sdk",
        "description": "Jotty SDK for Rust",
        "output_dir": "sdk/generated/rust",
        "additional_properties": {
            "packageName": "jotty-sdk",
            "packageVersion": "1.0.0"
        }
    },
    "csharp": {
        "generator": "csharp",
        "package_name": "Jotty.SDK",
        "description": "Jotty SDK for C#",
        "output_dir": "sdk/generated/csharp",
        "additional_properties": {
            "packageName": "Jotty.SDK",
            "packageVersion": "1.0.0",
            "targetFramework": "netstandard2.0"
        }
    },
    "dart": {
        "generator": "dart",
        "package_name": "jotty_sdk",
        "description": "Jotty SDK for Dart",
        "output_dir": "sdk/generated/dart",
        "additional_properties": {
            "pubName": "jotty_sdk",
            "pubVersion": "1.0.0"
        }
    }
}


def check_openapi_generator() -> bool:
    """Check if OpenAPI Generator is installed."""
    try:
        result = subprocess.run(
            ["openapi-generator-cli", "version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_openapi_generator() -> None:
    """Install OpenAPI Generator CLI."""
    print("📦 Installing OpenAPI Generator CLI...")
    
    # Try npm first (most common)
    try:
        subprocess.run(
            ["npm", "install", "-g", "@openapitools/openapi-generator-cli"],
            check=True
        )
        print("✅ Installed via npm")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try Docker
    try:
        subprocess.run(
            ["docker", "pull", "openapitools/openapi-generator-cli"],
            check=True
        )
        print("✅ Docker image pulled (use 'docker run' for generation)")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try Homebrew (macOS)
    try:
        subprocess.run(
            ["brew", "install", "openapi-generator"],
            check=True
        )
        print("✅ Installed via Homebrew")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    raise RuntimeError(
        "Could not install OpenAPI Generator. Please install manually:\n"
        "  npm: npm install -g @openapitools/openapi-generator-cli\n"
        "  Docker: docker pull openapitools/openapi-generator-cli\n"
        "  Homebrew: brew install openapi-generator"
    )


def generate_sdk(
    openapi_spec: Path,
    config: Dict[str, any],
    dry_run: bool = False
) -> bool:
    """Generate SDK for a specific language."""
    generator = config["generator"]
    output_dir = Path(config["output_dir"])
    package_name = config["package_name"]
    
    print(f"\n🔧 Generating {generator} SDK...")
    print(f"   Package: {package_name}")
    print(f"   Output: {output_dir}")
    
    # Build additional properties string
    props = []
    for key, value in config.get("additional_properties", {}).items():
        props.append(f"{key}={value}")
    props_str = ",".join(props)
    
    # Clean output directory
    if output_dir.exists() and not dry_run:
        shutil.rmtree(output_dir)
    
    # Build command
    cmd = [
        "openapi-generator-cli", "generate",
        "-i", str(openapi_spec),
        "-g", generator,
        "-o", str(output_dir),
    ]
    
    if props_str:
        cmd.extend(["--additional-properties", props_str])
    
    if dry_run:
        print(f"   [DRY RUN] Would run: {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Generated {generator} SDK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate {generator} SDK")
        print(f"   Error: {e.stderr}")
        return False


def generate_all_sdks(
    openapi_spec: Path,
    languages: Optional[List[str]] = None,
    dry_run: bool = False
) -> Dict[str, bool]:
    """Generate SDKs for all or specified languages."""
    
    if not openapi_spec.exists():
        raise FileNotFoundError(f"OpenAPI spec not found: {openapi_spec}")
    
    if languages is None:
        languages = list(SDK_CONFIGS.keys())
    
    results = {}
    
    for lang in languages:
        if lang not in SDK_CONFIGS:
            print(f"⚠️  Unknown language: {lang}")
            results[lang] = False
            continue
        
        config = SDK_CONFIGS[lang]
        success = generate_sdk(openapi_spec, config, dry_run)
        results[lang] = success
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate multi-language SDKs from OpenAPI specification"
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("sdk/openapi.json"),
        help="Path to OpenAPI specification JSON file"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=list(SDK_CONFIGS.keys()),
        help="Specific languages to generate (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without actually generating"
    )
    parser.add_argument(
        "--install-generator",
        action="store_true",
        help="Install OpenAPI Generator CLI if not found"
    )
    
    args = parser.parse_args()
    
    # Check for OpenAPI Generator
    if not check_openapi_generator():
        if args.install_generator:
            install_openapi_generator()
        else:
            print("❌ OpenAPI Generator CLI not found!")
            print("   Install with: --install-generator")
            print("   Or manually: npm install -g @openapitools/openapi-generator-cli")
            sys.exit(1)
    
    # Generate SDKs
    print(f"📋 Generating SDKs from: {args.spec}")
    
    results = generate_all_sdks(
        args.spec,
        languages=args.languages,
        dry_run=args.dry_run
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 Generation Summary")
    print("="*60)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for lang, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {lang}")
    
    print(f"\n{success_count}/{total_count} SDKs generated successfully")
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
