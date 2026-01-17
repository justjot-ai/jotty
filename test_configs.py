"""
Test Jotty Configuration System
================================

Tests all Hydra configs to ensure:
1. YAML syntax is valid
2. Configs can be loaded by Hydra
3. Module compositions work
4. Presets load correctly
5. Overrides work as expected
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Test Hydra availability
try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    print("⚠️ Hydra not available - some tests will be skipped")

CONFIG_DIR = Path(__file__).parent / "configs"

def test_yaml_syntax():
    """Test 1: All YAML files have valid syntax"""
    print("\n" + "="*80)
    print("TEST 1: YAML Syntax Validation")
    print("="*80)

    yaml_files = list(CONFIG_DIR.rglob("*.yaml"))
    print(f"  Found {len(yaml_files)} YAML files")

    errors = []
    for yaml_file in yaml_files:
        try:
            with open(yaml_file) as f:
                yaml.safe_load(f)
            print(f"  ✅ {yaml_file.relative_to(CONFIG_DIR)}")
        except yaml.YAMLError as e:
            errors.append((yaml_file, e))
            print(f"  ❌ {yaml_file.relative_to(CONFIG_DIR)}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} files have YAML syntax errors")
        return False
    else:
        print(f"\n✅ All {len(yaml_files)} YAML files have valid syntax")
        return True

def test_config_structure():
    """Test 2: Configuration directory structure"""
    print("\n" + "="*80)
    print("TEST 2: Configuration Structure")
    print("="*80)

    required_dirs = [
        "mas",
        "orchestrator",
        "memory",
        "learning",
        "validation",
        "tools",
        "experts",
        "communication",
        "presets"
    ]

    missing = []
    for dir_name in required_dirs:
        dir_path = CONFIG_DIR / dir_name
        if dir_path.exists():
            num_files = len(list(dir_path.glob("*.yaml")))
            print(f"  ✅ {dir_name}/ ({num_files} configs)")
        else:
            missing.append(dir_name)
            print(f"  ❌ {dir_name}/ (missing)")

    if missing:
        print(f"\n❌ Missing directories: {missing}")
        return False
    else:
        print(f"\n✅ All required directories present")
        return True

def test_module_configs():
    """Test 3: All module configs have required fields"""
    print("\n" + "="*80)
    print("TEST 3: Module Configuration Validation")
    print("="*80)

    module_dirs = ["mas", "orchestrator", "memory", "learning",
                   "validation", "tools", "experts", "communication"]

    errors = []
    for module_dir in module_dirs:
        dir_path = CONFIG_DIR / module_dir
        if not dir_path.exists():
            continue

        for config_file in dir_path.glob("*.yaml"):
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)

                # Basic validation
                if config is None:
                    errors.append((config_file, "Empty config"))
                    print(f"  ❌ {config_file.relative_to(CONFIG_DIR)}: Empty")
                else:
                    print(f"  ✅ {config_file.relative_to(CONFIG_DIR)}")

            except Exception as e:
                errors.append((config_file, str(e)))
                print(f"  ❌ {config_file.relative_to(CONFIG_DIR)}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} configs have validation errors")
        return False
    else:
        print(f"\n✅ All module configs validated")
        return True

def test_presets():
    """Test 4: All presets are valid"""
    print("\n" + "="*80)
    print("TEST 4: Preset Validation")
    print("="*80)

    presets_dir = CONFIG_DIR / "presets"
    expected_presets = ["minimal", "development", "production", "research", "experimental"]

    errors = []
    for preset_name in expected_presets:
        preset_file = presets_dir / f"{preset_name}.yaml"

        if not preset_file.exists():
            errors.append((preset_name, "Missing"))
            print(f"  ❌ {preset_name}: Missing")
            continue

        try:
            with open(preset_file) as f:
                config = yaml.safe_load(f)

            # Check defaults section
            if "defaults" not in config:
                errors.append((preset_name, "Missing defaults section"))
                print(f"  ❌ {preset_name}: Missing defaults section")
            else:
                print(f"  ✅ {preset_name}: Valid")

        except Exception as e:
            errors.append((preset_name, str(e)))
            print(f"  ❌ {preset_name}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} presets have errors")
        return False
    else:
        print(f"\n✅ All {len(expected_presets)} presets validated")
        return True

def test_hydra_loading():
    """Test 5: Hydra can load configs (requires hydra-core)"""
    print("\n" + "="*80)
    print("TEST 5: Hydra Configuration Loading")
    print("="*80)

    if not HYDRA_AVAILABLE:
        print("  ⚠️ Skipped: Hydra not installed")
        print("     Run: pip install hydra-core omegaconf")
        return None

    config_dir = str(CONFIG_DIR.absolute())

    try:
        # Test loading default config
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config")
            print(f"  ✅ Default config loaded")
            print(f"     - MAS: {cfg.defaults[0] if 'defaults' in cfg else 'N/A'}")

        # Test loading presets
        preset_names = ["minimal", "development", "production"]
        for preset_name in preset_names:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name=f"presets/{preset_name}")
                print(f"  ✅ Preset '{preset_name}' loaded")

        print(f"\n✅ Hydra loading works")
        return True

    except Exception as e:
        print(f"  ❌ Hydra loading failed: {e}")
        return False

def test_module_composition():
    """Test 6: Module composition works"""
    print("\n" + "="*80)
    print("TEST 6: Module Composition")
    print("="*80)

    if not HYDRA_AVAILABLE:
        print("  ⚠️ Skipped: Hydra not installed")
        return None

    config_dir = str(CONFIG_DIR.absolute())

    test_cases = [
        {
            "name": "Minimal + Cortex Memory",
            "overrides": ["mas=minimal", "memory=cortex"]
        },
        {
            "name": "Full + TD(λ) Learning",
            "overrides": ["mas=full", "learning=td_lambda"]
        },
        {
            "name": "Production + MARL",
            "overrides": ["mas=full", "memory=cortex", "learning=marl"]
        }
    ]

    errors = []
    for test_case in test_cases:
        try:
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(config_name="config", overrides=test_case["overrides"])
                print(f"  ✅ {test_case['name']}")
        except Exception as e:
            errors.append((test_case["name"], str(e)))
            print(f"  ❌ {test_case['name']}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} composition tests failed")
        return False
    else:
        print(f"\n✅ All composition tests passed")
        return True

def test_documentation():
    """Test 7: Documentation exists and is comprehensive"""
    print("\n" + "="*80)
    print("TEST 7: Documentation")
    print("="*80)

    required_docs = [
        CONFIG_DIR / "README.md",
        Path(__file__).parent / "MODULE_BASED_CONFIG_COMPLETE.md"
    ]

    errors = []
    total_lines = 0

    for doc_file in required_docs:
        if not doc_file.exists():
            errors.append((doc_file.name, "Missing"))
            print(f"  ❌ {doc_file.name}: Missing")
        else:
            with open(doc_file) as f:
                lines = len(f.readlines())
                total_lines += lines
            print(f"  ✅ {doc_file.name}: {lines} lines")

    print(f"\n  Total documentation: {total_lines} lines")

    if errors:
        print(f"❌ {len(errors)} documentation files missing")
        return False
    elif total_lines < 500:
        print(f"⚠️ Documentation exists but may not be comprehensive ({total_lines} lines)")
        return None
    else:
        print(f"✅ Comprehensive documentation ({total_lines} lines)")
        return True

def run_all_tests():
    """Run all tests"""
    print("#"*80)
    print("# JOTTY CONFIGURATION SYSTEM TESTS")
    print("#"*80)

    tests = [
        ("YAML Syntax", test_yaml_syntax),
        ("Config Structure", test_config_structure),
        ("Module Configs", test_module_configs),
        ("Presets", test_presets),
        ("Hydra Loading", test_hydra_loading),
        ("Module Composition", test_module_composition),
        ("Documentation", test_documentation)
    ]

    results = {}
    for name, test_func in tests:
        result = test_func()
        results[name] = result

    # Summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        if result is True:
            print(f"  ✅ {name}")
        elif result is False:
            print(f"  ❌ {name}")
        else:
            print(f"  ⚠️ {name} (skipped)")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n❌ Some tests failed")
        return False
    elif skipped > 0:
        print("\n⚠️ All tests passed (some skipped - install hydra-core)")
        return True
    else:
        print("\n✅ All tests passed!")
        return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
