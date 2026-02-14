"""
MAS-Bench Component Tests
=========================

Unit tests for all MAS-Bench components:
- AndroidDevice + UITreeParser (android-automation skill)
- Hybrid action router (planner executor_type awareness)
- Action-type dimension in TD-Lambda learning
- MASBenchRunner evaluation harness

All tests are mocked — no real Android device or emulator needed.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

# =============================================================================
# Android Automation Tests
# =============================================================================


class TestUITreeParser:
    """Tests for UITreeParser XML-to-JSON conversion."""

    def _get_parser(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "android_tools",
            Path(__file__).resolve().parent.parent / "skills" / "android-automation" / "tools.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.UITreeParser

    def test_parse_simple_button(self):
        """Parse a simple Button element."""
        xml = '''<hierarchy rotation="0">
            <node class="android.widget.Button" text="Submit"
                  bounds="[100,200][300,250]" clickable="true"
                  focusable="true" enabled="true" selected="false"
                  checked="false" scrollable="false" long-clickable="false"
                  content-desc="" resource-id="com.app:id/btn_submit" />
        </hierarchy>'''
        parser = self._get_parser()
        result = parser.parse(xml)

        assert result["node_count"] >= 1
        assert len(result["interactive_elements"]) >= 1

        # Find the button in interactive elements
        btn = result["interactive_elements"][0]
        assert btn["role"] == "button"
        assert btn["name"] == "Submit"
        assert btn["clickable"] is True
        assert btn["center"]["x"] == 200
        assert btn["center"]["y"] == 225

    def test_parse_edittext(self):
        """EditText should map to 'textbox' role."""
        xml = '''<hierarchy rotation="0">
            <node class="android.widget.EditText" text=""
                  bounds="[10,50][500,100]" clickable="true"
                  focusable="true" enabled="true" selected="false"
                  checked="false" scrollable="false" long-clickable="false"
                  content-desc="Search" resource-id="com.app:id/search_input" />
        </hierarchy>'''
        parser = self._get_parser()
        result = parser.parse(xml)

        assert len(result["interactive_elements"]) >= 1
        elem = result["interactive_elements"][0]
        assert elem["role"] == "textbox"
        assert elem["name"] == "Search"  # Falls back to content-desc

    def test_parse_bounds(self):
        """Bounds string parsing."""
        parser = self._get_parser()
        assert parser._parse_bounds("[0,0][1080,2400]") == {
            "left": 0, "top": 0, "right": 1080, "bottom": 2400
        }
        assert parser._parse_bounds("") == {}
        assert parser._parse_bounds("invalid") == {}

    def test_parse_nested_hierarchy(self):
        """Parse nested layout with children."""
        xml = '''<hierarchy rotation="0">
            <node class="android.widget.LinearLayout" bounds="[0,0][1080,2400]"
                  clickable="false" focusable="false" enabled="true"
                  selected="false" checked="false" scrollable="false"
                  long-clickable="false" text="" content-desc="" resource-id="">
                <node class="android.widget.TextView" text="Hello"
                      bounds="[10,10][200,50]" clickable="false" focusable="false"
                      enabled="true" selected="false" checked="false"
                      scrollable="false" long-clickable="false"
                      content-desc="" resource-id="" />
                <node class="android.widget.Button" text="OK"
                      bounds="[10,60][200,100]" clickable="true" focusable="true"
                      enabled="true" selected="false" checked="false"
                      scrollable="false" long-clickable="false"
                      content-desc="" resource-id="com.app:id/ok_btn" />
            </node>
        </hierarchy>'''
        parser = self._get_parser()
        result = parser.parse(xml)

        assert result["node_count"] >= 3
        # Only the button should be interactive
        interactive = result["interactive_elements"]
        assert any(e["name"] == "OK" and e["role"] == "button" for e in interactive)

    def test_max_depth_limits_tree(self):
        """max_depth should limit tree parsing."""
        xml = '''<hierarchy rotation="0">
            <node class="android.widget.FrameLayout" bounds="[0,0][1080,2400]"
                  clickable="false" focusable="false" enabled="true"
                  selected="false" checked="false" scrollable="false"
                  long-clickable="false" text="" content-desc="" resource-id="">
                <node class="android.widget.LinearLayout" bounds="[0,0][1080,2400]"
                      clickable="false" focusable="false" enabled="true"
                      selected="false" checked="false" scrollable="false"
                      long-clickable="false" text="" content-desc="" resource-id="">
                    <node class="android.widget.Button" text="Deep"
                          bounds="[10,10][100,50]" clickable="true" focusable="true"
                          enabled="true" selected="false" checked="false"
                          scrollable="false" long-clickable="false"
                          content-desc="" resource-id="" />
                </node>
            </node>
        </hierarchy>'''
        parser = self._get_parser()

        # depth=1 should not reach the button (depth 2)
        result = parser.parse(xml, max_depth=1)
        deep_buttons = [e for e in result["interactive_elements"] if e["name"] == "Deep"]
        assert len(deep_buttons) == 0

    def test_parse_invalid_xml(self):
        """Invalid XML should return error, not crash."""
        parser = self._get_parser()
        result = parser.parse("<not valid xml>><>")
        assert result["node_count"] == 0
        assert "error" in result

    def test_class_to_role_mapping(self):
        """Verify class-to-role mapping for key widget types."""
        parser = self._get_parser()
        assert parser._CLASS_TO_ROLE["android.widget.CheckBox"] == "checkbox"
        assert parser._CLASS_TO_ROLE["android.widget.Switch"] == "switch"
        assert parser._CLASS_TO_ROLE["android.widget.RecyclerView"] == "list"
        assert parser._CLASS_TO_ROLE["android.webkit.WebView"] == "document"

    def test_resource_id_as_fallback_name(self):
        """When text and desc are empty, resource_id should be used for name."""
        xml = '''<hierarchy rotation="0">
            <node class="android.widget.ImageButton" text=""
                  bounds="[900,10][960,60]" clickable="true" focusable="true"
                  enabled="true" selected="false" checked="false"
                  scrollable="false" long-clickable="false"
                  content-desc="" resource-id="com.app:id/menu_icon" />
        </hierarchy>'''
        parser = self._get_parser()
        result = parser.parse(xml)
        elem = result["interactive_elements"][0]
        assert elem["name"] == "menu_icon"


class TestAndroidDevice:
    """Tests for AndroidDevice class with mocked uiautomator2."""

    def _get_module(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "android_tools",
            Path(__file__).resolve().parent.parent / "skills" / "android-automation" / "tools.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_singleton_pattern(self):
        """AndroidDevice should be a singleton."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        d1 = mod.AndroidDevice.get_instance()
        d2 = mod.AndroidDevice.get_instance()
        assert d1 is d2
        mod.AndroidDevice.reset()

    def test_reset_clears_singleton(self):
        """reset() should clear the singleton."""
        mod = self._get_module()
        d1 = mod.AndroidDevice.get_instance()
        mod.AndroidDevice.reset()
        d2 = mod.AndroidDevice.get_instance()
        assert d1 is not d2
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_tap_calls_device_click(self):
        """tap() should call device.click(x, y)."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        dev._device = mock_device

        result = dev.tap(100, 200)
        mock_device.click.assert_called_once_with(100, 200)
        assert result["x"] == 100
        assert result["y"] == 200
        assert result["action"] == "tap"
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_swipe_calls_device_swipe(self):
        """swipe() should call device.swipe with coordinates."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        dev._device = mock_device

        result = dev.swipe(0, 500, 0, 100, duration=0.3)
        mock_device.swipe.assert_called_once_with(0, 500, 0, 100, duration=0.3)
        assert result["from"] == [0, 500]
        assert result["to"] == [0, 100]
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_type_text(self):
        """type_text() should call device.send_keys."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        dev._device = mock_device

        result = dev.type_text("hello world")
        mock_device.send_keys.assert_called_once_with("hello world")
        assert result["text"] == "hello world"
        assert result["action"] == "type"
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_type_text_clear_first(self):
        """type_text(clear_first=True) should clear then type."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        dev._device = mock_device

        result = dev.type_text("new text", clear_first=True)
        mock_device.clear_text.assert_called_once()
        mock_device.send_keys.assert_called_once_with("new text")
        assert result["cleared"] is True
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_press_key(self):
        """press_key() should call device.press."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        dev._device = mock_device

        result = dev.press_key("home")
        mock_device.press.assert_called_once_with("home")
        assert result["key"] == "home"
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_launch_app(self):
        """launch_app() should call device.app_start."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        dev._device = mock_device

        result = dev.launch_app("com.google.android.youtube")
        mock_device.app_start.assert_called_once_with(
            "com.google.android.youtube", wait=True
        )
        assert result["package"] == "com.google.android.youtube"
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_find_element_found(self):
        """find_element() should return element info when found."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        mock_element = MagicMock()
        mock_element.exists = True
        mock_element.info = {
            "text": "Submit",
            "className": "android.widget.Button",
            "resourceId": "com.app:id/btn",
            "contentDescription": "",
            "bounds": {"left": 100, "top": 200, "right": 300, "bottom": 250},
            "clickable": True,
            "focusable": True,
            "selected": False,
            "checked": False,
            "enabled": True,
        }
        mock_device.return_value = mock_element
        dev._device = mock_device

        result = dev.find_element(text="Submit")
        assert result["found"] is True
        assert result["text"] == "Submit"
        assert result["center"]["x"] == 200
        assert result["center"]["y"] == 225
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_find_element_not_found(self):
        """find_element() should return found=False when not found."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        mock_element = MagicMock()
        mock_element.exists = False
        mock_device.return_value = mock_element
        dev._device = mock_device

        result = dev.find_element(text="Nonexistent")
        assert result["found"] is False
        mod.AndroidDevice.reset()

    @pytest.mark.unit
    def test_scroll_down(self):
        """scroll(down) should swipe from center-low to center-high."""
        mod = self._get_module()
        mod.AndroidDevice.reset()
        dev = mod.AndroidDevice.get_instance()

        mock_device = MagicMock()
        dev._device = mock_device
        dev._screen_size = (1080, 2400)

        result = dev.scroll(direction="down", steps=5)
        mock_device.swipe.assert_called_once()
        assert result["direction"] == "down"
        mod.AndroidDevice.reset()


# =============================================================================
# Hybrid Action Router Tests
# =============================================================================


class TestHybridActionRouter:
    """Tests for executor_type propagation in planner."""

    @pytest.mark.unit
    def test_format_skills_includes_executor_type(self):
        """_format_skills_for_planner should include executor_type when present."""
        from Jotty.core.agents.agentic_planner import TaskPlanner

        planner = TaskPlanner.__new__(TaskPlanner)
        planner._fast_lm = None

        skills = [
            {
                'name': 'android-automation',
                'description': 'Mobile GUI automation',
                'executor_type': 'gui',
                'tools': [{'name': 'tap_tool'}],
            },
            {
                'name': 'http-client',
                'description': 'HTTP requests',
                'executor_type': 'api',
                'tools': [{'name': 'http_get_tool'}],
            },
            {
                'name': 'web-search',
                'description': 'Web search',
                'tools': [{'name': 'search_web_tool'}],
            },
        ]

        # Mock registry to avoid loading real skills
        with patch("Jotty.core.registry.skills_registry.get_skills_registry", return_value=None):
            result = planner._format_skills_for_planner(skills)

        assert result[0]['executor_type'] == 'gui'
        assert result[1]['executor_type'] == 'api'
        # web-search has no executor_type in metadata — should not have it
        assert 'executor_type' not in result[2]

    @pytest.mark.unit
    def test_executor_type_in_planner_signature(self):
        """ExecutionPlanningSignature should mention hybrid routing."""
        from Jotty.core.agents.planner_signatures import ExecutionPlanningSignature
        docstring = ExecutionPlanningSignature.__doc__
        assert "HYBRID ACTION ROUTING" in docstring
        assert "executor_type" in docstring
        assert "ALWAYS prefer API" in docstring


# =============================================================================
# TD-Lambda Action Type Tests
# =============================================================================


class TestActionTypeLearning:
    """Tests for action_type dimension in GroupedValueBaseline."""

    def _make_baseline(self):
        from Jotty.core.learning.td_lambda import GroupedValueBaseline
        return GroupedValueBaseline(ema_alpha=0.3)

    def test_update_creates_composite_key(self):
        """update_group with action_type should create composite key."""
        bl = self._make_baseline()
        bl.update_group("research", 0.8, action_type="api")
        assert "research:api" in bl.group_baselines
        assert "research:api" in bl.group_counts
        assert bl.group_counts["research:api"] == 1

    def test_get_baseline_prefers_composite(self):
        """get_baseline should prefer composite key when enough samples."""
        bl = self._make_baseline()
        # Build up 3+ samples for composite key
        for _ in range(4):
            bl.update_group("automation", 0.9, action_type="api")
        for _ in range(4):
            bl.update_group("automation", 0.3, action_type="gui")

        api_baseline = bl.get_baseline("automation", action_type="api")
        gui_baseline = bl.get_baseline("automation", action_type="gui")

        # API should have higher baseline than GUI
        assert api_baseline > gui_baseline

    def test_get_baseline_falls_back_without_action_type(self):
        """get_baseline without action_type should use task_type key."""
        bl = self._make_baseline()
        for _ in range(4):
            bl.update_group("analysis", 0.7)

        baseline = bl.get_baseline("analysis")
        assert baseline > 0.5  # Should reflect the 0.7 rewards

    def test_get_best_action_type(self):
        """get_best_action_type should return the highest-performing strategy."""
        bl = self._make_baseline()
        # API performs well
        for _ in range(5):
            bl.update_group("navigation", 0.95, action_type="api")
        # GUI performs poorly
        for _ in range(5):
            bl.update_group("navigation", 0.4, action_type="gui")

        best = bl.get_best_action_type("navigation")
        assert best == "api"

    def test_get_best_action_type_insufficient_data(self):
        """get_best_action_type should return None with too few samples."""
        bl = self._make_baseline()
        bl.update_group("new_task", 0.5, action_type="api")
        assert bl.get_best_action_type("new_task") is None

    def test_update_without_action_type_backward_compatible(self):
        """update_group without action_type should work as before."""
        bl = self._make_baseline()
        bl.update_group("research", 0.8)
        assert "research" in bl.group_baselines
        assert "research" in bl.group_counts
        # No composite keys should exist
        composite_keys = [k for k in bl.group_baselines if ":" in k and not k.startswith("domain:")]
        assert len(composite_keys) == 0

    def test_both_base_and_composite_updated(self):
        """update_group with action_type should update both base and composite."""
        bl = self._make_baseline()
        bl.update_group("research", 0.8, action_type="deeplink")

        assert "research" in bl.group_baselines
        assert "research:deeplink" in bl.group_baselines
        assert bl.group_counts["research"] == 1
        assert bl.group_counts["research:deeplink"] == 1


# =============================================================================
# MAS-Bench Runner Tests
# =============================================================================


class TestMASBenchResult:
    """Tests for MASBenchResult dataclass."""

    def test_step_ratio(self):
        """step_ratio should be total_steps / optimal_steps."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult
        r = MASBenchResult(task_id="t1", total_steps=10, optimal_steps=5)
        assert r.step_ratio == 2.0

    def test_step_ratio_zero_optimal(self):
        """step_ratio should handle optimal_steps=0."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult
        r = MASBenchResult(task_id="t1", total_steps=5, optimal_steps=0)
        assert r.step_ratio == 5.0

    def test_shortcut_success_rate(self):
        """SSR should be successes / calls."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult
        r = MASBenchResult(task_id="t1", shortcut_calls=10, shortcut_successes=8)
        assert r.shortcut_success_rate == 0.8

    def test_gui_shortcut_ratio(self):
        """GSAR should be shortcut / gui steps."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult
        r = MASBenchResult(task_id="t1", gui_steps=2, shortcut_steps=6)
        assert r.gui_shortcut_ratio == 3.0

    def test_to_dict(self):
        """to_dict should include all metrics."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult
        r = MASBenchResult(task_id="t1", success=True, total_steps=3,
                           optimal_steps=2, gui_steps=1, shortcut_steps=2)
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["success"] is True
        assert d["step_ratio"] == 1.5
        assert d["gui_shortcut_ratio"] == 2.0


class TestMASBenchRunner:
    """Tests for MASBenchRunner evaluation harness."""

    def _make_runner(self):
        from Jotty.core.orchestration.benchmarking import MASBenchRunner
        return MASBenchRunner()

    def test_add_task(self):
        """add_task should register tasks."""
        runner = self._make_runner()
        runner.add_task("t1", "Search on YouTube", app="com.youtube", optimal_steps=3)
        assert len(runner.tasks) == 1
        assert runner.tasks[0]["task_id"] == "t1"

    def test_classify_step_gui(self):
        """GUI skills should be classified as 'gui'."""
        runner = self._make_runner()
        assert runner.classify_step("android-automation") == "gui"
        assert runner.classify_step("browser-automation") == "gui"

    def test_classify_step_shortcut(self):
        """API skills should be classified as 'shortcut'."""
        runner = self._make_runner()
        assert runner.classify_step("http-client") == "shortcut"
        assert runner.classify_step("pmi-market-data") == "shortcut"
        assert runner.classify_step("web-search") == "shortcut"

    def test_classify_step_unknown_defaults_shortcut(self):
        """Unknown skills default to 'shortcut'."""
        runner = self._make_runner()
        assert runner.classify_step("custom-skill") == "shortcut"

    def test_evaluate_execution(self):
        """evaluate_execution should compute all metrics."""
        runner = self._make_runner()
        task = {
            'task_id': 't1',
            'description': 'Add laptop to cart',
            'app': 'com.amazon',
            'optimal_steps': 3,
            'difficulty': 2,
            'cross_app': False,
        }
        steps = [
            {'skill_name': 'http-client', 'status': 'completed'},
            {'skill_name': 'android-automation', 'status': 'completed'},
            {'skill_name': 'http-client', 'status': 'completed'},
        ]
        result = runner.evaluate_execution(
            task, steps, success=True, execution_time=15.0, token_cost=2.5
        )

        assert result.success is True
        assert result.total_steps == 3
        assert result.gui_steps == 1
        assert result.shortcut_steps == 2
        assert result.shortcut_calls == 2
        assert result.shortcut_successes == 2
        assert result.step_ratio == 1.0
        assert result.gui_shortcut_ratio == 2.0

    def test_evaluate_execution_with_failures(self):
        """Shortcut failures should reduce SSR."""
        runner = self._make_runner()
        task = {'task_id': 't2', 'description': 'test', 'optimal_steps': 2}
        steps = [
            {'skill_name': 'http-client', 'status': 'completed'},
            {'skill_name': 'http-client', 'status': 'failed'},
            {'skill_name': 'android-automation', 'status': 'completed'},
        ]
        result = runner.evaluate_execution(
            task, steps, success=False, execution_time=30.0, token_cost=5.0
        )

        assert result.shortcut_calls == 2
        assert result.shortcut_successes == 1
        assert result.shortcut_success_rate == 0.5

    def test_aggregate_metrics(self):
        """compute_aggregate_metrics should compute all 7 MAS-Bench metrics."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult, MASBenchRunner

        results = [
            MASBenchResult(task_id="t1", success=True, total_steps=4,
                           optimal_steps=3, gui_steps=1, shortcut_steps=3,
                           shortcut_calls=3, shortcut_successes=3,
                           execution_time_sec=10.0, token_cost_k=2.0,
                           difficulty_level=1),
            MASBenchResult(task_id="t2", success=False, total_steps=8,
                           optimal_steps=5, gui_steps=5, shortcut_steps=3,
                           shortcut_calls=3, shortcut_successes=2,
                           execution_time_sec=30.0, token_cost_k=5.0,
                           difficulty_level=2),
        ]

        metrics = MASBenchRunner.compute_aggregate_metrics(results)

        assert metrics['SR'] == 0.5  # 1/2 success
        assert metrics['SSR'] == 5 / 6  # 5/6 shortcut successes
        assert metrics['MS'] == 6.0  # (4+8)/2
        assert metrics['total_tasks'] == 2

    def test_aggregate_empty_results(self):
        """Empty results should return zeroed metrics."""
        from Jotty.core.orchestration.benchmarking import MASBenchRunner
        metrics = MASBenchRunner.compute_aggregate_metrics([])
        assert metrics['SR'] == 0
        assert metrics['MS'] == 0

    def test_aggregate_by_difficulty(self):
        """Should break down SR by difficulty level."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult, MASBenchRunner

        results = [
            MASBenchResult(task_id="t1", success=True, difficulty_level=1),
            MASBenchResult(task_id="t2", success=True, difficulty_level=1),
            MASBenchResult(task_id="t3", success=False, difficulty_level=2),
        ]

        metrics = MASBenchRunner.compute_aggregate_metrics(results)
        assert metrics['SR_L1'] == 1.0  # 2/2
        assert metrics['SR_L2'] == 0.0  # 0/1

    def test_aggregate_single_vs_cross_app(self):
        """Should break down SR by single-app vs cross-app."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult, MASBenchRunner

        results = [
            MASBenchResult(task_id="t1", success=True, is_cross_app=False),
            MASBenchResult(task_id="t2", success=False, is_cross_app=True),
            MASBenchResult(task_id="t3", success=True, is_cross_app=True),
        ]

        metrics = MASBenchRunner.compute_aggregate_metrics(results)
        assert metrics['SR_single_app'] == 1.0
        assert metrics['SR_cross_app'] == 0.5

    def test_summary_format(self):
        """summary() should return formatted string."""
        from Jotty.core.orchestration.benchmarking import MASBenchResult, MASBenchRunner

        runner = MASBenchRunner()
        results = [
            MASBenchResult(task_id="t1", success=True, total_steps=3,
                           optimal_steps=3, shortcut_calls=2, shortcut_successes=2,
                           execution_time_sec=5.0, token_cost_k=1.0),
        ]

        summary = runner.summary(results)
        assert "MAS-Bench Results" in summary
        assert "Success Rate" in summary
        assert "100.0%" in summary


# =============================================================================
# Tool Schema Output Tests for Android Tools
# =============================================================================


class TestAndroidToolSchemas:
    """Verify all android-automation tools have proper docstring schemas."""

    def _get_tools(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "android_tools",
            Path(__file__).resolve().parent.parent / "skills" / "android-automation" / "tools.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {name: getattr(mod, name) for name in mod.__all__}

    def test_all_tools_have_returns_section(self):
        """Every android tool should have a Returns section in docstring."""
        from Jotty.core.agents._execution_types import ToolSchema
        tools = self._get_tools()
        missing = []
        for name, func in tools.items():
            schema = ToolSchema.from_tool_function(func, name)
            if not schema.outputs:
                missing.append(name)
        assert not missing, f"Tools missing output declarations: {missing}"

    def test_tap_tool_schema(self):
        """tap_tool should have x, y required params and x, y, action outputs."""
        from Jotty.core.agents._execution_types import ToolSchema
        tools = self._get_tools()
        schema = ToolSchema.from_tool_function(tools["tap_tool"], "tap_tool")
        assert "x" in schema.required_param_names
        assert "y" in schema.required_param_names
        assert "x" in schema.output_field_names
        assert "action" in schema.output_field_names

    def test_screenshot_tool_schema(self):
        """screenshot_tool should declare image_base64, width, height outputs."""
        from Jotty.core.agents._execution_types import ToolSchema
        tools = self._get_tools()
        schema = ToolSchema.from_tool_function(tools["screenshot_tool"], "screenshot_tool")
        assert "image_base64" in schema.output_field_names
        assert "width" in schema.output_field_names
        assert "height" in schema.output_field_names

    def test_get_ui_tree_tool_schema(self):
        """get_ui_tree_tool should declare tree, node_count, interactive_elements outputs."""
        from Jotty.core.agents._execution_types import ToolSchema
        tools = self._get_tools()
        schema = ToolSchema.from_tool_function(tools["get_ui_tree_tool"], "get_ui_tree_tool")
        assert "tree" in schema.output_field_names
        assert "node_count" in schema.output_field_names
        assert "interactive_elements" in schema.output_field_names

    def test_tool_count(self):
        """Should have 16 android tools."""
        tools = self._get_tools()
        assert len(tools) == 16
