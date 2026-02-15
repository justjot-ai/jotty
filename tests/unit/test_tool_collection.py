"""
Tool Collection Unit Tests
============================

Comprehensive unit tests for ToolCollection class.
All tests use mocks -- no real file I/O, no network, no HuggingFace Hub, no MCP.
Runs offline and fast (<1s each).

Covers:
- ToolCollection initialization and basic properties
- __len__, __iter__, __getitem__ dunder methods
- from_local() with collection.json and skill directories
- from_hub() with mocked HuggingFace Hub
- from_mcp() context manager with mocked MCP
- save_to_local() serialization
- to_skill_definitions() with dict tools and tool objects
- list_tools() with dict and object tools
- Error handling and edge cases
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

# Try importing the module under test
try:
    from Jotty.core.capabilities.registry.tool_collection import (
        HUGGINGFACE_HUB_AVAILABLE,
        MCP_AVAILABLE,
        ToolCollection,
    )

    TOOL_COLLECTION_AVAILABLE = True
except ImportError:
    TOOL_COLLECTION_AVAILABLE = False

skip_no_module = pytest.mark.skipif(
    not TOOL_COLLECTION_AVAILABLE, reason="tool_collection module not importable"
)


# =============================================================================
# Helpers
# =============================================================================


def _make_tool_dict(name="test_tool", description="A test tool", source="test"):
    """Create a minimal tool dict for testing."""
    return {
        "name": name,
        "description": description,
        "source": source,
    }


def _make_tool_dict_with_forward(name="exec_tool", description="Executable tool"):
    """Create a tool dict with a forward callable."""
    return {
        "name": name,
        "description": description,
        "forward": lambda x: f"Result: {x}",
    }


def _make_tool_dict_with_execute(name="exec_tool2", description="Executable tool 2"):
    """Create a tool dict with an execute callable."""
    return {
        "name": name,
        "description": description,
        "execute": lambda params: params.get("x", 0) * 2,
    }


def _make_tool_object(name="obj_tool", description="Object tool"):
    """Create a mock tool object with name and forward attributes."""
    tool = Mock()
    tool.name = name
    tool.description = description
    tool.forward = Mock(return_value="forwarded")
    return tool


# =============================================================================
# TestToolCollectionInit
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionInit:
    """Tests for ToolCollection.__init__ and basic properties."""

    def test_init_with_empty_tools(self):
        """ToolCollection initializes with an empty tools list."""
        collection = ToolCollection(tools=[], source="test")
        assert collection.tools == []
        assert collection.source == "test"
        assert collection.metadata == {}

    def test_init_with_tools(self):
        """ToolCollection stores tools and source correctly."""
        tools = [_make_tool_dict("a"), _make_tool_dict("b")]
        collection = ToolCollection(tools=tools, source="local")
        assert len(collection.tools) == 2
        assert collection.source == "local"

    def test_init_default_source(self):
        """Default source is 'local'."""
        collection = ToolCollection(tools=[])
        assert collection.source == "local"

    def test_init_with_metadata(self):
        """Metadata dict is stored correctly."""
        meta = {"version": "1.0", "author": "tester"}
        collection = ToolCollection(tools=[], metadata=meta)
        assert collection.metadata == meta

    def test_init_none_metadata_becomes_empty_dict(self):
        """None metadata defaults to empty dict."""
        collection = ToolCollection(tools=[], metadata=None)
        assert collection.metadata == {}

    def test_converted_tools_initially_none(self):
        """Internal _converted_tools cache starts as None."""
        collection = ToolCollection(tools=[])
        assert collection._converted_tools is None


# =============================================================================
# TestToolCollectionDunders
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionDunders:
    """Tests for __len__, __iter__, __getitem__."""

    def test_len_empty(self):
        """Empty collection has length 0."""
        collection = ToolCollection(tools=[])
        assert len(collection) == 0

    def test_len_with_tools(self):
        """Length matches number of tools."""
        tools = [_make_tool_dict() for _ in range(5)]
        collection = ToolCollection(tools=tools)
        assert len(collection) == 5

    def test_iter(self):
        """Iteration yields all tools in order."""
        tools = [_make_tool_dict(f"tool_{i}") for i in range(3)]
        collection = ToolCollection(tools=tools)
        iterated = list(collection)
        assert len(iterated) == 3
        assert iterated[0]["name"] == "tool_0"
        assert iterated[2]["name"] == "tool_2"

    def test_getitem(self):
        """Index access returns the correct tool."""
        tools = [_make_tool_dict("first"), _make_tool_dict("second")]
        collection = ToolCollection(tools=tools)
        assert collection[0]["name"] == "first"
        assert collection[1]["name"] == "second"

    def test_getitem_negative_index(self):
        """Negative index works like a normal list."""
        tools = [_make_tool_dict("a"), _make_tool_dict("b")]
        collection = ToolCollection(tools=tools)
        assert collection[-1]["name"] == "b"

    def test_getitem_out_of_range(self):
        """Out-of-range index raises IndexError."""
        collection = ToolCollection(tools=[_make_tool_dict()])
        with pytest.raises(IndexError):
            _ = collection[5]


# =============================================================================
# TestToolCollectionListTools
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionListTools:
    """Tests for list_tools() method."""

    def test_list_tools_empty(self):
        """Empty collection returns empty list."""
        collection = ToolCollection(tools=[])
        assert collection.list_tools() == []

    def test_list_tools_dict_tools(self):
        """Dict tools are listed with name, description, source."""
        tools = [
            {"name": "t1", "description": "desc1", "source": "hub"},
            {"name": "t2", "description": "desc2"},
        ]
        collection = ToolCollection(tools=tools, source="local")
        result = collection.list_tools()
        assert len(result) == 2
        assert result[0]["name"] == "t1"
        assert result[0]["source"] == "hub"
        # Second tool has no explicit source, falls back to collection source
        assert result[1]["source"] == "local"

    def test_list_tools_object_tools(self):
        """Object tools (with name/description attrs) are listed correctly."""
        tool_obj = _make_tool_object("my_tool", "my desc")
        collection = ToolCollection(tools=[tool_obj], source="mcp")
        result = collection.list_tools()
        assert len(result) == 1
        assert result[0]["name"] == "my_tool"
        assert result[0]["description"] == "my desc"
        assert result[0]["source"] == "mcp"

    def test_list_tools_mixed(self):
        """Mixed dict and object tools are both listed."""
        tools = [
            _make_tool_dict("dict_tool"),
            _make_tool_object("obj_tool"),
        ]
        collection = ToolCollection(tools=tools, source="test")
        result = collection.list_tools()
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert "dict_tool" in names
        assert "obj_tool" in names

    def test_list_tools_missing_name_defaults_to_unknown(self):
        """Tool dict without name shows 'unknown'."""
        collection = ToolCollection(tools=[{"description": "no name"}])
        result = collection.list_tools()
        assert result[0]["name"] == "unknown"


# =============================================================================
# TestToolCollectionFromLocal
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionFromLocal:
    """Tests for from_local() class method."""

    def test_from_local_nonexistent_path_raises(self):
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ToolCollection.from_local("/nonexistent/path/to/collection")

    def test_from_local_collection_json(self, tmp_path):
        """Loads tools from collection.json in directory."""
        tools_data = {
            "tools": [
                {"name": "tool_a", "description": "Tool A"},
                {"name": "tool_b", "description": "Tool B"},
            ]
        }
        collection_json = tmp_path / "collection.json"
        collection_json.write_text(json.dumps(tools_data))

        collection = ToolCollection.from_local(str(tmp_path))
        assert collection.source == "local"
        assert len(collection) == 2
        assert collection.metadata["path"] == str(tmp_path)
        assert collection.metadata["tool_count"] == 2

    def test_from_local_empty_collection_json(self, tmp_path):
        """Empty tools array in collection.json yields empty collection."""
        collection_json = tmp_path / "collection.json"
        collection_json.write_text(json.dumps({"tools": []}))

        collection = ToolCollection.from_local(str(tmp_path))
        assert len(collection) == 0

    def test_from_local_no_collection_json_uses_skills_registry(self, tmp_path):
        """Without collection.json, falls back to SkillsRegistry scan."""
        mock_registry = MagicMock()
        mock_registry.loaded_skills = {}
        mock_registry.init = MagicMock()

        with patch(
            "Jotty.core.registry.skills_registry.SkillsRegistry",
            return_value=mock_registry,
        ):
            collection = ToolCollection.from_local(str(tmp_path))
            mock_registry.init.assert_called_once()
            assert len(collection) == 0

    def test_from_local_skills_registry_with_skills(self, tmp_path):
        """SkillsRegistry fallback converts loaded skills to tool dicts."""
        mock_skill = MagicMock()
        mock_skill.description = "A skill"
        mock_skill.tools = {"do_thing": lambda: None}

        mock_registry = MagicMock()
        mock_registry.loaded_skills = {"my_skill": mock_skill}
        mock_registry.init = MagicMock()

        with patch(
            "Jotty.core.registry.skills_registry.SkillsRegistry",
            return_value=mock_registry,
        ):
            collection = ToolCollection.from_local(str(tmp_path))
            assert len(collection) == 1
            assert collection.tools[0]["name"] == "my_skill"
            assert collection.tools[0]["source"] == "local"

    def test_from_local_accepts_path_object(self, tmp_path):
        """Accepts pathlib.Path as well as str."""
        collection_json = tmp_path / "collection.json"
        collection_json.write_text(json.dumps({"tools": [{"name": "x"}]}))

        collection = ToolCollection.from_local(tmp_path)
        assert len(collection) == 1


# =============================================================================
# TestToolCollectionFromHub
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionFromHub:
    """Tests for from_hub() class method."""

    def test_from_hub_without_huggingface_raises(self):
        """Raises ImportError when huggingface_hub is not available."""
        with patch("Jotty.core.registry.tool_collection.HUGGINGFACE_HUB_AVAILABLE", False):
            with pytest.raises(ImportError, match="huggingface_hub not available"):
                ToolCollection.from_hub("some-slug", trust_remote_code=True)

    def test_from_hub_without_trust_remote_code_raises(self):
        """Raises ValueError when trust_remote_code=False."""
        with patch("Jotty.core.registry.tool_collection.HUGGINGFACE_HUB_AVAILABLE", True):
            with pytest.raises(ValueError, match="trust_remote_code=True"):
                ToolCollection.from_hub("some-slug", trust_remote_code=False)

    def test_from_hub_loads_collection_tools(self):
        """Successfully loads tools from mocked Hub collection."""
        mock_item = Mock()
        mock_item.item_id = "user/tool-space"
        mock_item.item_type = "space"

        mock_collection = Mock()
        mock_collection.items = [mock_item]

        mock_tool_dict = {
            "name": "hub_tool",
            "description": "A hub tool",
            "forward": Mock(),
            "tool_object": Mock(),
            "source": "hub",
            "repo_id": "user/tool-space",
        }

        with (
            patch("Jotty.core.registry.tool_collection.HUGGINGFACE_HUB_AVAILABLE", True),
            patch(
                "Jotty.core.registry.tool_collection.get_collection",
                return_value=mock_collection,
            ),
            patch.object(ToolCollection, "_load_tool_from_hub_space", return_value=mock_tool_dict),
        ):
            collection = ToolCollection.from_hub("user/my-collection", trust_remote_code=True)
            assert collection.source == "hub"
            assert len(collection) == 1
            assert collection.metadata["collection_slug"] == "user/my-collection"

    def test_from_hub_skips_non_space_items(self):
        """Non-space items in Hub collection are ignored."""
        model_item = Mock()
        model_item.item_id = "user/my-model"
        model_item.item_type = "model"

        space_item = Mock()
        space_item.item_id = "user/my-space"
        space_item.item_type = "space"

        mock_collection = Mock()
        mock_collection.items = [model_item, space_item]

        with (
            patch("Jotty.core.registry.tool_collection.HUGGINGFACE_HUB_AVAILABLE", True),
            patch(
                "Jotty.core.registry.tool_collection.get_collection",
                return_value=mock_collection,
            ),
            patch.object(
                ToolCollection,
                "_load_tool_from_hub_space",
                return_value={"name": "tool", "description": "d"},
            ) as mock_load,
        ):
            collection = ToolCollection.from_hub("user/collection", trust_remote_code=True)
            # Only the space item should trigger a load
            mock_load.assert_called_once_with("user/my-space", None, True)

    def test_from_hub_handles_load_failure_gracefully(self):
        """Failed individual tool loads are skipped, not fatal."""
        space_item = Mock()
        space_item.item_id = "user/broken-space"
        space_item.item_type = "space"

        mock_collection = Mock()
        mock_collection.items = [space_item]

        with (
            patch("Jotty.core.registry.tool_collection.HUGGINGFACE_HUB_AVAILABLE", True),
            patch(
                "Jotty.core.registry.tool_collection.get_collection",
                return_value=mock_collection,
            ),
            patch.object(
                ToolCollection,
                "_load_tool_from_hub_space",
                side_effect=RuntimeError("load failed"),
            ),
        ):
            collection = ToolCollection.from_hub("user/collection", trust_remote_code=True)
            assert len(collection) == 0

    def test_from_hub_with_token(self):
        """Token is passed to get_collection."""
        mock_collection = Mock()
        mock_collection.items = []

        with (
            patch("Jotty.core.registry.tool_collection.HUGGINGFACE_HUB_AVAILABLE", True),
            patch(
                "Jotty.core.registry.tool_collection.get_collection",
                return_value=mock_collection,
            ) as mock_get,
        ):
            ToolCollection.from_hub("slug", token="hf_test_token", trust_remote_code=True)
            mock_get.assert_called_once_with("slug", token="hf_test_token")


# =============================================================================
# TestToolCollectionFromMCP
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionFromMCP:
    """Tests for from_mcp() context manager."""

    def test_from_mcp_without_mcp_raises(self):
        """Raises ImportError when MCP is not available."""
        with patch("Jotty.core.registry.tool_collection.MCP_AVAILABLE", False):
            with pytest.raises(ImportError, match="MCP not available"):
                with ToolCollection.from_mcp(Mock(), trust_remote_code=True):
                    pass

    def test_from_mcp_without_trust_remote_code_raises(self):
        """Raises ValueError when trust_remote_code=False."""
        with patch("Jotty.core.registry.tool_collection.MCP_AVAILABLE", True):
            with pytest.raises(ValueError, match="trust_remote_code=True"):
                with ToolCollection.from_mcp(Mock(), trust_remote_code=False):
                    pass

    def test_from_mcp_with_mcpadapt(self):
        """Loads tools via mcpadapt adapter when available."""
        mock_tool = Mock()
        mock_tool.name = "mcp_tool"
        mock_tool.description = "An MCP tool"
        mock_tool.forward = Mock()

        mock_adapt_cm = MagicMock()
        mock_adapt_cm.__enter__ = Mock(return_value=[mock_tool])
        mock_adapt_cm.__exit__ = Mock(return_value=False)

        mock_mcpadapt = Mock(return_value=mock_adapt_cm)
        mock_adapter = Mock()

        with (
            patch("Jotty.core.registry.tool_collection.MCP_AVAILABLE", True),
            patch.dict(
                "sys.modules",
                {
                    "mcpadapt": Mock(),
                    "mcpadapt.core": Mock(MCPAdapt=mock_mcpadapt),
                    "mcpadapt.oagents_adapter": Mock(
                        oagentsAdapter=Mock(return_value=mock_adapter)
                    ),
                },
            ),
        ):
            with ToolCollection.from_mcp(Mock(), trust_remote_code=True) as collection:
                assert collection.source == "mcp"
                assert len(collection) == 1
                assert collection.tools[0]["name"] == "mcp_tool"


# =============================================================================
# TestToolCollectionSaveToLocal
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionSaveToLocal:
    """Tests for save_to_local() method."""

    def test_save_creates_directory(self, tmp_path):
        """Creates output directory if it does not exist."""
        output_dir = tmp_path / "new_dir" / "collection"
        collection = ToolCollection(tools=[], source="test")
        collection.save_to_local(str(output_dir))
        assert output_dir.exists()

    def test_save_creates_collection_json(self, tmp_path):
        """Writes collection.json file."""
        collection = ToolCollection(
            tools=[_make_tool_dict("my_tool", "desc")],
            source="hub",
            metadata={"key": "value"},
        )
        collection.save_to_local(str(tmp_path))

        collection_json = tmp_path / "collection.json"
        assert collection_json.exists()

        data = json.loads(collection_json.read_text())
        assert data["source"] == "hub"
        assert data["metadata"]["key"] == "value"
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "my_tool"

    def test_save_dict_tools_excludes_callables(self, tmp_path):
        """Forward/execute callables are excluded from serialized output."""
        tool = _make_tool_dict_with_forward("fn_tool", "has forward")
        collection = ToolCollection(tools=[tool], source="test")
        collection.save_to_local(str(tmp_path))

        data = json.loads((tmp_path / "collection.json").read_text())
        tool_data = data["tools"][0]
        assert "forward" not in tool_data
        assert "execute" not in tool_data
        assert tool_data["name"] == "fn_tool"

    def test_save_object_tools(self, tmp_path):
        """Tool objects are serialized with name, description, type."""
        tool_obj = _make_tool_object("obj_tool", "An object tool")
        collection = ToolCollection(tools=[tool_obj], source="local")
        collection.save_to_local(str(tmp_path))

        data = json.loads((tmp_path / "collection.json").read_text())
        tool_data = data["tools"][0]
        assert tool_data["name"] == "obj_tool"
        assert tool_data["description"] == "An object tool"
        assert tool_data["tool_type"] == "Mock"

    def test_save_and_reload_roundtrip(self, tmp_path):
        """Saved collection can be loaded back via from_local."""
        tools = [
            _make_tool_dict("t1", "first"),
            _make_tool_dict("t2", "second"),
        ]
        collection = ToolCollection(tools=tools, source="hub", metadata={"v": 1})
        collection.save_to_local(str(tmp_path))

        reloaded = ToolCollection.from_local(str(tmp_path))
        assert len(reloaded) == 2
        assert reloaded.source == "local"
        assert reloaded.tools[0]["name"] == "t1"

    def test_save_empty_collection(self, tmp_path):
        """Empty collection saves valid JSON with empty tools array."""
        collection = ToolCollection(tools=[], source="test")
        collection.save_to_local(str(tmp_path))

        data = json.loads((tmp_path / "collection.json").read_text())
        assert data["tools"] == []


# =============================================================================
# TestToolCollectionToSkillDefinitions
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionToSkillDefinitions:
    """Tests for to_skill_definitions() method."""

    def test_to_skill_definitions_dict_tool(self):
        """Dict tool is converted to a SkillDefinition."""
        tool = _make_tool_dict_with_forward("dict_tool", "A dict tool")
        collection = ToolCollection(tools=[tool], source="test")

        with patch("Jotty.core.registry.skills_registry.SkillDefinition") as MockSkillDef:
            mock_instance = MagicMock()
            MockSkillDef.return_value = mock_instance

            result = collection.to_skill_definitions()
            assert len(result) == 1
            MockSkillDef.assert_called_once()
            call_kwargs = MockSkillDef.call_args
            assert call_kwargs[1]["name"] == "dict_tool"

    def test_to_skill_definitions_object_tool(self):
        """Tool object with name and forward is converted."""
        tool_obj = _make_tool_object("obj_tool", "Object tool")
        collection = ToolCollection(tools=[tool_obj], source="hub")

        with patch("Jotty.core.registry.skills_registry.SkillDefinition") as MockSkillDef:
            mock_instance = MagicMock()
            MockSkillDef.return_value = mock_instance

            result = collection.to_skill_definitions()
            assert len(result) == 1

    def test_to_skill_definitions_unknown_format_skipped(self):
        """Tools of unknown format (no name/forward, not dict) are skipped."""
        unknown_tool = 42  # Not a dict, no name/forward attrs
        collection = ToolCollection(tools=[unknown_tool], source="test")

        with patch("Jotty.core.registry.skills_registry.SkillDefinition"):
            result = collection.to_skill_definitions()
            assert len(result) == 0

    def test_to_skill_definitions_empty(self):
        """Empty collection yields empty list."""
        collection = ToolCollection(tools=[], source="test")

        with patch("Jotty.core.registry.skills_registry.SkillDefinition"):
            result = collection.to_skill_definitions()
            assert result == []

    def test_dict_to_skill_definition_with_forward(self):
        """_dict_to_skill_definition creates executor that calls forward."""
        tool = _make_tool_dict_with_forward("fwd_tool", "Forward tool")
        collection = ToolCollection(tools=[tool], source="test")

        with patch("Jotty.core.registry.skills_registry.SkillDefinition") as MockSkillDef:
            MockSkillDef.return_value = MagicMock()
            collection._dict_to_skill_definition(tool)
            MockSkillDef.assert_called_once()
            call_kwargs = MockSkillDef.call_args[1]
            assert call_kwargs["name"] == "fwd_tool"
            assert "source" in call_kwargs["metadata"]

    def test_dict_to_skill_definition_with_execute(self):
        """_dict_to_skill_definition handles tools with execute key."""
        tool = _make_tool_dict_with_execute("exec_tool", "Execute tool")
        collection = ToolCollection(tools=[tool], source="test")

        with patch("Jotty.core.registry.skills_registry.SkillDefinition") as MockSkillDef:
            MockSkillDef.return_value = MagicMock()
            result = collection._dict_to_skill_definition(tool)
            assert result is not None

    def test_tool_object_to_skill_definition(self):
        """_tool_object_to_skill_definition converts tool object."""
        tool_obj = _make_tool_object("obj", "desc")
        collection = ToolCollection(tools=[tool_obj], source="hub")

        with patch("Jotty.core.registry.skills_registry.SkillDefinition") as MockSkillDef:
            MockSkillDef.return_value = MagicMock()
            result = collection._tool_object_to_skill_definition(tool_obj)
            assert result is not None
            call_kwargs = MockSkillDef.call_args[1]
            assert call_kwargs["name"] == "obj"
            assert call_kwargs["metadata"]["tool_type"] == "Mock"

    def test_dict_to_skill_definition_returns_none_on_error(self):
        """Returns None if SkillDefinition construction fails."""
        tool = {"name": "bad"}
        collection = ToolCollection(tools=[tool], source="test")

        with patch(
            "Jotty.core.registry.skills_registry.SkillDefinition",
            side_effect=Exception("fail"),
        ):
            result = collection._dict_to_skill_definition(tool)
            assert result is None

    def test_tool_object_to_skill_definition_returns_none_on_error(self):
        """Returns None if SkillDefinition construction fails for object."""
        tool_obj = _make_tool_object()
        collection = ToolCollection(tools=[tool_obj], source="test")

        with patch(
            "Jotty.core.registry.skills_registry.SkillDefinition",
            side_effect=Exception("fail"),
        ):
            result = collection._tool_object_to_skill_definition(tool_obj)
            assert result is None


# =============================================================================
# TestToolCollectionEdgeCases
# =============================================================================


@pytest.mark.unit
@skip_no_module
class TestToolCollectionEdgeCases:
    """Tests for edge cases and mixed scenarios."""

    def test_collection_with_none_tool_values(self):
        """Tool dict with None values for name/description."""
        tool = {"name": None, "description": None}
        collection = ToolCollection(tools=[tool])
        result = collection.list_tools()
        assert len(result) == 1
        # get() with default should handle None in the dict vs missing key
        assert result[0]["name"] is None

    def test_collection_with_extra_metadata_in_tool_dict(self):
        """Extra keys in tool dict do not cause errors."""
        tool = {
            "name": "extra",
            "description": "has extras",
            "extra_key": "extra_value",
            "nested": {"a": 1},
        }
        collection = ToolCollection(tools=[tool])
        assert len(collection) == 1

    def test_from_local_with_path_object(self, tmp_path):
        """from_local works with a Path object directly."""
        (tmp_path / "collection.json").write_text(json.dumps({"tools": [{"name": "path_tool"}]}))
        collection = ToolCollection.from_local(tmp_path)
        assert len(collection) == 1

    def test_save_to_local_accepts_path_object(self, tmp_path):
        """save_to_local works with a Path object."""
        out = tmp_path / "out"
        collection = ToolCollection(tools=[_make_tool_dict()], source="x")
        collection.save_to_local(out)
        assert (out / "collection.json").exists()

    def test_multiple_tools_of_different_types(self):
        """Collection can hold dict tools and object tools simultaneously."""
        tools = [
            _make_tool_dict("dict1"),
            _make_tool_object("obj1"),
            _make_tool_dict("dict2"),
        ]
        collection = ToolCollection(tools=tools)
        assert len(collection) == 3
        result = collection.list_tools()
        assert len(result) == 3
