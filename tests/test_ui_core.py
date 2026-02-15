"""
Tests for UI Core Module
=========================
Tests for StatusTaxonomy (status_taxonomy.py) and A2UI widgets/helpers (a2ui.py).
"""
import pytest
import json
from unittest.mock import Mock, MagicMock, patch


# =============================================================================
# StatusTaxonomy Tests
# =============================================================================

@pytest.mark.unit
class TestStatusTaxonomyCanonicalStatuses:
    """Tests for the CANONICAL_STATUSES class attribute."""

    def test_canonical_statuses_has_four_keys(self):
        """CANONICAL_STATUSES contains exactly backlog, in_progress, completed, failed."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        expected = {'backlog', 'in_progress', 'completed', 'failed'}
        assert set(StatusTaxonomy.CANONICAL_STATUSES.keys()) == expected

    def test_each_canonical_status_has_required_fields(self):
        """Each canonical status has label, description, kanban_column, aliases."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        for key, config in StatusTaxonomy.CANONICAL_STATUSES.items():
            assert 'label' in config, f"{key} missing label"
            assert 'description' in config, f"{key} missing description"
            assert 'kanban_column' in config, f"{key} missing kanban_column"
            assert 'aliases' in config, f"{key} missing aliases"
            assert isinstance(config['aliases'], list), f"{key} aliases must be list"

    def test_canonical_kanban_columns_match_keys(self):
        """Each kanban_column matches its canonical key."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        for key, config in StatusTaxonomy.CANONICAL_STATUSES.items():
            assert config['kanban_column'] == key

    def test_backlog_aliases(self):
        """Backlog has expected aliases."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        aliases = StatusTaxonomy.CANONICAL_STATUSES['backlog']['aliases']
        assert 'todo' in aliases
        assert 'pending' in aliases
        assert 'not_started' in aliases

    def test_in_progress_aliases(self):
        """In Progress has expected aliases."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        aliases = StatusTaxonomy.CANONICAL_STATUSES['in_progress']['aliases']
        assert 'active' in aliases
        assert 'doing' in aliases
        assert 'wip' in aliases

    def test_completed_aliases(self):
        """Completed has expected aliases."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        aliases = StatusTaxonomy.CANONICAL_STATUSES['completed']['aliases']
        assert 'done' in aliases
        assert 'finished' in aliases
        assert 'closed' in aliases

    def test_failed_aliases(self):
        """Failed has expected aliases."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        aliases = StatusTaxonomy.CANONICAL_STATUSES['failed']['aliases']
        assert 'error' in aliases
        assert 'blocked' in aliases
        assert 'cancelled' in aliases


@pytest.mark.unit
class TestStatusTaxonomyInit:
    """Tests for StatusTaxonomy.__init__."""

    def test_default_init_no_custom_mapping(self):
        """Default init creates empty custom_mapping."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.custom_mapping == {}

    def test_init_with_custom_mapping(self):
        """Custom mapping is stored properly."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        custom = {'my_status': 'completed'}
        taxonomy = StatusTaxonomy(custom_mapping=custom)
        assert taxonomy.custom_mapping == custom

    def test_init_builds_alias_lookup(self):
        """Init builds _alias_to_canonical reverse lookup."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        # Canonical statuses map to themselves
        assert taxonomy._alias_to_canonical['backlog'] == 'backlog'
        assert taxonomy._alias_to_canonical['in_progress'] == 'in_progress'
        # Aliases map to canonical
        assert taxonomy._alias_to_canonical['todo'] == 'backlog'
        assert taxonomy._alias_to_canonical['doing'] == 'in_progress'
        assert taxonomy._alias_to_canonical['done'] == 'completed'
        assert taxonomy._alias_to_canonical['error'] == 'failed'

    def test_init_with_none_custom_mapping(self):
        """None custom_mapping defaults to empty dict."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy(custom_mapping=None)
        assert taxonomy.custom_mapping == {}


@pytest.mark.unit
class TestStatusTaxonomyNormalize:
    """Tests for StatusTaxonomy.normalize method."""

    def test_normalize_canonical_statuses(self):
        """Canonical statuses normalize to themselves."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.normalize('backlog') == 'backlog'
        assert taxonomy.normalize('in_progress') == 'in_progress'
        assert taxonomy.normalize('completed') == 'completed'
        assert taxonomy.normalize('failed') == 'failed'

    def test_normalize_aliases_to_backlog(self):
        """All backlog aliases normalize correctly."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        for alias in ['todo', 'pending', 'not_started', 'queue', 'waiting']:
            assert taxonomy.normalize(alias) == 'backlog', f"'{alias}' should normalize to 'backlog'"

    def test_normalize_aliases_to_in_progress(self):
        """All in_progress aliases normalize correctly."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        for alias in ['active', 'doing', 'working', 'started', 'wip']:
            assert taxonomy.normalize(alias) == 'in_progress', f"'{alias}' should normalize to 'in_progress'"

    def test_normalize_aliases_to_completed(self):
        """All completed aliases normalize correctly."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        for alias in ['done', 'finished', 'closed', 'resolved', 'complete']:
            assert taxonomy.normalize(alias) == 'completed', f"'{alias}' should normalize to 'completed'"

    def test_normalize_aliases_to_failed(self):
        """All failed aliases normalize correctly."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        for alias in ['error', 'blocked', 'cancelled', 'rejected']:
            assert taxonomy.normalize(alias) == 'failed', f"'{alias}' should normalize to 'failed'"

    def test_normalize_case_insensitive(self):
        """Normalize handles mixed case."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.normalize('TODO') == 'backlog'
        assert taxonomy.normalize('Done') == 'completed'
        assert taxonomy.normalize('IN_PROGRESS') == 'in_progress'
        assert taxonomy.normalize('FAILED') == 'failed'

    def test_normalize_replaces_spaces_with_underscores(self):
        """Normalize converts spaces to underscores."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.normalize('in progress') == 'in_progress'
        assert taxonomy.normalize('not started') == 'backlog'

    def test_normalize_replaces_dashes_with_underscores(self):
        """Normalize converts dashes to underscores."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.normalize('in-progress') == 'in_progress'
        assert taxonomy.normalize('not-started') == 'backlog'

    def test_normalize_unknown_status_defaults_to_backlog(self):
        """Unknown statuses default to backlog."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.normalize('nonexistent') == 'backlog'
        assert taxonomy.normalize('foobar') == 'backlog'
        assert taxonomy.normalize('') == 'backlog'

    def test_normalize_custom_mapping_takes_priority(self):
        """Custom mapping is checked before built-in aliases."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        custom = {'todo': 'completed'}  # Override built-in alias
        taxonomy = StatusTaxonomy(custom_mapping=custom)
        assert taxonomy.normalize('todo') == 'completed'

    def test_normalize_custom_mapping_for_unknown_status(self):
        """Custom mapping handles custom client statuses."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        custom = {'reviewing': 'in_progress', 'deployed': 'completed'}
        taxonomy = StatusTaxonomy(custom_mapping=custom)
        assert taxonomy.normalize('reviewing') == 'in_progress'
        assert taxonomy.normalize('deployed') == 'completed'


@pytest.mark.unit
class TestStatusTaxonomyMethods:
    """Tests for to_kanban_column, get_label, get_all_statuses, create_kanban_columns."""

    def test_to_kanban_column_canonical(self):
        """to_kanban_column returns correct column for canonical statuses."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.to_kanban_column('backlog') == 'backlog'
        assert taxonomy.to_kanban_column('in_progress') == 'in_progress'
        assert taxonomy.to_kanban_column('completed') == 'completed'
        assert taxonomy.to_kanban_column('failed') == 'failed'

    def test_to_kanban_column_alias(self):
        """to_kanban_column resolves aliases before returning column."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.to_kanban_column('done') == 'completed'
        assert taxonomy.to_kanban_column('wip') == 'in_progress'
        assert taxonomy.to_kanban_column('todo') == 'backlog'
        assert taxonomy.to_kanban_column('error') == 'failed'

    def test_to_kanban_column_unknown_defaults_to_backlog(self):
        """to_kanban_column maps unknown statuses to backlog column."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.to_kanban_column('mystery') == 'backlog'

    def test_get_label_canonical(self):
        """get_label returns human-readable labels."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.get_label('backlog') == 'Backlog'
        assert taxonomy.get_label('in_progress') == 'In Progress'
        assert taxonomy.get_label('completed') == 'Completed'
        assert taxonomy.get_label('failed') == 'Failed'

    def test_get_label_alias(self):
        """get_label resolves alias then returns label."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.get_label('wip') == 'In Progress'
        assert taxonomy.get_label('done') == 'Completed'
        assert taxonomy.get_label('todo') == 'Backlog'

    def test_get_label_unknown_defaults_to_backlog_label(self):
        """get_label for unknown status returns Backlog label."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        assert taxonomy.get_label('xyz') == 'Backlog'

    def test_get_all_statuses(self):
        """get_all_statuses returns all four canonical statuses."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        statuses = taxonomy.get_all_statuses()
        assert isinstance(statuses, list)
        assert set(statuses) == {'backlog', 'in_progress', 'completed', 'failed'}

    def test_create_kanban_columns_structure(self):
        """create_kanban_columns returns list of column dicts with id, title, items."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        columns = taxonomy.create_kanban_columns()
        assert isinstance(columns, list)
        assert len(columns) == 4
        for col in columns:
            assert 'id' in col
            assert 'title' in col
            assert 'items' in col
            assert col['items'] == []

    def test_create_kanban_columns_order_and_content(self):
        """create_kanban_columns returns columns with correct ids and titles."""
        from Jotty.core.interface.ui.status_taxonomy import StatusTaxonomy
        taxonomy = StatusTaxonomy()
        columns = taxonomy.create_kanban_columns()
        column_map = {c['id']: c['title'] for c in columns}
        assert column_map['backlog'] == 'Backlog'
        assert column_map['in_progress'] == 'In Progress'
        assert column_map['completed'] == 'Completed'
        assert column_map['failed'] == 'Failed'


@pytest.mark.unit
class TestStatusMapperGlobal:
    """Tests for the global status_mapper instance."""

    def test_status_mapper_is_status_taxonomy(self):
        """Global status_mapper is a StatusTaxonomy instance."""
        from Jotty.core.interface.ui.status_taxonomy import status_mapper, StatusTaxonomy
        assert isinstance(status_mapper, StatusTaxonomy)

    def test_status_mapper_normalize_works(self):
        """Global status_mapper can normalize statuses."""
        from Jotty.core.interface.ui.status_taxonomy import status_mapper
        assert status_mapper.normalize('todo') == 'backlog'
        assert status_mapper.normalize('done') == 'completed'

    def test_status_mapper_has_no_custom_mapping(self):
        """Global status_mapper has empty custom_mapping."""
        from Jotty.core.interface.ui.status_taxonomy import status_mapper
        assert status_mapper.custom_mapping == {}


# =============================================================================
# A2UI Widget Tests
# =============================================================================

@pytest.mark.unit
class TestA2UIWidgetBase:
    """Tests for the A2UIWidget base class."""

    def test_to_dict_raises_not_implemented(self):
        """Base A2UIWidget.to_dict raises NotImplementedError."""
        from Jotty.core.interface.ui.a2ui import A2UIWidget
        widget = A2UIWidget()
        with pytest.raises(NotImplementedError):
            widget.to_dict()

    def test_to_json_calls_to_dict(self):
        """to_json serializes to_dict result to JSON string."""
        from Jotty.core.interface.ui.a2ui import A2UIWidget
        widget = A2UIWidget()
        # Patch to_dict on the instance to return a dict
        widget.to_dict = lambda: {"type": "test", "value": 42}
        result = widget.to_json()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"type": "test", "value": 42}


@pytest.mark.unit
class TestA2UIText:
    """Tests for the A2UIText widget."""

    def test_text_without_style(self):
        """A2UIText without style produces type and text only."""
        from Jotty.core.interface.ui.a2ui import A2UIText
        widget = A2UIText("Hello world")
        result = widget.to_dict()
        assert result == {"type": "text", "text": "Hello world"}

    def test_text_with_style(self):
        """A2UIText with style includes style field."""
        from Jotty.core.interface.ui.a2ui import A2UIText
        widget = A2UIText("Bold text", style="bold")
        result = widget.to_dict()
        assert result == {"type": "text", "text": "Bold text", "style": "bold"}

    def test_text_with_italic_style(self):
        """A2UIText with italic style."""
        from Jotty.core.interface.ui.a2ui import A2UIText
        widget = A2UIText("Italic text", style="italic")
        result = widget.to_dict()
        assert result["style"] == "italic"

    def test_text_none_style_omitted(self):
        """A2UIText with None style omits style key."""
        from Jotty.core.interface.ui.a2ui import A2UIText
        widget = A2UIText("Some text", style=None)
        result = widget.to_dict()
        assert "style" not in result

    def test_text_to_json(self):
        """A2UIText.to_json produces valid JSON."""
        from Jotty.core.interface.ui.a2ui import A2UIText
        widget = A2UIText("test")
        result = json.loads(widget.to_json())
        assert result["type"] == "text"
        assert result["text"] == "test"

    def test_text_empty_string(self):
        """A2UIText handles empty string."""
        from Jotty.core.interface.ui.a2ui import A2UIText
        widget = A2UIText("")
        result = widget.to_dict()
        assert result["text"] == ""


@pytest.mark.unit
class TestA2UICard:
    """Tests for the A2UICard widget."""

    def test_card_with_all_fields(self):
        """A2UICard with title, subtitle, body string, no footer."""
        from Jotty.core.interface.ui.a2ui import A2UICard
        card = A2UICard(title="Title", subtitle="Subtitle", body="Body text")
        result = card.to_dict()
        assert result["type"] == "card"
        assert result["title"] == "Title"
        assert result["subtitle"] == "Subtitle"
        assert result["body"] == {"type": "text", "text": "Body text"}

    def test_card_minimal_no_fields(self):
        """A2UICard with no fields produces just type."""
        from Jotty.core.interface.ui.a2ui import A2UICard
        card = A2UICard()
        result = card.to_dict()
        assert result == {"type": "card"}

    def test_card_body_as_widget(self):
        """A2UICard body as a widget calls to_dict."""
        from Jotty.core.interface.ui.a2ui import A2UICard, A2UIText
        text_widget = A2UIText("Widget body", style="bold")
        card = A2UICard(title="Test", body=text_widget)
        result = card.to_dict()
        assert result["body"] == {"type": "text", "text": "Widget body", "style": "bold"}

    def test_card_body_as_widget_list(self):
        """A2UICard body as list of widgets."""
        from Jotty.core.interface.ui.a2ui import A2UICard, A2UIText
        widgets = [A2UIText("Line 1"), A2UIText("Line 2")]
        card = A2UICard(title="Test", body=widgets)
        result = card.to_dict()
        assert isinstance(result["body"], list)
        assert len(result["body"]) == 2
        assert result["body"][0] == {"type": "text", "text": "Line 1"}
        assert result["body"][1] == {"type": "text", "text": "Line 2"}

    def test_card_body_list_with_mixed_items(self):
        """A2UICard body list with widget and non-widget items."""
        from Jotty.core.interface.ui.a2ui import A2UICard, A2UIText
        items = [A2UIText("widget"), {"type": "raw", "data": "plain dict"}]
        card = A2UICard(body=items)
        result = card.to_dict()
        assert result["body"][0] == {"type": "text", "text": "widget"}
        assert result["body"][1] == {"type": "raw", "data": "plain dict"}

    def test_card_footer_as_widget(self):
        """A2UICard footer as a single widget."""
        from Jotty.core.interface.ui.a2ui import A2UICard, A2UIButton
        btn = A2UIButton(label="Click")
        card = A2UICard(title="Test", footer=btn)
        result = card.to_dict()
        assert result["footer"]["type"] == "button"
        assert result["footer"]["label"] == "Click"

    def test_card_footer_as_widget_list(self):
        """A2UICard footer as list of widgets."""
        from Jotty.core.interface.ui.a2ui import A2UICard, A2UIButton
        buttons = [A2UIButton(label="OK"), A2UIButton(label="Cancel")]
        card = A2UICard(footer=buttons)
        result = card.to_dict()
        assert isinstance(result["footer"], list)
        assert len(result["footer"]) == 2
        assert result["footer"][0]["label"] == "OK"
        assert result["footer"][1]["label"] == "Cancel"

    def test_card_footer_list_with_non_widget(self):
        """A2UICard footer list with non-widget items passes through."""
        from Jotty.core.interface.ui.a2ui import A2UICard
        footer_items = [{"type": "custom", "label": "raw"}]
        card = A2UICard(footer=footer_items)
        result = card.to_dict()
        assert result["footer"][0] == {"type": "custom", "label": "raw"}

    def test_card_none_body_omitted(self):
        """A2UICard with None body omits body key."""
        from Jotty.core.interface.ui.a2ui import A2UICard
        card = A2UICard(title="Only title")
        result = card.to_dict()
        assert "body" not in result

    def test_card_none_footer_omitted(self):
        """A2UICard with None footer omits footer key."""
        from Jotty.core.interface.ui.a2ui import A2UICard
        card = A2UICard(title="Only title")
        result = card.to_dict()
        assert "footer" not in result

    def test_card_to_json(self):
        """A2UICard.to_json produces valid JSON."""
        from Jotty.core.interface.ui.a2ui import A2UICard
        card = A2UICard(title="JSON test", body="body content")
        result = json.loads(card.to_json())
        assert result["type"] == "card"
        assert result["title"] == "JSON test"


@pytest.mark.unit
class TestA2UIList:
    """Tests for the A2UIList widget."""

    def test_list_with_items(self):
        """A2UIList renders items correctly."""
        from Jotty.core.interface.ui.a2ui import A2UIList
        items = [{"title": "Task 1"}, {"title": "Task 2", "status": "done"}]
        widget = A2UIList(items)
        result = widget.to_dict()
        assert result["type"] == "list"
        assert result["items"] == items

    def test_list_empty_items(self):
        """A2UIList with empty list."""
        from Jotty.core.interface.ui.a2ui import A2UIList
        widget = A2UIList([])
        result = widget.to_dict()
        assert result["items"] == []

    def test_list_to_json(self):
        """A2UIList.to_json produces valid JSON."""
        from Jotty.core.interface.ui.a2ui import A2UIList
        widget = A2UIList([{"title": "item"}])
        result = json.loads(widget.to_json())
        assert result["type"] == "list"
        assert len(result["items"]) == 1


@pytest.mark.unit
class TestA2UIImage:
    """Tests for the A2UIImage widget."""

    def test_image_url_only(self):
        """A2UIImage with only url."""
        from Jotty.core.interface.ui.a2ui import A2UIImage
        widget = A2UIImage(url="https://example.com/img.png")
        result = widget.to_dict()
        assert result == {"type": "image", "url": "https://example.com/img.png"}

    def test_image_with_alt(self):
        """A2UIImage with alt text."""
        from Jotty.core.interface.ui.a2ui import A2UIImage
        widget = A2UIImage(url="https://example.com/img.png", alt="A photo")
        result = widget.to_dict()
        assert result["alt"] == "A photo"

    def test_image_with_caption(self):
        """A2UIImage with caption."""
        from Jotty.core.interface.ui.a2ui import A2UIImage
        widget = A2UIImage(url="https://example.com/img.png", caption="Figure 1")
        result = widget.to_dict()
        assert result["caption"] == "Figure 1"

    def test_image_with_all_fields(self):
        """A2UIImage with all optional fields."""
        from Jotty.core.interface.ui.a2ui import A2UIImage
        widget = A2UIImage(url="https://example.com/img.png", alt="Alt", caption="Cap")
        result = widget.to_dict()
        assert result["type"] == "image"
        assert result["url"] == "https://example.com/img.png"
        assert result["alt"] == "Alt"
        assert result["caption"] == "Cap"

    def test_image_none_alt_omitted(self):
        """A2UIImage with None alt omits alt key."""
        from Jotty.core.interface.ui.a2ui import A2UIImage
        widget = A2UIImage(url="https://example.com/img.png", alt=None)
        result = widget.to_dict()
        assert "alt" not in result

    def test_image_none_caption_omitted(self):
        """A2UIImage with None caption omits caption key."""
        from Jotty.core.interface.ui.a2ui import A2UIImage
        widget = A2UIImage(url="https://example.com/img.png", caption=None)
        result = widget.to_dict()
        assert "caption" not in result

    def test_image_to_json(self):
        """A2UIImage.to_json produces valid JSON."""
        from Jotty.core.interface.ui.a2ui import A2UIImage
        widget = A2UIImage(url="https://example.com/img.png")
        result = json.loads(widget.to_json())
        assert result["type"] == "image"


@pytest.mark.unit
class TestA2UIButton:
    """Tests for the A2UIButton widget."""

    def test_button_label_only(self):
        """A2UIButton with label only uses secondary variant."""
        from Jotty.core.interface.ui.a2ui import A2UIButton
        widget = A2UIButton(label="Click Me")
        result = widget.to_dict()
        assert result["type"] == "button"
        assert result["label"] == "Click Me"
        assert result["variant"] == "secondary"
        assert "action" not in result

    def test_button_with_action(self):
        """A2UIButton with action dict."""
        from Jotty.core.interface.ui.a2ui import A2UIButton
        action = {"type": "url", "url": "https://example.com"}
        widget = A2UIButton(label="Go", action=action)
        result = widget.to_dict()
        assert result["action"] == action

    def test_button_primary_variant(self):
        """A2UIButton with primary variant."""
        from Jotty.core.interface.ui.a2ui import A2UIButton
        widget = A2UIButton(label="Submit", variant="primary")
        result = widget.to_dict()
        assert result["variant"] == "primary"

    def test_button_none_action_omitted(self):
        """A2UIButton with None action omits action key."""
        from Jotty.core.interface.ui.a2ui import A2UIButton
        widget = A2UIButton(label="Test", action=None)
        result = widget.to_dict()
        assert "action" not in result

    def test_button_to_json(self):
        """A2UIButton.to_json produces valid JSON."""
        from Jotty.core.interface.ui.a2ui import A2UIButton
        widget = A2UIButton(label="btn")
        result = json.loads(widget.to_json())
        assert result["label"] == "btn"


@pytest.mark.unit
class TestA2UISeparator:
    """Tests for the A2UISeparator widget."""

    def test_separator_to_dict(self):
        """A2UISeparator returns type separator."""
        from Jotty.core.interface.ui.a2ui import A2UISeparator
        widget = A2UISeparator()
        result = widget.to_dict()
        assert result == {"type": "separator"}

    def test_separator_to_json(self):
        """A2UISeparator.to_json produces valid JSON."""
        from Jotty.core.interface.ui.a2ui import A2UISeparator
        widget = A2UISeparator()
        result = json.loads(widget.to_json())
        assert result == {"type": "separator"}


@pytest.mark.unit
class TestA2UISection:
    """Tests for the A2UISection widget."""

    def test_section_minimal(self):
        """A2UISection with required fields only."""
        from Jotty.core.interface.ui.a2ui import A2UISection
        content = {"columns": []}
        widget = A2UISection(section_type="kanban-board", content=content)
        result = widget.to_dict()
        assert result["type"] == "section"
        assert result["section_type"] == "kanban-board"
        assert result["content"] == content
        assert "title" not in result
        assert "props" not in result

    def test_section_with_title(self):
        """A2UISection with title."""
        from Jotty.core.interface.ui.a2ui import A2UISection
        widget = A2UISection(section_type="chart", content={"type": "bar"}, title="Revenue")
        result = widget.to_dict()
        assert result["title"] == "Revenue"

    def test_section_with_props(self):
        """A2UISection with props."""
        from Jotty.core.interface.ui.a2ui import A2UISection
        props = {"theme": "dark", "animate": True}
        widget = A2UISection(section_type="chart", content={}, props=props)
        result = widget.to_dict()
        assert result["props"] == props

    def test_section_content_as_string(self):
        """A2UISection with string content (e.g. mermaid)."""
        from Jotty.core.interface.ui.a2ui import A2UISection
        widget = A2UISection(section_type="mermaid", content="graph TD; A-->B;")
        result = widget.to_dict()
        assert result["content"] == "graph TD; A-->B;"

    def test_section_empty_props_omitted(self):
        """A2UISection with empty props (default) omits props key."""
        from Jotty.core.interface.ui.a2ui import A2UISection
        widget = A2UISection(section_type="test", content={})
        result = widget.to_dict()
        assert "props" not in result

    def test_section_none_props_defaults_to_empty(self):
        """A2UISection with None props defaults to empty dict (omitted from output)."""
        from Jotty.core.interface.ui.a2ui import A2UISection
        widget = A2UISection(section_type="test", content={}, props=None)
        assert widget.props == {}
        result = widget.to_dict()
        assert "props" not in result

    def test_section_to_json(self):
        """A2UISection.to_json produces valid JSON."""
        from Jotty.core.interface.ui.a2ui import A2UISection
        widget = A2UISection(section_type="chart", content={"data": [1, 2, 3]}, title="Test")
        result = json.loads(widget.to_json())
        assert result["type"] == "section"
        assert result["section_type"] == "chart"


# =============================================================================
# A2UIBuilder Tests
# =============================================================================

@pytest.mark.unit
class TestA2UIBuilder:
    """Tests for the A2UIBuilder class."""

    def test_builder_empty_build(self):
        """Empty builder produces response with empty content."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        builder = A2UIBuilder()
        result = builder.build()
        assert result == {"role": "assistant", "content": []}

    def test_builder_add_text(self):
        """Builder add_text adds text widget."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = A2UIBuilder().add_text("Hello").build()
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "Hello"}

    def test_builder_add_text_with_style(self):
        """Builder add_text with style."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = A2UIBuilder().add_text("Bold", style="bold").build()
        assert result["content"][0]["style"] == "bold"

    def test_builder_add_card(self):
        """Builder add_card adds card widget."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = A2UIBuilder().add_card(title="Card", body="Body").build()
        assert result["content"][0]["type"] == "card"
        assert result["content"][0]["title"] == "Card"

    def test_builder_add_list(self):
        """Builder add_list adds list widget."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        items = [{"title": "Item 1"}]
        result = A2UIBuilder().add_list(items).build()
        assert result["content"][0]["type"] == "list"
        assert result["content"][0]["items"] == items

    def test_builder_add_image(self):
        """Builder add_image adds image widget."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = A2UIBuilder().add_image("https://img.com/a.png", alt="pic").build()
        assert result["content"][0]["type"] == "image"
        assert result["content"][0]["url"] == "https://img.com/a.png"

    def test_builder_add_button(self):
        """Builder add_button adds button widget."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = A2UIBuilder().add_button("Click", variant="primary").build()
        assert result["content"][0]["type"] == "button"
        assert result["content"][0]["label"] == "Click"
        assert result["content"][0]["variant"] == "primary"

    def test_builder_add_separator(self):
        """Builder add_separator adds separator widget."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = A2UIBuilder().add_separator().build()
        assert result["content"][0] == {"type": "separator"}

    def test_builder_add_section(self):
        """Builder add_section adds section widget."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = A2UIBuilder().add_section("mermaid", "graph TD; A-->B;", title="Diagram").build()
        assert result["content"][0]["type"] == "section"
        assert result["content"][0]["section_type"] == "mermaid"

    def test_builder_add_widget_generic(self):
        """Builder add_widget accepts any A2UIWidget subclass."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder, A2UISeparator
        sep = A2UISeparator()
        result = A2UIBuilder().add_widget(sep).build()
        assert result["content"][0] == {"type": "separator"}

    def test_builder_chaining(self):
        """Builder methods return self for chaining."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        builder = A2UIBuilder()
        returned = builder.add_text("text")
        assert returned is builder

    def test_builder_chaining_multiple(self):
        """Builder chains multiple widgets correctly."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = (
            A2UIBuilder()
            .add_text("Title", style="bold")
            .add_separator()
            .add_card(title="Card", body="content")
            .add_button("OK", variant="primary")
            .build()
        )
        assert len(result["content"]) == 4
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "separator"
        assert result["content"][2]["type"] == "card"
        assert result["content"][3]["type"] == "button"

    def test_builder_to_json(self):
        """Builder to_json produces valid JSON of built response."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        builder = A2UIBuilder().add_text("Test")
        result = json.loads(builder.to_json())
        assert result["role"] == "assistant"
        assert len(result["content"]) == 1

    def test_builder_response_structure(self):
        """Builder.build() always includes role and content keys."""
        from Jotty.core.interface.ui.a2ui import A2UIBuilder
        result = A2UIBuilder().add_text("hi").build()
        assert "role" in result
        assert result["role"] == "assistant"
        assert "content" in result
        assert isinstance(result["content"], list)


# =============================================================================
# A2UI Helper Functions Tests
# =============================================================================

@pytest.mark.unit
class TestIsA2UIResponse:
    """Tests for the is_a2ui_response helper function."""

    def test_widget_instance_is_a2ui(self):
        """A2UIWidget instance returns True."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response, A2UIText
        assert is_a2ui_response(A2UIText("test")) is True

    def test_valid_dict_response_is_a2ui(self):
        """Dict with role, content, and valid widget type returns True."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        response = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}]
        }
        assert is_a2ui_response(response) is True

    def test_dict_with_card_type_is_a2ui(self):
        """Dict with card type widget returns True."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        response = {
            "role": "assistant",
            "content": [{"type": "card", "title": "test"}]
        }
        assert is_a2ui_response(response) is True

    def test_dict_with_section_type_is_a2ui(self):
        """Dict with section type widget returns True."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        response = {
            "role": "assistant",
            "content": [{"type": "section", "section_type": "chart"}]
        }
        assert is_a2ui_response(response) is True

    def test_dict_with_list_type_is_a2ui(self):
        """Dict with list type widget returns True."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        response = {
            "role": "assistant",
            "content": [{"type": "list", "items": []}]
        }
        assert is_a2ui_response(response) is True

    def test_plain_string_not_a2ui(self):
        """Plain string returns False."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        assert is_a2ui_response("just text") is False

    def test_empty_dict_not_a2ui(self):
        """Empty dict returns False."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        assert is_a2ui_response({}) is False

    def test_dict_missing_role_not_a2ui(self):
        """Dict without role key returns False."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        response = {"content": [{"type": "text", "text": "hi"}]}
        assert is_a2ui_response(response) is False

    def test_dict_empty_content_not_a2ui(self):
        """Dict with empty content list returns False."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        response = {"role": "assistant", "content": []}
        assert is_a2ui_response(response) is False

    def test_dict_invalid_widget_type_not_a2ui(self):
        """Dict with invalid widget type returns False."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        response = {
            "role": "assistant",
            "content": [{"type": "unknown_widget"}]
        }
        assert is_a2ui_response(response) is False

    def test_none_not_a2ui(self):
        """None returns False."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        assert is_a2ui_response(None) is False

    def test_integer_not_a2ui(self):
        """Integer returns False."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        assert is_a2ui_response(42) is False

    def test_content_not_list_not_a2ui(self):
        """Dict with content as string (not list) returns False."""
        from Jotty.core.interface.ui.a2ui import is_a2ui_response
        response = {"role": "assistant", "content": "not a list"}
        assert is_a2ui_response(response) is False


@pytest.mark.unit
class TestConvertToA2UIResponse:
    """Tests for the convert_to_a2ui_response helper function."""

    def test_convert_widget_instance(self):
        """A2UIWidget instance gets wrapped in response structure."""
        from Jotty.core.interface.ui.a2ui import convert_to_a2ui_response, A2UIText
        widget = A2UIText("test")
        result = convert_to_a2ui_response(widget)
        assert result["role"] == "assistant"
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "test"}

    def test_convert_valid_dict_returns_as_is(self):
        """Already valid A2UI dict is returned as-is."""
        from Jotty.core.interface.ui.a2ui import convert_to_a2ui_response
        response = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}]
        }
        result = convert_to_a2ui_response(response)
        assert result is response  # Same reference

    def test_convert_plain_string(self):
        """Plain string gets wrapped as text widget."""
        from Jotty.core.interface.ui.a2ui import convert_to_a2ui_response
        result = convert_to_a2ui_response("Hello world")
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello world"

    def test_convert_integer(self):
        """Integer gets converted to text widget via str()."""
        from Jotty.core.interface.ui.a2ui import convert_to_a2ui_response
        result = convert_to_a2ui_response(42)
        assert result["content"][0]["text"] == "42"

    def test_convert_none(self):
        """None gets converted to text widget with 'None' string."""
        from Jotty.core.interface.ui.a2ui import convert_to_a2ui_response
        result = convert_to_a2ui_response(None)
        assert result["content"][0]["text"] == "None"

    def test_convert_non_a2ui_dict(self):
        """Non-A2UI dict gets converted to text widget via str()."""
        from Jotty.core.interface.ui.a2ui import convert_to_a2ui_response
        result = convert_to_a2ui_response({"key": "value"})
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"


@pytest.mark.unit
class TestFormatTaskList:
    """Tests for the format_task_list helper function."""

    def test_format_task_list_without_title(self):
        """format_task_list without title produces list widget directly."""
        from Jotty.core.interface.ui.a2ui import format_task_list
        tasks = [{"title": "Task 1"}, {"title": "Task 2"}]
        result = format_task_list(tasks)
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "list"
        assert result["content"][0]["items"] == tasks

    def test_format_task_list_with_title(self):
        """format_task_list with title wraps list in a card."""
        from Jotty.core.interface.ui.a2ui import format_task_list
        tasks = [{"title": "Task 1"}]
        result = format_task_list(tasks, title="My Tasks")
        assert result["content"][0]["type"] == "card"
        assert result["content"][0]["title"] == "My Tasks"
        # Body should be the list widget
        assert result["content"][0]["body"]["type"] == "list"

    def test_format_task_list_empty(self):
        """format_task_list with empty tasks."""
        from Jotty.core.interface.ui.a2ui import format_task_list
        result = format_task_list([])
        assert result["content"][0]["type"] == "list"
        assert result["content"][0]["items"] == []


@pytest.mark.unit
class TestFormatCard:
    """Tests for the format_card helper function."""

    def test_format_card_basic(self):
        """format_card creates card response."""
        from Jotty.core.interface.ui.a2ui import format_card
        result = format_card(title="Title", body="Body text")
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "card"
        assert result["content"][0]["title"] == "Title"
        assert result["content"][0]["body"] == {"type": "text", "text": "Body text"}

    def test_format_card_with_subtitle(self):
        """format_card with subtitle."""
        from Jotty.core.interface.ui.a2ui import format_card
        result = format_card(title="Title", body="Body", subtitle="Sub")
        assert result["content"][0]["subtitle"] == "Sub"

    def test_format_card_no_subtitle(self):
        """format_card without subtitle omits it."""
        from Jotty.core.interface.ui.a2ui import format_card
        result = format_card(title="T", body="B")
        assert "subtitle" not in result["content"][0]


@pytest.mark.unit
class TestFormatText:
    """Tests for the format_text helper function."""

    def test_format_text_basic(self):
        """format_text creates text response."""
        from Jotty.core.interface.ui.a2ui import format_text
        result = format_text("Hello")
        assert result["role"] == "assistant"
        assert result["content"][0] == {"type": "text", "text": "Hello"}

    def test_format_text_with_style(self):
        """format_text with style."""
        from Jotty.core.interface.ui.a2ui import format_text
        result = format_text("Bold text", style="bold")
        assert result["content"][0]["style"] == "bold"

    def test_format_text_no_style(self):
        """format_text without style omits style key."""
        from Jotty.core.interface.ui.a2ui import format_text
        result = format_text("Plain text")
        assert "style" not in result["content"][0]


@pytest.mark.unit
class TestFormatSection:
    """Tests for the format_section helper function."""

    def test_format_section_with_dict_content(self):
        """format_section with dict content."""
        from Jotty.core.interface.ui.a2ui import format_section
        content = {"columns": [{"id": "todo", "title": "To Do", "items": []}]}
        result = format_section("kanban-board", content, title="Sprint Tasks")
        assert result["role"] == "assistant"
        section = result["content"][0]
        assert section["type"] == "section"
        assert section["section_type"] == "kanban-board"
        assert section["content"] == content
        assert section["title"] == "Sprint Tasks"

    def test_format_section_with_string_content(self):
        """format_section with string content (e.g. mermaid)."""
        from Jotty.core.interface.ui.a2ui import format_section
        result = format_section("mermaid", "graph TD; A-->B;", title="Architecture")
        section = result["content"][0]
        assert section["content"] == "graph TD; A-->B;"

    def test_format_section_with_props(self):
        """format_section with props."""
        from Jotty.core.interface.ui.a2ui import format_section
        props = {"theme": "dark"}
        result = format_section("chart", {"type": "bar"}, props=props)
        assert result["content"][0]["props"] == props

    def test_format_section_no_title_or_props(self):
        """format_section without optional fields omits them."""
        from Jotty.core.interface.ui.a2ui import format_section
        result = format_section("json", {"data": 123})
        section = result["content"][0]
        assert "title" not in section
        assert "props" not in section
